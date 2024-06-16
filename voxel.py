import argparse
from ctypes import *
from contextlib import contextmanager
import sys
import tty
import termios
import pyaudio
import threading
import time
import numpy as np
import queue
import soundfile as sf
from scipy.signal import iirnotch, lfilter
import os
from datetime import datetime

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Class to store recording and processing parameters
class VoxDat:
    def __init__(self):
        self.devindex = 0
        self.threshold = 0.3
        self.saverecs = 8
        self.hangdelay = 6
        self.chunk = 8192
        self.devrate = 44100
        self.current = 0
        self.rcnt = 0
        self.recordflag = False
        self.running = False
        self.peakflag = False
        self.raw_data = []
        self.noise_floor_samples = []
        self.notch_filter_enabled = False
        self.noise_filter_enabled = False
        self.normalize_audio_enabled = False
        self.normalize_mode = 'fly'
        self.samplequeue = queue.Queue()
        self.preque = queue.Queue()

# Suppress ALSA errors
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

class StreamProcessor(threading.Thread):
    def __init__(self, pdat: VoxDat):
        super().__init__(daemon=True)
        self.pdat = pdat
        self.file = None

    def update_noise_floor(self, data):
        current_noise = np.mean(np.abs(data))
        self.pdat.noise_floor_samples.append(current_noise)
        if len(self.pdat.noise_floor_samples) > 100:
            self.pdat.noise_floor_samples.pop(0)
        self.pdat.noise_floor = np.mean(self.pdat.noise_floor_samples)

    def apply_notch_filter(self, data, fs, freq, quality_factor):
        b, a = iirnotch(freq, quality_factor, fs)
        return lfilter(b, a, data)

    def run(self):
        while self.pdat.running:
            data = self.pdat.samplequeue.get()
            if data is None:
                time.sleep(0.1)
                continue

            data2 = np.frombuffer(data, dtype=np.int16)
            if self.pdat.noise_filter_enabled:
                pass

            if self.pdat.notch_filter_enabled:
                for freq in [50, 100, 150]:
                    data2 = self.apply_notch_filter(data2, self.pdat.devrate, freq, 30)

            self.update_noise_floor(data2)

            peak = np.max(np.abs(data2))
            self.pdat.current = (100 * peak) / 2**15

            if self.pdat.current > self.pdat.threshold:
                self.pdat.rt.reset_timer(time.time())

            if self.pdat.recordflag:
                if self.pdat.normalize_mode == 'fly':
                    self._write_data_on_the_fly(data2)
                else:
                    self.pdat.raw_data.append(data2)
            else:
                if self.pdat.recordflag and self.file:
                    self.file.close()
                    self.file = None

                if self.pdat.rcnt == self.pdat.saverecs:
                    self.pdat.preque.get_nowait()
                else:
                    self.pdat.rcnt += 1
                self.pdat.preque.put(data)

    def _write_data_on_the_fly(self, data):
        if not self.file:
            self._open_new_file()
        if self.pdat.normalize_audio_enabled:
            data = data / np.max(np.abs(data))
            data = (data * (2**15 - 1)).astype(np.int16)
        self.file.write(data)

    def _open_new_file(self):
        self.filename = self._generate_filename()
        print("Opening file " + self.filename + "\r")
        self.file = sf.SoundFile(self.filename, mode='w', samplerate=self.pdat.devrate, channels=CHANNELS, format='FLAC')

    def _generate_filename(self):
        now = datetime.now()
        base_path = os.path.join("recordings", now.strftime("%Y-%m"), f"week_{now.strftime('%U')}")
        os.makedirs(base_path, exist_ok=True)
        return os.path.join(base_path, now.strftime("%Y%m%d-%H%M%S.flac"))

    def save_recording(self):
        if self.pdat.normalize_mode == 'post' and self.pdat.raw_data:
            data = np.concatenate(self.pdat.raw_data)
            if self.pdat.normalize_audio_enabled:
                data = data / np.max(np.abs(data))
                data = (data * (2**15 - 1)).astype(np.int16)
            self.filename = self._generate_filename()
            print("Saving file " + self.filename + "\r")
            sf.write(self.filename, data, self.pdat.devrate, format='FLAC')
            self.pdat.raw_data = []

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
        self.save_recording()

class RecordTimer(threading.Thread):
    def __init__(self, pdat: VoxDat):
        super().__init__(daemon=True)
        self.pdat = pdat
        self.timer = 0

    def run(self):
        while self.pdat.running:
            if time.time() - self.timer < self.pdat.hangdelay:
                self.pdat.recordflag = True
            elif time.time() - self.timer > self.pdat.hangdelay + 1:
                self.pdat.recordflag = False
                self.pdat.processor.save_recording()
            if self.pdat.peakflag:
                self._display_peak_info()
            time.sleep(1)

    def reset_timer(self, timer: float):
        self.timer = timer

    def _display_peak_info(self):
        nf = min(int(self.pdat.current), 99)
        nf2 = nf if nf <= 50 else int(min(50 + (nf - 50) / 3, 72))
        nf = nf if nf > 0 else 1
        rf = "*" if self.pdat.recordflag else ""
        noise_floor_normalized = (self.pdat.noise_floor / (2**15 - 1)) * 100
        print("\r" + " " * 80 + "\r", end="")
        print(f"Noise floor: {noise_floor_normalized:.2f}, Current: {self.pdat.current:.2f}, Threshold: {self.pdat.threshold:.2f}{rf}", end="\r")

class KBListener(threading.Thread):
    def __init__(self, pdat: VoxDat):
        super().__init__(daemon=True)
        self.pdat = pdat

    def run(self):
        self.pdat.ttyfd = sys.stdin.fileno()
        self.pdat.ttysettings = termios.tcgetattr(self.pdat.ttyfd)
        while self.pdat.running:
            ch = self._getch()
            self._handle_keypress(ch)

    def _getch(self):
        try:
            tty.setraw(self.pdat.ttyfd)
            ch = sys.stdin.read(1)
        finally:
            self._reset_terminal()
        return ch

    def _reset_terminal(self):
        termios.tcsetattr(self.pdat.ttyfd, termios.TCSADRAIN, self.pdat.ttysettings)

    def _handle_keypress(self, ch):
        if ch in ["h", "?"]:
            self._print_help()
        elif ch == "k":
            self._print_peak_info()
        elif ch == "v":
            self._set_trigger_level()
        elif ch == "f":
            self._print_filename()
        elif ch == "r":
            self._toggle_recording()
        elif ch == "p":
            self.pdat.peakflag = not self.pdat.peakflag
        elif ch == "n":
            self._toggle_normalization()
        elif ch == "N":
            self._toggle_noise_filter()
        elif ch == "H":
            self._toggle_notch_filter()
        elif ch == "M":
            self._toggle_normalization_mode()
        elif ch == "q":
            self._quit()

    def _print_help(self):
        print("h: help, f: show filename, k: show peak level, p: show peak")
        print("q: quit, r: record on/off, v: set trigger level")
        print("n: toggle normalization, N: toggle noise filter, H: toggle notch filter")
        print("M: toggle normalization mode (fly/post)")

    def _print_peak_info(self):
        print(f"Peak/Trigger: {self.pdat.current:.2f} {self.pdat.threshold}")

    def _set_trigger_level(self):
        self._reset_terminal()
        self.pdat.peakflag = False
        try:
            newpeak = float(input("New Peak Limit: "))
        except ValueError:
            newpeak = 0
        if newpeak:
            self.pdat.threshold = newpeak
        else:
            print("? Number not recognized")
        self.pdat.peakflag = True

    def _print_filename(self):
        if self.pdat.recordflag:
            print("Filename: " + self.pdat.processor.filename)
        else:
            print("Not recording")

    def _toggle_recording(self):
        if self.pdat.recordflag:
            self.pdat.recordflag = False
            self.pdat.processor.save_recording()
            print("Recording disabled")
        else:
            self.pdat.recordflag = True
            self.pdat.threshold = (self.pdat.noise_floor / (2**15 - 1)) * 100 * 1.5
            self.pdat.rt.reset_timer(time.time())
            print("Recording enabled")

    def _toggle_normalization(self):
        self.pdat.normalize_audio_enabled = not self.pdat.normalize_audio_enabled
        status = "enabled" if self.pdat.normalize_audio_enabled else "disabled"
        print(f"Normalization {status}")

    def _toggle_noise_filter(self):
        self.pdat.noise_filter_enabled = not self.pdat.noise_filter_enabled
        status = "enabled" if self.pdat.noise_filter_enabled else "disabled"
        print(f"Noise filter {status}")

    def _toggle_notch_filter(self):
        self.pdat.notch_filter_enabled = not self.pdat.notch_filter_enabled
        status = "enabled" if self.pdat.notch_filter_enabled else "disabled"
        print(f"Notch filter {status}")

    def _toggle_normalization_mode(self):
        self.pdat.normalize_mode = 'post' if self.pdat.normalize_mode == 'fly' else 'fly'
        print(f"Normalization mode set to {self.pdat.normalize_mode}")

    def _quit(self):
        print("Quitting...")
        self.pdat.recordflag = False
        self.pdat.running = False
        self._reset_terminal()
        time.sleep(0.5)

def display_config(args):
    print("\nCurrent Configuration:")
    print(f"  Chunk size: {args.chunk}")
    print(f"  Device number: {args.devno}")
    print(f"  Records to buffer ahead of threshold: {args.saverecs}")
    print(f"  Minimum volume threshold: {args.threshold}")
    print(f"  Seconds to record after input drops below threshold: {args.hangdelay}")
    print(f"  Notch filter: {'enabled' if args.notch else 'disabled'}")
    print(f"  Noise filter: {'enabled' if args.noise else 'disabled'}")
    print(f"  Normalization: {'enabled' if args.normalize else 'disabled'}")
    print(f"  Normalization mode: {args.normalizemode}")

# Main script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=['record', 'listdevs'], help="'record' or 'listdevs'")
    parser.add_argument("-c", "--chunk", type=int, default=8192, help="Chunk size [8192]")
    parser.add_argument("-d", "--devno", type=int, default=2, help="Device number [2]")
    parser.add_argument("-s", "--saverecs", type=int, default=8, help="Records to buffer ahead of threshold [8]")
    parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Minimum volume threshold (0.1-99) [0.3]")
    parser.add_argument("-l", "--hangdelay", type=int, default=6, help="Seconds to record after input drops below threshold [6]")
    parser.add_argument("-n", "--notch", action='store_true', help="Enable notch filter")
    parser.add_argument("-N", "--noise", action='store_true', help="Enable noise filter")
    parser.add_argument("-m", "--normalize", action='store_true', help="Enable normalization")
    parser.add_argument("-M", "--normalizemode", choices=['fly', 'post'], default='fly', help="Normalization mode: 'fly' or 'post' [fly]")
    args = parser.parse_args()

    display_config(args)  # Display configuration before proceeding

    pdat = VoxDat()
    pdat.devindex = args.devno
    pdat.threshold = args.threshold
    pdat.saverecs = args.saverecs
    pdat.hangdelay = args.hangdelay
    pdat.chunk = args.chunk
    pdat.notch_filter_enabled = args.notch
    pdat.noise_filter_enabled = args.noise
    pdat.normalize_audio_enabled = args.normalize
    pdat.normalize_mode = args.normalizemode

    with noalsaerr():
        pdat.pyaudio = pyaudio.PyAudio()

    if args.command == "listdevs":
        print("Device Information:")
        for i in range(pdat.pyaudio.get_device_count()):
            dev_info = pdat.pyaudio.get_device_info_by_index(i)
            print(f"Device {i}: {dev_info['name']} - Max Input Channels: {dev_info['maxInputChannels']} - Host API: {dev_info['hostApi']}")
    else:
        pdat.running = True
        pdat.rt = RecordTimer(pdat)
        pdat.processor = StreamProcessor(pdat)
        pdat.processor.start()
        pdat.rt.start()

        dev_info = pdat.pyaudio.get_device_info_by_index(pdat.devindex)
        if dev_info['maxInputChannels'] < CHANNELS:
            print(f"Error: Device {pdat.devindex} does not support {CHANNELS} channel(s). Please select a valid device.")
            print("Listing all devices again to help you select:")
            for i in range(pdat.pyaudio.get_device_count()):
                dev_info = pdat.pyaudio.get_device_info_by_index(i)
                print(f"Device {i}: {dev_info['name']} - Max Input Channels: {dev_info['maxInputChannels']} - Host API: {dev_info['hostApi']}")
            sys.exit(1)

        pdat.devrate = int(dev_info.get('defaultSampleRate'))
        pdat.devstream = pdat.pyaudio.open(format=FORMAT,
                                           channels=CHANNELS,
                                           rate=pdat.devrate,
                                           input=True,
                                           input_device_index=pdat.devindex,
                                           frames_per_buffer=pdat.chunk,
                                           stream_callback=pdat.processor.ReadCallback)
        pdat.devstream.start_stream()

        pdat.km = KBListener(pdat)
        pdat.km.start()

        while pdat.running:
            time.sleep(1)

    print("Done.")
