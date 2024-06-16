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
MAX_INT16 = 2**15 - 1

# Class to store recording and processing parameters
class VoxDat:
    def __init__(self):
        self.devindex = 0
        self.threshold_multiplier = 0.9  # Default multiplier for standard deviation
        self.saverecs = 8
        self.hangdelay = 6
        self.chunk = 8192
        self.devrate = 44100
        self.current = 0
        self.rcnt = 0
        self.recordflag = False
        self.running = False
        self.peakflag = False
        self.show_diagnostics = False
        self.raw_data = []
        self.noise_floor_samples = []
        self.noise_floor_avg = 0
        self.noise_floor_std = 0
        self.threshold = 0
        self.notch_filter_enabled = False
        self.noise_filter_enabled = False
        self.normalize_audio_enabled = False
        self.normalize_mode = 'post'
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
        print("StreamProcessor initialized")

    def update_noise_floor_and_threshold(self, data):
        print("Updating noise floor and threshold")
        current_noise = np.mean(np.abs(data))
        print(f"Current noise: {current_noise}")
        self.pdat.noise_floor_samples.append(current_noise)
        if len(self.pdat.noise_floor_samples) > 100:
            self.pdat.noise_floor_samples.pop(0)
        self.pdat.noise_floor_avg = np.mean(self.pdat.noise_floor_samples)
        self.pdat.noise_floor_std = np.std(self.pdat.noise_floor_samples)
        self.pdat.threshold = self.pdat.noise_floor_avg + (self.pdat.noise_floor_std * self.pdat.threshold_multiplier)
        print(f"Noise Floor Avg: {self.pdat.noise_floor_avg}, Noise Floor Std: {self.pdat.noise_floor_std}, Threshold: {self.pdat.threshold}")

    def apply_notch_filter(self, data, fs, freq, quality_factor):
        print(f"Applying notch filter: fs={fs}, freq={freq}, quality_factor={quality_factor}")
        b, a = iirnotch(freq, quality_factor, fs)
        return lfilter(b, a, data)

    def run(self):
        while self.pdat.running:
            data = self.pdat.samplequeue.get()
            if data is None:
                time.sleep(0.1)
                continue

            data2 = np.frombuffer(data, dtype=np.int16)
            print(f"Processing chunk of data: {data2[:10]}...")

            if self.pdat.notch_filter_enabled:
                print("Notch filter enabled")
                for freq in [50, 100, 150]:
                    data2 = self.apply_notch_filter(data2, self.pdat.devrate, freq, 30)

            self.update_noise_floor_and_threshold(data2)

            peak = np.max(np.abs(data2))
            self.pdat.current = (100 * peak) / MAX_INT16
            print(f"Peak: {peak}, Current: {self.pdat.current}")

            if self.pdat.current > self.pdat.threshold:
                self.pdat.rt.reset_timer(time.time())
                self.pdat.recordflag = True
                print("Recording triggered")
            else:
                self.pdat.recordflag = False
                print("Recording not triggered")

            if self.pdat.recordflag:
                if self.pdat.normalize_mode == 'fly':
                    self._write_data_on_the_fly(data2)
                else:
                    self.pdat.raw_data.append(data2)
                print("Data appended to raw_data")
            else:
                if self.file:
                    self.file.close()
                    self.file = None
                    print("File closed")

                if self.pdat.rcnt == self.pdat.saverecs:
                    self.pdat.preque.get_nowait()
                else:
                    self.pdat.rcnt += 1
                self.pdat.preque.put(data)
                print("Data appended to preque")

            if self.pdat.show_diagnostics and int(time.time()) % 1 == 0:
                self._print_diagnostics()

    def _print_diagnostics(self):
        print(f"\rNoise Floor Avg: {self.pdat.noise_floor_avg:.2f}, Noise Floor Std: {self.pdat.noise_floor_std:.2f}, Threshold: {self.pdat.threshold:.2f}, Current Peak: {self.pdat.current:.2f} (Multiplier: {self.pdat.threshold_multiplier})", end="\r")

    def _write_data_on_the_fly(self, data):
        print("Writing data on the fly")
        if not self.file:
            self._open_new_file()
        if self.pdat.normalize_audio_enabled:
            data = data / np.max(np.abs(data))
            data = (data * MAX_INT16).astype(np.int16)
        self.file.write(data)

    def _open_new_file(self):
        self.filename = self._generate_filename()
        print(f"Opening file {self.filename}")
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
                data = (data * MAX_INT16).astype(np.int16)
            self.filename = self._generate_filename()
            print(f"Saving file {self.filename}")
            sf.write(self.filename, data, self.pdat.devrate, format='FLAC')
            self.pdat.raw_data = []
            print("Raw data saved and cleared")

    def ReadCallback(self, indata, framecount, timeinfo, status):
        print("ReadCallback triggered")
        self.pdat.samplequeue.put(indata)
        if self.pdat.running:
            return (None, pyaudio.paContinue)
        else:
            return (None, pyaudio.paAbort)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print("File closed in close method")
        self.save_recording()
        print("Recording saved in close method")

class RecordTimer(threading.Thread):
    def __init__(self, pdat: VoxDat):
        super().__init__(daemon=True)
        self.pdat = pdat
        self.timer = 0
        print("RecordTimer initialized")

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
        print(f"Timer reset to {self.timer}")

    def _display_peak_info(self):
        rf = "*" if self.pdat.recordflag else ""
        noise_floor_normalized = (self.pdat.noise_floor_avg / MAX_INT16) * 100
        threshold_normalized = (self.pdat.threshold / MAX_INT16) * 100
        print("\r" + " " * 80 + "\r", end="")
        print(f"Noise floor: {noise_floor_normalized:.2f} Threshold: {threshold_normalized:.2f} Peak: {self.pdat.current:.2f} {rf}", end="\r")

class KBListener(threading.Thread):
    def __init__(self, pdat: VoxDat):
        super().__init__(daemon=True)
        self.pdat = pdat
        self.pdat.ttyfd = sys.stdin.fileno()
        self.pdat.old_settings = termios.tcgetattr(self.pdat.ttyfd)
        tty.setcbreak(self.pdat.ttyfd)
        print("KBListener initialized")

    def run(self):
        self.pdat.old_settings = termios.tcgetattr(self.pdat.ttyfd)
        tty.setcbreak(self.pdat.ttyfd)
        try:
            while self.pdat.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    self._handle_input(ch)
        finally:
            termios.tcsetattr(self.pdat.ttyfd, termios.TCSADRAIN, self.pdat.old_settings)

    def _handle_input(self, ch):
        if ch == "h" or ch == "?":
            self._print_help()
        elif ch == "k":
            self._print_peak_info()
        elif ch == "v":
            self._set_threshold_multiplier()
        elif ch == "f":
            self._print_filename()
        elif ch == "r":
            self._toggle_recording()
        elif ch == "d":
            self._toggle_diagnostics()
        elif ch == "q":
            self._quit()

    def _print_help(self):
        print("""
        h,?: Print this help
        k: Display Peak, Threshold, and Noise floor levels
        v: Set threshold multiplier (default 0.9)
        f: Display filename of current recording
        r: Start/Stop recording
        d: Toggle diagnostics on/off
        q: Quit
        """)

    def _print_peak_info(self):
        self.pdat.peakflag = True
        print(f"Threshold: {self.pdat.threshold:.2f}, Noise Floor Avg: {self.pdat.noise_floor_avg:.2f}, Noise Floor Std: {self.pdat.noise_floor_std:.2f}, Current: {self.pdat.current:.2f}")

    def _set_threshold_multiplier(self):
        print(f"Current threshold multiplier: {self.pdat.threshold_multiplier}")
        self.pdat.threshold_multiplier = float(input("Enter new threshold multiplier: "))

    def _print_filename(self):
        print(f"Filename: {self.pdat.processor.filename}")

    def _toggle_recording(self):
        self.pdat.recordflag = not self.pdat.recordflag
        status = "started" if self.pdat.recordflag else "stopped"
        print(f"Recording {status}")

    def _toggle_diagnostics(self):
        self.pdat.show_diagnostics = not self.pdat.show_diagnostics
        status = "enabled" if self.pdat.show_diagnostics else "disabled"
        print(f"Diagnostics {status}")

    def _quit(self):
        self.pdat.running = False
        print("Quitting...")
        self.pdat.processor.close()

def init_stream(pdat: VoxDat):
    with noalsaerr():
        p = pyaudio.PyAudio()
        devinfo = p.get_device_info_by_index(pdat.devindex)
        print(f"Recording from: {devinfo.get('name')}")
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=int(devinfo["defaultSampleRate"]), input=True, input_device_index=pdat.devindex, frames_per_buffer=pdat.chunk, stream_callback=pdat.processor.ReadCallback)
        return p, stream

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Recorder")
    parser.add_argument("-c", "--chunk", type=int, default=8192, help="Chunk size for audio processing")
    parser.add_argument("-d", "--device", type=int, default=1, help="Audio device number")
    parser.add_argument("-s", "--saverecs", type=int, default=8, help="Number of records to buffer ahead of threshold")
    parser.add_argument("-m", "--multiplier", type=float, default=0.9, help="Threshold multiplier for recording trigger")
    parser.add_argument("-t", "--threshold", type=float, help="Manual threshold override")
    parser.add_argument("-n", "--normalize", action="store_true", help="Enable normalization of audio")
    parser.add_argument("-x", "--notchfilter", action="store_true", help="Enable notch filter for audio")
    parser.add_argument("-f", "--file", type=str, help="Output file path")
    parser.add_argument("-r", "--rate", type=int, default=44100, help="Sampling rate")
    parser.add_argument("-q", "--quality", type=int, default=30, help="Notch filter quality factor")
    parser.add_argument("-g", "--debug", action="store_true", help="Show diagnostic information")
    return parser.parse_args()

def main():
    args = parse_args()
    pdat = VoxDat()
    pdat.chunk = args.chunk
    pdat.devindex = args.device
    pdat.saverecs = args.saverecs
    pdat.threshold_multiplier = args.multiplier
    pdat.devrate = args.rate
    pdat.normalize_audio_enabled = args.normalize
    pdat.notch_filter_enabled = args.notchfilter
    pdat.show_diagnostics = args.debug
    if args.threshold:
        pdat.threshold = args.threshold

    pdat.running = True
    pdat.processor = StreamProcessor(pdat)
    pdat.rt = RecordTimer(pdat)
    pdat.kb = KBListener(pdat)

    pdat.processor.start()
    pdat.rt.start()
    pdat.kb.start()

    p, stream = init_stream(pdat)

    try:
        while pdat.running:
            time.sleep(1)
    except KeyboardInterrupt:
        pdat.running = False
        pdat.processor.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("\nRecording stopped.")

if __name__ == "__main__":
    main()
