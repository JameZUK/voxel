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
from scipy.signal import butter, lfilter, iirnotch
import webrtcvad
from collections import deque

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 44100  # Ensure 48kHz sample rate
FRAME_DURATION_MS = 10  # Use 20ms frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # Calculate frame size in samples
FRAME_BYTES = FRAME_SIZE * 2  # 16-bit PCM, so 2 bytes per sample

class VoxDat:
    def __init__(self):
        self.devindex = self.threshold = self.saverecs = self.hangdelay = self.chunk = self.devrate = self.current = self.rcnt = 0
        self.recordflag = self.running = self.peakflag = False
        self.rt = self.km = self.ttysettings = self.ttyfd = self.pyaudio = self.devstream = self.processor = None
        self.preque = self.samplequeue = None

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_filter(data, freq, fs, quality=30):
    nyq = 0.5 * fs
    norm_freq = freq / nyq
    b, a = iirnotch(norm_freq, quality)
    y = lfilter(b, a, data)
    return y

def apply_notch_filters(data, fs):
    for freq in [50, 100, 150, 60, 120, 180]:
        data = notch_filter(data, freq, fs)
    return data

def frame_generator(buffer, frame_bytes):
    offset = 0
    while offset + frame_bytes <= len(buffer):
        frame = buffer[offset:offset + frame_bytes]
        offset += frame_bytes
        yield frame

class StreamProcessor(threading.Thread):
    def __init__(self, pdat: VoxDat, normalize: bool, filter: bool, filter_timing: str, vad_mode: int):
        threading.Thread.__init__(self)
        self.daemon = True
        self.pdat = pdat
        self.rt = self.pdat.rt
        self.file = None
        self.filename = "No File"
        self.normalize = normalize
        self.filter = filter
        self.filter_timing = filter_timing
        self.vad = webrtcvad.Vad(vad_mode) if vad_mode is not None else None
        self.audio_buffer = bytes()

        # Validate the VAD frame length
        print(f"Validating VAD with SAMPLE_RATE: {SAMPLE_RATE}, FRAME_DURATION_MS: {FRAME_DURATION_MS}")
        if not webrtcvad.valid_rate_and_frame_length(SAMPLE_RATE, FRAME_DURATION_MS):
            print(f"Invalid rate or frame length for VAD: SAMPLE_RATE={SAMPLE_RATE}, FRAME_DURATION_MS={FRAME_DURATION_MS}")
            raise ValueError("Invalid rate or frame length for VAD")
        print("Validation successful for VAD")

    def normalize_audio(self, data):
        # Normalize the audio to have a maximum of 0.99 of the maximum possible value
        peak = np.max(np.abs(data))
        if peak > 0:
            normalization_factor = (2**15 - 1) / peak * 0.99
            data = np.int16(data * normalization_factor)
        return data
    
    def apply_filters(self, data):
        # Apply bandpass filter and notch filters
        data = butter_bandpass_filter(data, lowcut=300, highcut=3400, fs=self.pdat.devrate)  # Bandpass filter
        data = apply_notch_filters(data, fs=self.pdat.devrate)  # Notch filters
        return data

    def is_speech(self, data):
        # Perform VAD on the audio data
        if self.vad is None:
            return True
        try:
            return self.vad.is_speech(data, SAMPLE_RATE)
        except webrtcvad.Error as e:
            print(f"VAD error: {e}")
            return False

    def process_audio_buffer(self):
        for frame in frame_generator(self.audio_buffer, FRAME_BYTES):
            if self.filter and self.filter_timing == 'before':
                frame = self.apply_filters(np.frombuffer(frame, dtype=np.int16)).tobytes()
            if self.is_speech(frame):
                if self.recordflag:
                    if self.filter and self.filter_timing == 'after':
                        frame = self.apply_filters(np.frombuffer(frame, dtype=np.int16)).tobytes()
                    if self.normalize:
                        frame = self.normalize_audio(np.frombuffer(frame, dtype=np.int16)).tobytes()
                    if not self.file:
                        self.filename = time.strftime("%Y%m%d-%H%M%S.flac")
                        print("opening file " + self.filename + "\r")
                        self.file = sf.SoundFile(self.filename, mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS, format='FLAC')
                    self.file.write(np.frombuffer(frame, dtype=np.int16))
                else:
                    if self.pdat.rcnt == self.pdat.saverecs:
                        self.pdat.preque.get_nowait()
                    else:
                        self.pdat.rcnt += 1
                    self.pdat.preque.put(frame)
                peak = np.max(np.abs(np.frombuffer(frame, dtype=np.int16)))  # Peak calculation
                peak_normalized = (100 * peak) / 2**15  # Normalized peak calculation
                self.pdat.current = peak_normalized  # Adjusted peak storage
                if self.pdat.current > self.pdat.threshold:
                    self.rt.reset_timer(time.time())
        # Keep remaining bytes in the buffer
        self.audio_buffer = self.audio_buffer[len(self.audio_buffer) - (len(self.audio_buffer) % FRAME_BYTES):]

    def run(self):
        while self.pdat.running:
            data = self.pdat.samplequeue.get(1)
            if data is None:
                time.sleep(0.1)
            else:
                self.audio_buffer += data
                self.process_audio_buffer()
             
    def ReadCallback(self, indata, framecount, timeinfo, status):
        self.pdat.samplequeue.put(indata)
        if self.pdat.running:
            return (None, pyaudio.paContinue)
        else:
            return (None, pyaudio.paAbort)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            self.filename = "No File"

class RecordTimer(threading.Thread):
    def __init__(self, pdat: VoxDat):
        threading.Thread.__init__(self)
        self.pdat = pdat
        self.daemon = True
        self.timer = 0
        
    def run(self):
        while self.pdat.running:
            if time.time() - self.timer < self.pdat.hangdelay:
                self.pdat.recordflag = True
            if time.time() - self.timer > self.pdat.hangdelay + 1:
                self.pdat.recordflag = False
                self.pdat.processor.close()
            if self.pdat.peakflag:
                nf = min(int(self.pdat.current), 99)  # Ensure nf is an integer
                nf2 = nf
                if nf > 50:
                    nf = int(min(50 + (nf - 50) / 3, 72))
                if nf <= 0:
                    nf = 1
                rf = "*" if self.pdat.recordflag else ""
                print(f"{'#' * nf} {nf2}{rf}\r")
            time.sleep(1)
                
    def reset_timer(self, timer: float):
        self.timer = timer

class KBListener(threading.Thread):
    def __init__(self, pdat: VoxDat):
        threading.Thread.__init__(self)
        self.pdat = pdat
        self.daemon = True

    def treset(self):
        termios.tcsetattr(self.pdat.ttyfd, termios.TCSADRAIN, self.pdat.ttysettings)

    def getch(self):
        try:
            tty.setraw(self.pdat.ttyfd)
            ch = sys.stdin.read(1)
            self.treset()
        finally:
            self.treset()
        return ch
    
    def rstop(self):
        self.pdat.rt.reset_timer(0)
        self.pdat.recordflag = False
        self.pdat.threshold = 100
        self.pdat.processor.close()

    def run(self):
        self.pdat.ttyfd = sys.stdin.fileno()
        self.pdat.ttysettings = termios.tcgetattr(self.pdat.ttyfd)
        while self.pdat.running:
            ch = self.getch()
            if ch in ["h", "?"]:
                print("h: help, f: show filename, k: show peak level, p: show peak")
                print("q: quit, r: record on/off, v: set trigger level, n: toggle normalization, F: toggle filter, T: toggle filter timing (before/after), V: toggle VAD")
            elif ch == "k":
                print(f"Peak/Trigger: {self.pdat.current:.2f} {self.pdat.threshold}")  # Display peak with 2 decimal places
            elif ch == "v":
                self.treset()
                pf = self.pdat.peakflag
                self.pdat.peakflag = False
                try:
                    newpeak = float(input("New Peak Limit: "))  # Changed to float
                except ValueError:
                    newpeak = 0
                if newpeak == 0:
                    print("? Number not recognized")
                else:
                    self.pdat.threshold = newpeak
                self.pdat.peakflag = pf
            elif ch == "f":
                if self.pdat.recordflag:
                    print("Filename: " + self.pdat.processor.filename)
                else:
                    print("Not recording")
            elif ch == "r":
                if self.pdat.recordflag:
                    self.rstop()
                    print("Recording disabled")
                else:
                    self.pdat.recordflag = True
                    self.pdat.threshold = 0.3  # Adjusted default threshold
                    self.pdat.rt.reset_timer(time.time())
                    print("Recording enabled")
            elif ch == "p":
                self.pdat.peakflag = not self.pdat.peakflag
            elif ch == "n":
                self.pdat.processor.normalize = not self.pdat.processor.normalize
                state = "enabled" if self.pdat.processor.normalize else "disabled"
                print(f"Normalization {state}")
            elif ch == "F":
                self.pdat.processor.filter = not self.pdat.processor.filter
                state = "enabled" if self.pdat.processor.filter else "disabled"
                print(f"Filtering {state}")
            elif ch == "T":
                if self.pdat.processor.filter_timing == 'before':
                    self.pdat.processor.filter_timing = 'after'
                else:
                    self.pdat.processor.filter_timing = 'before'
                print(f"Filter timing: {self.pdat.processor.filter_timing}")
            elif ch == "V":
                if self.pdat.processor.vad is None:
                    vad_mode = int(input("Enter VAD mode (0-3): "))
                    self.pdat.processor.vad = webrtcvad.Vad(vad_mode)
                else:
                    self.pdat.processor.vad = None
                state = "enabled" if self.pdat.processor.vad is not None else "disabled"
                print(f"VAD {state}")
            elif ch == "q":
                print("Quitting...")
                self.rstop()
                self.pdat.running = False
                self.treset()
                time.sleep(0.5)

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=['record', 'listdevs'], help="'record' or 'listdevs'")
parser.add_argument("-c", "--chunk", type=int, default=8192, help="Chunk size [8192]")
parser.add_argument("-d", "--devno", type=int, default=2, help="Device number [2]")
parser.add_argument("-s", "--saverecs", type=int, default=8, help="Records to buffer ahead of threshold [8]")
parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Minimum volume threshold (0.1-99) [0.3]")  # Adjusted default threshold to float
parser.add_argument("-l", "--hangdelay", type=int, default=6, help="Seconds to record after input drops below threshold [6]")
parser.add_argument("-n", "--normalize", action="store_true", help="Normalize audio [False]")  # Added normalization option
parser.add_argument("-F", "--filter", action="store_true", help="Apply filtering to audio [False]")  # Added filtering option
parser.add_argument("--filter-timing", choices=['before', 'after'], default='before', help="When to apply filtering: before or after peak detection [before]")  # Added filter timing option
parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], help="Set VAD sensitivity (0-3) [None]")  # Added VAD mode option
args = parser.parse_args()
pdat = VoxDat()

pdat.devindex = args.devno
pdat.threshold = args.threshold
pdat.saverecs = args.saverecs
pdat.hangdelay = args.hangdelay
pdat.chunk = args.chunk

with noalsaerr():
    pdat.pyaudio = pyaudio.PyAudio()

if args.command == "listdevs":
    print("Device Information:")
    for i in range(pdat.pyaudio.get_device_count()):
        dev_info = pdat.pyaudio.get_device_info_by_index(i)
        print(f"Device {i}: {dev_info['name']} - Max Input Channels: {dev_info['maxInputChannels']} - Host API: {dev_info['hostApi']}")
else:
    pdat.samplequeue = queue.Queue()
    pdat.preque = queue.Queue()

    pdat.running = True
    pdat.rt = RecordTimer(pdat)
    pdat.processor = StreamProcessor(pdat, normalize=args.normalize, filter=args.filter, filter_timing=args.filter_timing, vad_mode=args.vad_mode)  # Pass normalize, filter, filter_timing, and vad_mode arguments
    pdat.processor.start()
    pdat.rt.start()

    # Select the correct ALSA device with valid input channels
    dev_info = pdat.pyaudio.get_device_info_by_index(pdat.devindex)
    if dev_info['maxInputChannels'] < CHANNELS:
        print(f"Error: Device {pdat.devindex} does not support {CHANNELS} channel(s). Please select a valid device.")
        print("Listing all devices again to help you select:")
        for i in range(pdat.pyaudio.get_device_count()):
            dev_info = pdat.pyaudio.get_device_info_by_index(i)
            print(f"Device {i}: {dev_info['name']} - Max Input Channels: {dev_info['maxInputChannels']} - Host API: {dev_info['hostApi']}")
        sys.exit(1)
    
    pdat.devrate = SAMPLE_RATE  # Ensure the sample rate is set to 48kHz
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
