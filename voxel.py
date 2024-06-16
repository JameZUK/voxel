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
import os
from datetime import datetime, timedelta
from scipy.signal import butter, lfilter, iirnotch

FORMAT = pyaudio.paInt16
CHANNELS = 1

class VoxDat:
    def __init__(self):
        self.devindex = self.threshold = self.saverecs = self.hangdelay = self.chunk = self.devrate = self.current = self.rcnt = 0
        self.recordflag = self.running = self.peakflag = self.normalize = self.filter = False
        self.rt = self.km = self.ttysettings = self.ttyfd = self.pyaudio = self.devstream = self.processor = None
        self.preque = self.samplequeue = None
        self.debug_info = {}
        self.record_start_time = None
        self.listening = False
        self.harmonics = 4  # Default number of harmonics to filter

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

def find_dominant_frequency(data, fs):
    freqs = np.fft.fftfreq(len(data), 1/fs)
    fft_spectrum = np.abs(np.fft.fft(data))
    dominant_freq = freqs[np.argmax(fft_spectrum)]
    return abs(dominant_freq)

def notch_filter(data, freq, fs, quality=30):
    nyq = 0.5 * fs
    norm_freq = freq / nyq
    b, a = iirnotch(norm_freq, quality)
    y = lfilter(b, a, data)
    return y

def apply_notch_filter(data, fs, harmonics):
    dominant_freq = find_dominant_frequency(data, fs)
    sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Dominant frequency for notch filter: {dominant_freq} Hz\n")
    sys.stdout.flush()
    data = notch_filter(data, dominant_freq, fs)
    for harmonic in range(2, harmonics + 1):  # Apply additional notch filters for the specified harmonics
        data = notch_filter(data, dominant_freq * harmonic, fs)
    return data

def normalize_audio(data):
    peak = np.max(np.abs(data))
    if peak > 0:
        normalization_factor = (2**15 - 1) / peak * 0.99
        data = np.int16(data * normalization_factor)
    return data

def post_process(filename, devrate, apply_filter=False, apply_normalize=False, harmonics=4):
    data, samplerate = sf.read(filename, dtype='int16')
    sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Loaded recording for post-processing.\n")
    sys.stdout.flush()
    if apply_filter:
        sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Applying bandpass filter.\n")
        sys.stdout.flush()
        data = butter_bandpass_filter(data, lowcut=300, highcut=3400, fs=samplerate)
        sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Bandpass filter applied.\n")
        sys.stdout.flush()
        
        sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Applying notch filter.\n")
        sys.stdout.flush()
        data = apply_notch_filter(data, fs=samplerate, harmonics=harmonics)
        sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Notch filter applied.\n")
        sys.stdout.flush()
    if apply_normalize:
        sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Normalizing audio.\n")
        sys.stdout.flush()
        data = normalize_audio(data)
        sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Audio normalized.\n")
        sys.stdout.flush()
    sf.write(filename, data, samplerate)
    sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Post-processing completed and saved.\n")
    sys.stdout.flush()

class StreamProcessor(threading.Thread):
    def __init__(self, pdat: VoxDat):
        threading.Thread.__init__(self)
        self.daemon = True
        self.pdat = pdat
        self.rt = self.pdat.rt
        self.file = None
        self.filename = "No File"

    def run(self):
        while self.pdat.running:
            data = self.pdat.samplequeue.get(1)
            if data is None:
                time.sleep(0.1)
            else:
                data2 = np.frombuffer(data, dtype=np.int16)
                peak = np.max(np.abs(data2))  # Peak calculation
                peak_normalized = (100 * peak) / 2**15  # Normalized peak calculation
                self.pdat.current = peak_normalized  # Adjusted peak storage
                if self.pdat.current > self.pdat.threshold:
                    self.rt.reset_timer(time.time())
                if self.pdat.recordflag:
                    if not self.file:
                        self.pdat.record_start_time = datetime.now()
                        now = self.pdat.record_start_time
                        month_folder = now.strftime("%Y-%m")
                        week_folder = now.strftime("Week_%U")
                        directory = os.path.join("recordings", month_folder, week_folder)
                        os.makedirs(directory, exist_ok=True)
                        self.filename = os.path.join(directory, now.strftime("%Y%m%d-%H%M%S.flac"))
                        sys.stdout.write(f"\n{now.strftime('%Y-%m-%d %H:%M:%S')} - Opening file {self.filename}\n")
                        sys.stdout.flush()
                        self.file = sf.SoundFile(self.filename, mode='w', samplerate=self.pdat.devrate, channels=CHANNELS, format='FLAC')
                        if self.pdat.rcnt != 0:
                            self.pdat.rcnt = 0
                            while True:
                                try:
                                    data3 = self.pdat.preque.get_nowait()
                                    data3 = np.frombuffer(data3, dtype=np.int16)
                                    self.file.write(data3)
                                except queue.Empty:
                                    break
                    self.file.write(data2)
                else:
                    if self.pdat.rcnt == self.pdat.saverecs:
                        self.pdat.preque.get_nowait()
                    else:
                        self.pdat.rcnt += 1
                    self.pdat.preque.put(data)
             
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
            end_time = datetime.now()
            recording_duration = end_time - self.pdat.record_start_time
            sys.stdout.write(f"\n{end_time.strftime('%Y-%m-%d %H:%M:%S')} - Closing file {self.filename}\n")
            sys.stdout.write(f"{end_time.strftime('%Y-%m-%d %H:%M:%S')} - Recording duration: {recording_duration}\n")
            sys.stdout.flush()
            if self.pdat.filter or self.pdat.normalize:
                sys.stdout.write(f"\n{end_time.strftime('%Y-%m-%d %H:%M:%S')} - Starting post-processing of {self.filename}\n")
                sys.stdout.flush()
                post_process(self.filename, self.pdat.devrate, self.pdat.filter, self.pdat.normalize, self.pdat.harmonics)
                sys.stdout.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Post-processing completed for {self.filename}\n")
                sys.stdout.flush()
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
                if not self.pdat.recordflag:
                    self.pdat.recordflag = True
                    self.pdat.record_start_time = datetime.now()
                    sys.stdout.write(f"\n{self.pdat.record_start_time.strftime('%Y-%m-%d %H:%M:%S')} - Recording started\n")
                    sys.stdout.flush()
            if time.time() - self.timer > self.pdat.hangdelay + 1:
                if self.pdat.recordflag:
                    self.pdat.recordflag = False
                    self.pdat.processor.close()
                    sys.stdout.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Recording stopped\n")
                    sys.stdout.flush()
            if self.pdat.peakflag:
                nf = min(int(self.pdat.current), 99)  # Ensure nf is an integer
                nf2 = nf
                if nf > 50:
                    nf = int(min(50 + (nf - 50) / 3, 72))
                if nf <= 0:
                    nf = 1
                rf = "*" if self.pdat.recordflag else ""
                sys.stdout.write(f"{'#' * nf} {nf2}{rf}\r")
                sys.stdout.flush()
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
                sys.stdout.write("\nh: help, f: show filename, k: show peak level, p: show peak\n")
                sys.stdout.write("q: quit, r: record on/off, v: set trigger level, n: toggle normalization, F: toggle filtering, d: show debug info\n")
                sys.stdout.flush()
            elif ch == "k":
                sys.stdout.write(f"\nPeak/Trigger: {self.pdat.current:.2f} {self.pdat.threshold}\n")
                sys.stdout.flush()
            elif ch == "v":
                self.treset()
                pf = self.pdat.peakflag
                self.pdat.peakflag = False
                try:
                    newpeak = float(input("\nNew Peak Limit: "))
                except ValueError:
                    newpeak = 0
                if newpeak == 0:
                    sys.stdout.write("\n? Number not recognized\n")
                    sys.stdout.flush()
                else:
                    self.pdat.threshold = newpeak
                self.pdat.peakflag = pf
            elif ch == "f":
                if self.pdat.recordflag:
                    sys.stdout.write(f"\nFilename: {self.pdat.processor.filename}\n")
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\nNot recording\n")
                    sys.stdout.flush()
            elif ch == "r":
                if self.pdat.recordflag:
                    self.rstop()
                    sys.stdout.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Recording disabled\n")
                    sys.stdout.flush()
                else:
                    self.pdat.recordflag = True
                    self.pdat.threshold = 0.3
                    self.pdat.rt.reset_timer(time.time())
                    sys.stdout.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Recording enabled\n")
                    sys.stdout.flush()
            elif ch == "p":
                self.pdat.peakflag = not self.pdat.peakflag
            elif ch == "n":
                self.pdat.normalize = not self.pdat.normalize
                state = "enabled" if self.pdat.normalize else "disabled"
                sys.stdout.write(f"\nNormalization {state}\n")
                sys.stdout.flush()
            elif ch == "F":
                self.pdat.filter = not self.pdat.filter
                state = "enabled" if self.pdat.filter else "disabled"
                sys.stdout.write(f"\nFiltering {state}\n")
                sys.stdout.flush()
            elif ch == "d":
                sys.stdout.write(f"\nNormalization: {'enabled' if self.pdat.normalize else 'disabled'}\n")
                sys.stdout.write(f"Filtering: {'enabled' if self.pdat.filter else 'disabled'}\n")
                sys.stdout.flush()
            elif ch == "q":
                sys.stdout.write("\nQuitting...\n")
                sys.stdout.flush()
                self.rstop()
                self.pdat.running = False
                self.treset()
                time.sleep(0.5)

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=['record', 'listdevs'], help="'record' or 'listdevs'")
parser.add_argument("-c", "--chunk", type=int, default=8192, help="Chunk size [8192]")
parser.add_argument("-d", "--devno", type=int, default=2, help="Device number [2]")
parser.add_argument("-s", "--saverecs", type=int, default=8, help="Records to buffer ahead of threshold [8]")
parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Minimum volume threshold (0.1-99) [0.3]")
parser.add_argument("-l", "--hangdelay", type=int, default=6, help="Seconds to record after input drops below threshold [6]")
parser.add_argument("-n", "--normalize", action="store_true", help="Normalize audio [False]")
parser.add_argument("-F", "--filter", action="store_true", help="Apply filtering to audio [False]")
parser.add_argument("--harmonics", type=int, default=4, help="Number of harmonics to apply notch filter [4]")
args = parser.parse_args()
pdat = VoxDat()

pdat.devindex = args.devno
pdat.threshold = args.threshold
pdat.saverecs = args.saverecs
pdat.hangdelay = args.hangdelay
pdat.chunk = args.chunk
pdat.normalize = args.normalize
pdat.filter = args.filter
pdat.harmonics = args.harmonics

with noalsaerr():
    pdat.pyaudio = pyaudio.PyAudio()

if args.command == "listdevs":
    sys.stdout.write("Device Information:\n")
    for i in range(pdat.pyaudio.get_device_count()):
        dev_info = pdat.pyaudio.get_device_info_by_index(i)
        sys.stdout.write(f"Device {i}: {dev_info['name']} - Max Input Channels: {dev_info['maxInputChannels']} - Host API: {dev_info['hostApi']}\n")
else:
    pdat.samplequeue = queue.Queue()
    pdat.preque = queue.Queue()

    pdat.running = True
    pdat.rt = RecordTimer(pdat)
    pdat.processor = StreamProcessor(pdat)
    pdat.processor.start()
    pdat.rt.start()

    dev_info = pdat.pyaudio.get_device_info_by_index(pdat.devindex)
    if dev_info['maxInputChannels'] < CHANNELS:
        sys.stdout.write(f"Error: Device {pdat.devindex} does not support {CHANNELS} channel(s). Please select a valid device.\n")
        sys.stdout.write("Listing all devices again to help you select:\n")
        for i in range(pdat.pyaudio.get_device_count()):
            dev_info = pdat.pyaudio.get_device_info_by_index(i)
            sys.stdout.write(f"Device {i}: {dev_info['name']} - Max Input Channels: {dev_info['maxInputChannels']} - Host API: {dev_info['hostApi']}\n")
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

    sys.stdout.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Listening started\n")
    sys.stdout.flush()
    pdat.km = KBListener(pdat)
    pdat.km.start()

    while pdat.running:
        if not pdat.recordflag and not pdat.listening:
            pdat.listening = True
            sys.stdout.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Listening...\n")
            sys.stdout.flush()
        elif pdat.recordflag and pdat.listening:
            pdat.listening = False
        time.sleep(1)

sys.stdout.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Listening stopped\n")
sys.stdout.write("Done.\n")
sys.stdout.flush()
