#!/usr/bin/python3

import argparse
import sounddevice as sd
import threading
import time
import numpy as np
import queue
import wave
import sys
import tty
import termios

FORMAT = np.int16
CHANNELS = 1

class VoxDat:
    def __init__(self):
        self.devindex = self.threshold = self.saverecs = self.hangdelay = self.chunk = self.devrate = self.current = self.rcnt = 0
        self.recordflag = self.running = self.peakflag = False
        self.rt = self.km = self.ttysettings = self.ttyfd = self.devstream = self.processor = None
        self.preque = self.samplequeue = None

class StreamProcessor(threading.Thread):
    def __init__(self, pdat: VoxDat):
        threading.Thread.__init__(self)
        self.daemon = True
        self.pdat = pdat
        self.rt = self.pdat.rt
        self.wf = None
        self.filename = "No File"
        
    def run(self):
        while self.pdat.running:
            data = self.pdat.samplequeue.get(1)
            if data is None:
                time.sleep(0.1)
            else:
                data2 = np.frombuffer(data, dtype=FORMAT)
                peak = np.average(np.abs(data2))
                peak = (100 * peak) / 2**12
                self.pdat.current = int(peak)
                if self.pdat.current > self.pdat.threshold:
                    self.rt.reset_timer(time.time())
                if self.pdat.recordflag:
                    if not self.wf:
                        self.filename = time.strftime("%Y%m%d-%H%M%S.wav")
                        print("opening file " + self.filename + "\r")
                        self.wf = wave.open(self.filename, 'wb')
                        self.wf.setnchannels(CHANNELS)
                        self.wf.setsampwidth(2)  # 2 bytes for FORMAT np.int16
                        self.wf.setframerate(self.pdat.devrate)
                        if self.pdat.rcnt != 0:
                            self.pdat.rcnt = 0
                            while True:
                                try:
                                    data3 = self.pdat.preque.get_nowait()
                                    self.wf.writeframes(data3)
                                except queue.Empty:
                                    break
                    self.wf.writeframes(data)
                else:
                    if self.pdat.rcnt == self.pdat.saverecs:
                        self.pdat.preque.get_nowait()
                    else:
                        self.pdat.rcnt += 1
                    self.pdat.preque.put(data)
             
    def ReadCallback(self, indata, frames, time, status):
        self.pdat.samplequeue.put(indata.tobytes())
        if self.pdat.running:
            return
        else:
            raise sd.CallbackStop()

    def close(self):
        if self.wf:
            self.wf.close()
            self.wf = False
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
                nf = min(self.pdat.current, 99)
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
                print("q: quit, r: record on/off, v: set trigger level")
            elif ch == "k":
                print(f"Peak/Trigger: {self.pdat.current} {self.pdat.threshold}")
            elif ch == "v":
                self.treset()
                pf = self.pdat.peakflag
                self.pdat.peakflag = False
                try:
                    newpeak = int(input("New Peak Limit: "))
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
                    self.pdat.threshold = 1
                    self.pdat.rt.reset_timer(time.time())
                    print("Recording enabled")
            elif ch == "p":
                self.pdat.peakflag = not self.pdat.peakflag
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
parser.add_argument("-t", "--threshold", type=int, default=99, help="Minimum volume threshold (1-99) [99]")
parser.add_argument("-l", "--hangdelay", type=int, default=6, help="Seconds to record after input drops below threshold [6]")
args = parser.parse_args()
pdat = VoxDat()

pdat.devindex = args.devno
pdat.threshold = args.threshold
pdat.saverecs = args.saverecs
pdat.hangdelay = args.hangdelay
pdat.chunk = args.chunk

if args.command == "listdevs":
    print("Device Information:")
    for i, device in enumerate(sd.query_devices()):
        print(f"Device {i}: {device['name']} - Max Input Channels: {device['max_input_channels']} - Host API: {device['hostapi']}")
else:
    pdat.samplequeue = queue.Queue()
    pdat.preque = queue.Queue()

    pdat.running = True
    pdat.rt = RecordTimer(pdat)
    pdat.processor = StreamProcessor(pdat)
    pdat.processor.start()
    pdat.rt.start()

    # Select the correct ALSA device with valid input channels
    dev_info = sd.query_devices(pdat.devindex)
    if dev_info['max_input_channels'] < CHANNELS:
        print(f"Error: Device {pdat.devindex} does not support {CHANNELS} channel(s). Please select a valid device.")
        sys.exit(1)
    
    pdat.devrate = int(dev_info['default_samplerate'])
    pdat.devstream = sd.InputStream(device=pdat.devindex,
                                    channels=CHANNELS,
                                    samplerate=pdat.devrate,
                                    callback=pdat.processor.ReadCallback,
                                    blocksize=pdat.chunk,
                                    dtype=FORMAT)
    pdat.devstream.start()

    pdat.km = KBListener(pdat)
    pdat.km.start()

    while pdat.running:
        time.sleep(1)

print("Done.")
