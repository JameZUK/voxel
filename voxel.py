import argparse
import sys
import threading
import time
import tty
import termios
import select
import numpy as np

class KBListener(threading.Thread):
    def __init__(self, pdat):
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
        if ch == 'q':
            self.pdat.running = False
            print("Quit command received")
        # Add more input handling if necessary

class StreamProcessor:
    def __init__(self):
        self.noise_floor_avg = 0
        self.noise_floor_std = 0
        self.threshold = 0
        self.multiplier = 0.9
        self.recording = False
        print("StreamProcessor initialized")

    def update_noise_floor(self, current_noise):
        if self.noise_floor_avg == 0:
            self.noise_floor_avg = current_noise
        else:
            self.noise_floor_avg = (self.noise_floor_avg + current_noise) / 2
        self.noise_floor_std = self.calculate_noise_floor_std(current_noise)
        self.threshold = self.noise_floor_avg + self.noise_floor_std * self.multiplier

    def calculate_noise_floor_std(self, current_noise):
        # Using a placeholder for standard deviation calculation
        return abs(current_noise - self.noise_floor_avg)

    def process_chunk(self, data_chunk):
        current_noise = self.calculate_current_noise(data_chunk)
        self.update_noise_floor(current_noise)
        peak = self.calculate_peak(data_chunk)
        current_peak = self.calculate_current_peak(peak)
        print(f"Processing chunk of data: {data_chunk}...")
        print(f"Updating noise floor and threshold")
        print(f"Current noise: {current_noise}")
        print(f"Noise Floor Avg: {self.noise_floor_avg}, Noise Floor Std: {self.noise_floor_std}, Threshold: {self.threshold}")
        print(f"Peak: {peak}, Current: {current_peak}")

        if current_peak > self.threshold:
            self.recording = True
            print("Recording triggered")
        else:
            self.recording = False
            print("Recording not triggered")

        if self.recording:
            self.save_data(data_chunk)
            print("Data appended to preque")

        print(f"ReadCallback triggered, Noise Floor Std: {self.noise_floor_std}, Threshold: {self.threshold}, Current Peak: {current_peak} (Multiplier: {self.multiplier})")

    def calculate_current_noise(self, data_chunk):
        return max(abs(min(data_chunk)), max(data_chunk))

    def calculate_peak(self, data_chunk):
        return max(data_chunk)

    def calculate_current_peak(self, peak):
        return peak / 32767 * 100  # Example conversion

    def save_data(self, data_chunk):
        # Placeholder for saving data logic
        pass

class RecordTimer(threading.Thread):
    def __init__(self, pdat):
        super().__init__(daemon=True)
        self.pdat = pdat
        print("RecordTimer initialized")

    def run(self):
        while self.pdat.running:
            if self.pdat.recording:
                print("Recording...")
                # Implement recording duration logic
            time.sleep(1)

class VoxDat:
    def __init__(self):
        self.running = True
        self.chunk = 8192
        self.devindex = 1
        self.saverecs = False
        self.ttyfd = None
        self.old_settings = None
        print("VoxDat initialized")

def parse_args():
    parser = argparse.ArgumentParser(description="Audio recording script")
    parser.add_argument('--chunk', type=int, default=8192, help='Chunk size for audio processing')
    parser.add_argument('--device', type=int, default=1, help='Audio device number')
    parser.add_argument('--saverecs', action='store_true', help='Save recordings')
    return parser.parse_args()

def main():
    args = parse_args()
    pdat = VoxDat()
    pdat.chunk = args.chunk
    pdat.devindex = args.device
    pdat.saverecs = args.saverecs

    sp = StreamProcessor()
    rt = RecordTimer(pdat)
    kb = KBListener(pdat)

    kb.start()
    rt.start()

    # Simulation of reading audio chunks from the device
    try:
        while pdat.running:
            # Simulated chunk of audio data
            simulated_chunk = np.random.randint(-32768, 32767, pdat.chunk)
            sp.process_chunk(simulated_chunk.tolist())
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        pdat.running = False
        kb.join()
        rt.join()
        print("Recording stopped.")

if __name__ == "__main__":
    main()
