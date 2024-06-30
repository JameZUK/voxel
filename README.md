# Voxel Recorder

Voxel Recorder is a versatile audio recording and processing utility designed for both interactive and non-interactive environments. It supports real-time audio input, signal processing, and various user-defined configurations to tailor the recording process to your needs. The script is built with Python and leverages several powerful libraries such as `pyaudio`, `numpy`, and `scipy` for high-quality audio processing.

## Features

- **Real-time audio recording** with configurable device selection and chunk size.
- **Volume threshold-based recording** to trigger recording based on audio input levels.
- **Automatic post-processing** with options for normalization and filtering.
- **Dynamic folder structure** for organizing recordings by month and week.
- **Maximum recording length** to prevent long continuous recordings, automatically stopping and starting new recordings as needed.
- **Keyboard input handling** for interactive control of recording parameters (only in interactive mode).

## Command Line Options

- `command` (required): The main command to run. Options are:
  - `record`: Start recording audio.
  - `listdevs`: List available audio devices.
- `-c, --chunk` (optional): Chunk size for audio processing. Default is `8192`.
- `-d, --devno` (optional): Device number for audio input. Default is `2`.
- `-s, --saverecs` (optional): Number of records to buffer ahead of threshold. Default is `8`.
- `-t, --threshold` (optional): Minimum volume threshold to trigger recording (0.1-99). Default is `0.3`.
- `-l, --hangdelay` (optional): Seconds to continue recording after input drops below threshold. Default is `6`.
- `-n, --normalize` (optional): Normalize audio (enabled with this flag).
- `-F, --filter` (optional): Apply filtering to audio (enabled with this flag).
- `--harmonics` (optional): Number of harmonics to apply notch filter. Default is `4`.
- `--rootpath` (optional): Root path for storing recordings. Default is `recordings`.
- `--maxlength` (optional): Maximum recording length in minutes. Default is `0` (unlimited).

## Usage

### Listing Audio Devices

To list available audio devices:

```bash
python voxel.py listdevs
Starting a Recording

To start recording audio with default settings:

bash

python voxel.py record

To start recording with custom settings:

bash

python voxel.py record -c 4096 -d 1 -s 10 -t 0.5 -l 10 -n -F --harmonics 3 --rootpath /path/to/recordings --maxlength 60

Running as a Systemd Service

To run this script as a systemd service, create a service file like the following:

ini

[Unit]
Description=Voxel Recording Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/james/voxel/voxel.py record -c 8192 -d 2 -s 8 -t 0.3 -l 6 -n --rootpath /path/to/recordings --maxlength 60
WorkingDirectory=/home/james/voxel
StandardOutput=syslog
StandardError=syslog
Restart=always
User=james

[Install]
WantedBy=multi-user.target

Replace /path/to/recordings with the actual path where you want the recordings to be saved and adjust other parameters as necessary.
Interactive Controls

When running interactively, the following keyboard commands are supported:

    h, ?: Display help.
    f: Show current filename.
    k: Show peak level.
    v: Set new trigger level.
    r: Toggle recording on/off.
    p: Toggle peak display.
    n: Toggle normalization.
    F: Toggle filtering.
    d: Display debug information.
    q: Quit the application.

Dependencies

Ensure you have the following Python packages installed:

    argparse
    ctypes
    pyaudio
    numpy
    soundfile
    scipy

You can install them using pip:

bash

pip install pyaudio numpy soundfile scipy

Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to follow the code style and add tests for any new features or bug fixes.
License

This project is licensed under the MIT License. See the LICENSE file for details.
