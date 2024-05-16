# voxel
command-line voice-activated recorder.

This is a voice-activated recorder with a command line interface only (no GUI). When it's running it accepts single-letter commands:
* h - Print some help
* f - Print the current recording filename
* k - Print the peak and trigger levels
* q - Quit
* p - Start or stop the peak level meter
* r - Turn recording on/off
* v - Set the sound trigger level. You'll be prompted for a peak level

Help for the command-line interface

usage: voxel.py COMMAND [-h] [-c CHUNK] [-d DEVNO] [-s SAVERECS] [-t THRESHOLD] [-l HANGDELAY]

COMMAND is: 'record' to enter record mode or 'listdevs' to list the sound devices

    Requires Python3 and the modules python3-pyaudio python3-numpy libasound2-dev
    
    With thanks to https://github.com/russinnes/py-vox-recorder, on which this code is loosely based.

Summary of Changes:

    Filtering Functions:
        butter_highpass, butter_lowpass, butter_highpass_filter, butter_lowpass_filter: Implemented high-pass and low-pass filters.
        notch_filter: Implemented a notch filter to target specific frequencies (e.g., mains hum at 50 Hz and 60 Hz).

    Normalization and Filtering:
        normalize_audio: Added normalization function.
        apply_filters: Added function to apply high-pass, low-pass, and notch filters.
        Applied filtering to the audio data before or after peak calculation based on filter_timing.

    Command-Line Options and Keyboard Controls:
        Added --normalize and --filter command-line options.
        Added --filter-timing command-line option to specify when to apply filtering (before or after peak calculation).
        Added keyboard controls n to toggle normalization, F to toggle filtering, and T to toggle filter timing.

Steps to Run the Updated Script:

    Install soundfile and scipy:

    sh

pip install soundfile scipy

Run the Script:

    List devices to find a suitable input device:

    sh

python3 voxel.py listdevs

Start recording using a valid device number with normalization and filtering enabled, and specify when to apply filtering:

sh

        python3 voxel.py record -d <valid_device_number> --normalize --filter --filter-timing before

Replace <valid_device_number> with the number of the device that supports input channels.

This version of the script now allows you to choose whether the filtering takes place before or after the peak calculation, which can be specified via the --filter-timing command-line option or toggled during recording with the T key.
