r"""
This script generates a stereo FLAC audio file containing a 60 Hz sine wave.

Dependencies:
- numpy
- pydub

Usage (command line):
    python generate_flac.py

Arguments:
    (none)

Example:
    python generate_flac.py

Description:
    The script creates a 60 Hz sine wave at 96 kHz sample rate, 16-bit PCM, and 60 seconds duration.
    The generated audio is stereo and exported as '60hz_sine.flac' in the current directory.

Note:
    - Ensure the required dependencies are installed:
        pip install numpy pydub
    - ffmpeg must be installed and available in your system PATH for pydub to export FLAC files.
"""

import numpy as np
from pydub import AudioSegment
import io

# Audio parameters
sample_rate = 96000  # 96 kHz for high-definition audio
frequency = 60  # 60 Hz sine wave
duration = 60.0  # 60 seconds
amplitude = 0.5  # Amplitude (0.5 to avoid clipping)

# Generate time array
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate sine wave (16-bit PCM for pydub compatibility, scaled to avoid clipping)
sine_wave = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

# Convert to stereo (pydub expects 2 channels for FLAC)
sine_wave_stereo = np.stack((sine_wave, sine_wave), axis=1)

# Create an AudioSegment from the raw audio data
audio = AudioSegment(
    sine_wave_stereo.tobytes(),
    frame_rate=sample_rate,
    sample_width=2,  # 16-bit PCM (2 bytes)
    channels=2  # Stereo
)

# Export directly to FLAC
audio.export('60hz_sine.flac', format='flac')