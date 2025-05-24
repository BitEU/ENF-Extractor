r"""
This script analyzes an audio file and prints the dominant frequency (in Hz) found in the file.

Dependencies:
- librosa
- numpy

Usage (command line):
    python verifyHz.py <audio_file_path>

Arguments:
    <audio_file_path>   Path to the audio file to analyze (e.g., WAV, FLAC, MP3).

Example:
    python verifyHz.py C:\Users\USERNAME\Downloads\ENF\60hzhighdef.flac

Description:
    The script loads the specified audio file, computes its Fast Fourier Transform (FFT),
    and identifies the frequency with the highest amplitude (dominant frequency).
    The result is printed to the console.

Note:
    - Ensure the required dependencies are installed:
        pip install librosa numpy
"""

import sys
import librosa
import numpy as np

def measure_frequency(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Compute FFT
    fft = np.fft.fft(y)
    frequencies = np.fft.fftfreq(len(fft), 1/sr)
    
    # Get positive frequencies only
    positive_freqs = frequencies[:len(frequencies)//2]
    magnitudes = np.abs(fft)[:len(frequencies)//2]
    
    # Find the frequency with the highest amplitude
    peak_freq = positive_freqs[np.argmax(magnitudes)]
    
    print(f"Dominant frequency: {peak_freq:.5f} Hz")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verifyHz.py <audio_file_path>")
        sys.exit(1)
    # Accept Windows-style paths with backslashes from the command line
    audio_file = sys.argv[1]
    measure_frequency(audio_file)