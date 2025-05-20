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

# Use raw string for the file path to avoid unicode escape errors
measure_frequency(r'C:\Users\Steven\Downloads\ENF\60hzhighdef.flac')