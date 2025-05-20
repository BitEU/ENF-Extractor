#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import librosa
import argparse
import csv
import os
import sys
import subprocess
import tempfile
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AdaptiveGoertzelENF:
    def __init__(self, sample_rate=22050, nominal_freq=60.0, freq_tolerance=0.5, 
                 window_length=1.0, overlap=0.5, harmonics=[1, 2, 3]):
        """
        Initialize the Adaptive Goertzel ENF extractor.
        
        Parameters:
        - sample_rate: Audio sample rate in Hz
        - nominal_freq: Nominal mains frequency (50Hz for Europe/Asia, 60Hz for Americas)
        - freq_tolerance: Frequency search range around nominal (+/- Hz)
        - window_length: Analysis window length in seconds
        - overlap: Window overlap ratio (0-1)
        - harmonics: List of harmonics to analyze (1=fundamental, 2=second harmonic, etc.)
        """
        self.sample_rate = sample_rate
        self.nominal_freq = nominal_freq
        self.freq_tolerance = freq_tolerance
        self.window_length = window_length
        self.overlap = overlap
        self.harmonics = harmonics
        
        # Calculate window parameters
        self.window_samples = int(window_length * sample_rate)
        self.hop_samples = int(self.window_samples * (1 - overlap))
        
        # Frequency search parameters
        self.freq_resolution = 0.01  # Hz resolution for frequency search
        self.search_freqs = np.arange(
            nominal_freq - freq_tolerance,
            nominal_freq + freq_tolerance + self.freq_resolution,
            self.freq_resolution
        )
    
    def goertzel_algorithm(self, samples, target_freq, sample_rate):
        """
        Implement the Goertzel algorithm for single frequency detection.
        
        Parameters:
        - samples: Input signal samples
        - target_freq: Target frequency to detect
        - sample_rate: Sample rate of the signal
        
        Returns:
        - magnitude: Magnitude of the target frequency component
        """
        N = len(samples)
        k = int(N * target_freq / sample_rate)
        w = 2 * np.pi * k / N
        
        # Goertzel coefficients
        cosine = np.cos(w)
        coeff = 2 * cosine
        
        # Initialize variables
        q0 = q1 = q2 = 0.0
        
        # Process samples
        for sample in samples:
            q0 = coeff * q1 - q2 + sample
            q2 = q1
            q1 = q0
        
        # Calculate magnitude
        real = q1 - q2 * cosine
        imag = q2 * np.sin(w)
        magnitude = np.sqrt(real**2 + imag**2)
        
        return magnitude
    
    def adaptive_frequency_estimation(self, window_samples):
        """
        Adaptively estimate the ENF within the frequency tolerance range.
        
        Parameters:
        - window_samples: Audio samples for current window
        
        Returns:
        - estimated_freq: Estimated ENF frequency
        - confidence: Confidence metric for the estimation
        """
        # Apply window function to reduce spectral leakage
        windowed_samples = window_samples * signal.windows.hann(len(window_samples))
        
        # Search for peak frequency using Goertzel algorithm
        magnitudes = []
        for freq in self.search_freqs:
            mag = self.goertzel_algorithm(windowed_samples, freq, self.sample_rate)
            magnitudes.append(mag)
        
        magnitudes = np.array(magnitudes)
        
        # Find peak frequency
        peak_idx = np.argmax(magnitudes)
        estimated_freq = self.search_freqs[peak_idx]
        peak_magnitude = magnitudes[peak_idx]
        
        # Calculate confidence based on peak prominence
        # Higher confidence when peak is well-defined above noise floor
        sorted_mags = np.sort(magnitudes)
        noise_floor = np.mean(sorted_mags[:-int(len(sorted_mags)*0.1)])  # Bottom 90%
        confidence = (peak_magnitude - noise_floor) / peak_magnitude if peak_magnitude > 0 else 0
        
        return estimated_freq, confidence
    
    def harmonic_enhancement(self, window_samples):
        """
        Enhance ENF detection using harmonic analysis.
        
        Parameters:
        - window_samples: Audio samples for current window
        
        Returns:
        - enhanced_freq: ENF estimate enhanced by harmonic analysis
        - combined_confidence: Combined confidence from all harmonics
        """
        harmonic_estimates = []
        harmonic_confidences = []
        
        for harmonic in self.harmonics:
            # Search around the harmonic frequency
            harmonic_nominal = self.nominal_freq * harmonic
            harmonic_search_freqs = np.arange(
                harmonic_nominal - self.freq_tolerance,
                harmonic_nominal + self.freq_tolerance + self.freq_resolution,
                self.freq_resolution
            )
            
            # Apply bandpass filter around harmonic
            nyquist = self.sample_rate / 2
            low_freq = max(0.1, (harmonic_nominal - 2*self.freq_tolerance) / nyquist)
            high_freq = min(0.99, (harmonic_nominal + 2*self.freq_tolerance) / nyquist)
            
            try:
                b, a = signal.butter(4, [low_freq, high_freq], btype='band')
                filtered_samples = signal.filtfilt(b, a, window_samples)
            except ValueError:
                filtered_samples = window_samples
            
            # Find peak in this harmonic
            magnitudes = []
            for freq in harmonic_search_freqs:
                mag = self.goertzel_algorithm(filtered_samples, freq, self.sample_rate)
                magnitudes.append(mag)
            
            magnitudes = np.array(magnitudes)
            peak_idx = np.argmax(magnitudes)
            
            # Convert harmonic frequency back to fundamental
            fundamental_estimate = harmonic_search_freqs[peak_idx] / harmonic
            harmonic_estimates.append(fundamental_estimate)
            
            # Calculate confidence for this harmonic
            peak_mag = magnitudes[peak_idx]
            noise_floor = np.mean(np.sort(magnitudes)[:-int(len(magnitudes)*0.1)])
            confidence = (peak_mag - noise_floor) / peak_mag if peak_mag > 0 else 0
            harmonic_confidences.append(confidence)
        
        # Weight estimates by confidence and proximity to nominal frequency
        weights = []
        for est, conf in zip(harmonic_estimates, harmonic_confidences):
            # Higher weight for estimates closer to nominal frequency
            distance_weight = 1.0 / (1.0 + abs(est - self.nominal_freq))
            weights.append(conf * distance_weight)
        
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
            enhanced_freq = np.average(harmonic_estimates, weights=weights)
            combined_confidence = np.average(harmonic_confidences, weights=weights)
        else:
            enhanced_freq = self.nominal_freq
            combined_confidence = 0.0
        
        return enhanced_freq, combined_confidence
    
    def extract_audio_from_video(self, video_file):
        """
        Extract audio from video file using FFmpeg.
        
        Parameters:
        - video_file: Path to video file
        
        Returns:
        - temp_audio_file: Path to temporary audio file
        """
        # Create temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_file = temp_audio.name
        temp_audio.close()
        
        try:
            # Use FFmpeg to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_file,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                temp_audio_file
            ]
            
            # Run FFmpeg with suppressed output
            with open(os.devnull, 'w') as devnull:
                subprocess.run(cmd, stdout=devnull, stderr=devnull, check=True)
            
            return temp_audio_file
            
        except subprocess.CalledProcessError as e:
            # Clean up temporary file on error
            if os.path.exists(temp_audio_file):
                os.unlink(temp_audio_file)
            raise ValueError(f"FFmpeg failed to extract audio: {e}")
        except FileNotFoundError:
            # Clean up temporary file on error
            if os.path.exists(temp_audio_file):
                os.unlink(temp_audio_file)
            raise ValueError("FFmpeg not found. Please install FFmpeg to process video files.")
    
    def load_audio_file(self, audio_file):
        """
        Load audio file, handling both audio and video formats.
        
        Parameters:
        - audio_file: Path to audio or video file
        
        Returns:
        - audio: Audio samples
        - sr: Sample rate
        """
        temp_audio_file = None
        
        try:
            # First try to load directly with librosa (works for most audio formats)
            audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            print(f"Loaded audio directly: {len(audio)} samples at {sr} Hz")
            return audio, sr
            
        except:
            # If direct loading fails, try extracting audio with FFmpeg
            print("Direct audio loading failed, attempting to extract audio with FFmpeg...")
            try:
                temp_audio_file = self.extract_audio_from_video(audio_file)
                audio, sr = librosa.load(temp_audio_file, sr=self.sample_rate, mono=True)
                print(f"Extracted and loaded audio: {len(audio)} samples at {sr} Hz")
                return audio, sr
                
            except Exception as e:
                raise ValueError(f"Could not load audio from {audio_file}. "
                               f"Make sure FFmpeg is installed for video files. Error: {e}")
            
            finally:
                # Clean up temporary file
                if temp_audio_file and os.path.exists(temp_audio_file):
                    os.unlink(temp_audio_file)
    def extract_enf_trace(self, audio_file, use_harmonic_enhancement=True):
        """
        Extract ENF trace from audio file.
        
        Parameters:
        - audio_file: Path to audio/video file
        - use_harmonic_enhancement: Whether to use harmonic enhancement
        
        Returns:
        - time_stamps: Array of time stamps (seconds)
        - enf_estimates: Array of ENF estimates (Hz)
        - confidences: Array of confidence scores
        """
        # Load audio file (handles both audio and video)
        try:
            audio, sr = self.load_audio_file(audio_file)
        except Exception as e:
            raise ValueError(str(e))
        
        # Initialize output arrays
        time_stamps = []
        enf_estimates = []
        confidences = []
        
        # Process audio in overlapping windows
        num_windows = int((len(audio) - self.window_samples) / self.hop_samples) + 1
        
        for i in range(num_windows):
            start_idx = i * self.hop_samples
            end_idx = start_idx + self.window_samples
            
            if end_idx > len(audio):
                break
            
            # Extract window
            window_samples = audio[start_idx:end_idx]
            
            # Calculate time stamp (center of window)
            time_stamp = (start_idx + self.window_samples / 2) / self.sample_rate
            
            # Estimate ENF for this window
            if use_harmonic_enhancement:
                freq_estimate, confidence = self.harmonic_enhancement(window_samples)
            else:
                freq_estimate, confidence = self.adaptive_frequency_estimation(window_samples)
            
            # Store results
            time_stamps.append(time_stamp)
            enf_estimates.append(freq_estimate)
            confidences.append(confidence)
            
            # Progress indicator
            if i % 1 == 0:
                print(f"Processed {i+1}/{num_windows} windows ({100*(i+1)/num_windows:.1f}%)")
        
        return np.array(time_stamps), np.array(enf_estimates), np.array(confidences)
    
    def post_process_enf(self, enf_estimates, confidences, median_filter_size=5, confidence_threshold=0.1):
        """
        Post-process ENF estimates to remove outliers and smooth the trace.
        
        Parameters:
        - enf_estimates: Raw ENF estimates
        - confidences: Confidence scores
        - median_filter_size: Size of median filter for outlier removal
        - confidence_threshold: Minimum confidence threshold
        
        Returns:
        - processed_enf: Post-processed ENF estimates
        """
        processed_enf = enf_estimates.copy()
        
        # 1. Remove low-confidence estimates
        low_confidence_mask = confidences < confidence_threshold
        processed_enf[low_confidence_mask] = np.nan
        
        # 2. Remove statistical outliers (beyond 3 standard deviations)
        valid_estimates = processed_enf[~np.isnan(processed_enf)]
        if len(valid_estimates) > 0:
            mean_enf = np.mean(valid_estimates)
            std_enf = np.std(valid_estimates)
            outlier_mask = np.abs(processed_enf - mean_enf) > 3 * std_enf
            processed_enf[outlier_mask] = np.nan
        
        # 3. Interpolate missing values
        valid_indices = ~np.isnan(processed_enf)
        if np.sum(valid_indices) > 1:
            processed_enf = np.interp(
                np.arange(len(processed_enf)),
                np.where(valid_indices)[0],
                processed_enf[valid_indices]
            )
        
        # 4. Apply median filter to smooth the trace
        if median_filter_size > 1:
            processed_enf = signal.medfilt(processed_enf, kernel_size=median_filter_size)
        
        return processed_enf
    
    def save_to_csv(self, time_stamps, enf_estimates, confidences, output_file):
        """
        Save ENF trace to CSV file.
        
        Parameters:
        - time_stamps: Array of time stamps
        - enf_estimates: Array of ENF estimates
        - confidences: Array of confidence scores
        - output_file: Output CSV file path
        """
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Time_s', 'ENF_Hz', 'Confidence'])
            
            # Write data
            for t, f, c in zip(time_stamps, enf_estimates, confidences):
                writer.writerow([f'{t:.3f}', f'{f:.6f}', f'{c:.6f}'])
        
        print(f"ENF trace saved to {output_file}")
    
    def plot_enf_trace(self, time_stamps, enf_estimates, confidences, output_file=None):
        """
        Plot ENF trace with confidence scores.
        
        Parameters:
        - time_stamps: Array of time stamps
        - enf_estimates: Array of ENF estimates
        - confidences: Array of confidence scores
        - output_file: Optional output file for saving plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot ENF trace
        ax1.plot(time_stamps, enf_estimates, 'b-', linewidth=1)
        ax1.axhline(y=self.nominal_freq, color='r', linestyle='--', alpha=0.7, label=f'Nominal ({self.nominal_freq} Hz)')
        ax1.set_ylabel('ENF (Hz)')
        ax1.set_title('Electrical Network Frequency Trace')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot confidence scores
        ax2.plot(time_stamps, confidences, 'g-', linewidth=1)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Confidence Scores')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()


def main():
    # Custom help if user runs: python main_v1.py help
    if len(sys.argv) == 2 and sys.argv[1].lower() == "help":
        print("""
        Usage: python main_v1.py <input_file> [options]

        Extract ENF from audio/video files using adaptive Goertzel algorithm.

        Positional arguments:
            input_file            Input audio/video file (supports MP4, AVI, MP3, WAV, etc.)

        Optional arguments:
            -o, --output          Output CSV file (default: input_filename_enf.csv)
            -f, --frequency       Nominal mains frequency (default: 60 Hz)
            -t, --tolerance       Frequency search tolerance (+/- Hz, default: 0.5 Hz)
            -w, --window          Analysis window length in seconds (default: 1.0)
            -r, --overlap         Window overlap ratio (default: 0.5)
            --sample-rate         Audio sample rate (default: 22050)
            --harmonics           Harmonics to analyze (default: 1 2 3)
            --no-harmonics        Disable harmonic enhancement
            --plot                Generate and display plot
            --plot-output         Save plot to file
            help                  Show this help message and exit

        Example:
            python main_v1.py myaudio.wav -f 60 -w 1 --plot

        Please provide the path to your audio or video file as the first argument.
        """)
        sys.exit(0)
    
    
    parser = argparse.ArgumentParser(description='Extract ENF from audio/video files using adaptive Goertzel algorithm')
    parser.add_argument('input_file', help='Input audio/video file (supports MP4, AVI, MP3, WAV, etc.)')
    parser.add_argument('-o', '--output', help='Output CSV file (default: input_filename_enf.csv)')
    parser.add_argument('-f', '--frequency', type=float, default=60.0, help='Nominal mains frequency (default: 60 Hz)')
    parser.add_argument('-t', '--tolerance', type=float, default=0.5, help='Frequency search tolerance (+/- Hz, default: 0.5 Hz)')
    parser.add_argument('-w', '--window', type=float, default=1.0, help='Analysis window length in seconds (default: 1.0)')
    parser.add_argument('-r', '--overlap', type=float, default=0.5, help='Window overlap ratio (default: 0.5)')
    parser.add_argument('--sample-rate', type=int, default=22050, help='Audio sample rate (default: 22050)')
    parser.add_argument('--harmonics', nargs='+', type=int, default=[1, 2, 3], help='Harmonics to analyze (default: 1 2 3)')
    parser.add_argument('--no-harmonics', action='store_true', help='Disable harmonic enhancement')
    parser.add_argument('--plot', action='store_true', help='Generate and display plot')
    parser.add_argument('--plot-output', help='Save plot to file')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return
    
    # Create output filename if not specified
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = input_path.parent / f"{input_path.stem}_enf.csv"
    
    # Check for required dependencies
    try:
        import librosa
        import scipy
        import matplotlib
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install numpy scipy librosa matplotlib")
        return
    
    # Initialize ENF extractor
    extractor = AdaptiveGoertzelENF(
        sample_rate=args.sample_rate,
        nominal_freq=args.frequency,
        freq_tolerance=args.tolerance,
        window_length=args.window,
        overlap=args.overlap,
        harmonics=args.harmonics
    )
    
    print(f"Extracting ENF from: {args.input_file}")
    print(f"Nominal frequency: {args.frequency} Hz")
    print(f"Search tolerance: +/- {args.tolerance} Hz")
    print(f"Window length: {args.window} seconds")
    print(f"Harmonics: {args.harmonics if not args.no_harmonics else 'disabled'}")
    print("-" * 50)
    
    try:
        # Extract ENF trace
        time_stamps, enf_estimates, confidences = extractor.extract_enf_trace(
            args.input_file, 
            use_harmonic_enhancement=not args.no_harmonics
        )
        
        # Post-process the ENF trace
        processed_enf = extractor.post_process_enf(enf_estimates, confidences)
        
        # Save to CSV
        extractor.save_to_csv(time_stamps, processed_enf, confidences, args.output)
        
        # Generate statistics
        print("-" * 50)
        print(f"ENF Statistics:")
        print(f"  Duration: {time_stamps[-1]:.1f} seconds")
        print(f"  Mean ENF: {np.mean(processed_enf):.6f} Hz")
        print(f"  Std ENF: {np.std(processed_enf):.6f} Hz")
        print(f"  Min ENF: {np.min(processed_enf):.6f} Hz")
        print(f"  Max ENF: {np.max(processed_enf):.6f} Hz")
        print(f"  Mean Confidence: {np.mean(confidences):.3f}")
        
        # Generate plot if requested
        if args.plot or args.plot_output:
            extractor.plot_enf_trace(time_stamps, processed_enf, confidences, args.plot_output)
    
    except Exception as e:
        print(f"Error processing file: {e}")
        if "FFmpeg" in str(e):
            print("\nTo process video files, please install FFmpeg:")
            print("- Windows: Download from https://ffmpeg.org/download.html")
            print("- Linux: sudo apt install ffmpeg")
            print("- macOS: brew install ffmpeg")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
