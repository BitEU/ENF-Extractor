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
import concurrent.futures
import datetime
import time
import itertools

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- Default configuration values ---
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_NOMINAL_FREQ = 60.0
DEFAULT_FREQ_TOLERANCE = 0.1
DEFAULT_WINDOW_LENGTH = 2.0
DEFAULT_OVERLAP = 0.5
DEFAULT_HARMONICS = [1, 2, 3]
DEFAULT_MEDIAN_FILTER_SIZE = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.1

# --- ENF Speed/Time-Scale Detection Defaults ---
DEFAULT_CANDIDATE_ENF_FREQS = [50.0, 60.0]  # Hz, can be expanded if needed
DEFAULT_MIN_SPEED_FACTOR = 0.5              # Minimum speed factor to check (e.g., 0.5x)
DEFAULT_MAX_SPEED_FACTOR = 2.5              # Maximum speed factor to check (e.g., 2.5x)
DEFAULT_SPEED_FACTOR_STEP = 0.01            # Step size for speed factor search
# ------------------------------------

class AdaptiveGoertzelENF:
    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, nominal_freq=DEFAULT_NOMINAL_FREQ, 
                 freq_tolerance=DEFAULT_FREQ_TOLERANCE, window_length=DEFAULT_WINDOW_LENGTH, 
                 overlap=DEFAULT_OVERLAP, harmonics=DEFAULT_HARMONICS,
                 candidate_enf_freqs=DEFAULT_CANDIDATE_ENF_FREQS,
                 min_speed_factor=DEFAULT_MIN_SPEED_FACTOR,
                 max_speed_factor=DEFAULT_MAX_SPEED_FACTOR,
                 speed_factor_step=DEFAULT_SPEED_FACTOR_STEP):
        """
        Initialize the Adaptive Goertzel ENF extractor.
        
        Parameters:
        - sample_rate: Audio sample rate in Hz
        - nominal_freq: Nominal mains frequency (50Hz for Europe/Asia, 60Hz for Americas)
        - freq_tolerance: Frequency search range around nominal (+/- Hz)
        - window_length: Analysis window length in seconds
        - overlap: Window overlap ratio (0-1)
        - harmonics: List of harmonics to analyze (1=fundamental, 2=second harmonic, etc.)
        - candidate_enf_freqs: List of candidate ENF frequencies to consider for speed detection
        - min_speed_factor: Minimum speed factor to check (e.g., 0.5x)
        - max_speed_factor: Maximum speed factor to check (e.g., 2.5x)
        - speed_factor_step: Step size for speed factor search
        """
        self.sample_rate = sample_rate
        self.nominal_freq = nominal_freq
        self.freq_tolerance = freq_tolerance
        self.window_length = window_length
        self.overlap = overlap
        self.harmonics = harmonics

        # Speed/ENF detection config
        self.candidate_enf_freqs = candidate_enf_freqs
        self.min_speed_factor = min_speed_factor
        self.max_speed_factor = max_speed_factor
        self.speed_factor_step = speed_factor_step
        
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

    def goertzel_vectorized(self, samples, freqs, sample_rate, use_gpu=False):
        """
        Vectorized Goertzel for multiple frequencies.
        """
        N = len(samples)
        if use_gpu and GPU_AVAILABLE:
            xp = cp
            samples = cp.asarray(samples)
            freqs = cp.asarray(freqs)
        else:
            xp = np

        k = N * freqs / sample_rate
        w = 2 * xp.pi * k / N
        cosine = xp.cos(w)
        coeff = 2 * cosine

        q0 = xp.zeros_like(freqs)
        q1 = xp.zeros_like(freqs)
        q2 = xp.zeros_like(freqs)

        for sample in samples:
            q0 = coeff * q1 - q2 + sample
            q2 = q1
            q1 = q0

        real = q1 - q2 * cosine
        imag = q2 * xp.sin(w)
        magnitude = xp.sqrt(real**2 + imag**2)
        if use_gpu and GPU_AVAILABLE:
            return cp.asnumpy(magnitude)
        return magnitude

    def goertzel_vectorized_batch(self, windows, freqs, sample_rate):
        """
        Vectorized Goertzel for multiple windows and frequencies using GPU.
        windows: shape (num_windows, window_samples)
        freqs: shape (num_freqs,)
        Returns: magnitudes, shape (num_windows, num_freqs)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is not available for GPU processing.")
        xp = cp
        windows = cp.asarray(windows)  # (num_windows, window_samples)
        freqs = cp.asarray(freqs)      # (num_freqs,)
        N = windows.shape[1]
        num_windows = windows.shape[0]
        num_freqs = freqs.shape[0]

        k = N * freqs / sample_rate
        w = 2 * xp.pi * k / N
        cosine = xp.cos(w)
        coeff = 2 * cosine

        # Prepare for broadcasting: (num_windows, num_freqs)
        q0 = xp.zeros((num_windows, num_freqs), dtype=windows.dtype)
        q1 = xp.zeros((num_windows, num_freqs), dtype=windows.dtype)
        q2 = xp.zeros((num_windows, num_freqs), dtype=windows.dtype)

        # Process all windows in parallel
        stream = cp.cuda.Stream()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        with stream:
            start_event.record()
            for n in range(N):
                sample = windows[:, n][:, None]  # shape (num_windows, 1)
                q0 = coeff * q1 - q2 + sample
                q2 = q1
                q1 = q0
            end_event.record()

        # Wait for completion
        end_event.synchronize()
        elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
        print(f"GPU batch completed in {elapsed_time/1000:.2f} seconds")

        real = q1 - q2 * cosine
        imag = q2 * xp.sin(w)
        magnitude = xp.sqrt(real**2 + imag**2)
        return cp.asnumpy(magnitude)  # shape (num_windows, num_freqs)
    
    def adaptive_frequency_estimation(self, window_samples, use_gpu=False):
        """
        Adaptively estimate the ENF within the frequency tolerance range.
        
        Parameters:
        - window_samples: Audio samples for current window
        - use_gpu: Whether to use GPU acceleration
        
        Returns:
        - estimated_freq: Estimated ENF frequency
        - confidence: Confidence metric for the estimation
        """
        # Apply window function to reduce spectral leakage
        windowed_samples = window_samples * signal.windows.hann(len(window_samples))
        
        # Search for peak frequency using vectorized Goertzel algorithm
        magnitudes = self.goertzel_vectorized(windowed_samples, self.search_freqs, self.sample_rate, use_gpu=use_gpu)
        
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
    
    def harmonic_enhancement(self, window_samples, use_gpu=False):
        """
        Enhance ENF detection using harmonic analysis.
        
        Parameters:
        - window_samples: Audio samples for current window
        - use_gpu: Whether to use GPU acceleration
        
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
            magnitudes = self.goertzel_vectorized(filtered_samples, harmonic_search_freqs, self.sample_rate, use_gpu=use_gpu)
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
    
    def extract_enf_trace(self, audio_file, use_harmonic_enhancement=True, max_workers=None, use_gpu=False):
        try:
            audio, sr = self.load_audio_file(audio_file)
        except Exception as e:
            raise ValueError(str(e))

        num_windows = int((len(audio) - self.window_samples) / self.hop_samples) + 1

        # Prepare all windows at once
        window_indices = [
            (i * self.hop_samples, i * self.hop_samples + self.window_samples)
            for i in range(num_windows)
            if (i * self.hop_samples + self.window_samples) <= len(audio)
        ]
        windows = np.stack([audio[start:end] for start, end in window_indices])
        time_stamps = np.array([
            (start + self.window_samples / 2) / self.sample_rate
            for start, end in window_indices
        ])

        if use_gpu:
            if not use_harmonic_enhancement:
                # Already batched, keep as is
                windows_hann = windows * signal.windows.hann(self.window_samples)
                magnitudes = self.goertzel_vectorized_batch(windows_hann, self.search_freqs, self.sample_rate)
                peak_indices = np.argmax(magnitudes, axis=1)
                enf_estimates = self.search_freqs[peak_indices]
                peak_mags = magnitudes[np.arange(len(peak_indices)), peak_indices]
                sorted_mags = np.sort(magnitudes, axis=1)
                noise_floor = np.mean(sorted_mags[:, :-int(magnitudes.shape[1]*0.1)], axis=1)
                confidences = (peak_mags - noise_floor) / np.where(peak_mags > 0, peak_mags, 1)
                # Progress indicator for GPU
                return time_stamps, enf_estimates, confidences
            else:
                # --- BATCHED HARMONIC ENHANCEMENT ---
                harmonic_estimates = []
                harmonic_confidences = []
                for harmonic in self.harmonics:
                    harmonic_nominal = self.nominal_freq * harmonic
                    harmonic_search_freqs = np.arange(
                        harmonic_nominal - self.freq_tolerance,
                        harmonic_nominal + self.freq_tolerance + self.freq_resolution,
                        self.freq_resolution
                    )
                    # Bandpass filter all windows (on CPU, fast enough)
                    nyquist = self.sample_rate / 2
                    low_freq = max(0.1, (harmonic_nominal - 2*self.freq_tolerance) / nyquist)
                    high_freq = min(0.99, (harmonic_nominal + 2*self.freq_tolerance) / nyquist)
                    try:
                        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
                        filtered_windows = signal.filtfilt(b, a, windows, axis=1)
                    except ValueError:
                        filtered_windows = windows
                    # Goertzel batch on GPU
                    magnitudes = self.goertzel_vectorized_batch(filtered_windows, harmonic_search_freqs, self.sample_rate)
                    peak_indices = np.argmax(magnitudes, axis=1)
                    # Convert harmonic frequency back to fundamental
                    fundamental_estimate = harmonic_search_freqs[peak_indices] / harmonic
                    harmonic_estimates.append(fundamental_estimate)
                    # Confidence
                    peak_mags = magnitudes[np.arange(len(peak_indices)), peak_indices]
                    sorted_mags = np.sort(magnitudes, axis=1)
                    noise_floor = np.mean(sorted_mags[:, :-int(magnitudes.shape[1]*0.1)], axis=1)
                    confidence = (peak_mags - noise_floor) / np.where(peak_mags > 0, peak_mags, 1)
                    harmonic_confidences.append(confidence)
                # Combine harmonics (vectorized)
                harmonic_estimates = np.stack(harmonic_estimates, axis=1)  # (num_windows, num_harmonics)
                harmonic_confidences = np.stack(harmonic_confidences, axis=1)
                # Weight by confidence and proximity to nominal
                distance_weights = 1.0 / (1.0 + np.abs(harmonic_estimates - self.nominal_freq))
                weights = harmonic_confidences * distance_weights
                weights_sum = np.sum(weights, axis=1, keepdims=True)
                weights = np.where(weights_sum > 0, weights / weights_sum, 0)
                enf_estimates = np.sum(harmonic_estimates * weights, axis=1)
                confidences = np.sum(harmonic_confidences * weights, axis=1)
                # Progress indicator for GPU
                return time_stamps, enf_estimates, confidences

        # --- CPU fallback (original logic) ---
        def process_window(i):
            start_idx = i * self.hop_samples
            end_idx = start_idx + self.window_samples
            if end_idx > len(audio):
                return None
            window_samples = audio[start_idx:end_idx]
            time_stamp = (start_idx + self.window_samples / 2) / self.sample_rate
            if use_harmonic_enhancement:
                freq_estimate, confidence = self.harmonic_enhancement(window_samples, use_gpu=use_gpu)
            else:
                freq_estimate, confidence = self.adaptive_frequency_estimation(window_samples, use_gpu=use_gpu)
            return i, time_stamp, freq_estimate, confidence

        results = [None] * num_windows
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(process_window, i): i for i in range(num_windows)}
            for future in concurrent.futures.as_completed(future_to_index):
                res = future.result()
                if res is not None:
                    i, t, f, c = res
                    results[i] = (t, f, c)
                completed += 1
                print(f"Processed {completed}/{num_windows} windows ({100*completed/num_windows:.1f}%)")

        time_stamps_out, enf_estimates_out, confidences_out = [], [], []
        for r in results:
            if r is not None:
                t, f, c = r
                time_stamps_out.append(t)
                enf_estimates_out.append(f)
                confidences_out.append(c)
        return np.array(time_stamps_out), np.array(enf_estimates_out), np.array(confidences_out)
    
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

    def detect_speed_factor(self, enf_estimates, confidences, nominal_freq, freq_tolerance):
        """
        Detect if the ENF trace has been shifted due to speedup/slowdown.
        Returns the most likely original ENF, detected speed factor, and a summary of candidates.
        """
        valid = confidences > 0.2
        if np.sum(valid) < 5:
            return nominal_freq, 1.0, []

        observed = enf_estimates[valid]
        factors = np.arange(self.min_speed_factor, self.max_speed_factor + self.speed_factor_step, self.speed_factor_step)
        candidate_freqs = self.candidate_enf_freqs

        best_score = -np.inf
        best_result = (nominal_freq, 1.0, [])
        candidates = []

        for true_freq, factor in itertools.product(candidate_freqs, factors):
            expected = true_freq * factor
            mean_diff = np.mean(np.abs(observed - expected))
            std_diff = np.std(observed - expected)
            score = -mean_diff - std_diff
            candidates.append({
                'true_freq': true_freq,
                'factor': factor,
                'expected': expected,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'score': score
            })
            if score > best_score:
                best_score = score
                best_result = (true_freq, factor, candidates)

        return best_result

    def analyze_enf_and_speed(self, enf_estimates, confidences):
        """
        Analyze ENF trace for possible speedup/slowdown and report findings.
        """
        # Use the nominal frequency and tolerance from the instance
        true_freq, factor, candidates = self.detect_speed_factor(
            enf_estimates, confidences, self.nominal_freq, self.freq_tolerance
        )
        print("\n--- ENF Speed Analysis ---")
        print(f"Detected ENF: {true_freq:.2f} Hz")
        print(f"Detected speed factor: {factor:.3f}x")
        if abs(factor - 1.0) > 0.05:
            print(f"WARNING: Audio appears to be {'sped up' if factor > 1.0 else 'slowed down'} by ~{factor:.2f}x")
            print(f"Observed ENF is {true_freq * factor:.2f} Hz, expected {true_freq:.2f} Hz")
        else:
            print("No significant speedup/slowdown detected.")
        print("Top candidate ENF/factor pairs:")
        for c in sorted(candidates, key=lambda x: -x['score'])[:5]:
            print(f"  ENF: {c['true_freq']:.2f} Hz, factor: {c['factor']:.3f}x, observed: {c['expected']:.2f} Hz, mean diff: {c['mean_diff']:.4f}, score: {c['score']:.4f}")
        print("--- End ENF Speed Analysis ---\n")
        return true_freq, factor, candidates


def main():
    # Custom help if user runs: python main_v1.py help
    if len(sys.argv) == 2 and sys.argv[1].lower() == "help":
        print(f"""
        Usage: python main_v1.py <input_file> [options]


        Extract ENF from audio/video files using adaptive Goertzel algorithm.


        Positional arguments:
            input_file            Input audio/video file (supports MP4, AVI, MP3, WAV, etc.)


        Optional arguments:
            -o, --output          Output CSV file (default: input_filename_enf.csv)

            -f, --frequency       Nominal mains frequency (default: {DEFAULT_NOMINAL_FREQ} Hz).
                                  Set to 50 for most of Europe/Asia, 60 for Americas.

            -t, --tolerance       Frequency search tolerance (+/- Hz, default: {DEFAULT_FREQ_TOLERANCE} Hz).
                                  Higher values increase processing time and may reduce accuracy by including more noise,
                                  but can help if the ENF deviates significantly from nominal.

            -w, --window          Analysis window length in seconds (default: {DEFAULT_WINDOW_LENGTH}).
                                  Increasing window length improves frequency resolution and confidence,
                                  but increases processing time per window and reduces the number of output points (lower time resolution).

            -r, --overlap         Window overlap ratio (0-1, default: {DEFAULT_OVERLAP}).
                                  Higher overlap increases the number of output points and smoothness,
                                  but increases total processing time.

            --sample-rate         Audio sample rate (default: {DEFAULT_SAMPLE_RATE}).
                                  Higher sample rates may improve accuracy for high-frequency harmonics,
                                  but increase memory usage and processing time.

            --harmonics           Harmonics to analyze (default: {" ".join(map(str, DEFAULT_HARMONICS))}).
                                  Including more harmonics can improve robustness and confidence,
                                  but increases processing time.

            --no-harmonics        Disable harmonic enhancement (faster, but may reduce accuracy/confidence).

            --plot                Generate and display plot of ENF trace and confidence.

            --plot-output         Save plot to file.


            --max-workers         Maximum number of parallel workers (default: use all available CPU cores).
                                  More workers can speed up processing on multicore systems.

            --use-gpu             Use GPU acceleration for ENF extraction (requires CuPy and CUDA-capable GPU).
                                  Greatly speeds up processing for files longer than 20 seconds if supported.

            help                  Show this help message and exit.


        Example:
            python main_v1.py myaudio.wav -f 60 -w 2 --plot --max-workers 8


        Notes:
            - Increasing window length (-w) improves accuracy/confidence but reduces output points and increases processing time.
            - Increasing overlap (-r) increases output points and smoothness but also increases processing time.
            - Increasing tolerance (-t) or harmonics increases processing time and may affect accuracy.
            - Use --use-gpu for large files if you have a compatible GPU.


        Please provide the path to your audio or video file as the first argument.
        """)
        sys.exit(0)
    
    
    parser = argparse.ArgumentParser(description='Extract ENF from audio/video files using adaptive Goertzel algorithm')
    parser.add_argument('input_file', help='Input audio/video file (supports MP4, AVI, MP3, WAV, etc.)')
    parser.add_argument('-o', '--output', help='Output CSV file (default: input_filename_enf.csv)')
    parser.add_argument('-f', '--frequency', type=float, default=DEFAULT_NOMINAL_FREQ,
        help=f'Nominal mains frequency (default: {DEFAULT_NOMINAL_FREQ} Hz). Set to 50 for most of Europe/Asia, 60 for Americas. Does not affect processing time or accuracy directly.')
    parser.add_argument('-t', '--tolerance', type=float, default=DEFAULT_FREQ_TOLERANCE,
        help=f'Frequency search tolerance (+/- Hz, default: {DEFAULT_FREQ_TOLERANCE} Hz). Higher values increase processing time and may reduce accuracy by including more noise, but can help if the ENF deviates significantly from nominal.')
    parser.add_argument('-w', '--window', type=float, default=DEFAULT_WINDOW_LENGTH,
        help=f'Analysis window length in seconds (default: {DEFAULT_WINDOW_LENGTH}). Increasing window length improves frequency resolution and confidence, but increases processing time per window and reduces the number of output points (lower time resolution).')
    parser.add_argument('-r', '--overlap', type=float, default=DEFAULT_OVERLAP,
        help=f'Window overlap ratio (0-1, default: {DEFAULT_OVERLAP}). Higher overlap increases the number of output points and smoothness, but increases total processing time.')
    parser.add_argument('--sample-rate', type=int, default=DEFAULT_SAMPLE_RATE,
        help=f'Audio sample rate (default: {DEFAULT_SAMPLE_RATE}). Higher sample rates may improve accuracy for high-frequency harmonics, but increase memory usage and processing time.')
    parser.add_argument('--harmonics', nargs='+', type=int, default=DEFAULT_HARMONICS,
        help=f'Harmonics to analyze (default: {" ".join(map(str, DEFAULT_HARMONICS))}). Including more harmonics can improve robustness and confidence, but increases processing time.')
    parser.add_argument('--no-harmonics', action='store_true',
                        help='Disable harmonic enhancement (faster, but may reduce accuracy/confidence).')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and display plot of ENF trace and confidence.')
    parser.add_argument('--plot-output',
                        help='Save plot to file.')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: use all available CPU cores). More workers can speed up processing on multicore systems.')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration for ENF extraction (requires CuPy and CUDA-capable GPU). Greatly speeds up processing for large files if supported.')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return
    
    # Create output filename if not specified
    if args.output is None:
        input_path = Path(args.input_file)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        args.output = input_path.parent / f"{input_path.stem}_enf_{timestamp}.csv"
    
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
    
    # Determine number of workers
    if args.max_workers is not None:
        num_workers = args.max_workers
    else:
        num_workers = os.cpu_count()

    # Show GPU status
    gpu_status = "enabled" if args.use_gpu and GPU_AVAILABLE else "not enabled"
    print(f"Workers: {num_workers} (CPU threads)")
    print(f"GPU: {gpu_status}")
    if args.use_gpu and GPU_AVAILABLE:
        device = cp.cuda.Device()
        print(f"Using GPU device: {device.id} ({cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()})")
    elif args.use_gpu and not GPU_AVAILABLE:
        print("WARNING: --use-gpu specified but CuPy is not installed or no CUDA device is available.")

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
        start_time = time.time()  # Start timing

        # Extract ENF trace
        time_stamps, enf_estimates, confidences = extractor.extract_enf_trace(
            args.input_file, 
            use_harmonic_enhancement=not args.no_harmonics,
            max_workers=args.max_workers,
            use_gpu=args.use_gpu
        )
        
        # Post-process the ENF trace
        processed_enf = extractor.post_process_enf(enf_estimates, confidences)
        
        # Save to CSV
        extractor.save_to_csv(time_stamps, processed_enf, confidences, args.output)

        # --- ENF speed analysis ---
        extractor.analyze_enf_and_speed(processed_enf, confidences)
        
        end_time = time.time()  # End timing

        # Generate statistics
        print("-" * 50)
        print(f"ENF Statistics:")
        print(f"  Duration: {time_stamps[-1]:.1f} seconds")
        print(f"  Mean ENF: {np.mean(processed_enf):.6f} Hz")
        print(f"  Std ENF: {np.std(processed_enf):.6f} Hz")
        print(f"  Min ENF: {np.min(processed_enf):.6f} Hz")
        print(f"  Max ENF: {np.max(processed_enf):.6f} Hz")
        print(f"  Mean Confidence: {np.mean(confidences):.3f}")
        print(f"  Processing Time: {end_time - start_time:.2f} seconds")
        
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
