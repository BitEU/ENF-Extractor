# ENF Extractor

**ENF Extractor** is a Python tool for extracting Electrical Network Frequency (ENF) traces from audio and video files using an adaptive Goertzel algorithm. It supports robust ENF estimation, harmonic enhancement, confidence scoring, and exports results to CSV for further analysis. The tool is designed for forensic, research, and signal analysis applications.

My objective for this project is to identify and generate frequency ranges for content, be it 50Hz of 60Hz, but also to figure out if a video was sped up or slowed down (audio taken in a 60Hz country sped up to 2X now registering as 120Hz).

---

## Features

- **Extract ENF traces** from audio and video files (WAV, MP3, MP4, AVI, etc.)
- **Adaptive Goertzel algorithm** for precise frequency tracking
- **Harmonic enhancement** for improved robustness and confidence
- **Confidence scoring** for each ENF estimate
- **Batch processing** with multi-core CPU and optional GPU acceleration (CuPy + CUDA)
- **Post-processing**: outlier removal, interpolation, and median filtering
- **CSV export** of time-stamped ENF and confidence values
- **Optional plotting** of ENF traces and confidence scores

---

## Installation

1. **Clone the repository:** ```git clone https://github.com/yourusername/enf-extractor.git cd enf-extractor```
2. **Install dependencies:** ```pip install numpy scipy librosa matplotlib```
    - For GPU acceleration (optional):
     ```pip install cupy```
   - For video file support, [FFmpeg](https://ffmpeg.org/download.html) must be installed and available in your system path.

---

## Usage

```python main_v3.py <input_file> [options]```


**Positional arguments:**
- `<input_file>`: Path to audio or video file (WAV, MP3, MP4, AVI, etc.)

**Key options:**
- `-o, --output <file>`: Output CSV file (default: `<input_file>_enf_<timestamp>.csv`)
- `-f, --frequency <Hz>`: Nominal mains frequency (default: 60.0; use 50 for Europe/Asia)
- `-t, --tolerance <Hz>`: Frequency search tolerance (default: 0.1)
- `-w, --window <sec>`: Analysis window length in seconds (default: 2.0)
- `-r, --overlap <ratio>`: Window overlap ratio (0-1, default: 0.5)
- `--sample-rate <Hz>`: Audio sample rate (default: 22050)
- `--harmonics <list>`: Harmonics to analyze (default: 1 2 3)
- `--no-harmonics`: Disable harmonic enhancement
- `--plot`: Display ENF trace and confidence plot
- `--plot-output <file>`: Save plot to file
- `--max-workers <N>`: Number of parallel workers (default: all CPU cores)
- `--use-gpu`: Use GPU acceleration (requires CuPy and CUDA GPU)
- `help`: Show detailed help

**GPU Example:**

```python main_v3.py 60hz_sine.flac --use-gpu```

```
Workers: 12 (CPU threads)
GPU: enabled
Using GPU device: 0 (NVIDIA GeForce RTX 3070 Ti)
Extracting ENF from: 60hz_sine.flac
Nominal frequency: 60.0 Hz
Search tolerance: +/- 0.1 Hz
Window length: 2.0 seconds
Harmonics: [1, 2, 3]
--------------------------------------------------
Loaded audio directly: 1323000 samples at 22050 Hz
GPU batch completed in 2.73 seconds
GPU batch completed in 2.81 seconds
GPU batch completed in 2.84 seconds
ENF trace saved to 60hz_sine_enf_20250520145620.csv
--------------------------------------------------
ENF Statistics:
  Duration: 59.0 seconds
  Mean ENF: 59.959675 Hz
  Std ENF: 0.000000 Hz
  Min ENF: 59.959675 Hz
  Max ENF: 59.959675 Hz
  Mean Confidence: 0.502
  Processing Time: 9.89 seconds
```

![Screenshot 2025-05-20 151224](https://github.com/user-attachments/assets/18b8148d-921d-4785-8169-dfa2bd1cca37)

**CPU Example:**

```python main_v3.py 60hz_sine.flac```

```
Workers: 12 (CPU threads)
GPU: not enabled
Extracting ENF from: 60hz_sine.flac
Nominal frequency: 60.0 Hz
Search tolerance: +/- 0.1 Hz
Window length: 2.0 seconds
Harmonics: [1, 2, 3]
--------------------------------------------------
Loaded audio directly: 1323000 samples at 22050 Hz
Processed 1/59 windows (1.7%)
Processed 2/59 windows (3.4%)
Processed 3/59 windows (5.1%)
Processed 4/59 windows (6.8%)
Processed 5/59 windows (8.5%)
Processed 6/59 windows (10.2%)
Processed 7/59 windows (11.9%)
Processed 8/59 windows (13.6%)
Processed 9/59 windows (15.3%)
Processed 10/59 windows (16.9%)
Processed 11/59 windows (18.6%)
Processed 12/59 windows (20.3%)
Processed 13/59 windows (22.0%)
Processed 14/59 windows (23.7%)
Processed 15/59 windows (25.4%)
Processed 16/59 windows (27.1%)
Processed 17/59 windows (28.8%)
Processed 18/59 windows (30.5%)
Processed 19/59 windows (32.2%)
Processed 20/59 windows (33.9%)
Processed 21/59 windows (35.6%)
Processed 22/59 windows (37.3%)
Processed 23/59 windows (39.0%)
Processed 24/59 windows (40.7%)
Processed 25/59 windows (42.4%)
Processed 26/59 windows (44.1%)
Processed 27/59 windows (45.8%)
Processed 28/59 windows (47.5%)
Processed 29/59 windows (49.2%)
Processed 30/59 windows (50.8%)
Processed 31/59 windows (52.5%)
Processed 32/59 windows (54.2%)
Processed 33/59 windows (55.9%)
Processed 34/59 windows (57.6%)
Processed 35/59 windows (59.3%)
Processed 36/59 windows (61.0%)
Processed 37/59 windows (62.7%)
Processed 38/59 windows (64.4%)
Processed 39/59 windows (66.1%)
Processed 40/59 windows (67.8%)
Processed 41/59 windows (69.5%)
Processed 42/59 windows (71.2%)
Processed 43/59 windows (72.9%)
Processed 44/59 windows (74.6%)
Processed 45/59 windows (76.3%)
Processed 46/59 windows (78.0%)
Processed 47/59 windows (79.7%)
Processed 48/59 windows (81.4%)
Processed 49/59 windows (83.1%)
Processed 50/59 windows (84.7%)
Processed 51/59 windows (86.4%)
Processed 52/59 windows (88.1%)
Processed 53/59 windows (89.8%)
Processed 54/59 windows (91.5%)
Processed 55/59 windows (93.2%)
Processed 56/59 windows (94.9%)
Processed 57/59 windows (96.6%)
Processed 58/59 windows (98.3%)
Processed 59/59 windows (100.0%)
ENF trace saved to 60hz_sine_enf_20250520151450.csv
--------------------------------------------------
ENF Statistics:
  Duration: 59.0 seconds
  Mean ENF: 59.959675 Hz
  Std ENF: 0.000000 Hz
  Min ENF: 59.959675 Hz
  Max ENF: 59.959675 Hz
  Mean Confidence: 0.502
  Processing Time: 18.11 seconds
```

![Screenshot 2025-05-20 151702](https://github.com/user-attachments/assets/71853a79-ad8f-4759-8308-1cfa55400109)
![Screenshot 2025-05-20 151636](https://github.com/user-attachments/assets/19167b36-0c53-474b-84dc-367e22371d86)


---

## Notes

- For video files, ensure [FFmpeg](https://ffmpeg.org/download.html) is installed.
- Increasing window length improves frequency resolution but reduces time resolution.
- Harmonic enhancement increases robustness but may increase processing time.
- GPU acceleration is recommended for large files if supported.

---

## References

- Jenkins, C. (2011). *An investigative approach to configuring forensic electric network frequency databases.*
- Hua et al. (2021). *Detection of Electric Network Frequency in Audio Recordings.*
- Su, H. (2014). *Temporal and spatial alignment of multimedia signals.*

---

## License

MIT License
