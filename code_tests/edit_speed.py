#!/usr/bin/env python3
r"""
This script changes the playback speed of an audio or video file using FFmpeg.

Dependencies:
•   ffmpeg (must be installed and available in your system PATH)
•   Python standard libraries: argparse, subprocess, sys, os, pathlib, json

Usage (command line):
    python edit_speed.py <input_file> <speed_factor> [output_file]

Arguments:
    <input_file>      Path to the input media file (audio or video, e.g., MP3, WAV, MP4, MKV).
    <speed_factor>    Speed multiplication factor (e.g., 2.0 for double speed, 0.5 for half speed).
    [output_file]     (Optional) Path for the output file. If not provided, a filename is auto-generated.

Example:
    python edit_speed.py video.mp4 2.0
    python edit_speed.py audio.mp3 0.5 slow_audio.mp3

Description:
    The script uses FFmpeg to change the playback speed of the specified media file. 
    For video files, both audio and video streams are adjusted to match the new speed.
    For audio files, only the audio stream is processed.
    The output file is saved with a name indicating the speed factor if not specified.
    The script does not perform pitch correction; changing speed will also change pitch.

Note:
    - Ensure FFmpeg is installed and accessible from your command line.
    - The script will prompt before overwriting existing output files.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json


def get_output_filename(input_file, speed_factor):
    """Generate output filename based on input file and speed factor."""
    input_path = Path(input_file)
    stem = input_path.stem
    suffix = input_path.suffix
    
    # Format speed factor for filename (remove trailing zeros)
    speed_str = f"{speed_factor:g}".replace('.', '_')
    
    return f"{stem}_speed{speed_str}x{suffix}"


def is_video_file(filename):
    """Check if file is likely a video file based on extension."""
    video_extensions = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', 
        '.m4v', '.3gp', '.ogv', '.ts', '.mts', '.m2ts'
    }
    return Path(filename).suffix.lower() in video_extensions


def get_sample_rate(input_file):
    """Get the sample rate of the input audio file using ffprobe."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=sample_rate',
                '-of', 'json',
                input_file
            ],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        return int(info['streams'][0]['sample_rate'])
    except Exception as e:
        print(f"Warning: Could not determine sample rate, defaulting to 44100 Hz. ({e})")
        return 44100


def change_speed(input_file, speed_factor, output_file=None):
    """Change playback speed of media file using FFmpeg, without pitch correction."""
    from pathlib import Path
    import os
    import subprocess

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False

    if output_file is None:
        output_file = get_output_filename(input_file, speed_factor)

    # Check if output file already exists
    if os.path.exists(output_file):
        response = input(f"Output file '{output_file}' already exists. Overwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return False

    try:
        sample_rate = get_sample_rate(input_file)

        if is_video_file(input_file):
            cmd = [
                'ffmpeg', '-i', input_file,
                '-filter_complex',
                f'[0:v]setpts={1/speed_factor}*PTS[v];[0:a]asetrate={sample_rate*speed_factor},aresample={sample_rate}[a]',
                '-map', '[v]', '-map', '[a]',
                '-y', output_file
            ]
        else:
            cmd = [
                'ffmpeg', '-i', input_file,
                '-filter:a', f'asetrate={sample_rate*speed_factor},aresample={sample_rate}',
                '-y', output_file
            ]

        print(f"Processing: {input_file} -> {output_file}")
        print(f"Speed factor: {speed_factor}x")
        print("Running FFmpeg...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Successfully created: {output_file}")
            return True
        else:
            print("✗ FFmpeg error:")
            print(result.stderr)
            return False

    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Change playback speed of audio/video files using FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4 2.0                    # 2x speed
  %(prog)s audio.mp3 0.5 slow_audio.mp3    # Half speed
  %(prog)s song.flac 1.25                  # 1.25x speed
  %(prog)s lecture.wav 1.5                 # 1.5x speed
        """
    )
    
    parser.add_argument('input_file', help='Input media file')
    parser.add_argument('speed_factor', type=float, 
                       help='Speed multiplication factor (e.g., 2.0 for double speed, 0.5 for half speed)')
    parser.add_argument('output_file', nargs='?', 
                       help='Output file (optional, auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Validate speed factor
    if args.speed_factor <= 0:
        print("Error: Speed factor must be positive")
        sys.exit(1)
    
    # Warn about extreme speed factors
    if args.speed_factor > 100:
        response = input(f"Speed factor {args.speed_factor}x is very high. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            sys.exit(0)
    
    if args.speed_factor < 0.01:
        response = input(f"Speed factor {args.speed_factor}x is very low. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            sys.exit(0)
    
    success = change_speed(args.input_file, args.speed_factor, args.output_file)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
