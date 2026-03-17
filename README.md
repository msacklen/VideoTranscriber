# Video Transcriber

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Windows](https://img.shields.io/badge/platform-windows-lightgrey.svg)
![Downloads](https://img.shields.io/github/downloads/msacklen/VideoTranscriber/total)

A Windows desktop application that transcribes audio/video files using faster-whisper.
Currently features a CLI, support for english & finnish transcription and GPU acceleration.
Built for internal use, so bugs can be expected. Might develop further to add support for other languages and other necessary features.

## Requirements

- **Windows 7/8/10/11** (64-bit recommended)
- **4GB RAM minimum** (8GB+ recommended for large files)
- **Optional**: NVIDIA GPU with CUDA for acceleration
- **Disk space**: 500MB-3GB for model files (downloaded on first use)

## Installation

### Option 1: Download Executable (Recommended)

1. Go to the [Releases page](https://github.com/msacklen/VideoTranscriber/releases)
2. Download `VideoTranscriber.exe`
3. Run the executable (no installation needed)

### Option 2: Run from Source

```bash
# Clone the repository
git clone https://github.com/msacklen/VideoTranscriber.git
cd VideoTranscriber

# Install dependencies
pip install -r requirements.txt

# Run the script
python VideoTranscriber.py
