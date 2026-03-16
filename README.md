# Video Transcriber
A Windows desktop application that transcribes audio/video files using faster-whisper.
Currently features a CLI, support for english & finnish transcription and GPU acceleration.
Built for internal use, so bugs can be expected. Might develop further to add support for other languages and other necessary features.

## Installation

### Option 1: Download Pre-built Executable (Recommended)

1. Go to the [Releases page](https://github.com/msacklen/VideoTranscriber/releases)
2. Download `VideoTranscriber.exe`
3. Run the executable (no installation needed)

### Option 2: Run from Source (WIP)

```bash
# Clone the repository
git clone https://github.com/msacklen/VideoTranscriber.git
cd VideoTranscriber

# Install dependencies
pip install -r requirements.txt

# Run the script
python VideoTranscriber.py
