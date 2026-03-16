import os
import sys
import time
from pathlib import Path
import argparse
from faster_whisper import WhisperModel
import warnings
import tkinter as tk
from tkinter import filedialog
import subprocess

# Suppress some warnings
# warnings.filterwarnings("ignore")

def test_ffmpeg_bundled():
    """Test if ffmpeg is accessible"""
    ffmpeg_path = get_ffmpeg_path()
    print(f"FFmpeg path: {ffmpeg_path}")
    
    try:
        # Try to get ffmpeg version
        result = subprocess.run([ffmpeg_path, '-version'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg working: {version_line}")
            return True
        else:
            print("✗ FFmpeg returned error")
            return False
    except Exception as e:
        print(f"✗ FFmpeg test failed: {e}")
        return False

def get_ffmpeg_path():
    """Get the path to ffmpeg executable, whether running as script or bundled exe"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = sys._MEIPASS
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Construct path to ffmpeg.exe
    ffmpeg_path = os.path.join(base_path, 'ffmpeg.exe')
    
    # Verify it exists
    if os.path.exists(ffmpeg_path):
        return ffmpeg_path
    else:
        # Fallback to system PATH if bundled version not found
        return 'ffmpeg'

def setup_ffmpeg_environment():
    """Add ffmpeg to PATH for subprocess calls"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        # Add to PATH so subprocess can find it
        os.environ['PATH'] = base_path + os.pathsep + os.environ['PATH']
        return True
    return False

def setup_ffmpeg():
    """Setup FFmpeg path if bundled with executable"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = sys._MEIPASS
        ffmpeg_path = os.path.join(base_path, 'ffmpeg.exe')
        if os.path.exists(ffmpeg_path):
            # Add to PATH
            os.environ['PATH'] = base_path + os.pathsep + os.environ['PATH']
            return True
    return False

def check_gpu_availability():
    """Check if CUDA is available for GPU acceleration"""
    try:
        import ctypes
        # Simple check for NVIDIA CUDA DLLs
        ctypes.WinDLL('nvcuda.dll')
        
        # Try to get GPU info using nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
                print(f"✓ NVIDIA GPU detected: {gpu_name}")
                return True
        except:
            pass
        
        # Alternative check
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✓ NVIDIA GPU detected via PyTorch: {gpu_name}")
                return True
        except ImportError:
            pass
        
        return True  # CUDA DLL exists but couldn't get details
    except:
        return False

def select_file_gui():
    """Open a graphical file picker dialog"""
    # Create root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Bring dialog to front
    root.attributes('-topmost', True)
    
    # Show file picker dialog
    file_path = filedialog.askopenfilename(
        title="Select audio or video file for transcription",
        filetypes=[
            ("All supported files", "*.mp3 *.mp4 *.wav *.m4a *.flac *.ogg *.mov *.avi *.mkv *.webm"),
            ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
            ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def get_file_from_user():
    """Get input file path from user via GUI or command line"""
    print("=" * 60)
    print("WHISPER TRANSCRIPTION TOOL")
    print("By msacklen")
    print("=" * 60)
    
    # Check if file was provided as command line argument (for drag-and-drop)
    if len(sys.argv) > 1:
        file_path = sys.argv[1].strip('"')
        if os.path.exists(file_path):
            print(f"Input file (from drag-and-drop): {file_path}")
            return file_path
    
    # Ask user for input method preference
    print("\nSelect input method:")
    print("1. Browse file with graphical file picker")
    print("2. Enter file path manually")
    print("-" * 40)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            print("\nOpening file browser...")
            file_path = select_file_gui()
            if file_path:
                print(f"Selected file: {file_path}")
                return file_path
            else:
                print("No file selected. Please try manual entry or exit.")
                # Fall back to manual entry if no file selected
                choice = '2'
                continue
        
        if choice == '2':
            while True:
                file_path = input("\nEnter the path to your video/audio file: ").strip().strip('"')
                
                if not file_path:
                    retry = input("No file provided. Try again? (y/n): ").lower()
                    if retry != 'y':
                        return None
                    continue
                
                # Remove quotes if user pasted with quotes
                file_path = file_path.strip('"\'')
                
                if os.path.exists(file_path):
                    return file_path
                else:
                    print(f"File not found: {file_path}")
                    retry = input("Try another path? (y/n): ").lower()
                    if retry != 'y':
                        return None
        else:
            print("Invalid choice. Please enter 1 or 2.")

def select_compute_device():
    """Ask user whether to use CPU or GPU"""
    print("\n" + "-" * 40)
    print("COMPUTE DEVICE SELECTION")
    print("-" * 40)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        print("✓ GPU acceleration is available!")
        print("\nSelect compute device:")
        print("1. GPU (faster, requires NVIDIA GPU)")
        print("2. CPU (slower, but compatible with all systems)")
        print("-" * 40)
        
        while True:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == '1':
                return "cuda", "float16"
            elif choice == '2':
                return "cpu", "int8"
            print("Invalid choice. Please enter 1 or 2.")
    else:
        print("⚠ No NVIDIA GPU detected or CUDA not available.")
        print("Falling back to CPU mode.")
        print("(GPU acceleration requires NVIDIA GPU with CUDA)")
        return "cpu", "int8"

def select_model_size():
    """Ask user which model size to use"""
    print("\n" + "-" * 40)
    print("MODEL SIZE SELECTION")
    print("-" * 40)
    print("Available models (speed vs accuracy):")
    print("1. tiny     (fastest, least accurate, ~75MB)")
    print("2. base     (fast, good balance, ~150MB)")
    print("3. small    (slower, more accurate, ~500MB)")
    print("4. medium   (slow, very accurate, ~1.5GB)")
    print("5. large-v3 (slowest, most accurate, ~3GB)")
    print("-" * 40)
    
    model_map = {
        '1': "tiny",
        '2': "base",
        '3': "small",
        '4': "medium",
        '5': "large-v3"
    }
    
    while True:
        choice = input("Enter your choice (1-5) [default: 2]: ").strip() or "2"
        if choice in model_map:
            selected = model_map[choice]
            print(f"Selected model: {selected}")
            return selected
        print("Invalid choice. Please enter 1-5.")

def select_language_option():
    """Ask user whether to transcribe in English or translate from Finnish"""
    print("\n" + "-" * 40)
    print("LANGUAGE SELECTION")
    print("-" * 40)
    print("Select language transcription:")
    print("1. English")
    print("2. Translate from Finnish to English")
    print("-" * 40)
    print("Note: Option 1 works best for English content but can handle mixed content.")
    print("      Option 2 specifically translates Finnish to English.\n")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            return "en", "transcribe"  # Language: English, Task: transcribe
        elif choice == '2':
            return "fi", "translate"   # Language: Finnish, Task: translate
        print("Invalid choice. Please enter 1 or 2.")

def format_time(seconds):
    """Format seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def transcribe_file(file_path, language, task, device, compute_type, model_size):
    """Transcribe the file using faster-whisper with progress display"""
    
    print(f"\nLoading Whisper model: {model_size}...")
    print(f"Device: {device.upper()}, Compute type: {compute_type}")
    print("This may take a moment on first run.")
    
    try:
        # Initialize model with selected device
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root=None,  # Can specify custom download path
            cpu_threads=0 if device == "cuda" else os.cpu_count() // 2,  # Use half of CPU cores
            num_workers=1
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"⚠ Error loading model with {device}/{compute_type}: {e}")
        
        if device == "cuda":
            print("Falling back to CPU...")
            try:
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                print("✓ Model loaded on CPU instead.")
                device = "cpu"
                compute_type = "int8"
            except Exception as e:
                print(f"✗ Failed to load model even on CPU: {e}")
                return None, None
        else:
            print(f"✗ Failed to load model: {e}")
            return None, None
    
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    
    # Get file size for info
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB")
    
    start_time = time.time()
    
    try:
        # Language description for user
        if language == "en":
            lang_desc = "English (or mixed content)"
        else:
            lang_desc = "Finnish"
            
        if task == "translate":
            task_desc = "Translation to English"
        else:
            task_desc = "Transcription"
            
        print(f"🔄 Processing: {task_desc} for {lang_desc}...")
        
        # Run transcription/translation with progress tracking
        segments, info = model.transcribe(
            file_path,
            language=language,  # Explicitly set language (en or fi)
            task=task,          # transcribe or translate
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,    # Voice Activity Detection filter
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5
            ),
            without_timestamps=False,
            condition_on_previous_text=True,
            initial_prompt=None,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        # Display info
        print(f"\n📊 Processing info:")
        print(f"   - Language: {info.language} (confidence: {info.language_probability:.2f})")
        print(f"   - Duration: {info.duration:.2f} seconds\n")
        
        # Collect all segments for output file
        all_segments = []
        last_percentage = -1
        
        # Process segments with progress display
        for i, segment in enumerate(segments):
            all_segments.append(segment)
            
            # Calculate progress
            current_progress = (segment.end / info.duration) * 100
            progress_percentage = int(current_progress)
            
            # Display progress bar at 5% intervals or at the end
            if progress_percentage % 5 == 0 and progress_percentage != last_percentage:
                bar_length = 40
                filled = int(bar_length * current_progress // 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                # Calculate ETA
                elapsed = time.time() - start_time
                if current_progress > 0:
                    total_eta = elapsed / (current_progress / 100)
                    remaining = total_eta - elapsed
                    eta_str = format_time(remaining)
                else:
                    eta_str = "calculating..."
                
                print(f"\rProgress: |{bar}| {progress_percentage}% | ETA: {eta_str}", end="", flush=True)
                last_percentage = progress_percentage
            
            # Display the transcribed text (but not too frequently)
            if i % 3 == 0 or current_progress > 99:
                timestamp = format_time(segment.start)
                text = segment.text.strip()
                if len(text) > 70:
                    text = text[:67] + "..."
                print(f"\n[{timestamp}] {text}")
        
        print()  # New line after progress bar
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Processing completed in {format_time(elapsed_time)}")
        
        return all_segments, info
        
    except Exception as e:
        print(f"\n✗ Error during transcription: {e}")
        return None, None

def save_transcript(segments, info, original_file_path, language, task, device, model_size):
    """Save the transcript to a text file"""
    
    # Create output filename
    original_path = Path(original_file_path)
    output_filename = original_path.stem + "_transcript.txt"
    output_path = original_path.parent / output_filename
    
    print(f"\n💾 Saving transcript to: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 70 + "\n")
            f.write("WHISPER TRANSCRIPTION\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("📁 FILE INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Source file: {original_path.name}\n")
            f.write(f"File size: {os.path.getsize(original_file_path) / (1024 * 1024):.1f} MB\n")
            f.write(f"Duration: {info.duration:.2f} seconds\n\n")
            
            f.write("🤖 PROCESSING INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model: {model_size}\n")
            f.write(f"Device: {device.upper()}\n")
            f.write(f"Processing language: {language.upper()}\n")
            f.write(f"Task: {'Translation to English' if task == 'translate' else 'Transcription'}\n")
            f.write(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("📝 TRANSCRIPT\n")
            f.write("=" * 70 + "\n\n")
            
            # Write segments with timestamps
            for i, segment in enumerate(segments):
                timestamp = format_time(segment.start)
                f.write(f"[{timestamp}] {segment.text.strip()}\n")
                
                # Add a blank line between speakers or paragraphs occasionally
                if i > 0 and i % 20 == 0:
                    f.write("\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF TRANSCRIPT\n")
            f.write("=" * 70 + "\n")
        
        print(f"✓ Transcript saved successfully!")
        return output_path
        
    except Exception as e:
        print(f"✗ Error saving transcript: {e}")
        return None

def main():
    """Main function"""
    test_ffmpeg_bundled()
    setup_ffmpeg_environment()
    try:
        # Setup FFmpeg if bundled
        setup_ffmpeg()
        
        # Get input file
        file_path = get_file_from_user()
        if not file_path:
            print("\nNo file selected. Exiting.")
            input("\nPress Enter to exit...")
            return
        
        # Select compute device (CPU/GPU)
        device, compute_type = select_compute_device()
        
        # Select model size
        model_size = select_model_size()
        
        # Get language and task
        language, task = select_language_option()
        
        # Confirm settings
        print("\n" + "=" * 60)
        print("CONFIRMATION")
        print("=" * 60)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Device: {device.upper()}")
        print(f"Model: {model_size}")
        
        if language == "en":
            print(f"Mode: English/Mixed Transcription")
        else:
            print(f"Mode: Finnish to English Translation")
        
        print("-" * 60)
        
        confirm = input("Start processing? (y/n): ").lower()
        if confirm != 'y':
            print("Operation cancelled.")
            input("\nPress Enter to exit...")
            return
        
        # Perform transcription/translation
        segments, info = transcribe_file(file_path, language, task, device, compute_type, model_size)
        
        if segments:
            # Save transcript
            output_path = save_transcript(segments, info, file_path, language, task, device, model_size)
            
            if output_path:
                print(f"\n✅ Process completed successfully!")
                print(f"📄 Transcript saved as: {output_path}")
                
                # Ask if user wants to open the folder
                open_folder = input("\nOpen containing folder? (y/n): ").lower()
                if open_folder == 'y':
                    folder_path = os.path.dirname(output_path)
                    os.startfile(folder_path)
            else:
                print("\n❌ Failed to save transcript.")
        else:
            print("\n❌ Transcription failed.")
        
        print("\n" + "=" * 60)
        input("Press Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user.")
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()