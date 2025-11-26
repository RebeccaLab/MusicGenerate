#!/usr/bin/env python3
"""
Music Generation App using xLSTM
Launches a Gradio web interface for generating MIDI music.
"""

import os
import json
from typing import List

# -----------------------------
# Set CUDA_HOME if not already set
# -----------------------------
import subprocess
import shutil

def find_cuda_home():
    """Find CUDA installation directory"""
    # Check if already set
    if "CUDA_HOME" in os.environ:
        cuda_home = os.environ["CUDA_HOME"]
        if os.path.exists(cuda_home):
            return cuda_home
    
    # Try to find nvcc in PATH
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        # Extract CUDA_HOME from nvcc path (usually /path/to/cuda/bin/nvcc)
        cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        if os.path.exists(cuda_home):
            return cuda_home
    
    # Check common CUDA installation paths (including more version numbers)
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.7",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.5",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-11.8",
        "/usr/local/cuda-11.7",
        "/usr/local/cuda-11.0",
        "/opt/cuda",
        "/usr/local/cuda-*",  # Will be expanded by glob
    ]
    
    # Check explicit paths
    for path in cuda_paths:
        if "*" in path:
            continue
        if os.path.exists(path):
            # Verify it has the necessary components
            if os.path.exists(os.path.join(path, "bin", "nvcc")) or \
               os.path.exists(os.path.join(path, "include", "cuda.h")):
                return path
    
    # Try to find via glob pattern for versioned CUDA directories
    import glob
    for pattern in ["/usr/local/cuda-*", "/opt/cuda-*"]:
        matches = glob.glob(pattern)
        for match in sorted(matches, reverse=True):  # Try newest first
            if os.path.exists(os.path.join(match, "bin", "nvcc")) or \
               os.path.exists(os.path.join(match, "include", "cuda.h")):
                return match
    
    # Check /usr for system-installed CUDA
    if os.path.exists("/usr/bin/nvcc") or os.path.exists("/usr/include/cuda.h"):
        return "/usr"
    
    # Try to find via find command (more thorough search)
    try:
        # Search for nvcc in common locations
        for search_dir in ["/usr/local", "/opt", "/usr"]:
            result = subprocess.run(
                ["find", search_dir, "-name", "nvcc", "-type", "f"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                nvcc_path = result.stdout.strip().split('\n')[0]
                cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
                if os.path.exists(cuda_home):
                    return cuda_home
    except:
        pass
    
    # Last resort: check if PyTorch has CUDA and try to infer from torch installation
    try:
        import torch
        if torch.cuda.is_available() and hasattr(torch.version, 'cuda'):
            cuda_version = torch.version.cuda
            major_minor = '.'.join(cuda_version.split('.')[:2])
            
            # Try versioned paths first
            potential_paths = [
                f"/usr/local/cuda-{major_minor}",
                f"/usr/local/cuda-{cuda_version}",
                "/usr/local/cuda",
            ]
            
            for path in potential_paths:
                if os.path.exists(path) and (os.path.exists(os.path.join(path, "bin", "nvcc")) or 
                                            os.path.exists(os.path.join(path, "include", "cuda.h"))):
                    return path
            
            # If PyTorch can use CUDA, try to find where its CUDA libraries are
            # This might give us a hint about system CUDA location
            try:
                import torch.utils.cpp_extension
                # This will fail if CUDA_HOME not set, but we can catch it
                try:
                    cuda_paths = torch.utils.cpp_extension.include_paths(device_type="cuda")
                    if cuda_paths:
                        # Extract potential CUDA_HOME from include paths
                        for inc_path in cuda_paths:
                            # Include paths are usually like /path/to/cuda/include
                            potential_cuda = os.path.dirname(inc_path)
                            if os.path.exists(potential_cuda) and os.path.exists(os.path.join(potential_cuda, "bin", "nvcc")):
                                return potential_cuda
                except:
                    pass
            except:
                pass
    except:
        pass
    
    return None

# Initialize flag for CUDA toolkit detection
cuda_toolkit_missing = False

cuda_home = find_cuda_home()

# If not found, try /usr/local/cuda (common symlink location)
if not cuda_home and os.path.exists("/usr/local/cuda"):
    # Check if it's a symlink and resolve it
    if os.path.islink("/usr/local/cuda"):
        resolved = os.path.realpath("/usr/local/cuda")
        if os.path.exists(resolved):
            cuda_home = resolved
    elif os.path.exists("/usr/local/cuda/bin/nvcc") or os.path.exists("/usr/local/cuda/include/cuda.h"):
        cuda_home = "/usr/local/cuda"

if cuda_home:
    os.environ["CUDA_HOME"] = cuda_home
    print(f"✓ CUDA_HOME set to: {cuda_home}")
    
    # Verify nvcc is accessible
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        try:
            result = subprocess.run([nvcc_path, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✓ CUDA compiler found: {version_line}")
        except Exception as e:
            print(f"⚠ Warning: Could not verify nvcc: {e}")
else:
    # Final fallback: try to find CUDA through system libraries (before importing torch/xlstm)
    print("⚠ CUDA_HOME not found automatically.")
    print("  Searching for CUDA installation via system libraries...")
    
    # Try to find via ldconfig first (doesn't require imports)
    try:
        result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'libcudart' in line and '=>' in line:
                    lib_path = line.split('=>')[1].strip()
                    lib_real = os.path.realpath(lib_path)
                    if "/lib64/" in lib_real:
                        potential_cuda = lib_real.split("/lib64/")[0]
                    elif "/lib/" in lib_real:
                        potential_cuda = lib_real.split("/lib/")[0]
                    else:
                        potential_cuda = os.path.dirname(os.path.dirname(lib_real))
                    
                    if os.path.exists(potential_cuda) and (os.path.exists(os.path.join(potential_cuda, "bin", "nvcc")) or 
                                                          os.path.exists(os.path.join(potential_cuda, "include", "cuda.h"))):
                        cuda_home = potential_cuda
                        os.environ["CUDA_HOME"] = cuda_home
                        print(f"✓ Found CUDA via ldconfig: {cuda_home}")
                        break
    except Exception as e:
        pass

# Now import torch (can be done before xlstm)
import gradio as gr
import torch
import torch.nn.functional as F
import pretty_midi
import mido

# If still not found, try using torch to locate CUDA
if not cuda_home:
    try:
        if torch.cuda.is_available():
            # Try to get CUDA library path from PyTorch
            try:
                import ctypes
                import ctypes.util
                
                # Find libcudart (CUDA runtime library)
                libcudart_path = ctypes.util.find_library("cudart")
                if libcudart_path:
                    # Resolve the actual path
                    libcudart_real = os.path.realpath(libcudart_path)
                    # CUDA_HOME is typically the parent of lib64 or lib
                    if "/lib64/" in libcudart_real:
                        potential_cuda = libcudart_real.split("/lib64/")[0]
                    elif "/lib/" in libcudart_real:
                        potential_cuda = libcudart_real.split("/lib/")[0]
                    else:
                        potential_cuda = os.path.dirname(os.path.dirname(libcudart_real))
                    
                    if os.path.exists(potential_cuda) and (os.path.exists(os.path.join(potential_cuda, "bin", "nvcc")) or 
                                                          os.path.exists(os.path.join(potential_cuda, "include", "cuda.h"))):
                        cuda_home = potential_cuda
                        os.environ["CUDA_HOME"] = cuda_home
                        print(f"✓ Found CUDA via library path: {cuda_home}")
                    else:
                        # Try parent directories
                        for parent in [os.path.dirname(potential_cuda), os.path.dirname(os.path.dirname(potential_cuda))]:
                            if os.path.exists(parent) and (os.path.exists(os.path.join(parent, "bin", "nvcc")) or 
                                                          os.path.exists(os.path.join(parent, "include", "cuda.h"))):
                                cuda_home = parent
                                os.environ["CUDA_HOME"] = cuda_home
                                print(f"✓ Found CUDA via library path: {cuda_home}")
                                break
            except Exception as e:
                pass
    
    except ImportError:
        pass
    
    # If still not found, try common defaults (but verify they're valid)
    if 'cuda_home' not in locals() or not cuda_home:
        default_paths = ["/usr/local/cuda", "/opt/cuda"]
        for default_path in default_paths:
            if os.path.exists(default_path) and (os.path.exists(os.path.join(default_path, "include", "cuda.h")) or
                                                 os.path.exists(os.path.join(default_path, "bin", "nvcc"))):
                cuda_home = default_path
                os.environ["CUDA_HOME"] = cuda_home
                print(f"✓ Using default CUDA path: {cuda_home}")
                break
    
    # Final check - make sure CUDA_HOME is set and valid
    if 'cuda_home' not in locals() or not cuda_home:
        print("  ❌ Could not automatically find CUDA_HOME")
        print("  Attempting to find nvcc...")
        
        # Try one more search for nvcc in common locations
        search_dirs = ["/usr/local", "/opt", "/usr"]
        for search_dir in search_dirs:
            try:
                result = subprocess.run(
                    ["find", search_dir, "-name", "nvcc", "-type", "f"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    nvcc_found = result.stdout.strip().split('\n')[0]
                    # Extract CUDA_HOME (parent of bin directory)
                    cuda_home = os.path.dirname(os.path.dirname(nvcc_found))
                    os.environ["CUDA_HOME"] = cuda_home
                    print(f"✓ Found nvcc at: {nvcc_found}")
                    print(f"✓ Setting CUDA_HOME to: {cuda_home}")
                    break
            except Exception as e:
                continue
        
        # If still not found, give clear instructions
        if 'cuda_home' not in locals() or not cuda_home:
            print("\n" + "="*60)
            print("  CUDA Toolkit not found!")
            print("="*60)
            print("\n  PyTorch can use CUDA, but xlstm requires the CUDA Toolkit")
            print("  (which includes nvcc compiler) to compile CUDA code.\n")
            print("  You have two options:\n")
            print("  OPTION 1: Install CUDA Toolkit")
            print("  -----------------------------")
            print("  1. Download CUDA Toolkit from:")
            print("     https://developer.nvidia.com/cuda-downloads")
            print("  2. Install it (usually installs to /usr/local/cuda)")
            print("  3. Set CUDA_HOME:")
            print("     export CUDA_HOME=/usr/local/cuda")
            print("  4. Add to PATH:")
            print("     export PATH=$CUDA_HOME/bin:$PATH\n")
            print("  OPTION 2: Use CPU mode (temporary workaround)")
            print("  --------------------------------------------")
            print("  Edit app.py and change backend to 'vanilla' in the config")
            print("  (line ~106: backend=\"vanilla\")\n")
            print("="*60)
            print("\n  ⚠ CUDA Toolkit not found")
            print("  xlstm requires CUDA Toolkit to be installed (even for CPU mode)")
            print("  because it imports CUDA modules during startup.\n")
            
            cuda_toolkit_missing = True  # Flag to force CPU mode

# Import torch BEFORE xlstm to help refine CUDA_HOME if needed
import gradio as gr
import torch
import torch.nn.functional as F
import pretty_midi
import mido

# xlstm cannot import without a valid CUDA Toolkit installation
if cuda_toolkit_missing and not os.environ.get("CUDA_HOME"):
    # Don't set an invalid CUDA_HOME - let the validation below catch it
    pass

# -----------------------------
# Verify CUDA availability
# -----------------------------
# Default to CUDA if available, fall back to CPU only if CUDA fails

# Try CUDA first (default), fall back to CPU only if unavailable
cuda_available = torch.cuda.is_available() and not cuda_toolkit_missing
if cuda_available:
    try:
        # Test CUDA with a simple operation
        test_tensor = torch.zeros(1).cuda()
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_device_count = torch.cuda.device_count()
        print(f"✓ CUDA is available and working!")
        print(f"  Device: {cuda_device_name}")
        print(f"  Device count: {cuda_device_count}")
        print(f"  CUDA version: {torch.version.cuda}")
        device = "cuda"
        backend = "cuda"
    except Exception as e:
        print(f"⚠ CUDA detected but not working: {e}")
        print("  Falling back to CPU mode")
        cuda_available = False
        device = "cpu"
        backend = "vanilla"
else:
    # Fall back to CPU mode only if CUDA is not available
    if cuda_toolkit_missing:
        print("ℹ CUDA Toolkit not installed, using CPU mode (CUDA driver is available but toolkit needed)")
    else:
        print("ℹ CUDA not available, using CPU mode")
    device = "cpu"
    backend = "vanilla"  # CPU fallback

print(f"Using device: {device}, backend: {backend}")

# -----------------------------
# Verify CUDA_HOME is valid before importing xlstm
# -----------------------------
cuda_home_current = os.environ.get("CUDA_HOME", "")

# If CUDA toolkit was missing and CUDA_HOME is not set or invalid, fail early
if cuda_toolkit_missing:
    if not cuda_home_current or cuda_home_current == "/usr":
        # CUDA_HOME is not set or invalid - provide installation instructions
        print("\n" + "="*70)
        print("ERROR: CUDA Toolkit Required")
        print("="*70)
        print("\nxlstm requires the CUDA Toolkit to be installed.")
        print("Even if you want to use CPU mode, xlstm needs CUDA headers during import.\n")
        print("Your system has:")
        print("  ✓ CUDA Driver (PyTorch can use GPU)")
        print("  ✗ CUDA Toolkit (required by xlstm)\n")
        print("SOLUTION: Install CUDA Toolkit (System-wide)")
        print("-" * 70)
        print("\nCUDA Toolkit must be installed system-wide (venv cannot install it).")
        print("This requires administrator/sudo access.\n")
        print("1. Download CUDA Toolkit from:")
        print("   https://developer.nvidia.com/cuda-downloads")
        print("\n2. Select:")
        print("   - Operating System: Linux")
        print("   - Architecture: x86_64")
        print("   - Distribution: Your Linux distro")
        print("   - Version: 12.x (to match PyTorch CUDA 12.8)")
        print("   - Installer Type: deb (network) or deb (local)")
        print("\n3. Follow the installation commands shown on the download page.")
        print("   Example for Debian/Ubuntu:")
        print("   wget https://developer.download.nvidia.com/compute/cuda/repos/...")
        print("   sudo dpkg -i cuda-repo-*.deb")
        print("   sudo apt-key add /var/cuda-repo-*/7fa2af80.pub")
        print("   sudo apt-get update")
        print("   sudo apt-get -y install cuda-toolkit-12-x")
        print("\n4. After installation, verify:")
        print("   ls /usr/local/cuda/bin/nvcc")
        print("   # Should show: /usr/local/cuda/bin/nvcc")
        print("\n5. Set environment variables for this session:")
        print("   export CUDA_HOME=/usr/local/cuda")
        print("   export PATH=$CUDA_HOME/bin:$PATH")
        print("\n6. Add to ~/.bashrc to make permanent:")
        print("   echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc")
        print("   echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc")
        print("   source ~/.bashrc")
        print("\n7. Run the app again:")
        print("   python3 app.py")
        print("\nHELPER SCRIPT:")
        print("   A helper script is available: install_cuda_toolkit.sh")
        print("   Run it with: sudo ./install_cuda_toolkit.sh")
        print("   (It provides guidance but you may need to follow official install steps)")
        print("\nNOTE: If you don't have sudo access, contact your system administrator.")
        print("="*70)
        raise RuntimeError(
            "CUDA Toolkit is required. Please install it from "
            "https://developer.nvidia.com/cuda-downloads and set CUDA_HOME."
        )

if cuda_home_current:
    # Check if CUDA_HOME points to a valid CUDA installation
    cuda_home_valid = (
        os.path.exists(cuda_home_current) and
        (os.path.exists(os.path.join(cuda_home_current, "include", "cuda.h")) or
         os.path.exists(os.path.join(cuda_home_current, "bin", "nvcc")) or
         os.path.exists(os.path.join(cuda_home_current, "lib64", "libcudart.so")) or
         os.path.exists(os.path.join(cuda_home_current, "lib", "libcudart.so")))
    )
    
    if not cuda_home_valid:
        print("\n" + "="*70)
        print("ERROR: CUDA Toolkit not properly installed")
        print("="*70)
        print(f"\nCUDA_HOME is set to: {cuda_home_current}")
        print("But this path doesn't contain a valid CUDA Toolkit installation.")
        print("\nxlstm requires the CUDA Toolkit (not just the driver) to work.")
        print("Even in CPU mode, xlstm needs CUDA headers during import.\n")
        print("SOLUTION: Install CUDA Toolkit")
        print("-" * 70)
        print("1. Download CUDA Toolkit from:")
        print("   https://developer.nvidia.com/cuda-downloads")
        print("2. Select:")
        print("   - OS: Linux")
        print("   - Architecture: x86_64")
        print("   - Distribution: Your Linux distro (e.g., Ubuntu/Debian)")
        print("   - Version: 12.x (matches your PyTorch CUDA 12.8)")
        print("3. Follow installation instructions (usually .deb package)")
        print("4. After installation, verify:")
        print("   ls /usr/local/cuda/bin/nvcc")
        print("5. Set CUDA_HOME:")
        print("   export CUDA_HOME=/usr/local/cuda")
        print("   export PATH=$CUDA_HOME/bin:$PATH")
        print("6. Run the app again:")
        print("   python3 app.py")
        print("="*70)
        raise RuntimeError(
            f"CUDA_HOME={cuda_home_current} is not a valid CUDA Toolkit installation. "
            "Please install CUDA Toolkit and set CUDA_HOME to the installation directory."
        )

# -----------------------------
# Model Configuration
# -----------------------------
try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig,
    )
except OSError as e:
    if "CUDA_HOME" in str(e):
        print("\n" + "="*70)
        print("ERROR: xlstm import failed - CUDA Toolkit issue")
        print("="*70)
        print("\nxlstm tried to import CUDA modules but CUDA_HOME is not valid.")
        print("\nSOLUTION: Install CUDA Toolkit")
        print("-" * 70)
        print("1. Download from: https://developer.nvidia.com/cuda-downloads")
        print("2. Install the toolkit (not just driver)")
        print("3. Set CUDA_HOME to installation path (usually /usr/local/cuda)")
        print("4. Run again")
        print("="*70)
        raise RuntimeError("CUDA Toolkit required. Please install it and set CUDA_HOME.") from e
    else:
        raise

# Configure xLSTM model
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend=backend,  # Automatically set based on CUDA availability
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=256,  # Sequence length
    num_blocks=7,  # Number of stacked blocks
    embedding_dim=128,  # Embedding dimension
    slstm_at=[1],  # Which blocks use sLSTM
)

# -----------------------------
# Config / Paths
# -----------------------------
STYLE_MIDI_MAP = {
    "country": "midi_styles/country.mid",
    "hiphop": "midi_styles/hiphop.mid",
    "jazz": "midi_styles/jazz.mid",
    "rock": "midi_styles/rock.mid",
}

WORD_TO_ID_PATH = "word_to_id.json"
ID_TO_WORD_PATH = "id_to_word.json"
CHECKPOINT_PATH = "xlstm_best_model.pt"

# -----------------------------
# Load vocab
# -----------------------------
def load_vocab(word_to_id_path=WORD_TO_ID_PATH, id_to_word_path=ID_TO_WORD_PATH):
    with open(word_to_id_path, "r") as f:
        word_to_id = json.load(f)
    with open(id_to_word_path, "r") as f:
        raw = json.load(f)
        id_to_word = {int(k): v for k, v in raw.items()}
    return word_to_id, id_to_word

word_to_id, id_to_word = load_vocab()

# -----------------------------
# Model loading
# -----------------------------
model = None
embedding = None
classifier = None

try:
    model = xLSTMBlockStack(cfg).to(device)
    embedding = torch.nn.Embedding(303, cfg.embedding_dim).to(device)
    classifier = torch.nn.Linear(cfg.embedding_dim, 303).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # Try to load with strict=False to handle shape mismatches
        try:
            model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
            embedding.load_state_dict(ckpt.get("embedding_state_dict", {}), strict=False)
            classifier.load_state_dict(ckpt.get("classifier_state_dict", {}), strict=False)
            print("Model loaded successfully (some weights may not match - using strict=False)")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Continuing with uninitialized model weights...")
    else:
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")
except Exception as e:
    print(f"Model init failed: {e}")
    raise


# -----------------------------
# Generation
# -----------------------------
def generate_music(start_tokens, model, embedding, classifier,
                   word_to_id, id_to_word, max_length=1024, device="cpu"):

    if model is None or embedding is None or classifier is None:
        raise RuntimeError("Model / embedding / classifier not initialized. Make sure cfg and checkpoint are loaded.")

    model.eval()
    generated = [word_to_id[t] for t in start_tokens]

    for _ in range(max_length):
        input_ids = torch.tensor(
            generated[-cfg.context_length:], dtype=torch.long
        ).unsqueeze(0).to(device)
        x_embed = embedding(input_ids)

        with torch.no_grad():
            output = model(x_embed)
            logits = classifier(output)

        prob = torch.softmax(logits[0, -1], dim=-1)
        next_id = torch.multinomial(prob, 1).item()
        generated.append(next_id)

        if id_to_word[next_id].startswith("[END]"):
            break

    return [id_to_word[i] for i in generated]


# -----------------------------
# Token → MIDI
# -----------------------------
def tokens_to_midi(tokens, bpm=120):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    beat = 60.0 / bpm
    grid = beat / 4  # resolution
    time = 0

    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t.startswith("DELTA_"):
            delta = int(t.split("_")[1])
            time += delta * grid
            i += 1

        elif t.startswith("PITCH_") and i + 1 < len(tokens) and tokens[i + 1].startswith("VEL_"):
            pitch = int(tokens[i].split("_")[1])
            vel = min(int(tokens[i + 1].split("_")[1]) * 8, 127)
            inst.notes.append(
                pretty_midi.Note(velocity=vel, pitch=pitch, start=time, end=time + grid)
            )
            i += 2
        else:
            i += 1

    midi.instruments.append(inst)
    return midi


# -----------------------------
# MIDI ↔ Pianoroll Helpers
# -----------------------------
def midi_to_pianoroll(midi_path, fs=8):
    """
    Read MIDI file and convert to piano-roll: [128, T] tensor
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    roll = torch.tensor(pm.get_piano_roll(fs=fs), dtype=torch.float32)  # [128, T]
    return roll


def pianoroll_to_midi(piano_roll, fs=8, program=0):
    """
    Convert piano-roll: [128, T] back to single instrument MIDI
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    piano_roll = piano_roll > 0  # bool mask
    notes, frames = piano_roll.shape

    for pitch in range(notes):
        row = piano_roll[pitch]
        on = False
        start = 0

        for i, val in enumerate(row):
            if val and not on:
                on = True
                start = i
            elif not val and on:
                on = False
                end = i
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start / fs,
                        end=end / fs,
                    )
                )

        # If still playing at the end, close it
        if on:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start / fs,
                    end=frames / fs,
                )
            )

    midi.instruments.append(instrument)
    return midi


# -----------------------------
# Mix two MIDI (Pianoroll mix)
# -----------------------------
def mix_midi(user_midi_path, style, alpha=0.5, fs=8):
    """
    Mix two MIDI files using piano-roll blending:
    alpha closer to 1 -> more style MIDI
    alpha closer to 0 -> more generated MIDI
    """
    if style not in STYLE_MIDI_MAP:
        return user_midi_path

    style_path = STYLE_MIDI_MAP[style]
    if not os.path.exists(style_path):
        return user_midi_path

    # Convert to piano-roll
    pr1 = midi_to_pianoroll(user_midi_path, fs=fs)   # Generated
    pr2 = midi_to_pianoroll(style_path, fs=fs)       # Style MIDI

    # Align lengths
    T = max(pr1.shape[1], pr2.shape[1])
    if pr1.shape[1] < T:
        pr1 = F.pad(pr1, (0, T - pr1.shape[1]))
    if pr2.shape[1] < T:
        pr2 = F.pad(pr2, (0, T - pr2.shape[1]))

    # Mix: higher alpha = more style
    mixed_roll = alpha * pr2 + (1.0 - alpha) * pr1

    # Binarize: on/off
    mixed_roll = (mixed_roll > 0.5).float()

    # Convert back to MIDI
    midi = pianoroll_to_midi(mixed_roll, fs=fs, program=0)
    out_path = f"mixed_{style}.mid"
    midi.write(out_path)
    return out_path


# -----------------------------
# Gradio Call Function
# -----------------------------
def run(start_genre, delta, pitch, vel, alpha, do_mix, mix_style):

    start_tokens = [
        f"[GENRE_{start_genre.upper()}]",
        f"DELTA_{int(delta)}",
        f"PITCH_{int(pitch)}",
        f"VEL_{int(vel)}"
    ]

    tokens = generate_music(start_tokens, model, embedding, classifier,
                            word_to_id, id_to_word, device=device)

    midi = tokens_to_midi(tokens)
    out = "generated.mid"
    midi.write(out)

    if do_mix:
        return mix_midi(out, mix_style, alpha=float(alpha))

    return out


# -----------------------------
# Gradio UI
# -----------------------------
if __name__ == "__main__":
    with gr.Blocks(title="MIDI Generator") as demo:
        gr.Markdown("## 🎵 MIDI Generator with Style Mix")

        genre = gr.Dropdown(["CLASSICAL", "COUNTRY", "HIPHOP", "JAZZ", "ROCK"],
                            label="Genre")
        delta = gr.Number(value=5, label="DELTA")
        pitch = gr.Number(value=95, label="PITCH")
        vel = gr.Number(value=10, label="VEL")

        alpha = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.05,
            label="Mix Alpha (0 = Pure Generated, 1 = Pure Style)"
        )

        do_mix = gr.Checkbox(label="Mix with style MIDI", value=False)
        mix_style = gr.Dropdown(
            list(STYLE_MIDI_MAP.keys()),
            label="Mix Style",
            value="jazz"
        )

        btn = gr.Button("Generate MIDI")
        out_file = gr.File(label="Output MIDI")

        btn.click(run, [genre, delta, pitch, vel, alpha, do_mix, mix_style], out_file)

    demo.launch(share=False)

