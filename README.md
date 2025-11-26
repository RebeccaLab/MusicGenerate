# MusicGenerate

Neural music generation using xLSTM to create MIDI sequences with style mixing.

## Setup

1. **Set up Python environment with pyenv and venv**:
```bash
# Initialize pyenv in current shell (if not already in PATH)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install Python version (e.g., 3.11)
pyenv install 3.11.0
pyenv local 3.11.0

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install torch torchvision torchaudio xlstm pretty_midi mido gradio note-seq jupyter
```

3. **Required files** (should already be present):
   - `xlstm_best_model.pt`
   - `word_to_id.json`
   - `id_to_word.json`
   - `midi_styles/*.mid` files

## Usage

1. **Activate virtual environment** (if not already active):
   ```bash
   source venv/bin/activate
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook 20251114.ipynb
   ```

3. **Use the interface**:
   - Select genre (Classical, Country, Hip-hop, Jazz, Rock)
   - Set DELTA, PITCH, VEL parameters
   - Optional: Enable style mixing and adjust alpha (0 = generated, 1 = style)
   - Click "Generate MIDI" and download the result
