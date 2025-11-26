# MusicGenerate

Neural music generation using xLSTM to create MIDI sequences with style mixing.

## Setup
- Use pyenv to install python 3.13.0
```bash
pyenv install 3.13.0
pyenv shell 3.13.0
python -m venv env
source env/bin/activate
```
- Install dependencies
```bash
pip install -r requirements.txt
```

**Required files** (should already be present):
   - `xlstm_best_model.pt`
   - `word_to_id.json`
   - `id_to_word.json`
   - `midi_styles/*.mid` files

## Usage

1. **Activate virtual environment** (if not already active):
   ```bash
   source env/bin/activate
   ```

2. **Run the app**:
   ```bash
   python app.py
   ```

3. **Access the interface**:
   - The app will launch locally (typically at `http://127.0.0.1:7860`)
   - Select genre (Classical, Country, Hip-hop, Jazz, Rock)
   - Set DELTA, PITCH, VEL parameters
   - Optional: Enable style mixing and adjust alpha (0 = generated, 1 = style)
   - Click "Generate MIDI" and download the result

**Note**: The app is configured for CPU mode by default. If you have CUDA installed and want to use GPU, change `backend="vanilla"` to `backend="cuda"` in `app.py`.
