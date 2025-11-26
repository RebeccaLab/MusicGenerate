#!/usr/bin/env python3
"""
Music Generation App using xLSTM
Launches a Gradio web interface for generating MIDI music.
"""

import gradio as gr
import torch
import torch.nn.functional as F
import json
import os
import pretty_midi
import mido
from typing import List

# -----------------------------
# Model Configuration
# -----------------------------
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

# Configure xLSTM model
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla",  # Use CPU (change to "cuda" if GPU available)
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

