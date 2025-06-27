#!/usr/bin/env python
# ecg_autoencoder_example.py
"""
Simple demo: create synthetic ECG images + time-series and train
a 1-D convolutional auto-encoder on 4-second single-lead strips.
No IPython/Jupyter is required.
"""

import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageDraw

import neurokit2 as nk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------------------------------
# 1. CONFIG
# -----------------------------------------------------------------------------
OUT_DIR = "../data/ecg_demo"
N_SAMPLES = 100          # keep it tiny for a quick run; scale up later
SAMPLING_RATE = 500      # Hz
DURATION_S = 4           # seconds  -> 500*4 = 2000 samples
SIG_LEN = int(SAMPLING_RATE * DURATION_S)
IMG_W, IMG_H = 600, 200  # pixels for the output PNG
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "timeseries"), exist_ok=True)

# -----------------------------------------------------------------------------
# 2. HELPER: ADD “PAPER” GRID + VISUAL NOISE
# -----------------------------------------------------------------------------
def add_paper_background(img: Image.Image) -> Image.Image:
    """Overlay ECG paper grid and some random artefacts."""
    draw = ImageDraw.Draw(img)
    # small boxes: 1 mm every 5 px (scale is arbitrary here)
    for x in range(0, IMG_W, 25):  # dark lines every 5 boxes
        width = 2 if x % 125 == 0 else 1
        draw.line([(x, 0), (x, IMG_H)], fill=(200, 200, 200), width=width)
    for y in range(0, IMG_H, 25):
        width = 2 if y % 125 == 0 else 1
        draw.line([(0, y), (IMG_W, y)], fill=(200, 200, 200), width=width)
    # random small scribbles / noise (5–10 per image)
    for _ in range(random.randint(5, 10)):
        x0 = random.randint(0, IMG_W - 20)
        y0 = random.randint(0, IMG_H - 10)
        x1 = x0 + random.randint(5, 20)
        y1 = y0 + random.randint(2, 10)
        draw.line([(x0, y0), (x1, y1)], fill=(150, 150, 150), width=1)
    return img

# -----------------------------------------------------------------------------
# 3. SYNTHETIC DATASET
# -----------------------------------------------------------------------------
def create_one_sample(idx: int):
    # Simulate a clean ECG
    hr = random.randint(50, 100)               # random heart-rate
    ecg = nk.ecg_simulate(duration=DURATION_S,
                          sampling_rate=SAMPLING_RATE,
                          heart_rate=hr,
                          noise=0)             # keep underlying series noise-free

    # Random shift: rotate the signal so that the first R-peak occurs at a
    # random point within the 1st second, not always at t=0
    shift = random.randint(0, SAMPLING_RATE)   # ≤1 s
    ecg = np.roll(ecg, shift)

    # Save raw series for the auto-encoder
    np.save(os.path.join(OUT_DIR, "timeseries", f"ecg_{idx:04d}.npy"), ecg)

    # Plot to PNG with grid
    fig, ax = plt.subplots(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    ax.plot(np.linspace(0, DURATION_S, SIG_LEN), ecg, lw=1, color="black")
    ax.axis("off")
    # tight layout so waveform fills the canvas
    plt.subplots_adjust(0, 0, 1, 1)

    # Draw to an RGBA buffer and convert to PIL to add grid
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    png = np.asarray(renderer.buffer_rgba())[:, :, :3]  # Discard alpha if needed
    png = png.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(png)
    plt.close(fig)

    img = add_paper_background(img)
    img.save(os.path.join(OUT_DIR, "images", f"ecg_{idx:04d}.png"))

print("Generating synthetic ECG strips...")
for i in range(N_SAMPLES):
    create_one_sample(i)
print(f"Saved {N_SAMPLES} PNGs + .npy files in {OUT_DIR}/")

# -----------------------------------------------------------------------------
# 4. LOAD DATA FOR AUTO-ENCODER
# -----------------------------------------------------------------------------
X = np.stack([
    np.load(os.path.join(OUT_DIR, "timeseries", f))
    for f in sorted(os.listdir(os.path.join(OUT_DIR, "timeseries")))
])
# shape → (batch, timesteps, 1)
X = X[..., np.newaxis].astype("float32")

# Simple train/val split
split = int(0.8 * N_SAMPLES)
X_train, X_val = X[:split], X[split:]

# -----------------------------------------------------------------------------
# 5. BUILD & TRAIN 1-D CONVOLUTIONAL AUTO-ENCODER
# -----------------------------------------------------------------------------
TIMESTEPS = SIG_LEN

def build_autoencoder():
    inputs = keras.Input(shape=(TIMESTEPS, 1))
    x = layers.Conv1D(16, 7, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 5, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling1D(2)(x)              # latent dim ~ 500×32

    x = layers.Conv1D(32, 5, activation="relu", padding="same")(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 7, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1D(1, 7, activation="linear", padding="same")(x)

    autoencoder = keras.Model(inputs, decoded, name="ecg_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


model = build_autoencoder()
model.summary(line_length=120)

print("\nTraining auto-encoder (quick demo, 5 epochs)…")
history = model.fit(
    X_train, X_train,
    epochs=5,
    batch_size=16,
    validation_data=(X_val, X_val),
    verbose=1
)

# -----------------------------------------------------------------------------
# 6. QUICK VISUAL CHECK OF RECONSTRUCTIONS + PNG PREVIEW
# -----------------------------------------------------------------------------
from PIL import Image

timeseries_dir = os.path.join(OUT_DIR, "timeseries")
ts_files       = sorted(os.listdir(timeseries_dir))
val_files      = ts_files[split:]                     # filenames that ended up in X_val

if len(X_val) > 0:
    idx   = random.randint(0, len(X_val) - 1)        # pick a random validation sample
    original = X_val[idx].squeeze()
    recon    = model.predict(X_val[idx: idx + 1])[0].squeeze()

    # ------------------------------------------------------
    # find & load the matching PNG image
    # ------------------------------------------------------
    # e.g. "ecg_0083.npy"  ->  "../images/ecg_0083.png"
    npy_name  = val_files[idx]
    png_name  = npy_name.replace(".npy", ".png")
    png_path  = os.path.join(OUT_DIR, "images", png_name)
    ecg_img   = Image.open(png_path)

    # ------------------------------------------------------
    # build a 1×2 figure (left: PNG, right: overlayed signals)
    # ------------------------------------------------------
    t = np.arange(TIMESTEPS) / SAMPLING_RATE

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.5),
                            gridspec_kw={"width_ratios": [1.2, 2]})

    # --- left panel: show the image exactly as generated ---
    axs[0].imshow(ecg_img)
    axs[0].set_title("Input 4 s ECG image")
    axs[0].axis("off")

    # --- right panel: original vs reconstruction ---
    axs[1].plot(t, original, label="Original", lw=1)
    axs[1].plot(t, recon,   label="Reconstruction", lw=1, linestyle="--")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title("Auto-encoder reconstruction")
    axs[1].legend()

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "example_reconstruction.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"\nSaved combined image + reconstruction plot to {out_png}")

