#!/usr/bin/env bash
# MeDS — pinned-version installer for a reproducible CUDA 11.8 / PyTorch 2.1.2 env.
#
# Use this when you want exact pins (recommended for reproducing paper numbers).
# For loose pins use `pip install -r requirements.txt` instead.
#
# Usage:  bash install_packages.sh

set -euo pipefail

# --- PyTorch (CUDA 11.8) --------------------------------------------------
pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# --- Core scientific stack ------------------------------------------------
pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0 \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    scikit-image==0.22.0 \
    sympy==1.12

# --- Vision / image I/O ---------------------------------------------------
pip install --no-cache-dir \
    Pillow==10.2.0 \
    opencv-python==4.9.0.80

# --- Model utilities ------------------------------------------------------
pip install --no-cache-dir \
    timm==0.9.16 \
    ptflops==0.7

# --- Plotting / CLI / logging ---------------------------------------------
pip install --no-cache-dir \
    matplotlib==3.8.4 \
    tabulate==0.9.0 \
    tqdm==4.66.1 \
    colorama==0.4.6 \
    natsort==8.4.0

# --- GPU-accelerated AD evaluator (optional but recommended) --------------
# `ader_evaluator` in utils.py falls back to the CPU sklearn path if absent.
pip install --no-cache-dir adeval || \
    echo "[WARN] adeval install failed — evaluation_batch will use the CPU fallback."

echo "[OK] MeDS dependencies installed."
