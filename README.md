# Memory-Distilled Selection (MeDS)

Reference implementation for the ICML 2026 submission **"Memory-Distilled Selection for Noise-Robust Anomaly Detection"** (Paper #5685, under review).

MeDS is a training framework that makes unsupervised anomaly detection robust to **contaminated training data** — i.e. training sets that contain an unknown fraction of anomalous samples. It is not a single model; it is a three-stage *training algorithm* that can be applied on top of existing AD baselines.

This repository provides MeDS applied to two strong baselines:

| Model | Folder | Backbone | Decoder |
|---|---|---|---|
| **Noisy Dinomaly** | [dinomaly/](dinomaly/) | DINOv2 ViT-S/B/L (+ many other ViTs) | linear-attention bottleneck |
| **Noisy INP-Former** | [inpformer/](inpformer/) | DINOv2 ViT-S/B/L | informative-patch prototypes |

Reported MVTec-AD I-AUROC at **40% noise**: **99.16% (Dinomaly+MeDS)** and **99.17% (INP-Former+MeDS)** — state-of-the-art on MVTec-AD, VisA, and Real-IAD under noisy settings. See the paper, Table 1.

---

## The MeDS pipeline

MeDS executes three stages in order:

```
            ┌───────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   train →  │ 1. Bootstrap      │ →  │ 2. Score         │ →  │ 3. Progressive   │
   data     │    memory         │    │    distillation  │    │    fine-tuning   │
            │    ensemble       │    │  (init s_θ)      │    │  on clean subset │
            └───────────────────┘    └──────────────────┘    └──────────────────┘
                low-pass filter        early-learning bias       self-selection
```

**Stage 1 — Bootstrap memory ensemble.** Build `B = 100` sparse memory banks by randomly subsampling patch features from a frozen ViT encoder. Sparse subsampling acts as a **low-pass filter** that separates normal from anomalous patches (paper §4.1, Theorem 1).

**Stage 2 — Memory-score distillation.** Train a small reconstruction-score network `s_θ` to predict the memory-bank anomaly score. The early-learning bias of neural nets sharpens the normal/anomaly boundary, fixing the limit imposed by the frozen encoder (paper §4.2).

**Stage 3 — Fine-tune with progressive selection.** Iteratively fine-tune `s_θ` on a self-selected clean subset `S_t`, growing the trusted set as the model improves (MAD-based thresholding, paper §4.3). This yields fine-grained pixel localization without overfitting to anomalous samples.

The result: stable performance **across all noise ratios** (0%, 10%, 20%, 40%) with no noise-ratio-specific hyperparameter tuning.

---

## Repository layout

```
MeDS/
├── README.md                  # this file — framework overview
├── LICENSE                    # MIT
├── requirements.txt           # combined deps for both models (loose pins)
├── install_packages.sh        # exact pins for reproducing paper numbers
├── Dockerfile                 # nvidia/cuda:11.8 + miniconda + pinned deps
├── release.sh                 # docker build + run wrapper
├── .gitignore
│
├── shared/                    # infrastructure used by both models
│   ├── dinov1/                # DINOv1 ViT
│   ├── dinov2/                # DINOv2 ViT (default backbone)
│   ├── beit/                  # BEiT ViT
│   ├── flops_profiler/        # FLOPs accounting
│   └── optimizers/            # StableAdamW, RAdam, AdaBelief, ...
│
├── dinomaly/                  # Noisy Dinomaly + MeDS
│   ├── README.md
│   ├── _meds_paths.py         # adds ../shared to sys.path
│   ├── dataset.py             # MVTec / VisA / Real-IAD datasets
│   ├── utils.py               # losses, anomaly maps, ader_evaluator (GPU metrics)
│   ├── models/uad.py          # ViTill, ViTillv2, ViTAD, ReContrast
│   ├── step_1_memory_score_generation.py
│   ├── step_2_distillation.py
│   ├── step_3_data_selection_with_distilled_model.py
│   ├── step_3_data_selection_with_memory_score.py
│   ├── real_iad/              # Real-IAD variants
│   ├── data/                  # data preparation utilities + noisy-CSV files
│   └── scripts/run_step_*.sh
│
└── inpformer/                 # Noisy INP-Former + MeDS
    ├── README.md
    ├── _meds_paths.py
    ├── dataset.py             # MVTec / Real-IAD datasets
    ├── utils.py               # ader_evaluator (GPU metrics)
    ├── aug_funcs.py           # rotation / flip / grey augmentations
    ├── models/uad.py          # INP_Former (informative-patch prototypes)
    ├── step_1_memory_score_generation.py
    ├── step_2_distillation.py
    ├── step_3_data_selection_with_distilled_model.py
    ├── step_3_data_selection_with_memory_score.py
    ├── inp_former_multi_class*.py    # multi-class baseline variants
    ├── baseline_original.py          # non-MeDS reference run
    └── scripts/run_step_*.sh
```

### Why `shared/` and `_meds_paths.py`?

Dinov1/Dinov2/BEiT backbones and optimizers are byte-identical between the two models, so they live once in `shared/`. Each model script imports `_meds_paths` as its first line; that helper prepends `MeDS/shared/` to `sys.path`, so existing imports like `from dinov1.utils import trunc_normal_` and `from optimizers import StableAdamW` work unchanged.

### Optimized metric backend

Both models compute pixel/image AUROC, AP, F1, and AUPRO through a single GPU-accelerated function — `ader_evaluator` — backed by the [`adeval`](https://pypi.org/project/adeval/) library. The Dinomaly side originally used a CPU sklearn + custom `compute_pro` path; it now calls `ader_evaluator` by default and falls back to the CPU path if `adeval` is not installed. To force the CPU path:

```python
metrics = evaluation_batch(model, loader, device, use_adeval=False)
```

---

## Installation

You have three options, in increasing order of reproducibility:

### Option A — `pip install -r requirements.txt` (loose pins)

```bash
git clone <this-repo> MeDS
cd MeDS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Good for quick experimentation against whatever versions of PyTorch / sklearn / etc. happen to resolve.

### Option B — `install_packages.sh` (exact pins, host install)

Reproduces the exact environment the paper numbers were obtained with: **PyTorch 2.1.2 + CUDA 11.8**, plus pinned numpy / sklearn / timm / opencv versions.

```bash
bash install_packages.sh
```

CUDA 11.8 + a recent NVIDIA driver must already be available on the host. See `install_packages.sh` for the full list of pins.

### Option C — Docker (fully isolated, recommended)

Provides a self-contained `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04` environment with Miniconda + all pinned dependencies pre-installed. Build & launch:

```bash
bash release.sh
```

`release.sh` builds the image (tag `meds:local` by default), removes any previous container of the same name, and launches an interactive shell with:

| Flag | Why |
|---|---|
| `--gpus all` | GPU passthrough (required for training and the GPU evaluator) |
| `--shm-size=16g` | DataLoader workers need shared memory beyond the 64 MB default |
| `--ipc=host` | NCCL / shared-tensor IPC across worker processes |
| `-v $DATA_HOST:/data` | Host dataset root (`MVTec-AD`, `VisA`, `Real-IAD`, noisy variants) |
| `-v $OUTPUT_HOST:/outputs` | Host output root (memory scores, checkpoints, metrics) |
| `-v $CODE_HOST:/workspace/MeDS` | Live source mount — edits don't need a rebuild |

Override defaults via env vars before running, e.g.:

```bash
IMAGE_NAME=my/meds:dev \
DATA_HOST=/mnt/datasets \
OUTPUT_HOST=/mnt/runs/meds \
bash release.sh
```

The container drops you in `/workspace/MeDS`. From there you can run the per-model pipeline (`cd dinomaly && bash scripts/run_step_1.sh`, etc.) — see the per-model READMEs.

#### `adeval`

`adeval` is optional but recommended — it makes pixel-level evaluation ~10× faster. It is installed by `install_packages.sh` and the Dockerfile. On hosts without it, `evaluation_batch` automatically falls back to a CPU sklearn + `compute_pro` path.

CUDA-enabled PyTorch (≥1.12) is required for the GPU evaluator and for training in any reasonable time.

---

## Datasets

| Dataset | Source | Used for |
|---|---|---|
| MVTec-AD | https://www.mvtec.com/company/research/datasets/mvtec-ad | both models, primary benchmark |
| VisA | https://github.com/amazon-science/spot-diff | both models |
| Real-IAD | https://realiad4ad.github.io/Real-IAD/ | Dinomaly (scripts in `dinomaly/real_iad/`) |

### Noisy dataset construction

To inject label noise into MVTec-AD, use the pre-computed CSV files in `dinomaly/data/MVTech/` (or generate your own):

```bash
cd MeDS/dinomaly
python data/create_noisy_dataset.py \
    --original_dataset /path/to/mvtec_anomaly_detection \
    --csv_file        data/MVTech/train_mvtec_nr10_seed_0.csv \
    --output_dir      /path/to/MVTec_nr10_seed_0
```

CSV naming: `train_mvtec_nr{ratio}_seed_{seed}.csv` with `nr ∈ {0, 10, 20, 40}` and `seed ∈ {0, 10, 20, 30}`. `nr0` is the original clean dataset.

---

## Quick start

Each model has identical step semantics; pick a folder and run the three stages in order.

```bash
cd MeDS/dinomaly            # or MeDS/inpformer
bash scripts/run_step_1.sh           # bootstrap memory ensemble → memory scores
bash scripts/run_step_2.sh           # distill memory scores → init s_θ
bash scripts/run_step_3_distilled.sh # progressive selection + fine-tune

# Alternative for stage 3: select using raw memory scores instead of the distilled model
bash scripts/run_step_3_memory.sh
```

See the per-model READMEs for argument-level documentation:

- [dinomaly/README.md](dinomaly/README.md)
- [inpformer/README.md](inpformer/README.md)

---
