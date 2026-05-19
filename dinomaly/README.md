# Noisy Dinomaly

The Dinomaly architecture (frozen ViT encoder + trainable linear-attention bottleneck/decoder) trained with **Memory-Distilled Selection (MeDS)** for robust industrial anomaly detection under data contamination.

Anomaly maps are computed via cosine similarity between encoder and decoder features. MeDS wraps the standard Dinomaly training loop in a three-stage pipeline (bootstrap memory ensemble → score distillation → progressive data selection) that maintains high performance even when the training set contains up to **40% mislabeled / anomalous samples**.

Reported MVTec-AD I-AUROC at 40% noise: **99.16%** (Table 1, paper).

> Looking for the framework overview, the INP-Former variant, or shared infrastructure? See the top-level [README](../README.md).

## Pipeline

### Stage 1 — Memory score generation

Extracts ViT features from training images, builds ensemble memory banks by random subsampling, and computes cosine-distance anomaly scores for every training sample.

```bash
bash scripts/run_step_1.sh
# or
python step_1_memory_score_generation.py \
    --dataset MVTec-AD \
    --data_path  /path/to/MVTec_nr10_seed_0 \
    --output_dir /path/to/memory_scores/nr10_seed_0 \
    --encoder dinov2reg_vit_base_14 \
    --ensemble_size 100 \
    --subsampling_ratio 0.1
```

| Argument | Default | Meaning |
|---|---|---|
| `--ensemble_size` | `100` | Number of bootstrapped memories `B` |
| `--subsampling_ratio` | `0.1` | Fraction `ρ` of patches kept per memory (paper §4.1) |
| `--dataset` | `MVTec-AD` | `MVTec-AD` or `VisA` |
| `--encoder` | `dinov2reg_vit_base_14` | Frozen backbone — see *Supported backbones* |

### Stage 2 — Model distillation

Trains the Dinomaly bottleneck + decoder to predict the Stage-1 memory anomaly scores using SmoothL1/MSE loss, jointly across all product categories.

```bash
bash scripts/run_step_2.sh
# or
python step_2_distillation.py \
    --memory_output_dir /path/to/memory_scores/nr10_seed_0 \
    --output_dir        /path/to/distilled/nr10_seed_0 \
    --loss_function l2 \
    --n_iters 10000 \
    --batch_size 16
```

| Argument | Default | Meaning |
|---|---|---|
| `--loss_function` | `l2` | Distillation loss — `l1` or `l2` |
| `--n_iters` | `10000` | Distillation training iterations |
| `--batch_size` | `16` | Joint batch across categories |

### Stage 3 — Data selection + fine-tuning

Uses the distilled model to score training samples, applies MAD-based outlier detection to filter noisy samples, and fine-tunes Dinomaly on the progressively selected clean subset `S_t`.

```bash
bash scripts/run_step_3_distilled.sh
# or
python step_3_data_selection_with_distilled_model.py \
    --distill_output_dir /path/to/distilled/nr10_seed_0 \
    --output_dir         /path/to/final/nr10_seed_0 \
    --mad_factor 3
```

| Argument | Default | Meaning |
|---|---|---|
| `--mad_factor` | `3` | MAD multiplier `k` (paper Eq. 12). Typical range 0–3 |
| `--distill_output_dir` | — | Path to Stage-2 distilled model |

An alternative Stage-3 script (`step_3_data_selection_with_memory_score.py` / `scripts/run_step_3_memory.sh`) selects samples using raw memory scores instead of the distilled model — used in the ablation (paper Table 4).

### Real-IAD

The Real-IAD benchmark uses single-class training (one model per product); the variants live in `real_iad/`:

```bash
python real_iad/step_1_memory_score_generation.py     --data_path /path/to/real_iad ...
python real_iad/step_2_distillation.py                --data_path /path/to/real_iad ...
python real_iad/step_3_data_selection_with_distilled_model.py     --data_path /path/to/real_iad ...
# sep_ variant separates fine-tuning per category instead of jointly:
python real_iad/step_3_data_selection_sep_with_distilled_model.py --data_path /path/to/real_iad ...
```

## Data preparation

Create a noisy version of MVTec-AD using the pre-computed CSVs in `data/MVTech/`:

```bash
python data/create_noisy_dataset.py \
    --original_dataset /path/to/mvtec_anomaly_detection \
    --csv_file        data/MVTech/train_mvtec_nr10_seed_0.csv \
    --output_dir      /path/to/MVTec_nr10_seed_0
```

CSV naming: `train_mvtec_nr{ratio}_seed_{seed}.csv` — `ratio ∈ {0, 10, 20, 40}`, `seed ∈ {0, 10, 20, 30}`. `nr0` is the original clean dataset.

The test split and ground-truth masks are copied as-is. The `train/good/` folder is rebuilt from the CSV: genuine good samples plus defective samples injected as noise (renamed with their defect type prefix so filenames don't collide).

## Model

**`ViTill`** (`models/uad.py`) — the default Dinomaly model. Frozen ViT encoder, trainable bottleneck (`bMlp`) and `LinearAttention` decoder. Reconstruction score = cosine distance between encoder features and reconstructed features.

Variants in the same file:
- `ViTillCat` — multi-scale feature concatenation
- `ViTillv2` / `ViTillv3` — alternate bottleneck designs
- `ViTAD` — pure ViT decoder
- `ReContrast` — contrastive variant

## Supported backbones

| Architecture | Sizes |
|---|---|
| DINOv2 (default) | ViT-S/14, ViT-B/14, ViT-L/14 |
| DINOv2 w/ registers | ViT-S/14, ViT-B/14, ViT-L/14 |
| DINOv1 | ViT-S/8, ViT-S/16, ViT-B/8, ViT-B/16 |
| MAE | ViT-B/16, ViT-L/16, ViT-H/14 |
| iBOT | ViT-S/16, ViT-B/16, ViT-L/16 |
| BEiTv2 | ViT-B/16, ViT-L/16 |
| D-iGPT | ViT-B/16, ViT-L/14 |
| DeiT | ViT-S/16, ViT-B/16 |

Selected via `--encoder`. Backbone code lives in `../shared/{dinov1,dinov2,beit}/` and is loaded via `_meds_paths.py`.

## Evaluation

`utils.evaluation_batch` returns `[I-AUROC, I-AP, I-F1, P-AUROC, P-AP, P-F1, P-AUPRO]`. It uses the GPU-accelerated `ader_evaluator` (backed by the [`adeval`](https://pypi.org/project/adeval/) library) by default and is roughly 10× faster than the CPU sklearn + `compute_pro` path. The CPU implementation is kept as a fallback:

```python
metrics = evaluation_batch(model, loader, device, use_adeval=False)
```

## Project structure

```
dinomaly/
├── README.md
├── _meds_paths.py                                    # adds ../shared to sys.path
├── dataset.py                                        # MVTec / VisA / Real-IAD / LOCO / AeBAD / DRAEM datasets
├── utils.py                                          # losses, anomaly maps, ader_evaluator, evaluation_batch
├── models/
│   ├── uad.py                                        # ViTill, ViTillCat, ViTAD, ViTillv2, ViTillv3, ReContrast
│   ├── vision_transformer.py                         # Block, LinearAttention, bMlp, FeatureJitter
│   └── vit_encoder.py                                # pretrained ViT loader
├── step_1_memory_score_generation.py                 # MVTec / VisA Stage 1
├── step_2_distillation.py                            # MVTec / VisA Stage 2
├── step_3_data_selection_with_distilled_model.py     # MVTec / VisA Stage 3 (default)
├── step_3_data_selection_with_memory_score.py        # MVTec / VisA Stage 3 (memory-score variant)
├── real_iad/                                         # Real-IAD Stage 1/2/3 variants
│   ├── _meds_paths.py
│   ├── step_1_memory_score_generation.py
│   ├── step_2_distillation.py
│   ├── step_3_data_selection_with_distilled_model.py
│   └── step_3_data_selection_sep_with_distilled_model.py
├── data/                                             # noisy-dataset generator + CSV files
│   ├── create_noisy_dataset.py
│   └── MVTech/train_mvtec_nr*_seed_*.csv
└── scripts/
    ├── run_step_1.sh
    ├── run_step_2.sh
    ├── run_step_3_distilled.sh
    └── run_step_3_memory.sh
```

## Requirements

See the top-level [requirements.txt](../requirements.txt). The shared backbones in `../shared/` are loaded automatically via `_meds_paths.py`; no installation step is needed for them.
