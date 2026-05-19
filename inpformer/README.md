# Noisy INP-Former

INP-Former (Informative Patches Transformer) trained with the **Memory-Distilled Selection (MeDS)** framework for robust industrial anomaly detection under data contamination.

INP-Former represents each product category through a small set of *informative patch prototypes* mined from the training data; the anomaly score is the distance between a test patch and its nearest prototype. MeDS wraps this baseline in a three-stage pipeline (bootstrap memory ensemble → score distillation → progressive data selection) that maintains high detection accuracy even when up to **40% of the training data is anomalous**.

Reported MVTec-AD I-AUROC at 40% noise: **99.17%** (Table 1, paper).

> Looking for the framework overview, the Dinomaly variant, or shared infrastructure? See the top-level [README](../README.md).

## Pipeline

The three stages mirror the Dinomaly variant — same CLI, same outputs, different model class.

### Stage 1 — Memory score generation

Extracts ViT features from the noisy training set, builds an ensemble of `B = 100` sparse memory banks by random subsampling, and writes a per-sample memory anomaly score.

```bash
bash scripts/run_step_1.sh
# or
python step_1_memory_score_generation.py \
    --dataset MVTec-AD \
    --data_path  /path/to/MVTec_nr10_seed_0 \
    --output_dir /path/to/memory_scores/nr10_seed_0 \
    --encoder dinov2reg_vit_base_14 \
    --ensemble_size 100 \
    --subsampling_ratio 0.1 \
    --device cuda:0
```

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `MVTec-AD` | `MVTec-AD` or `VisA` |
| `--ensemble_size` | `100` | Number of bootstrapped memories `B` |
| `--subsampling_ratio` | `0.1` | Fraction `ρ` of patches kept per memory (paper §4.1) |
| `--encoder` | `dinov2reg_vit_base_14` | Frozen backbone — see *Supported backbones* |

### Stage 2 — Memory score distillation

Initializes an INP-Former reconstruction-score network from scratch and distills the Stage-1 memory scores into it. Early-learning bias sharpens the normal/anomaly boundary beyond what the frozen ViT alone can express.

```bash
bash scripts/run_step_2.sh
# or
python step_2_distillation.py \
    --dataset MVTec-AD \
    --data_path        /path/to/MVTec_nr10_seed_0 \
    --memory_output_dir /path/to/memory_scores/nr10_seed_0 \
    --output_dir       /path/to/distilled/nr10_seed_0 \
    --loss_function l2 \
    --n_iters 10000 \
    --batch_size 16 \
    --INP_num 6
```

| Argument | Default | Meaning |
|---|---|---|
| `--loss_function` | `l2` | Distillation loss — `l1` or `l2` |
| `--n_iters` | `10000` | Distillation training iterations |
| `--INP_num` | `6` | Number of informative patch prototypes per category |

### Stage 3 — Progressive selection + fine-tuning

Uses the distilled model to score each training sample, applies MAD-based outlier filtering, and fine-tunes INP-Former on the progressively expanding clean subset `S_t`.

```bash
bash scripts/run_step_3_distilled.sh
# or
python step_3_data_selection_with_distilled_model.py \
    --dataset MVTec-AD \
    --data_path        /path/to/MVTec_nr10_seed_0 \
    --distill_output_dir /path/to/distilled/nr10_seed_0 \
    --output_dir       /path/to/final/nr10_seed_0 \
    --mad_factor 3 \
    --beta_end 0.5 \
    --n_iters 10000
```

| Argument | Default | Meaning |
|---|---|---|
| `--mad_factor` | `3` | MAD multiplier `k` (paper Eq. 12). Typical range 0–3 |
| `--beta_end` | `0.5` | Fraction of total iterations during which memory scores still influence selection |
| `--distill_initialized` | `True` | Warm-start fine-tuning from the Stage-2 model |

There is also an alternative Stage-3 script (`step_3_data_selection_with_memory_score.py` / `scripts/run_step_3_memory.sh`) that selects samples using the raw memory scores from Stage 1 instead of the distilled model — useful for ablation against Table 4.

### Baseline (no MeDS)

`baseline_original.py` runs INP-Former training on the noisy data without MeDS — the reference number against which the MeDS gains are measured.

```bash
bash scripts/run_baseline.sh
```

### Multi-class variants

`inp_former_multi_class.py`, `inp_former_multi_class_stage2_distillation.py`, `inp_former_multi_class_stage3_data_selection.py` — original multi-class implementations from the paper authors, kept verbatim for reproducibility.

## Model

**`INP_Former`** (`models/uad.py`) — implements informative-patch prototypes on top of a frozen DINOv2 backbone. Key components:

- **Aggregation blocks** (`models/vision_transformer.py:Aggregation_Block`) — pool patch tokens into a fixed number `INP_num` of prototype slots
- **Prototype blocks** (`models/vision_transformer.py:Prototype_Block`) — cross-attend prototypes against patch features; the anomaly map is built from the resulting attention distances
- Stores `self.distance` for visualization — used by `evaluation_batch_vis_ZS` in `utils.py`

## Supported backbones

| Architecture | Sizes |
|---|---|
| DINOv2 (default) | ViT-S/14, ViT-B/14, ViT-L/14 |
| DINOv2 w/ registers | ViT-S/14, ViT-B/14, ViT-L/14 |
| DINOv1 | ViT-S/8, ViT-S/16, ViT-B/8, ViT-B/16 |
| MAE | ViT-B/16, ViT-L/16, ViT-H/14 |
| iBOT | ViT-S/16, ViT-B/16, ViT-L/16 |
| BEiTv2 | ViT-B/16, ViT-L/16 |

Selected via `--encoder` (e.g. `dinov2reg_vit_base_14`, `dinov1_vit_small_8`). The backbone code lives in `../shared/{dinov1,dinov2,beit}/` and is shared with the Dinomaly variant.

## Evaluation

All metrics (image-level AUROC / AP / F1, pixel-level AUROC / AP / F1 / AUPRO) are computed by the GPU-accelerated `ader_evaluator` in `utils.py`. It batches the pixel-AUPRO and pixel-AUROC computations on CUDA and is roughly an order of magnitude faster than the equivalent CPU sklearn + custom `compute_pro` path. Install `adeval` from PyPI to enable it.

## Project structure

```
inpformer/
├── README.md
├── _meds_paths.py                                    # adds ../shared to sys.path
├── dataset.py                                        # MVTec / Real-IAD dataset classes
├── utils.py                                          # ader_evaluator, evaluation_batch, losses
├── aug_funcs.py                                      # rotation / translation / hflip / grey augmentations
├── models/
│   ├── uad.py                                        # INP_Former model
│   ├── vision_transformer.py                         # Mlp, Aggregation_Block, Prototype_Block
│   └── vit_encoder.py                                # pretrained ViT loader
├── step_1_memory_score_generation.py
├── step_2_distillation.py
├── step_3_data_selection_with_distilled_model.py
├── step_3_data_selection_with_memory_score.py
├── baseline_original.py                              # INP-Former without MeDS
├── inp_former_multi_class.py                         # multi-class baseline
├── inp_former_multi_class_stage2_distillation.py
├── inp_former_multi_class_stage3_data_selection.py
└── scripts/
    ├── run_step_1.sh
    ├── run_step_2.sh
    ├── run_step_3_distilled.sh
    ├── run_step_3_memory.sh
    └── run_baseline.sh
```

## Requirements

See the top-level [requirements.txt](../requirements.txt). The shared backbones in `../shared/` are loaded automatically via `_meds_paths.py`; no installation step is needed for them.
