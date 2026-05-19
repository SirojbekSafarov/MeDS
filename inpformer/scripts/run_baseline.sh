#!/bin/bash

PYTHON_FILE="baseline_original.py"

# Array of noise ratios and seeds
noise_ratios=(10)
seeds=(0 10 20 30)

for ratio in "${noise_ratios[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Running for noise ratio: $ratio, seed: $seed"

        # VisA (default); for MVTec-AD switch DATA_PATH and --dataset
        DATA_PATH="/path/to/datasets/VisA-noisy/visa_noise_ratio${ratio}/VisA_20220922_pytorch_nr${ratio}_seed_${seed}"
        SAVE_DIR="/path/to/experiments/inpformer/visa/baseline/noise_ratio_${ratio}/seed_${seed}"

        CUDA_VISIBLE_DEVICES=0 python $PYTHON_FILE \
        --data_path "$DATA_PATH" \
        --output_dir "$SAVE_DIR" \
        --dataset "VisA"  # or "MVTec-AD"

        echo "Cleaning up..."
        sleep 5  # give some time for CUDA memory to release
    done
done
