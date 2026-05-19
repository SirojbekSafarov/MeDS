#!/bin/bash

PYTHON_FILE="step_2_distillation.py"

# Array of noise ratios and seeds
noise_ratios=(0 10 20 40)
seeds=(0)
ensemble_size=(100)
subsampling_ratio=(0.1)
loss_function='l1' #l2

for ensemble in "${ensemble_size[@]}"; do
    for subsampling in "${subsampling_ratio[@]}"; do
        for ratio in "${noise_ratios[@]}"; do
            for seed in "${seeds[@]}"; do
                scaled=$(awk "BEGIN {printf \"%.0f\", $subsampling * 100}")
                echo "Running for noise ratio: $ratio, seed: $seed, ensemble: $ensemble, subsampling: $scaled"

                # MVTec-AD (default); for VisA switch DATA_PATH and --dataset
                DATA_PATH="/path/to/datasets/MVTec-AD-noisy/mvtec_noise_ratio${ratio}/MVTech_nr${ratio}_seed_${seed}"
                SAVE_DIR="/path/to/experiments/inpformer/mvtec/stage_2_distillation/noise_ratio_${ratio}/seed_${seed}/ensemble_${ensemble}_p${scaled}/${loss_function}"
                MEMORY_DIR="/path/to/experiments/dinomaly/mvtec/memory_scores/noise_ratio_${ratio}/seed_${seed}/ensemble_${ensemble}_p${scaled}"

                CUDA_VISIBLE_DEVICES=0 python $PYTHON_FILE \
                --data_path "$DATA_PATH" \
                --output_dir "$SAVE_DIR" \
                --memory_output_dir "$MEMORY_DIR" \
                --loss_function "$loss_function" \
                --dataset "MVTec-AD"

                echo "Cleaning up..."
                sleep 5  # give some time for CUDA memory to release
            done
        done
    done
done
