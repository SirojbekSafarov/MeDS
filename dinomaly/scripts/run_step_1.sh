#!/bin/bash

# Set Python file name
PYTHON_FILE="step_1_memory_score_generation.py"

# Array of noise ratios and seeds
noise_ratios=(40)
seeds=(0 10 20 30)
ensemble_size=(100)
subsampling_ratio=(0.1)

subsampling=0.1
echo "Subsampling is [$subsampling]"
scaled=$(printf "%.0f" "$(echo "$subsampling * 100" | bc -l)")
echo "Scaled is [$scaled]"

for ensemble in "${ensemble_size[@]}"; do
    for subsampling in "${subsampling_ratio[@]}"; do
        for ratio in "${noise_ratios[@]}"; do
            for seed in "${seeds[@]}"; do
                scaled=$(awk "BEGIN {printf \"%.0f\", $subsampling * 100}")               
                echo "Running for noise ratio: $ratio, seed: $seed, ensemble: $ensemble, subsampling: $scaled"

                DATA_PATH="/path/to/datasets/MVTec-AD-noisy/mvtec_noise_ratio${ratio}/MVTech_nr${ratio}_seed_${seed}"
                SAVE_DIR="/path/to/experiments/dinomaly/mvtec/memory_scores/noise_ratio_${ratio}/seed_${seed}/ensemble_${ensemble}_p${scaled}"

                CUDA_VISIBLE_DEVICES=0 python $PYTHON_FILE \
                --data_path "$DATA_PATH" \
                --output_dir "$SAVE_DIR" \
                --ensemble_size "$ensemble" \
                --subsampling_ratio "$subsampling" \
                --dataset "MVTec-AD"


                # Optional memory cleanup
                echo "Cleaning up..."
                sleep 5  # give some time for CUDA memory to release
            done
        done
    done
done