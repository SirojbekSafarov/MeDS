#!/bin/bash

# Set Python file name
PYTHON_FILE="noisy_dinomaly_uni_step_1_memory_score_generation.py"  # 🔁 Replace with the actual filename

# Array of noise ratios and seeds
noise_ratios=(0)
seeds=(0)
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

                DATA_PATH="/research/workspaces/sirojbek/visa_noisy/visa_noise_ratio${ratio}/VisA_20220922_pytorch_nr${ratio}_seed_${seed}"
                # DATA_PATH="/research/workspaces/sirojbek/mvtec_noisy/mvtec_noise_ratio${ratio}/MVTech_nr${ratio}_seed_${seed}"
                SAVE_DIR="/research/experiments/siroj/academic/noisy_ad/dinomaly/visa/dino_v2_vit_base_backbone/memory_scores_folder/noise_ratio_${ratio}/seed_${seed}/ensemble_${ensemble}_p${scaled}"

                CUDA_VISIBLE_DEVICES=0 python $PYTHON_FILE \
                --data_path "$DATA_PATH" \
                --output_dir "$SAVE_DIR" \
                --ensemble_size "$ensemble" \
                --subsampling_ratio "$subsampling" \
                --dataset "VisA"  #MVTec-AD


                # Optional memory cleanup
                echo "Cleaning up..."
                sleep 5  # give some time for CUDA memory to release
            done
        done
    done
done