#!/bin/bash

# Set Python file name
PYTHON_FILE="step_3_data_selection_with_distilled_model.py"

# Array of noise ratios and seeds
noise_ratios=(10)
seeds=(0)
ensemble_size=(100)
subsampling_ratio=(0.1)
loss_function='l2' #l1
mad_factors=(0 1 2 3)

for ensemble in "${ensemble_size[@]}"; do
    for subsampling in "${subsampling_ratio[@]}"; do
        for ratio in "${noise_ratios[@]}"; do
            for seed in "${seeds[@]}"; do
                for mad_factor in "${mad_factors[@]}"; do
                    scaled=$(awk "BEGIN {printf \"%.0f\", $subsampling * 100}")               
                    echo "Running for noise ratio: $ratio, seed: $seed, ensemble: $ensemble, subsampling: $scaled"

                    DATA_PATH="/path/to/datasets/MVTec-AD-noisy/mvtec_noise_ratio${ratio}/MVTech_nr${ratio}_seed_${seed}"
                    SAVE_DIR="/path/to/experiments/dinomaly/mvtec/stage_3_data_selection/distilled/noise_ratio_${ratio}/seed_${seed}/z${mad_factor}/ensemble_${ensemble}_p${scaled}/${loss_function}"
                    DISTILL_OUT_DIR="/path/to/experiments/dinomaly/mvtec/stage_2_distillation/noise_ratio_${ratio}/seed_${seed}/ensemble_${ensemble}_p${scaled}/${loss_function}"

                    CUDA_VISIBLE_DEVICES=0 python $PYTHON_FILE \
                    --data_path "$DATA_PATH" \
                    --output_dir "$SAVE_DIR" \
                    --distill_output_dir "$DISTILL_OUT_DIR" \
                    --mad_factor "$mad_factor" \
                    --dataset "MVTec-AD"


                    # Optional memory cleanup
                    echo "Cleaning up..."
                    sleep 5  # give some time for CUDA memory to release
                done
            done
        done
    done
done