#!/bin/bash

# Set Python file name
PYTHON_FILE="noisy_dinomaly_uni_step_3_data_selection_with_distilled_model.py"  # 🔁 Replace with the actual filename

# Array of noise ratios and seeds
noise_ratios=(0 10 20 40)
# seeds=(0 10 20 30)
seeds=(0)
ensemble_size=(100)
subsampling_ratio=(0.1)
loss_function='l1' #l2
mad_factors=(1)

for ensemble in "${ensemble_size[@]}"; do
    for subsampling in "${subsampling_ratio[@]}"; do
        for ratio in "${noise_ratios[@]}"; do
            for seed in "${seeds[@]}"; do
                for mad_factor in "${mad_factors[@]}"; do
                    scaled=$(awk "BEGIN {printf \"%.0f\", $subsampling * 100}")               
                    echo "Running for noise ratio: $ratio, seed: $seed, ensemble: $ensemble, subsampling: $scaled"

                    # DATA_PATH="/research/workspaces/sirojbek/visa_noisy/visa_noise_ratio${ratio}/VisA_20220922_pytorch_nr${ratio}_seed_${seed}"
                    # SAVE_DIR="/research/experiments/siroj/academic/noisy_ad/inpformer/visa/dino_v2_vit_base_backbone/multi_class/stage_3_pseudo_label_selection/distilled_model_as_dataselector/with_distilled_as_seed_model/noise_ratio_${ratio}/seed_${seed}/z${mad_factor}/ensemble_${ensemble}_p${scaled}/${loss_function}"
                    # DISTILL_OUT_DIR="/research/experiments/siroj/academic/noisy_ad/inpformer/visa/dino_v2_vit_base_backbone/multi_class/stage_2_distillation/noise_ratio_${ratio}/seed_${seed}/ensemble_${ensemble}_p${scaled}/${loss_function}"

                    DATA_PATH="/research/workspaces/sirojbek/mvtec_noisy/mvtec_noise_ratio${ratio}/MVTech_nr${ratio}_seed_${seed}"
                    SAVE_DIR="/research/experiments/siroj/academic/noisy_ad/inpformer/mvtec/dino_v2_vit_base_backbone/multi_class/stage_3_pseudo_label_selection/distilled_model_as_dataselector/with_distilled_as_seed_model/noise_ratio_${ratio}/seed_${seed}/z${mad_factor}/ensemble_${ensemble}_p${scaled}/${loss_function}"
                    DISTILL_OUT_DIR="/research/experiments/siroj/academic/noisy_ad/inpformer/mvtec/dino_v2_vit_base_backbone/multi_class/stage_2_distillation/noise_ratio_${ratio}/seed_${seed}/ensemble_${ensemble}_p${scaled}/${loss_function}"

                    CUDA_VISIBLE_DEVICES=0 python $PYTHON_FILE \
                    --data_path "$DATA_PATH" \
                    --output_dir "$SAVE_DIR" \
                    --distill_output_dir "$DISTILL_OUT_DIR" \
                    --mad_factor "$mad_factor" \
                    --dataset "MVTec-AD" \
                    # --dataset "VisA" \
                    # --distill_initialized "False"


                    # Optional memory cleanup
                    echo "Cleaning up..."
                    sleep 5  # give some time for CUDA memory to release
                done
            done
        done
    done
done