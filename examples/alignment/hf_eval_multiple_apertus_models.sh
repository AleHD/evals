#!/bin/bash

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(
    # RLVR models
    # ["Apertus3-8B_iter_1678000-tulu3-sft-RLVR-105"]="/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_105/hf_actor"
    # ["Apertus3-8B_iter_1678000-tulu3-sft-RLVR-450"]="/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_450/hf_actor"
    # ["Apertus3-8B_iter_1678000-tulu3-sft-RLVR-560"]="/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_560/hf_actor"
    ["Apertus3-8B_iter_1678000-tulu3-sft-RLVR-MR-800"]="/capstor/store/cscs/swissai/infra01/reasoning/models/mr_800/hf_actor"
    
    # SFT models
    ["Apertus-8B-SFT"]="/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-8B_iter_1678000-tulu3-sft/checkpoint-13446/"
    ["Apertus-8B-7.04T-it1678000--Tulu3-SFT-tulu"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/chat-template/Apertus8B-tokens7.04T-it1678000-tulu-swissai-tulu-3-sft-0225/checkpoints/6d5f11d2873ecb4d/checkpoint-13446"
    ["Apertus-8B-7.04T-it1678000--Tulu3-SFT-tulu_special_token"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/chat-template/Apertus8B-tokens7.04T-it1678000-tulu_special_token-swissai-tulu-3-sft-0225/checkpoints/9b811fb20bdd09a4/checkpoint-13446"
    ["Apertus8B-tokens7.4T-it1728000-ademamix-swissai-tulu-3-sft-0225"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/token-count/Apertus8B-tokens7.4T-it1728000-ademamix-swissai-tulu-3-sft-0225/checkpoints/aba7b1a3121290e5/checkpoint-13000"
    ["Apertus8B-tokens7.4T-it1728000-ademamix-swissai-tulu-3-sft-0225-max_grad_norm_0.1"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/Apertus8B-tokens7.4T-it1728000-ademamix-swissai-tulu-3-sft-0225-max_grad_norm_0.1/checkpoints/206c53edb3c43a3a/checkpoint-13000"
    ["Apertus-70B-15T-it1155828--Tulu3-SFT"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/token-count/Apertus70B-tokens15T-it1155828-ademamix-swissai-tulu-3-sft-0225/checkpoints/c9b2910640c220b1/checkpoint-13446"
)

export WANDB_ENTITY=apertus
export WANDB_PROJECT=swissai-evals-v0.0.3
# Launch evaluation jobs for each model
echo "Launching evaluation jobs for ${#MODEL_CHECKPOINTS[@]} models..."
job_count=0
for MODEL in "${!MODEL_CHECKPOINTS[@]}"; do
    CKPT_PATH="${MODEL_CHECKPOINTS[$MODEL]}"
    job_count=$((job_count + 1))
    
    echo "Launching job $job_count/${#MODEL_CHECKPOINTS[@]}: $MODEL"
    echo "  Checkpoint path: $CKPT_PATH"
    
    sbatch --job-name eval-$MODEL scripts/evaluate_hf.sbatch "$CKPT_PATH" "$MODEL"
    
    # Add a small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done

echo "All evaluation jobs submitted successfully!"
