#!/bin/bash

export TOKENIZER=alehc/swissai-tokenizer
export BOS=true

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(    
    # Base pretrained models
    ["Apertus8B-tokens7.04T-it1678000"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens7.04T-it1678000"
    ["Apertus8B-tokens7.4T-it1728000"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens7.4T-it1728000"
    ["Apertus70B-tokens15T-it1155828"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens15T-it1155828"
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
