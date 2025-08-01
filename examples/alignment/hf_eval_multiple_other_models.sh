#!/bin/bash

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(
    # EuroLLM models
    ["EuroLLM-9B"]="utter-project/EuroLLM-9B"
    ["EuroLLM-22B-Preview"]="utter-project/EuroLLM-22B-Preview"
    
    # Gemma models
    ["gemma-3-4b-it"]="google/gemma-3-4b-it"
    ["gemma-3-12b-it"]="google/gemma-3-12b-it"
    ["gemma-3-27b-it"]="google/gemma-3-27b-it"
    
    # K2 models
    ["K2-Chat"]="K2-Chat"
    
    # Llama models
    ["Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"
    ["Llama-3.3-70B-Instruct"]="meta-llama/Llama-3.3-70B-Instruct"
    
    # OLMo models (base)
    ["OLMo-2-1124-7B"]="allenai/OLMo-2-1124-7B"
    ["OLMo-2-0325-32B"]="allenai/OLMo-2-0325-32B"
    
    # OLMo models (fine-tuned variants)
    ["OLMo-2-1124-7B-SFT"]="allenai/OLMo-2-1124-7B-SFT"
    ["OLMo-2-1124-7B-DPO"]="allenai/OLMo-2-1124-7B-DPO"
    ["OLMo-2-1124-7B-Instruct"]="allenai/OLMo-2-1124-7B-Instruct"
    ["OLMo-2-0325-32B-SFT"]="allenai/OLMo-2-0325-32B-SFT"
    ["OLMo-2-0325-32B-DPO"]="allenai/OLMo-2-0325-32B-DPO"
    ["OLMo-2-0325-32B-Instruct"]="allenai/OLMo-2-0325-32B-Instruct"
    
    # Qwen 2.5 models
    ["Qwen2.5-7B"]="Qwen/Qwen2.5-7B"
    ["Qwen2.5-7B-Instruct"]="Qwen/Qwen2.5-7B-Instruct"
    ["Qwen2.5-14B-Instruct"]="Qwen/Qwen2.5-14B-Instruct"
    ["Qwen2.5-32B-Instruct"]="Qwen/Qwen2.5-32B-Instruct"
    ["Qwen2.5-72B-Instruct"]="Qwen/Qwen2.5-72B-Instruct"
    
    # Qwen 3 models
    ["Qwen3-1.7B"]="Qwen/Qwen3-1.7B"
    ["Qwen3-4B"]="Qwen/Qwen3-4B"
    ["Qwen3-8B"]="Qwen/Qwen3-8B"
    ["Qwen3-14B"]="Qwen/Qwen3-14B"
    ["Qwen3-32B"]="Qwen/Qwen3-32B"
    
    # SmolLM models
    ["SmolLM3-3B"]="HuggingFaceTB/SmolLM3-3B"
)

export WANDB_ENTITY=apertus
export WANDB_PROJECT=swissai-evals-v0.0.3
# Launch evaluation jobs for each model
echo "Launching evaluation jobs for ${#MODEL_CHECKPOINTS[@]} non-Apertus models..."
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
