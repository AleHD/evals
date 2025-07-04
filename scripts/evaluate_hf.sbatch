#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --job-name=evaluation
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --partition=normal

# Aux functions.
usage() {
	echo "Usage: sbatch evaluate.sbatch <model> <iteration> <tokens-per-iter> <name>"
	echo "Runs evaluation of the specified model. The 'model' should either be a megatron checkpoint or a huggingface path/name. The script determines which one is given by looking for 'model/latest_checkpointed_iteration.txt', if it is not found a huggingface model is assumed."
	echo "The 'iteration' and 'tokens-per-iter' is used to calculate the consumed tokens of the model."
	echo "The 'name' shall be used as an unique identifier."
	echo "In addition of these three necessary positional arguments, you can specify the following bash environment variables. Unless explicitly stated, all of these are optional:"
	echo " TOKENIZER: A huggingface tokenizer path/name. Needed if 'model' is a megatron checkpoint, otherwise the default will be set as the same value as 'model'."
	echo " BS: Batch size."
	echo " REVISION: Only used in huggingface models. If set, this revision of the model will be evaluated."
	echo " SIZE: The (approximate) size of the model in billions of parameters. Used to set model parallelism (needs to be set in larger models)."
	echo " LIMIT: The --limit argument to pass to lm-evaluation-harness."
	echo " BOD: Set this to 'true' if you wish to prepend the BOD token when evaluating models."
	echo " LOGS_ROOT: Where are your evaluation wandb&harness logs going to."
	echo " TASKS: Tasks to run with lm eval harness."
	echo "For more information see the README: https://github.com/swiss-ai/evals?tab=readme-ov-file."
}
die() {
	echo "$*" >& 2
	exit 1
}

# Wakeup logs.
set -e
echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

# Grab variables and arguments.
if (( $# != 4 )); then
	usage
	die "Invalid usage: Invalid argument count"
fi
MODEL=$1
IT=$2
TOKENS_PER_ITER=$3
NAME=$4
BS=${BS:-"auto:20"}
SIZE=${SIZE:-1}
BOD=${BOD:-false}
LOGS_ROOT=${LOGS_ROOT:-$SCRATCH/eval-logs}
TASKS=${TASKS:-swissai_eval}
export HF_HOME=${HF_HOME:-/capstor/store/cscs/swissai/infra01/hf_home/}
GPUS_PER_NODE=4

# Print configuration.
echo "Configuration set:"
printf "MODEL=$MODEL\nIT=$IT\nTOKENS_PER_ITER=$TOKENS_PER_ITER\nNAME=$NAME\nTOKENIZER=$TOKENIZER\nBS=$BS\nREVISION=$REVISION\nSIZE=$SIZE\nLIMIT=$LIMIT\nBOD=$BOD\nTASKS=$TASKS\\nHF_HOME=$HF_HOME\n\n"

# Generic envs.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Huggingface checkpoint detected!"
CONSUMED_TOKENS=$(( IT*TOKENS_PER_ITER ))
if [ ! -z ${REVISION+x} ]; then
    echo "Using revision=$REVISION"
    MAYBE_REVISION=",revision=$REVISION"
    MAYBE_DOWNLOAD_REVISION="--revision=$REVISION"
fi
if [ -z ${TOKENIZER+x} ]; then
    echo "TOKENIZER not set, using TOKENIZER=$MODEL"
    TOKENIZER=$MODEL
fi

echo "Consumed tokens for this iteration: $CONSUMED_TOKENS"

# Determine model parallel based on size.
if (( SIZE <= 35 )); then
	MP=1
	CONVERT_MP=1
elif (( SIZE <= 70 )); then
	MP=2
	CONVERT_MP=4
else
	MP=4
	CONVERT_MP=4
fi
echo "Using model size ~${SIZE}B, model parallel size of $MP, conversion parallel size of $CONVERT_MP"

# Prepare logs root.
RUN_ROOT=$LOGS_ROOT/$NAME/iter_$IT
HARNESS_DIR=$RUN_ROOT/harness
export WANDB_DIR=$RUN_ROOT
mkdir -p $HARNESS_DIR
echo $CONSUMED_TOKENS > $RUN_ROOT/consumed_tokens.txt

export HF_HUB_OFFLINE=1

# Finally, run the actual evaluations.
HF_CHECKPOINT_PATH=$MODEL
DP=$(( GPUS_PER_NODE / MP ))
COMMON_MODEL_ARGS="pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER,max_length=4096$MAYBE_REVISION,attn_implementation=flash_attention_2,dtype=bfloat16"
if [[ $BOD = true ]]; then
	echo "Adding BOS token."
	COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,add_bos_token=True"
fi
COMMON_EVAL_ARGS=(
	--trust_remote_code
	--batch_size $BS
	--tasks $TASKS
	--output $HARNESS_DIR/eval_$SLURM_JOBID
	--max_batch_size 128
	--cache_requests true
	--model hf
	--log_samples
	--write_out
	$LIMIT_ARGS
)
if [[ $DP -eq 1 ]]; then  # Only use model parallel.
	export WORLD_SIZE=1
	CMD="lm_eval --model_args=$COMMON_MODEL_ARGS,parallelize=True ${COMMON_EVAL_ARGS[@]}"
elif (( MP > 1 )); then  # Use data parallel and model parallel.
	CMD="accelerate launch --multi_gpu --num_processes $DP -m lm_eval --model_args=$COMMON_MODEL_ARGS,parallelize=True ${COMMON_EVAL_ARGS[@]}"
else  # Only use data parallel.
	CMD="accelerate launch -m lm_eval --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"
fi
echo "Final command: $CMD"
srun -ul --environment=./env.toml bash -c " \
    $CMD
"

# Goodbye.
echo "Evaluation finished"
echo "Now run 'python scripts/update_wandb.py $LOGS_ROOT --name $NAME --it $IT' to update wandb space."
echo "END TIME: $(date)"
