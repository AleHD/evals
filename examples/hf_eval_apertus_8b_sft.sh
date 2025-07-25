export TOKENIZER=alehc/swissai-tokenizer
export BOS=true

MODEL=Apertus-8B-SFT
CKPT_PATH=/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-8B_iter_1678000-tulu3-sft/checkpoint-13446/
sbatch --job-name eval-$MODEL scripts/evaluate_hf.sbatch $CKPT_PATH $MODEL
