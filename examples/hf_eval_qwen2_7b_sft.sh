MODEL=Qwen2.5-7B-Instruct
CKPT_PATH=Qwen/Qwen2.5-7B-Instruct
sbatch --job-name eval-$MODEL scripts/evaluate_hf.sbatch $CKPT_PATH $MODEL