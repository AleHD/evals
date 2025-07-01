export TOKENIZER=alehc/swissai-tokenizer
export BOS=true

MODEL=Apertus-8B-converted
TOK_PER_IT=$(( 4096*1024 ))
IT=1678000
CKPT_PATH=/capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus8B-it1678000/
sbatch --job-name eval-$MODEL-$IT scripts/evaluate_hf.sbatch $CKPT_PATH $IT $TOK_PER_IT $MODEL
