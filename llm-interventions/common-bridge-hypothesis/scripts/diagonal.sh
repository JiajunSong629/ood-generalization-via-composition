#!/bin/bash
export PYTHONWARNINGS="ignore::FutureWarning"
# MODEL_NAMES=("gpt2" "gpt2-xl" "llama2-7b" "gemma-7b" "falcon-7b" "mistral-7b" "olmo-7b" "gemma2-9b" "pythia-7b")
MODEL_NAMES=("llama3-8b")

for model in "${MODEL_NAMES[@]}"; do
	echo "DIAGONAL..ING $model"
	python diagonal.py --model_name=$model --K=60 --random=True
done
