#!/bin/bash

export PYTHONWARNINGS="ignore::FutureWarning"
MODEL_NAMES=("gpt2" "gpt2-xl" "llama2-7b" "llama3-8b" "gemma-7b" "gemma2-9b" "falcon-7b" "mistral-7b" "olmo-7b" "pythia-7b")

for model in "${MODEL_NAMES[@]}"; do
	echo "SHUFFLE..ING ${model}"
	python shuffle.py --model_name="${model}" --n_exp=10 --K=10 --method=diagonal
done >"shuffle.log" 2>&1
