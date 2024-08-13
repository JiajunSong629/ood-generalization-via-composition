#!/bin/bash

MODEL_NAMES=("gpt2" "gpt2-xl" "llama2-7b" "gemma-7b" "falcon-7b" "mistral-7b" "olmo-7b")

for model in "${MODEL_NAMES[@]}";
do
    echo "PREPING $model"
    python prep.py --model_name=$model
done