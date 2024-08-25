#!/bin/bash

# MODEL_NAMES=("gpt2" "gpt2-xl" "llama2-7b" "gemma-7b" "falcon-7b" "mistral-7b" "olmo-7b")
# MODEL_NAMES=("llama3-8b")
# MODEL_NAMES=("falcon2-11b")
# MODEL_NAMES=("llama2-70b")
MODEL_NAMES=("gemma-2-9b")

for model in "${MODEL_NAMES[@]}";
do
    echo "PREPING $model"
    python prep.py --model_name=$model
done