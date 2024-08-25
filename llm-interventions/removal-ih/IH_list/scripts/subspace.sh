#!/bin/bash

export PYTHONWARNINGS="ignore::FutureWarning"
MODEL_NAMES=("llama2-7b" "gemma-7b" "falcon-7b" "mistral-7b" "olmo-7b")

for model in "${MODEL_NAMES[@]}";
do
    echo "SUBSPACE...ING $model"
    python subspace.py --model_name=$model --K=40 --rank=10 --method="largest"
done