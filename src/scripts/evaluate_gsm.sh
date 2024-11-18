#!/bin/bash

# Define arrays for models and evaluation scripts
models=("llama3-8b" "gemma-7b" "gemma2-9b" "mistral-7b" "pythia-7b" "olmo-7b" "falcon-7b")
tasks=("gsm")
scripts=("evaluate_projection.py" "evaluate_shuffle.py")

# Iterate through each model
for model in "${models[@]}"; do
    # For each model, run all tasks and scripts
    for task in "${tasks[@]}"; do
        for script in "${scripts[@]}"; do
            python "$script" \
                --model_names "$model" \
                --task_name "$task"
        done
    done
done