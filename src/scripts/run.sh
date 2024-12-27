#!/bin/bash

models=("llama2-7b") # Choose from `config.MODEL_CLASSES.keys()`
tasks=("copying")  # Choose from `config.TASKS_CONFIGS.keys()`
scripts=("evaluate_removal.py") # Choose from evaluate_[removal/projection/shuffle].py

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        for script in "${scripts[@]}"; do
            python $script --model_names $model --task_name $task
        done
    done
done