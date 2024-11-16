#!/bin/bash

python evaluate_shuffle.py \
    --model_names llama2-7b,llama3-8b,gemma-7b,gemma2-9b,mistral-7b,pythia-7b,olmo-7b \
    --task_name gsm