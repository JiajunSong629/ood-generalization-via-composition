#!/bin/bash

export PYTHONWARNINGS="ignore::FutureWarning"
MODEL_NAMES=("llama2-7b" "falcon-7b" "mistral-7b" "olmo-7b")

for model in "${MODEL_NAMES[@]}";
do
    echo "PROJECT..ING $model"
    # python project.py --model_name=$model --component="QK" --proj_out=True --method=diagonal
    # python project.py --model_name=$model --component="QK" --proj_out=False --method=diagonal
    # python project.py --model_name=$model --component="OV" --proj_out=True --method=diagonal
    # python project.py --model_name=$model --component="OV" --proj_out=False --method=diagonal

    # python project.py --model_name=$model --component="QK" --proj_out=True --method=subspace
    # python project.py --model_name=$model --component="QK" --proj_out=False --method=subspace
    # python project.py --model_name=$model --component="OV" --proj_out=True --method=subspace
    # python project.py --model_name=$model --component="OV" --proj_out=False --method=subspace

    python project.py --model_name=$model --component="QK" --proj_out=True --method=subset
    python project.py --model_name=$model --component="QK" --proj_out=False --method=subset
    python project.py --model_name=$model --component="OV" --proj_out=True --method=subset
    python project.py --model_name=$model --component="OV" --proj_out=False --method=subset

done