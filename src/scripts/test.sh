python evaluate_shuffle.py \
    --model_names llama2-7b \
    --task_name copying

python evaluate_shuffle.py \
    --model_names llama2-7b \
    --task_name icl

python evaluate_projection.py \
    --model_names gpt2 \
    --task_name copying

python evaluate_projection.py \
    --model_names gpt2 \
    --task_name icl
