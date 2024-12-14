# Define arrays for models and evaluation scripts
models=("llama2-7b")
tasks=("icl")
scripts=("evaluate_removal.py")


for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        for script in "${scripts[@]}"; do
            python $script --model_names $model --task_name $task
        done
    done
done