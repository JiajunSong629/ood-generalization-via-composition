#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="logs/run_${timestamp}.log"

# Define arrays for models and evaluation scripts
models=("gpt2" "gpt2-xl" "llama2-7b" "llama3-8b" "gemma-7b" "gemma2-9b" "falcon-7b" "mistral-7b" "olmo-7b" "pythia-7b")
tasks=("ioi")
# scripts=("evaluate_projection.py" "evaluate_shuffle.py")
scripts=("evaluate_projection.py" "evaluate_shuffle.py")

# Export PYTHONWARNINGS to ignore warnings
export PYTHONWARNINGS="ignore"

# Log the start of execution
{
    echo "=== Execution started at $(date) ==="
    echo "Running with models: ${models[*]}"
    echo "Tasks: ${tasks[*]}"
    echo "Scripts: ${scripts[*]}"
    echo "----------------------------------------"
} | tee -a "$logfile"

# Function to filter out warning messages
filter_warnings() {
    grep -v "WARNING:" | grep -v "UserWarning:" | grep -v "FutureWarning:" | grep -v "DeprecationWarning:"
}

# Iterate through each model
for model in "${models[@]}"; do
    # For each model, run all tasks and scripts
    for task in "${tasks[@]}"; do
        for script in "${scripts[@]}"; do
            # Log the command being executed
            {
                echo "=== Running command at $(date) ==="
                echo "Command: python $script --model_names $model --task_name $task"
                echo "-----------------------------------------------------------------"
            } | tee -a "$logfile"
            
            # Execute command with unbuffered output
            PYTHONWARNINGS="ignore" python -u -W ignore "$script" \
                --model_names "$model" \
                --task_name "$task" 2>&1 \
                | filter_warnings \
                | tee -a "$logfile"
            
            echo "----------------------------------------" | tee -a "$logfile"
        done
    done
done

# Log completion
{
    echo "=== Execution completed at $(date) ==="
    echo "Log saved to: $logfile"
} | tee -a "$logfile"

echo "Execution completed. Log saved to: $logfile"