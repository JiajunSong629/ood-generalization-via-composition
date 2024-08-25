#!/bin/bash

# Set model_name variable
model_name="llama2_70b"

# python src/llama/model_weights.py
# python main.py --model_name $model_name
# Define the array of values
# mask_head_values=(50)
mask_head_values=(50 100 150 200 250)
# mask_head_values=(10 20 30 40 50)

random_mask_head_seeds=(37 42 134 1567 8787)
# Iterate over the values and execute the Python script with each one
# python main.py --model_name $model_name --task_name "gsm"
for value in "${mask_head_values[@]}"
do
  # echo "Executing: python main.py --mask_head $value"
  # python main.py --mask_head $value --model_name $model_name --task_name "gsm"
  for random_mask_head_seed in "${random_mask_head_seeds[@]}"
  do
    echo "Executing: python main.py --mask_head $value random masked"
    python main.py --mask_head $value  --random_mask_head --model_name $model_name --task_name "gsm" --random_mask_head_seed $random_mask_head_seed 
  done
done

exit 0
python main.py -s symbol --model_name $model_name
# Iterate over the values and execute the Python script with each one
for value in "${mask_head_values[@]}"
do
  echo "Executing: python main.py --mask_head $value"
  python main.py --mask_head $value -s symbol --model_name $model_name
  echo "Executing: python main.py --mask_head $value random masked"
  python main.py --mask_head $value  --random_mask_head -s symbol --model_name $model_name
done
exit 0


