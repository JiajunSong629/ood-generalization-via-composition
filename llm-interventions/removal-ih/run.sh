#!/bin/bash

# export CUDA_VISIBLE_DEVICES=5
# Define the array of values
# mask_head_values=(5 10 15 20 25 30 40 50)
mask_head_values=(10 20 30 40 50)
# mask_head_values=( $(seq 2 2 50) )
echo "mask_head_values = ${mask_head_values[@]}"

random_mask_head_seeds=(37 42 134 1567 8787)
# random_mask_head_seeds=(42)



# model_names=("llama2_7b")
# model_names=("gemma_7b_it")
# model_names=("falcon_7b")
# model_names=("olmo_7b")
# model_names=("pythia_7b")
# model_names=("falcon_7b" "gemma_7b_it" "llama2_7b" "mistral_7b" "olmo_7b")
# model_names=("llama3_8b" "falcon2_11b" "pythia_7b")
# model_names=("llama3_8b")
# model_names=("gemma2_9b")
model_names=("gemma_7b")



######################### fuzzy copy
# task_name="fuzzy_copy"
# settings=("upper")
# n_examples=10
# seq_len=15
#########################

######################### ioi
# task_name="ioi"
# settings=("origin" "symbol")
# n_examples=10
# seq_len=5
# add --multiple_choice!!
#########################

######################### permute_label
task_name="permute_label"
settings=("origin" "permute" "symbol")
n_examples=20
seq_len=2
## add  --multiple_choice !!
#########################


# Iterate over each model name
for model_name in "${model_names[@]}"
do
  # Iterate over each setting
  for setting in "${settings[@]}"
  do
    python main.py  --model_name $model_name -s $setting --seq_len $seq_len --task_name $task_name --n_examples $n_examples  --multiple_choice
    # Iterate over each value in the mask head values array
    for value in "${mask_head_values[@]}"
    do
      echo "Executing: python main.py --mask_head $value with --task_name $task_name setting $setting"
      python main.py  --model_name $model_name --mask_head $value -s $setting --seq_len $seq_len --task_name $task_name  --n_examples $n_examples  --multiple_choice
      for random_mask_head_seed in "${random_mask_head_seeds[@]}"
      do
        echo "Executing with random mask: python main.py --mask_head $value random masked with --task_name $task_name setting $setting"
        python main.py  --model_name $model_name --mask_head $value --random_mask_head -s $setting --seq_len $seq_len --task_name $task_name  --n_examples $n_examples --random_mask_head_seed $random_mask_head_seed  --multiple_choice
      done
    done
  done
done

