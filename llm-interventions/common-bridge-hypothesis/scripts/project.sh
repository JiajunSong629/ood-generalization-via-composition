#!/bin/bash

export PYTHONWARNINGS="ignore::FutureWarning"
MODEL_NAMES=("gpt2" "gpt2-xl" "llama2-7b" "llama3-8b" "gemma-7b" "gemma2-9b" "falcon-7b" "mistral-7b" "olmo-7b" "pythia-7b")

# # make directory for out-proj-many
# mkdir -p "out-proj-many"
# for model in "${MODEL_NAMES[@]}"; do
# 	echo "PROJECT..ING ${model}"
# 	for component in "QK" "OV"; do
# 		for proj_out in "True" "False"; do
# 			python project.py \
# 				--out_dir="out-proj-many" \
# 				--K1_prop=0.25 --K0=10 \
# 				--model_name="${model}" --component="${component}" --proj_out="${proj_out}" \
# 				--method=diagonal
# 		done
# 	done
# done >"out-proj-many/project.log" 2>&1

# # make directory for out-proj-few
# mkdir -p "out-proj-few"
# for model in "${MODEL_NAMES[@]}"; do
# 	echo "PROJECT..ING ${model}"
# 	for component in "QK" "OV"; do
# 		for proj_out in "True" "False"; do
# 			python project.py \
# 				--out_dir="out-proj-few" \
# 				--K1=50 --K0=10 \
# 				--model_name="${model}" --component="${component}" --proj_out="${proj_out}" \
# 				--method=diagonal
# 		done
# 	done
# done >"out-proj-few/project.log" 2>&1

# # make directory for out-proj-include-all-many
# mkdir -p "out-proj-include-all-many"
# for model in "${MODEL_NAMES[@]}"; do
# 	echo "PROJECT..ING ${model}"
# 	for component in "QK" "OV"; do
# 		for proj_out in "True" "False"; do
# 			python project.py \
# 				--out_dir="out-proj-include-all-many" \
# 				--K1_prop=0.25 --K0=50 \
# 				--model_name="${model}" --component="${component}" --proj_out="${proj_out}" \
# 				--method=diagonal
# 		done
# 	done
# done >"out-proj-include-all-many/project.log" 2>&1

# make directory for out-proj-more
mkdir -p "out-proj-more"
for model in "${MODEL_NAMES[@]}"; do
	echo "PROJECT..ING ${model}"
	for component in "QK" "OV"; do
		for proj_out in "True" "False"; do
			python project.py \
				--out_dir="out-proj-more" \
				--K1_prop=0.3 --K0=10 \
				--model_name="${model}" --component="${component}" --proj_out="${proj_out}" \
				--method=diagonal
		done
	done
done >"out-proj-more/project.log" 2>&1
