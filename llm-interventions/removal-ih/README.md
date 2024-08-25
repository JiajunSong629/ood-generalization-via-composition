# Inference LLM on tasks with IH removal


## Requirements

It is tested under Ubuntu Linux 20.04 and Python 3.11 environment and requires some packages to be installed.

 - Pytorch >= 1.12.1 (guide is [here](https://pytorch.org/get-started/locally/))
 - Transformers >= 4.37

#### Install
```
conda create -n llm-interv  python=3.10 -y
conda activate llm-interv
pip install --upgrade pip 
pip install -r requirements.txt
```

## Usage
#### Inference
- To evaluate the model on tasks, run:

```bash
bash run.sh
```
Modify `tasks`, `settings`, `models` accordingly.
Tasks include `fuzzy_copy`, `ioi`, `permute_label`, `multiple_choice (ioi, permute_label)`.

This will save raw results into `./result/<task_name>/<task_name>_<model_name>_eval_result/<setting>`.

- To evaluate the `llama2-70b` model on `CoT` task, run:
```bash
bash run_70b.sh
```

#### Gather accs

```bash
cd ./result
bash acc.sh
```
This will save aggregated results into `./result/<task_name>/<task_name>_<model_name>_eval_result/accs.json`.
We can adjust the tasks and model accordingly.

#### visualize accs
in `./notebook`.


#### Get IH lists
1. see details in `IH_list`
```
cd IH_list
```
following instructions to get IH list for different models.


2. Move IH lists to the `src`
For example:
```
cp ./IH_list/out/gemma-2-9b/IH.json src/gemma/
cd src/gemma
mv IH.json IH_gemma2_9b.json
```

print out head id and paste into `ada_gemma.py`
```
cd src/gemma
python src/gemma/get_id.py
```
modify `file_path`.

