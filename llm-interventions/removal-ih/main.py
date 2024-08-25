import torch
import os
import random
import json
from dataclasses import dataclass, asdict, field

import argparse

from src.tasks import *
from src.evaluator import Evaluator, EvalResult, make_prompt_to_solution, generate_prompt_to_solution, make_prompt_to_loss
from src.utils import Timer, time_str, extract_numerical


SEED = 1234
random.seed(SEED)
# set seed for model.generate determisnistic
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For CUDA operations, if applicable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_head",type = int, default = 0)
    parser.add_argument("--random_mask_head", action='store_true')
    parser.add_argument("-s", "--setting", type=str, choices=['origin', 'symbol', 'permute', 'upper', 'past'], help="The action to perform.", default="origin")
    parser.add_argument("--seq_len", type = int, default = 5)
    parser.add_argument("--model_name", type=str, help="The model to evaluate.", default="llama2_7b")
    parser.add_argument("--task_name", type=str, help="The task to evaluate.", default="ioi")
    parser.add_argument("--n_examples", type=int, help="Number of demonstration examples.", default=10)
    parser.add_argument("--multiple_choice", help="whether format it as multiple choice question", action='store_true')
    parser.add_argument("--random_mask_head_seed",type = int, default = 42)

    return parser.parse_args()




def t1(item, label=None):
    if label is None:
        return f"{item}:"
    return f"{item}:{label}, "


def t2(item, label=None):
    if label is None:
        return f"Item:{item} Label:"
    return f"Item:{item} Label:{label}, "


def t3(item, label=None):
    if label is None:
        return f"A:{item} B:"
    return f"A:{item} B:{label}, "

def t4(item, label=None):
    if label is None:
        return f"{item} is"
    return f"{item} is {label}, "

# MODELS = {
#     "llama2": _llama2,
#     "mistral": _mistral,
# }
MODELS = {
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
    "llama2_13b": "meta-llama/Llama-2-13b-hf",
    "llama2_70b": "meta-llama/Llama-2-70b-hf",
    "mistral_7b": "mistralai/Mistral-7B-v0.1",
    "mistral_8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "gemma_2b_it": "google/gemma-1.1-2b-it",
    "gemma_7b_it": "google/gemma-7b-it",
    "gemma_7b": "google/gemma-7b",
    "gemma2_9b": "google/gemma-2-9b",
    "falcon_7b": "tiiuae/falcon-7b",
    "olmo_7b": "allenai/OLMo-1.7-7B-hf",
    "pythia_7b": "EleutherAI/pythia-6.9b",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B",
    "falcon2_11b": "tiiuae/falcon-11B",
}
TASKS = {
    "num_translation": NumTranslationGenerator,
    "permute_label": PermuteLabelsGenerator,
    "city": CityGenerator,
    "ioi": IOIGenerator,
    "fuzzy_copy": FuzzyCopyGenerator,
    "gsm": GSMGenerator,
}
TEMPLATES = {"1": t1, "2": t2, "3": t3, "4": t4}


def main(
    model_name = "llama2_7b",
    task_name = "permute_label",
    template = 4,
    n_examples=10,
    n_experiments=100,
    top_k = 3, # change to 3 for fuzzy_copt len=10 and mask_head = 30
    seq_len = 20
):
    args = parse_args()
    ### mask heads
    mask_head = args.mask_head
    random_mask_head = args.random_mask_head
    seq_len = args.seq_len
    model_name = args.model_name
    task_name = args.task_name
    print(f"task name is {task_name}")
    n_examples = args.n_examples

    setting = args.setting
    permute = True if setting == "permute" else False
    symbol = True if setting == "symbol" else False

    timer = Timer()

    # func_model = MODELS[model_name]
    model_path = MODELS[model_name]
    func_task = TASKS[task_name]
    kwargs_func_tasks = {}

    if args.multiple_choice:
        kwargs_func_tasks.update({"multiple_choice": True})
    
    if task_name in ["ioi"]:
        kwargs_func_tasks.update({"symbol": symbol})
    elif task_name == "permute_label":
        kwargs_func_tasks.update({"permute": permute, "symbol": symbol}) 
    elif task_name == "fuzzy_copy":
        kwargs_func_tasks.update({"n_test": 8, "setting": setting}) 

    func_template = TEMPLATES[str(template)]

    # prompt_to_solution = func_model(mask_head = mask_head, random_mask_head = random_mask_head, beam_width = top_k)
    if args.multiple_choice:
        assert task_name in ["ioi","permute_label"], "only implemented ioi, permute_label task for now"
        prompt_to_solution = make_prompt_to_loss(model_path = model_path, mask_head = mask_head, random_mask_head = random_mask_head, random_mask_head_seed = args.random_mask_head_seed)
    else:
        if task_name == "gsm":
            # for CoT dataset, use generate function
            prompt_to_solution = generate_prompt_to_solution(model_path = model_path, mask_head = mask_head, random_mask_head = random_mask_head, seq_len = seq_len, random_mask_head_seed = args.random_mask_head_seed)
        else:
            prompt_to_solution = make_prompt_to_solution(model_path = model_path, mask_head = mask_head, random_mask_head = random_mask_head, beam_width = top_k, seq_len = seq_len, random_mask_head_seed = args.random_mask_head_seed)


    evaluator = Evaluator(prompt_to_solution)
    ds = func_task(template=func_template, n_examples=n_examples, **kwargs_func_tasks)
    n_experiments = n_experiments
    accs = 0
    topk_accs = 0
    results = []

    print(f"load model done, start evaluation, use time {time_str(timer.end())}")
    for _ in range(n_experiments):
        if args.multiple_choice:
            prompt, answer, choices = next(ds)
            result = evaluator.eval_loss(prompt, answer, choices)
            accs += result.accuracy
            results.append(asdict(result))
            if (_+1) % 20 == 0:
                print(f"===== {task_name} =========== {_+1}")
                time_elapsed = timer.end()
                print(f"use time {time_str(time_elapsed)} | {time_str(time_elapsed / (_+1)* n_experiments)}")
            continue

        prompt, answer = next(ds)
        if task_name == "gsm":
            result = evaluator.eval_gsm(prompt, answer)
            ground_truth = extract_numerical(result['answer'], from_ground_truth=True)
            prediction, model_output = extract_numerical(result['solution'], from_ground_truth=False)
            result['ground_truth'] = ground_truth
            result['prediction'] = prediction
            result['model_output'] = model_output
            result['accuracy'] = prediction == ground_truth
            accs += result['accuracy']
            results.append(result)
        else:
            result = evaluator.eval(prompt, answer)
            accs += result.accuracy
            topk_accs += result.topk_accuracy
            results.append(asdict(result))

        if (_+1) % 20 == 0:
            print(f"===== {task_name} =========== {_+1}")
            time_elapsed = timer.end()
            print(f"use time {time_str(time_elapsed)} | {time_str(time_elapsed / (_+1)* n_experiments)}")
    if task_name == "gsm":
        results = [{"acc": accs / n_experiments}] + results
    else:
        results = [{"acc": accs / n_experiments}, {f"top{top_k}acc": topk_accs / n_experiments}] + results
    # os.makedirs(f"{task_name}_{model_name}_eval_result", exist_ok=True)

    if args.multiple_choice:
        file_path = f"./result/multiple_choice/{task_name}/{model_name}/"
    else:
        file_path = f"./result/{task_name}/{model_name}/"
    
    if task_name in ["permute_label", "ioi", "fuzzy_copy"]:
        file_path += f"{task_name}_{model_name}_eval_result_seed{args.random_mask_head_seed}/{setting}"
    elif task_name in ["gsm"]:
        file_path += f"{task_name}_{model_name}_eval_result_seed{args.random_mask_head_seed}"
    else:
        file_path += f"{task_name}_{model_name}_eval_result"

    file_name = f"{n_examples}_template{template}"
        
    if mask_head > 0:
        file_name += f"mask{mask_head}"
    if random_mask_head:
        file_name += f"random_baseline"

    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    with open(
        f"{file_path}/{file_name}.json",
        "w",
    ) as f:
        json.dump(results, f, indent = 4)

    print(f"save to {file_path}/{file_name}")
    print(f"save out, total use time {time_str(timer.end())}")

if __name__ == "__main__":
    main()

