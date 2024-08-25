import json
import argparse
from pprint import pprint
import copy
from pdb import set_trace as pds

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, indent = 4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--setting", type=str, choices=['symbol', 'permute', 'upper', 'past'], help="The action to perform.", default="origin")
    parser.add_argument("--tasks", type=str, choices=['permute_label', 'ioi', 'fuzzy_copy',"gsm"], default="permute_label")
    parser.add_argument("-m", "--model_name", type=str, help="The model to evaluate.", default="llama2_7b")
    parser.add_argument("--n_examples", type=int, help="Number of demonstration examples.", default=10)
    parser.add_argument("--multiple_choice", help="whether format it as multiple choice question", action='store_true')

    return parser.parse_args()


def extract(
        model_name = "llama2_7b", 
        task_name = "permute_label",
        setting = "origin",
        n_examples = 10,
        template = 4,
        mask_head = 0,
        random_mask_head = False,
        multiple_choice = False, 
        random_mask_head_seed = 42
    ):

    if multiple_choice:
        file_path = f"./multiple_choice/{task_name}/{model_name}/"
    else:
        file_path = f"./{task_name}/{model_name}/"
    
    if task_name in ["permute_label", "ioi", "fuzzy_copy"]:
        file_path += f"{task_name}_{model_name}_eval_result_seed{random_mask_head_seed}/{setting}" 
    else:
        file_path += f"{task_name}_{model_name}_eval_result"
    
    if task_name == "gsm":
        if random_mask_head:
            file_path += f"_seed{random_mask_head_seed}"


    file_name = f"{n_examples}_template{template}"
        
    if mask_head > 0:
        file_name += f"mask{mask_head}"
    if random_mask_head:
        file_name += f"random_baseline"

    load_file_path = f"{file_path}/{file_name}.json"
    print(f"load from: {load_file_path}")
    data = load_json(file_path=load_file_path)
    
    '''
    print(load_file_path)
    print(len(data))
    print(data[0])
    print(data[1])
    
    return
    '''

    res = {
        "model": model_name, "task": task_name, "n_exmaple": n_examples, "template": template, "setting": setting,
        "mask_head": mask_head,
        "random_mask_head": random_mask_head,
        "random_mask_head_seed": random_mask_head_seed,
        "acc": data[0]['acc'],
        "topkacc": list(data[1].values())[0]
        }
    return res

def main():
    args = parse_args()

    model_name = args.model_name
    task_name = args.tasks
    if task_name == "permute_label":
        settings = ["origin", "permute", "symbol"]
    elif task_name == "ioi":
        settings = ["origin", "symbol"]
    elif task_name == "fuzzy_copy":
        # settings = ["past", "upper"]
        settings = ["upper"]
    elif task_name == "gsm":
        settings = ["None"]

    if args.multiple_choice:
        file_path = f"./multiple_choice/{task_name}/{model_name}/"
    else:
        file_path = f"./{task_name}/{model_name}/"
    
    accs = []
    for setting in settings:
        res = extract(
            model_name = model_name,
            task_name = task_name, 
            setting = setting, 
            n_examples = args.n_examples, 
            multiple_choice = args.multiple_choice,
            )
        accs.append(res)
        ## add duplicate for random_mask_head = True, for mask_head = 0
        # Make a deep copy of res, then modify and append the copy
        res_copy = copy.deepcopy(res)
        res_copy["random_mask_head"] = True 
        accs.append(res_copy)
        # pprint(accs)
        # assert False
        
            # for mask_head in [x for x in range(2, 51, 2)]:
        for mask_head in [x for x in [10,20,30,40,50]]:
        # for mask_head in [x for x in [50,100,150,200,250]]:
            res = extract(
                model_name = model_name,
                task_name = task_name, 
                mask_head = mask_head, 
                random_mask_head = False, 
                setting = setting,
                n_examples = args.n_examples,
                multiple_choice = args.multiple_choice,
            )
            accs.append(res)
            for random_mask_head_seed in [37, 42, 134, 1567, 8787]:
                res = extract(
                    model_name = model_name,
                    task_name = task_name, 
                    mask_head = mask_head, 
                    random_mask_head = True, 
                    setting = setting,
                    n_examples = args.n_examples,
                    multiple_choice = args.multiple_choice,
                    random_mask_head_seed = random_mask_head_seed
                )
                accs.append(res)
            
    ####################

    print(len(accs))
    pprint(accs[:10])
    print(f"save to {file_path}/accs.json")
    save_json(accs, f"{file_path}/accs.json")



if __name__ == '__main__':
    main()
