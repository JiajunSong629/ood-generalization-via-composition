import torch
from transformers import LlamaForCausalLM
import os
import random
import json

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, indent = 4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)

# Function to load JSON data from a file line by line
def load_json_lines(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

import time 

class Timer(object):

    def __init__(self):

        self.start()

    def start(self):
        self.v = time.time()

    def end(self):
        return time.time() - self.v


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t > 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="seed", default=42)
    parser.add_argument("--random_mask_head", action='store_true')
    return parser.parse_args()




# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the 'factory/classnames.txt'
# file_path = os.path.join(current_dir, "IH_scores_Llama_2_70b_hf.json")
file_path = os.path.join(current_dir, "IH_llama2_70b.json")


def load_model(model_path = "meta-llama/Llama-2-70b-hf"):
    # Load the model
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, output_attentions=False, low_cpu_mem_usage=True, device_map="auto", attn_implementation="eager")
    # Save the state_dict only run once!
    state_dict_path = "llama_2_70b_state_dict.pth"  # Specify your path here
    torch.save(model.state_dict(), state_dict_path)
    
    return model




## edit the model state_dict
def edit_model_weight(ckpt, ids_to_mask):
    '''
    Masks out the query weights for specified attention heads in the model's checkpoint.

    Parameters:
    - ckpt: The state_dict of the model.
    - ids_to_mask: A list of [layer_id, head_id] pairs indicating which heads to mask.

    Returns:
    - A modified checkpoint with the specified heads' query weights set to zero.

    ### details
    Note: In llama2-70b, the num_key_value is 8, num_attention_heads is 64, indicating they use grouped-query attention, 
    see https://huggingface.co/TheBloke/Llama-2-70B-fp16/blob/main/config.json
    see details in https://github.com/huggingface/transformers/blob/8e3e145b427196e014f37aa42ba890b9bc94275e/src/transformers/models/llama/configuration_llama.py#L53-L60


    see model weight shape below:
    model.layers.0.self_attn.q_proj.weight: torch.Size([8192, 8192])
    model.layers.0.self_attn.k_proj.weight: torch.Size([1024, 8192])
    model.layers.0.self_attn.v_proj.weight: torch.Size([1024, 8192])
    model.layers.0.self_attn.o_proj.weight: torch.Size([8192, 8192])

    '''
    for layer_id, head_id in ids_to_mask:
        # Adjusted keys based on the provided model structure
        q_proj_key = f"model.layers.{layer_id}.self_attn.q_proj.weight"
        o_proj_key = f"model.layers.{layer_id}.self_attn.o_proj.weight"

        # Note: The size calculation might need to be adjusted based on the model's architecture.
        # For LLaMA, it seems the projection might be split differently. 
        # will need to confirm how the model splits its attention heads.
        num_attention_heads = 64  # Example, adjust based on actual model configuration
        head_size = ckpt[q_proj_key].size(0) // num_attention_heads
        start_index = head_id * head_size
        end_index = start_index + head_size

        # Zero out the weights for the specified head
        ckpt[q_proj_key][start_index:end_index, :] = 0

        # Zero out the weights for the specified columns in o_proj
        # Assuming normal operation without pretraining_tp partitioning
        ckpt[o_proj_key][:, start_index:end_index] = 0

    return ckpt

def main():
    # load_model()
    # return

    args = parse_args()
    seed = args.seed
    random_mask_head = args.random_mask_head
    num_heads = 300 
    # heads = load_json_lines(file_path)
    # all_mask_layer_head_id = [dic["IH"] for dic in heads[:num_heads]]

    heads = load_json(file_path)
    all_mask_layer_head_id = [[dic[str(idx)][0],dic[str(idx)][1]] for idx,dic in enumerate(heads[:num_heads])]
    
    # mask_layer_head_id = get_top_layer_head_ids_llama(scale = "70b", num_heads = 300)
    all_pairs = [(x, y) for x in range(80) for y in range(64)]
    
    timer = Timer()
    modified_state_dict_path = os.path.join(current_dir,"ckpt", "llama_2_70b_state_dict.pth")
    
    rnd = random.Random()
    rnd.seed(seed)
    
    # for mask_head in [10,20,30,40,50]:
    # for mask_head in [100, 150, 200, 250]:
    for mask_head in [50]:
        print("load ckpt ...")
        ckpt = torch.load(modified_state_dict_path)
        print(f"load done, time use {time_str(timer.end())}")

        if random_mask_head:
            # Randomly subsample mask_head indices without replacement
            saved_state_dict_path = f"llama_2_70b_mask{mask_head}_random_seed{seed}.pth"
            mask_layer_head_id = rnd.sample(all_pairs, mask_head)
            # print("random sample mask_layer_head_id: ", mask_layer_head_id)
        else:
            saved_state_dict_path = f"llama_2_70b_mask{mask_head}.pth"
            mask_layer_head_id = all_mask_layer_head_id[:mask_head]
            # print("slice mask_layer_head_id: ", mask_layer_head_id)
        edited_ckpt = edit_model_weight(ckpt, mask_layer_head_id)
        torch.save(edited_ckpt, os.path.join(current_dir,"ckpt", saved_state_dict_path))
        print(f"mask head: {mask_head} | time elpsed {time_str(timer.end())}")

        # Free memory
        print("free mem")
        del edited_ckpt
        del ckpt
        torch.cuda.empty_cache()
        print(f"mask head: {mask_head} done | time elpsed {time_str(timer.end())}")


    return

if __name__ == "__main__":
    main()

