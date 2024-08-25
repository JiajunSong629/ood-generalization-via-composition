import torch
import numpy as np
import os
from transformers import set_seed
from transformers import GPT2Model, GPT2Config
from transformers import GPT2Tokenizer

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM
from transformers import GemmaForCausalLM
import json
from pdb import set_trace as pds
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

# Function to save a list of JSON objects to a file, each JSON object on a new line
def save_json_lines(data, file_path):
    with open(file_path, 'w') as file:
        for json_obj in data:
            json_line = json.dumps(json_obj)
            file.write(json_line + '\n')

SEED = 2024

set_seed(SEED)
np.random.seed(SEED)


def measure_IH_shifting(attentions, T0, num_layer, num_heads, seq_len, is_IH=True): # / updated / #
    """
    Given all attention outputs (the one after softmax), which is of shape
    (num_layer, sample_size, num_head, seq_len, seq_len), returns # / updated / #
        IH_list, a list of (layer, head) pairs, which is ordered from
            most likely to be an induction head to least likely
        scores, a list of float, which is the induction head score used to
            rank the (layer, head) pairs
    """
    scores = np.zeros((num_layer, num_heads))
    #for layer in range(num_layer):  # / updated / #
    #    for head in range(num_heads): # / updated / #
    #        A = attentions[layer, head]  # / updated / #
    #        A_adjusted = np.zeros((seq_len, seq_len)) # / updated / #
    #        A_adjusted[1:, 1:] = A[1:, 1:] / np.sum(A[1:, 1:], axis=1, keepdims=True) # / updated / #
    #        diag1 = np.diag(A_adjusted, -(T0 - 1))[1:] # / updated / #
    #        diag2 = np.diag(A_adjusted, -(2 * T0 - 1))[1:] # / updated / #
    #        diag = np.concatenate((diag1[:-T0], diag1[-T0:] + diag2)) # / updated / #
    #        scores[layer, head] = np.mean(diag) # / updated / #

    offset = -(T0 - 1) if is_IH else -1 # / updated / #
    sample_size = attentions.shape[1] # / updated / #
    for layer in range(num_layer): # / updated / #
        for head in range(num_heads): # / updated / #
            A = attentions[layer, :, head] # / updated / #
            A_adjusted = np.zeros((sample_size, seq_len, seq_len)) # / updated / #
            A_adjusted[:, 1:, 1:] = A[:, 1:, 1:] / np.sum(A[:, 1:, 1:], axis=2, keepdims=True) # / updated / #
            scores[layer, head] = np.mean(np.array([np.mean(np.diag(A_adjusted[i], offset)[1:]) for i in range(sample_size)])) # / updated / #

    idx_sort = np.argsort(scores, axis=None)[::-1]
    head_list = [  # / updated / #
        [idx_sort[j] // num_heads, idx_sort[j] % num_heads]
        for j in range(len(idx_sort))
    ]

    ##zhuoyan
    sorted_scores = scores.flatten()[idx_sort]
    return head_list, sorted_scores.tolist() # / updated / #

# Load model

# configuration = GPT2Config()
# model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
# configuration = model.config
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# vocab_size = 50257
# T0 = 5
# num_layer = 12
# num_heads = 12
# d_model = 768
# d_head = d_model // num_heads

# print(model)
# print(configuration)

print("===========================================")


# model_path = "meta-llama/Llama-2-70b-hf"
# model_path = "mistralai/Mixtral-8x7B-v0.1"


model_path = "meta-llama/Llama-2-7b-chat-hf"
model_path = "mistralai/Mistral-7B-v0.1"
model_path = "google/gemma-7b-it"
model_path = "tiiuae/falcon-7b"
model_path = "allenai/OLMo-1.7-7B-hf"
model_path = "EleutherAI/pythia-6.9b"

for model_path in [
    # "meta-llama/Llama-2-7b-hf",
    # "mistralai/Mistral-7B-v0.1",
    "google/gemma-7b-it",
    # "tiiuae/falcon-7b",
    # "allenai/OLMo-1.7-7B-hf",
    # "EleutherAI/pythia-6.9b",
    # "meta-llama/Meta-Llama-3-8B",
    # "tiiuae/falcon-11B",
]:
    model_name = model_path.split("/")[1].replace("-","_").replace(".","_")
    NUM_TOP_HEADS = 1000
    json_path = f'IH_scores_{model_name}_{NUM_TOP_HEADS}.json'

    print(f"investigate model: {model_name}")
    print(f"save to {json_path}")



    if model_path == "meta-llama/Llama-2-7b-hf":

        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True)


        print(model)
        print(model.config)

        vocab_size = 32000
        T0 = 25    # / updated / #
        num_layer = 32
        num_heads = 32
        d_model = 4096
        d_head = d_model // num_heads

        print("===========================================")
    
    if model_path == "meta-llama/Meta-Llama-3-8B":

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True)


        print(model)
        print(model.config)

        vocab_size = 128256
        T0 = 25    # / updated / #
        num_layer = 32
        num_heads = 32
        d_model = 4096
        d_head = d_model // num_heads

        print("===========================================")
    

    if model_path == "mistralai/Mistral-7B-v0.1":
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print(model)
        print(model.config)

        vocab_size = 32000
        T0 = 25    # / updated / #
        num_layer = 32
        num_heads = 32
        d_model = 4096
        d_head = d_model // num_heads
        
    if model_path == "meta-llama/Llama-2-70b-hf":
        # model_path = "meta-llama/Llama-2-7b-chat-hf"

        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True, low_cpu_mem_usage=True, device_map="auto")

        print(model)
        print(model.config)

        vocab_size = 32000
        T0 = 25    # / updated / #
        num_layer = 80
        num_heads = 64
        d_model = 8192 
        d_head = d_model // num_heads

        print("===========================================")

    if model_path == "mistralai/Mixtral-8x7B-v0.1":
        # model_path = "meta-llama/Llama-2-7b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True, low_cpu_mem_usage=True, device_map="auto")

        print(model)
        print(model.config)

        vocab_size = 32000
        T0 = 25    # / updated / #
        num_layer = 80
        num_heads = 64
        d_model = 8192 
        d_head = d_model // num_heads

        print("===========================================")
        assert False

    if model_path == "google/gemma-7b-it":

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GemmaForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True, attn_implementation="eager")


        print(model)
        print(model.config)

        vocab_size = 256000
        T0 = 25    # / updated / #
        num_layer = 28
        num_heads = 16
        d_head = 256

        print("===========================================")

    if model_path == "tiiuae/falcon-7b":

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True)


        print(model)
        print(model.config)

        vocab_size = 65024
        T0 = 25    # / updated / #
        num_layer = 32
        num_heads = 71
        d_model = 4544
        d_head = d_model // num_heads

        print("===========================================")
            
    if model_path == "tiiuae/falcon-11B":

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True)


        print(model)
        print(model.config)

        vocab_size = 65024
        T0 = 25    # / updated / #
        num_layer = 60
        num_heads = 32
        d_model = 4096

        d_head = d_model // num_heads

        print("===========================================")
    
    if model_path == "allenai/OLMo-1.7-7B-hf":

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True)


        print(model)
        print(model.config)

        vocab_size = 50304
        T0 = 25    # / updated / #
        num_layer = 32
        num_heads = 32
        d_model = 4096
        d_head = d_model // num_heads

        print("===========================================")

    if model_path == "EleutherAI/pythia-6.9b":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, output_attentions=True)


        print(model)
        print(model.config)

        vocab_size = 50432
        T0 = 25    # / updated / #
        num_layer = 32
        num_heads = 32
        d_model = 4096
        d_head = d_model // num_heads

        print("===========================================")



    # Prepare inputs, randomly repeated sequence  
    # sample_int = ( # / updated / #
    #    np.random.randint(low=0, high=vocab_size, size=T0) # / updated / #
    #    .repeat(3) # / updated / #
    #    .reshape(T0, -1) # / updated / #
    #    .T.ravel() # / updated / #
    #)

    sample_int = np.zeros((100, 2*T0)) # / updated / #
    for i in range(100): # / updated / #
        sample_int[i] = ( # / updated / #
            np.random.randint(low=0, high=vocab_size, size=T0) # / updated / #
            .repeat(2) # / updated / #
            .reshape(T0, -1) # / updated / #
            .T.ravel()) # / updated / #

    #input_ids = torch.Tensor(sample_int).long().unsqueeze(0).cuda() # / updated / #
    input_ids = torch.Tensor(sample_int).long().cuda() # / updated / #
    seq_len = input_ids.size(1)

    model.to("cuda")
    with torch.no_grad():
        output = model(input_ids)
    #attentions = np.array([a.numpy(force=True)[0] for a in output.attentions]) # / updated / #
    attentions = np.array([a.numpy(force=True) for a in output.attentions]) # / updated / #
    IH_list, scores = measure_IH_shifting(attentions, T0, num_layer, num_heads, seq_len, is_IH=True) # / updated / #

    print("IH_list: ", IH_list[:NUM_TOP_HEADS])
    print("scores: ", scores[:NUM_TOP_HEADS])

    # Prepare the data for JSON saving
    combined_data = [{"IH": [int(pair[0]), int(pair[1])], "score": float(score)} for pair, score in zip(IH_list, scores)]

    # Save the combined data to a JSON file
    import json
    model_name = model_path.split("/")[1].replace("-","_").replace(".","_")
    json_path = f'IH_scores_{model_name}_{NUM_TOP_HEADS}.json'
    save_json_lines(combined_data, json_path)
