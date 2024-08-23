import json
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

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

# file_path = os.path.join(current_dir, "IH_scores_OLMo_1_7_7B_hf_1000.json")

# heads = load_json_lines(file_path)
# print(type(heads))
# print(len(heads))
# print(heads[:5])


# def get_top_layer_head_ids_gemma(scale = "7b", num_heads = 200, file_path = file_path):
#     heads = load_json_lines(file_path)
#     ids = [dic["IH"] for dic in heads[:num_heads]]
#     return ids
# print(get_top_layer_head_ids_gemma())


file_path = os.path.join(current_dir, "IH_olmo_7b.json")
def get_top_layer_head_ids(scale = "7b", num_heads = 200, file_path = file_path):
    heads = load_json(file_path)
    ids = [[dic[str(idx)][0],dic[str(idx)][1]] for idx,dic in enumerate(heads[:num_heads])]
    return ids

print(get_top_layer_head_ids())

