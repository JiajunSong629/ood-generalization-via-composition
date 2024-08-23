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

# heads = load_json_lines("IH_scores_Llama_2_70b_hf_1000.json")
# print(type(heads))
# print(len(heads))
# print(heads[:5])

import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the 'factory/classnames.txt'
# file_path = os.path.join(current_dir, "IH_scores_Llama_2_7b_hf_1000.json")
# file_path = os.path.join(current_dir, "IH_scores_Llama_2_70b_hf.json")
# file_path = os.path.join(current_dir, "IH_scores_Meta_Llama_3_8B_1000.json")


# def get_top_layer_head_ids_llama(scale = "7b", num_heads = 200, file_path = file_path):
#     if scale == "7b":
#         ids = [[8, 26],
#             [11, 15],
#             [6, 9],
#             [6, 30],
#             [12, 26],
#             [16, 19],
#             [21, 30],
#             [26, 28],
#             [7, 13],
#             [6, 11],
#             [7, 4],
#             [17, 22],
#             [19, 15],
#             [12, 2],
#             [18, 30],
#             [13, 11],
#             [22, 22],
#             [3, 6],
#             [24, 3],
#             [16, 24],
#             [7, 28],
#             [8, 31],
#             [11, 18],
#             [19, 10],
#             [11, 2],
#             [22, 27],
#             [24, 11],
#             [13, 23],
#             [22, 8],
#             [20, 8],
#             [17, 16],
#             [21, 1],
#             [20, 29],
#             [19, 9],
#             [21, 16],
#             [22, 19],
#             [24, 8],
#             [23, 7],
#             [20, 30],
#             [21, 27],
#             [25, 17],
#             [6, 14],
#             [21, 28],
#             [15, 14],
#             [23, 20],
#             [20, 27],
#             [20, 1],
#             [21, 5],
#             [24, 30],
#             [23, 8]]
#     elif scale == "70b":
#         heads = load_json_lines(file_path)
#         ids = [dic["IH"] for dic in heads[:num_heads]]
#     return ids

# def get_top_layer_head_ids_llama(scale = "70b", num_heads = 300, file_path = file_path):
#     heads = load_json_lines(file_path)
#     ids = [dic["IH"] for dic in heads[:num_heads]]
#     return ids

# file_path = os.path.join(current_dir, "IH_llama2_7b.json")
file_path = os.path.join(current_dir, "IH_llama3_8b.json")
def get_top_layer_head_ids(scale = "7b", num_heads = 200, file_path = file_path):
    heads = load_json(file_path)
    ids = [[dic[str(idx)][0],dic[str(idx)][1]] for idx,dic in enumerate(heads[:num_heads])]
    return ids

print(get_top_layer_head_ids())