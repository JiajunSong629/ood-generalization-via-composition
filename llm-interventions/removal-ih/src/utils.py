import time 
import re

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


def extract_numerical(text, from_ground_truth = False):
    '''
    This function for extract numerical numbers from a answer for exmaple in GSM dataset.
    '''
    pattern = r'\n#### (\d+)'
    match = re.search(pattern, text)
    if from_ground_truth:
        if match:
            ground_truth = match.group(1)  # This extracts the numerical part
            return ground_truth
        else:
            print("cannot extract ground truth from answer")
        return ""
    else:
        if match:
            final_answer = match.group(1)  # This extracts the numerical part
            # Optionally, truncate the text up to the matched pattern
            end_position = match.end()
            text = text[:end_position]
            # print("Extracted number:", final_answer)
            # print("Truncated text:", text)
            return final_answer, text
        else:
            print("Pattern not found in the generated text.")
        return "Not find", text

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

# Function to save a list of JSON objects to a file, each JSON object on a new line
def save_json_lines(data, file_path):
    with open(file_path, 'w') as file:
        for json_obj in data:
            json_line = json.dumps(json_obj)
            file.write(json_line + '\n')
