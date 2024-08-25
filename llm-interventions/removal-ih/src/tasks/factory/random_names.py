import numpy as np
import sys
from collections import Counter
import random

import json

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, indent = 4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)


# I'll read from a file named "classnames.txt" and convert its contents into a list
file_path = 'classnames.txt'

def read_txt(file_path):
    long = []
    # Reading the file and converting each line into an item in a list
    with open(file_path, 'r') as file:
        for line in file.readlines():
            name = line[10:].split("\n")[0].lower().replace("(","").replace(")","").replace("/","")
            name = name.split(" ")
            name = [i for i in name if i != ""]
            long += name
            if "" in name:
                print(name)
            
            
        retlist = sorted(set(long))

    return retlist


def main():
    #################################### classnames
    
    names = read_txt(file_path)
    print("total names: ", len(names))
    print("names: ", names[:5])

    
    

    # count = Counter(names)
    # print(type(count))
    # for key,val in count.items():
    #     if val > 1:
    #         print(key)
    # return

    random.seed(123)  # You can choose any number as your seed

    res = []
    while len(res) < 500:
        tokens_ori = list(random.sample(names, 15))
        tokens = tokens_ori[:]
        tokens += tokens_ori[:5]
        
        
        sen = " ".join(tokens)

        output = tokens_ori[5:]
        sen_out = " ".join(output)


        ex = {"input": sen, "output": sen_out}
        res.append(ex)
    
    print(len(res))
    print(res[:5])



    # save_json(res,"data/factory/random_factory.json")




if __name__ == "__main__":
    main()

