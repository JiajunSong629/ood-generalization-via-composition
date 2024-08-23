import torch
import random

MAPPING = {
    "volleyball": "animal",
    "onions": "sport",
    "broccoli": "sport",
    "hockey": "animal",
    "kale": "sport",
    "beet": "sport",
    "golf": "animal",
    "horse": "plant/vegetable",
    "corn": "sport",
    "football": "animal",
    "luge": "animal",
    "bowling": "animal",
    "beans": "sport",
    "archery": "animal",
    "sheep": "plant/vegetable",
    "zucchini": "sport",
    "goldfish": "plant/vegetable",
    "duck": "plant/vegetable",
    "leopard": "plant/vegetable",
    "lacrosse": "animal",
    "badminton": "animal",
    "lion": "plant/vegetable",
    "celery": "sport",
    "porcupine": "plant/vegetable",
    "wolf": "plant/vegetable",
    "lettuce": "sport",
    "camel": "plant/vegetable",
    "billiards": "animal",
    "zebra": "plant/vegetable",
    "radish": "sport",
    "llama": "plant/vegetable",
    "cat": "plant/vegetable",
    "elephant": "plant/vegetable",
    "monkey": "plant/vegetable",
    "panda": "plant/vegetable",
    "cucumber": "sport",
    "peas": "sport",
    "tomato": "sport",
    "spinach": "sport",
    "carrots": "sport",
    "rugby": "animal",
    "cycling": "animal",
    "baseball": "animal",
    "tennis": "animal",
    "judo": "animal",
}


REMAP = {"animal": "$#", "sport": "!%", "plant/vegetable": "&*"}
REMAP_PERMUTE = {"animal": "animal", "sport": "sport", "plant/vegetable": "plant"}
REMAP_ORIG = {"animal": "sport", "sport": "plant", "plant/vegetable": "animal"}

def generator(n_examples: int = 10, template: callable = None, permute = False, symbol=False, balanced_sample = True, **args):
    if permute:
        remap = REMAP_PERMUTE
    elif symbol:
        remap = REMAP
    else:
        remap = REMAP_ORIG

    # Organize items by category, for banlanced sampling
    categories = {"animal": [], "sport": [], "plant/vegetable": []}
    for item, label in MAPPING.items():
        categories[label].append(item)

    item_label_pair = [(i, l) for i, l in MAPPING.items()]
    while True:
        prompt = ""
        selected = set()
        
        if balanced_sample:
            # Ensure one item from each category
            for category, items in categories.items():
                item = random.choice(items)
                label = remap[category]
                prompt += template(item, label)
                selected.add(item)

            # Add more items to reach n_examples, if necessary
            remaining_items = [(item, label) for item, label in MAPPING.items() if item not in selected]
            for item, label in random.choices(remaining_items, k=n_examples - len(categories)):
                label = remap[label]
                prompt += template(item, label)
                selected.add(item)

        else:
            for i, l in random.choices(item_label_pair, k=n_examples):
                label = remap[l]
                prompt += template(i, label)
                selected.add(i)

        item, answer = random.choice(
            [(i, l) for i, l in item_label_pair if i not in selected]
        )
        prompt += template(item)  # f"{item}:"

        test_answer = remap[answer]


        if "multiple_choice" in args and args["multiple_choice"]:
            choices = list(remap.values())
            yield prompt, test_answer, choices
            continue
        yield prompt, test_answer
