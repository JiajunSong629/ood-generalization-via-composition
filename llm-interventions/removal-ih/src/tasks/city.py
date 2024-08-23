import random


MAPPING = {
    "Toronto": "Canada",
    "Montreal": "Canada",
    "Calgary": "Canada",
    "Ottawa": "Canada",
    "Edmonton": "Canada",
    "Winnipeg": "Canada",
    "Mississauga": "Canada",
    "Vancouver": "Canada",
    "Brampton": "Canada",
    "Hamilton": "Canada",
    "Atlanta": "US",
    "Miami": "US",
    "Orlando": "US",
    "Charleston": "US",
    "Savannah": "US",
    "Nashville": "US",
    "Chicago": "US",
    "Detroit": "US",
    "Minneapolis": "US",
    "Seattle": "US",
    "Denver": "US",
    "London": "England",
    "Bath": "England",
    "Manchester": "England",
    "Liverpool": "England",
    "Oxford": "England",
    "Cambridge": "England",
    "Bristol": "England",
    "York": "England",
    "Brighton": "England",
    "Birmingham": "England",
    "Leeds": "England",
    "Sheffield": "England",
    "Leicester": "England",
    "Nottingham": "England",
}

REMAP = {"Canada": "$#", "US": "!%", "England": "&*"}


def generator(n_examples: int = 10, template: callable = None):
    item_label_pair = [(i, l) for i, l in MAPPING.items()]
    while True:
        prompt = ""
        selected = set()
        for i, l in random.choices(item_label_pair, k=n_examples):
            prompt += template(i, REMAP[l])  # f"{i}:{REMAP[l]}, "
            selected.add(i)

        item, answer = random.choice(
            [(i, l) for i, l in item_label_pair if i not in selected]
        )
        prompt += template(item)  # f"{item}:"

        yield prompt, REMAP[answer]
