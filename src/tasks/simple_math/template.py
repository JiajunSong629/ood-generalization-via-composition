import random

SYMBOL_NAMES = ["$#", "@%", "!&", "*#", "#@", "%$", "&*", "##", "@!", "%%"]
SYMBOL_ITEMS = ["!@", "#$", "%^", "&*", "*@", "$#", "@&", "#%", "!$", "**"]


def get_random_name(use_symbols=False):
    if use_symbols:
        return random.choice(SYMBOL_NAMES)
    return random.choice(
        [
            "Alice",
            "Bob",
            "Charlie",
            "Diana",
            "Eve",
            "Frank",
            "Grace",
            "Hank",
            "Ivy",
            "Jack",
            "Karen",
            "Leo",
            "Maya",
            "Noah",
            "Olivia",
            "Peter",
        ]
    )


def get_random_item(use_symbols=False):
    if use_symbols:
        return random.choice(SYMBOL_ITEMS)
    return random.choice(
        [
            "apples",
            "bananas",
            "candies",
            "oranges",
            "pencils",
            "erasers",
            "notebooks",
            "chocolates",
            "books",
            "toys",
            "shirts",
            "bottles",
        ]
    )


class TaskTemplate:
    def __init__(
        self,
        type_name,
        question_template,
        deduction_template,
        variable_generator,
        answer_generator,
    ):
        self.type = type_name
        self.question_template = question_template
        self.deduction_template = deduction_template
        self.variable_generator = variable_generator
        self.answer_generator = answer_generator

    def generate(self, setting="original", show_deduction=True):
        while True:
            variables = self.variable_generator(setting)
            answer_dict = self.answer_generator(variables)
            answer_dict.update(variables)

            # Ensure positive numbers
            if all(isinstance(v, (int, str)) or v > 0 for v in answer_dict.values()):
                formatted_question = self.question_template.format(**answer_dict)
                if show_deduction:
                    formatted_question += "\n" + self.deduction_template.format(
                        **answer_dict
                    )

                # Extract answer variable from the last line
                answer_var = self._extract_answer_variable()
                return {
                    "type": self.type,
                    "question": formatted_question,
                    "answer": str(answer_dict[answer_var]),
                }

    def _extract_answer_variable(self):
        last_line = self.deduction_template.split("\n")[-1]
        import re

        match = re.search(r"{(\w+)}", last_line)
        return match.group(1) if match else None


task_templates = [
    TaskTemplate(
        type_name="Find the Total Price",
        question_template="""Question: {name} buys {qty1} {item1} for ${price1} each and {qty2} {item2} for ${price2} each. What is the total cost in dollars?""",
        deduction_template="""Let's solve this step by step:
1) Cost of {item1}: {qty1} × ${price1} = ${cost1}
2) Cost of {item2}: {qty2} × ${price2} = ${cost2}
3) Total cost: ${cost1} + ${cost2} = ${total}
#### {total}""",
        variable_generator=lambda setting="original": {
            "name": get_random_name(setting == "symbol"),
            "qty1": random.randint(1, 5),
            "item1": get_random_item(setting == "symbol"),
            "price1": random.randint(1, 5),
            "qty2": random.randint(1, 5),
            "item2": get_random_item(setting == "symbol"),
            "price2": random.randint(1, 5),
        },
        answer_generator=lambda vars: {
            "cost1": vars["qty1"] * vars["price1"],
            "cost2": vars["qty2"] * vars["price2"],
            "total": vars["qty1"] * vars["price1"] + vars["qty2"] * vars["price2"],
        },
    ),
    TaskTemplate(
        type_name="How Many Items",
        question_template="""Question: {name} buys {qty1} {item1}, {qty2} {item2}, and {qty3} {item3}. How many items does {name} buy in total?""",
        deduction_template="""Let's solve this step by step:
1) First, list all items:
   * {qty1} {item1}
   * {qty2} {item2}
   * {qty3} {item3}
2) Add all quantities: {qty1} + {qty2} + {qty3} = {total}
#### {total}""",
        variable_generator=lambda setting="original": {
            "name": get_random_name(setting == "symbol"),
            "qty1": random.randint(1, 10),
            "item1": get_random_item(setting == "symbol"),
            "qty2": random.randint(1, 10),
            "item2": get_random_item(setting == "symbol"),
            "qty3": random.randint(1, 10),
            "item3": get_random_item(setting == "symbol"),
        },
        answer_generator=lambda vars: {
            "total": vars["qty1"] + vars["qty2"] + vars["qty3"],
        },
    ),
    TaskTemplate(
        type_name="Simple Change Calculation",
        question_template="""Question: {name} has ${money}. {name} buys {qty} {item} for ${price} each. How much change does {name} get?""",
        deduction_template="""Let's solve this step by step:
1) Calculate total cost: {qty} × ${price} = ${total_cost}
2) Calculate change: ${money} - ${total_cost} = ${change}
#### {change}""",
        variable_generator=lambda setting="original": {
            "name": get_random_name(setting == "symbol"),
            "money": random.randint(20, 50),
            "qty": random.randint(1, 5),
            "item": get_random_item(setting == "symbol"),
            "price": random.randint(2, 8),
        },
        answer_generator=lambda vars: {
            "total_cost": vars["qty"] * vars["price"],
            "change": vars["money"] - (vars["qty"] * vars["price"]),
        },
    ),
    TaskTemplate(
        type_name="Distribute Items Equally",
        question_template="""Question: {name} has {total_items} {item} and wants to put them into {boxes} boxes equally. How many {item} will be in each box?""",
        deduction_template="""Let's solve this step by step:
1) Divide {total_items} by {boxes}: {total_items} ÷ {boxes} = {per_box}
#### {per_box}""",
        variable_generator=lambda setting="original": {
            "name": get_random_name(setting == "symbol"),
            "total_items": random.randint(12, 48),
            "item": get_random_item(setting == "symbol"),
            "boxes": random.choice([2, 3, 4, 6]),
        },
        answer_generator=lambda vars: {
            "per_box": vars["total_items"] // vars["boxes"],
        },
    ),
]


def generate_task_with_context(setting="original", num_shots=2):
    """Generate a task with balanced in-context examples.

    Args:
        setting (str): Either "original" or "symbol" for name/item style
        num_shots (int): Number of example questions per template type

    Returns:
        dict: Contains the full prompt with balanced examples and the target question
    """
    # Generate balanced examples
    examples = []
    shots_per_template = num_shots // len(task_templates)
    remaining_shots = num_shots % len(task_templates)

    for template in task_templates:
        # Generate shots for this template
        template_shots = shots_per_template + (1 if remaining_shots > 0 else 0)
        remaining_shots -= 1 if remaining_shots > 0 else 0

        for _ in range(template_shots):
            examples.append(template.generate(setting, show_deduction=True))

    # Shuffle examples to avoid pattern recognition
    random.shuffle(examples)

    # Generate target question
    target = random.choice(task_templates).generate(setting, show_deduction=False)

    # Combine into prompt
    full_prompt = ""
    for example in examples:
        full_prompt += f"{example['question']}\n\n"
    full_prompt += target["question"]

    return {"prompt": full_prompt, "answer": target["answer"], "type": target["type"]}


if __name__ == "__main__":
    # Test the task generation with context
    task = generate_task_with_context(setting="symbol", num_shots=4)
    print("Generated Task:")
    print("-" * 50)
    print(task["prompt"])
    print("-" * 50)
    print(f"Answer: {task['answer']}")