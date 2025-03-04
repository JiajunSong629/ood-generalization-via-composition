import random

SYMBOL_NAMES = [
    "$#",
    "@%",
    "!&",
    "*#",
    "#@",
    "%$",
    "&*",
    "##",
    "@!",
    "%%",
    "!#",
    "@$",
    "#%",
    "$^",
    "%&",
    "&@",
    "*!",
    "#*",
    "@&",
    "!%",
    "$@",
    "#!",
    "%#",
    "&$",
    "!@",
    "*%",
    "@#",
    "!*",
    "#&",
    "%@",
    "&!",
    "*$",
    "@^",
    "#%",
    "$*",
    "%!",
    "&@",
    "*#",
    "!$",
    "@%",
]

SYMBOL_ITEMS = [
    "!@",
    "#$",
    "%^",
    "&*",
    "*@",
    "$#",
    "@&",
    "#%",
    "!$",
    "**",
    "@!",
    "#@",
    "$%",
    "%*",
    "&!",
    "*#",
    "!&",
    "@$",
    "#^",
    "%@",
    "&*",
    "*!",
    "@#",
    "#!",
    "$@",
    "%#",
    "&$",
    "*%",
    "!@",
    "@^",
    "#*",
    "$!",
    "%@",
    "&#",
    "*$",
    "!%",
    "@&",
    "#@",
    "$*",
    "%!",
]


def get_random_name(setting):
    if setting == "symbol":
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


def get_random_item(setting):
    if setting == "symbol":
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
        self.type_name = type_name
        self.question_template = question_template
        self.deduction_template = deduction_template
        self.variable_generator = variable_generator
        self.answer_generator = answer_generator

    def generate(self, setting, show_deduction=True):
        variables = self.variable_generator(setting)
        answer_dict = self.answer_generator(variables)
        answer_dict.update(variables)

        formatted_question = self.question_template.format(**answer_dict)
        formatted_deduction = self.deduction_template.format(**answer_dict)
        if show_deduction:
            formatted_question += "\n" + formatted_deduction

        # Extract answer variable from the last line
        import re

        match = re.search(r"#### (\w+)", formatted_deduction)
        answer = match.group(1) if match else None

        return {
            "type_name": self.type_name,
            "question": formatted_question,
            "answer": answer,
        }


task_templates = [
    TaskTemplate(
        type_name="Tree Logging Calculation",
        question_template="""Question: {name} is cutting up wood for his wood-burning stove. Each {pine} tree makes {pine_logs} logs, each {maple} tree makes {maple_logs} logs, and each {walnut} tree makes {walnut_logs} logs. If {name} cuts up {pine_count} {pine} trees, {maple_count} {maple} trees, and {walnut_count} {walnut} trees, how many logs does he get?""",
        deduction_template="""Answer: First find the total number of {pine} logs by multiplying the number of trees by the number of logs per tree: {pine_logs} logs/{pine} * {pine_count} {pine} = <<{pine_logs}*{pine_count}={total_pine}>>{total_pine} logs
Then do the same thing for the maple trees: {maple_logs} logs/{maple} * {maple_count} {maple} = <<{maple_logs}*{maple_count}={total_maple}>>{total_maple} logs
And do the same thing for the walnut trees: {walnut_logs} logs/{walnut} * {walnut_count} {walnut} = <<{walnut_logs}*{walnut_count}={total_walnut}>>{total_walnut} logs
Finally, add up the number of logs from each type of tree to find the total number: {total_pine} logs + {total_maple} logs + {total_walnut} logs = <<{total_pine}+{total_maple}+{total_walnut}={total}>>{total} logs
#### {total}""",
        variable_generator=lambda setting: {
            "name": get_random_name(setting),
            "pine": get_random_item(setting),
            "maple": get_random_item(setting),
            "walnut": get_random_item(setting),
            "pine_logs": random.randint(50, 100),
            "maple_logs": random.randint(40, 80),
            "walnut_logs": random.randint(60, 120),
            "pine_count": random.randint(5, 10),
            "maple_count": random.randint(2, 5),
            "walnut_count": random.randint(3, 6),
        },
        answer_generator=lambda vars: {
            "total_pine": vars["pine_logs"] * vars["pine_count"],
            "total_maple": vars["maple_logs"] * vars["maple_count"],
            "total_walnut": vars["walnut_logs"] * vars["walnut_count"],
            "total": (
                vars["pine_logs"] * vars["pine_count"]
                + vars["maple_logs"] * vars["maple_count"]
                + vars["walnut_logs"] * vars["walnut_count"]
            ),
        },
    ),
    TaskTemplate(
        type_name="Fruit Roll-Up Contest",
        question_template="""Question: {A} and {B} are having a contest to see who can eat the most fruit roll-ups, so they unroll as many as they can find. Unfortunately, someone makes a mistake and {A}'s was {A_roll} roll-ups wide and {A_num_roll} rolls up long while {B}'s was {B_roll} roll-ups wide and {B_num_roll} roll-ups long. If they both ate their entire amount, how many did they eat on average?""",
        deduction_template="""Answer: {A} ate {A_total} because {A_roll} x {A_num_roll} = <<{A_roll}*{A_num_roll}={A_total}>>{A_total}.
{B} ate {B_total} because {B_roll} x {B_num_roll} = <<{B_roll}*{B_num_roll}={B_total}>>{B_total}.
In total they ate {total} because {A_total} + {B_total} = <<{A_total}+{B_total}={total}>>{total}.
On average they ate {average} because {total} / 2 = <<{total}/2={average}>>{average}.
#### {average}""",
        variable_generator=lambda setting: {
            "A": get_random_name(setting),
            "B": get_random_name(setting),
            "A_roll": random.choice([2, 3, 4]),
            "A_num_roll": random.choice([2, 4, 8]),
            "B_roll": random.choice([2, 3, 4]),
            "B_num_roll": random.choice([2, 4, 8]),
        },
        answer_generator=lambda vars: {
            "A_total": vars["A_roll"] * vars["A_num_roll"],
            "B_total": vars["B_roll"] * vars["B_num_roll"],
            "total": vars["A_roll"] * vars["A_num_roll"]
            + vars["B_roll"] * vars["B_num_roll"],
            "average": (
                vars["A_roll"] * vars["A_num_roll"]
                + vars["B_roll"] * vars["B_num_roll"]
            )
            // 2,
        },
    ),
    TaskTemplate(
        type_name="Batting cages",
        question_template="""Question: {name1} and {name2} went to the batting cages. Each token gets you {pitches} pitches. {name1} used {tokens1} tokens and {name2} used {tokens2} tokens. {name1} hit the ball {hits1} times and {name2} hit the ball {hits2} times. How many pitches did {name1} and {name2} miss altogether?""",
        deduction_template="""Answer: {name1} used {tokens1} tokens which are worth {pitches} pitches each, so {tokens1} tokens x {pitches} pitches = <<{tokens1}*{pitches}={total_pitches1}>>{total_pitches1} pitches.
{name2} used {tokens2} tokens x {pitches} pitches = <<{tokens2}*{pitches}={total_pitches2}>>{total_pitches2} pitches.
Together, {name1} and {name2} received {total_pitches1} + {total_pitches2} = <<{total_pitches1}+{total_pitches2}={total_pitches}>>{total_pitches} pitches.
{name1} hit the ball {hits1} times + {name2}’s {hits2} hits = <<{hits1}+{hits2}={total_hits}>>{total_hits} total hits.
Of the {total_pitches} pitches – {total_hits} total hits = <<{total_pitches}-{total_hits}={misses}>>{misses} misses.
#### {misses}""",
        variable_generator=lambda setting: {
            "name1": get_random_name(setting),
            "name2": get_random_name(setting),
            "pitches": random.randint(10, 20),
            "tokens1": random.randint(10, 20),
            "tokens2": random.randint(10, 20),
            "hits1": random.randint(10, 20),
            "hits2": random.randint(10, 20),
        },
        answer_generator=lambda vars: {
            "total_pitches1": vars["tokens1"] * vars["pitches"],
            "total_pitches2": vars["tokens2"] * vars["pitches"],
            "total_pitches": vars["tokens1"] * vars["pitches"]
            + vars["tokens2"] * vars["pitches"],
            "total_hits": vars["hits1"] + vars["hits2"],
            "misses": (
                vars["tokens1"] * vars["pitches"] + vars["tokens2"] * vars["pitches"]
            )
            - (vars["hits1"] + vars["hits2"]),
        },
    ),
    TaskTemplate(
        type_name="Waterslide",
        question_template="""Question: The biggest {waterslide} at Five Flags is {big_slide} feet long, and people slide down at {big_speed} feet/minute. The second biggest {waterslide} is {small_slide} feet long, but steeper, so people slide down at {small_speed} feet/minute. How much longer does it take to ride the biggest {waterslide} compared to the second biggest {waterslide}?""",
        deduction_template="""Answer:
First find the ride length of the biggest {waterslide}: {big_slide} feet / {big_speed} feet/minute = <<{big_slide}/{big_speed}={big_time}>>{big_time} minutes
Then find the ride length of the second biggest {waterslide}: {small_slide} feet / {small_speed} feet/minute = <<{small_slide}/{small_speed}={small_time}>>{small_time} minutes
Then subtract the ride length of the second longest slide from the longest slide to find the difference: {big_time} minutes - {small_time} minutes = <<{big_time}-{small_time}={difference}>>{difference} minutes
#### {difference}""",
        variable_generator=lambda setting: {
            "waterslide": get_random_item(setting),
            "big_slide": random.choice([480, 600]),
            "small_slide": random.choice([120, 180]),
            "big_speed": random.choice([20, 30]),
            "small_speed": random.choice([40, 60]),
        },
        answer_generator=lambda vars: {
            "big_time": vars["big_slide"] // vars["big_speed"],
            "small_time": vars["small_slide"] // vars["small_speed"],
            "difference": vars["big_slide"] // vars["big_speed"]
            - vars["small_slide"] // vars["small_speed"],
        },
    ),
    TaskTemplate(
        type_name="Water left",
        question_template="""Question: Two girls each got 1/{part} of the {amount} liters of {water}. Then a boy got {boy_amount} liters of {water}. How many liters of {water} were left?""",
        deduction_template="""Answer:
Each of the girls got {amount} x 1/{part} = <<{amount}*{part}={girl_amount}>>{girl_amount} liters of water.
So the two girls got a total of {girl_amount} x 2 = <<{girl_amount}*2={total_girl_amount}>>{total_girl_amount} liters.
Thus, a total of {total_girl_amount} + {boy_amount} = <<{total_girl_amount}+{boy_amount}={total_amount}>>{total_amount} liters of water were gotten by the two girls and the boy.
Therefore, {amount} - {total_amount} = <<{amount}-{total_amount}={left}>>{left} liters of water were left.
#### {left}""",
        variable_generator=lambda setting: {
            "part": random.choice([3, 4]),
            "amount": random.choice([12, 24, 60]),
            "water": get_random_item(setting),
            "boy_amount": random.randint(2, 5),
        },
        answer_generator=lambda vars: {
            "girl_amount": vars["amount"] // vars["part"],
            "total_girl_amount": (vars["amount"] // vars["part"]) * 2,
            "total_amount": ((vars["amount"] // vars["part"]) * 2) + vars["boy_amount"],
            "left": vars["amount"]
            - (((vars["amount"] // vars["part"]) * 2) + vars["boy_amount"]),
        },
    ),
    TaskTemplate(
        type_name="Butcher Sales",
        question_template="""Question: {name} is a butcher. {pronoun} sells {rate}kg of meat every hour {pronoun} works, and {pronoun} works {hours} hours a day. {name2} gives {pronoun} a {animal} that weighs {weight}kg. How many days will it take {name} to sell the meat?""",
        deduction_template="""Answer:
In a day {name} sells {rate}kg of meat every hour {pronoun} works, and {pronoun} works {hours} hours a day.
So in a day {name} sells {rate} * {hours} = <<{rate}*{hours}={daily}>>{daily}kg of meat.
It will take {name} {weight} / {daily} = <<{weight}/{daily}={days}>>{days} days to sell all the meat from the {animal}.
#### {days}""",
        variable_generator=lambda setting: {
            "name": get_random_name(setting),
            "name2": get_random_name(setting),
            "rate": random.choice([5, 10]),
            "hours": random.choice([6, 9]),
            "animal": get_random_item(setting),
            "weight": random.choice([360, 540, 900]),
            "pronoun": "they",
        },
        answer_generator=lambda vars: {
            "daily": vars["rate"] * vars["hours"],
            "days": vars["weight"] // (vars["rate"] * vars["hours"]),
        },
    ),
    TaskTemplate(
        type_name="Pencil Pairs",
        question_template="""Question: There is space for {total} {items} in the box. If there are {missing} {items} missing from the box, how many pairs of {items} are in the box?""",
        deduction_template="""Answer:
In the box there are {total} {items} - {missing} {items} = <<{total}-{missing}={actual}>>{actual} {items}.
Dividing into pairs we have {actual} {items} / 2 {items}/pair = {pairs} pairs of {items}
#### {pairs}""",
        variable_generator=lambda setting="original": {
            "total": random.choice([10, 12, 14, 16, 18, 20]),
            "items": get_random_item(setting),
            "missing": random.choice([2, 4, 6]),
        },
        answer_generator=lambda vars: {
            "actual": vars["total"] - vars["missing"],
            "pairs": (vars["total"] - vars["missing"]) // 2,
        },
    ),
    TaskTemplate(
        type_name="Total cards",
        question_template="""Question: A boy has {total} {items}. His brother has {diff} fewer {items} than he has. How many {items} do they have together?""",
        deduction_template="""Answer:
His brother has {total} - {diff} = <<{total}-{diff}={brother_total}>>{brother_total} {items}.
Together, they have {total} + {brother_total} = <<{total}+{brother_total}={total_together}>>{total_together} {items}.
#### {total_together}""",
        variable_generator=lambda setting: {
            "total": random.randint(15, 25),
            "items": get_random_item(setting),
            "diff": random.randint(2, 5),
        },
        answer_generator=lambda vars: {
            "brother_total": vars["total"] - vars["diff"],
            "total_together": vars["total"] + vars["total"] - vars["diff"],
        },
    ),
    TaskTemplate(
        type_name="Boat Rentals",
        question_template="""Question: {name1} and {name2} are at the beach. {name1} rents a canoe for ${price1} an hour and {name2} rents a banana boat raft for ${price2} an hour. If {name1} uses the boat for {hours1} hours and {name2} uses the raft for {hours2} hours, how much will they pay for their rentals, altogether?""",
        deduction_template="""Answer:
{name1} would have to pay ${price1} x {hours1} = $<<{price1}*{hours1}={total1}>>{total1}
{name2} would have to pay ${price2} x {hours2} = $<<{price2}*{hours2}={total2}>>{total2}
All together, {name1} and {name2} will have to pay ${total1} + ${total2} = $<<{total1}+{total2}={total_together}>>{total_together}
#### {total_together}""",
        variable_generator=lambda setting: {
            "name1": get_random_name(setting),
            "name2": get_random_name(setting),
            "price1": random.randint(10, 20),
            "price2": random.randint(10, 20),
            "hours1": random.randint(1, 3),
            "hours2": random.randint(1, 3),
        },
        answer_generator=lambda vars: {
            "total1": vars["price1"] * vars["hours1"],
            "total2": vars["price2"] * vars["hours2"],
            "total_together": vars["price1"] * vars["hours1"]
            + vars["price2"] * vars["hours2"],
        },
    ),
    TaskTemplate(
        type_name="Playlist Hours",
        question_template="""Question: The number of {songs} in a playlist is {total}. If John has {num} such playlists, and each {songs} is {hours} hours long, how many hours will the {num} playlists last in total?""",
        deduction_template="""Answer:
Since each playlist has {total} {songs}, the total number of {songs} in the {num} playlists is {num}*{total}= <<{num}*{total}={total_songs}>>{total_songs}.
If each {songs} is {hours} hours long, the {total_songs} {songs} in the {num} playlists last a total of {total_songs}*{hours} = <<{total_songs}*{hours}={total_hours}>>{total_hours} hours
#### {total_hours}""",
        variable_generator=lambda setting: {
            "songs": get_random_item(setting),
            "total": random.randint(10, 20),
            "num": random.randint(2, 4),
            "hours": random.randint(2, 4),
        },
        answer_generator=lambda vars: {
            "total_songs": vars["num"] * vars["total"],
            "total_hours": vars["num"] * vars["total"] * vars["hours"],
        },
    ),
    TaskTemplate(
        type_name="Doughnuts",
        question_template="""Question: A {box} holds {num_dozen} dozen {items}. If the family ate {num} {items}, how many {items} are left?""",
        deduction_template="""Answer: {num_dozen} dozen {items} are equal to {num_dozen} x 12 = <<{num_dozen}*12={total}>>{total} {items}.
Since {num} {items} were eaten, therefore {total} - {num} = <<{total}-{num}={left}>>{left} {items} are left.
#### {left}""",
        variable_generator=lambda setting: {
            "box": get_random_item(setting),
            "items": get_random_item(setting),
            "num_dozen": random.randint(2, 5),
            "num": random.randint(2, 5),
        },
        answer_generator=lambda vars: {
            "total": vars["num_dozen"] * 12,
            "left": vars["num_dozen"] * 12 - vars["num"],
        },
    ),
    TaskTemplate(
        type_name="",
        question_template="""Question: {name} wants to buy a Samsung TV worth ${price}. She works for a delivery service company for a month earning ${hourly} per hour for a {hours}-hour workweek. How many more hours does she have to work to buy the TV?""",
        deduction_template="""Answer: In a week, {name} earns {hours}*{hourly} = $<<{hours}*{hourly}={weekly}>>{weekly}.
In a month, she earns {weekly}*4 = $<<{weekly}*4={monthly}>>{monthly}.
If the TV was worth ${price}, the total amount she has to work for to buy the TV is ${price}-{monthly} = $<<{price}-{monthly}={left}>>{left}.
Since she earns ${hourly} per hour, she'll have to work for ${left}/${hourly} = <<{left}/{hourly}={hours_left}>>{hours_left} more hours to buy the TV.
#### {hours_left}""",
        variable_generator=lambda setting: {
            "name": get_random_name(setting),
            "price": random.choice([4000, 5000]),
            "hourly": random.choice([10, 20]),
            "hours": random.choice([30, 35, 40]),
        },
        answer_generator=lambda vars: {
            "weekly": vars["hours"] * vars["hourly"],
            "monthly": vars["hours"] * vars["hourly"] * 4,
            "left": vars["price"] - vars["hours"] * vars["hourly"] * 4,
            "hours_left": (vars["price"] - vars["hours"] * vars["hourly"] * 4)
            // vars["hourly"],
        },
    ),
    TaskTemplate(
        type_name="",
        question_template="""Question: {name} wants to buy a {item1} that costs ${price1}, a {item2} that costs ${price2}, and a {item3} that costs ${price3}. She has saved ${saved} from her allowance, and her mother gave her ${mother} more. How much more money does {name} need to buy the {item1}, the {item2}, and the {item3}?""",
        deduction_template="""Answer:
The total cost of the {item1}, the {item2}, and the {item3} is ${price1} + ${price2} + ${price3} = $<<{price1}+{price2}+{price3}={total}>>{total}.
The total amount of money from the allowance and her mother is ${saved} + ${mother} = $<<{saved}+{mother}={total_saved}>>{total_saved}.
{name} needs ${total} − ${total_saved} = $<<{total}-{total_saved}={left}>>{left}.
#### {left}""",
        variable_generator=lambda setting: {
            "name": get_random_name(setting),
            "item1": get_random_item(setting),
            "item2": get_random_item(setting),
            "item3": get_random_item(setting),
            "price1": random.randint(10, 20),
            "price2": random.randint(10, 20),
            "price3": random.randint(10, 20),
            "saved": random.randint(10, 20),
            "mother": random.randint(10, 20),
        },
        answer_generator=lambda vars: {
            "total": vars["price1"] + vars["price2"] + vars["price3"],
            "total_saved": vars["saved"] + vars["mother"],
            "left": vars["price1"]
            + vars["price2"]
            + vars["price3"]
            - vars["saved"]
            - vars["mother"],
        },
    ),
]


def generate_task_with_context(setting, num_shots):
    """Generate a task with balanced in-context examples.

    Args:
        setting (str): Either "original" or "symbol" for name/item style
        num_shots (int): Number of example questions per template type

    Returns:
        dict: Contains the full prompt with balanced examples and the target question
    """
    indices = list(range(len(task_templates)))
    random.shuffle(indices)
    example_indices = indices[: num_shots + 1]

    full_prompt = ""
    for index in example_indices[:-1]:
        example = task_templates[index].generate(setting, show_deduction=True)
        full_prompt += f"{example['question']}\n\n"

    target_template = task_templates[example_indices[-1]]
    target_example = target_template.generate(setting, show_deduction=False)
    # print(target_example)
    full_prompt += target_example["question"]
    answer = target_example["answer"]

    return {"prompt": full_prompt, "answer": answer, "type": target_template.type_name}
