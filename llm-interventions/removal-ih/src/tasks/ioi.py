from typing import Union, List, Optional
import warnings
import torch as t
import numpy as np
from transformers import AutoTokenizer
import random
import copy
import re
np.random.seed(42)  # Example of setting the seed to 42

NAMES = [
    "Aaron",
    "Adam",
    "Alan",
    "Alex",
    "Alice",
    "Amy",
    "Anderson",
    "Andre",
    "Andrew",
    "Andy",
    "Anna",
    "Anthony",
    "Arthur",
    "Austin",
    "Blake",
    "Brandon",
    "Brian",
    "Carter",
    "Charles",
    "Charlie",
    "Christian",
    "Christopher",
    "Clark",
    "Cole",
    "Collins",
    "Connor",
    "Crew",
    "Crystal",
    "Daniel",
    "David",
    "Dean",
    "Edward",
    "Elizabeth",
    "Emily",
    "Eric",
    "Eva",
    "Ford",
    "Frank",
    "George",
    "Georgia",
    "Graham",
    "Grant",
    "Henry",
    "Ian",
    "Jack",
    "Jacob",
    "Jake",
    "James",
    "Jamie",
    "Jane",
    "Jason",
    "Jay",
    "Jennifer",
    "Jeremy",
    "Jessica",
    "John",
    "Jonathan",
    "Jordan",
    "Joseph",
    "Joshua",
    "Justin",
    "Kate",
    "Kelly",
    "Kevin",
    "Kyle",
    "Laura",
    "Leon",
    "Lewis",
    "Lisa",
    "Louis",
    "Luke",
    "Madison",
    "Marco",
    "Marcus",
    "Maria",
    "Mark",
    "Martin",
    "Mary",
    "Matthew",
    "Max",
    "Michael",
    "Michelle",
    "Morgan",
    "Patrick",
    "Paul",
    "Peter",
    "Prince",
    "Rachel",
    "Richard",
    "River",
    "Robert",
    "Roman",
    "Rose",
    "Ruby",
    "Russell",
    "Ryan",
    "Sarah",
    "Scott",
    "Sean",
    "Simon",
    "Stephen",
    "Steven",
    "Sullivan",
    "Taylor",
    "Thomas",
    "Tyler",
    "Victoria",
    "Warren",
    "William",
]


BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]


PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]

OBJECTS = [
    "ring",
    # "kiss",
    # "bone",
    "basketball",
    # "computer",
    "necklace",
    "drink",
    "snack",

    "money",
    "backpack",
    "bottle",
    "book",

]


SYMBOLS = [
    "#$",
    "&^",
    "!@",
    ")%",
    "-+",
    "=/",
    "[&",
]


def generator(**args):
    while True:
        if "symbol" in args and args["symbol"]:
            name_A, name_B = np.random.choice(SYMBOLS, size=2, replace=False)
        else:
            name_A, name_B = np.random.choice(NAMES, size=2, replace=False)

        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)

        prompt = random.choice(BABA_TEMPLATES)
        prompt = prompt.replace("[A]", name_A)
        prompt = prompt.replace("[B]", name_B)
        prompt = prompt.replace("[PLACE]", place)
        prompt = prompt.replace("[OBJECT]", obj)

        prompt, answer = prompt[: -len(name_A) - 1], f"{name_A}"

        if "multiple_choice" in args and args["multiple_choice"]:
            choices = [f"{name_A}", f"{name_B}"]
            yield prompt, answer, choices
            continue

        yield prompt, answer
