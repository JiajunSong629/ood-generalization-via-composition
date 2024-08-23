import random

num_english_pair = [
    [1, "one"],
    [2, "two"],
    [3, "three"],
    [4, "four"],
    [5, "five"],
    [6, "six"],
    [7, "seven"],
    [8, "eight"],
    [9, "nine"],
]


def generator(n_examples: int = 2, template: callable = None):
    while True:
        prompt = ""
        for n, e in random.sample(
            num_english_pair,
            k=n_examples,
        ):
            prompt += template(n, e)  # f"{n}:{e}, "

        n, answer = random.choice(num_english_pair)
        prompt += template(n)  # f"{n}:"

        yield prompt, answer
