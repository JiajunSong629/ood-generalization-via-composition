import json
import os
import re
import random

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join(current_dir,"factory/GSM", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS




def doc_to_text(doc):
    return "Question: {}Answer:".format(doc['question'])

def doc_to_target(doc):
    return " " + doc['answer']

def generator(n_examples: int = 8, template: callable = None):
    ## load data
    test_examples = get_examples("test")
    train_examples = get_examples("train")
    while True:
        test_doc = random.choices(test_examples, k=1)[0]
        fewshotex = random.sample(train_examples, n_examples)

        labeled_examples = (
            "\n\n".join(
                [
                    # TODO: temporarily!!!
                    doc_to_text(doc) + doc_to_target(doc)
                    for doc in fewshotex
                ]
            )
            + "\n\n"
        )

        example = doc_to_text(test_doc)

        prompt = labeled_examples + example

        answer = test_doc['answer']

        yield prompt, answer


def main():
    test_examples = get_examples("test")
    print(type(test_examples))
    doc = test_examples[1]
    print(doc)
    qn = doc["question"]
    ans = doc["answer"]
    sample_len = 100
    print(qn.strip())
    print(ans.strip())

    print("=================")
    doc = random.choices(test_examples, k=1)[0]
    print(doc_to_text(doc))

    


if __name__ == "__main__":
    main()
