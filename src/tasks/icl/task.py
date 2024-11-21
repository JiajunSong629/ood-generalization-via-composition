from typing import List, Optional, Union, Dict, Any
import torch.nn.functional as F
import numpy as np
import copy
import random
import dataclasses

import src.api.task as task_api
import src.api.result as result_api
import src.api.model as model_api


@dataclasses.dataclass
class GenerationExampleResult:
    prompt: str
    expected_answer: str
    model_solution: str
    correct: bool
    prob: float
    multiple_choice_logprob: Dict[str, float]


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


REMAP_SYMBOL = {"animal": "$#", "sport": "!%", "plant/vegetable": "&*"}
REMAP_PERMUTE = {"animal": "animal", "sport": "sport", "plant/vegetable": "plant"}
REMAP_ORIG = {"animal": "sport", "sport": "plant", "plant/vegetable": "animal"}


def template(item: str, label: str = None) -> str:
    if label is None:
        return f"{item} is"
    return f"{item} is {label}, "


class ICLTask(task_api.Task):
    def __init__(self, setting: str, num_shots: int, balanced_sample):
        super().__init__()
        self._setting = setting
        self._num_shots = num_shots
        self._balanced_sample = balanced_sample

    def get_task_details(self) -> Dict[str, Any]:
        return {
            "name": "icl",
            "description": "icl",
            "setting": self._setting,
            "num_shots": self._num_shots,
            "balanced_sample": self._balanced_sample,
        }

    @property
    def choices(self):
        if self._setting == "permute":
            return ["animal", "sport", "plant/vegetable"]
        elif self._setting == "original":
            return ["sport", "plant", "animal"]
        elif self._setting == "symbol":
            return [" $#,", " !%,", " &*,"]
        else:
            raise ValueError(f"Invalid setting: {self._setting}")

    def get_prompts_and_answers(
        self, num_samples: int, balanced_sample: bool = True, seed: int = None
    ):
        if seed is not None:
            random.seed(seed)

        # Organize items by category, for banlanced sampling
        categories = {"animal": [], "sport": [], "plant/vegetable": []}
        for item, label in MAPPING.items():
            categories[label].append(item)

        if self._setting == "permute":
            remap = REMAP_PERMUTE
        elif self._setting == "original":
            remap = REMAP_ORIG
        elif self._setting == "symbol":
            remap = REMAP_SYMBOL
        else:
            raise ValueError(f"Invalid setting: {self._setting}")

        item_label_pair = [(i, l) for i, l in MAPPING.items()]
        prompts, answers = [], []

        for _ in range(num_samples):
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
                remaining_items = [
                    (item, label)
                    for item, label in MAPPING.items()
                    if item not in selected
                ]
                for item, label in random.choices(
                    remaining_items, k=self._num_shots - len(categories)
                ):
                    label = remap[label]
                    prompt += template(item, label)
                    selected.add(item)

            else:
                for i, l in random.choices(item_label_pair, k=self._num_shots):
                    label = remap[l]
                    prompt += template(i, label)
                    selected.add(i)

            item, answer = random.choice(
                [(i, l) for i, l in item_label_pair if i not in selected]
            )
            prompt += template(item)

            test_answer = remap[answer]

            prompts.append(prompt)
            answers.append(test_answer)

        return prompts, answers

    def evaluate_model(
        self,
        model: model_api.Model,
        num_samples: int,
        task_random_seed: Optional[int] = None,
    ) -> float:
        prompts, answers = self.get_prompts_and_answers(
            num_samples,
            self._balanced_sample,
            task_random_seed,
        )

        if model.model_meta["model_name"].startswith("gpt"):
            batch_size = 100
        else:
            batch_size = 1

        solutions = model.generate_text(
            prompts, max_new_tokens=3, batch_size=batch_size
        )

        acc = [answer in solution for answer, solution in zip(answers, solutions)]
        logprob = model.cond_log_prob(
            prompts, [self.choices for _ in range(num_samples)]
        )

        eval_results = []
        for prompt, answer, solution, is_correct, pred_logprob in zip(
            prompts, answers, solutions, acc, logprob
        ):
            logprob_on_correct = pred_logprob[self.choices.index(f" {answer},")]

            eval_results.append(
                GenerationExampleResult(
                    prompt=prompt,
                    expected_answer=answer,
                    model_solution=solution,
                    correct=bool(is_correct),
                    prob=np.exp(logprob_on_correct),
                    multiple_choice_logprob={
                        c: lp for c, lp in zip(self.choices, pred_logprob)
                    },
                )
            )

        # Save results to JSON
        result = result_api.ICLResult(
            task_details=self.get_task_details(),
            model_details=copy.deepcopy(model.model_meta),
            acc=float(np.mean(acc)),
            prob=float(np.mean([r.prob for r in eval_results])),
            num_samples=num_samples,
            random_seed=task_random_seed,
            examples=eval_results,
        )

        return result
