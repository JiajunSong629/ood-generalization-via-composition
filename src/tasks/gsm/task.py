import os
import json
import re
import copy
import random
from typing import List, Optional, Union
from tqdm import tqdm

import src.api.task as task
import src.api.model as model
import src.api.result as result_api

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def doc_to_text(doc):
    return "Question: {}Answer:".format(doc["question"])


def doc_to_target(doc):
    return " " + doc["answer"]


class GSMTask(task.Task):
    def __init__(
        self,
        num_shots: Optional[int] = 3,
        num_beams: Optional[int] = 1,
        num_outputs: Optional[int] = 1,
        max_new_tokens: Optional[int] = 100,
    ):
        super().__init__()
        self._num_shots = num_shots
        self._num_beams = num_beams
        self._num_outputs = num_outputs
        self._max_new_tokens = max_new_tokens

    def get_task_details(self) -> task.TaskMetadata:
        return {
            "name": "gsm",
            "description": "gsm",
            "num_shots": self._num_shots,
            "num_outputs": self._num_outputs,
            "max_new_tokens": self._max_new_tokens,
        }

    def _extract_answer(self, completion: str) -> str:
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS

    def _generate_examples(self, num_samples: int):
        path = os.path.join(os.path.dirname(__file__), "data")
        with open(os.path.join(path, "train.jsonl")) as fh:
            train_examples = [json.loads(line) for line in fh.readlines() if line]
        with open(os.path.join(path, "test.jsonl")) as fh:
            test_examples = [json.loads(line) for line in fh.readlines() if line]

        for examples in [train_examples, test_examples]:
            for ex in examples:
                ex.update(question=ex["question"] + "\n")
                ex.update(answer=ex["answer"] + "<|endoftext|>")

        prompts, answers = [], []
        for _ in tqdm(range(num_samples)):
            test_doc = random.choices(test_examples, k=1)[0]
            fewshotex = random.sample(train_examples, self._num_shots)

            labeled_examples = (
                "\n\n".join(
                    [doc_to_text(doc) + doc_to_target(doc) for doc in fewshotex]
                )
                + "\n\n"
            )

            example = doc_to_text(test_doc)
            prompt = labeled_examples + example

            prompts.append(prompt)
            answers.append(test_doc["answer"])

        return prompts, answers

    def evaluate_model(
        self,
        model: model.Model,
        num_samples: Optional[int] = None,
        task_random_seed: Optional[int] = 0,
        batch_size: Optional[int] = 8,
    ) -> Union[task.ScoreData, List[task.ScoreData]]:
        if task_random_seed is not None:
            random.seed(task_random_seed)

        prompts, answers = self._generate_examples(num_samples)

        acc = 0
        examples = []

        for prompt, answer in tqdm(zip(prompts, answers)):
            response = model.generate_text(
                prompt,
                max_new_tokens=self._max_new_tokens,
                num_beams=self._num_beams,
                num_outputs=self._num_outputs,
                batch_size=batch_size,
            )
            answer = self._extract_answer(answer)
            if isinstance(response, list):
                model_answer = [self._extract_answer(r) for r in response]
                cur_result = answer in model_answer
            else:
                model_answer = self._extract_answer(response)
                cur_result = model_answer == answer

            examples.append(
                result_api.GSMGenerationExampleResult(
                    prompt=prompt,
                    expected_answer=answer,
                    model_solution=response,
                    model_answer=model_answer,
                    correct=cur_result,
                )
            )
            acc += cur_result

        result = result_api.GSMResult(
            model_details=copy.deepcopy(model.model_meta),
            task_details=self.get_task_details(),
            accuracy=acc / num_samples,
            num_samples=num_samples,
            random_seed=task_random_seed,
            examples=examples,
        )

        return result


if __name__ == "__main__":
    from src.models.huggingface_models import HFModel

    model = HFModel(model_name="llama2-7b", device="cuda", quantize=True)
    task = GSMTask(max_new_tokens=128, num_shots=10)
    results = task.evaluate_model(model, num_samples=3, batch_size=1)
    results.save("results.json")
