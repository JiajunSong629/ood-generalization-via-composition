from typing import List, Optional, Dict, Any
import dataclasses
import json
import src.api.task as task_api
import src.api.model as model_api


@dataclasses.dataclass
class CopyingResult:
    task_details: Dict[str, Any]
    model_details: Dict[str, Any]
    prob: float
    err: float
    probs: List[float]
    errs: List[float]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @staticmethod
    def save_multiple(results: List["CopyingResult"], path: str):
        results_dict = [dataclasses.asdict(result) for result in results]
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)


@dataclasses.dataclass
class FuzzyCopyingResult:
    logprob: List[float]
    task_meta: Dict[str, Any]
    model_meta: Dict[str, Any]
    inputs: List[str]
    targets: List[str]


@dataclasses.dataclass
class ICLResult:
    task_details: Dict[str, Any]
    model_details: Dict[str, Any]
    accuracy: float
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]
    balanced_sampling: bool

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @staticmethod
    def save_multiple(results: List["ICLResult"], path: str):
        results_dict = [dataclasses.asdict(result) for result in results]
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)


@dataclasses.dataclass
class GenerationExampleResult:
    prompt: str
    expected_answer: str
    model_solution: str
    correct: bool
    multiple_choice_logprob: Dict[str, float]


@dataclasses.dataclass
class GSMGenerationExampleResult:
    prompt: str
    expected_answer: str
    model_solution: str
    model_answer: str
    correct: bool


@dataclasses.dataclass
class GSMResult:
    task_details: Dict[str, Any]
    model_details: Dict[str, Any]
    accuracy: float
    num_samples: int
    random_seed: int
    examples: List[GSMGenerationExampleResult]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @staticmethod
    def save_multiple(results: List["ICLResult"], path: str):
        results_dict = [dataclasses.asdict(result) for result in results]
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)
