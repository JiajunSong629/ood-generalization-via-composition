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
class ICLResult:
    task_details: Dict[str, Any]
    model_details: Dict[str, Any]
    accuracy: float
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @staticmethod
    def save_multiple(results: List["ICLResult"], path: str):
        results_dict = [dataclasses.asdict(result) for result in results]
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)


@dataclasses.dataclass
class GSMResult:
    task_details: Dict[str, Any]
    model_details: Dict[str, Any]
    accuracy: float
    num_samples: int
    random_seed: int
    examples: List[Dict[str, Any]]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @staticmethod
    def save_multiple(results: List["GSMResult"], path: str):
        results_dict = [dataclasses.asdict(result) for result in results]
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)
