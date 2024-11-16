import os
import json

import src.api.result as result_api


def load_results(model_name, component):
    cur_dir = os.path.dirname(os.path.abspath("__file__"))
    result_dir = os.path.join(cur_dir, "results")

    fname = f"{model_name}_symbol_20_shuffle_{component}.json"
    result_path = os.path.join(result_dir, fname)

    with open(result_path, "r") as f:
        results = json.load(f)

    results = [result_api.ICLResult(**r) for r in results]

    return results


def pretty_print_result(result: result_api.CopyingResult):
    if "shuffle_meta" in result.model_details:
        shuffle_meta = result.model_details["shuffle_meta"]
        print(
            "Shuffle",
            shuffle_meta["shuffled_type"],
            shuffle_meta["shuffled_component"],
            shuffle_meta["seed"],
        )
    else:
        print("Original")

    print(f"{result.prob:.2f}, {1 - result.err:.2f}")


def summarise_results(results):
    d = {}

    for r in results:
        if "shuffle_meta" in r.model_details:
            shuffle_meta = r.model_details["shuffle_meta"]
            shuffle_type = shuffle_meta["shuffled_type"]
            if shuffle_type not in d:
                d[shuffle_type] = {"accuracy": []}
            d[shuffle_type]["accuracy"].append(r.accuracy)

        else:
            d["original"] = {"accuracy": r.accuracy}

    return d


if __name__ == "__main__":
    from src.config import MODEL_CLASSES

    for model_name in MODEL_CLASSES:
        for component in ["qk", "ov"]:
            try:
                print(model_name, component)
                results = load_results(model_name, component)
                d = summarise_results(results)
                print(d)
            except FileNotFoundError:
                print(f"No results for {model_name}")
