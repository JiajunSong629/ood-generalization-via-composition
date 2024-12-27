# OOD Generalization via Composition

This repository contains code necessary to reproduce the experiments and figures in the paper **"Out-of-Distribution Generalization via Composition: a lens through induction heads in Transformers".**

## Usage

```
conda create -n ood python=3.10
conda activate ood
pip install -r requirements.txt
pip install -e .
```

To reproduce the experiments, specify the models, tasks, and scripts in `src/scripts/run.sh` and run the following command:
```
bash src/scripts/run.sh
```

The results will be saved by default under `src/{task}/results/` for each task. You may run `report.py` inside each task directory to generate the figures and summary statistics for the results.


## Structure of the import files

```
├── src/                           # Main source code
│   ├── api/                       # Core APIs and abstractions
│   ├── scripts/                   # Scripts for running experiments
│   ├── models/                    # Model and intervention implementations
│   │   ├── huggingface_models.py  # HuggingFace model wrappers
│   │   └── removal_models.py      # Models with head removal capability
│   │   └── proj_models.py         # Models with head projection capability
│   │   └── shuffle_models.py      # Models with head shuffling capability
│   ├── tasks/                     
│   │   ├── copying/               # Copying task
│   │   ├── icl/                   # In-context learning task
│   │   ├── ioi/                   # Indirect object identification
│   │   └── fuzzycopy/             # Fuzzy copying task
|   |
│   ├── config.py                  # Configuration file
|   |
│   ├── evaluate_shuffle.py        # Evaluate shuffled models
│   ├── evaluate_projection.py     # Evaluate projection models
│   ├── evaluate_scaling.py        # Evaluate scaling models
│   └── evaluate_removal.py        # Evaluate removal models
│
├── synthetic-experiments/        # Synthetic experiments
│   ├── main.py                   # Main entry point
│   ├── config.yaml               # Configuration file
│   └── run.sh                    # Example run scripts
|
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT license
```


## Citing this work

```
@article{song2024out,
  title={Out-of-distribution generalization via composition: a lens through induction heads in transformers},
  author={Song, Jiajun and Xu, Zhuoyan and Zhong, Yiqiao},
  journal={arXiv preprint arXiv:2408.09503},
  year={2024}
}
```

## License

This repository is licensed under the MIT license. See the LICENSE file for more details.
