# Movie Recommender System

The repository has the following structure:

```
movie-recommender-system
├── README.md               # The top-level README
│
├── data
│   ├── external            # Data from third party sources
│   ├── interim             # Intermediate data that has been transformed.
│   └── raw                 # The original, immutable data
│
├── models                  # Trained and serialized models, final checkpoints
│
├── notebooks               #  Jupyter notebooks. Naming convention is a number (for ordering),
│                               and a short delimited description, e.g.
│                               "1.0-initial-data-exporation.ipynb"            
│ 
├── references              # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports
│   ├── figures             # Generated graphics and figures to be used in reporting
│   └── final_report.pdf    # Report containing data exploration, solution exploration, training process, and evaluation
│
└── benchmark
│    ├── data                # dataset used for evaluation 
│    └── evaluate.py         # script that performs evaluation of the given model
│
└── scripts                 # Source code for use in this      assignment
```

## Installation
Before using scripts, you need to install **requirements.txt** with pip:
``` bash
    pip install -r requirements.txt
```

## Getting data
To fetch the dataset, launch `scripts/get_data.py`:
``` bash
    python scripts/get_data.py
```

## Evaluation
To run evaluation process, run `benchmark/evaluate.py`:
``` bash
    python benchmark/evaluate.py
```