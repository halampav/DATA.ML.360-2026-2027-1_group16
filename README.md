# DATA.ML.360-2026-2027-1 Group 16

Recommender Systems course project on MovieLens 1M.

## Team

- **Jindřich Halabala** (154807984)
- **Pavel Halama** (154808080)
- **Amala Vijayan Pillai Soudha** (153106068)

Repository: https://github.com/halampav/DATA.ML.360-2026-2027-1_group16

## Requirements

- Python 3.11 (see `.python-version`)
- Dependencies pinned in `requirements.txt`
- Exact full environment in `requirements-lock.txt`

## Setup

**Clone the repository and enter the project folder:**

```bash
git clone https://github.com/halampav/DATA.ML.360-2026-2027-1_group16.git
cd DATA.ML.360-2026-2027-1_group16
```

**Create and activate a virtual environment:**

```bash
python3.11 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**Install the dependencies:**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Optional
# python -m pip install -r requirements-lock.txt
```

## Data

The dataset is **not** included in the repository because of the GroupLens usage terms.
The best option is to store it in the git-ignored `data/` folder inside the project.

1. Download **MovieLens 1M** from https://grouplens.org/datasets/movielens/1m/.
2. Unzip it. The folder must contain `ratings.dat`, `movies.dat` and `users.dat`.

## Running

```bash
python run_all.py --data-dir data/ml-1m
```

It takes about 5 minutes.

| Argument | Description | Default |
|---|---|---|
| `--data-dir` | Folder with the MovieLens files. | None |
| `--config` | Path to the YAML config file. | `config.yaml` |
| `--output-dir` | Folder for the results. | `results` |

What the command does:

1. Splits the data temporally (train / validation / test).
2. Trains the final baseline configurations from `config.yaml` on train + validation and evaluates
   them on the test split.
3. Computes a random and a perfect ranker on the same test users as reference bounds.
4. Checks our metric implementations (`src/evaluation.py`) against scikit-learn.

Outputs in `results/`:

| File | Content |
|---|---|
| `results.csv` | Test results; columns match the "Part 1 - Baselines" sheet of the results template. |
| `reference_bounds.csv` | Random and perfect ranker on the same test users. |
| `metric_check.csv` | Largest difference between our metrics and scikit-learn's. |

## Notebooks

The notebooks are committed with their outputs, so they can be read without running anything:

- `notebooks/01_data_analysis.ipynb`: data characterization (the data figures and numbers in the report).
- `notebooks/02_tuning.ipynb`: hyperparameter tuning on the validation split; the chosen values are in
  `config.yaml`. Re-running it takes about 3 hours.

The random seed is `seed: 42` in `config.yaml`; the tuning notebook uses the same value. Both baseline
models are deterministic.

## Configuration (`config.yaml`)

All names and parameters are in `config.yaml` so that nothing needs to be changed in the code.
