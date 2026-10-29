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

| Argument | Description | Default |
|---|---|---|
| `--data-dir` | Folder with the MovieLens files. | None |
| `--config` | Path to the YAML config file. | `config.yaml` |

## Configuration (`config.yaml`)

All names and parameters are in `config.yaml` so that nothing needs to be changed in the code.
