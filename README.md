# DATA.ML.360-2026-2027-1 Group 16

Recommender Systems course project on MovieLens 1M.

## Team

- **Jindřich Halabala** (154807984)
- **Pavel Halama** (154808080)
- **Amala Vijayan Pillai Soudha** (153106068)

Repository: https://github.com/halampav/DATA.ML.360-2026-2027-1_group16
Excel with results: https://drive.google.com/file/d/12fiw6fWJJNYX4NtnCuRQVqbtq5d1Tx6y/view?usp=sharing

## Requirements

- Python 3.11 (see `.python-version`)
- Dependencies pinned in `requirements.txt`
- Exact full environment in `requirements-lock.txt`
- PyTorch is the CPU-only build: `requirements.txt` adds the PyTorch CPU package index, so pip does not
  download the multi-GB CUDA build on Linux. No GPU is needed; all runs use one CPU thread and fixed seeds.

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

It takes about 10 minutes.

| Argument | Description | Default |
|---|---|---|
| `--data-dir` | Folder with the MovieLens files. | None |
| `--config` | Path to the YAML config file. | `config.yaml` |
| `--output-dir` | Folder for the results. | `results` |

What the command does:

1. Splits the data temporally (train / validation / test).
2. Trains the final baseline and Part 2 configurations from `config.yaml` on train + validation and
   evaluates them on the test split.
3. Computes a random and a perfect ranker on the same test users as reference bounds.
4. Checks our metric implementations (`src/evaluation.py`) against scikit-learn.

Outputs in `results/`:

| File | Content |
|---|---|
| `results.csv` | Test results; columns match the "Part 1 - Baselines" sheet of the results template. |
| `reference_bounds.csv` | Random and perfect ranker on the same test users. |
| `metric_check.csv` | Largest difference between our metrics and scikit-learn's. |

### Part 2 analysis

```bash
python -m src.part2_analysis --data-dir data/ml-1m
```

Trains the tuned Part 2 models (and the two baselines) on train + validation and evaluates them on
test. Takes about 40 minutes; `--no-seeds` and `--no-sparsity` skip the slow parts, `--models`
selects models.

1. Overall metrics with a 95% CI over test users, and the disaggregated results by user activity and
   item popularity.
2. Paired t-tests on per-user NDCG@10 for every pair of models (Holm-corrected for 15 comparisons).
3. Seed variability: ALS, BPR and TwoTower retrained with the 5 seeds in `part2.seeds`.
4. Sparsity ablation: every model trained on 50% and 20% of the train + validation ratings (3 random
   subsamples each, `part2.sparsity_seeds`) with the same hyperparameters, evaluated on the full test set.

Outputs in `results/part2_analysis/`:

| File | Content |
|---|---|
| `part2_overall.csv` | Test metrics of every model (seed 42) with the user CI. |
| `part2_subgroups.csv` | Metrics by user activity (cold / low / high) and item popularity (top 20% / rest). |
| `part2_per_user_ndcg.csv` | NDCG@10 of every model for every test user (input of the paired tests). |
| `part2_paired_tests.csv` | Paired t-test, mean difference with 95% CI and Holm-adjusted p-value per model pair. |
| `part2_seed_variability.csv` | Mean, 95% CI, min and max over seeds (stochastic models). |
| `part2_sparsity.csv` | Mean and 95% CI per model and training fraction, with the model ranking per fraction. |
| `part2_runs.csv` | Every single training run behind the two tables above. |

The same results in the exact columns of the results template (`templates/Recommender_Systems_Results_Template.xlsm`),
ready to paste into its Part 2 sheets, are written to `results/`: `results_part2_models.csv`,
`results_part2_disaggregated.csv` and `results_part2_sparsity.csv`. They can be rebuilt from the CSVs above without
training anything: `python -m src.part2_analysis --data-dir data/ml-1m --tables-only`.

### Part 3 analysis (Track A: MMR re-ranking)

```bash
python -m src.part3_analysis --data-dir data/ml-1m
```

Re-ranks the Part 2 ALS model with MMR (Maximal Marginal Relevance) over the ALS top-100, using binary genre
vectors and the item distance 1 - Jaccard similarity. Takes about 40 seconds. Settings are under `part3` in
`config.yaml`. The MMR code and the metrics are in `src/part3_mmr.py`.

1. Validation sweep over λ (`part3.lambdas`, 1 = plain ALS, 0 = only diversity); λ is chosen here
   (`part3.chosen_lambda`).
2. Test sweep with the same λ values (the accuracy-diversity frontier): P/R/NDCG@10, intra-list diversity,
   catalogue coverage, Gini, calibration error and the share of recommendations to the 20% most-rated movies.
3. Paired t-tests of every λ against plain ALS (NDCG@10, ILD, calibration error; Holm-corrected).
4. Who pays: the chosen λ against plain ALS per user group (activity, taste breadth) and per genre.
5. Candidate-pool ablation: the chosen λ with pools of 10 to 500 candidates.

Outputs in `results/part3_analysis/`:

| File | Content |
|---|---|
| `part3_frontier.csv` | All metrics for every λ, on validation and on test. |
| `part3_per_user.csv` | Per-user test metrics for every λ (input of the paired tests). |
| `part3_paired_tests.csv` | Every λ against plain ALS: mean difference with 95% CI and Holm-adjusted p-value. |
| `part3_who_pays.csv` | Chosen λ against plain ALS per user group. |
| `part3_genre_exposure.csv` | Share of recommendation slots per genre (plain ALS, chosen λ, λ = 0) and in the catalogue. |
| `part3_pool_size.csv` | Chosen λ and plain ALS with different candidate-pool sizes. |

The test frontier in the columns of the "Part 3A - Diversity" sheet of the results template is written to
`results/results_part3_diversity.csv`.

## Notebooks

The notebooks are committed with their outputs, so they can be read without running anything:

- `notebooks/01_data_analysis.ipynb`: data characterization (the data figures and numbers in the report).
- `notebooks/02_tuning.ipynb`: hyperparameter tuning on the validation split; the chosen values are in
  `config.yaml`. Re-running it takes about 3 hours.
- `notebooks/03_part2.ipynb`: Part 2 tuning. Tunes UserKNN, ImplicitALS, ImplicitBPR and TwoTower on
  validation (the same Optuna budget for every model, code in `src/tuning.py`). Re-running it takes about
  2 hours.
- `notebooks/04_part2_results.ipynb`: plots of the Part 2 test results (overall, paired tests, seeds,
  subgroups, sparsity). Only reads the CSVs from `src/part2_analysis.py`, runs in seconds; the figures are
  also saved to `results/part2_analysis/figures/`.
- `notebooks/05_part3_mmr.ipynb`: plots and discussion of the Part 3 results (frontier, paired tests, coverage /
  Gini / calibration / popularity, who pays, candidate pool). Only reads the CSVs from `src/part3_analysis.py`;
  the figures are also saved to `results/part3_analysis/figures/`.

The random seed is `seed: 42` in `config.yaml`; the tuning notebook uses the same value. Both baseline
models are deterministic.

## Configuration (`config.yaml`)

All names and parameters are in `config.yaml` so that nothing needs to be changed in the code.
