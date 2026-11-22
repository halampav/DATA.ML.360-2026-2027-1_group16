"""Hyperparameter tuning of the Part 2 models on the validation split.

- `MODELS`: every tuned model with its config.yaml key and its Optuna search space.
- `tune`: Optuna search on validation; every model gets the same number of trials.
- `trials_table`: all trials of a study as a table.

Used by notebooks/03_part2.ipynb.
"""
import time
from dataclasses import dataclass
from typing import Callable

import optuna
import pandas as pd

from src.evaluation import evaluate_recommender
from src.model import Dataset, ImplicitALS, ImplicitBPR, UserKNN
from src.model.two_tower import TwoTower


# ---------------------------------------------------------------------------------------------
# Search spaces (validation only). None in a list means "off".
# ---------------------------------------------------------------------------------------------

def user_knn_space(trial: optuna.Trial) -> dict:
    # The same space as ItemKNN in Part 1 (k, similarity, gamma), plus min_support
    return {"k": trial.suggest_int("k", 5, 200, log=True),
            "similarity_measure": trial.suggest_categorical("similarity_measure",
                                                            ["cosine", "pearson", "jaccard"]),
            "gamma": trial.suggest_categorical("gamma", [None, 10, 25, 50, 100]),
            "min_support": trial.suggest_categorical("min_support", [1, 5, 10, 20, 50, 100, 200])}


def als_space(trial: optuna.Trial) -> dict:
    return {"factors": trial.suggest_int("factors", 16, 256, log=True),
            "regularization": trial.suggest_float("regularization", 1e-4, 10, log=True),
            "alpha": trial.suggest_float("alpha", 0.1, 100, log=True),   # confidence = 1 + alpha
            "iterations": trial.suggest_int("iterations", 5, 50)}


def bpr_space(trial: optuna.Trial) -> dict:
    return {"factors": trial.suggest_int("factors", 16, 256, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "regularization": trial.suggest_float("regularization", 1e-5, 0.1, log=True),
            "iterations": trial.suggest_int("iterations", 20, 500, log=True)}


def two_tower_space(trial: optuna.Trial) -> dict:
    return {"dim": trial.suggest_categorical("dim", [32, 64, 128]),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [16, 32, 64]),
            "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
            "epochs": trial.suggest_int("epochs", 2, 20),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
            "temperature": trial.suggest_float("temperature", 0.02, 0.5, log=True),
            "id_dropout": trial.suggest_float("id_dropout", 0.0, 0.5)}


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    cls: type
    config_key: str           # key under `models` in config.yaml
    search_space: Callable[[optuna.Trial], dict]
    uses_seed: bool           # random training -> the seed is passed to the model

    def build(self, params: dict, seed: int | None):
        params = dict(params)
        if self.uses_seed:
            params["seed"] = seed
        return self.cls(**params)


MODELS = {
    "UserKNN": ModelSpec(UserKNN, "user_knn", user_knn_space, uses_seed=False),
    "ImplicitALS": ModelSpec(ImplicitALS, "implicit_als", als_space, uses_seed=True),
    "ImplicitBPR": ModelSpec(ImplicitBPR, "implicit_bpr", bpr_space, uses_seed=True),
    "TwoTower": ModelSpec(TwoTower, "two_tower", two_tower_space, uses_seed=True),
}


def final_params(name: str, config: dict) -> dict:
    """Hyperparameters of `name` from config.yaml (without the seed, which `build` sets)."""
    params = config["models"][MODELS[name].config_key]
    return {key: value for key, value in params.items() if key != "seed"}


def describe(params: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


# ---------------------------------------------------------------------------------------------
# Tuning on validation
# ---------------------------------------------------------------------------------------------

def tune(name: str, train: Dataset, val_ratings: pd.DataFrame, catalogue: set, k: int,
         threshold: int, n_trials: int, seed: int, metric: str = "ndcg_warm") -> optuna.Study:
    """Optuna (TPE, fixed seed) search over `MODELS[name].search_space`, maximising `metric` on
    validation. Random models are trained with `seed` in every trial, so trials differ only in
    their hyperparameters. All validation metrics are stored on the trials as user_attrs.
    """
    spec = MODELS[name]

    def objective(trial):
        params = spec.search_space(trial)
        start = time.time()
        model = spec.build(params, seed)
        model.train(train)
        metrics = evaluate_recommender(model, train.ratings, val_ratings, catalogue, k, threshold)
        metrics["seconds"] = round(time.time() - start, 1)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        print(f"{name} trial {trial.number:3d}: {metric}={metrics[metric]:.4f} "
              f"ndcg={metrics['ndcg']:.4f} coverage={metrics['coverage']:.3f} "
              f"({metrics['seconds']} s) {describe(params)}", flush=True)
        return metrics[metric]

    study = optuna.create_study(direction="maximize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """One row per finished trial: hyperparameters and validation metrics, best first."""
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df = df[df["state"] == "COMPLETE"].drop(columns="state")
    df.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in df.columns]
    df.insert(0, "model", study.study_name)
    return df.sort_values("value", ascending=False).reset_index(drop=True)
