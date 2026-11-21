from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.stats import ttest_rel

from src.data.splitting import split
from src.evaluation import (
    catalogue_coverage,
    evaluate_model,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevant_by_user,
)
from src.model import Dataset
from src.model.simple_models import ItemKNN, MostPopular, UserKNN
from src.model.implicit_models import ImplicitALS, ImplicitBPR
from src.model.two_tower import TwoTower


# Model name -> (class, key in config.yaml under `models`)
MODELS = {
    "MostPopular": (MostPopular, "most_popular"),
    "ItemKNN": (ItemKNN, "item_knn"),
    "UserKNN": (UserKNN, "user_knn"),
    "ImplicitALS": (ImplicitALS, "implicit_als"),
    "ImplicitBPR": (ImplicitBPR, "implicit_bpr"),
    "TwoTower": (TwoTower, "two_tower"),
}
# Models with random initialisation / sampling. Only these are re-run with several seeds:
# the others give the same result every time, so repeating them would add nothing.
STOCHASTIC = {"ImplicitALS", "ImplicitBPR", "TwoTower"}
# Self-implemented models (the hard Part 2 requirement); ALS and BPR come from `implicit`
SELF_IMPLEMENTED = {"MostPopular", "ItemKNN", "UserKNN", "TwoTower"}


def model_settings(name: str, config: dict) -> dict:
    """The final hyperparameters of a model from config.yaml (first MostPopular variant)."""
    settings = config["models"][MODELS[name][1]]
    return settings[0] if name == "MostPopular" else dict(settings)


def build_model(name: str, config: dict, seed: int | None = None):
    """A fresh model with the tuned hyperparameters; `seed` replaces the config seed."""
    settings = model_settings(name, config)
    if seed is not None and name in STOCHASTIC:
        settings["seed"] = seed
    return MODELS[name][0](**settings)


def per_user_metrics(recommendations: dict, relevant: dict, k: int) -> pd.DataFrame:
    """P@k, R@k and NDCG@k of every evaluated user, one row per user (sorted by user id).

    The per-user scores are what the paired tests and the subgroup tables are built from.
    """
    rows = []
    for u in sorted(relevant):
        ranked = recommendations[u]
        if len(ranked[:k]) != len(set(ranked[:k])):
            raise ValueError(f"Duplicate recommendations for user {u}")
        rows.append({
            "user_id": u,
            "precision": precision_at_k(ranked, relevant[u], k),
            "recall": recall_at_k(ranked, relevant[u], k),
            "ndcg": ndcg_at_k(ranked, relevant[u], k),
        })
    return pd.DataFrame(rows)


def user_activity_groups(history: pd.DataFrame, users: list[int]) -> tuple[pd.Series, int]:
    """Splits `users` into Cold (no history ratings), Low and High activity.

    Low/High are split at the median number of history ratings over all users in `history`.
    Returns the group of each user and that median.
    """
    activity = history.groupby("user_id").size()
    threshold = float(activity.median())
    counts = pd.Series(users, index=users).map(activity).fillna(0)
    groups = pd.Series(
        np.where(counts.eq(0), "Cold", np.where(counts.le(threshold), "Low", "High")),
        index=users,
    )
    return groups, int(threshold)


def popularity_buckets(history: pd.DataFrame, catalogue: set[int]) -> tuple[dict[str, set[int]], int]:
    """Splits the catalogue into the 20% most-rated movies in `history` and the other 80%.

    Ties in the number of ratings are broken by movie id, so the split is reproducible.
    Returns both item sets and the size of the popular one.
    """
    if not catalogue:
        raise ValueError("Empty catalogue")
    counts = history.groupby("item_id").size().to_dict()
    sorted_ids = sorted(catalogue, key=lambda item: (-counts.get(item, 0), item))
    n_popular = max(1, math.ceil(len(sorted_ids) * 0.2))
    popular = set(sorted_ids[:n_popular])
    tail = set(sorted_ids[n_popular:])
    return {"Popular top 20%": popular, "Long-tail other 80%": tail}, n_popular


def user_group_rows(model_name, metrics: pd.DataFrame, history: pd.DataFrame,
                    recs: dict, catalogue: set[int], k: int) -> list[dict]:
    """Disaggregated rows by user activity: mean metrics of the Cold, Low and High users."""
    groups, cutoff = user_activity_groups(history, metrics["user_id"].tolist())
    df = metrics.copy()
    df["segment"] = df["user_id"].map(groups)
    output = []
    for group, part in df.groupby("segment", sort=True):
        output.append({
            "model": model_name, "dimension": "User activity", "segment": group,
            "users_evaluated": len(part),
            "precision_at_10": part["precision"].mean(),
            "recall_at_10": part["recall"].mean(),
            "ndcg_at_10": part["ndcg"].mean(),
            "catalogue_coverage": catalogue_coverage(
                {u: recs[u] for u in part["user_id"]}, catalogue, k
            ),
            "notes": f"Warm users split at median {cutoff} training ratings; cold=0",
        })
    return output


def item_popularity_rows(model_name, recs: dict, relevant: dict,
                         groups: dict[str, set[int]], k: int) -> list[dict]:
    """Disaggregated rows by item popularity: how well the model finds popular vs long-tail hits.

    For each group only the user's relevant movies from that group count as relevant, and only
    users with at least one such movie are evaluated. The top-k lists are not changed. Coverage
    is the share of the group's movies that appear in any of these top-k lists.
    """
    output = []
    for name, items in groups.items():
        group_relevant = {u: rel.intersection(items) for u, rel in relevant.items()}
        eligible = [u for u in sorted(relevant) if group_relevant[u]]
        if not eligible:
            continue
        recalls = [recall_at_k(recs[u], group_relevant[u], k) for u in eligible]
        ndcgs = [ndcg_at_k(recs[u], group_relevant[u], k) for u in eligible]
        precisions = [precision_at_k(recs[u], group_relevant[u], k) for u in eligible]
        reached = {item for u in eligible for item in recs[u][:k] if item in items}
        output.append({
            "model": model_name, "dimension": "Item popularity", "segment": name,
            "users_evaluated": len(eligible),
            "precision_at_10": float(np.mean(precisions)),
            "recall_at_10": float(np.mean(recalls)),
            "ndcg_at_10": float(np.mean(ndcgs)),
            "catalogue_coverage": len(reached) / len(items),
            "notes": (f"{len(items)} catalogue items; matching relevant items only; "
                      "original top-10 positions retained; group coverage uses group catalogue"),
        })
    return output


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-distribution confidence interval of the mean.

    The t-distribution (not the normal one) because some samples are small (5 seeds).
    With a single value there is no spread to estimate, so the interval is NaN.
    """
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half = stats.t.ppf((1 + confidence) / 2, len(values) - 1) * stats.sem(values)
    return mean, mean - half, mean + half


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values.

    With 15 model pairs, ~0.75 of them would come out "significant" at 0.05 by luck alone.
    Holm controls that: the smallest p-value is multiplied by m, the next by m-1, ..., and the
    running maximum keeps the adjusted values in the same order as the raw ones.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


def paired_comparisons(score_data: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Paired t-test on per-user NDCG@10 for every pair of models.

    Paired: both models are scored on the same test users, so we test the per-user
    difference. That removes the (large) user-to-user spread that both models share.
    """
    output = []
    for a, b in combinations(models, 2):
        diff = score_data[a].to_numpy(dtype=float) - score_data[b].to_numpy(dtype=float)
        result = ttest_rel(score_data[a], score_data[b])
        mean, low, high = mean_ci(diff)
        output.append({
            "model_a": a, "model_b": b, "users_paired": len(diff),
            "mean_ndcg_a": float(score_data[a].mean()),
            "mean_ndcg_b": float(score_data[b].mean()),
            "mean_difference_a_minus_b": mean,
            "difference_ci95_low": low, "difference_ci95_high": high,
            "paired_t_statistic": float(result.statistic),
            "p_value_two_sided": float(result.pvalue),
        })
    output = pd.DataFrame(output)
    output["p_value_holm"] = holm(output["p_value_two_sided"])
    output["significant_at_0_05_holm"] = output["p_value_holm"] < 0.05
    return output


def evaluate_run(name: str, model, history: Dataset, users: list[int], relevant: dict,
                 catalogue: set[int], k: int,
                 already_rated: dict | None = None) -> tuple[dict, dict]:
    """Train `model` on `history`, rank the full catalogue for the test users and score it.

    `already_rated`: {user: movies rated in the full train + validation} for the sparsity
    ablation. The model only knows the subsample, so it would recommend movies the user
    rated in the dropped part. Those can never be relevant in test (MovieLens has one rating
    per user and movie), and they filled ~35% of MostPopular's top-10 slots at 50% of the data.
    That measures the evaluation, not the model, so they are removed: ask for k more movies
    than dropped ones and keep the first k the user has not rated.
    """
    model.train(history)
    if already_rated is None:
        recs = {u: model.rank(u, k) for u in users}
    else:
        known = history.ratings.groupby("user_id")["item_id"].agg(set)
        recs = {}
        for u in users:
            rated = already_rated.get(u, set())
            n_dropped = len(rated - known.get(u, set()))
            recs[u] = [i for i in model.rank(u, k + n_dropped) if i not in rated][:k]
    results = evaluate_model(recs, relevant, catalogue, k)
    return recs, results


def run_row(name: str, fraction: float, seed, repetition: str | None, results: dict) -> dict:
    """One training run in the long table of all runs (main, extra seeds and sparsity).

    `repetition` groups the runs whose models are ranked against each other: the main run,
    and each subsample of the sparsity ablation. Extra-seed runs get None (not ranked).
    """
    return {
        "model": name, "training_fraction": fraction, "repetition": repetition,
        "seed": seed if name in STOCHASTIC else None,
        "precision_at_10": results["precision"], "recall_at_10": results["recall"],
        "ndcg_at_10": results["ndcg"], "catalogue_coverage": results["coverage"],
        "users_evaluated": results["users_evaluated"],
    }


def summarise_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI over the repeated runs of each model and training fraction.

    At 100% of the data the repetitions are seeds (stochastic models only).
    In the sparsity ablation they are different random subsamples (and seeds), so there
    also the deterministic models get an interval.
    """
    rows = []
    for (fraction, name), part in runs.groupby(["training_fraction", "model"], sort=False):
        row = {"model": name, "training_fraction": fraction, "runs": len(part),
               "seeds": ", ".join(str(int(s)) for s in part["seed"].dropna().unique()) or "-"}
        for metric in ["precision_at_10", "recall_at_10", "ndcg_at_10", "catalogue_coverage"]:
            mean, low, high = mean_ci(part[metric])
            row[metric] = mean
            if metric == "ndcg_at_10":
                row["ndcg_ci95_low"], row["ndcg_ci95_high"] = low, high
                row["ndcg_min"], row["ndcg_max"] = part[metric].min(), part[metric].max()
        rows.append(row)
    summary = pd.DataFrame(rows)
    # Rank of each model within its training fraction (1 = best mean NDCG@10)
    summary["rank_by_mean_ndcg"] = (summary.groupby("training_fraction")["ndcg_at_10"]
                                    .rank(ascending=False, method="min").astype(int))
    return summary.sort_values(["training_fraction", "rank_by_mean_ndcg"],
                               ascending=[False, True]).reset_index(drop=True)


def run_analysis(data: Dataset, cfg: dict, out_dir: Path, only: set[str] | None = None,
                 with_seeds: bool = True, with_sparsity: bool = True):
    """The whole Part 2 test evaluation; writes the CSVs listed in the README to `out_dir`.

    Every model is trained on train + validation and evaluated on test with the tuned
    hyperparameters from config.yaml. `only` restricts it to some models.
    """
    s = split(data, cfg["split"])
    history_ratings = pd.concat([s.train.ratings, s.val.ratings], ignore_index=True)
    history = s.train.with_ratings(history_ratings)
    test = s.test.ratings
    k = cfg["evaluation"]["k"]
    threshold = cfg["evaluation"]["relevance_threshold"]
    if k != 10:
        raise ValueError("The official results template uses @10; please use evaluation.k = 10")
    catalogue = set(data.movies["item_id"])
    relevant = relevant_by_user(test, threshold)
    users = sorted(relevant)
    groups, n_popular = popularity_buckets(history_ratings, catalogue)
    out_dir.mkdir(parents=True, exist_ok=True)
    part2 = cfg["part2"]
    main_seed = cfg["seed"]

    names = list(MODELS)
    if only is not None:
        unknown = only - set(names)
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")
        names = [name for name in names if name in only]

    # 1. Main run: tuned config (seed 42), overall + disaggregated metrics, per-user NDCG
    overall_rows, subgroup_rows, runs = [], [], []
    scores = pd.DataFrame({"user_id": users})
    for name in names:
        print(f"Training and evaluating {name} ({len(users)} test users)...", flush=True)
        recs, results = evaluate_run(name, build_model(name, cfg), history, users,
                                     relevant, catalogue, k)
        user_metrics = per_user_metrics(recs, relevant, k)
        overall_rows.append({
            "model": name,
            "self_implemented": name in SELF_IMPLEMENTED,
            "precision_at_10": results["precision"],
            "recall_at_10": results["recall"],
            "ndcg_at_10": results["ndcg"],
            "catalogue_coverage": results["coverage"],
            "users_evaluated": results["users_evaluated"],
            "evaluation_split": "held-out test",
            "training_data": "train + validation",
            "settings": str(model_settings(name, cfg)),
        })
        runs.append(run_row(name, 1.0, main_seed, "full data", results))
        scores[name] = user_metrics["ndcg"].values
        subgroup_rows.extend(user_group_rows(name, user_metrics, history_ratings,
                                             recs, catalogue, k))
        subgroup_rows.extend(item_popularity_rows(name, recs, relevant, groups, k))
        print(f"  NDCG@10={results['ndcg']:.4f}, Recall@10={results['recall']:.4f}", flush=True)

    # 95% CI over test users of each model's NDCG@10 (how much the number depends on
    # which users happened to be in the test period)
    for row in overall_rows:
        _, row["ndcg_ci95_low_users"], row["ndcg_ci95_high_users"] = mean_ci(scores[row["model"]])

    pd.DataFrame(overall_rows).to_csv(out_dir / "part2_overall.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(out_dir / "part2_subgroups.csv", index=False)
    scores.to_csv(out_dir / "part2_per_user_ndcg.csv", index=False)
    if len(names) >= 2:
        paired_comparisons(scores, names).to_csv(out_dir / "part2_paired_tests.csv", index=False)

    # 2. Seed variability: the stochastic models retrained with the other seeds
    if with_seeds:
        for seed in part2["seeds"]:
            if seed == main_seed:
                continue  # already in the main run
            for name in [n for n in names if n in STOCHASTIC]:
                print(f"Seed {seed}: {name}...", flush=True)
                _, results = evaluate_run(name, build_model(name, cfg, seed), history, users,
                                          relevant, catalogue, k)
                runs.append(run_row(name, 1.0, seed, None, results))
                print(f"  NDCG@10={results['ndcg']:.4f}", flush=True)

    # 3. Sparsity ablation: the same hyperparameters trained on a random fraction of the
    # train + validation ratings, evaluated on the same (full) test set. Each repetition
    # draws a different subsample with its seed, and stochastic models also use that seed.
    if with_sparsity:
        already_rated = history_ratings.groupby("user_id")["item_id"].agg(set).to_dict()
        for fraction in part2["sparsity_fractions"]:
            for seed in part2["sparsity_seeds"]:
                sample = history_ratings.sample(frac=fraction, random_state=seed)
                sampled = history.with_ratings(sample.reset_index(drop=True))
                for name in names:
                    print(f"Sparsity {fraction:.0%}, sample seed {seed}: {name}...", flush=True)
                    _, results = evaluate_run(name, build_model(name, cfg, seed), sampled,
                                              users, relevant, catalogue, k, already_rated)
                    runs.append(run_row(name, fraction, seed,
                                        f"{fraction:.0%} sample, seed {seed}", results))
                    print(f"  NDCG@10={results['ndcg']:.4f}", flush=True)

    runs = pd.DataFrame(runs)
    # Rank of each model within one repetition, to see whether the order is stable
    # across subsamples (NaN for the extra-seed runs, which are not a full repetition)
    runs["rank_in_repetition"] = (runs.groupby("repetition")["ndcg_at_10"]
                                  .rank(ascending=False, method="min"))
    runs.to_csv(out_dir / "part2_runs.csv", index=False)
    summary = summarise_runs(runs)
    summary[summary["training_fraction"] == 1.0].to_csv(
        out_dir / "part2_seed_variability.csv", index=False)
    if with_sparsity:
        summary.to_csv(out_dir / "part2_sparsity.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWritten to: {out_dir}  (popular item count: {n_popular})")
    print("Copy the final tuned results into the official results template.")


def main():
    """Command-line entry point: python -m src.part2_analysis --data-dir data/ml-1m"""
    parser = argparse.ArgumentParser(description="Part 2 evaluation")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/part2_analysis"))
    parser.add_argument("--models", nargs="+", default=None, choices=list(MODELS))
    parser.add_argument("--no-seeds", action="store_true",
                        help="skip re-running the stochastic models with extra seeds")
    parser.add_argument("--no-sparsity", action="store_true", help="skip the sparsity ablation")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ds = Dataset.from_config(str(args.data_dir), cfg["load"])
    run_analysis(ds, cfg, args.output_dir, only=set(args.models) if args.models else None,
                 with_seeds=not args.no_seeds, with_sparsity=not args.no_sparsity)


if __name__ == "__main__":
    main()
