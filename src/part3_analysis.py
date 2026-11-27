"""Part 3, Track A: MMR re-ranking of the Part 2 ALS model, and what it costs.

python -m src.part3_analysis --data-dir data/ml-1m

1. Validation sweep: ALS trained on train, lambda swept over `part3.lambdas`, evaluated on the
   validation users with training history (the Part 2 tuning objective). Lambda is chosen here.
2. Test sweep: ALS trained on train + validation, the same lambdas, evaluated on all test users
   (as in Part 2, cold users included). This is the accuracy-diversity frontier we report.
3. Paired t-tests on test: every lambda against lambda = 1 (= plain ALS), per-user NDCG@10.
4. Who pays: the chosen lambda against plain ALS per user group (activity, taste breadth), and
   which genres and how many popular movies gain or lose exposure.
5. Candidate-pool ablation: the chosen lambda with pools of different size.

Outputs go to results/part3_analysis/; the rows of the results template to
results/results_part3_diversity.csv. The figures are drawn by notebooks/05_part3_mmr.ipynb.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_rel

from src.data.splitting import split
from src.evaluation import catalogue_coverage, relevant_by_user
from src.model import Dataset
from src.model.implicit_models import ImplicitALS
from src.part2_analysis import holm, mean_ci, per_user_metrics, popularity_buckets, user_activity_groups
from src.part3_mmr import (
    calibration_error,
    candidate_pool,
    genre_distribution,
    genre_vectors,
    gini_index,
    intra_list_diversity,
    mmr_select,
)

BASE_MODEL = "ImplicitALS"
ITEM_DISTANCE = "1 - Jaccard similarity of binary genre vectors"


def build_pools(model, users: list[int], size: int, vectors: pd.DataFrame) -> dict:
    """Candidate pool of every user, computed once and re-used for every lambda and pool size."""
    return {u: candidate_pool(model, u, size, vectors) for u in users}


def rerank(pools: dict, lambda_rel: float, k: int, pool_size: int) -> dict[int, list[int]]:
    """MMR top-k list of every user, using the first `pool_size` candidates of the pool.

    The pool is the base ranking, so its first `pool_size` items are the base top-`pool_size`.
    """
    recs = {}
    for u, (items, scores, similarity) in pools.items():
        if not items:
            recs[u] = []
            continue
        n = min(pool_size, len(items))
        picked = mmr_select(scores[:n], similarity[:n, :n], k, lambda_rel)
        recs[u] = [items[p] for p in picked]
    return recs


def liked_profiles(history: pd.DataFrame, threshold: int, vectors: pd.DataFrame) -> dict:
    """Genre distribution of the movies every user liked (rating >= threshold) in `history`."""
    liked = history[history["rating"] >= threshold].groupby("user_id")["item_id"].apply(list)
    return {u: genre_distribution(items, vectors) for u, items in liked.items()}


def per_user_scores(recs: dict, relevant: dict, profiles: dict, popular: set[int],
                    vectors: pd.DataFrame, k: int) -> pd.DataFrame:
    """Accuracy, diversity, calibration and popular share of every evaluated user's list."""
    scores = per_user_metrics(recs, relevant, k)
    scores["ild"] = [intra_list_diversity(recs[u][:k], vectors) for u in scores["user_id"]]
    scores["calibration_error"] = [calibration_error(profiles.get(u), recs[u][:k], vectors)
                                   for u in scores["user_id"]]
    scores["popular_share"] = [np.mean([i in popular for i in recs[u][:k]]) if recs[u] else np.nan
                               for u in scores["user_id"]]
    return scores


def summarise(scores: pd.DataFrame, recs: dict, catalogue: set[int], k: int) -> dict:
    """One row of the frontier: mean per-user metrics plus the list-level coverage and Gini."""
    return {
        "users": len(scores),
        "precision": scores["precision"].mean(),
        "recall": scores["recall"].mean(),
        "ndcg": scores["ndcg"].mean(),
        "ild": scores["ild"].mean(),                       # NaN (empty lists) are skipped
        "users_with_ild": int(scores["ild"].notna().sum()),
        "coverage": catalogue_coverage(recs, catalogue, k),
        "gini": gini_index(recs, catalogue, k),
        "calibration_error": scores["calibration_error"].mean(),
        "users_with_calibration": int(scores["calibration_error"].notna().sum()),
        "popular_share": scores["popular_share"].mean(),
    }


def sweep(split_name: str, pools: dict, history: pd.DataFrame, relevant: dict,
          vectors: pd.DataFrame, catalogue: set[int], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates every lambda on one split. Returns the frontier and the per-user scores."""
    p3, k = cfg["part3"], cfg["evaluation"]["k"]
    threshold = cfg["evaluation"]["relevance_threshold"]
    profiles = liked_profiles(history, threshold, vectors)
    popular = popularity_buckets(history, catalogue)[0]["Popular top 20%"]

    frontier, per_user = [], []
    for lam in p3["lambdas"]:
        recs = rerank(pools, lam, k, p3["candidates"])
        scores = per_user_scores(recs, relevant, profiles, popular, vectors, k)
        frontier.append({"split": split_name, "lambda": lam, **summarise(scores, recs, catalogue, k)})
        per_user.append(scores.assign(split=split_name, **{"lambda": lam}))
        print(f"   {split_name} lambda={lam:.1f}: NDCG@10={frontier[-1]['ndcg']:.4f} "
              f"ILD={frontier[-1]['ild']:.4f}", flush=True)
    return pd.DataFrame(frontier), pd.concat(per_user, ignore_index=True)


def paired_tests(per_user: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Paired t-test of every lambda < 1 against lambda = 1 (plain ALS) on a per-user metric.

    Users with a NaN value (empty list, no liked history) are left out of that metric's test.
    Holm-corrected over the compared lambdas.
    """
    wide = per_user.pivot(index="user_id", columns="lambda", values=metric)
    rows = []
    for lam in sorted(wide.columns, reverse=True):
        if lam == 1.0:
            continue
        pair = wide[[lam, 1.0]].dropna()
        diff = pair[lam] - pair[1.0]
        mean, low, high = mean_ci(diff)
        rows.append({
            "metric": metric, "lambda": lam, "users_paired": len(pair),
            "mean_base": pair[1.0].mean(), "mean_mmr": pair[lam].mean(),
            "mean_difference": mean, "difference_ci95_low": low, "difference_ci95_high": high,
            "relative_change": mean / pair[1.0].mean(),
            "p_value_two_sided": float(ttest_rel(pair[lam], pair[1.0]).pvalue),
        })
    rows = pd.DataFrame(rows)
    rows["p_value_holm"] = holm(rows["p_value_two_sided"])
    rows["significant_at_0_05_holm"] = rows["p_value_holm"] < 0.05
    return rows


def taste_breadth_groups(profiles: dict, users: list[int]) -> tuple[pd.Series, float]:
    """Splits users with liked history into Narrow / Broad taste at the median genre entropy.

    Entropy (bits) of the genre distribution of the movies the user liked: low = the user likes
    few genres. Users without liked history are left out.
    """
    entropy = {}
    for u in users:
        p = profiles.get(u)
        if p is not None:
            p = p[p > 0]
            entropy[u] = float(-(p * np.log2(p)).sum())
    entropy = pd.Series(entropy)
    threshold = float(entropy.median())
    return pd.Series(np.where(entropy <= threshold, "Narrow taste", "Broad taste"),
                     index=entropy.index), threshold


def who_pays(per_user: pd.DataFrame, history: pd.DataFrame, profiles: dict, chosen: float) -> pd.DataFrame:
    """Chosen lambda against plain ALS per user group: means, paired NDCG difference with CI."""
    users = sorted(per_user["user_id"].unique())
    activity, activity_cut = user_activity_groups(history, users)
    breadth, breadth_cut = taste_breadth_groups(profiles, users)
    dimensions = {
        "User activity": (activity, {"Cold": "0 history ratings",
                                     "Low": f"<= {activity_cut} history ratings (median)",
                                     "High": f"> {activity_cut} history ratings"}),
        "Taste breadth": (breadth, {"Narrow taste": f"genre entropy <= {breadth_cut:.2f} bits (median)",
                                    "Broad taste": f"genre entropy > {breadth_cut:.2f} bits"}),
    }
    base = per_user[per_user["lambda"] == 1.0].set_index("user_id")
    mmr = per_user[per_user["lambda"] == chosen].set_index("user_id")
    rows = []
    for dimension, (groups, definitions) in dimensions.items():
        for group, definition in definitions.items():
            members = groups.index[groups == group]
            if len(members) == 0:
                continue
            diff = mmr.loc[members, "ndcg"] - base.loc[members, "ndcg"]
            mean, low, high = mean_ci(diff)
            row = {"dimension": dimension, "group": group, "definition": definition,
                   "users": len(members), "ndcg_difference": mean,
                   "ndcg_difference_ci95_low": low, "ndcg_difference_ci95_high": high}
            for metric in ["ndcg", "ild", "calibration_error", "popular_share"]:
                row[f"{metric}_base"] = base.loc[members, metric].mean()
                row[f"{metric}_mmr"] = mmr.loc[members, metric].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def genre_exposure(recs_by_lambda: dict, vectors: pd.DataFrame, catalogue: set[int]) -> pd.DataFrame:
    """Share of all recommendation slots that goes to every genre, per lambda.

    Uses the same per-movie split of the weight over its genres as the calibration error.
    The catalogue share of every genre is added for comparison.
    """
    rows = {"catalogue": genre_distribution(sorted(catalogue), vectors)}
    for lam, recs in recs_by_lambda.items():
        rows[f"lambda={lam:.1f}"] = genre_distribution([i for items in recs.values() for i in items], vectors)
    return pd.DataFrame(rows, index=vectors.columns).rename_axis("genre").reset_index()


def pool_ablation(model, history: pd.DataFrame, users: list[int], relevant: dict,
                  vectors: pd.DataFrame, catalogue: set[int], cfg: dict) -> pd.DataFrame:
    """The chosen lambda (and plain ALS) with candidate pools of different size."""
    p3, k = cfg["part3"], cfg["evaluation"]["k"]
    threshold = cfg["evaluation"]["relevance_threshold"]
    pools = build_pools(model, users, max(p3["pool_sizes"]), vectors)
    profiles = liked_profiles(history, threshold, vectors)
    popular = popularity_buckets(history, catalogue)[0]["Popular top 20%"]
    rows = []
    for size in p3["pool_sizes"]:
        for lam in sorted({1.0, p3["chosen_lambda"]}):
            recs = rerank(pools, lam, k, size)
            scores = per_user_scores(recs, relevant, profiles, popular, vectors, k)
            rows.append({"pool_size": size, "lambda": lam, **summarise(scores, recs, catalogue, k)})
        print(f"   pool {size}: done", flush=True)
    return pd.DataFrame(rows)


def template_rows(test: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The test frontier in the columns of the "Part 3A - Diversity" sheet of the results template."""
    p3 = cfg["part3"]
    settings = ", ".join(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
                         for key, value in cfg["models"]["implicit_als"].items())
    rows = []
    for _, r in test.iterrows():
        if r["lambda"] == 1.0:
            note = "Reference: lambda=1 keeps the ALS ranking (= unmodified base model)"
        elif r["lambda"] == p3["chosen_lambda"]:
            note = "Chosen operating point (on validation)"
        else:
            note = ""
        rows.append({
            "Base model": f"ImplicitALS ({settings}); trained on train + validation",
            "Re-ranking method": f"MMR over the ALS top-{p3['candidates']}",
            "Trade-off parameter": f"lambda={r['lambda']:.1f}",
            "Item distance": ITEM_DISTANCE,
            "Precision@10": round(r["precision"], 4),
            "Recall@10": round(r["recall"], 4),
            "NDCG@10": round(r["ndcg"], 4),
            "Intra-list diversity": round(r["ild"], 4),
            "Catalogue coverage": round(r["coverage"], 4),
            "Gini": round(r["gini"], 4),
            "Calibration error": round(r["calibration_error"], 4),
            "Notes": "; ".join(x for x in [note, f"{int(r['users'])} test users, {int(r['users'] - r['users_with_ild'])} cold users get no list",
                                           "calibration error = Jensen-Shannon divergence (base 2)"] if x),
        })
    return pd.DataFrame(rows)


def run_analysis(data: Dataset, cfg: dict, out_dir: Path) -> None:
    """Runs all Part 3 experiments and writes the CSVs (see the module docstring)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k, threshold = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    chosen = cfg["part3"]["chosen_lambda"]
    s = split(data, cfg["split"])
    vectors = genre_vectors(data.movies)
    catalogue = set(data.movies["item_id"])
    start = time.time()

    # 1. validation: ALS trained on train, users with training history (Part 2 tuning objective)
    print("validation sweep (ALS trained on train):", flush=True)
    model = ImplicitALS(**cfg["models"]["implicit_als"])
    model.train(s.train)
    val_relevant = relevant_by_user(s.val.ratings, threshold)
    train_users = set(s.train.ratings["user_id"])
    val_users = sorted(u for u in val_relevant if u in train_users)
    pools = build_pools(model, val_users, cfg["part3"]["candidates"], vectors)
    val_frontier, _ = sweep("validation", pools, s.train.ratings,
                            {u: val_relevant[u] for u in val_users}, vectors, catalogue, cfg)

    # 2. test: ALS trained on train + validation, all test users with a relevant movie
    print("test sweep (ALS trained on train + validation):", flush=True)
    history = s.train.with_ratings(pd.concat([s.train.ratings, s.val.ratings], ignore_index=True))
    model = ImplicitALS(**cfg["models"]["implicit_als"])
    model.train(history)
    test_relevant = relevant_by_user(s.test.ratings, threshold)
    test_users = sorted(test_relevant)
    pools = build_pools(model, test_users, cfg["part3"]["candidates"], vectors)
    test_frontier, per_user = sweep("test", pools, history.ratings, test_relevant,
                                    vectors, catalogue, cfg)
    pd.concat([val_frontier, test_frontier]).to_csv(out_dir / "part3_frontier.csv", index=False)
    per_user.to_csv(out_dir / "part3_per_user.csv", index=False)

    # 3. paired tests against plain ALS
    tests = pd.concat([paired_tests(per_user, m) for m in ["ndcg", "ild", "calibration_error"]])
    tests.to_csv(out_dir / "part3_paired_tests.csv", index=False)

    # 4. who pays: user groups, genre exposure
    profiles = liked_profiles(history.ratings, threshold, vectors)
    who_pays(per_user, history.ratings, profiles, chosen).to_csv(out_dir / "part3_who_pays.csv", index=False)
    recs_by_lambda = {lam: rerank(pools, lam, k, cfg["part3"]["candidates"]) for lam in [1.0, chosen, 0.0]}
    genre_exposure(recs_by_lambda, vectors, catalogue).to_csv(out_dir / "part3_genre_exposure.csv", index=False)

    # 5. candidate-pool size
    print("candidate-pool ablation (test):", flush=True)
    pool_ablation(model, history.ratings, test_users, test_relevant, vectors, catalogue,
                  cfg).to_csv(out_dir / "part3_pool_size.csv", index=False)

    template_rows(test_frontier, cfg).to_csv(out_dir.parent / "results_part3_diversity.csv", index=False)
    print(f"done in {time.time() - start:.0f} s; results in {out_dir}", flush=True)


def main():
    """Command-line entry point: python -m src.part3_analysis --data-dir data/ml-1m"""
    parser = argparse.ArgumentParser(description="Part 3 (Track A): MMR re-ranking")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/part3_analysis"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = Dataset.from_config(str(args.data_dir), cfg["load"])
    run_analysis(data, cfg, args.output_dir)


if __name__ == "__main__":
    main()
