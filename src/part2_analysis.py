from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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


def model_registry(config: dict):
    c = config["models"]
    return {
        "MostPopular": MostPopular(**c["most_popular"][0]),
        "ItemKNN": ItemKNN(**c["item_knn"]),
        "UserKNN": UserKNN(**c["user_knn"]),
        "ImplicitALS": ImplicitALS(**c["implicit_als"]),
        "ImplicitBPR": ImplicitBPR(**c["implicit_bpr"]),
    }


def per_user_metrics(recommendations: dict, relevant: dict, k: int) -> pd.DataFrame:
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
    activity = history.groupby("user_id").size()
    threshold = float(activity.median())
    counts = pd.Series(users, index=users).map(activity).fillna(0)
    groups = pd.Series(
        np.where(counts.eq(0), "Cold", np.where(counts.le(threshold), "Low", "High")),
        index=users,
    )
    return groups, int(threshold)


def popularity_buckets(history: pd.DataFrame, catalogue: set[int]) -> tuple[dict[str, set[int]], int]:
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


def paired_comparisons(score_data: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("ImplicitALS", "ImplicitBPR")]
    output = []
    for a, b in comparisons:
        sa = score_data[a].to_numpy(dtype=float)
        sb = score_data[b].to_numpy(dtype=float)
        result = ttest_rel(sa, sb)
        output.append({
            "model_a": a, "model_b": b, "users_paired": len(score_data),
            "mean_ndcg_a": float(np.mean(sa)), "mean_ndcg_b": float(np.mean(sb)),
            "mean_difference_a_minus_b": float(np.mean(sa - sb)),
            "paired_t_statistic": float(result.statistic),
            "p_value_two_sided": float(result.pvalue),
            "significant_at_0_05_unadjusted": bool(result.pvalue < 0.05),
        })
    return pd.DataFrame(output)


def run_analysis(data: Dataset, cfg: dict, out_dir: Path, only: set[str] | None = None):
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

    all_models = model_registry(cfg)
    if only is not None:
        unknown = only - set(all_models)
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")
        models = {name: obj for name, obj in all_models.items() if name in only}
    else:
        models = all_models

    overall_rows, subgroup_rows, scores = [], [], pd.DataFrame({"user_id": users})
    for name, model in models.items():
        print(f"Training and evaluating {name} ({len(users)} test users)...", flush=True)
        model.train(history)
        recs = {u: model.rank(u, k) for u in users}
        results = evaluate_model(recs, relevant, catalogue, k)
        user_metrics = per_user_metrics(recs, relevant, k)
        config_key = {
            "MostPopular": "most_popular",
            "ItemKNN": "item_knn",
            "UserKNN": "user_knn",
            "ImplicitALS": "implicit_als",
            "ImplicitBPR": "implicit_bpr",
        }[name]
        settings = cfg["models"][config_key]
        if name == "MostPopular":
            settings = settings[0]
        overall_rows.append({
            "model": name,
            "precision_at_10": results["precision"],
            "recall_at_10": results["recall"],
            "ndcg_at_10": results["ndcg"],
            "catalogue_coverage": results["coverage"],
            "users_evaluated": results["users_evaluated"],
            "evaluation_split": "held-out test",
            "training_data": "train + validation",
            "settings": str(settings),
        })
        if len(user_metrics):
            scores[name] = user_metrics["ndcg"].values
        subgroup_rows.extend(user_group_rows(name, user_metrics, history_ratings,
                                             recs, catalogue, k))
        subgroup_rows.extend(item_popularity_rows(name, recs, relevant, groups, k))
        print(f"  NDCG@10={results['ndcg']:.4f}, Recall@10={results['recall']:.4f}", flush=True)

    pd.DataFrame(overall_rows).to_csv(out_dir / "part2_overall.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(out_dir / "part2_subgroups.csv", index=False)
    if all(c in scores for c in ("ImplicitALS", "ImplicitBPR")):
        paired_comparisons(scores).to_csv(out_dir / "part2_paired_tests.csv", index=False)
    print(f"\nWritten to: {out_dir}  (popular item count: {n_popular})")
    print("Copy the final tuned results into the official results template.")


def main():
    parser = argparse.ArgumentParser(description="Part 2 evaluation")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/part2_analysis"))
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["MostPopular", "ItemKNN", "UserKNN", "ImplicitALS", "ImplicitBPR"])
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ds = Dataset.from_config(str(args.data_dir), cfg["load"])
    run_analysis(ds, cfg, args.output_dir, only=set(args.models) if args.models else None)


if __name__ == "__main__":
    main()
