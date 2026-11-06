import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from src.model import Dataset
from src.data.splitting import split
from src.evaluation import evaluate_recommender
from src.model.simple_models import ItemKNN, MostPopular


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def most_popular_settings(variant: dict) -> str:
    """Human-readable description of a MostPopular variant for the results table."""
    window = "all-time" if variant["window_days"] is None else f"last {variant['window_days']} days"
    counted = f"only ratings >= {variant['positive_threshold']}" if variant["only_positive"] else "all ratings"
    return f"{window}, {counted}"


def result_row(model: str, settings: str, similarity: str, k, metrics: dict, notes: str) -> dict:
    """One row in the format of the 'Part 1 - Baselines' sheet of the results template."""
    # How the model does on users known from history vs. new users (cold start)
    for segment, name in [("warm", "users with history"), ("cold", "new users")]:
        if f"ndcg_{segment}" in metrics:
            notes += (f"; NDCG@10 {name}: {metrics[f'ndcg_{segment}']:.4f} "
                      f"(n={metrics[f'users_evaluated_{segment}']})")
    return {
        "Model": model,
        "Variant / settings": settings,
        "Similarity": similarity,
        "k": k,
        "Precision@10": round(metrics["precision"], 4),
        "Recall@10": round(metrics["recall"], 4),
        "NDCG@10": round(metrics["ndcg"], 4),
        "Catalogue coverage": round(metrics["coverage"], 4),
        "Seed": "-",
        "Notes": "deterministic (no randomness); " + notes,
    }


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    names = [m["name"] for m in config.get("team", {}).get("members", [])]

    print("args:", args, "\n")

    print("names:")
    for name in names:
        print(f" - {name}")

    load_cfg = config["load"]
    data_dir = args.data_dir or load_cfg.get("data_dir")
    if data_dir is None:
        raise SystemExit("Pass the MovieLens folder with --data-dir (or set load.data_dir in the config)")

    ds = Dataset.from_config(data_dir, load_cfg)

    print("\ndataset:")
    print(f" - users:   {ds.n_users}")
    print(f" - items:   {ds.n_items}")
    print(f" - ratings: {ds.n_ratings}")

    s = split(ds, config["split"])

    print("\nsplit:")
    print(s.summary())

    # Hyperparameters were chosen on validation (notebooks/02_tuning.ipynb). For the final test
    # evaluation the models are trained on train + validation: validation is the month right
    # before the test period, so leaving it out would throw away the most recent history.
    history = s.train.with_ratings(pd.concat([s.train.ratings, s.val.ratings], ignore_index=True))
    test = s.test.ratings

    # Ranked against the full catalogue: every movie in movies.dat
    catalogue = set(ds.movies["item_id"])
    k = config["evaluation"]["k"]
    threshold = config["evaluation"]["relevance_threshold"]

    def evaluate(model):
        start = time.time()
        model.train(history)
        metrics = evaluate_recommender(model, history.ratings, test, catalogue, k=k, threshold=threshold)
        print(f"   NDCG@{k}={metrics['ndcg']:.4f} P@{k}={metrics['precision']:.4f} "
              f"R@{k}={metrics['recall']:.4f} coverage={metrics['coverage']:.4f} "
              f"({time.time() - start:.0f} s)")
        return metrics

    print(f"\ntest evaluation (trained on train + validation, k={k}, relevant = rating >= {threshold}):")
    rows = []

    for i, variant in enumerate(config["models"]["most_popular"]):
        settings = most_popular_settings(variant)
        print(f" - MostPopular ({settings})")
        notes = "selected on validation" if i == 0 else "alternative variant"
        rows.append(result_row("MostPopular", settings, "", "", evaluate(MostPopular(**variant)), notes))

    knn_cfg = config["models"]["item_knn"]
    weighting = ("no significance weighting" if knn_cfg["gamma"] is None
                 else f"significance weighting gamma={knn_cfg['gamma']}")
    settings = f"mean-centred prediction, {weighting}, min_support={knn_cfg['min_support']}"
    print(f" - ItemKNN (k={knn_cfg['k']}, {knn_cfg['similarity_measure']}, {settings})")
    rows.append(result_row("ItemKNN", settings, knn_cfg["similarity_measure"], knn_cfg["k"],
                           evaluate(ItemKNN(**knn_cfg)), "k and similarity tuned on validation"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\nresults written to {results_path}")


if __name__ == "__main__":
    main()
