import argparse
from pathlib import Path

import yaml

from src.model import Dataset
from src.data.splitting import split
from src.model.simple_models import UserKNN, ItemKNN, MostPopular


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    names = [m["name"] for m in config.get("team", {}).get("members", [])]

    print("args:", args, "\n")

    print("names:")
    for name in names:
        print(f" - {name}")

    load_cfg = config["load"]
    data_dir = args.data_dir or load_cfg["data_dir"]

    ds = Dataset.from_config(data_dir, load_cfg)

    print("\ndataset:")
    print(f" - users:   {ds.n_users}")
    print(f" - items:   {ds.n_items}")
    print(f" - ratings: {ds.n_ratings}")

    s = split(ds, config["split"])

    print("\nsplit:")
    print(s.summary())

    model = UserKNN()
    model.train(s.train)
    print(model.predict(635, 1))
    print(model.rank(635, 10))

    model = ItemKNN()
    model.train(s.train)
    print(model.predict(635, 1))
    print(model.rank(635, 10))

    model = MostPopular()
    model.train(s.train)
    print(model.rank(635, 10))


if __name__ == "__main__":
    main()
