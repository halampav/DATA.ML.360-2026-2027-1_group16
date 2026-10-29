import argparse
from pathlib import Path

import yaml


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


if __name__ == "__main__":
    main()
