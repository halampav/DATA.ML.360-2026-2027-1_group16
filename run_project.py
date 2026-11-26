"""Run the existing Group 16 baseline/Part 2 evaluation and Part 3 notebook.

Usage (from repository root): python run_project.py --data-dir data/ml-1m

Does not modify the original Part 1/2 or MMR implementation.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reproduce Group 16 project results")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = (root / args.data_dir).resolve()
    config = (root / args.config).resolve()
    output_dir = (root / args.output_dir).resolve()

    for filename in ("ratings.dat", "movies.dat", "users.dat"):
        if not (data_dir / filename).is_file():
            parser.error(f"Missing {data_dir / filename}. Download and extract MovieLens 1M first.")
    if not config.is_file():
        parser.error(f"Missing config: {config}")

    # The committed notebook assumes data/ml-1m and results/ relative to repo root.
    standard_data_dir = (root / "data/ml-1m").resolve()
    standard_output_dir = (root / "results").resolve()
    if data_dir != standard_data_dir or output_dir != standard_output_dir or config != (root / "config.yaml").resolve():
        parser.error("Part 3 notebook currently requires default paths: --data-dir data/ml-1m, --config config.yaml and --output-dir results")

    # Part 1 & 2: invoke the existing, unchanged project entrypoint.
    subprocess.run([
        sys.executable, str(root / "run_all.py"),
        "--data-dir", str(data_dir),
        "--config", str(config),
        "--output-dir", str(output_dir),
    ], cwd=root, check=True)

    # Part 3: execute the existing notebook's Python cells without notebook UI dependencies.
    notebook = root / "notebooks/05_part3_mmr.ipynb"
    with notebook.open(encoding="utf-8") as stream:
        cells = json.load(stream)["cells"]
    scope = {"__name__": "__main__"}
    print("\nRunning existing Part 3 notebook cells...", flush=True)
    for i, cell in enumerate(cells, start=1):
        if cell.get("cell_type") == "code":
            code = "".join(cell.get("source", []))
            if code.strip():
                exec(compile(code, f"{notebook.name}:cell_{i}", "exec"), scope)
    print("\nDone. Part 3 CSVs and plot are in results/", flush=True)


if __name__ == "__main__":
    main()
