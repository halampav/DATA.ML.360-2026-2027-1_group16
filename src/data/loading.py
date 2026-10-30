from pathlib import Path
import pandas as pd



def _load_table(data_dir: str, load_cfg: dict, table: str) -> pd.DataFrame:
    table_cfg = load_cfg["files"][table]
    return pd.read_csv(
        Path(data_dir) / table_cfg["filename"],
        sep=load_cfg["separator"],
        engine="python",
        header=None,
        names=table_cfg["columns"],
        encoding=load_cfg["encoding"],
    )

def load_ratings(data_dir: str, load_cfg: dict) -> pd.DataFrame:
    return _load_table(data_dir, load_cfg, "ratings")

def load_users(data_dir: str, load_cfg: dict) -> pd.DataFrame:
    return _load_table(data_dir, load_cfg, "users")

def load_movies(data_dir: str, load_cfg: dict) -> pd.DataFrame:
    df = _load_table(data_dir, load_cfg, "movies")
    df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")
    df["genres"] = df["genres"].str.split("|")
    return df
