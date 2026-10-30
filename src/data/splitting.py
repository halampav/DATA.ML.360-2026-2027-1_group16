import pandas as pd

from src.model import Split


def split(ratings: pd.DataFrame, split_cfg: dict) -> Split:
    val_frac = split_cfg["val_frac"]
    test_frac = split_cfg["test_frac"]
    ts = ratings["ts"]

    t_val = int(ts.quantile(1 - val_frac - test_frac))
    t_test = int(ts.quantile(1 - test_frac))

    return Split(
        train=ratings[ts <= t_val].reset_index(drop=True),
        val=ratings[(ts > t_val) & (ts <= t_test)].reset_index(drop=True),
        test=ratings[ts > t_test].reset_index(drop=True),
        t_val=t_val,
        t_test=t_test,
    )
