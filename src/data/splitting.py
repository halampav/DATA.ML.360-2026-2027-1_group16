from src.model import Dataset, Split


def split(ds: Dataset, split_cfg: dict) -> Split:
    val_frac = split_cfg["val_frac"]
    test_frac = split_cfg["test_frac"]
    ratings = ds.ratings
    ts = ratings["ts"]

    t_val = int(ts.quantile(1 - val_frac - test_frac))
    t_test = int(ts.quantile(1 - test_frac))

    return Split(
        train=ds.with_ratings(ratings[ts <= t_val].reset_index(drop=True)),
        val=ds.with_ratings(ratings[(ts > t_val) & (ts <= t_test)].reset_index(drop=True)),
        test=ds.with_ratings(ratings[ts > t_test].reset_index(drop=True)),
        t_val=t_val,
        t_test=t_test,
    )
