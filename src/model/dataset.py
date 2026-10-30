from dataclasses import dataclass

import pandas as pd

from src.data import load_ratings, load_movies, load_users


@dataclass(frozen=True)
class Dataset:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame

    @property
    def n_users(self) -> int:
        return self.ratings["user_id"].nunique()

    @property
    def n_items(self) -> int:
        return self.ratings["item_id"].nunique()

    @property
    def n_ratings(self) -> int:
        return len(self.ratings)

    @classmethod
    def from_config(cls: type["Dataset"], data_dir: str, load_cfg: dict) -> "Dataset":
        return cls(
            ratings=load_ratings(data_dir, load_cfg),
            movies=load_movies(data_dir, load_cfg),
            users=load_users(data_dir, load_cfg),
        )
