from dataclasses import dataclass, replace

import pandas as pd
import scipy.sparse as sp

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

    def with_ratings(self, ratings: pd.DataFrame) -> "Dataset":
        return replace(self, ratings=ratings)

    def rating_matrix(self) -> tuple[sp.csr_matrix, pd.Index, pd.Index]:
        """Sparse user x item matrix of ratings (0 = not rated).

        Returns the matrix plus the user and item ids of its rows and columns,
        so row `users.get_loc(user_id)` holds that user's ratings.
        """
        users = pd.Index(sorted(self.ratings["user_id"].unique()))
        items = pd.Index(sorted(self.ratings["item_id"].unique()))
        rows = users.get_indexer(self.ratings["user_id"])
        cols = items.get_indexer(self.ratings["item_id"])
        values = self.ratings["rating"].to_numpy(dtype=float)
        matrix = sp.csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))
        return matrix, users, items

    @classmethod
    def from_config(cls: type["Dataset"], data_dir: str, load_cfg: dict) -> "Dataset":
        return cls(
            ratings=load_ratings(data_dir, load_cfg),
            movies=load_movies(data_dir, load_cfg),
            users=load_users(data_dir, load_cfg),
        )
