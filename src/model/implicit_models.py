import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from src.model.dataset import Dataset
from src.model.models import RecommendationModel


class _ImplicitModel(RecommendationModel):
    """Shared parts of ImplicitALS and ImplicitBPR (library `implicit`).

    Trained on binary feedback: rating >= positive_threshold -> 1.
    Subclasses implement `_build`, which returns the library model.
    """

    def __init__(self, positive_threshold: int = 4, seed: int = 42) -> None:
        super().__init__()
        self.positive_threshold = positive_threshold
        self.seed = seed
        self.data: Dataset | None = None

    def _build(self):
        raise NotImplementedError

    def train(self, data: Dataset) -> None:
        self.data = data
        r = data.ratings
        self.user_ids = pd.Index(sorted(r["user_id"].unique()))
        self.item_ids = pd.Index(sorted(r["item_id"].unique()))
        rows = self.user_ids.get_indexer(r["user_id"])
        cols = self.item_ids.get_indexer(r["item_id"])
        shape = (len(self.user_ids), len(self.item_ids))

        # all rated items -> excluded from recommendations
        self.seen = sp.csr_matrix(
            (np.ones(len(r), dtype=np.float32), (rows, cols)), shape=shape)

        # positive feedback -> training matrix
        pos = r["rating"].to_numpy() >= self.positive_threshold
        positives = sp.csr_matrix(
            (np.ones(pos.sum(), dtype=np.float32), (rows[pos], cols[pos])), shape=shape)

        self.model = self._build()
        self.model.fit(positives, show_progress=False)

    def predict(self, user: int, movie: int) -> float:
        if self.data is None:
            raise Exception("Model has not been trained yet")
        if user not in self.user_ids or movie not in self.item_ids:
            return np.nan
        u = self.user_ids.get_loc(user)
        i = self.item_ids.get_loc(movie)
        return float(self.model.user_factors[u] @ self.model.item_factors[i])

    def rank(self, user: int, n: int) -> list[int]:
        """Top-n unseen movies; [] for users unknown in train."""
        if self.data is None:
            raise Exception("Model has not been trained yet")
        if user not in self.user_ids:
            return []
        u = self.user_ids.get_loc(user)
        idx, _ = self.model.recommend(
            u, self.seen[u], N=n, filter_already_liked_items=True)
        return self.item_ids[idx].tolist()


class ImplicitALS(_ImplicitModel):
    def __init__(self, factors: int = 64, regularization: float = 0.01,
                 iterations: int = 15, alpha: float = 1.0,
                 positive_threshold: int = 4, seed: int = 42) -> None:
        super().__init__(positive_threshold, seed)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

    def _build(self):
        return AlternatingLeastSquares(
            factors=self.factors, regularization=self.regularization,
            iterations=self.iterations, alpha=self.alpha,
            random_state=self.seed, calculate_training_loss=False,
            num_threads=1)


class ImplicitBPR(_ImplicitModel):
    def __init__(self, factors: int = 64, learning_rate: float = 0.01,
                 regularization: float = 0.01, iterations: int = 100,
                 positive_threshold: int = 4, seed: int = 42) -> None:
        super().__init__(positive_threshold, seed)
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations

    def _build(self):
        return BayesianPersonalizedRanking(
            factors=self.factors, learning_rate=self.learning_rate,
            regularization=self.regularization, iterations=self.iterations,
            random_state=self.seed, num_threads=1)
