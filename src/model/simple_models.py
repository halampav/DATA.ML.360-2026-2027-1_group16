from typing import Callable

import numpy as np
import scipy.sparse as sp

from src.model.dataset import Dataset
from src.model.models import RecommendationModel

def _dense(m) -> np.ndarray:
    """Helper function for flattening np arrays and CSR matrices"""
    if hasattr(m, "toarray"):
        m = m.toarray()
    return np.asarray(m).ravel()


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, with 0 wherever den is 0."""
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


class RowSimilarity:
    """Helper class that extracts similair code for similarities in UserKNN and ItemKNN

    Pass user-item matrix for UserKNN and item-user matrix for ItemKNN
    """
    MEASURES = ["cosine", "pearson", "jaccard"]

    def __init__(self, matrix: sp.csr_matrix, measure: str, gamma: int | None) -> None:
        self.matrix = matrix
        self.measure = measure # Similarity measure
        self.gamma = gamma # parameter for importance weighting

        # binary matrix rated → 1, did not → 0
        self.rated = matrix.copy()
        self.rated.data[:] = 1.0

        # Number of ratings in each row (for Jaccard)
        self.n_rated = _dense(self.rated.sum(axis=1))

        # Length of each row's rating vector (for cosine)
        self.norms = np.sqrt(_dense(matrix.multiply(matrix).sum(axis=1)))

        # Each row's mean rating (for Pearson and the prediction formula)
        self.means = _dense(matrix.sum(axis=1)) / np.maximum(self.n_rated, 1)

        # Ratings-row_mean, and their squares (for Pearson)
        row_of_entry = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        self.centered = matrix.copy()
        self.centered.data -= self.means[row_of_entry]
        self.centered_sq = self.centered.multiply(self.centered).tocsr()

    def between(self, rows: np.ndarray, others: np.ndarray) -> np.ndarray:
        """len(rows) x len(others) array of similarities."""
        sims = self._raw(rows, others)
        if self.gamma is None:
            return sims
        # trust similarities less when the two rows have few ratings in common
        n_co_rated = (self.rated[rows] @ self.rated[others].T).toarray()
        return sims * np.minimum(n_co_rated, self.gamma) / self.gamma

    def _raw(self, rows: np.ndarray, others: np.ndarray) -> np.ndarray:
        if self.measure == "cosine":
            dots = (self.matrix[rows] @ self.matrix[others].T).toarray()
            return _safe_divide(dots, np.outer(self.norms[rows], self.norms[others]))

        if self.measure == "jaccard":
            both = (self.rated[rows] @ self.rated[others].T).toarray()
            either = self.n_rated[rows][:, None] + self.n_rated[others][None, :] - both
            return _safe_divide(both, either)

        if self.measure == "pearson":
            # pearson over co-rated entries: multiplying by `rated` keeps only entries the other row rated too
            num = (self.centered[rows] @ self.centered[others].T).toarray()
            var_rows = (self.centered_sq[rows] @ self.rated[others].T).toarray()
            var_others = (self.rated[rows] @ self.centered_sq[others].T).toarray()
            return _safe_divide(num, np.sqrt(var_rows * var_others))
        raise Exception(f"Unknown similarity measure {self.measure}")


class _KNN(RecommendationModel):
    """Shared parts of UserKNN and ItemKNN; subclasses implement `_scorer`."""

    def __init__(self, k: int = 25, similarity_measure: str = "cosine",
                 gamma: int | None = 50) -> None:
        """`gamma`: similarities based on fewer co-ratings than this are scaled
        down by n_co_rated / gamma (significance weighting). None turns it off."""
        super().__init__()
        if similarity_measure not in RowSimilarity.MEASURES:
            raise ValueError(f"Unknown similarity measure: {similarity_measure}, "
                             f"choose one of {RowSimilarity.MEASURES}")
        self.k = k
        self.sim = similarity_measure
        self.gamma = gamma
        self.data: Dataset | None = None

    def train(self, data: Dataset) -> None:
        self.data = data

        # Get CSC and CSR rating matrices for fast lookup
        self.matrix, self.user_ids, self.item_ids = data.rating_matrix()
        self.matrix_csc = self.matrix.tocsc()

    def _scorer(self, u: int) -> Callable[[int], float]:
        """Returns a function that predicts the rating of user row `u` for item column i.
        Whatever depends only on u is computed once here, not once per item."""
        raise NotImplementedError

    def predict(self, user: int, movie: int) -> float:
        if self.data is None:
            raise Exception("Model has not been trained yet")

        # cold start: user or movie not known, we are unable to predict
        if user not in self.user_ids or movie not in self.item_ids:
            return np.nan
        u = self.user_ids.get_loc(user)
        i = self.item_ids.get_loc(movie)
        return self._scorer(u)(i)

    def rank(self, user: int, n: int) -> list[int]:
        """Top-n movie ids for `user`, scored with the same formula as `predict`.

        Skips movies the user already rated in train. Returns [] for users unknown in train.
        """
        if self.data is None:
            raise Exception("Model has not been trained yet")
        if user not in self.user_ids:
            return []
        u = self.user_ids.get_loc(user)
        score_func = self._scorer(u)

        seen = set(self.matrix[u].indices)
        scores = {i: score_func(i) for i in range(len(self.item_ids)) if i not in seen}
        scores = {i: s for i, s in scores.items() if not np.isnan(s)}

        best = sorted(scores, key=scores.get, reverse=True)[:n]
        return self.item_ids[best].tolist()


class UserKNN(_KNN):
    def train(self, data: Dataset) -> None:
        super().train(data)
        # rows = users
        self.similarity = RowSimilarity(self.matrix, self.sim, self.gamma)
        self.means = self.similarity.means

    def _scorer(self, u: int) -> Callable[[int], float]:
        # similarity of u to every user, computed once and reused for every movie
        all_users = np.arange(self.matrix.shape[0])
        sims = self.similarity.between(np.array([u]), all_users)[0]
        return lambda i: self._score(u, i, sims)

    def _score(self, u: int, i: int, sims: np.ndarray) -> float:
        """pred(u, i) given the similarities of user row `u` to all users.

        pred(u, i) = avg(u) + sum(sim(u, v) * (r(v, i) - avg(v))) / sum(|sim(u, v)|)
        over the k users v most similar to u who rated movie i.
        """
        # users who rated the movie and their ratings of it; same as matrix_csc[:, i].indices / .data,
        # but reading the CSC arrays directly is ~3x faster than slicing out a new sparse column
        start, end = self.matrix_csc.indptr[i], self.matrix_csc.indptr[i + 1]
        raters = self.matrix_csc.indices[start:end]
        ratings = self.matrix_csc.data[start:end]
        keep = raters != u
        raters, ratings = raters[keep], ratings[keep]

        # k most similar raters
        top = np.argsort(-sims[raters], kind="stable")[:self.k]
        raters, ratings = raters[top], ratings[top]
        weights = sims[raters]
        if np.abs(weights).sum() == 0:
            return np.nan

        deviations = ratings - self.means[raters]
        return float(self.means[u] + weights @ deviations / np.abs(weights).sum())


class ItemKNN(_KNN):
    def train(self, data: Dataset) -> None:
        super().train(data)
        # rows = items
        self.similarity = RowSimilarity(self.matrix.T.tocsr(), self.sim, self.gamma)
        self.means = self.similarity.means

    def _scorer(self, u: int) -> Callable[[int], float]:
        # movies the user rated and their ratings; same as matrix[u].indices / .data
        start, end = self.matrix.indptr[u], self.matrix.indptr[u + 1]
        rated = self.matrix.indices[start:end]
        ratings = self.matrix.data[start:end]

        # similarity of every movie to the movies u rated (n_items x n_rated), computed once
        # and reused for every movie; only these similarities are ever needed for u
        all_items = np.arange(self.matrix.shape[1])
        sims = self.similarity.between(all_items, rated)
        return lambda i: self._score(i, rated, ratings, sims[i])

    def _score(self, i: int, rated: np.ndarray, ratings: np.ndarray, sims: np.ndarray) -> float:
        """pred(u, i) given the movies u rated, their ratings and their similarities to movie i.

        pred(u, i) = avg(i) + sum(sim(i, j) * (r(u, j) - avg(j))) / sum(|sim(i, j)|)
        over the k movies j most similar to i that user u rated.
        """
        keep = rated != i
        rated, ratings, sims = rated[keep], ratings[keep], sims[keep]

        # k rated movies most similar to i
        top = np.argsort(-sims, kind="stable")[:self.k]
        rated, ratings, weights = rated[top], ratings[top], sims[top]
        if np.abs(weights).sum() == 0:
            return np.nan

        deviations = ratings - self.means[rated]
        return float(self.means[i] + weights @ deviations / np.abs(weights).sum())


class MostPopular(RecommendationModel):
    def __init__(self, only_positive: bool = False, positive_threshold: int = 4,
                 window_days: int | None = None) -> None:
        """Recommends the movies with the most ratings in train.

        `only_positive`: count only ratings >= `positive_threshold`.
        `window_days`: count only ratings from the last `window_days` days of train
        (counted back from the last training rating). None = all-time popularity.
        """
        super().__init__()
        self.only_positive = only_positive
        self.positive_threshold = positive_threshold
        self.window_days = window_days
        self.data: Dataset | None = None

    def train(self, data: Dataset) -> None:
        self.data = data
        ratings = data.ratings

        if self.window_days is not None:
            window_start = ratings["ts"].max() - self.window_days * 24 * 60 * 60
            ratings = ratings[ratings["ts"] > window_start]
        if self.only_positive:
            ratings = ratings[ratings["rating"] >= self.positive_threshold]

        # number of counted ratings per movie, most popular first;
        # ties are broken by movie id so the order is reproducible
        counts = ratings["item_id"].value_counts().sort_index()
        self.popularity = counts.sort_values(ascending=False, kind="stable")

        # movies each user already rated (from all of train, not only the window)
        self.seen = data.ratings.groupby("user_id")["item_id"].agg(set)

    def predict(self, user: int, movie: int) -> float:
        # Most popular can only rank
        return np.nan

    def rank(self, user: int, n: int) -> list[int]:
        """Top-n most popular movies the user has not rated yet.

        Users unknown in train get the plain top-n.
        """
        if self.data is None:
            raise Exception("Model has not been trained yet")
        seen = self.seen.get(user, set())

        recommendations = []
        for movie in self.popularity.index:
            if movie not in seen:
                recommendations.append(movie)
            if len(recommendations) == n:
                break
        return recommendations
