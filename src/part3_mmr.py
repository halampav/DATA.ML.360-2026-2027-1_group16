"""Part 3, Track A: MMR re-ranking on genre vectors and the beyond-accuracy metrics.

Item representation: the binary genre vector of a movie (18 MovieLens genres, 1 = the movie has
that genre). Item distance: 1 - Jaccard similarity of two genre vectors, i.e.
1 - |genres in common| / |genres of either movie|. The same distance is used inside MMR and in the
intra-list diversity metric.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def genre_vectors(movies: pd.DataFrame) -> pd.DataFrame:
    """Binary genre vector of every movie: one row per item_id, one 0/1 column per genre."""
    genres = sorted({g for movie_genres in movies["genres"] for g in movie_genres})
    rows = [[int(g in movie_genres) for g in genres] for movie_genres in movies["genres"]]
    return pd.DataFrame(rows, index=movies["item_id"].to_numpy(), columns=genres)


def jaccard_similarity(vectors: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard similarity of binary vectors (one row per movie).

    |A and B| is the dot product of two 0/1 vectors, |A or B| = |A| + |B| - |A and B|.
    """
    v = np.asarray(vectors, dtype=float)
    common = v @ v.T
    sizes = v.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - common
    return np.divide(common, union, out=np.zeros_like(common), where=union > 0)


def mmr_select(scores, similarity: np.ndarray, k: int, lambda_rel: float) -> list[int]:
    """Greedy Maximal Marginal Relevance (Carbonell & Goldstein 1998; Lecture 5.1, slide 9).

    `scores` are the base model's scores of the candidates (in the base model's order) and
    `similarity` their pairwise similarity. The scores are min-max scaled to [0, 1] so that they are
    on the same scale as the similarity. Each step adds the candidate i with the highest

        lambda * relevance(i) - (1 - lambda) * max similarity(i, already selected)

    lambda = 1 keeps the base ranking, lambda = 0 only maximises diversity. Ties go to the candidate
    the base model ranked higher (np.argmax returns the first maximum).
    Returns the positions of the selected candidates, in the order they were picked.
    """
    scores = np.asarray(scores, dtype=float)
    low, high = scores.min(), scores.max()
    relevance = (scores - low) / (high - low) if high > low else np.ones_like(scores)

    redundancy = np.zeros(len(scores))     # max similarity to the selected items, 0 at the start
    available = np.ones(len(scores), dtype=bool)
    selected = []
    for _ in range(min(k, len(scores))):
        mmr = lambda_rel * relevance - (1 - lambda_rel) * redundancy
        mmr[~available] = -np.inf
        best = int(np.argmax(mmr))
        selected.append(best)
        available[best] = False
        redundancy = np.maximum(redundancy, similarity[best])
    return selected


def candidate_pool(model, user: int, size: int, vectors: pd.DataFrame):
    """The base model's top-`size` movies for `user`, their scores and their genre similarity.

    Returns ([], empty, empty) for users the model cannot rank (no training history).
    """
    items = model.rank(user, size)
    if not items:
        return [], np.empty(0), np.empty((0, 0))
    scores = np.array([model.predict(user, i) for i in items])
    return items, scores, jaccard_similarity(vectors.loc[items].to_numpy())


def mmr_rerank(model, user: int, vectors: pd.DataFrame, k: int = 10,
               candidates: int = 100, lambda_rel: float = 0.7) -> list[int]:
    """Top-k list for `user`: MMR over the base model's top-`candidates` movies.

    MMR can only reorder this pool, so it never recommends a movie outside the base top-`candidates`.
    """
    items, scores, similarity = candidate_pool(model, user, candidates, vectors)
    return [items[p] for p in mmr_select(scores, similarity, k, lambda_rel)] if items else []


# ---------------------------------------------------------------- beyond-accuracy metrics (slide 7)

def intra_list_diversity(items: list[int], vectors: pd.DataFrame) -> float:
    """Mean item distance (1 - Jaccard of genre vectors) over all pairs in one list.

    NaN for lists with fewer than 2 movies (cold users get an empty list from ALS).
    """
    if len(items) < 2:
        return float("nan")
    similarity = jaccard_similarity(vectors.loc[items].to_numpy())
    upper = np.triu_indices(len(items), k=1)
    return float(np.mean(1 - similarity[upper]))


def genre_distribution(items, vectors: pd.DataFrame) -> np.ndarray | None:
    """Share of every genre in a set of movies; None for an empty set.

    Every movie has weight 1, split equally over its genres (as in Steck 2018), so a movie with
    four genres does not count four times as much as a movie with one.
    """
    if len(items) == 0:
        return None
    rows = vectors.loc[list(items)].to_numpy(dtype=float)
    per_movie = rows / rows.sum(axis=1, keepdims=True)
    total = per_movie.sum(axis=0)
    return total / total.sum()


def calibration_error(profile: np.ndarray | None, items: list[int], vectors: pd.DataFrame) -> float:
    """How far the genre mix of a list is from the user's own genre mix.

    `profile` is the genre distribution of the movies the user liked in the training history.
    We use the Jensen-Shannon divergence (base 2, so it lies in [0, 1]); unlike the KL divergence
    of Steck 2018 it is defined when the list misses a genre the user likes, so no smoothing is
    needed. 0 = same genre mix. NaN if the user has no liked movies or the list is empty.
    """
    recommended = genre_distribution(items, vectors)
    if profile is None or recommended is None:
        return float("nan")
    return float(jensenshannon(profile, recommended, base=2) ** 2)


def gini_index(recommendations: dict, catalogue: set[int], k: int) -> float:
    """Gini coefficient of how often each catalogue movie appears in the top-k lists.

    0 = every movie is recommended equally often, 1 = all recommendations go to one movie.
    Movies never recommended count as 0, so the whole catalogue is included.
    """
    counts = Counter(i for items in recommendations.values() for i in items[:k])
    values = np.sort(np.array([counts.get(i, 0) for i in catalogue], dtype=float))
    n = len(values)
    if values.sum() == 0:
        return 0.0
    return float(2 * np.sum(np.arange(1, n + 1) * values) / (n * values.sum()) - (n + 1) / n)
