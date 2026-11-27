"""Part 3 (Track A): MMR re-ranking on movie genres and the beyond-accuracy metrics.

Item distance everywhere: 1 - Jaccard similarity of the genre sets of two movies.
"""
import numpy as np
from collections import Counter
from itertools import combinations
from scipy.spatial.distance import jensenshannon


def mmr_rerank(model, user_id, movies, top_k=10,
               candidate_count=100, lambda_rel=0.7):
    """MMR re-ranking of the model's top-`candidate_count` movies (Lecture 5.1, slide 9).

    Picks `top_k` movies one by one; each step takes the candidate with the highest
    lambda * relevance - (1 - lambda) * max Jaccard genre similarity to the movies already picked.
    Relevance is the model score min-max scaled within the candidates. lambda = 1 keeps the model's
    ranking, lambda = 0 only maximises genre diversity. Ties go to the higher-ranked candidate.
    """

    candidates = model.rank(user_id, candidate_count)

    if not candidates:
        return []

    genre_map = dict(zip(movies["item_id"], movies["genres"]))
    genres = {i: set(genre_map.get(i, [])) for i in candidates}

    scores = [model.predict(user_id, i) for i in candidates]
    low, high = min(scores), max(scores)

    relevance = {
        i: (s - low) / (high - low) if high > low else 1.0
        for i, s in zip(candidates, scores)
    }

    def similarity(i, j):
        intersection = genres[i] & genres[j]
        union = genres[i] | genres[j]
        return len(intersection) / len(union) if union else 0.0

    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:

        def mmr_score(i):
            redundancy = max(
                (similarity(i, j) for j in selected),
                default=0.0
            )
            return lambda_rel * relevance[i] - (1 - lambda_rel) * redundancy

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return selected


# The metric functions below are the ones from notebooks/05_part3_mmr.ipynb, moved here so that
# src/part3_analysis.py can use them. The only change is that the genre map, the genre list and the
# catalogue are passed as arguments instead of being notebook globals.


def genre_sets(movies):
    """{item_id: set of genres} of every movie."""
    return {i: set(g) for i, g in zip(movies["item_id"], movies["genres"])}


def diversity(recs, genre_map):
    """Intra-list diversity: mean distance (1 - Jaccard of genres) over all pairs in one list."""
    pairs = list(combinations(recs, 2))
    if not pairs:
        return 0.0

    distances = []
    for a, b in pairs:
        x, y = genre_map[a], genre_map[b]
        similarity = len(x & y) / len(x | y) if (x | y) else 0
        distances.append(1 - similarity)

    return np.mean(distances)


def gini_exposure(recs, catalogue):
    """Gini of how often each catalogue movie is recommended (0 = equal, 1 = one movie gets all)."""
    counts = Counter(i for r in recs.values() for i in r)
    values = np.sort(np.array(
        [counts.get(i, 0) for i in catalogue], dtype=float
    ))
    n = len(values)
    if values.sum() == 0:
        return 0.0
    return (2 * np.sum(np.arange(1, n + 1) * values)
            / (n * values.sum()) - (n + 1) / n)


def genre_distribution(items, genre_map, all_genres):
    """Share of each genre in `all_genres` over a set of movies; None for an empty set."""
    counts = Counter(
        genre
        for item in items
        for genre in genre_map.get(item, set())
    )
    values = np.array([counts[g] for g in all_genres], dtype=float)
    return values / values.sum() if values.sum() else None


def calibration_error(profile, output):
    """Squared Jensen-Shannon distance (base 2) between two genre distributions; NaN if one is None."""
    if profile is None or output is None:
        return float("nan")
    return jensenshannon(profile, output, base=2) ** 2
