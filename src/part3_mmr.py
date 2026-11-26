def mmr_rerank(model, user_id, movies, top_k=10,
               candidate_count=100, lambda_rel=0.7):

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
