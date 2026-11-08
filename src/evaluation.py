
# 1. Precision@k

def precision_at_k(recommended, relevant, k):
    correct = 0

    for movie in recommended[:k]:
        if movie in relevant:
            correct += 1

    return correct / k


# 2. Recall@k

def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0

    correct = 0

    for movie in recommended[:k]:
        if movie in relevant:
            correct += 1

    return correct / len(relevant)



# 3. NDCG@k

import math

import numpy as np
from sklearn.metrics import ndcg_score, precision_score, recall_score

def ndcg_at_k(recommended, relevant, k):
    dcg = 0

    for position, movie in enumerate(recommended[:k], start=1):
        if movie in relevant:
            dcg += 1 / math.log2(position + 1)

    ideal_dcg = 0

    for position in range(1, min(len(relevant), k) + 1):
        ideal_dcg += 1 / math.log2(position + 1)

    if ideal_dcg == 0:
        return 0

    return dcg / ideal_dcg




# 4. Catalogue Coverage

def catalogue_coverage(recommendations, catalogue, k):
    unique_movies = set()

    for movie_list in recommendations.values():
        for movie in movie_list[:k]:
            if movie in catalogue:
                unique_movies.add(movie)

    if len(catalogue) == 0:
        return 0

    return len(unique_movies) / len(catalogue)


# 5. Evaluate Model

def evaluate_model(recommendations, relevant_by_user, catalogue, k):
    precision = 0
    recall = 0
    ndcg = 0
    count = 0
    valid_recommendations = {}

    for user, movies in recommendations.items():
        relevant = relevant_by_user.get(user, set())

        if len(relevant) == 0:
            continue

        top_k = movies[:k]

        if len(top_k) != len(set(top_k)):
            raise ValueError("Duplicate recommendations")

        precision += precision_at_k(movies, relevant, k)
        recall += recall_at_k(movies, relevant, k)
        ndcg += ndcg_at_k(movies, relevant, k)

        valid_recommendations[user] = top_k
        count += 1

    if count == 0:
        raise ValueError("No users to evaluate")

    return {
        "precision": precision / count,
        "recall": recall / count,
        "ndcg": ndcg / count,
        "coverage": catalogue_coverage(
            valid_recommendations, catalogue, k
        ),
        "users_evaluated": count
    }



# Relevant movies per user

def relevant_by_user(ratings, threshold):
    """{user: set of movies the user rated >= threshold} for users with at least one such movie."""
    positive = ratings[ratings["rating"] >= threshold]
    return positive.groupby("user_id")["item_id"].apply(set).to_dict()


# Evaluate a recommender model

def evaluate_recommender(model, train, val, catalogue, k, threshold):
    relevant = relevant_by_user(val, threshold)

    # Every user with at least one relevant movie is evaluated, also users with no train
    # ratings (cold start). A model that cannot recommend anything for them returns [] and
    # scores 0 for them, so the numbers show that weakness instead of hiding it.
    users = sorted(relevant)
    recs = {u: model.rank(u, k) for u in users}
    results = evaluate_model(recs, relevant, catalogue, k)

    # The same metrics separately for users known from train (warm) and new users (cold)
    train_users = set(train["user_id"])
    segments = {
        "warm": {u: r for u, r in recs.items() if u in train_users},
        "cold": {u: r for u, r in recs.items() if u not in train_users},
    }
    for segment, segment_recs in segments.items():
        if len(segment_recs) == 0:
            continue
        segment_results = evaluate_model(segment_recs, relevant, catalogue, k)
        for name in ["precision", "recall", "ndcg", "users_evaluated"]:
            results[name + "_" + segment] = segment_results[name]

    return results


# Reference rankers: a floor and a ceiling for the metrics

def reference_recommendations(history, eval_ratings, catalogue, k, threshold, seed):
    """Top-k lists of two reference rankers for every user with a relevant movie in `eval_ratings`.

    Random: k random catalogue movies the user has not rated in `history` (fixed seed).
    Perfect ranker: the user's relevant movies first, i.e. it knows the answers. Its NDCG is 1,
    but its precision is not, because many users have fewer than k relevant movies.
    """
    relevant = relevant_by_user(eval_ratings, threshold)
    seen = history.groupby("user_id")["item_id"].agg(set)
    items = np.array(sorted(catalogue))
    rng = np.random.default_rng(seed)

    random_recs = {}
    for user in sorted(relevant):
        unseen = np.setdiff1d(items, list(seen.get(user, set())))
        random_recs[user] = rng.choice(unseen, size=k, replace=False).tolist()
    perfect_recs = {user: sorted(relevant[user])[:k] for user in sorted(relevant)}
    return {"Random": random_recs, "Perfect ranker": perfect_recs}


# Check our metrics against scikit-learn

def compare_with_sklearn(recommendations, relevant_by_user, catalogue, k):
    """Largest difference between our P@k, R@k, NDCG@k and scikit-learn's, over all given users.

    scikit-learn scores items instead of taking a ranked list, so every catalogue movie gets a
    score: the recommended movies get k, k-1, ..., 1 by position and all other movies 0.
    Only users with exactly k recommendations are compared (then precision means the same in both).
    """
    position = {item: i for i, item in enumerate(sorted(catalogue))}
    differences = {"precision": [], "recall": [], "ndcg": []}
    for user, recs in recommendations.items():
        relevant = relevant_by_user.get(user, set())
        if len(relevant) == 0 or len(recs) != k:
            continue
        y_true = np.zeros(len(position))
        y_true[[position[item] for item in relevant]] = 1
        y_score = np.zeros(len(position))
        y_score[[position[item] for item in recs]] = np.arange(k, 0, -1)
        y_pred = (y_score > 0).astype(int)

        differences["precision"].append(precision_at_k(recs, relevant, k) - precision_score(y_true, y_pred))
        differences["recall"].append(recall_at_k(recs, relevant, k) - recall_score(y_true, y_pred))
        differences["ndcg"].append(ndcg_at_k(recs, relevant, k) - ndcg_score([y_true], [y_score], k=k))

    return {metric: {"users": len(diffs), "max_abs_difference": float(np.max(np.abs(diffs)))}
            for metric, diffs in differences.items()}
