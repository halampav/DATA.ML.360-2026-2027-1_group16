
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



# Evaluate a recommender model

def evaluate_recommender(model, train, val, catalogue, k, threshold):
    positive = val[val["rating"] >= threshold]
    relevant = positive.groupby("user_id")["item_id"].apply(set).to_dict()

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
