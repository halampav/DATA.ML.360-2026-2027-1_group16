# 1. Precision@10
def precision_at_10(recommended, relevant):
    correct = 0

    for movie in recommended[:10]:
        if movie in relevant:
            correct += 1

    return correct / 10

# 2. Recall@10
def recall_at_10(recommended, relevant):
    if len(relevant) == 0:
        return 0

    correct = 0

    for movie in recommended[:10]:
        if movie in relevant:
            correct += 1

    return correct / len(relevant)


# 3. NDCG@10

import math

def ndcg_at_10(recommended, relevant):
    dcg = 0

    for position, movie in enumerate(recommended[:10], start=1):
        if movie in relevant:
            dcg += 1 / math.log2(position + 1)

    ideal_dcg = 0

    for position in range(1, min(len(relevant), 10) + 1):
        ideal_dcg += 1 / math.log2(position + 1)

    if ideal_dcg == 0:
        return 0
      
    return dcg / ideal_dcg


# 4. Catalogue Coverage

def catalogue_coverage(recommendations, catalogue):
    unique_movies = set()

    for movie_list in recommendations.values():
        for movie in movie_list[:10]:
            if movie in catalogue:
                unique_movies.add(movie)

    return len(unique_movies) / len(catalogue)

