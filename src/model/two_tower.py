from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.model.dataset import Dataset
from src.model.models import RecommendationModel

class UserTower(nn.Module):
    """Encodes user features to embedding of size `dim`

    Every categorical feature (user id, gender, age group, occupation) has its own embedding
    table. The embeddings are concatenated and passed through a small MLP.
    """

    def __init__(self, sizes: dict[str, int], dim: int, embedding_dim: int, hidden: int) -> None:
        """`sizes`: number of distinct values of each feature."""
        super().__init__()
        # padding_idx=0 -- users with idx 0 get no personilized embedding (for new users)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(size, embedding_dim, padding_idx=0 if name == "user_idx" else None)
            for name, size in sizes.items()})
        self.mlp = nn.Sequential(nn.Linear(len(sizes) * embedding_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """`x`: batch of user features, every value has shape [B] and dtype long:
            user_idx    0 = unknown user (no train ratings), 1..n = train users
            gender      0..1
            age         0..6   (the 7 age groups of users.dat)
            occupation  0..20
        Returns user vectors, shape [B, dim], with length 1.
        """
        embedded = [self.embeddings[name](x[name]) for name in self.embeddings]   # each [B, E]
        out = self.mlp(torch.cat(embedded, dim=1))
        # Normalize so that dot product == cosine similarity
        return F.normalize(out, dim=1)


class ItemTower(nn.Module):
    """Movie features -> movie vector of size `dim`.

    The movie id has an embedding table; the genres (multi-hot) and the year go through a linear
    layer. Everything is concatenated and passed through a small MLP.
    """

    def __init__(self, sizes: dict[str, int], dim: int, embedding_dim: int, hidden: int) -> None:
        super().__init__()
        # Movies with no train ratings (index 0) get a zero id vector: genres and year only
        self.id_embedding = nn.Embedding(sizes["item_idx"], embedding_dim, padding_idx=0)
        self.genre_layer = nn.Linear(sizes["genres"], embedding_dim)
        self.year_layer = nn.Linear(1, embedding_dim)
        self.mlp = nn.Sequential(nn.Linear(3 * embedding_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """`x`: batch of item features:
            item_idx  [B] long      0 = movie with no train ratings, 1..n = train movies
            genres    [B, G] float  multi-hot, 1 = movie has that genre
            year      [B, 1] float  release year, standardised (0 = average or unknown year)
        Returns item vectors, shape [B, dim], with length 1.
        """
        parts = [self.id_embedding(x["item_idx"]), self.genre_layer(x["genres"]),
                 self.year_layer(x["year"])]
        return F.normalize(self.mlp(torch.cat(parts, dim=1)), dim=1)


def in_batch_softmax_loss(user_vectors: torch.Tensor, item_vectors: torch.Tensor,
                          item_rows: torch.Tensor, log_q: torch.Tensor,
                          temperature: float) -> torch.Tensor:
    """Sampled softmax with in-batch negatives and logQ correction.

    user_vectors [B, dim], item_vectors [B, dim]: row b of both is one positive pair.
    item_rows [B]: which movie each pair is (to find the same movie twice in one batch).
    log_q [B]: log of the probability that movie b is sampled into a batch.
    Returns the mean loss over the batch.
    """
    # logits[b, j] = score of user b with the movie of pair j; the diagonal are the positives
    logits = user_vectors @ item_vectors.T / temperature                      # [B, B]

    # logQ correction: popular movies are in many pairs, so they show up as negatives far more
    # often than the rest. Subtracting log(sampling probability) from column j removes that
    # advantage; without it the model learns to push popular movies down.
    logits = logits - log_q[None, :]

    # A popular movie can appear in two pairs of one batch. For user b it is then a positive, not
    # a negative, so those "accidental hits" are masked out (except the diagonal itself).
    same_movie = item_rows[:, None] == item_rows[None, :]
    same_movie.fill_diagonal_(False)
    logits = logits.masked_fill(same_movie, float("-inf"))

    # Softmax over the batch for each user, the correct class of row b is column b
    targets = torch.arange(len(logits))
    return F.cross_entropy(logits, targets)


class TwoTower(RecommendationModel):
    def __init__(self, dim: int = 64, embedding_dim: int = 32, hidden: int = 128,
                 epochs: int = 10, batch_size: int = 1024, learning_rate: float = 1e-3,
                 weight_decay: float = 0.0, temperature: float = 0.1, id_dropout: float = 0.1,
                 positive_threshold: int = 4, seed: int = 42, num_threads: int = 1,
                 verbose: bool = False,
                 on_epoch_end: Callable[[int, "TwoTower"], None] | None = None) -> None:
        """
        `dim`: size of the user and movie vectors.
        `embedding_dim`: size of each feature embedding.
        `hidden`: width of the hidden layer of both MLPs.
        `temperature`: the cosine scores are divided by it before the softmax; smaller = sharper.
        `id_dropout`: share of training pairs in which the user id and the movie id are hidden
        (set to 0), so the towers also learn to work from the features alone (cold start).
        `on_epoch_end(epoch, model)` is called after every epoch with the item and user
        vectors already updated, so `model.rank` works there (e.g. to watch validation NDCG).
        """
        super().__init__()
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.id_dropout = id_dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.positive_threshold = positive_threshold
        self.seed = seed
        self.num_threads = num_threads
        self.verbose = verbose
        self.on_epoch_end = on_epoch_end
        self.data: Dataset | None = None

    # --- features ------------------------------------------------------------------------

    def _user_features(self, data: Dataset) -> dict[str, torch.Tensor]:
        """One row per user in users.dat (also users with no train ratings: cold start)."""
        users = data.users.sort_values("user_id").reset_index(drop=True)
        self.user_ids = pd.Index(users["user_id"])

        # ID index only for users that have train ratings; everyone else shares index 0
        train_users = pd.Index(sorted(data.ratings["user_id"].unique()))
        user_idx = train_users.get_indexer(users["user_id"]) + 1   # -1 (unknown) -> 0
        gender = (users["gender"] == "M").astype(int)
        age = pd.Index(sorted(users["age"].unique())).get_indexer(users["age"])

        self.user_sizes = {"user_idx": len(train_users) + 1, "gender": 2,
                           "age": int(age.max()) + 1,
                           "occupation": int(users["occupation"].max()) + 1}
        return {"user_idx": torch.tensor(user_idx, dtype=torch.long),
                "gender": torch.tensor(gender.to_numpy(), dtype=torch.long),
                "age": torch.tensor(age, dtype=torch.long),
                "occupation": torch.tensor(users["occupation"].to_numpy(), dtype=torch.long)}

    def _item_features(self, data: Dataset) -> dict[str, torch.Tensor]:
        """One row per movie in movies.dat, so the whole catalogue can be ranked."""
        movies = data.movies.sort_values("item_id").reset_index(drop=True)
        self.item_ids = pd.Index(movies["item_id"])

        train_items = pd.Index(sorted(data.ratings["item_id"].unique()))
        item_idx = train_items.get_indexer(movies["item_id"]) + 1   # -1 (unknown) -> 0

        genre_names = sorted({g for genres in movies["genres"] for g in genres})
        genres = np.zeros((len(movies), len(genre_names)), dtype=np.float32)
        for row, movie_genres in enumerate(movies["genres"]):
            for g in movie_genres:
                genres[row, genre_names.index(g)] = 1

        year = movies["year"].astype(float)
        year = ((year - year.mean()) / year.std()).fillna(0).to_numpy(dtype=np.float32)

        self.item_sizes = {"item_idx": len(train_items) + 1, "genres": len(genre_names)}
        return {"item_idx": torch.tensor(item_idx, dtype=torch.long),
                "genres": torch.tensor(genres),
                "year": torch.tensor(year).unsqueeze(1)}

    # --- training --------------------------------------------------------------------------

    def train(self, data: Dataset) -> None:
        self.data = data
        # Same seed + 1 thread + deterministic algorithms -> identical runs on CPU
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.num_threads)
        torch.use_deterministic_algorithms(True)

        self.user_features = self._user_features(data)
        self.item_features = self._item_features(data)

        # Training pairs: row numbers into the feature tables above
        r = data.ratings
        positives = r[r["rating"] >= self.positive_threshold]
        user_rows = torch.tensor(self.user_ids.get_indexer(positives["user_id"]), dtype=torch.long)
        item_rows = torch.tensor(self.item_ids.get_indexer(positives["item_id"]), dtype=torch.long)

        # logQ: batches are drawn uniformly from the pairs, so a movie lands in a batch with
        # probability proportional to its number of positive pairs
        counts = torch.bincount(item_rows, minlength=len(self.item_ids)).float()
        self.log_q = torch.log(counts.clamp(min=1) / counts.sum())

        # Movies each user already rated (all ratings, not only positives): never recommended
        self.seen = {user: self.item_ids.get_indexer(items)
                     for user, items in r.groupby("user_id")["item_id"]}

        self.user_tower = UserTower(self.user_sizes, self.dim, self.embedding_dim, self.hidden)
        self.item_tower = ItemTower(self.item_sizes, self.dim, self.embedding_dim, self.hidden)
        parameters = list(self.user_tower.parameters()) + list(self.item_tower.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        generator = torch.Generator().manual_seed(self.seed)   # fixes the shuffling order
        loader = DataLoader(TensorDataset(user_rows, item_rows), batch_size=self.batch_size,
                            shuffle=True, drop_last=True, generator=generator)

        self.losses = []   # mean training loss per epoch, e.g. for a plot
        for epoch in range(1, self.epochs + 1):
            self.user_tower.train()
            self.item_tower.train()
            total = 0.0
            for batch_users, batch_items in loader:
                user_x = {name: values[batch_users] for name, values in self.user_features.items()}
                item_x = {name: values[batch_items] for name, values in self.item_features.items()}
                user_x["user_idx"] = self._drop_ids(user_x["user_idx"])
                item_x["item_idx"] = self._drop_ids(item_x["item_idx"])

                loss = in_batch_softmax_loss(self.user_tower(user_x), self.item_tower(item_x),
                                             batch_items, self.log_q[batch_items], self.temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()

            self.losses.append(total / len(loader))
            if self.verbose:
                print(f"epoch {epoch}: loss {self.losses[-1]:.4f}")
            if self.on_epoch_end is not None:
                self._compute_vectors()
                self.on_epoch_end(epoch, self)

        self._compute_vectors()

    def _drop_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Replaces each id by 0 ("unknown") with probability `id_dropout`.

        Without this, index 0 never occurs in training, and a cold-start user or movie would get
        an input the towers have never seen.
        """
        drop = torch.rand(ids.shape) < self.id_dropout
        return ids.masked_fill(drop, 0)

    @torch.no_grad()
    def _compute_vectors(self) -> None:
        """User and item vectors for everyone, computed once (the towers are independent)."""
        self.user_tower.eval()
        self.item_tower.eval()
        self.user_vectors = self.user_tower(self.user_features).numpy()
        self.item_vectors = self.item_tower(self.item_features).numpy()

    # --- using the model -------------------------------------------------------------------

    def predict(self, user: int, movie: int) -> float:
        """Dot-product score (not a rating); NaN for users or movies outside the data."""
        if self.data is None:
            raise Exception("Model has not been trained yet")
        if user not in self.user_ids or movie not in self.item_ids:
            return np.nan
        u = self.user_ids.get_loc(user)
        i = self.item_ids.get_loc(movie)
        return float(self.user_vectors[u] @ self.item_vectors[i])

    def rank(self, user: int, n: int) -> list[int]:
        """Top-n unseen movies of the whole catalogue.

        Exact search over all movies: with ~3,900 movies it is one matrix-vector product, so the
        approximate nearest-neighbour search used for huge catalogues is not needed.
        Users with no train ratings are ranked from their demographic features.
        """
        if self.data is None:
            raise Exception("Model has not been trained yet")
        if user not in self.user_ids:
            return []
        scores = self.item_vectors @ self.user_vectors[self.user_ids.get_loc(user)]
        scores[self.seen.get(user, [])] = -np.inf
        # stable sort: equal scores are ordered by movie id, so the result is reproducible
        top = np.argsort(-scores, kind="stable")[:n]
        return self.item_ids[top].tolist()
