"""
Typicality scoring for TypiClust.

Original (Euclidean):
    Typicality(x) = ( 1/K * Σ_{x_i ∈ KNN(x)} ||x - x_i||_2 )^(-1)

Modified (Cosine):
    Typicality_cosine(x) = 1/K * Σ_{x_i ∈ KNN(x)} cos_sim(x, x_i)

In both variants K = min(20, cluster_size) and KNN is computed *within* the
cluster's feature vectors.

Reference:
    Hacohen, G., Dekel, O., & Weinshall, D. (2022).
    Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.
    ICML 2022. https://arxiv.org/abs/2202.02794
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Original: Euclidean inverse-mean-distance
# ---------------------------------------------------------------------------

def compute_typicality(features: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Typicality(x) = ( mean_{i in KNN(x)} ||x - x_i||_2 )^(-1)

    KNN uses Euclidean distance; computed within the provided feature set
    (i.e. per-cluster when called from TypiClust).

    Args:
        features: Array of shape (N, D).
        k:        Neighbour count K (clamped to N-1 automatically).

    Returns:
        scores: Shape (N,), float64.  Higher = more typical.
    """
    N = len(features)
    if N < 2:
        return np.ones(N, dtype=np.float64)

    k = min(k, N - 1)

    nn = NearestNeighbors(
        n_neighbors=k + 1,          # +1: kneighbors includes the point itself
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )
    nn.fit(features)
    distances, _ = nn.kneighbors(features)

    # Column 0 is the self-distance (0.0); skip it.
    avg_dist = distances[:, 1:].mean(axis=1)
    avg_dist = np.where(avg_dist == 0.0, 1e-10, avg_dist)

    return 1.0 / avg_dist


# ---------------------------------------------------------------------------
# Modified: Cosine mean-similarity
# ---------------------------------------------------------------------------

def compute_typicality_cosine(features: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Typicality_cosine(x) = mean_{i in KNN(x)} cos_sim(x, x_i)

    KNN is found using **cosine distance** (= 1 - cosine_similarity), so the
    nearest neighbours are the most similar points.  The typicality score is
    then the *mean cosine similarity* to those neighbours — a value in (-1, 1]
    where higher means more typical.

    For L2-normalised features this is equivalent to the mean dot-product with
    the K nearest neighbours.

    Args:
        features: Array of shape (N, D).
        k:        Neighbour count K (clamped to N-1 automatically).

    Returns:
        scores: Shape (N,), float64.  Higher = more typical.
    """
    N = len(features)
    if N < 2:
        return np.ones(N, dtype=np.float64)

    k = min(k, N - 1)

    # sklearn cosine metric = 1 - cosine_similarity
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",           # 'auto' may fall back to brute for cosine
        n_jobs=-1,
    )
    nn.fit(features)
    cosine_distances, _ = nn.kneighbors(features)   # values in [0, 2]

    # Convert distance back to similarity and drop the self-neighbour (col 0).
    cosine_sims = 1.0 - cosine_distances[:, 1:]     # shape (N, k), values in [-1, 1]

    # Mean cosine similarity to K nearest neighbours.
    scores = cosine_sims.mean(axis=1)                # (N,)

    return scores
