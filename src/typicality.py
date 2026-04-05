# Typicality scoring for TypiClust.
#
# Original (Euclidean):
#   Typicality(x) = (1/K * sum of ||x - x_i||_2 for KNN(x))^(-1)
#
# Modified (Cosine):
#   Typicality_cosine(x) = 1/K * sum of cos_sim(x, x_i) for KNN(x)
#
# K = min(20, cluster_size), KNN computed within the cluster.
# Reference: Hacohen et al. (2022), ICML - Active Learning on a Budget.

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality(features: np.ndarray, k: int = 20) -> np.ndarray:
    # inverse of mean Euclidean distance to K nearest neighbours
    N = len(features)
    if N < 2:
        return np.ones(N, dtype=np.float64)

    k = min(k, N - 1)

    nn = NearestNeighbors(
        n_neighbors=k + 1,          # +1 because kneighbors includes the point itself
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )
    nn.fit(features)
    distances, _ = nn.kneighbors(features)

    # column 0 is self-distance (0.0), skip it
    avg_dist = distances[:, 1:].mean(axis=1)
    avg_dist = np.where(avg_dist == 0.0, 1e-10, avg_dist)

    return 1.0 / avg_dist


def compute_typicality_cosine(features: np.ndarray, k: int = 20) -> np.ndarray:
    # mean cosine similarity to K nearest neighbours (found via cosine distance)
    N = len(features)
    if N < 2:
        return np.ones(N, dtype=np.float64)

    k = min(k, N - 1)

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(features)
    cosine_distances, _ = nn.kneighbors(features)

    # convert distance back to similarity, drop self-neighbour
    cosine_sims = 1.0 - cosine_distances[:, 1:]
    scores = cosine_sims.mean(axis=1)

    return scores
