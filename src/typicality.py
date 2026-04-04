"""
Typicality scoring for TypiClust.

Typicality(x) = ( 1/K * Σ_{x_i ∈ KNN(x)} ||x - x_i||_2 )^(-1)

where K = min(20, cluster_size) and KNN is computed using Euclidean distance
within the cluster's feature vectors.

Reference:
    Hacohen, G., Dekel, O., & Weinshall, D. (2022).
    Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.
    ICML 2022. https://arxiv.org/abs/2202.02794
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality(features: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Compute typicality scores for all points in `features` using KNN within
    that feature set.

    Typicality(x) = ( mean_{i in KNN(x)} ||x - x_i||_2 )^(-1)

    The neighbours are restricted to the provided `features` array — when
    called per-cluster this means KNN is computed inside the cluster only.

    Args:
        features: Array of shape (N, D) — the points to score.
        k:        Number of nearest neighbours K.  Automatically clamped to
                  min(k, N-1) so the function is safe for small clusters.

    Returns:
        scores: Array of shape (N,), dtype float64.  Higher = more typical.
    """
    N = len(features)

    # Need at least 2 points to compute a meaningful neighbour distance.
    if N < 2:
        return np.ones(N, dtype=np.float64)

    # K = min(requested_k, cluster_size - 1)  — paper §3.2
    k = min(k, N - 1)

    nn = NearestNeighbors(
        n_neighbors=k + 1,   # +1 because kneighbors includes the point itself
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )
    nn.fit(features)
    distances, _ = nn.kneighbors(features)

    # Column 0 is the self-distance (= 0.0); skip it.
    avg_dist = distances[:, 1:].mean(axis=1)           # shape (N,)

    # Guard against duplicate points (avg_dist == 0 → infinite typicality).
    avg_dist = np.where(avg_dist == 0.0, 1e-10, avg_dist)

    return 1.0 / avg_dist
