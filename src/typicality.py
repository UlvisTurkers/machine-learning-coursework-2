"""
Typicality scoring for TypiClust.

Typicality measures how representative a point is within its local
neighbourhood. Following Hacohen et al. (2022), the typicality of a sample x
is defined as the inverse of its average distance to its k nearest neighbours:

    typicality(x) = 1 / mean_{i in kNN(x)} || x - x_i ||

Higher typicality => more central / typical within its cluster.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality(
    features: np.ndarray,
    k: int = 20,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute typicality scores for every point in `features`.

    Args:
        features: Array of shape (N, D) containing embeddings.
        k:        Number of nearest neighbours (excluding self).
        metric:   Distance metric passed to sklearn NearestNeighbors.

    Returns:
        scores: Array of shape (N,) with typicality scores (higher = more typical).
    """
    if k >= len(features):
        k = len(features) - 1

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto", n_jobs=-1)
    nn.fit(features)
    distances, _ = nn.kneighbors(features)

    # distances[:, 0] is distance to self (= 0); exclude it.
    avg_dist = distances[:, 1:].mean(axis=1)  # (N,)

    # Avoid division by zero for duplicate points.
    avg_dist = np.where(avg_dist == 0, 1e-10, avg_dist)
    scores = 1.0 / avg_dist
    return scores


def select_typical_per_cluster(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    typicality_scores: np.ndarray,
) -> np.ndarray:
    """
    For each cluster, return the index of the most typical unlabelled point.

    Args:
        features:           Array of shape (N, D). (Unused directly; kept for
                            API symmetry with caller in active_learning.py.)
        cluster_labels:     Integer cluster assignment for each of the N points.
        typicality_scores:  Typicality score for each of the N points.

    Returns:
        selected: Array of indices (one per unique cluster) of the most typical
                  point in that cluster.
    """
    unique_clusters = np.unique(cluster_labels)
    selected = []
    for c in unique_clusters:
        mask = cluster_labels == c
        indices = np.where(mask)[0]
        best = indices[np.argmax(typicality_scores[indices])]
        selected.append(best)
    return np.array(selected)
