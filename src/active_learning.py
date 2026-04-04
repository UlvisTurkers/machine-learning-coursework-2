"""
TypiClust (TPC_RP) active learning selection strategy.

Algorithm (Hacohen et al., ICML 2022):
  1. Extract SimCLR features for the unlabelled pool.
  2. Apply a random projection to reduce dimensionality.
  3. Run k-means with k = budget to cluster the projected features.
  4. Within each cluster select the most typical point.
  5. Return those indices for labelling.

Reference:
    Hacohen, G., Dekel, O., & Weinshall, D. (2022).
    Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.
    ICML 2022. https://arxiv.org/abs/2202.02794
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

from .typicality import compute_typicality, select_typical_per_cluster


# ---------------------------------------------------------------------------
# Random projection helper
# ---------------------------------------------------------------------------

def random_projection(
    features: np.ndarray,
    out_dim: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Project `features` (N, D) -> (N, out_dim) using a Gaussian random matrix.

    The projection matrix is drawn from N(0, 1/out_dim) so that the expected
    pairwise distances are approximately preserved (Johnson-Lindenstrauss).

    Args:
        features: Input array of shape (N, D).
        out_dim:  Target dimensionality after projection.
        seed:     Random seed for reproducibility.

    Returns:
        Projected array of shape (N, out_dim).
    """
    rng = np.random.default_rng(seed)
    D = features.shape[1]
    W = rng.standard_normal((D, out_dim)) / np.sqrt(out_dim)
    return features @ W


# ---------------------------------------------------------------------------
# Core TPC_RP selector
# ---------------------------------------------------------------------------

class TypiClustSelector:
    """
    TPC_RP active learning selector.

    Args:
        proj_dim:    Dimensionality to project features to before clustering.
                     Set to None to skip random projection.
        knn_k:       Number of neighbours for typicality computation.
        kmeans_init: Number of k-means initialisations ('k-means++' or int).
        seed:        Global random seed for reproducibility.
    """

    def __init__(
        self,
        proj_dim: int | None = 64,
        knn_k: int = 20,
        kmeans_init: str | int = "k-means++",
        seed: int = 42,
    ):
        self.proj_dim = proj_dim
        self.knn_k = knn_k
        self.kmeans_init = kmeans_init
        self.seed = seed

    def select(
        self,
        features: np.ndarray,
        budget: int,
        already_labelled: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Select `budget` indices from `features` to query next.

        Args:
            features:         Array of shape (N, D) for the *entire* unlabelled
                              pool (after SimCLR feature extraction).
            budget:           Number of samples to select this round.
            already_labelled: Boolean or index array of already-labelled samples
                              to exclude from selection. If None, all samples
                              are considered unlabelled.

        Returns:
            selected_indices: Array of shape (budget,) with pool indices.
        """
        N = len(features)

        # Build mask of candidates (unlabelled samples only).
        if already_labelled is not None:
            labelled_mask = np.zeros(N, dtype=bool)
            labelled_mask[already_labelled] = True
        else:
            labelled_mask = np.zeros(N, dtype=bool)

        candidate_idx = np.where(~labelled_mask)[0]
        candidate_feats = features[candidate_idx]  # (M, D)

        # Clamp budget to available candidates.
        budget = min(budget, len(candidate_idx))

        # ---------------------------------------------------------------
        # Step 1: optional random projection
        # ---------------------------------------------------------------
        if self.proj_dim is not None and self.proj_dim < candidate_feats.shape[1]:
            projected = random_projection(candidate_feats, self.proj_dim, seed=self.seed)
        else:
            projected = candidate_feats

        # ---------------------------------------------------------------
        # Step 2: k-means clustering (k = budget)
        # ---------------------------------------------------------------
        if budget <= 10:
            kmeans = KMeans(
                n_clusters=budget,
                init=self.kmeans_init,
                n_init=10,
                random_state=self.seed,
            )
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=budget,
                init=self.kmeans_init,
                n_init=5,
                random_state=self.seed,
                batch_size=max(256, budget * 2),
            )
        cluster_labels = kmeans.fit_predict(projected)

        # ---------------------------------------------------------------
        # Step 3: typicality scoring in projected space
        # ---------------------------------------------------------------
        k_eff = min(self.knn_k, len(projected) - 1)
        scores = compute_typicality(projected, k=k_eff)

        # ---------------------------------------------------------------
        # Step 4: pick most typical point per cluster
        # ---------------------------------------------------------------
        local_selected = select_typical_per_cluster(projected, cluster_labels, scores)

        # Map local candidate indices back to original pool indices.
        selected_indices = candidate_idx[local_selected]
        return selected_indices


# ---------------------------------------------------------------------------
# Active learning loop
# ---------------------------------------------------------------------------

def run_active_learning_loop(
    features: np.ndarray,
    labels: np.ndarray,
    selector: TypiClustSelector,
    initial_budget: int,
    query_budget: int,
    n_rounds: int,
    train_fn,
    eval_fn,
    seed: int = 42,
) -> dict:
    """
    Run the full active learning loop.

    Args:
        features:       SimCLR features for the *training* pool, shape (N, D).
        labels:         Ground-truth labels for the training pool, shape (N,).
        selector:       TypiClustSelector instance.
        initial_budget: Number of random samples to label before round 1.
        query_budget:   Number of samples to query per round.
        n_rounds:       Total number of active learning rounds.
        train_fn:       Callable(labelled_indices, labels) -> model.
        eval_fn:        Callable(model) -> float (test accuracy).
        seed:           Seed for initial random selection.

    Returns:
        history: dict with keys 'labelled_counts' and 'accuracies'.
    """
    rng = np.random.default_rng(seed)
    labelled = rng.choice(len(features), size=initial_budget, replace=False).tolist()

    history = {"labelled_counts": [], "accuracies": []}

    for round_idx in range(n_rounds):
        model = train_fn(np.array(labelled), labels[labelled])
        acc = eval_fn(model)

        history["labelled_counts"].append(len(labelled))
        history["accuracies"].append(acc)
        print(f"Round {round_idx + 1}/{n_rounds} | labelled={len(labelled):4d} | acc={acc:.4f}")

        if round_idx < n_rounds - 1:
            new_idx = selector.select(
                features,
                budget=query_budget,
                already_labelled=np.array(labelled),
            )
            labelled.extend(new_idx.tolist())

    return history
