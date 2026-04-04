"""
Active learning selection strategies for TypiClust on CIFAR-10.

Strategies
----------
TypiClust          — Typicality + K-Means clustering (Hacohen et al., ICML 2022)
RandomSelection    — Uniform random baseline
UncertaintySelection — Least-confidence (lowest max softmax)
MarginSelection    — Smallest margin between top-2 softmax scores

All classes share the interface::

    selector = Strategy(features, ...)          # instantiate once per AL run
    indices  = selector.select(budget_B,        # call each round
                               labeled_indices)

Reference:
    Hacohen, G., Dekel, O., & Weinshall, D. (2022).
    Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.
    ICML 2022. https://arxiv.org/abs/2202.02794
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .typicality import compute_typicality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unlabeled_mask(n: int, labeled_indices: np.ndarray | None) -> np.ndarray:
    """Return a boolean mask of shape (n,): True where the sample is unlabeled."""
    mask = np.ones(n, dtype=bool)
    if labeled_indices is not None and len(labeled_indices) > 0:
        mask[labeled_indices] = False
    return mask


def _kmeans(k: int, features: np.ndarray, seed: int) -> np.ndarray:
    """
    Cluster `features` into `k` clusters.
    Uses KMeans for k ≤ 50 and MiniBatchKMeans for k > 50 (paper §3.2).

    Returns integer cluster-label array of shape (N,).
    """
    if k <= 50:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=seed,
        )
    else:
        model = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=5,
            random_state=seed,
            batch_size=max(1024, k * 4),
        )
    return model.fit_predict(features).astype(np.int32)


# ---------------------------------------------------------------------------
# TypiClust
# ---------------------------------------------------------------------------

class TypiClust:
    """
    TypiClust active learning selector (Hacohen et al., ICML 2022).

    Selects the most *typical* (highest-density) unlabeled point from each of
    the B largest clusters that contain no already-labeled examples.

    Args:
        features:     L2-normalised 512-dim numpy array of shape (N, D) for
                      **all** training samples (labeled + unlabeled).
        max_clusters: Upper bound on the number of K-Means clusters K.
                      Paper default: 500.
        min_cluster_size: Skip clusters with fewer than this many points
                      when selecting (paper uses 5).
        seed:         Random seed for K-Means.
    """

    MIN_CLUSTER_SIZE = 5

    def __init__(
        self,
        features: np.ndarray,
        max_clusters: int = 500,
        min_cluster_size: int = 5,
        seed: int = 42,
    ):
        self.features = features            # (N, D)
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.seed = seed
        self._N = len(features)

    # ------------------------------------------------------------------

    def select(
        self,
        budget_B: int,
        labeled_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Select `budget_B` unlabeled samples to query next.

        Algorithm
        ---------
        1. K = min(|labeled| + budget_B, max_clusters)
        2. K-Means on ALL features with K clusters
           (KMeans if K ≤ 50, MiniBatchKMeans otherwise)
        3. Identify *uncovered* clusters — clusters containing no labeled sample
        4. Sort uncovered clusters by size (number of members) descending
        5. For each of the B largest uncovered clusters (in order):
           a. K_nn = min(20, cluster_size)
           b. Skip if cluster_size < min_cluster_size (default 5)
           c. Compute Typicality for every point in the cluster using its
              K_nn nearest neighbours **within the cluster**
           d. Select the point with the highest typicality score
        6. If fewer than budget_B points have been selected after step 5
           (e.g. many tiny clusters were skipped), continue iterating over
           remaining uncovered clusters until budget is filled or pool exhausted
        7. If still short, fill remaining slots with random unlabeled samples

        Args:
            budget_B:        Number of samples to select.
            labeled_indices: 1-D integer array of already-labeled pool indices.
                             Pass None or empty array at the very first round.

        Returns:
            selected: np.ndarray of shape (≤ budget_B,) with pool indices.
        """
        if labeled_indices is None:
            labeled_indices = np.array([], dtype=np.int64)
        labeled_indices = np.asarray(labeled_indices, dtype=np.int64)

        n_labeled = len(labeled_indices)
        budget_B  = min(budget_B, self._N - n_labeled)
        if budget_B <= 0:
            return np.array([], dtype=np.int64)

        # ----------------------------------------------------------------
        # Step 1: determine K
        # ----------------------------------------------------------------
        K = min(n_labeled + budget_B, self.max_clusters)
        K = max(K, 1)

        # ----------------------------------------------------------------
        # Step 2: cluster ALL features
        # ----------------------------------------------------------------
        cluster_labels = _kmeans(K, self.features, self.seed)   # (N,)

        # ----------------------------------------------------------------
        # Step 3: identify uncovered clusters
        # ----------------------------------------------------------------
        labeled_set = set(labeled_indices.tolist())

        # For each cluster: collect its member indices and whether any are labeled
        cluster_members: dict[int, list[int]] = {c: [] for c in range(K)}
        for idx in range(self._N):
            cluster_members[cluster_labels[idx]].append(idx)

        uncovered = []   # list of (cluster_id, member_indices)
        for c_id, members in cluster_members.items():
            if not any(m in labeled_set for m in members):
                uncovered.append((c_id, members))

        # ----------------------------------------------------------------
        # Step 4: sort uncovered clusters by size (descending)
        # ----------------------------------------------------------------
        uncovered.sort(key=lambda x: len(x[1]), reverse=True)

        # ----------------------------------------------------------------
        # Steps 5 & 6: select the most typical point from each cluster
        # ----------------------------------------------------------------
        selected: list[int] = []
        skipped:  list[tuple[int, list[int]]] = []   # clusters below min_size

        for c_id, members in uncovered:
            if len(selected) >= budget_B:
                break

            cluster_size = len(members)

            if cluster_size < self.min_cluster_size:
                skipped.append((c_id, members))
                continue

            # K_nn = min(20, cluster_size)  — per-cluster neighbour count
            k_nn = min(20, cluster_size)

            cluster_feats = self.features[members]       # (cluster_size, D)
            scores = compute_typicality(cluster_feats, k=k_nn)

            best_local = int(np.argmax(scores))
            selected.append(members[best_local])

        # ----------------------------------------------------------------
        # Step 6 (continued): drain skipped clusters if still short
        # ----------------------------------------------------------------
        if len(selected) < budget_B:
            # Sort skipped by size desc and try them too
            skipped.sort(key=lambda x: len(x[1]), reverse=True)
            for c_id, members in skipped:
                if len(selected) >= budget_B:
                    break
                cluster_size = len(members)
                k_nn = min(20, cluster_size)
                cluster_feats = self.features[members]
                scores = compute_typicality(cluster_feats, k=k_nn)
                best_local = int(np.argmax(scores))
                selected.append(members[best_local])

        # ----------------------------------------------------------------
        # Step 7: final fallback — random unlabeled samples
        # ----------------------------------------------------------------
        if len(selected) < budget_B:
            already = set(selected) | labeled_set
            remaining = [i for i in range(self._N) if i not in already]
            rng = np.random.default_rng(self.seed)
            n_fill = min(budget_B - len(selected), len(remaining))
            fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
            selected.extend(fill)

        return np.array(selected[:budget_B], dtype=np.int64)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomSelection:
    """
    Baseline: sample `budget_B` unlabeled points uniformly at random.

    Args:
        features: Feature array of shape (N, D).  Only N is used.
        seed:     Random seed.
    """

    def __init__(self, features: np.ndarray, seed: int = 42):
        self._N   = len(features)
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        budget_B: int,
        labeled_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Args:
            budget_B:        Number of samples to select.
            labeled_indices: Already-labeled pool indices (excluded).

        Returns:
            selected: np.ndarray of shape (budget_B,).
        """
        if labeled_indices is None:
            labeled_indices = np.array([], dtype=np.int64)
        labeled_set = set(np.asarray(labeled_indices, dtype=np.int64).tolist())

        candidates = np.array([i for i in range(self._N) if i not in labeled_set])
        budget_B   = min(budget_B, len(candidates))
        return self._rng.choice(candidates, size=budget_B, replace=False)


# ---------------------------------------------------------------------------
# Uncertainty sampling (least-confidence)
# ---------------------------------------------------------------------------

class UncertaintySelection:
    """
    Least-confidence active learning: query the unlabeled points for which
    the classifier is least confident (lowest max softmax probability).

    Args:
        features:        Feature array of shape (N, D).  Used to identify
                         the unlabeled pool; actual predictions are obtained
                         via `predict_proba_fn`.
        predict_proba_fn: Callable that takes a 1-D array of pool indices and
                         returns a 2-D float array of shape (len(indices), C)
                         containing class probabilities (rows sum to 1).
                         Update this attribute between rounds as the model
                         improves::

                             selector.predict_proba_fn = new_fn

    Example::

        def proba_fn(indices):
            X = torch.tensor(train_feats[indices])
            with torch.no_grad():
                logits = model(X.to(device))
            return torch.softmax(logits, dim=1).cpu().numpy()

        selector = UncertaintySelection(train_feats, proba_fn)
    """

    def __init__(
        self,
        features: np.ndarray,
        predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    ):
        self._N              = len(features)
        self.predict_proba_fn = predict_proba_fn

    def select(
        self,
        budget_B: int,
        labeled_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Args:
            budget_B:        Number of samples to select.
            labeled_indices: Already-labeled pool indices (excluded).

        Returns:
            selected: np.ndarray of shape (budget_B,) — the `budget_B` unlabeled
                      indices with the lowest max softmax probability.
        """
        if labeled_indices is None:
            labeled_indices = np.array([], dtype=np.int64)
        labeled_set = set(np.asarray(labeled_indices, dtype=np.int64).tolist())

        candidates = np.array([i for i in range(self._N) if i not in labeled_set])
        budget_B   = min(budget_B, len(candidates))

        probs        = self.predict_proba_fn(candidates)  # (M, C)
        max_probs    = probs.max(axis=1)                  # (M,)
        # Least confident = smallest max probability
        sorted_local = np.argsort(max_probs)              # ascending
        return candidates[sorted_local[:budget_B]]


# ---------------------------------------------------------------------------
# Margin sampling
# ---------------------------------------------------------------------------

class MarginSelection:
    """
    Margin sampling: query the unlabeled points with the smallest difference
    between the top-1 and top-2 softmax probabilities.

    Args:
        features:        Feature array of shape (N, D).
        predict_proba_fn: Same spec as in UncertaintySelection.
    """

    def __init__(
        self,
        features: np.ndarray,
        predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    ):
        self._N              = len(features)
        self.predict_proba_fn = predict_proba_fn

    def select(
        self,
        budget_B: int,
        labeled_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Args:
            budget_B:        Number of samples to select.
            labeled_indices: Already-labeled pool indices (excluded).

        Returns:
            selected: np.ndarray of shape (budget_B,) — the `budget_B` unlabeled
                      indices with the smallest top-1 minus top-2 probability gap.
        """
        if labeled_indices is None:
            labeled_indices = np.array([], dtype=np.int64)
        labeled_set = set(np.asarray(labeled_indices, dtype=np.int64).tolist())

        candidates = np.array([i for i in range(self._N) if i not in labeled_set])
        budget_B   = min(budget_B, len(candidates))

        probs      = self.predict_proba_fn(candidates)     # (M, C)
        # Partition to get top-2 efficiently (avoids full sort of C classes)
        if probs.shape[1] >= 2:
            top2       = np.partition(probs, -2, axis=1)[:, -2:]   # (M, 2)
            margins    = top2[:, 1] - top2[:, 0]                    # top1 - top2
        else:
            margins    = probs[:, 0]                                 # single class

        # Smallest margin = most uncertain
        sorted_local = np.argsort(margins)                  # ascending
        return candidates[sorted_local[:budget_B]]


# ---------------------------------------------------------------------------
# Active learning loop
# ---------------------------------------------------------------------------

def run_active_learning_loop(
    features: np.ndarray,
    labels: np.ndarray,
    selector: TypiClust | RandomSelection | UncertaintySelection | MarginSelection,
    initial_budget: int,
    query_budget: int,
    n_rounds: int,
    train_fn: Callable,
    eval_fn: Callable,
    on_model_update: Callable | None = None,
    seed: int = 42,
) -> dict:
    """
    Generic active learning loop compatible with all selector classes.

    Args:
        features:        SimCLR features for the training pool, shape (N, D).
        labels:          Ground-truth labels, shape (N,).
        selector:        Any selector instance with a
                         ``select(budget_B, labeled_indices)`` method.
        initial_budget:  Random samples to label before round 1.
        query_budget:    Samples queried per round.
        n_rounds:        Total number of active learning rounds.
        train_fn:        ``(labeled_idx: ndarray, labels: ndarray) -> model``
        eval_fn:         ``(model) -> float``  — returns test accuracy.
        on_model_update: Optional callback ``(model, labeled_idx) -> None``
                         called after each ``train_fn``.  Useful for updating
                         ``predict_proba_fn`` on Uncertainty/Margin selectors::

                             def update(model, idx):
                                 selector.predict_proba_fn = make_proba_fn(model)

        seed:            Seed for the initial random selection.

    Returns:
        history: dict with keys ``'labelled_counts'`` and ``'accuracies'``.
    """
    rng     = np.random.default_rng(seed)
    labeled = rng.choice(len(features), size=initial_budget, replace=False).tolist()

    history: dict[str, list] = {"labelled_counts": [], "accuracies": []}

    for round_idx in range(n_rounds):
        labeled_arr = np.array(labeled, dtype=np.int64)

        model = train_fn(labeled_arr, labels[labeled_arr])
        acc   = eval_fn(model)

        history["labelled_counts"].append(len(labeled))
        history["accuracies"].append(float(acc))
        print(
            f"Round {round_idx + 1:>2d}/{n_rounds} | "
            f"labeled={len(labeled):5d} | acc={acc:.4f}"
        )

        if on_model_update is not None:
            on_model_update(model, labeled_arr)

        if round_idx < n_rounds - 1:
            new_idx = selector.select(query_budget, labeled_indices=labeled_arr)
            labeled.extend(int(i) for i in new_idx)

    return history
