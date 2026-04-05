# Active learning selection strategies for TypiClust on CIFAR-10.
#
# Strategies:
#   TypiClust - typicality + K-Means clustering (Hacohen et al., ICML 2022)
#   TypiClustCosine - same but with cosine typicality instead of Euclidean
#   RandomSelection - uniform random baseline
#   UncertaintySelection - least-confidence (lowest max softmax)
#   MarginSelection - smallest margin between top-2 softmax scores

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .typicality import compute_typicality, compute_typicality_cosine


def _unlabeled_mask(n: int, labeled_indices: np.ndarray | None) -> np.ndarray:
    # boolean mask where True = unlabeled
    mask = np.ones(n, dtype=bool)
    if labeled_indices is not None and len(labeled_indices) > 0:
        mask[labeled_indices] = False
    return mask


def _kmeans(k: int, features: np.ndarray, seed: int) -> np.ndarray:
    # KMeans for k <= 50, MiniBatchKMeans for larger k (paper section 3.2)
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


class TypiClust:
    # selects the most typical unlabeled point from the largest uncovered clusters

    MIN_CLUSTER_SIZE = 5

    def __init__(
        self,
        features: np.ndarray,
        max_clusters: int = 500,
        min_cluster_size: int = 5,
        seed: int = 42,
    ):
        self.features = features
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.seed = seed
        self._N = len(features)

    def select(
        self,
        budget_B: int,
        labeled_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        # select budget_B unlabeled samples using typicality-based clustering
        if labeled_indices is None:
            labeled_indices = np.array([], dtype=np.int64)
        labeled_indices = np.asarray(labeled_indices, dtype=np.int64)

        n_labeled = len(labeled_indices)
        budget_B  = min(budget_B, self._N - n_labeled)
        if budget_B <= 0:
            return np.array([], dtype=np.int64)

        # determine number of clusters
        K = min(n_labeled + budget_B, self.max_clusters)
        K = max(K, 1)

        cluster_labels = _kmeans(K, self.features, self.seed)
        labeled_set = set(labeled_indices.tolist())

        # group samples by cluster
        cluster_members: dict[int, list[int]] = {c: [] for c in range(K)}
        for idx in range(self._N):
            cluster_members[cluster_labels[idx]].append(idx)

        # find clusters with no labeled samples
        uncovered = []
        for c_id, members in cluster_members.items():
            if not any(m in labeled_set for m in members):
                uncovered.append((c_id, members))

        # sort by size descending so we pick from the biggest clusters first
        uncovered.sort(key=lambda x: len(x[1]), reverse=True)

        selected: list[int] = []
        skipped:  list[tuple[int, list[int]]] = []

        for c_id, members in uncovered:
            if len(selected) >= budget_B:
                break

            cluster_size = len(members)

            if cluster_size < self.min_cluster_size:
                skipped.append((c_id, members))
                continue

            k_nn = min(20, cluster_size)
            cluster_feats = self.features[members]
            scores = compute_typicality(cluster_feats, k=k_nn)

            best_local = int(np.argmax(scores))
            selected.append(members[best_local])

        # try skipped (small) clusters if we still need more
        if len(selected) < budget_B:
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

        # fallback: fill remaining slots with random unlabeled samples
        if len(selected) < budget_B:
            already = set(selected) | labeled_set
            remaining = [i for i in range(self._N) if i not in already]
            rng = np.random.default_rng(self.seed)
            n_fill = min(budget_B - len(selected), len(remaining))
            fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
            selected.extend(fill)

        return np.array(selected[:budget_B], dtype=np.int64)


class TypiClustCosine(TypiClust):
    # same as TypiClust but uses cosine similarity instead of Euclidean distance
    # for the within-cluster typicality scoring

    def _score_cluster(self, members, k_nn):
        cluster_feats = self.features[members]
        return compute_typicality_cosine(cluster_feats, k=k_nn)

    def select(
        self,
        budget_B: int,
        labeled_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        if labeled_indices is None:
            labeled_indices = np.array([], dtype=np.int64)
        labeled_indices = np.asarray(labeled_indices, dtype=np.int64)

        n_labeled = len(labeled_indices)
        budget_B  = min(budget_B, self._N - n_labeled)
        if budget_B <= 0:
            return np.array([], dtype=np.int64)

        K = min(n_labeled + budget_B, self.max_clusters)
        K = max(K, 1)

        cluster_labels = _kmeans(K, self.features, self.seed)
        labeled_set = set(labeled_indices.tolist())

        cluster_members = {c: [] for c in range(K)}
        for idx in range(self._N):
            cluster_members[cluster_labels[idx]].append(idx)

        uncovered = []
        for c_id, members in cluster_members.items():
            if not any(m in labeled_set for m in members):
                uncovered.append((c_id, members))

        uncovered.sort(key=lambda x: len(x[1]), reverse=True)

        selected = []
        skipped  = []

        for c_id, members in uncovered:
            if len(selected) >= budget_B:
                break
            cluster_size = len(members)
            if cluster_size < self.min_cluster_size:
                skipped.append((c_id, members))
                continue
            k_nn = min(20, cluster_size)
            scores = self._score_cluster(members, k_nn)
            best_local = int(np.argmax(scores))
            selected.append(members[best_local])

        if len(selected) < budget_B:
            skipped.sort(key=lambda x: len(x[1]), reverse=True)
            for c_id, members in skipped:
                if len(selected) >= budget_B:
                    break
                k_nn = min(20, len(members))
                scores = self._score_cluster(members, k_nn)
                best_local = int(np.argmax(scores))
                selected.append(members[best_local])

        if len(selected) < budget_B:
            already = set(selected) | labeled_set
            remaining = [i for i in range(self._N) if i not in already]
            rng = np.random.default_rng(self.seed)
            n_fill = min(budget_B - len(selected), len(remaining))
            fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
            selected.extend(fill)

        return np.array(selected[:budget_B], dtype=np.int64)


class RandomSelection:
    # baseline: sample budget_B unlabeled points uniformly at random

    def __init__(self, features: np.ndarray, seed: int = 42):
        self._N   = len(features)
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        budget_B: int,
        labeled_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        if labeled_indices is None:
            labeled_indices = np.array([], dtype=np.int64)
        labeled_set = set(np.asarray(labeled_indices, dtype=np.int64).tolist())

        candidates = np.array([i for i in range(self._N) if i not in labeled_set])
        budget_B   = min(budget_B, len(candidates))
        return self._rng.choice(candidates, size=budget_B, replace=False)


class UncertaintySelection:
    # least-confidence: query samples with the lowest max softmax probability

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
        if labeled_indices is None:
            labeled_indices = np.array([], dtype=np.int64)
        labeled_set = set(np.asarray(labeled_indices, dtype=np.int64).tolist())

        candidates = np.array([i for i in range(self._N) if i not in labeled_set])
        budget_B   = min(budget_B, len(candidates))

        probs        = self.predict_proba_fn(candidates)
        max_probs    = probs.max(axis=1)
        sorted_local = np.argsort(max_probs)
        return candidates[sorted_local[:budget_B]]


class MarginSelection:
    # margin sampling: query samples with smallest gap between top-2 predictions

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
        if labeled_indices is None:
            labeled_indices = np.array([], dtype=np.int64)
        labeled_set = set(np.asarray(labeled_indices, dtype=np.int64).tolist())

        candidates = np.array([i for i in range(self._N) if i not in labeled_set])
        budget_B   = min(budget_B, len(candidates))

        probs = self.predict_proba_fn(candidates)
        if probs.shape[1] >= 2:
            top2    = np.partition(probs, -2, axis=1)[:, -2:]
            margins = top2[:, 1] - top2[:, 0]
        else:
            margins = probs[:, 0]

        sorted_local = np.argsort(margins)
        return candidates[sorted_local[:budget_B]]


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
    # generic active learning loop that works with all selector classes
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
