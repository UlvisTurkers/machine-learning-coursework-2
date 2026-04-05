# End-to-end validation of the TPC_RP active learning pipeline.
# Run from repo root: python -m tests.test_pipeline
# Prints PASS/FAIL for each test. Exits with code 1 if any test fails.

from __future__ import annotations

import sys
import os
import time
import traceback

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from pathlib import Path

_results: list[tuple[str, bool, str]] = []


class _Subset:
    # lightweight wrapper to truncate a dataset to the first N samples
    def __init__(self, ds, n):
        self.ds = ds
        self.n = n
        self.targets = ds.targets[:n] if hasattr(ds, "targets") else None

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.ds[idx]


def run_test(name, fn):
    print("\n" + "=" * 60)
    print("TEST: {}".format(name))
    print("=" * 60)
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print("  -> PASS  ({:.1f}s)".format(dt))
        _results.append((name, True, ""))
    except Exception as e:
        dt = time.time() - t0
        detail = traceback.format_exc()
        print("  -> FAIL  ({:.1f}s)".format(dt))
        print(detail)
        _results.append((name, False, str(e)))


# 1. SimCLR feature extraction

def test_simclr_features():
    from src.simclr import SimCLRModel, get_features
    from src.utils import load_cifar10

    model = SimCLRModel()
    train_ds, _ = load_cifar10(root="data")

    subset = _Subset(train_ds, 100)
    feats, labels = get_features(model, subset, batch_size=50,
                                 device=torch.device("cpu"), num_workers=0)

    assert feats.shape == (100, 512), \
        "Expected shape (100, 512), got {}".format(feats.shape)
    print("  Feature shape: {} -- OK".format(feats.shape))

    norms = np.linalg.norm(feats, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), \
        "Norms not ~1.0: min={:.4f}, max={:.4f}".format(norms.min(), norms.max())
    print("  Norms: min={:.6f}  max={:.6f}  mean={:.6f} -- OK".format(
        norms.min(), norms.max(), norms.mean()))

    assert labels.dtype == np.int64, "Labels dtype: {}".format(labels.dtype)
    assert labels.min() >= 0 and labels.max() <= 9
    print("  Labels: dtype={}, range=[{}, {}] -- OK".format(
        labels.dtype, labels.min(), labels.max()))


# 2. Typicality computation

def test_typicality():
    from src.typicality import compute_typicality, compute_typicality_cosine

    # dense cluster at origin + isolated outlier far away
    rng = np.random.default_rng(42)
    dense_points = rng.normal(loc=0.0, scale=0.1, size=(30, 2))
    sparse_point = np.array([[10.0, 10.0]])
    features = np.vstack([dense_points, sparse_point]).astype(np.float32)

    scores = compute_typicality(features, k=10)

    outlier_score = scores[30]
    best_dense_score = scores[:30].max()
    assert outlier_score < best_dense_score, \
        "Outlier score ({:.4f}) should be < dense cluster max ({:.4f})".format(
            outlier_score, best_dense_score)
    print("  Densest point typicality ({:.4f}) > outlier ({:.6f}) -- OK".format(
        best_dense_score, outlier_score))

    # with only 5 points, k=20 should clamp to k=4
    small = rng.normal(size=(5, 2)).astype(np.float32)
    scores_small = compute_typicality(small, k=20)
    assert scores_small.shape == (5,), \
        "Expected (5,), got {}".format(scores_small.shape)
    assert np.all(np.isfinite(scores_small)), "Non-finite scores"
    print("  K_nn capping with N=5, k=20 -> scores shape {} -- OK".format(
        scores_small.shape))

    # single-point edge case
    single = np.array([[1.0, 2.0]])
    scores_single = compute_typicality(single, k=20)
    assert scores_single.shape == (1,), "Single-point shape wrong"
    assert np.isfinite(scores_single[0]), "Single-point score not finite"
    print("  Single-point edge case -- OK")

    # cosine variant
    normed = features / np.linalg.norm(features, axis=1, keepdims=True)
    cos_scores = compute_typicality_cosine(normed, k=10)
    assert cos_scores.shape == (31,)
    assert np.all(np.isfinite(cos_scores)), "Non-finite cosine scores"
    assert cos_scores[:30].mean() > cos_scores[30], \
        "Dense cluster cosine scores should exceed outlier"
    print("  Cosine typicality: dense mean={:.4f} > outlier={:.4f} -- OK".format(
        cos_scores[:30].mean(), cos_scores[30]))


# 3. TypiClust selection

def test_typiclust_selection():
    from src.active_learning import TypiClust, TypiClustCosine

    # 3 tight clusters in 10-dim space
    rng = np.random.default_rng(42)
    N_PER_CLUSTER = 100
    D = 10
    centres = np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    clusters = []
    for c in centres:
        pts = rng.normal(loc=c, scale=0.3, size=(N_PER_CLUSTER, D)).astype(np.float32)
        clusters.append(pts)
    features = np.vstack(clusters)

    B = 3

    selector = TypiClust(features, max_clusters=500, min_cluster_size=5, seed=42)
    selected = selector.select(B, labeled_indices=None)

    assert len(selected) == B, \
        "Expected {} indices, got {}".format(B, len(selected))
    print("  Returned exactly B={} indices -- OK".format(B))

    assert len(set(selected.tolist())) == B, "Duplicate indices!"
    print("  No duplicate indices -- OK")

    # check that selected indices span all 3 clusters
    cluster_ids = set()
    for idx in selected:
        cluster_ids.add(idx // N_PER_CLUSTER)
    assert len(cluster_ids) == B, \
        "Selected from {} clusters, expected {}. IDs: {}".format(
            len(cluster_ids), B, cluster_ids)
    print("  Selected from {} distinct clusters -- OK".format(len(cluster_ids)))

    # already-labeled points should be excluded
    labeled = np.array([0, 100, 200], dtype=np.int64)
    selected2 = selector.select(B, labeled_indices=labeled)
    overlap = set(selected2.tolist()) & set(labeled.tolist())
    assert len(overlap) == 0, \
        "Selected indices overlap with labeled: {}".format(overlap)
    print("  Already-labeled exclusion -- OK")

    # cosine variant should also return the right number of unique indices
    normed = features / np.linalg.norm(features, axis=1, keepdims=True)
    cos_sel = TypiClustCosine(normed, max_clusters=500, min_cluster_size=5, seed=42)
    cos_selected = cos_sel.select(B, labeled_indices=None)
    assert len(cos_selected) == B, "Cosine variant: wrong count"
    assert len(set(cos_selected.tolist())) == B, "Cosine variant: duplicates"
    print("  TypiClustCosine: {} unique indices -- OK".format(B))


# 4. Classifier training

def test_classifier():
    from src.classifier import CIFARClassifier
    from src.utils import load_cifar10

    train_ds, test_ds = load_cifar10(root="data")
    device = torch.device("cpu")

    clf = CIFARClassifier(num_classes=10, device=device, seed=42, num_workers=0)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(train_ds), size=50, replace=False).astype(np.int64)

    hist = clf.train(indices, train_ds, epochs=5, batch_size=32,
                     seed=42, verbose=False)

    assert "train_loss" in hist and len(hist["train_loss"]) == 5
    assert "train_acc" in hist and len(hist["train_acc"]) == 5
    print("  Training returned 5-epoch history -- OK")

    small_test = _Subset(test_ds, 200)
    eval_res = clf.evaluate(small_test, batch_size=100)

    acc = eval_res["accuracy"]
    assert isinstance(acc, float), "Accuracy is not float: {}".format(type(acc))
    assert 0.0 <= acc <= 100.0, "Accuracy out of range: {}".format(acc)
    print("  Accuracy = {:.2f}% (float in [0,100]) -- OK".format(acc))

    # check that re-initialising weights actually changes them
    w_before = clf.model.conv1.weight.data.clone()
    clf._reset_weights(seed=99)
    w_after = clf.model.conv1.weight.data.clone()
    assert not torch.allclose(w_before, w_after), \
        "Weights did not change after _reset_weights"
    print("  Weight re-initialisation changes parameters -- OK")


# 5. Full pipeline (1 AL iteration on real CIFAR-10)

def test_full_pipeline():
    from src.simclr import SimCLRModel, get_features
    from src.active_learning import TypiClust
    from src.classifier import CIFARClassifier
    from src.utils import load_cifar10, set_seed

    set_seed(42)
    device = torch.device("cpu")
    train_ds, test_ds = load_cifar10(root="data")

    model = SimCLRModel()

    train_sub = _Subset(train_ds, 500)
    test_sub  = _Subset(test_ds, 200)

    train_feats, train_labels = get_features(model, train_sub, batch_size=100,
                                              device=device, num_workers=0)

    selector = TypiClust(train_feats, max_clusters=500,
                         min_cluster_size=5, seed=42)
    selected = selector.select(10, labeled_indices=None)

    assert len(selected) == 10, "Expected 10, got {}".format(len(selected))
    assert selected.max() < 500, "Index out of range"
    print("  TypiClust selected 10 indices from 500 samples -- OK")

    clf = CIFARClassifier(device=device, seed=42, num_workers=0)
    clf.train(selected, train_ds, epochs=5, batch_size=32,
              seed=42, verbose=False)

    eval_res = clf.evaluate(test_sub, batch_size=100)
    acc = eval_res["accuracy"]
    print("  Test accuracy after 1 round: {:.2f}%".format(acc))

    assert acc > 0.0, "Accuracy is 0% -- model is broken"
    print("  Accuracy > 0% (pipeline produces predictions) -- OK")


# 6. Reproducibility

def test_reproducibility():
    from src.active_learning import TypiClust
    from src.utils import set_seed

    rng = np.random.default_rng(0)
    features = rng.standard_normal((200, 64)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    set_seed(42)
    sel1 = TypiClust(features, max_clusters=500, min_cluster_size=5, seed=42)
    idx1 = sel1.select(10, labeled_indices=None)

    set_seed(42)
    sel2 = TypiClust(features, max_clusters=500, min_cluster_size=5, seed=42)
    idx2 = sel2.select(10, labeled_indices=None)

    assert np.array_equal(idx1, idx2), \
        "Indices differ!\n  Run 1: {}\n  Run 2: {}".format(idx1.tolist(), idx2.tolist())
    print("  Run 1: {}".format(idx1.tolist()))
    print("  Run 2: {}".format(idx2.tolist()))
    print("  Identical -- OK")

    # different seed should give different results
    set_seed(99)
    sel3 = TypiClust(features, max_clusters=500, min_cluster_size=5, seed=99)
    idx3 = sel3.select(10, labeled_indices=None)
    assert not np.array_equal(idx1, idx3), \
        "Different seeds produced identical results -- seeding may be broken"
    print("  Different seed -> different indices -- OK")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#  TPC_RP Pipeline Validation")
    print("#" * 60)

    run_test("1. SimCLR Feature Extraction", test_simclr_features)
    run_test("2. Typicality Computation",    test_typicality)
    run_test("3. TypiClust Selection",       test_typiclust_selection)
    run_test("4. Classifier Training",       test_classifier)
    run_test("5. Full Pipeline",             test_full_pipeline)
    run_test("6. Reproducibility",           test_reproducibility)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    for name, ok, detail in _results:
        status = "PASS" if ok else "FAIL"
        suffix = "" if ok else "  ({})".format(detail)
        print("  [{}] {}{}".format(status, name, suffix))

    print("\n{}/{} tests passed.".format(passed, total))

    if passed < total:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
