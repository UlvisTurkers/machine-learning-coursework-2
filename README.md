# TypiClust (TPC_RP) Active Learning on CIFAR-10

Implementation of the **TPC_RP** (Typicality-based Clustering with Random Projection) active learning algorithm from:

> Hacohen, G., Dekel, O., & Weinshall, D. (2022). **Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets**. *ICML 2022*. ([arXiv:2202.02794](https://arxiv.org/abs/2202.02794))

## Overview

TypiClust selects the most *typical* (representative) unlabelled examples for annotation. It operates in two phases:

1. **Self-supervised feature extraction** — a SimCLR model is trained on the unlabelled pool to produce rich representations without any labels.
2. **Typicality-based selection** — for each active learning round, examples are clustered and the most typical (highest-density) point in each cluster is selected for labelling.

TPC_RP adds a Random Projection step before clustering to reduce representation dimensionality and improve selection stability at low budgets.

## Project Structure

```
machine-learning-coursework-2/
├── notebooks/
│   ├── tpcrp_original.ipynb    # Original TPC_RP implementation
│   └── tpcrp_modified.ipynb    # Modified algorithm variant
├── src/
│   ├── simclr.py               # SimCLR self-supervised pre-training
│   ├── resnet.py               # ResNet-18 backbone architecture
│   ├── typicality.py           # Typicality scoring via KNN density
│   ├── active_learning.py      # TypiClust selection strategy (TPC_RP)
│   ├── classifier.py           # Supervised classifier training & evaluation
│   └── utils.py                # Data loading, plotting, seeding, logging
├── results/                    # Saved metrics and model checkpoints
├── plots/                      # Generated figures
└── requirements.txt
```

## Setup

### Local

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Google Colab

```python
!pip install -r requirements.txt
```

Then mount Google Drive and clone/upload the repository.

## How to Run

Open the notebooks in order:

| Notebook | Description |
|---|---|
| `notebooks/tpcrp_original.ipynb` | Baseline TPC_RP from the paper |
| `notebooks/tpcrp_modified.ipynb` | Modified variant with ablation analysis |

Both notebooks run end-to-end on CIFAR-10 and save results to `results/` and plots to `plots/`.

## Algorithm Summary

```
Given: unlabelled pool U, labelled set L (initially empty), budget b per round

1. Train SimCLR on U ∪ L (self-supervised, no labels used)
2. Extract features f(x) for all x in U
3. Project features: z = f(x) W  where W is a random projection matrix
4. Cluster z into k clusters (k-means, k = b)
5. For each cluster c_i:
     score(x) = 1 / mean distance to k nearest neighbours in z
     select x* = argmax score(x) for x in c_i
6. Query labels for selected points; move to L
7. Train supervised classifier on L; evaluate on test set
8. Repeat from step 4 (features from step 2 are reused)
```

## Dependencies

See [requirements.txt](requirements.txt). Key packages:

- `torch` / `torchvision` — model training
- `scikit-learn` — k-means clustering, metrics
- `scipy` — KNN density estimation
- `tqdm` — progress bars

## Results

Results (accuracy vs. label budget curves) are saved to `results/` as `.npy` / `.json` files and plotted to `plots/`.