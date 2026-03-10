# BioCLIP Clustering Pipeline — Complete Technical Documentation

> **Version:** 1.1 | **Platform:** Python 3.9+ / PyTorch 2.0+ | **Date:** March 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Foundation Models & Feature Extraction](#4-foundation-models--feature-extraction)
   - 4.1 [BioCLIP-2 (Primary Model)](#41-bioclip-2-primary-model)
   - 4.2 [BioCLIP (Variant)](#42-bioclip-variant)
   - 4.3 [CLIP (Baseline)](#43-clip-baseline)
   - 4.4 [DINOv2 (Vision-Only Baseline)](#44-dinov2-vision-only-baseline)
   - 4.5 [L2 Normalisation](#45-l2-normalisation)
   - 4.6 [Text Encoding and Prompt Ensembling](#46-text-encoding-and-prompt-ensembling)
5. [Clustering Algorithms](#5-clustering-algorithms)
   - 5.1 [K-Means](#51-k-means)
   - 5.2 [Text-Guided K-Means](#52-text-guided-k-means)
   - 5.3 [Zero-Shot Classification](#53-zero-shot-classification)
   - 5.4 [Gaussian Mixture Model (GMM)](#54-gaussian-mixture-model-gmm)
   - 5.5 [Agglomerative (Ward) Clustering](#55-agglomerative-ward-clustering)
   - 5.6 [Spectral Clustering](#56-spectral-clustering)
   - 5.7 [DBSCAN](#57-dbscan)
   - 5.8 [HDBSCAN](#58-hdbscan)
6. [Density Clustering Pipelines](#6-density-clustering-pipelines)
   - 6.1 [Why PCA, Not UMAP, as Input to Density Methods](#61-why-pca-not-umap-as-input-to-density-methods)
   - 6.2 [DBSCAN Full Pipeline](#62-dbscan-full-pipeline)
   - 6.3 [HDBSCAN Full Pipeline](#63-hdbscan-full-pipeline)
   - 6.4 [Coverage-Weighted ARI](#64-coverage-weighted-ari)
   - 6.5 [Noise Point Handling](#65-noise-point-handling)
7. [Evaluation Metrics](#7-evaluation-metrics)
   - 7.1 [Adjusted Rand Index (ARI)](#71-adjusted-rand-index-ari)
   - 7.2 [Normalized Mutual Information (NMI)](#72-normalized-mutual-information-nmi)
   - 7.3 [Hungarian Algorithm Accuracy](#73-hungarian-algorithm-accuracy)
   - 7.4 [Per-Class Accuracy](#74-per-class-accuracy)
   - 7.5 [Per-Cluster Purity](#75-per-cluster-purity)
   - 7.6 [Top-K Confusion Pairs](#76-top-k-confusion-pairs)
8. [Diagnostics Suite](#8-diagnostics-suite)
9. [Visualization](#9-visualization)
10. [Configuration System (YAML)](#10-configuration-system-yaml)
11. [CLI Pipeline & Data Flow](#11-cli-pipeline--data-flow)
12. [Datasets](#12-datasets)
13. [Cross-Model Comparison](#13-cross-model-comparison)
14. [Reproducibility](#14-reproducibility)
15. [Performance Considerations](#15-performance-considerations)
16. [Key Design Decisions & Rationales](#16-key-design-decisions--rationales)
17. [Dependencies](#17-dependencies)
18. [Running the Full Pipeline](#18-running-the-full-pipeline)

---

## 1. Project Overview

### Research Question

Can state-of-the-art biological foundation models — particularly **BioCLIP-2** — automatically group images of fine-grained biological species purely through unsupervised visual feature clustering, without any task-specific labels at inference time?

### What This System Does

The **BioCLIP Clustering Pipeline** is a modular, reproducible research system for evaluating whether biological images can be automatically partitioned by species using deep embeddings from foundation models. The pipeline:

1. **Extracts high-dimensional image embeddings** from four foundation models (BioCLIP-2, BioCLIP, CLIP, DINOv2).
2. **Applies eight unsupervised clustering algorithms** to discover natural groupings in embedding space.
3. **Evaluates all methods** rigorously against ground-truth species labels using ARI, NMI, Hungarian accuracy, and confusion analysis.
4. **Generates publication-ready figures** (300 DPI PNG + optional vector PDF) for each method, including UMAP scatter plots, confusion matrices, cluster-size distributions, and per-class accuracy breakdowns.
5. **Compares all four models** and eight methods head-to-head in cross-model comparison tables and heatmaps.

### Datasets

| Dataset | Species | Images | Format | Notes |
|---------|---------|--------|--------|-------|
| EU-Moth | 200 | 2,205 | ImageFolder | European moth species |
| CUB-200-2011 | 200 | 11,788 | CUB annotation files | Caltech-UCSD birds |
| MothSpeciesDK | 351 | ~19,000 | ImageFolder | Danish moth species, long-tailed |

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                     YAML Config File                                   │
│  (dataset path, model name, clustering methods, hyperparameters, ...)  │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
          ┌────────────────────▼─────────────────────┐
          │          cli/extract.py                   │
          │  • Discover & filter images (ImageFolder  │
          │    / CUB-200 / plugin)                    │
          │  • Load foundation model                  │
          │  • Batch GPU feature extraction           │
          │  • L2-normalize embeddings                │
          │  → features.npz  (N × D)                 │
          └────────────────────┬─────────────────────┘
                               │
          ┌────────────────────▼─────────────────────┐
          │          cli/cluster.py                   │
          │  • Load features                          │
          │  • Encode text labels (if needed)         │
          │  • Compute 2D UMAP projection (cache)     │
          │  • Run all 8 clustering methods:          │
          │    KMeans, Text-KMeans, Zero-shot,        │
          │    GMM, Agglomerative, Spectral,          │
          │    DBSCAN (PCA+sweep), HDBSCAN (sweep)   │
          │  → clusters_<method>.npz × 8             │
          │  → density figures + sweep CSVs          │
          └────────────────────┬─────────────────────┘
                               │
          ┌────────────────────▼─────────────────────┐
          │         cli/evaluate.py                   │
          │  • Load cluster assignments               │
          │  • For each method:                       │
          │    - ARI, NMI, Hungarian accuracy         │
          │    - Per-class breakdown                  │
          │    - Confusion pairs                      │
          │    - 8+ publication figures               │
          │  → metrics.json, metrics_summary.csv     │
          │  → figures/                              │
          └────────────────────┬─────────────────────┘
                               │
          ┌────────────────────▼─────────────────────┐
          │         cli/diagnose.py                   │
          │  • Feature quality statistics             │
          │  • 1-NN accuracy (embedding separability)│
          │  • Silhouette score                       │
          │  • Quick K-Means baseline                 │
          │  → diagnostics_report.txt                │
          │  → diagnostic_overview.png               │
          └──────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
bioclip_clustering/
├── PROJECT_DOCUMENTATION.md           ← This file
├── README.md
├── PRD.md                             # Product requirements document
├── requirements.txt                   # Python dependencies
│
├── configs/                           # One YAML per (dataset × model)
│   ├── eumoth.yaml                    # EU-Moth + BioCLIP-2 (primary)
│   ├── eumoth_bioclip.yaml            # EU-Moth + BioCLIP (ViT-B/16)
│   ├── eumoth_clip.yaml               # EU-Moth + CLIP ViT-L/14
│   ├── eumoth_dinov2.yaml             # EU-Moth + DINOv2 ViT-L/14
│   ├── cub200.yaml                    # CUB-200 + BioCLIP-2
│   ├── cub200_bioclip.yaml
│   ├── cub200_clip.yaml
│   ├── cub200_dinov2.yaml
│   ├── mothDK.yaml                    # MothSpeciesDK + BioCLIP-2
│   ├── mothDK_bioclip.yaml
│   ├── mothDK_clip.yaml
│   └── mothDK_dinov2.yaml
│
├── src/                               # Core library modules
│   ├── models/
│   │   └── extractors.py              # Feature extraction
│   ├── data/
│   │   ├── dataset.py                 # Image discovery & loading
│   │   └── plugins.py                 # Plugin registry for custom formats
│   ├── clustering/
│   │   ├── methods.py                 # 8 clustering algorithms
│   │   └── density/
│   │       ├── dbscan_pipeline.py     # Full DBSCAN diagnostic pipeline
│   │       └── hdbscan_pipeline.py    # Full HDBSCAN diagnostic pipeline
│   ├── evaluation/
│   │   └── metrics.py                 # ARI, NMI, Hungarian, confusion
│   ├── visualization/
│   │   └── plots.py                   # 8+ publication figure functions
│   ├── diagnostics/
│   │   └── suite.py                   # Feature quality & 1-NN accuracy
│   └── utils/
│       ├── io.py                      # Save/load, seed, config
│       └── pretty.py                  # Console formatting
│
├── cli/                               # Command-line entry points
│   ├── extract.py                     # Step 1: extract embeddings
│   ├── cluster.py                     # Step 2: run all clustering methods
│   ├── evaluate.py                    # Step 3: compute metrics + figures
│   ├── diagnose.py                    # Step 4: diagnostic report
│   ├── density.py                     # Standalone: density methods only
│   └── compare_models.py              # Cross-model comparison
│
├── experiments/
│   ├── run_pipeline.py                # End-to-end orchestrator
│   └── run_density_clustering.py      # Backwards-compatible wrapper
│
└── results/                           # Auto-created output (not in VCS)
    └── <dataset>_<model>/
        ├── features.npz
        ├── features_2d.npy
        ├── clusters_<method>.npz
        ├── metrics.json
        ├── metrics_summary.csv
        ├── dbscan_metrics.json / dbscan_sweep.csv
        ├── hdbscan_metrics.json / hdbscan_sweep.csv
        ├── density_metrics_combined.json
        ├── density_metrics_summary.csv
        ├── diagnostics_report.txt
        └── figures/
            ├── embedding_true_labels.png
            ├── embedding_clusters_<method>.png
            ├── confusion_matrix_<method>.png
            ├── top_confusions_<method>.png
            ├── cluster_sizes_<method>.png
            ├── per_class_accuracy_<method>.png
            ├── intra_inter_distances.png
            ├── method_comparison.png
            ├── dbscan_kdistance_plot.png
            ├── dbscan_eps_sweep.png
            ├── dbscan_umap_scatter.png
            ├── hdbscan_sweep.png
            ├── hdbscan_umap_scatter.png
            ├── hdbscan_outlier_scores.png
            ├── density_method_comparison.png
            └── diagnostic_overview.png
```

---

## 4. Foundation Models & Feature Extraction

All four models are wrapped behind a unified `BaseExtractor` interface:

```python
extractor = build_extractor(config)
features  = extractor.extract(image_paths)   # → (N, D) float32 NumPy array
```

The factory function `build_extractor(config)` routes based on `config['model']['name']`:

| Model name in config | Class | Library |
|---|---|---|
| `bioclip2` | `OpenCLIPExtractor` | `open_clip` |
| `bioclip` | `OpenCLIPExtractor` | `open_clip` |
| `clip` | `OpenCLIPExtractor` | `open_clip` |
| `dinov2` | `DINOv2Extractor` | `timm` |

---

### 4.1 BioCLIP-2 (Primary Model)

**Architecture:** ViT-L/14 (Vision Transformer, Large, patch size 14)
**Pretrained:** `hf-hub:imageomics/bioclip-2`
**Embedding dimension:** D = 768
**Input resolution:** 224 × 224 px
**Training data:** TreeOfLife-200M — 200 million biological images paired with structured taxonomic text

BioCLIP-2 is a CLIP-style contrastive model trained to align image embeddings with structured taxonomic text rather than generic web captions. During training, the model optimises:

```
L_contrastive = -1/N * Σᵢ log [ exp(sim(vᵢ, tᵢ) / τ) / Σⱼ exp(sim(vᵢ, tⱼ) / τ) ]
```

where `vᵢ` is the image embedding, `tᵢ` is the matching text embedding, `τ` is a learnable temperature (`logit_scale` = `log(1/τ)`, typically around 4.6, so `τ ≈ 0.01`), and `sim(·,·)` is cosine similarity. The sum in the denominator runs over all other text embeddings in the batch (negative samples).

The key difference from standard CLIP: the paired text in TreeOfLife-200M includes the full taxonomy (Kingdom → Phylum → Class → Order → Family → Genus → Species), so the model learns embeddings that are sensitive to the biological hierarchy. This means BioCLIP-2 embeddings for closely related species (same genus) are closer in embedding space than for distantly related ones — exactly the property that makes it useful for unsupervised species clustering.

**Fallback:** If `hf-hub:imageomics/bioclip-2` fails to load (not yet released or network error), the extractor automatically falls back to `hf-hub:imageomics/bioclip`.

---

### 4.2 BioCLIP (Variant)

**Architecture:** ViT-B/16 (smaller, patch size 16)
**Pretrained:** `hf-hub:imageomics/bioclip`
**Embedding dimension:** D = 512
**Training data:** TreeOfLife-10M subset

Same contrastive training paradigm as BioCLIP-2, but with a smaller architecture. Serves as an ablation to measure how much embedding capacity (ViT-B vs ViT-L) affects clustering quality.

---

### 4.3 CLIP (Baseline)

**Architecture:** ViT-L/14
**Pretrained:** `openai` (original CLIP from OpenAI)
**Embedding dimension:** D = 768
**Training data:** WIT (WebImageText) — 400 million web image–caption pairs

CLIP was not trained on biological taxonomic data. While it produces strong general visual features, scientific moth/bird names are largely out-of-distribution for the text encoder — they were rarely paired with images in web training data. Expected to underperform BioCLIP-2 on text-guided and zero-shot methods while still providing a strong visual-only baseline.

**Config difference:** CLIP requires separate `arch` and `pretrained` fields because `open_clip` needs to know both the architecture (`ViT-L-14`) and the pretrained checkpoint source (`openai`):

```yaml
model:
  name: clip
  arch: ViT-L-14
  pretrained: openai
```

The `OpenCLIPExtractor._load_model()` detects the presence of `arch` and calls:

```python
open_clip.create_model_and_transforms(arch, pretrained=pretrained)
```

vs. for HuggingFace Hub models (no `arch` field):

```python
open_clip.create_model_and_transforms(pretrained)   # pretrained = 'hf-hub:...'
```

---

### 4.4 DINOv2 (Vision-Only Baseline)

**Architecture:** ViT-L/14
**Pretrained:** `vit_large_patch14_dinov2.lvd142m` (via `timm`)
**Embedding dimension:** D = 1024
**Input resolution:** 518 × 518 px
**Training data:** LVD-142M (curated image-only dataset, no text)

DINOv2 is a self-supervised vision transformer trained with self-distillation with no labels (DINO). Unlike BioCLIP and CLIP, **DINOv2 has no text encoder** — it cannot be used for text-guided K-Means or zero-shot classification. It produces strong visual features through a combination of:

- **Self-supervised objective:** image → student network matches teacher network representations
- **Patch-level features:** 14×14 patches at 518×518 input = 37×37 = 1,369 patches per image
- **Register tokens** in DINOv2 v2 prevent patch collapse on background regions

**Critical implementation detail:** `vit_large_patch14_dinov2.lvd142m` expects **518 × 518** input, not 224 × 224. Naively applying 224 × 224 preprocessing causes:

```
AssertionError: Input height (224) doesn't match model (518).
```

The `DINOv2Extractor` avoids this by using `timm`'s data configuration resolution:

```python
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

data_config     = resolve_data_config({}, model=self.model)
self.preprocess = create_transform(**data_config)
self._input_size = tuple(data_config["input_size"])  # (3, 518, 518)
```

`resolve_data_config` reads the model's internal metadata to determine the correct input size, normalization statistics, and interpolation mode. This makes the extractor automatically adapt to any timm DINOv2 variant.

**Memory implication:** 518 × 518 images are 5.4× larger in area than 224 × 224 images. GPU memory consumption scales accordingly, requiring `batch_size: 16` (vs. 64 for other models) to fit on typical 8–16 GB GPUs.

**Incompatibilities in configs:** Since DINOv2 has no text encoder, the three DINOv2 configs (`eumoth_dinov2.yaml`, `cub200_dinov2.yaml`, `mothDK_dinov2.yaml`) explicitly omit `zero_shot` and `kmeans_text_guided` from the methods list.

---

### 4.5 L2 Normalisation

All extractors optionally apply L2 normalisation after extraction (controlled by `model.normalize_embeddings: true` in config):

```python
def _normalize(self, features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True)
```

Mathematically: for a feature vector **f** ∈ ℝ^D, the normalised version is:

```
f̂ = f / ||f||₂,   where   ||f̂||₂ = 1
```

**Why normalise?** After L2 normalisation:
- All vectors lie on the unit hypersphere S^(D-1)
- Euclidean distance and cosine similarity are related by: `||a - b||² = 2(1 - cos(a, b))`
- K-Means on normalised vectors implicitly maximises cosine similarity to centroids
- Dot product `a · b = cos(a, b)` for unit vectors — used in zero-shot classification

---

### 4.6 Text Encoding and Prompt Ensembling

For CLIP-family models (BioCLIP-2, BioCLIP, CLIP), `encode_text(class_names, prompt_templates)` produces one text embedding per class by averaging across multiple prompt templates:

**Algorithm:**

For each class with name `c` (e.g., `"abrostola_tripartita"`):

1. Parse readable name: `c_readable = c.replace('_', ' ')` → `"abrostola tripartita"`
2. Parse genus: `genus = c_readable.split()[0]` → `"abrostola"`
3. Fill each template:
   - `"{name}"` → `"abrostola tripartita"`
   - `"a photo of {name}"` → `"a photo of abrostola tripartita"`
   - `"a photo of {name}, a species"` → `"a photo of abrostola tripartita, a species"`
   - `"{genus} {name}"` → `"abrostola abrostola tripartita"` (redundant but consistent)
4. Tokenize all T templates, encode with text encoder:
   `E = text_encoder(templates) ∈ ℝ^(T × D)`
5. L2-normalise each: `Ê_t = E_t / ||E_t||₂` for each template t
6. Average: `μ = (1/T) Σₜ Êₜ`
7. Re-normalise: `t̂_c = μ / ||μ||₂`

**Why re-normalise after averaging?** The average of T unit vectors is not itself a unit vector (unless they are all identical). Re-normalising projects the ensemble back onto the unit sphere, ensuring consistent scale with image embeddings for dot-product similarity.

**Why taxonomic templates for BioCLIP-2?** BioCLIP-2 was trained on TreeOfLife-200M where each image is paired with a structured taxonomy string. Prompts that match this training distribution (e.g., genus + species) produce more central, stable embeddings than generic web-caption prompts like "a photo of X". Using multiple complementary templates further reduces variance.

**Output shape:** `(num_classes, D)` NumPy array, each row a unit vector.

---

## 5. Clustering Algorithms

All 8 algorithms share the same interface:

```python
cluster_ids = run_clustering(method_name, features, params, **kwargs)
# features: (N, D) float32 array
# cluster_ids: (N,) int32 array
```

Results in range `[0, K-1]` for partition methods; `-1` (noise) additionally for density methods.

---

### 5.1 K-Means

**Type:** Partition
**Requires `n_clusters`:** Yes
**Text encoder required:** No

Standard Lloyd's algorithm. Given N data points in ℝ^D and number of clusters K:

**Objective:** Minimise within-cluster sum of squared distances (WCSS / inertia):

```
J = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²
```

where `μₖ` is the centroid (mean) of cluster `Cₖ`.

**Algorithm:**
1. **Initialisation** (k-means++ by sklearn default): choose first centroid uniformly at random; choose each subsequent centroid with probability proportional to `D(xᵢ)²` (squared distance to nearest already-chosen centroid)
2. **Assignment:** assign each point to nearest centroid: `cᵢ = argmin_k ||xᵢ - μₖ||²`
3. **Update:** recompute centroids: `μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ`
4. Repeat 2–3 until assignments no longer change or `max_iter` reached

**Hyperparameters in config:**

```yaml
- name: kmeans
  n_clusters: 200
  n_init: 50         # 50 random restarts; best inertia wins
  max_iter: 500
  random_state: 42
```

`n_init: 50` is used rather than the default 10 because K-Means is sensitive to initialisation for fine-grained data with 200+ clusters. 50 restarts trades compute for stability.

---

### 5.2 Text-Guided K-Means

**Type:** Partition (semi-supervised initialisation)
**Requires `n_clusters`:** Yes
**Text encoder required:** Yes (BioCLIP-2, BioCLIP, or CLIP)

Identical to standard K-Means except the initial centroids are set to the **text embeddings** of the class names, rather than random or k-means++ points.

**Key distinction from zero-shot:**
- **Text-guided K-Means:** text embeddings are *only* the starting centroids. K-Means then iteratively refines them toward the actual empirical means of image clusters. Final centroids may drift far from original text positions.
- **Zero-shot:** text embeddings are *fixed* as the final prototypes. No refinement.

**Why use text initialisation?**
For BioCLIP-2, text embeddings of species names encode biological meaning. Starting K-Means from semantically meaningful positions (rather than random) often produces:
- Faster convergence (fewer iterations to good solution)
- Better final solution (avoids degenerate local minima)
- More interpretable clusters (each starts labelled as a known species)

**Hyperparameters:**

```yaml
- name: kmeans_text_guided
  n_clusters: 200
  n_init: 1          # 1 restart: start from text, not random
  max_iter: 300
  random_state: 42
```

`n_init: 1` because there is only one meaningful text-based initialisation (unlike random init where multiple restarts are needed for stability).

---

### 5.3 Zero-Shot Classification

**Type:** Direct assignment (no iterative refinement)
**Requires `n_clusters`:** No (uses one class per species name)
**Text encoder required:** Yes (BioCLIP-2, BioCLIP, or CLIP)

Each image is assigned to the class whose **text prototype** has the highest cosine similarity with the image embedding:

```
ŷᵢ = argmax_c  (f̂ᵢ · t̂_c)
```

where:
- `f̂ᵢ ∈ ℝ^D` is the L2-normalised image embedding
- `t̂_c ∈ ℝ^D` is the L2-normalised text embedding for class `c` (ensemble-averaged)
- `·` is dot product (= cosine similarity for unit vectors)

**Vectorised implementation:**

```python
similarity  = features @ text_centroids.T   # (N, C)
cluster_ids = similarity.argmax(axis=1)     # (N,)
```

**Interpretation:** This is exactly CLIP-style zero-shot image classification. No training or fine-tuning is required — the model uses only the semantic alignment learned during contrastive pretraining. The `cluster_ids` correspond directly to class indices, so the Hungarian algorithm finds the identity permutation (Hungarian accuracy = top-1 accuracy).

**Why dot product equals cosine similarity here:**
Both `f̂ᵢ` and `t̂_c` are L2-normalised (unit norm), so:
```
cos(f̂ᵢ, t̂_c) = (f̂ᵢ · t̂_c) / (||f̂ᵢ|| × ||t̂_c||) = f̂ᵢ · t̂_c
```

---

### 5.4 Gaussian Mixture Model (GMM)

**Type:** Probabilistic partition
**Requires `n_clusters`:** Yes (as `n_components`)
**Text encoder required:** No

Models the data as a mixture of K Gaussian distributions and assigns each point to the most probable component (hard assignment after soft estimation).

**Generative model:**

```
P(x) = Σₖ πₖ · N(x; μₖ, Σₖ)
```

where:
- `πₖ` = mixing weight for component k (Σₖ πₖ = 1)
- `N(x; μₖ, Σₖ)` = Gaussian density with mean `μₖ` and covariance `Σₖ`

**Fitting (EM algorithm):**

*E-step:* Compute posterior responsibilities:
```
rᵢₖ = P(z=k|xᵢ) = πₖ N(xᵢ; μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ; μⱼ, Σⱼ)
```

*M-step:* Update parameters:
```
Nₖ  = Σᵢ rᵢₖ
μₖ  = (1/Nₖ) Σᵢ rᵢₖ xᵢ
Σₖ  = (1/Nₖ) Σᵢ rᵢₖ (xᵢ - μₖ)(xᵢ - μₖ)ᵀ
πₖ  = Nₖ / N
```

Repeat until log-likelihood converges.

**Final assignment:** `ŷᵢ = argmax_k rᵢₖ`

**Covariance type — why `diag` not `full`:**

The `full` covariance requires storing and estimating K matrices of size D×D:

```
Memory (full)  = K × D² × 8 bytes
                = 200 × 768² × 8 bytes ≈ 947 MB   (for BioCLIP-2, D=768)
                = 351 × 768² × 8 bytes ≈ 1.66 GB  (for MothDK)
```

In practice, with N ≈ 2,205 images (EU-Moth) and D = 768 dimensions, the M-step for `full` covariance requires inverting nearly-singular matrices (N ≪ D), making the EM algorithm numerically unstable and memory-intensive.

**`diag` covariance** assumes independent feature dimensions — the covariance is a diagonal matrix parameterised by D scalars per component:

```
Memory (diag) = K × D × 8 bytes
              = 200 × 768 × 8 bytes ≈ 1.2 MB   (200× smaller)
```

This is both feasible and numerically stable, at the cost of ignoring feature correlations.

**Model selection:** BIC (Bayesian Information Criterion) is reported after fitting:
```
BIC = -2 log L̂ + p log N
```
where `p` is the number of free parameters. Lower BIC is better.

---

### 5.5 Agglomerative (Ward) Clustering

**Type:** Hierarchical partition
**Requires `n_clusters`:** Yes
**Text encoder required:** No

Bottom-up hierarchical clustering. Starts with each point as its own cluster, then greedily merges the pair of clusters that minimises the chosen linkage criterion.

**Ward linkage (default):** Merge the two clusters whose union minimises the increase in total within-cluster variance (WCSS):

```
Δ(A, B) = ||μ_A - μ_B||² × (|A| × |B|) / (|A| + |B|)
```

This is the squared Euclidean distance between centroids, weighted by cluster sizes. Ward linkage tends to produce compact, roughly equal-sized clusters — preferred for balanced species datasets.

**Alternative linkages supported:** `complete`, `average`, `single`

**Computational complexity:** O(N² log N) with Ward (using `sklearn`'s optimised implementation). Not suitable for N > 50,000.

**Hyperparameters:**

```yaml
- name: agglomerative
  n_clusters: 200
  linkage: ward
```

---

### 5.6 Spectral Clustering

**Type:** Graph-based partition
**Requires `n_clusters`:** Yes
**Text encoder required:** No

Spectral clustering maps data to a lower-dimensional space defined by the eigenvectors of a graph Laplacian, then applies K-Means in that space.

**Algorithm (nearest-neighbour affinity):**

1. **Build affinity graph:** for each point xᵢ, connect to its k nearest neighbours (k = `n_neighbors`). Define affinity `W_ij = 1` if j ∈ kNN(i) or i ∈ kNN(j), else 0.

2. **Compute normalised Laplacian:**
   ```
   L_sym = I - D^(-1/2) W D^(-1/2)
   ```
   where D is the diagonal degree matrix: `D_ii = Σⱼ W_ij`

3. **Eigendecomposition:** compute the K smallest eigenvectors of L_sym
   → embedding matrix U ∈ ℝ^(N×K), each row is a point's spectral embedding

4. **Normalise rows of U** to unit length

5. **K-Means** on U

**Why spectral clustering can find non-convex clusters:** Step 3 reveals the intrinsic geometry of the manifold rather than just Euclidean distance. Even if clusters are non-convex in the original space (e.g., two interleaved spirals), they appear linearly separable in the spectral embedding.

**Known warning:** sklearn emits `"Graph is not fully connected"` when some points are isolated in the affinity graph (typically when `n_neighbors` is too small). The pipeline suppresses this warning with `n_neighbors: 15` (default), which is large enough to avoid disconnected graphs for datasets up to ~10,000 images.

**Memory complexity:** O(N²) — the full affinity matrix is stored. Not recommended for N > 5,000.

**Hyperparameters:**

```yaml
- name: spectral
  n_clusters: 200
  n_init: 10
  random_state: 42
  n_neighbors: 15
```

---

### 5.7 DBSCAN

**Type:** Density-based (discovers number of clusters)
**Requires `n_clusters`:** No
**Text encoder required:** No

Density-Based Spatial Clustering of Applications with Noise. Groups together densely packed points, marks sparse points as noise.

**Key definitions:**

- **Core point:** a point xᵢ with at least `min_samples` points within radius `eps` (including itself): `|N_ε(xᵢ)| ≥ min_samples`
- **Border point:** not a core point, but within `eps` of some core point
- **Noise point:** neither core nor border; assigned label `-1`

**Algorithm:**
1. For each unvisited core point, start a new cluster
2. Recursively add all density-reachable points (points reachable via chain of core points within `eps`)
3. Mark all remaining points as noise

**Key parameter:** `eps` (neighbourhood radius). Too small → almost everything is noise; too large → everything merges into one cluster. The pipeline uses a full diagnostic k-distance plot + sweep to select `eps` automatically (see Section 6.2).

In the simple `run_dbscan()` method in `methods.py`, parameters come directly from config:

```yaml
- name: dbscan
# parameters configured in dbscan: block below
```

For the full diagnostic pipeline (PCA + k-distance + sweep + figures), route to `DBSCANPipeline.run()`.

---

### 5.8 HDBSCAN

**Type:** Hierarchical density-based (discovers number of clusters)
**Requires `n_clusters`:** No
**Text encoder required:** No
**Requires:** `pip install hdbscan`

Hierarchical DBSCAN addresses DBSCAN's fundamental weakness: a single global `eps` cannot handle datasets where cluster density varies across the space.

**Motivation for this dataset:** In EU-Moth with 200 species and variable sample counts per species, common species have more images → denser clusters; rare species have fewer images → sparser clusters. A single `eps` that separates dense clusters labels sparse ones as noise, and vice versa.

**Core concepts:**

1. **Mutual reachability distance:** smooths local density differences
   ```
   d_mreach(a, b) = max(core_dist_k(a), core_dist_k(b), d(a, b))
   ```
   where `core_dist_k(a)` = distance to k-th nearest neighbour (k = `min_samples`)

2. **Minimum spanning tree (MST):** build on the complete graph with `d_mreach` edge weights using Prim's or Kruskal's algorithm

3. **Cluster hierarchy:** remove MST edges in decreasing weight order; each connected component = cluster at that density level. Records births/deaths of clusters.

4. **Cluster extraction:** two modes:
   - **EOM (Excess of Mass):** find clusters that maximise the integral of their stability over the hierarchy. Merges nested sub-clusters → fewer, larger, more stable clusters. Good for compact, balanced data.
   - **Leaf:** extract the finest-grained leaf nodes. Preserves many small clusters. Recommended for long-tailed datasets.

5. **Outlier scores:** each point gets a score ∈ [0, 1] representing its membership robustness. Score near 0 = strong cluster member; score near 1 = likely outlier.

**Hyperparameters:**

```yaml
hdbscan:
  pca_components: 100
  sweep_range: [2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50, ...]
  cluster_selection_method: eom
```

`min_cluster_size` is swept automatically; the best value is selected by coverage-weighted ARI.

---

## 6. Density Clustering Pipelines

The full DBSCAN and HDBSCAN pipelines go far beyond the simple `fit_predict` call. They implement principled parameter selection, diagnostic visualisations, and correct metric computation.

---

### 6.1 Why PCA, Not UMAP, as Input to Density Methods

**Incorrect approach (from original notebook):** Run DBSCAN on UMAP-50D output with `min_dist=0.0`.

**Why this is wrong:**

DBSCAN and HDBSCAN are distance-based algorithms. They assume the density structure (distribution of pairwise distances) in the input space faithfully represents the real data geometry.

UMAP with `min_dist=0.0` **deliberately distorts** this structure: it compresses local neighbourhoods into tight balls separated by empty space (optimising for visual separation). The "clusters" DBSCAN finds are UMAP's artificially created density contrasts, not real biological groupings.

**Correct approach:** Apply PCA to the L2-normalised high-dimensional embeddings before running density methods.

**Why PCA is appropriate:**

- PCA is a *linear* projection: `z = (x - μ) W` where W ∈ ℝ^(D×p) contains the top p eigenvectors of the covariance matrix
- Euclidean distances in PCA space approximate Euclidean distances in the original space up to the variance captured
- 100 PCA dimensions retain ~85–90% of variance in BioCLIP embeddings while removing noise dimensions (the last principal components)
- No density artifacts are created

**PCA decomposition:**

```
Cov = (1/(N-1)) Xᵀ X   (X centred, N×D)
Cov V = V Λ             (eigendecomposition)
z = X V[:, :p]          (project to top-p components)
```

sklearn's `PCA(n_components=100, random_state=42).fit_transform(features)` handles this efficiently using truncated SVD.

UMAP is still used for 2D **visualisation** — it is just not fed as input to the density algorithms.

---

### 6.2 DBSCAN Full Pipeline

`DBSCANPipeline.run()` executes these steps in sequence:

**Step 0: PCA reduction**

```
features (N × D)  →  PCA(100)  →  features_pca (N × 100)
```

Reports variance retained.

**Step 1: k-distance plot for eps guidance**

For each point xᵢ, compute `d_k(xᵢ)` = distance to its k-th nearest neighbour (k = `min_samples`). Sort all `d_k` values in descending order and plot.

The "knee" (elbow) of this curve indicates the natural density transition. Points to the right of the knee are sparse (become noise at optimal eps); points to the left are dense (form clusters).

**Knee detection (Kneedle method):**

1. The curve connects point `P₁ = (0, d_k[0])` to `P_N = (N-1, d_k[N-1])`
2. For each point `Pᵢ = (i, d_k[i])`, compute its perpendicular distance from the line `P₁P_N`
3. The knee is at `argmax_i dist(Pᵢ, line)`

```python
line_vec  = p2 - p1
line_unit = line_vec / ||line_vec||
proj      = outer(dot(points - p1, line_unit), line_unit) + p1
distances = ||points - proj||   (perpendicular distances)
knee_idx  = argmax(distances)
```

The knee value becomes the suggested `eps`.

**Step 2: eps sweep**

Sweep 20 eps values in the range `[0.4 × knee, 1.8 × knee]`. For each eps:
- Fit DBSCAN
- Compute: ARI, NMI (on non-noise points), % noise, # clusters, **coverage-weighted ARI**
- The best eps maximises coverage-weighted ARI (see Section 6.4)

Produces a 4-panel plot: ARI/NMI vs eps, noise % vs eps, # clusters vs eps, weighted score vs eps.

**Step 3: Select best eps**

```python
best_eps = sweep_df.loc[sweep_df["weighted_score"].idxmax(), "eps"]
```

Manual override available via `dbscan.final_eps` in config.

**Step 4: Final fit and evaluation**

Fit DBSCAN with `best_eps` and `min_samples`. Compute all metrics (ARI, NMI, Hungarian accuracy, per-class accuracy, confusion pairs) on non-noise points only.

**Step 5: Visualisations**

- Side-by-side scatter (true labels vs DBSCAN clusters), noise points in grey ×
- Cluster size distribution (sorted bar + histogram)
- Top confusion pairs (after Hungarian mapping)

**Step 6: Save outputs**

`dbscan_metrics.json`, `dbscan_sweep.csv`

---

### 6.3 HDBSCAN Full Pipeline

`HDBSCANPipeline.run()` is the recommended density pipeline:

**Step 0: PCA reduction** (same as DBSCAN, 100 components)

**Step 1: min_cluster_size sweep**

Sweep across `sweep_range` (default: `[2, 3, 5, 10, 15, 20, 25, 30, ...]`). For each value:
- Fit HDBSCAN with `cluster_selection_method` (eom or leaf)
- Compute: ARI, NMI, % noise, # clusters, coverage-weighted ARI

Produces a 3-panel sweep plot: quality metrics, noise %, clusters found vs min_cluster_size.

**Step 2: Select best min_cluster_size**

```python
best_mcs = sweep_df.loc[sweep_df["weighted_ari"].idxmax(), "min_cluster_size"]
```

**Step 3: Final fit and evaluation**

Fit HDBSCAN with `best_mcs`. Computes same metrics as DBSCAN. Additionally returns `outlier_scores_` array.

**Step 4: Visualisations**

- Side-by-side scatter (same layout as DBSCAN for direct comparison)
- Cluster size distribution
- **Outlier score distribution** (HDBSCAN-specific): histogram of scores ∈ [0,1]
  - Bimodal (peak near 0, tail near 1) = healthy separation; most points belong clearly to clusters
  - Flat = min_cluster_size too small or data too uniformly distributed for HDBSCAN
- Top confusion pairs

**Step 5: Comparison plot** (generated in `cli/cluster.py` after both pipelines run)

3-panel bar chart: DBSCAN vs HDBSCAN on ARI/NMI/Accuracy, # clusters, noise %.

---

### 6.4 Coverage-Weighted ARI

**Problem with raw ARI for density methods:**

A degenerate DBSCAN configuration with `eps` too small finds a small number of tiny, perfectly-pure micro-clusters while labelling 97% of the data as noise. Since noise points are excluded from metrics, ARI = 1.0 — but the solution is useless (only 3% of images are "clustered").

**Solution: Coverage-Weighted ARI**

```
weighted_score = ARI × Coverage
Coverage       = 1 - noise_fraction
noise_fraction = n_noise / n_total
```

**Examples:**

| Scenario | ARI | noise% | Coverage | Weighted Score |
|---|---|---|---|---|
| Perfect purity on 3% | 1.00 | 97% | 0.03 | **0.03** (bad) |
| Good clustering on 98% | 0.97 | 2% | 0.98 | **0.95** (good) |
| Mediocre on 70% | 0.60 | 30% | 0.70 | **0.42** (mediocre) |
| High ARI but 50% noise | 0.80 | 50% | 0.50 | **0.40** |

The metric penalises solutions that achieve high purity by discarding most of the data. The best eps/min_cluster_size is the one that maximises `ARI × (1 - noise_fraction)`.

This is the **selection criterion** used automatically in both DBSCAN and HDBSCAN sweep steps.

---

### 6.5 Noise Point Handling

DBSCAN and HDBSCAN assign label `-1` to noise points (points that do not belong to any cluster). Including these in evaluation creates a **phantom cluster**: all noise points are grouped together as "cluster -1", which may contain images from all 200 species. This phantom cluster massively inflates the denominator in ARI, severely under-reporting true clustering quality.

**Policy:** All metrics (ARI, NMI, Hungarian accuracy, per-class accuracy) are computed **only on non-noise points**. Noise is:
1. Excluded from metric computation
2. Reported explicitly as `n_noise` and `noise_pct`
3. Shown visually in scatter plots as grey × markers

```python
valid_mask  = cluster_ids >= 0         # True for non-noise points
n_noise     = (~valid_mask).sum()
t = true_labels[valid_mask]
c = cluster_ids[valid_mask]
ari = adjusted_rand_score(t, c)        # computed on non-noise only
```

---

## 7. Evaluation Metrics

All metrics are computed in `src/evaluation/metrics.py`. For DBSCAN/HDBSCAN, noise points (label = -1) are excluded before any computation.

---

### 7.1 Adjusted Rand Index (ARI)

**Range:** [-1, 1] | **Higher is better** | **Random clustering → 0**

The Rand Index counts the fraction of pairs of samples that are either in the same cluster in both clusterings, or in different clusters in both. The *Adjusted* Rand Index corrects for chance:

```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

More formally, using the contingency table `nᵢⱼ` (number of samples in ground-truth class i and cluster j):

```
ARI = [Σᵢⱼ C(nᵢⱼ,2) - (Σᵢ C(aᵢ,2) × Σⱼ C(bⱼ,2)) / C(n,2)]
      / [(Σᵢ C(aᵢ,2) + Σⱼ C(bⱼ,2))/2 - (Σᵢ C(aᵢ,2) × Σⱼ C(bⱼ,2)) / C(n,2)]
```

where `C(n,2) = n(n-1)/2`, `aᵢ = Σⱼ nᵢⱼ` (row sums), `bⱼ = Σᵢ nᵢⱼ` (column sums).

**Interpretation:**
- ARI = 1.0: perfect match with ground truth
- ARI = 0.0: random clustering (expected score)
- ARI < 0: worse than random

**Advantage over Rand Index:** Unaffected by number of clusters or cluster sizes — comparable across different methods and datasets.

---

### 7.2 Normalized Mutual Information (NMI)

**Range:** [0, 1] | **Higher is better**

Measures the shared information between clustering assignment C and ground truth Y:

```
NMI(Y, C) = I(Y; C) / sqrt(H(Y) × H(C))
```

where:
- `I(Y; C) = H(Y) + H(C) - H(Y, C)` = mutual information
- `H(Y) = -Σᵢ P(Y=i) log P(Y=i)` = entropy of true labels
- `H(C) = -Σⱼ P(C=j) log P(C=j)` = entropy of cluster assignments
- `H(Y, C) = -Σᵢⱼ P(Y=i, C=j) log P(Y=i, C=j)` = joint entropy

The geometric mean normalisation `sqrt(H(Y) × H(C))` ensures NMI = 1 iff the clusterings are identical, and NMI = 0 iff they are fully independent — regardless of number of classes/clusters.

**sklearn's default normalisation:** `average_method='arithmetic'` uses `(H(Y) + H(C)) / 2` instead of geometric mean. The pipeline uses the default.

---

### 7.3 Hungarian Algorithm Accuracy

**Range:** [0, 1] | **Higher is better**

Clustering algorithms produce cluster IDs that are arbitrary integers. To compute classification accuracy, we need to find the **optimal one-to-one mapping** from cluster IDs to class IDs.

The Hungarian (Munkres) algorithm solves the **assignment problem** optimally in O(n³) time. Given a cost matrix `C[i,j] = number of samples where cluster i contains class j`, we find the permutation that maximises total matches:

```
max_π Σᵢ C[i, π(i)]
```

sklearn's `linear_sum_assignment(-C)` finds this by minimising the negated matrix.

**Implementation:**

```python
n = max(cluster_ids.max(), true_labels.max()) + 1
cost = np.zeros((n, n), dtype=np.int64)
for c, t in zip(cluster_ids, true_labels):
    cost[c, t] += 1

row_ind, col_ind = linear_sum_assignment(-cost)   # maximise matches
n_correct = cost[row_ind, col_ind].sum()
accuracy  = n_correct / len(true_labels)
```

**Why not simple accuracy with sorted IDs?** Cluster labels are arbitrary — cluster "0" might contain mostly species "127". The Hungarian algorithm finds the optimal bijection between cluster labels and class labels that maximises overall accuracy.

**Handling non-square matrices:** When the number of clusters differs from the number of classes (common for density methods), a square matrix is created with side `max(K, C)`, and unmatched rows/columns are zero. The algorithm still finds the globally optimal partial matching.

---

### 7.4 Per-Class Accuracy

After obtaining the cluster-to-class mapping `M` from the Hungarian algorithm:

```python
pred_labels = M[cluster_ids]        # map clusters to classes
mask_c      = true_labels == c      # all images of class c
acc_c       = (pred_labels[mask_c] == c).sum() / mask_c.sum()
```

Produces a DataFrame with per-species accuracy, sorted worst-first. This reveals which specific species are hardest to cluster (typically visually similar species, rare species with few images, or species with high within-class visual variation).

---

### 7.5 Per-Cluster Purity

For each cluster k, purity is the fraction of images belonging to the dominant class:

```
purity_k = max_c count(cluster_k ∩ class_c) / size(cluster_k)
```

A pure cluster (purity = 1) contains images from exactly one species. Low purity indicates a cluster that mixes multiple species.

---

### 7.6 Top-K Confusion Pairs

After applying the Hungarian mapping to get predicted class labels, the confusion matrix `CM` is built where `CM[i,j]` = number of times species i was predicted as species j. The diagonal (correct predictions) is zeroed, and the top K off-diagonal entries are returned:

```python
cm = confusion_matrix(true_err, pred_err, labels=list(range(n)))
np.fill_diagonal(cm, 0)
top_k_indices = np.argsort(cm.ravel())[::-1][:k]
```

This identifies which specific species pairs are most often confused. For moth datasets, common confusions are often between species in the same genus or with similar wing patterns (e.g., *Xestia* spp.).

---

## 8. Diagnostics Suite

`cli/diagnose.py` runs `src/diagnostics/suite.py` to assess **embedding quality** before committing to full clustering.

### Feature Statistics

Computes per-feature vector statistics:
- **Mean/std L2 norm:** should be near 1.0 ± small if normalised
- **Dead dimensions:** feature dimensions with zero variance across all samples — indicates preprocessing issues or model failure
- **Mean feature variance:** how spread the representations are across the dataset

### 1-NN Accuracy (Separability Upper Bound)

1-nearest-neighbour classification accuracy (with cosine distance) via 3-fold cross-validation on a subsample of ≤ 3,000 images.

```
1-NN Accuracy = fraction of images where nearest neighbour belongs to same species
```

This is the **upper bound on clustering accuracy** achievable on these embeddings. If 1-NN accuracy is 60%, no clustering algorithm can achieve more than ~60% Hungarian accuracy (because the embeddings themselves are not fully separable at that level). If 1-NN accuracy is high (>70%), the embeddings are well-separated and good clustering results are achievable.

### Silhouette Score (on 2D Projection)

Computed on the UMAP/t-SNE 2D projection (fast approximation):

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

where `a(i)` = mean intra-cluster distance, `b(i)` = mean nearest-cluster distance.

**Interpretation:**
- s > 0.5: strong cluster structure in 2D
- s ∈ [0.25, 0.5]: some cluster structure
- s < 0.25: weak or no cluster structure

Note: This is computed on the 2D projection, which is a lossy visualisation. The true high-dimensional silhouette would be more informative but computationally expensive for N > 1,000.

### Diagnostic Figure (2×2 Grid)

- **Top-left:** 2D scatter coloured by true labels (reference)
- **Top-right:** 2D scatter coloured by first available cluster assignments
- **Bottom-left:** Histogram of L2 norms (should peak near 1.0 if normalised)
- **Bottom-right:** Cluster size histogram

---

## 9. Visualization

All figures are generated in `src/visualization/plots.py` at 300 DPI (publication-ready), with optional vector PDF export. Each figure includes an embedded caption explaining how to read it — critical for figures with 200+ classes where naive approaches fail.

### Embedding Scatter (`embedding_*.png`)

2D UMAP or t-SNE projection coloured by class/cluster labels.

**Small datasets (≤ 20 classes):** Standard scatter with individual legend entries per class.

**Large datasets (> 20 classes, typical for these datasets with 200+):** Two-layer rendering to avoid unreadable legends:
1. **Grey backdrop:** all classes plotted as small grey dots — shows the overall embedding structure
2. **Highlighted overlay:** 12 selected classes plotted with distinct colours and a manageable legend

This "spotlight-on-grey" strategy allows readers to see both the global structure (full grey cloud) and specific example classes (coloured highlights) without an unreadable 200-entry legend.

### Confusion Analysis (`confusion_matrix_*.png`)

**Small datasets (≤ 30 classes):** Full annotated confusion matrix (rows = true classes, columns = predicted classes).

**Large datasets (> 30 classes):** Two-panel layout (both fit on one figure):
- **Left panel:** Per-class recall bar chart for the 40 worst-performing classes, sorted worst-first, colour-coded by accuracy
- **Right panel:** 25×25 submatrix showing only the most-confused species pairs (species with highest total confusion mass)

This approach makes the most important information readable even for 200-class problems where a full 200×200 confusion matrix would be illegible at any reasonable figure size.

### Intra/Inter-Class Distance Distributions (`intra_inter_distances.png`)

Violin + box plots comparing:
- **Intra-class (blue):** pairwise distances between images of the **same** species
- **Inter-class (red):** pairwise distances between images of **different** species

For clustering to be possible, inter-class distances must be significantly larger than intra-class distances (no overlap). If the distributions overlap heavily, the embedding space does not separate species well.

**Sampling:** O(N²) pairwise distances are expensive. The pipeline samples ≤ 20 images per class and ≤ 20,000 total pairs.

### Method Comparison (`method_comparison.png`)

Grouped bar chart with one group per method and three bars per group (Accuracy, ARI, NMI). Allows immediate visual comparison of all 8 methods on a single figure.

---

## 10. Configuration System (YAML)

All experiments are fully specified by YAML config files. No code changes are needed to run a new dataset, model, or hyperparameter combination.

### Full Config Schema

```yaml
dataset:
  name: <str>                           # Short identifier (e.g. "eumoth")
  format: imagefolder | cub | <plugin> # Data loader to use
  root: <path>                          # Root directory
  image_extensions: [jpg, jpeg, png]   # Allowed file extensions
  num_classes: <int>                    # Number of species (for reporting)

model:
  name: bioclip2 | bioclip | clip | dinov2
  pretrained: <hub_path_or_name>        # e.g. hf-hub:imageomics/bioclip-2
  arch: <str>                           # Optional: only for CLIP (ViT-L-14)
  normalize_embeddings: true            # Apply L2 normalisation
  batch_size: 64                        # GPU batch size (16 for DINOv2)
  num_workers: 8                        # DataLoader workers
  prompt_templates:                     # For text-guided methods (optional)
    - "{name}"
    - "a photo of {name}"
    - "a photo of {name}, a species"
    - "{genus} {name}, Lepidoptera"

clustering:
  methods:
    - name: kmeans
      n_clusters: 200
      n_init: 50
      random_state: 42
      max_iter: 500

    - name: kmeans_text_guided          # Skip if using DINOv2
      n_clusters: 200
      n_init: 1
      random_state: 42
      max_iter: 300

    - name: zero_shot                   # Skip if using DINOv2

    - name: agglomerative
      n_clusters: 200
      linkage: ward

    - name: spectral
      n_clusters: 200
      n_init: 10
      random_state: 42
      n_neighbors: 15

    - name: gmm
      n_clusters: 200
      covariance_type: diag             # 'full' infeasible for ≥512D
      random_state: 42
      max_iter: 300

    - name: dbscan                      # Full pipeline in dbscan: block
    - name: hdbscan                     # Full pipeline in hdbscan: block

evaluation:
  top_k_confusions: 20                  # How many confusion pairs to report

visualization:
  method: umap | tsne
  umap:
    n_neighbors: 30
    min_dist: 0.1
    metric: cosine
    random_state: 42
  tsne:
    perplexity: 40
    random_state: 42
  subsample: 5000                       # Max scatter plot points
  dpi: 300
  export_pdf: true                      # Also export vector PDF

output_dir: results/eumoth              # Where to write all outputs

seed: 42                                # Global random seed

dbscan:
  pca_components: 100
  min_samples: 5                        # Rule of thumb: ln(N) ≈ 8 for N=2205
  eps_range: null                       # null = auto from k-distance knee
  final_eps: null                       # null = auto from sweep

hdbscan:
  pca_components: 100
  sweep_range: [2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
  cluster_selection_method: eom         # eom | leaf
  final_min_cluster_size: null          # null = auto from sweep
  min_samples: null                     # null = default to min_cluster_size
```

### Config Conventions per Model

| Model | `arch` needed? | Text methods? | `batch_size` |
|---|---|---|---|
| `bioclip2` | No | Yes | 64 |
| `bioclip` | No | Yes | 64 |
| `clip` | Yes (`ViT-L-14`) | Yes (generic templates) | 64 |
| `dinov2` | No | **No** | **16** |

---

## 11. CLI Pipeline & Data Flow

### Individual Steps

```bash
# Step 1: Extract embeddings (slow, GPU-accelerated)
python cli/extract.py --config configs/eumoth.yaml

# Step 2: Run all clustering methods
python cli/cluster.py --config configs/eumoth.yaml
python cli/cluster.py --config configs/eumoth.yaml --skip-hdbscan  # if hdbscan not installed

# Step 3: Evaluate + generate publication figures
python cli/evaluate.py --config configs/eumoth.yaml

# Step 4: Diagnostic report
python cli/diagnose.py --config configs/eumoth.yaml
```

### End-to-End (All Steps)

```bash
python experiments/run_pipeline.py --config configs/eumoth.yaml
```

### Resume / Partial Re-run

```bash
# Skip extraction (features.npz already exists)
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-extract

# Skip density methods (HDBSCAN not installed)
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-hdbscan

# Re-run evaluation only (clusters already computed)
python experiments/run_pipeline.py --config configs/eumoth.yaml \
    --skip-extract --skip-cluster --skip-diagnose
```

### Density Methods Only

```bash
python cli/density.py --config configs/eumoth.yaml
python cli/density.py --config configs/eumoth.yaml --skip-dbscan   # HDBSCAN only
python cli/density.py --config configs/eumoth.yaml --skip-hdbscan  # DBSCAN only
```

### Data Flow Diagram

```
Images on Disk (JPEG/PNG)
        │
        ▼
[dataset.py: load_dataset()]
  • Discover subdirs as classes (ImageFolder)
  • Sort alphabetically (deterministic)
  • Filter corrupted images (PIL verify)
  → (image_paths[], labels[], class_names[])
        │
        ▼
[extractors.py: extractor.extract()]
  • Open image → PIL RGB
  • Apply model preprocessing (Resize, CenterCrop, Normalize)
  • Stack batch tensor → GPU
  • Forward pass (no_grad)
  • L2-normalise
  • CPU → numpy
  → features.npz: {features: (N,D), labels: (N,), paths: (N,), class_names: (C,)}
        │
        ├─────────────────────────────────────────────────────┐
        │ [UMAP/t-SNE 2D projection]                          │
        │  • compute once, cache to features_2d.npy           │
        │  • reused for all density visualisations            │
        ▼                                                     ▼
[Partition methods]                             [Density methods]
  KMeans → clusters_kmeans.npz                  PCA(100) → features_pca
  TextKMeans → clusters_kmeans_text_guided.npz  ├── DBSCAN:
  ZeroShot → clusters_zero_shot.npz             │     k-dist plot
  Agglomerative → clusters_agglomerative.npz    │     eps sweep
  Spectral → clusters_spectral.npz              │     best eps → fit
  GMM → clusters_gmm.npz                        │     → dbscan_metrics.json
                                                └── HDBSCAN:
                                                      mcs sweep
                                                      best mcs → fit
                                                      → hdbscan_metrics.json
        │
        ▼
[metrics.py: evaluate()]
  For each method:
  • Exclude noise (cluster_id == -1)
  • ARI, NMI
  • Hungarian matching → accuracy
  • Per-class accuracy, per-cluster purity
  • Top-K confusion pairs
  → metrics.json, metrics_summary.csv
        │
        ▼
[plots.py: generate figures]
  • embedding_true_labels.png
  • embedding_clusters_<method>.png × 8
  • confusion_matrix_<method>.png × 8
  • top_confusions_<method>.png × 8
  • cluster_sizes_<method>.png × 8
  • per_class_accuracy_<method>.png × 8
  • intra_inter_distances.png
  • method_comparison.png
```

---

## 12. Datasets

### ImageFolder Format

Standard format where each subdirectory is a class (species):

```
dataset_root/
├── Abrostola_tripartita/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── Acronicta_psi/
│   └── ...
└── ...
```

Class names are the subdirectory names, sorted alphabetically for determinism. Labels are 0-indexed integers corresponding to alphabetical order.

### CUB-200-2011 Format

Uses annotation files provided with the dataset:
- `images.txt` — maps image_id to file path
- `image_class_labels.txt` — maps image_id to class_id
- `classes.txt` — maps class_id to class name

The loader reads all three files and reconstructs full image paths.

### Corrupted Image Filtering

Before extraction, all discovered image paths are validated:

```python
from PIL import Image
img = Image.open(path)
img.verify()   # reads header only, does not decode full image
```

Corrupted images (truncated files, invalid headers) are silently excluded. This prevents batch-level failures during extraction. The number of excluded images is logged.

### Dataset Plugin System

Custom dataset formats can be added without modifying core code:

```python
# In src/data/plugins.py
@register_plugin("myformat")
def load_myformat(root, extensions, config):
    # Return: (image_paths, labels, class_names)
    ...
```

Then in YAML:
```yaml
dataset:
  format: myformat
```

---

## 13. Cross-Model Comparison

`cli/compare_models.py` reads `metrics_summary.csv` from multiple result directories and produces cross-model comparison figures.

### Usage (PowerShell single line)

```powershell
# EU-Moth: all 4 models
python cli/compare_models.py --dataset "EU-Moth (200 species)" --results results/eumoth:BioCLIP-2 results/eumoth_bioclip:BioCLIP results/eumoth_clip:CLIP results/eumoth_dinov2:DINOv2 --output results/comparison_eumoth --pdf

# CUB-200: all 4 models
python cli/compare_models.py --dataset "CUB-200-2011 (200 birds)" --results results/cub200:BioCLIP-2 results/cub200_bioclip:BioCLIP results/cub200_clip:CLIP results/cub200_dinov2:DINOv2 --output results/comparison_cub200 --pdf

# MothDK: all 4 models
python cli/compare_models.py --dataset "MothSpeciesDK (351 moths)" --results results/mothDK:BioCLIP-2 results/mothDK_bioclip:BioCLIP results/mothDK_clip:CLIP results/mothDK_dinov2:DINOv2 --output results/comparison_mothDK --pdf
```

### Outputs

| File | Description |
|---|---|
| `comparison_table.csv` | Full results: rows = (model, method), columns = ARI/NMI/accuracy |
| `heatmap_ari.png` | Methods × Models heatmap (ARI), N/A cells (DINOv2 text methods) in grey |
| `heatmap_accuracy.png` | Methods × Models heatmap (accuracy) |
| `heatmap_nmi.png` | Methods × Models heatmap (NMI) |
| `bar_ari_by_method.png` | Grouped bar chart: all methods, bars per model |
| `bar_accuracy_by_method.png` | Same for accuracy |
| `bar_nmi_by_method.png` | Same for NMI |
| `model_best_summary.png` | Each model's best method and score |

---

## 14. Reproducibility

### Global Seed Control

`set_seed(seed)` in `src/utils/io.py` sets seeds at all levels:

```python
import random
import numpy as np
import torch
import os

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
```

`CUBLAS_WORKSPACE_CONFIG=:4096:8` is required for deterministic CUDA operations (cuBLAS GEMM kernels). Set before launching Python if needed:

```powershell
# Windows PowerShell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

### Deterministic File Discovery

Images are discovered by sorting subdirectories and files alphabetically:

```python
classes = sorted([d for d in os.scandir(root) if d.is_dir()])
```

This ensures identical ordering across platforms and runs — no dependence on filesystem inode order.

### 2D Projection Caching

UMAP and t-SNE are stochastic. The 2D projection is computed once and cached to `features_2d.npy`. All subsequent CLI steps load the cached projection rather than recomputing it, ensuring visualisations are consistent across runs.

### Fixed Hyperparameters in Config

All randomness-carrying algorithms specify explicit `random_state` in the YAML config:

```yaml
- name: spectral
  random_state: 42
umap:
  random_state: 42
```

Given identical `features.npz`, identical config, and identical seed, two runs produce byte-identical `.npz` outputs and identical metric values.

---

## 15. Performance Considerations

### Computational Complexity

| Method | Time Complexity | Space Complexity | Notes |
|---|---|---|---|
| K-Means | O(N·K·D·n_iter) | O(N·D + K·D) | n_iter ≤ 500 |
| Text-KMeans | O(N·K·D·n_iter) | O(N·D + K·D) | n_iter ≤ 300, 1 restart |
| Zero-shot | O(N·C·D) | O(N·D + C·D) | Single forward pass |
| GMM (diag) | O(N·K·D·n_iter) | O(N·D + K·D) | n_iter ≤ 300 |
| Agglomerative | O(N² log N) | O(N²) | |
| Spectral | O(N²·K) | **O(N²)** | Not recommended for N>5,000 |
| DBSCAN (PCA+sweep) | O(N log N) per eps | O(N·p) | p=100 PCA dims |
| HDBSCAN (PCA+sweep) | O(N log N) per mcs | O(N·p) | p=100 PCA dims |

### Feature Extraction Wall Time (Approximate)

| Model | Dataset | Images | GPU (RTX 3080) | Notes |
|---|---|---|---|---|
| BioCLIP-2 | EU-Moth | 2,205 | ~3 min | batch=64 |
| BioCLIP | EU-Moth | 2,205 | ~2 min | batch=64 |
| CLIP | EU-Moth | 2,205 | ~3 min | batch=64 |
| DINOv2 | EU-Moth | 2,205 | ~10 min | batch=16, 518×518 |

DINOv2 is ~3–4× slower due to smaller batch size and larger input resolution.

### Subsampling for Expensive Operations

| Operation | Max samples | Reason |
|---|---|---|
| 1-NN accuracy | 3,000 | O(N²) with KD-tree |
| Silhouette score | 3,000 | O(N²) pairwise distances |
| Intra/inter distances | 20 per class | O(N²) pairwise |
| 2D scatter plots | 5,000–6,000 | Visual readability |

---

## 16. Key Design Decisions & Rationales

### Decision 1: PCA (not UMAP) as input to density methods

**Chosen:** PCA (100D) before DBSCAN/HDBSCAN
**Rejected:** UMAP-50D (used in original notebook)
**Reason:** UMAP invents density structure (compression artifacts); DBSCAN then finds UMAP artifacts rather than real biological clusters. PCA preserves global Euclidean distance structure. See Section 6.1.

### Decision 2: Coverage-Weighted ARI for parameter selection

**Chosen:** `weighted = ARI × (1 - noise_pct)` as selection criterion
**Rejected:** Raw ARI alone
**Reason:** Raw ARI rewards degenerate configurations (tiny perfect clusters, 97% noise). Coverage-weighted ARI selects configurations that cluster most of the data while maintaining quality. See Section 6.4.

### Decision 3: Noise point exclusion in all density metrics

**Chosen:** Exclude label=-1 before ARI/NMI/accuracy
**Rejected:** Include noise as its own cluster
**Reason:** Including noise creates a phantom "cluster -1" containing images from all 200 species, massively inflating the denominator in ARI. See Section 6.5.

### Decision 4: diag covariance for GMM

**Chosen:** `covariance_type: diag`
**Rejected:** `full`
**Reason:** `full` covariance with D=768, K=200 requires ~950 MB of parameters and nearly-singular matrix inversions (N=2205 ≪ D²=589,824). `diag` reduces this to 1.2 MB and is numerically stable. See Section 5.4.

### Decision 5: Text encoder loaded once, shared

**Chosen:** Pre-encode text centroids once before running clustering step
**Rejected:** Re-encode per method
**Reason:** If both `kmeans_text_guided` and `zero_shot` are in the config, the text encoder is loaded and run only once. The resulting `text_centroids` array is passed to both methods. This avoids a second expensive model load.

### Decision 6: 2D projection cached to disk

**Chosen:** Compute once → `features_2d.npy` → reuse
**Rejected:** Recompute per CLI step
**Reason:** UMAP on N=2,205 images (EU-Moth) or N=11,788 (CUB-200) takes significant time. All CLI steps (cluster.py, evaluate.py, diagnose.py) need the same projection for visualisation. Caching avoids redundant computation and ensures consistency (same 2D positions across all figures).

### Decision 7: Alphabetical class sorting

**Chosen:** Sort subdirectories and files alphabetically
**Rejected:** Filesystem order (inode-dependent)
**Reason:** Different operating systems (or the same OS after deletion/rewrite) may return directory entries in different orders. Alphabetical sorting ensures identical label assignments across machines and runs — critical for reproducibility and for comparing results between colleagues.

### Decision 8: DINOv2 configs exclude text-dependent methods

**Chosen:** Omit `kmeans_text_guided` and `zero_shot` from DINOv2 configs
**Rejected:** Include and let them fail at runtime
**Reason:** Clean failure avoidance. DINOv2 has no text encoder. Including text methods would cause `AttributeError` during clustering. Explicit exclusion in config is clearer and prevents wasted time.

### Decision 9: Separate config file per (dataset × model)

**Chosen:** 12 config files (3 datasets × 4 models)
**Rejected:** Single config with model list
**Reason:** Each combination has different optimal hyperparameters (`batch_size`, prompt templates, DINOv2 incompatibilities). Separate files make each experiment self-contained and independently re-runnable. Output directories are also cleanly separated.

### Decision 10: Spot-on-grey scatter for large label counts

**Chosen:** Grey backdrop + 12 highlighted classes for >20 classes
**Rejected:** Per-class colour for all 200 classes
**Reason:** With 200 species, no colour palette produces distinguishable colours. Plotting all 200 in distinct colours produces illegible rainbow noise. The grey-backdrop-with-highlights strategy communicates the overall embedding structure (via the grey cloud) while showing specific examples (via the coloured highlights).

---

## 17. Dependencies

### Core (required)

```
torch>=2.0.0
torchvision>=0.15.0
open_clip_torch>=2.24.0     # BioCLIP-2, BioCLIP, CLIP
scikit-learn>=1.3.0         # clustering, metrics, PCA
scipy>=1.11.0               # Hungarian algorithm (linear_sum_assignment)
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
tqdm>=4.65.0
PyYAML>=6.0
Pillow>=10.0.0
```

### Optional

```
umap-learn>=0.5.5           # UMAP 2D projection (falls back to t-SNE if missing)
hdbscan>=0.8.33             # HDBSCAN clustering (skip with --skip-hdbscan if missing)
timm>=0.9.0                 # DINOv2 extractor (only needed for DINOv2 configs)
```

### Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
pip install umap-learn hdbscan timm   # optional but recommended
```

---

## 18. Running the Full Pipeline

### Single Experiment (EU-Moth + BioCLIP-2)

```bash
python experiments/run_pipeline.py --config configs/eumoth.yaml
```

### All Models on EU-Moth

```bash
python experiments/run_pipeline.py --config configs/eumoth.yaml
python experiments/run_pipeline.py --config configs/eumoth_bioclip.yaml
python experiments/run_pipeline.py --config configs/eumoth_clip.yaml
python experiments/run_pipeline.py --config configs/eumoth_dinov2.yaml
```

### All Models on All Three Datasets (12 experiments total)

```bash
for cfg in configs/eumoth.yaml configs/eumoth_bioclip.yaml configs/eumoth_clip.yaml configs/eumoth_dinov2.yaml configs/cub200.yaml configs/cub200_bioclip.yaml configs/cub200_clip.yaml configs/cub200_dinov2.yaml configs/mothDK.yaml configs/mothDK_bioclip.yaml configs/mothDK_clip.yaml configs/mothDK_dinov2.yaml; do
    python experiments/run_pipeline.py --config $cfg
done
```

### Cross-Model Comparison (after all pipelines complete)

```powershell
# EU-Moth
python cli/compare_models.py --dataset "EU-Moth (200 species)" --results results/eumoth:BioCLIP-2 results/eumoth_bioclip:BioCLIP results/eumoth_clip:CLIP results/eumoth_dinov2:DINOv2 --output results/comparison_eumoth --pdf

# CUB-200
python cli/compare_models.py --dataset "CUB-200-2011 (200 birds)" --results results/cub200:BioCLIP-2 results/cub200_bioclip:BioCLIP results/cub200_clip:CLIP results/cub200_dinov2:DINOv2 --output results/comparison_cub200 --pdf

# MothDK
python cli/compare_models.py --dataset "MothSpeciesDK (351 moths)" --results results/mothDK:BioCLIP-2 results/mothDK_bioclip:BioCLIP results/mothDK_clip:CLIP results/mothDK_dinov2:DINOv2 --output results/comparison_mothDK --pdf
```

### Skip Flags Reference

| Flag | Effect |
|---|---|
| `--skip-extract` | Skip feature extraction (use cached `features.npz`) |
| `--skip-cluster` | Skip clustering step |
| `--skip-evaluate` | Skip evaluation + figure generation |
| `--skip-diagnose` | Skip diagnostics report |
| `--skip-density` | Skip both DBSCAN and HDBSCAN |
| `--skip-dbscan` | Skip DBSCAN only |
| `--skip-hdbscan` | Skip HDBSCAN only |

### Output Directory Contents After Full Run

```
results/eumoth/
├── features.npz                        # (N, 768) BioCLIP-2 embeddings
├── features_2d.npy                     # (N, 2) UMAP projection (cached)
├── clusters_kmeans.npz
├── clusters_kmeans_text_guided.npz
├── clusters_zero_shot.npz
├── clusters_agglomerative.npz
├── clusters_spectral.npz
├── clusters_gmm.npz
├── metrics.json                        # Full metrics dict (all methods)
├── metrics_summary.csv                 # One-row-per-method table
├── dbscan_metrics.json
├── dbscan_sweep.csv
├── hdbscan_metrics.json
├── hdbscan_sweep.csv
├── density_metrics_combined.json
├── density_metrics_summary.csv
├── diagnostics_report.txt
└── figures/
    ├── embedding_true_labels.png
    ├── intra_inter_distances.png
    ├── method_comparison.png
    ├── embedding_clusters_kmeans.png
    ├── confusion_matrix_kmeans.png
    ├── top_confusions_kmeans.png
    ├── cluster_sizes_kmeans.png
    ├── per_class_accuracy_kmeans.png
    ├── [same 5 figures for each of 8 methods]
    ├── dbscan_kdistance_plot.png
    ├── dbscan_eps_sweep.png
    ├── dbscan_umap_scatter.png
    ├── dbscan_cluster_sizes.png
    ├── dbscan_top_confusions.png
    ├── hdbscan_sweep.png
    ├── hdbscan_umap_scatter.png
    ├── hdbscan_outlier_scores.png
    ├── hdbscan_cluster_sizes.png
    ├── hdbscan_top_confusions.png
    ├── density_method_comparison.png
    └── diagnostic_overview.png
```

---

*End of documentation. All mathematics, implementation details, design decisions, and usage instructions are derived directly from the source code.*
