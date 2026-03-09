# Product Requirements Document (PRD)
## BioCLIP Clustering Pipeline

**Version:** 1.1
**Date:** 2026-03-09
**Domain:** Computational Biology / Computer Vision Research
**Author:** Shakil Ahmed

---

## 1. Overview

### 1.1 Purpose

The BioCLIP Clustering Pipeline is a modular, reproducible research system for evaluating whether biological images can be automatically grouped by species using visual embeddings from foundation models — without any labeled training data. The system extracts high-dimensional image embeddings using BioCLIP-2, applies multiple unsupervised clustering algorithms, and evaluates their accuracy against ground-truth species labels.

The primary use case is a **thesis experiment**: comparing clustering performance across algorithms and datasets to determine how well vision-language models pre-trained on biological data encode species-level visual distinctions.

### 1.2 Problem Statement

Fine-grained species recognition is expensive to scale because it requires expert-annotated labels. If a foundation model pre-trained on biological image-text pairs (e.g., BioCLIP-2 on TreeOfLife-200M) already encodes species identity in its embedding space, then unsupervised clustering could become a viable path to zero-shot species grouping.

**Research question:** Do BioCLIP-2 embeddings cluster into species-level groups without any supervision, and which clustering algorithm best recovers the ground-truth taxonomy?

### 1.3 Scope

- **In scope:** Feature extraction, clustering, evaluation, visualization, diagnostics, configuration management, and reproducibility tooling.
- **Out of scope:** Model fine-tuning, active learning, web-based UI, real-time inference, production deployment.

---

## 2. Goals and Non-Goals

### 2.1 Goals

| # | Goal |
|---|------|
| G1 | Extract L2-normalized image embeddings using BioCLIP-2 (and optionally BioCLIP, CLIP, DINOv2) in a single consistent interface. |
| G2 | Run seven clustering algorithms on the same feature set: K-Means, Text-Guided K-Means, Spectral, Agglomerative, DBSCAN, HDBSCAN, and GMM. |
| G3 | Evaluate every clustering result with ARI, NMI, Hungarian accuracy, per-class recall, and cluster purity. For density methods, use coverage-weighted ARI (ARI × coverage fraction) as the primary comparison metric. |
| G4 | Produce publication-ready figures (300 DPI PNG + PDF) for every experiment. |
| G5 | Enforce full reproducibility via fixed seeds and deterministic GPU settings. |
| G6 | Support any image-folder–style dataset by adding only a YAML config (no code changes). |
| G7 | Provide a CLI that allows each pipeline stage to be run independently or end-to-end, with granular skip flags for resuming partial runs. |

### 2.2 Non-Goals

- Real-time or online inference
- Supervised classification
- Hyperparameter search beyond the built-in density-method sweeps
- Multi-GPU / distributed training
- A web or GUI interface

---

## 3. Stakeholders

| Role | Person/Group | Interest |
|------|-------------|----------|
| Primary author | Thesis student | Build, run, and interpret experiments |
| Thesis supervisor | Academic advisor | Reproducibility, methodological rigor |
| Examiners / reviewers | Academic committee | Clarity of results, figure quality |
| Future researchers | Open-source community | Reusability for other biological datasets |

---

## 4. User Personas

### Persona A — Thesis Researcher
- Runs experiments frequently (daily), adjusting hyperparameters and datasets.
- Needs deterministic results, fast iteration, and publication-quality figures.
- Comfortable with the command line and Python.

### Persona B — Collaborator / Reviewer
- Reads the code and README to understand methodology.
- Needs clear documentation and reproducible results.
- May re-run experiments to verify findings.

### Persona C — Future Researcher
- Adapts the pipeline to a new biological dataset.
- Needs minimal friction: ideally just a new YAML config.

---

## 5. Functional Requirements

### 5.1 Feature Extraction (`cli/extract.py`)

| ID | Requirement |
|----|-------------|
| FR-E1 | Load images from an `imagefolder` directory (one subfolder per class) or a CUB-style metadata structure. |
| FR-E2 | Support model backends: BioCLIP-2, BioCLIP, CLIP (via `open_clip`), and DINOv2 (via `timm`). |
| FR-E3 | Extract embeddings in configurable mini-batches with GPU acceleration when available. |
| FR-E4 | Optionally L2-normalize all embeddings so that Euclidean distance equals cosine distance. |
| FR-E5 | Save embeddings, integer class labels, image paths, and class names to `features.npz`. |
| FR-E6 | Filter corrupted or unreadable images before extraction using a fast PIL header-only check (`Image.verify()`). Corrupted files are skipped and logged; no zero-vector placeholders are inserted (clean data only). Within-batch load failures still use a zero-vector placeholder to maintain batch alignment. |

### 5.2 Clustering (`cli/cluster.py`)

| ID | Requirement |
|----|-------------|
| FR-C1 | Load pre-computed features from `features.npz`. |
| FR-C2 | Run all methods listed in `clustering.methods` in the YAML config. |
| FR-C3 | Support K-Means with configurable `n_clusters`, `n_init`, `max_iter`. |
| FR-C4 | Support Text-Guided K-Means: initialize centroids from BioCLIP-2 text embeddings of class names using an ensemble of prompt templates (configurable in YAML). Each class gets one centroid from averaging N template embeddings, re-normalized to unit sphere. |
| FR-C5 | Support Spectral Clustering with nearest-neighbor affinity; suppress known connectivity warnings. |
| FR-C6 | Support Agglomerative (Ward linkage by default). |
| FR-C7 | Support DBSCAN with automatic `eps` selection via k-distance knee detection and optional grid sweep. Applies PCA dimensionality reduction before clustering (not UMAP — UMAP distorts the density structure DBSCAN relies on). |
| FR-C8 | Support HDBSCAN with automatic `min_cluster_size` selection via grid sweep. `min_cluster_size` must be ≥ 2; value of 1 is mathematically undefined and will raise an error. |
| FR-C9 | Support Gaussian Mixture Models (GMM) with configurable covariance type; hard-assigned at evaluation. |
| FR-C10 | Save cluster assignment arrays to per-method `.npz` files (`clusters_<method>.npz`). |
| FR-C11 | For density methods, report coverage-weighted ARI (ARI × coverage fraction) as the comparison verdict metric, in addition to raw ARI, NMI, and accuracy. |
| FR-C12 | Support granular skip flags: `--skip-density`, `--skip-dbscan`, `--skip-hdbscan` to omit individual density methods without modifying the config. |

### 5.3 Evaluation (`cli/evaluate.py`)

| ID | Requirement |
|----|-------------|
| FR-V1 | Compute Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) for every method. |
| FR-V2 | Compute Hungarian-matched accuracy: find the optimal one-to-one cluster→class mapping via the linear assignment algorithm. |
| FR-V3 | Compute per-class accuracy and per-cluster purity as DataFrames. |
| FR-V4 | Identify the top-K most common species confusion pairs. |
| FR-V5 | Exclude noise points (cluster label = −1 from DBSCAN/HDBSCAN) from all metric computations and report the count of excluded points. |
| FR-V6 | Save all metrics to `metrics.json` and a comparative summary to `metrics_summary.csv`. |

### 5.4 Visualization (`cli/evaluate.py`, `src/visualization/plots.py`)

| ID | Requirement |
|----|-------------|
| FR-P1 | Generate UMAP or t-SNE 2D scatter plots of embeddings, colored by true labels and by cluster assignments. t-SNE is used as fallback when `umap-learn` is not installed. |
| FR-P2 | For datasets with >20 classes, render a grey backdrop of all points and spotlight the 12 most populated classes with distinct colors. For datasets ≤20 classes, all classes are fully colored. |
| FR-P3 | Generate a two-panel confusion figure: per-class recall bar chart (worst 40 classes) and a most-confused species submatrix. For datasets ≤30 classes, generate a full annotated confusion matrix instead. |
| FR-P4 | Generate a horizontal bar chart of the top confusion pairs (True → Predicted). |
| FR-P5 | Generate a two-panel cluster size figure: sorted bar chart and histogram. |
| FR-P6 | Generate a violin + box plot comparing intra-class vs. inter-class pairwise distances (subsampled to ≤20 images/class and ≤20,000 pairs for speed). |
| FR-P7 | Generate a grouped bar chart comparing Accuracy, ARI, and NMI across all methods. |
| FR-P8 | Generate a per-species accuracy figure: distribution histogram and worst-20 bar chart. |
| FR-P9 | Save all figures at 300 DPI as PNG, and optionally as PDF, in `output_dir/figures/`. |
| FR-P10 | Each figure must include an embedded plain-text caption explaining how to read it. |

### 5.5 Diagnostics (`cli/diagnose.py`)

| ID | Requirement |
|----|-------------|
| FR-D1 | Compute and report feature quality statistics: sample count, embedding dimension, mean/std/min/max L2 norm, mean feature variance, and number of zero-variance (dead) dimensions. Warn if mean norm deviates from 1.0 by more than 0.05, or if dead dimensions exist. |
| FR-D2 | Compute and report 1-NN accuracy using cosine distance and 3-fold cross-validation (subsampled to ≤3,000 images for speed). This is a supervision-free separability upper bound. |
| FR-D3 | Compute and report silhouette score on the 2D UMAP/t-SNE projection (subsampled to ≤3,000 images). Values > 0.5 = strong separation; > 0.2 = moderate; ≤ 0.2 = weak. |
| FR-D4 | Run a quick K-Means baseline (n_init=10) and report Hungarian accuracy, ARI, and NMI as a sanity check. |
| FR-D5 | Save a 2×2 diagnostic figure: true-label scatter, cluster scatter, 2D norm distribution, cluster size histogram. |
| FR-D6 | Save a plain-text `diagnostics_report.txt` with all computed statistics. |

### 5.6 Standalone Density CLI (`cli/density.py`)

| ID | Requirement |
|----|-------------|
| FR-DN1 | Provide a standalone `cli/density.py` command that runs only the DBSCAN and HDBSCAN pipelines, independently of `cli/cluster.py`. |
| FR-DN2 | Support `--skip-dbscan` and `--skip-hdbscan` flags for running one density method at a time. |
| FR-DN3 | Print a side-by-side comparison table with coverage-weighted ARI verdict when both methods run. |
| FR-DN4 | Save `density_metrics_combined.json` and `density_metrics_summary.csv` with per-method results and weighted scores. |

### 5.7 Configuration System

| ID | Requirement |
|----|-------------|
| FR-CF1 | All experiments are specified via a YAML config file; no code changes are needed for a new dataset. |
| FR-CF2 | Config sections: `dataset`, `model`, `clustering`, `evaluation`, `visualization`, `dbscan`, `hdbscan`, `output_dir`, `seed`. |
| FR-CF3 | Prompt templates for text-guided K-Means are configurable in the YAML under `model.prompt_templates`. Templates support `{name}` (full species name) and `{genus}` (first word of name) placeholders. |
| FR-CF4 | The `dbscan` block controls PCA dimensionality reduction, `min_samples`, and optional `eps_range` / `final_eps` overrides. |
| FR-CF5 | The `hdbscan` block controls `sweep_range` (all values must be ≥ 2) and optional `final_min_cluster_size` override. |

### 5.8 Dataset Plugins

| ID | Requirement |
|----|-------------|
| FR-DS1 | Built-in support for `imagefolder` format (class name = subfolder name). |
| FR-DS2 | Built-in support for CUB-200-2011 format with metadata files. |
| FR-DS3 | A plugin registry (`src/data/plugins.py`) supports custom dataset formats via the `@register_plugin(key)` decorator. An iNaturalist-style JSON annotation plugin is pre-registered as a working example (`format: inaturalist`). |

---

## 6. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Reproducibility | Setting `seed` in the config fixes all random state (Python, NumPy, PyTorch). Setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` ensures deterministic GPU operations. The env var is also set programmatically at startup as a fallback. |
| NFR-2 | Performance | Feature extraction must support batched GPU inference. Spectral clustering warns the user when N > 5000 (O(N²) memory). Intra/inter distance computation is subsampled. 2D projections are cached to `features_2d.npy` and reused across pipeline stages. |
| NFR-3 | Portability | Must run on CPU-only machines (falls back from CUDA automatically). |
| NFR-4 | Extensibility | New clustering methods can be added by registering a function in `METHOD_MAP` in `src/clustering/methods.py`. New extractors can be added by subclassing `BaseExtractor`. New dataset formats can be added via `@register_plugin` in `src/data/plugins.py`. |
| NFR-5 | Output quality | All figures exported at 300 DPI; PNG for thesis inclusion; PDF for vector-quality publication. |
| NFR-6 | Graceful degradation | Missing optional packages (`hdbscan`, `umap-learn`, `timm`) raise clear `ImportError` messages. The `--skip-hdbscan` flag allows running without the `hdbscan` package. UMAP falls back to t-SNE automatically if `umap-learn` is not installed. |
| NFR-7 | Interpretability | All figures include embedded captions explaining how to read them. All code modules include "how to read" and "design rationale" docstrings. |

---

## 7. System Architecture

```
bioclip_clustering/
│
├── cli/                    # CLI entry points (one per pipeline stage)
│   ├── extract.py          # Stage 1: extract embeddings
│   ├── cluster.py          # Stage 2: run all clustering algorithms
│   ├── evaluate.py         # Stage 3: compute metrics and generate plots
│   ├── diagnose.py         # Stage 4: diagnostic report
│   └── density.py          # Standalone: DBSCAN + HDBSCAN only
│
├── experiments/
│   ├── run_pipeline.py          # End-to-end runner (calls all 4 stages)
│   └── run_density_clustering.py  # Backwards-compatible wrapper for cli/density.py
│
├── src/
│   ├── models/
│   │   └── extractors.py   # BaseExtractor, OpenCLIPExtractor, DINOv2Extractor, build_extractor()
│   ├── data/
│   │   ├── dataset.py      # Image discovery, label indexing, corrupted-image filter
│   │   └── plugins.py      # Dataset plugin registry + iNaturalist example
│   ├── clustering/
│   │   ├── methods.py      # Unified clustering API + METHOD_MAP (KMeans, Spectral, Agglomerative, DBSCAN, HDBSCAN, GMM)
│   │   └── density/
│   │       ├── dbscan_pipeline.py   # DBSCAN + k-distance + sweep + figures
│   │       └── hdbscan_pipeline.py  # HDBSCAN + sweep + outlier scores + figures
│   ├── evaluation/
│   │   └── metrics.py      # ARI, NMI, Hungarian accuracy, purity, confusion
│   ├── visualization/
│   │   └── plots.py        # All 8+ publication-ready figure functions
│   ├── diagnostics/
│   │   └── suite.py        # 1-NN accuracy, silhouette, feature stats, quick K-Means
│   └── utils/
│       ├── io.py            # NPZ save/load, seed control, config loading
│       └── pretty.py        # Console formatting helpers (headers, tables, verdicts)
│
├── configs/
│   ├── eumoth.yaml          # EU-Moth (200 species) config
│   ├── cub200.yaml          # CUB-200-2011 (200 bird species) config
│   └── mothDK.yaml          # MothSpeciesDK (351 Danish moth species) config
│
└── results/                 # Auto-created; not version-controlled
    └── <dataset>/
        ├── features.npz
        ├── features_2d.npy              # Cached 2D UMAP/t-SNE projection
        ├── clusters_<method>.npz
        ├── metrics.json
        ├── metrics_summary.csv
        ├── density_metrics_combined.json
        ├── density_metrics_summary.csv
        ├── diagnostics_report.txt
        └── figures/
            └── *.png / *.pdf
```

### Data Flow

```
Images on disk
      │
      ▼
[extract.py]  →  features.npz  (N × D embeddings + labels + paths)
                 features_2d.npy  (cached 2D projection, computed once)
      │
      ▼
[cluster.py]  →  clusters_<method>.npz  (N integer assignments, per method)
               →  density_metrics_combined.json / _summary.csv
               →  figures/dbscan_*.png, hdbscan_*.png
      │
      ▼
[evaluate.py] →  metrics.json, metrics_summary.csv, figures/*.png
      │
      ▼
[diagnose.py] →  diagnostics_report.txt, figures/diagnostic_overview.png
```

---

## 8. Datasets

| Dataset | Classes | Images | Format | Notes |
|---------|---------|--------|--------|-------|
| EU-Moth | 200 European moth species | ~2,200 | `imagefolder` | ~11 images/class |
| MothSpeciesDK | 351 Danish moth species | 18,999 | `imagefolder` | ~54 images/class |
| CUB-200-2011 | 200 bird species | ~11,788 | CUB metadata | Standard fine-grained benchmark |

EU-Moth and CUB-200-2011 share the same class count (200), making direct algorithm comparison meaningful. MothSpeciesDK tests pipeline scalability with a larger class count and more images per class.

---

## 9. Models Supported

| Model ID | Architecture | Embedding Dim | Pretrained On |
|----------|-------------|---------------|---------------|
| `bioclip2` | ViT-L/14 | 768 | TreeOfLife-200M (biology) |
| `bioclip` | ViT-B/16 | 512 | TreeOfLife-10M (biology) |
| `clip` | ViT-L/14 | 768 | LAION-2B (general) |
| `dinov2` | ViT-L/14 | 1024 | LVD-142M (general, self-supervised) |

BioCLIP-2 is the primary model; others serve as ablation baselines. DINOv2 is incompatible with `kmeans_text_guided` (no text encoder).

---

## 10. Clustering Algorithms

| Method | Requires K | Discovers K | Notes |
|--------|-----------|------------|-------|
| K-Means | Yes | No | Baseline; 50 restarts |
| K-Means (Text-Guided) | Yes | No | Text centroid initialization via BioCLIP-2 text encoder; prompt ensembling |
| Spectral | Yes | No | Nearest-neighbor affinity; O(N²) memory |
| Agglomerative | Yes | No | Ward linkage; produces compact clusters |
| DBSCAN | No | Yes | Requires PCA pre-processing; eps via k-distance knee |
| HDBSCAN | No | Yes | Hierarchical density; more robust than DBSCAN; min_cluster_size ≥ 2 |
| GMM | Yes | No | Soft clustering, hard-assigned at evaluation |

---

## 11. Evaluation Metrics

| Metric | Range | Interpretation |
|--------|-------|---------------|
| ARI (Adjusted Rand Index) | [−1, 1] | 0 = random; 1 = perfect; adjusted for chance |
| NMI (Normalized Mutual Information) | [0, 1] | Shared information between clusters and labels |
| Hungarian Accuracy | [0, 1] | Fraction correct under optimal cluster→class assignment |
| Per-Class Recall | [0, 1] | Per-species fraction correctly assigned |
| Per-Cluster Purity | [0, 1] | Fraction of cluster belonging to its dominant class |
| Coverage-Weighted ARI | [0, 1] | ARI × coverage fraction; primary verdict metric for density methods |

---

## 12. CLI Reference

```bash
# Stage 1: Extract embeddings
python cli/extract.py --config configs/eumoth.yaml

# Stage 2: Cluster (all methods)
python cli/cluster.py --config configs/eumoth.yaml

# Stage 3: Evaluate and plot
python cli/evaluate.py --config configs/eumoth.yaml

# Stage 4: Diagnostics
python cli/diagnose.py --config configs/eumoth.yaml

# All stages in sequence
python experiments/run_pipeline.py --config configs/eumoth.yaml

# Partial runs (skip flags)
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-extract
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-hdbscan
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-density
python experiments/run_pipeline.py --config configs/eumoth.yaml \
    --skip-extract --skip-cluster --skip-diagnose   # re-run evaluate only

# Standalone density clustering
python cli/density.py --config configs/eumoth.yaml
python cli/density.py --config configs/eumoth.yaml --skip-hdbscan
python cli/density.py --config configs/eumoth.yaml --skip-dbscan
```

---

## 13. Output Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| `features.npz` | `results/<dataset>/` | Embeddings (N×D), labels, image paths, class names |
| `features_2d.npy` | `results/<dataset>/` | Cached 2D UMAP/t-SNE projection; shared across pipeline stages |
| `clusters_<method>.npz` | `results/<dataset>/` | Cluster assignments per method |
| `metrics.json` | `results/<dataset>/` | Full metrics dict (JSON-safe) |
| `metrics_summary.csv` | `results/<dataset>/` | One-row-per-method summary table |
| `dbscan_metrics.json` | `results/<dataset>/` | DBSCAN sweep and final results |
| `dbscan_sweep.csv` | `results/<dataset>/` | DBSCAN eps sweep results table |
| `hdbscan_metrics.json` | `results/<dataset>/` | HDBSCAN sweep and final results |
| `hdbscan_sweep.csv` | `results/<dataset>/` | HDBSCAN min_cluster_size sweep results |
| `density_metrics_combined.json` | `results/<dataset>/` | Combined DBSCAN + HDBSCAN results with coverage-weighted ARI |
| `density_metrics_summary.csv` | `results/<dataset>/` | One-row-per-density-method summary including weighted_score |
| `diagnostics_report.txt` | `results/<dataset>/` | Plain-text diagnostic summary (feature stats, 1-NN accuracy, silhouette, K-Means baseline) |
| `figures/*.png` / `*.pdf` | `results/<dataset>/figures/` | All publication-ready figures |

### Generated Figures

| Figure | Content |
|--------|---------|
| `embedding_true_labels.png` | 2D UMAP/t-SNE, colored by ground-truth species |
| `embedding_clusters_<method>.png` | 2D UMAP/t-SNE, colored by cluster assignments |
| `confusion_matrix_<method>.png` | Per-class recall + most-confused species submatrix (full matrix for ≤30 classes) |
| `top_confusions_<method>.png` | Top-K misclassification pairs (True → Predicted) |
| `cluster_sizes_<method>.png` | Sorted bar chart + size histogram |
| `intra_inter_distances.png` | Intra-class vs. inter-class distance distributions (violin + box) |
| `method_comparison.png` | Grouped bar chart comparing Accuracy/ARI/NMI across all methods |
| `per_class_accuracy_<method>.png` | Distribution histogram + worst-20 species |
| `dbscan_kdistance_plot.png` | k-distance plot for eps selection |
| `dbscan_eps_sweep.png` | ARI/NMI/cluster count vs. eps sweep |
| `dbscan_umap_scatter.png` | UMAP scatter colored by DBSCAN assignments |
| `dbscan_cluster_sizes.png` | DBSCAN cluster size distribution |
| `dbscan_top_confusions.png` | Top confusion pairs for DBSCAN |
| `hdbscan_sweep.png` | HDBSCAN min_cluster_size sweep plot |
| `hdbscan_umap_scatter.png` | UMAP scatter colored by HDBSCAN assignments |
| `hdbscan_outlier_scores.png` | HDBSCAN per-point outlier score distribution |
| `density_method_comparison.png` | Side-by-side DBSCAN vs. HDBSCAN comparison figure |
| `diagnostic_overview.png` | 2×2 diagnostic panel (true labels, clusters, norm distribution, cluster sizes) |

---

## 14. Dependencies

| Package | Role | Optional |
|---------|------|----------|
| `torch` ≥ 2.0 | GPU inference | No |
| `open_clip_torch` | BioCLIP / CLIP extraction | No |
| `scikit-learn` | Clustering + metrics | No |
| `numpy`, `scipy` | Numerical computation | No |
| `matplotlib`, `seaborn` | Visualization | No |
| `pandas` | Tabular metrics | No |
| `tqdm` | Progress bars | No |
| `PyYAML` | Config parsing | No |
| `Pillow` | Image loading + validation | No |
| `umap-learn` | UMAP 2D projection (t-SNE fallback if absent) | Yes |
| `hdbscan` | HDBSCAN clustering | Yes |
| `timm` | DINOv2 extraction | Yes |
| `torchvision` | DINOv2 preprocessing | Yes |

---

## 15. Reproducibility Requirements

| Mechanism | Description |
|-----------|-------------|
| Global seed | Set via `seed` in YAML; applied to NumPy, Python random, and PyTorch (CPU + GPU) |
| Deterministic CUDA | `CUBLAS_WORKSPACE_CONFIG=:4096:8` set programmatically at startup; also set manually before running for full effect |
| Fixed K-Means restarts | `n_init=50` with fixed `random_state` |
| Fixed UMAP | `random_state` set in config |
| 2D projection cache | `features_2d.npy` is computed once and reused; avoids non-determinism from repeated UMAP runs |
| Committed configs | All YAML configs are version-controlled |
| Excluded results | `results/` directory is `.gitignore`d; only configs are committed |

---

## 16. Extensibility Guidelines

### Adding a new dataset
1. Create a new YAML config in `configs/`.
2. If the folder structure is non-standard, add a loader with `@register_plugin("mykey")` in `src/data/plugins.py`.
3. Run `python experiments/run_pipeline.py --config configs/<new>.yaml`.

### Adding a new clustering method
1. Implement a function `run_<method>(features, params, **kwargs) -> np.ndarray`.
2. Register it in `METHOD_MAP` in `src/clustering/methods.py`.
3. Add the method to the `clustering.methods` list in the relevant YAML config.

### Adding a new embedding model
1. Subclass `BaseExtractor` in `src/models/extractors.py`.
2. Implement `_load_model()` and `extract(image_paths)`.
3. Register the new model name in `build_extractor()`.

---

## 17. Constraints and Assumptions

| Constraint | Detail |
|------------|--------|
| Image format | JPEG, PNG supported; images must be RGB-convertible. Corrupted files are filtered before extraction via PIL header validation. |
| Dataset structure | Primary format assumes one subfolder per class (ImageFolder). Class names are derived from subfolder names; sorted alphabetically for deterministic label assignment. |
| Cluster count | For K-anchored methods, K is set to the number of ground-truth classes |
| HDBSCAN min_cluster_size | Must be ≥ 2. A value of 1 is undefined by the HDBSCAN algorithm and will raise a `ValueError`. Datasets with single-image classes will have those images treated as noise or absorbed into nearby clusters. |
| Evaluation | Requires ground-truth labels; purely unsupervised metrics (Silhouette, Davies-Bouldin) are used only in diagnostics, not as primary evaluation metrics |
| Spectral scaling | Not recommended for N > 5,000 due to O(N²) affinity matrix |
| Text-guided K-Means | Requires the extractor to be an `OpenCLIPExtractor` (BioCLIP family); incompatible with DINOv2 |
| Density method evaluation | Noise points (label = −1) are excluded from ARI/NMI/accuracy calculations. Coverage-weighted ARI accounts for this exclusion. |

---

## 18. Success Criteria

| Criterion | Target |
|-----------|--------|
| Reproduction | Given the same config and dataset, two independent runs produce byte-identical `.npz` files |
| Coverage | All 7 clustering methods complete without error on EU-Moth, CUB-200-2011, and MothSpeciesDK |
| Figure quality | All figures render cleanly at 300 DPI with readable labels |
| Extensibility | A collaborator can add a new dataset (ImageFolder format) by editing only a YAML file |
| Documentation | Every source file includes a module docstring explaining its purpose and every public function includes a docstring with a "how to read" or "design rationale" section |

---

## 19. Known Limitations

| Limitation | Mitigation |
|------------|-----------|
| DBSCAN and HDBSCAN produce noise points (label = −1), reducing the evaluated sample | Noise points are reported and excluded from metrics; coverage-weighted ARI captures the trade-off |
| Spectral clustering is O(N²) in memory | Documented; user warned; recommended N ≤ 5,000 |
| Text-guided K-Means depends on text-image alignment quality of BioCLIP-2 | Template ensembling reduces prompt sensitivity; templates are configurable per dataset |
| Per-class accuracy is undefined if K < true class count | Hungarian algorithm handles non-square cost matrices |
| 200+ classes exceed the perceptual limit of distinguishable colors | Spotlight-on-grey rendering strategy used for embedding plots |
| Single-image classes cannot form their own HDBSCAN cluster | Expected behavior; such images become noise or merge with nearest cluster; min_cluster_size=1 is invalid |
| 2D projection (UMAP/t-SNE) is stochastic | Fixed `random_state` in config ensures same projection per config; cached to `features_2d.npy` after first run |

---

*End of PRD — v1.1*
