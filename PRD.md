# Product Requirements Document (PRD)
## BioCLIP Clustering Pipeline

**Version:** 1.0
**Date:** 2026-03-03
**Domain:** Computational Biology / Computer Vision Research

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
| G3 | Evaluate every clustering result with ARI, NMI, Hungarian accuracy, per-class recall, and cluster purity. |
| G4 | Produce publication-ready figures (300 DPI PNG + PDF) for every experiment. |
| G5 | Enforce full reproducibility via fixed seeds and deterministic GPU settings. |
| G6 | Support any image-folder–style dataset by adding only a YAML config (no code changes). |
| G7 | Provide a CLI that allows each pipeline stage to be run independently or end-to-end. |

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
| FR-E5 | Save embeddings, integer class labels, and image paths to `features.npz`. |
| FR-E6 | Handle corrupt or unreadable images gracefully (log a warning; insert a zero vector as placeholder). |

### 5.2 Clustering (`cli/cluster.py`)

| ID | Requirement |
|----|-------------|
| FR-C1 | Load pre-computed features from `features.npz`. |
| FR-C2 | Run all methods listed in `clustering.methods` in the YAML config. |
| FR-C3 | Support K-Means with configurable `n_clusters`, `n_init`, `max_iter`. |
| FR-C4 | Support Text-Guided K-Means: initialize centroids from BioCLIP-2 text embeddings of class names using an ensemble of prompt templates. |
| FR-C5 | Support Spectral Clustering with nearest-neighbor affinity; suppress known connectivity warnings. |
| FR-C6 | Support Agglomerative (Ward linkage by default). |
| FR-C7 | Support DBSCAN with automatic `eps` selection via k-distance knee detection and an optional grid sweep. |
| FR-C8 | Support HDBSCAN with automatic `min_cluster_size` selection via grid sweep. |
| FR-C9 | Support Gaussian Mixture Models (GMM). |
| FR-C10 | Save cluster assignment arrays to per-method `.npz` files (`clusters_<method>.npz`). |

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
| FR-P1 | Generate UMAP or t-SNE 2D scatter plots of embeddings, colored by true labels and by cluster assignments. |
| FR-P2 | For datasets with >20 classes, render a grey backdrop of all points and spotlight the 12 most populated classes with distinct colors. |
| FR-P3 | Generate a two-panel confusion figure: per-class recall bar chart and a submatrix heatmap of the most-confused species. |
| FR-P4 | Generate a horizontal bar chart of the top confusion pairs (True → Predicted). |
| FR-P5 | Generate a two-panel cluster size figure: sorted bar chart and histogram. |
| FR-P6 | Generate a violin + box plot comparing intra-class vs. inter-class pairwise distances. |
| FR-P7 | Generate a grouped bar chart comparing Accuracy, ARI, and NMI across all methods. |
| FR-P8 | Generate a per-species accuracy figure: distribution histogram and worst-20 bar chart. |
| FR-P9 | Save all figures at 300 DPI as PNG, and optionally as PDF, in `output_dir/figures/`. |
| FR-P10 | Each figure must include an embedded plain-text caption explaining how to read it. |

### 5.5 Diagnostics (`cli/diagnose.py`)

| ID | Requirement |
|----|-------------|
| FR-D1 | Generate a diagnostic overview figure and a plain-text `diagnostics_report.txt`. |
| FR-D2 | For density methods (DBSCAN, HDBSCAN), output sweep plots: k-distance plot, eps sweep curve, UMAP scatter. |

### 5.6 Configuration System

| ID | Requirement |
|----|-------------|
| FR-CF1 | All experiments are specified via a YAML config file; no code changes are needed for a new dataset. |
| FR-CF2 | Config sections: `dataset`, `model`, `clustering`, `evaluation`, `visualization`, `dbscan`, `hdbscan`, `output_dir`, `seed`. |
| FR-CF3 | Prompt templates for text-guided K-Means are configurable in the YAML under `model.prompt_templates`. |
| FR-CF4 | The `dbscan` block controls PCA dimensionality reduction, `min_samples`, and optional `eps_range` / `final_eps` overrides. |
| FR-CF5 | The `hdbscan` block controls `sweep_range` and optional `final_min_cluster_size` override. |

### 5.7 Dataset Plugins

| ID | Requirement |
|----|-------------|
| FR-DS1 | Built-in support for `imagefolder` format (class name = subfolder name). |
| FR-DS2 | Built-in support for CUB-200-2011 format with metadata files. |
| FR-DS3 | New dataset formats can be added by implementing a loader in `src/data/plugins.py` without modifying core pipeline code. |

---

## 6. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Reproducibility | Setting `seed` in the config fixes all random state. Setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` ensures deterministic GPU operations. |
| NFR-2 | Performance | Feature extraction must support batched GPU inference. Spectral clustering warns the user when N > 5000 (O(N²) memory). |
| NFR-3 | Portability | Must run on CPU-only machines (falls back from CUDA automatically). |
| NFR-4 | Extensibility | New clustering methods can be added by registering a function in `METHOD_MAP` in `src/clustering/methods.py`. New extractors can be added by subclassing `BaseExtractor`. |
| NFR-5 | Output quality | All figures exported at 300 DPI; PNG for inclusion in thesis; PDF for vector-quality publication. |
| NFR-6 | Graceful degradation | Missing optional packages (`hdbscan`, `umap-learn`, `timm`) raise clear `ImportError` messages. The main pipeline continues without them unless they are required by the config. |
| NFR-7 | Interpretability | All figures include embedded captions explaining how to read them. All code modules include "how to read" and "design rationale" docstrings. |

---

## 7. System Architecture

```
bioclip_clustering/
│
├── cli/                    # CLI entry points (one per pipeline stage)
│   ├── extract.py          # Stage 1: extract embeddings
│   ├── cluster.py          # Stage 2: run clustering algorithms
│   ├── evaluate.py         # Stage 3: compute metrics and generate plots
│   └── diagnose.py         # Stage 4: diagnostic report
│
├── experiments/
│   └── run_pipeline.py     # End-to-end runner (calls all 4 stages)
│
├── src/
│   ├── models/
│   │   └── extractors.py   # BaseExtractor, OpenCLIPExtractor, DINOv2Extractor, build_extractor()
│   ├── data/
│   │   ├── dataset.py      # Image discovery and label indexing
│   │   └── plugins.py      # Dataset format plugins (imagefolder, CUB)
│   ├── clustering/
│   │   ├── methods.py      # Unified clustering API + METHOD_MAP dispatcher
│   │   └── density/
│   │       ├── dbscan_pipeline.py   # DBSCAN + k-distance + sweep
│   │       └── hdbscan_pipeline.py  # HDBSCAN + sweep
│   ├── evaluation/
│   │   └── metrics.py      # ARI, NMI, Hungarian accuracy, purity, confusion
│   ├── visualization/
│   │   └── plots.py        # All 7+ publication-ready figure functions
│   ├── diagnostics/
│   │   └── suite.py        # Diagnostic report generation
│   └── utils/
│       ├── io.py            # NPZ save/load, seed control
│       └── pretty.py        # Console formatting helpers
│
├── configs/
│   ├── eumoth.yaml          # EU-Moth (200 species) config
│   └── cub200.yaml          # CUB-200-2011 config
│
└── results/                 # Auto-created; not version-controlled
    └── <dataset>/
        ├── features.npz
        ├── clusters_<method>.npz
        ├── metrics.json
        ├── metrics_summary.csv
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
      │
      ▼
[cluster.py]  →  clusters_<method>.npz  (N integer assignments, per method)
      │
      ▼
[evaluate.py] →  metrics.json, metrics_summary.csv, figures/*.png
      │
      ▼
[diagnose.py] →  diagnostics_report.txt, figures/diagnostic_*.png
```

---

## 8. Datasets

| Dataset | Classes | Format | Notes |
|---------|---------|--------|-------|
| EU-Moth | 200 European moth species | `imagefolder` | ~11 images/class |
| CUB-200-2011 | 200 bird species | CUB metadata | Standard fine-grained benchmark |

Both datasets use 200 classes, making direct algorithm comparison meaningful.

---

## 9. Models Supported

| Model ID | Architecture | Embedding Dim | Pretrained On |
|----------|-------------|---------------|---------------|
| `bioclip2` | ViT-L/14 | 768 | TreeOfLife-200M (biology) |
| `bioclip` | ViT-B/16 | 512 | TreeOfLife-10M (biology) |
| `clip` | ViT-L/14 | 768 | LAION-2B (general) |
| `dinov2` | ViT-L/14 | 1024 | LVD-142M (general, self-supervised) |

BioCLIP-2 is the primary model; others serve as ablation baselines.

---

## 10. Clustering Algorithms

| Method | Requires K | Discovers K | Notes |
|--------|-----------|------------|-------|
| K-Means | Yes | No | Baseline; 50 restarts |
| K-Means (Text-Guided) | Yes | No | Text centroid initialization via BioCLIP-2 text encoder |
| Spectral | Yes | No | Nearest-neighbor affinity; O(N²) memory |
| Agglomerative | Yes | No | Ward linkage; produces compact clusters |
| DBSCAN | No | Yes | Requires PCA pre-processing; eps via k-distance knee |
| HDBSCAN | No | Yes | Hierarchical density; more robust than DBSCAN |
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

---

## 12. CLI Reference

```bash
# Stage 1: Extract embeddings
python cli/extract.py --config configs/eumoth.yaml

# Stage 2: Cluster
python cli/cluster.py --config configs/eumoth.yaml

# Stage 3: Evaluate and plot
python cli/evaluate.py --config configs/eumoth.yaml

# Stage 4: Diagnostics
python cli/diagnose.py --config configs/eumoth.yaml

# All stages in sequence
python experiments/run_pipeline.py --config configs/eumoth.yaml
```

---

## 13. Output Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| `features.npz` | `results/<dataset>/` | Embeddings (N×D), labels, image paths |
| `clusters_<method>.npz` | `results/<dataset>/` | Cluster assignments per method |
| `metrics.json` | `results/<dataset>/` | Full metrics dict (JSON-safe) |
| `metrics_summary.csv` | `results/<dataset>/` | One-row-per-method summary table |
| `diagnostics_report.txt` | `results/<dataset>/` | Plain-text diagnostic summary |
| `figures/*.png` / `*.pdf` | `results/<dataset>/figures/` | All publication-ready figures |

### Generated Figures

| Figure | Content |
|--------|---------|
| `embedding_true_labels.png` | 2D UMAP/t-SNE, colored by ground-truth species |
| `embedding_clusters_<method>.png` | 2D UMAP/t-SNE, colored by cluster assignments |
| `confusion_matrix_<method>.png` | Per-class recall + most-confused species submatrix |
| `top_confusions_<method>.png` | Top-K misclassification pairs (True → Predicted) |
| `cluster_sizes_<method>.png` | Sorted bar chart + size histogram |
| `intra_inter_distances.png` | Intra-class vs. inter-class distance distributions |
| `method_comparison.png` | Grouped bar chart comparing Accuracy/ARI/NMI across methods |
| `per_class_accuracy_<method>.png` | Distribution histogram + worst-20 species |
| `dbscan_kdistance_plot.png` | k-distance plot for eps selection |
| `dbscan_eps_sweep.png` | ARI/NMI/cluster count vs. eps sweep |
| `hdbscan_sweep.csv` | HDBSCAN parameter sweep results |
| `diagnostic_overview.png` | Multi-panel diagnostic summary |

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
| `Pillow` | Image loading | No |
| `umap-learn` | UMAP 2D projection | Yes |
| `hdbscan` | HDBSCAN clustering | Yes |
| `timm` | DINOv2 extraction | Yes |
| `torchvision` | DINOv2 preprocessing | Yes |

---

## 15. Reproducibility Requirements

| Mechanism | Description |
|-----------|-------------|
| Global seed | Set via `seed` in YAML; applied to NumPy, random, and PyTorch |
| Deterministic CUDA | `CUBLAS_WORKSPACE_CONFIG=:4096:8` environment variable |
| Fixed K-Means restarts | `n_init=50` with fixed `random_state` |
| Fixed UMAP | `random_state` set in config |
| Committed configs | All YAML configs are version-controlled |
| Excluded results | `results/` directory is `.gitignore`d; only configs are committed |

---

## 16. Extensibility Guidelines

### Adding a new dataset
1. Create a new YAML config in `configs/`.
2. If the folder structure is non-standard, add a loader in `src/data/plugins.py`.
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
| Image format | JPEG, PNG supported; images must be RGB-convertible |
| Dataset structure | Primary format assumes one subfolder per class (ImageFolder) |
| Cluster count | For K-anchored methods, K is set to the number of ground-truth classes |
| Evaluation | Requires ground-truth labels; purely unsupervised metric support (Silhouette, Davies-Bouldin) is not implemented |
| Spectral scaling | Not recommended for N > 5,000 due to O(N²) affinity matrix |
| Text-guided K-Means | Requires the extractor to be an `OpenCLIPExtractor` (BioCLIP family) |

---

## 18. Success Criteria

| Criterion | Target |
|-----------|--------|
| Reproduction | Given the same config and dataset, two independent runs produce byte-identical `.npz` files |
| Coverage | All 7 clustering methods complete without error on both EU-Moth and CUB-200-2011 |
| Figure quality | All figures render cleanly at 300 DPI with readable labels |
| Extensibility | A collaborator can add a new dataset (ImageFolder format) by editing only a YAML file |
| Documentation | Every source file includes a module docstring explaining its purpose and every public function includes a docstring with a "how to read" or "design rationale" section |

---

## 19. Known Limitations

| Limitation | Mitigation |
|------------|-----------|
| DBSCAN and HDBSCAN produce noise points (label = −1), reducing the evaluated sample | Noise points are reported and excluded from metrics |
| Spectral clustering is O(N²) in memory | Documented; user warned; recommended N ≤ 5,000 |
| Text-guided K-Means depends on text-image alignment quality of BioCLIP-2 | Template ensembling reduces prompt sensitivity |
| Per-class accuracy is undefined if K < true class count | Hungarian algorithm handles non-square cost matrices |
| 200 classes exceed the perceptual limit of distinguishable colors | Spotlight-on-grey rendering strategy used for embedding plots |

---

*End of PRD*
