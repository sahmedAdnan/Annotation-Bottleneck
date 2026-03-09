# BioCLIP Clustering Pipeline

A modular, reproducible pipeline for extracting image embeddings with BioCLIP-2
and clustering them across biological image datasets (EU-Moth, MothSpeciesDK, CUB-200-2011, and others).

---

## Project Structure

```
bioclip_clustering/
├── configs/                 # YAML configs per dataset/run
│   ├── eumoth.yaml          # EU-Moth (200 species)
│   ├── cub200.yaml          # CUB-200-2011 (200 bird species)
│   └── mothDK.yaml          # MothSpeciesDK (351 Danish moth species)
├── src/
│   ├── data/                # Dataset loading and image discovery
│   │   ├── dataset.py       # ImageFolder + CUB loaders, corrupted-image filter
│   │   └── plugins.py       # Plugin registry for custom formats (iNaturalist example included)
│   ├── models/              # Embedding extractors (BioCLIP-2, CLIP, DINOv2)
│   ├── clustering/          # KMeans, Text-Guided KMeans, Spectral, Agglomerative, DBSCAN, HDBSCAN, GMM
│   │   └── density/         # Full DBSCAN and HDBSCAN diagnostic pipelines
│   ├── evaluation/          # ARI, NMI, Hungarian accuracy, confusion analysis
│   ├── visualization/       # Publication-ready plots (300 DPI PNG + PDF)
│   ├── diagnostics/         # 1-NN accuracy, silhouette, feature quality report
│   └── utils/
│       ├── io.py            # NPZ save/load, seed control, config loading
│       └── pretty.py        # Console formatting helpers
├── cli/                     # CLI entry points (one per pipeline stage)
│   ├── extract.py           # Stage 1: extract embeddings
│   ├── cluster.py           # Stage 2: all clustering methods
│   ├── evaluate.py          # Stage 3: metrics + publication figures
│   ├── diagnose.py          # Stage 4: diagnostic report
│   └── density.py           # Standalone: DBSCAN + HDBSCAN only
├── experiments/
│   ├── run_pipeline.py      # End-to-end runner (all 4 stages)
│   └── run_density_clustering.py  # Backwards-compatible wrapper for cli/density.py
└── results/                 # Auto-created; not committed to git
    └── <dataset>/
        ├── features.npz
        ├── features_2d.npy          # Cached 2D UMAP/t-SNE projection
        ├── clusters_<method>.npz
        ├── metrics.json
        ├── metrics_summary.csv
        ├── density_metrics_combined.json
        ├── density_metrics_summary.csv
        ├── diagnostics_report.txt
        └── figures/
            └── *.png / *.pdf
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Optional extras

```bash
# For UMAP (recommended; t-SNE used as fallback if not installed)
pip install umap-learn

# For HDBSCAN
pip install hdbscan

# For DINOv2
pip install timm torchvision
```

---

## Usage

All CLI commands take a config file as the first argument.
Results are written to the path defined in your config under `output_dir`.

### Step 1 — Extract features

```bash
python cli/extract.py --config configs/eumoth.yaml
```

Outputs:
- `results/eumoth/features.npz` (embeddings + labels + paths)
- `results/eumoth/features_2d.npy` (cached 2D projection, computed on first use)

### Step 2 — Cluster

```bash
python cli/cluster.py --config configs/eumoth.yaml
```

Outputs:
- `results/eumoth/clusters_kmeans.npz`
- `results/eumoth/clusters_kmeans_text_guided.npz`
- `results/eumoth/clusters_agglomerative.npz`
- `results/eumoth/clusters_spectral.npz`
- `results/eumoth/clusters_dbscan.npz`
- `results/eumoth/clusters_hdbscan.npz`
- `results/eumoth/dbscan_metrics.json`, `dbscan_sweep.csv`
- `results/eumoth/hdbscan_metrics.json`, `hdbscan_sweep.csv`
- `results/eumoth/density_metrics_combined.json`, `density_metrics_summary.csv`
- `results/eumoth/figures/dbscan_*.png`, `hdbscan_*.png`

### Step 3 — Evaluate and plot

```bash
python cli/evaluate.py --config configs/eumoth.yaml
```

Outputs:
- `results/eumoth/metrics.json`
- `results/eumoth/metrics_summary.csv`
- `results/eumoth/figures/*.png`

### Step 4 — Diagnostics report

```bash
python cli/diagnose.py --config configs/eumoth.yaml
```

Outputs:
- `results/eumoth/diagnostics_report.txt` (1-NN accuracy, silhouette, feature stats, quick K-Means)
- `results/eumoth/figures/diagnostic_overview.png`

### Run everything at once

```bash
python experiments/run_pipeline.py --config configs/eumoth.yaml
```

#### Skip flags (for resuming or partial runs)

```bash
# Resume from clustering (features already extracted)
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-extract

# Skip only HDBSCAN (if not installed)
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-hdbscan

# Skip both density methods (DBSCAN + HDBSCAN)
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-density

# Re-run only evaluation figures
python experiments/run_pipeline.py --config configs/eumoth.yaml \
    --skip-extract --skip-cluster --skip-diagnose

# Available skip flags:
#   --skip-extract    Skip Step 1 (feature extraction)
#   --skip-cluster    Skip Step 2 (all clustering)
#   --skip-evaluate   Skip Step 3 (evaluation + figures)
#   --skip-diagnose   Skip Step 4 (diagnostics)
#   --skip-density    Skip DBSCAN + HDBSCAN within Step 2
#   --skip-dbscan     Skip DBSCAN only within Step 2
#   --skip-hdbscan    Skip HDBSCAN only within Step 2
```

### Run density clustering standalone

```bash
python cli/density.py --config configs/eumoth.yaml
python cli/density.py --config configs/eumoth.yaml --skip-hdbscan
python cli/density.py --config configs/eumoth.yaml --skip-dbscan
```

---

## Datasets

| Dataset | Classes | Images | Format | Config |
|---------|---------|--------|--------|--------|
| EU-Moth | 200 European moth species | ~2,200 | `imagefolder` | `eumoth.yaml` |
| MothSpeciesDK | 351 Danish moth species | 18,999 | `imagefolder` | `mothDK.yaml` |
| CUB-200-2011 | 200 bird species | ~11,788 | CUB metadata | `cub200.yaml` |

---

## Models Supported

| `model.name` | Architecture | Embedding Dim | Pretrained On |
|---|---|---|---|
| `bioclip2` | ViT-L/14 | 768 | TreeOfLife-200M (biology) |
| `bioclip` | ViT-B/16 | 512 | TreeOfLife-10M (biology) |
| `clip` | ViT-L/14 | 768 | LAION-2B (general) |
| `dinov2` | ViT-L/14 | 1024 | LVD-142M (self-supervised) |

Switch models by changing `model.name` and `model.pretrained` in the YAML config. No code changes needed.

> **Note:** `kmeans_text_guided` requires a CLIP-family model (BioCLIP-2, BioCLIP, or CLIP). DINOv2 has no text encoder and is incompatible with that clustering method.

---

## Clustering Methods

| Method | Requires K | Notes |
|--------|-----------|-------|
| `kmeans` | Yes | Baseline; 50 random restarts |
| `kmeans_text_guided` | Yes | Text centroid init via BioCLIP-2 text encoder; prompt template ensembling |
| `spectral` | Yes | Nearest-neighbor affinity; O(N²) memory — not recommended for N > 5,000 |
| `agglomerative` | Yes | Ward linkage |
| `dbscan` | No | Auto eps via k-distance knee; PCA pre-processing |
| `hdbscan` | No | Auto min_cluster_size sweep; min_cluster_size must be ≥ 2 |
| `gmm` | Yes | Gaussian Mixture Model; hard-assigned at evaluation |

---

## Adding a New Dataset

1. Create a new config file under `configs/` (copy `eumoth.yaml` as a template).
2. If your dataset uses a non-standard folder structure, add a loader plugin in
   `src/data/plugins.py` following the existing iNaturalist example (`@register_plugin("inaturalist")`).
3. Run the pipeline as above.

---

## Reproducing Results

Seeds are set globally at the start of every CLI command.
For deterministic GPU behavior, set:

```bash
# Linux / macOS
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Windows PowerShell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Windows Command Prompt
set CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Then run as usual.

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- open_clip_torch
- scikit-learn
- matplotlib, seaborn
- pandas, numpy, scipy
- tqdm, PyYAML, Pillow
- umap-learn (optional, recommended)
- hdbscan (optional)
- timm, torchvision (optional, for DINOv2)

---

## Methodology

Image embeddings were extracted using BioCLIP-2 (ViT-L/14), a vision-language foundation
model pre-trained on biological image-text pairs. Each image was resized and normalized
according to the model's preprocessing transform, then encoded with the frozen image encoder.
The resulting 768-dimensional embeddings were L2-normalized prior to clustering, so that
Euclidean distance in embedding space is equivalent to cosine distance.

Clustering was performed with multiple algorithms: K-Means, Text-Guided K-Means, Spectral
Clustering, Agglomerative (Ward linkage), DBSCAN, HDBSCAN, and GMM. For K-anchored methods
the number of clusters was set equal to the number of known species. Cluster quality was
measured with Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI), which are
both robust to class imbalance and invariant to cluster label permutation. Cluster-to-class
assignment accuracy was computed via the Hungarian algorithm. For density methods (DBSCAN,
HDBSCAN), a coverage-weighted ARI (ARI × coverage fraction) is used as the primary verdict
metric to account for noise rejection. Embeddings were projected to 2D for visualization
using UMAP with cosine metric (t-SNE used as fallback if umap-learn is not installed).

All experiments were run with a fixed random seed (42) for reproducibility.
Figures were generated at 300 DPI and exported as PNG and PDF.
