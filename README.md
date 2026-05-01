# BioCLIP Clustering and Annotation Pipeline

A modular, reproducible pipeline for unsupervised species identification from biological images. The pipeline extracts BioCLIP-2 image embeddings, clusters them with one of six unsupervised algorithms, and assigns species names to clusters via zero-shot text matching with a four-signal uncertainty quantification framework.

The pipeline supports two workflows:

1. **Labelled benchmark evaluation** — measure clustering and annotation accuracy on datasets with ground-truth species labels (EU-Moth, CUB-200-2011, MothSpeciesDK).
2. **Unlabelled specimen identification** — annotate unknown specimen images with confidence-tiered species labels and produce an interactive HTML review report for expert verification.

---

## Project Structure

```
bioclip_clustering/
├── configs/                    # YAML configs per dataset / model
├── src/
│   ├── data/                   # Dataset loading, subsampling, image discovery
│   ├── models/                 # Embedding extractors (BioCLIP-2, BioCLIP, CLIP, DINOv2)
│   ├── reduction/              # PCA, UMAP, dual-path reduction
│   ├── clustering/             # K-Means, Spectral, Agglomerative, GMM, DBSCAN, HDBSCAN
│   │   └── density/            # DBSCAN and HDBSCAN diagnostic pipelines
│   ├── evaluation/             # ARI, NMI, Hungarian accuracy, confusion analysis
│   ├── visualization/          # Publication-ready plots (300 DPI PNG + PDF)
│   ├── annotation/             # Centroid annotation, 4-signal uncertainty, expert review
│   └── utils/                  # I/O, seed control, console formatting
├── cli/
│   ├── extract.py              # Stage 1: extract embeddings
│   ├── cluster.py              # Stage 2: run clustering
│   ├── evaluate.py             # Stage 3: metrics + figures
│   ├── annotate.py             # Stage 4: zero-shot annotation + uncertainty + review
│   ├── compare_models.py       # Compare BioCLIP-2 vs BioCLIP vs CLIP vs DINOv2
│   └── reduction_analysis.py   # Compare PCA / UMAP / PCA+UMAP at multiple dimensions
├── experiments/
│   ├── run_pipeline.py         # End-to-end runner: extract -> cluster -> evaluate
│   └── unlabeled_pipeline/     # Real-world annotation for unknown specimens
│       ├── run_unlabeled.py    # GBIF / local / CSV candidate modes + HTML report
│       ├── gbif_candidates.py  # GBIF API queries with balanced per-family coverage
│       ├── species_report.py   # Interactive HTML report generator
│       └── config_example.yaml
└── results/                    # Auto-created (not committed)
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure dataset paths

Edit the YAML config for your dataset under `configs/` and set the `images_dir` (or `dataset.root`) field to the absolute path on your machine. The shipped configs use placeholder paths that must be replaced before running.

---

## Usage

All CLI commands take a YAML config as their argument. Outputs are written to the `output_dir` defined in the config.

### Workflow 1: Labelled benchmark evaluation

```bash
# End-to-end: extract -> cluster -> evaluate
python experiments/run_pipeline.py --config configs/eumoth.yaml

# Or run stages individually
python cli/extract.py  --config configs/eumoth.yaml
python cli/cluster.py  --config configs/eumoth.yaml
python cli/evaluate.py --config configs/eumoth.yaml
```

### Workflow 2: Annotate clusters with species names

```bash
# Annotate a specific clustering method
python cli/annotate.py --config configs/eumoth.yaml --method hdbscan

# After expert fills out the review CSV, merge corrections back
python cli/annotate.py --config configs/eumoth.yaml \
    --merge results/eumoth/review_template_filled.csv
```

### Workflow 3: Annotate unknown specimens

```bash
# GBIF mode (queries GBIF for candidate species in the region)
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml

# Local mode (uses dataset folder names as candidates, no internet needed)
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml --local

# CSV mode (uses pre-computed taxonomy file as candidate list)
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml \
    --taxonomy path/to/taxonomy.csv
```

The unlabelled pipeline produces an interactive HTML report (`species_report.html`) where each cluster can be reviewed and either accepted, corrected, or rejected by an expert.

### Workflow 4: Compare foundation models

```bash
python cli/compare_models.py --config configs/eumoth.yaml
```

Runs all clustering methods on BioCLIP-2, BioCLIP, CLIP, and DINOv2 embeddings and reports comparative metrics with heatmaps.

### Workflow 5: Find optimal dimensionality reduction

```bash
python cli/reduction_analysis.py --config configs/eumoth.yaml
```

Compares Full-D, PCA, UMAP, and PCA+UMAP at multiple target dimensions for each clustering method.

---

## Datasets

| Dataset | Classes | Images | Format | Config |
|---------|---------|--------|--------|--------|
| EU-Moth | 200 European moth species | 2,205 | `imagefolder` | `configs/eumoth.yaml` |
| CUB-200-2011 | 200 bird species | 11,788 | `imagefolder` | `configs/cub200.yaml` |
| MothSpeciesDK (full) | 351 Danish moth species | 18,999 | `imagefolder` | `configs/mothDK.yaml` |
| MothSpeciesDK (subsample) | 200 species, balanced | 1,961 | `imagefolder` | `configs/mothDK_subsample.yaml` |

---

## Foundation Models Supported

| `model.name` | Architecture | Embedding Dim | Pretrained On | Text Encoder |
|---|---|---|---|---|
| `bioclip2` | ViT-L/14 | 768 | TreeOfLife-200M | Yes |
| `bioclip` | ViT-B/16 | 512 | TreeOfLife-10M | Yes |
| `clip` | ViT-L/14 | 768 | LAION-400M | Yes |
| `dinov2` | ViT-L/14 | 1,024 | LVD-142M | No |

Switch models by changing `model.name` and `model.pretrained` in the YAML config. DINOv2 has no text encoder and cannot be used with the annotation pipeline.

---

## Clustering Methods

| Method | Requires K | Notes |
|--------|-----------|-------|
| `kmeans` | Yes | 50 random restarts, 500 max iterations |
| `agglomerative` | Yes | Ward linkage |
| `spectral` | Yes | Nearest-neighbour affinity; O(N²) memory |
| `gmm` | Yes | Diagonal covariance |
| `dbscan` | No | Auto `eps` via k-distance knee method |
| `hdbscan` | No | Sweep over `min_cluster_size`; `eom` or `leaf` selection |

Fixed-k methods are reduced with PCA (preserves global variance for Euclidean distances). Density-based methods are reduced with UMAP (preserves local neighbourhood structure for density estimation). This dual-path reduction is configured in each YAML file.

---

## Annotation Pipeline

The annotation stage assigns a species name to each cluster via zero-shot text matching in the original 768-dimensional CLIP space, and quantifies prediction uncertainty with four independent signals:

| Signal | Captures | Weight |
|---|---|---|
| Semantic (text margin) | Similar-looking species pairs | 0.35 |
| Spatial (membership probability) | Boundary or noise points | 0.25 |
| Cluster agreement | Misplaced images in good clusters | 0.25 |
| Taxonomic consistency | Clusters mixing different families | 0.15 |

The combined uncertainty score classifies each annotation into one of three confidence tiers:

- **`auto_accept`** — combined uncertainty below the review threshold (default 0.55)
- **`review`** — between review and human-required thresholds
- **`human_required`** — combined uncertainty above the human-required threshold (default 0.70)

A review CSV is exported for the `review` and `human_required` tiers. Expert corrections can be merged back with full provenance tracking.

---

## Reproducibility

A fixed random seed (42) is set globally at the start of every pipeline run, controlling Python's `random`, NumPy, and PyTorch generators. All hyperparameters are specified in YAML configuration files; no values are hardcoded in the source code.

For deterministic GPU behaviour, set:

```bash
# Linux / macOS
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Windows PowerShell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- open_clip_torch
- scikit-learn
- hdbscan
- umap-learn
- timm (for DINOv2)
- matplotlib, pandas, numpy, scipy, tqdm, PyYAML, Pillow

See `requirements.txt` for the exact pinned versions used in all reported experiments.

---

## Citation

If you use this pipeline in your research, please cite the underlying foundation models:

- **BioCLIP-2**: Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life*, CVPR 2024.
- **CLIP**: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
- **DINOv2**: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.

And the benchmark datasets used:

- **EU-Moth**: Korsch et al., *Deep Learning Pipeline for Automated Visual Moth Monitoring*, INFORMATIK 2021.
- **CUB-200-2011**: Wah et al., Caltech-UCSD Birds-200-2011 Dataset, Tech. Report CNS-TR-2011-001, 2011.
- **MothSpeciesDK**: Bjerge et al., *An Automated Light Trap to Monitor Moths Using Computer Vision-Based Tracking and Deep Learning*, Sensors 21(2), 2021.
