# BioCLIP Clustering Pipeline

A modular, reproducible pipeline for extracting image embeddings with BioCLIP-2
and clustering them across biological image datasets (EU-Moth, CUB-200-2011, and others).

---

## Project Structure

```
bioclip_clustering/
├── configs/                 # YAML configs per dataset/run
│   ├── eumoth.yaml
│   └── cub200.yaml
├── src/
│   ├── data/                # Dataset loading and image discovery
│   ├── models/              # Embedding extractors (BioCLIP-2, CLIP, DINOv2)
│   ├── clustering/          # KMeans, Spectral, Agglomerative, DBSCAN, HDBSCAN
│   ├── evaluation/          # ARI, NMI, Hungarian accuracy, confusion analysis
│   ├── visualization/       # Publication-ready plots
│   ├── diagnostics/         # Diagnostic report suite
│   └── utils/               # Saving, loading, seed control
├── cli/                     # CLI entry points
│   ├── extract.py
│   ├── cluster.py
│   ├── evaluate.py
│   └── diagnose.py
├── experiments/             # High-level run scripts
│   └── run_pipeline.py
└── results/                 # Auto-created; not committed to git
    └── figures/
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
# For UMAP
pip install umap-learn

# For HDBSCAN
pip install hdbscan
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

### Step 2 — Cluster

```bash
python cli/cluster.py --config configs/eumoth.yaml
```

Outputs:
- `results/eumoth/clusters_kmeans.npz`
- `results/eumoth/clusters_spectral.npz`
- (etc. for each method in config)

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
- `results/eumoth/diagnostics_report.txt`
- `results/eumoth/figures/diagnostic_*.png`

### Run everything at once

```bash
python experiments/run_pipeline.py --config configs/eumoth.yaml
```

---

## Adding a New Dataset

1. Create a new config file under `configs/` (copy `eumoth.yaml` as a template).
2. If your dataset uses a non-standard folder structure, add a loader plugin in
   `src/data/plugins.py` following the existing examples.
3. Run the pipeline as above.

---

## Reproducing Results

Seeds are set globally at the start of every CLI command.
For deterministic GPU behavior, set:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Then run as usual.

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- open_clip_torch
- scikit-learn
- matplotlib, seaborn
- pandas, numpy
- tqdm, PyYAML
- umap-learn (optional)
- hdbscan (optional)

---

## Methodology 

Image embeddings were extracted using BioCLIP-2 (ViT-L/14), a vision-language foundation
model pre-trained on biological image-text pairs. Each image was resized and normalized
according to the model's preprocessing transform, then encoded with the frozen image encoder.
The resulting 768-dimensional embeddings were L2-normalized prior to clustering, so that
Euclidean distance in embedding space is equivalent to cosine distance.

Clustering was performed with multiple algorithms: K-Means, Spectral Clustering,
Agglomerative (Ward linkage), and DBSCAN. For K-Means and Spectral, the number of
clusters was set equal to the number of known species. Cluster quality was measured with
Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI), which are both
robust to class imbalance and invariant to cluster label permutation. Cluster-to-class
assignment accuracy was computed via the Hungarian algorithm, which finds the
optimal one-to-one matching between predicted clusters and ground-truth classes.
Embeddings were projected to 2D for visualization using UMAP with cosine metric.

All experiments were run with a fixed random seed (42) for reproducibility.
Figures were generated at 300 DPI and exported as PNG and PDF.
