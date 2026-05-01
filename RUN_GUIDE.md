# BioCLIP Clustering Pipeline -- Run Guide

A step-by-step guide for running every task in the pipeline.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Task 1: Clustering Method Evaluation](#3-task-1-clustering-method-evaluation)
4. [Task 2: Species Annotation (Unlabeled Specimens)](#4-task-2-species-annotation)
5. [Task 3: Foundation Model Comparison](#5-task-3-foundation-model-comparison)
6. [Task 4: Dimensionality Reduction Analysis](#6-task-4-dimensionality-reduction-analysis)
7. [Task 5: Expert Review Workflow](#7-task-5-expert-review-workflow)
8. [Configuration Reference](#8-configuration-reference)
9. [Output Files Reference](#9-output-files-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Environment Setup

```bash
conda activate thesis
```

### Required Packages

- Python 3.10+
- PyTorch (with CUDA for GPU acceleration)
- open_clip_torch (for BioCLIP-2, BioCLIP, CLIP)
- scikit-learn, hdbscan, umap-learn
- numpy, pandas, matplotlib, seaborn

### Hardware

- GPU recommended for feature extraction (Step 1)
- BioCLIP-2 ViT-L/14 requires ~2.5 GB RAM for model weights
- Text encoding requires the model loaded in memory
- Clustering and evaluation run on CPU

---

## 2. Project Structure

```
bioclip_clustering/
├── cli/                            # Individual pipeline steps
│   ├── extract.py                  # Extract image embeddings
│   ├── cluster.py                  # Run clustering methods
│   ├── evaluate.py                 # Evaluate against ground truth
│   ├── annotate.py                 # Annotate clusters with species names
│   ├── compare_models.py           # Compare foundation models
│   └── reduction_analysis.py       # Find optimal dimensionality reduction
├── experiments/
│   ├── run_pipeline.py             # One-command full pipeline
│   └── unlabeled_pipeline/         # Annotation pipeline for unknown images
│       ├── run_unlabeled.py        # Main entry point (3 candidate modes)
│       ├── gbif_candidates.py      # GBIF API integration
│       ├── species_report.py       # HTML report generator
│       └── config_example.yaml     # Template configuration
├── configs/                        # YAML configs per dataset/model
├── src/                            # Core library modules
└── results/                        # Output directory (auto-created)
```

---

## 3. Task 1: Clustering Method Evaluation

**Goal:** Test how well different clustering methods group species using labeled data.

**Supported methods:** KMeans, Agglomerative, Spectral, GMM, DBSCAN, HDBSCAN

### Quick Start (one command)

```bash
python experiments/run_pipeline.py --config configs/eumoth.yaml
```

This runs 3 steps in sequence: extract, cluster, evaluate.

### Step-by-Step

```bash
# Step 1: Extract embeddings (saves features.npz)
python cli/extract.py --config configs/eumoth.yaml

# Step 2: Run all clustering methods defined in config
python cli/cluster.py --config configs/eumoth.yaml

# Step 3: Evaluate clustering quality + generate figures
python cli/evaluate.py --config configs/eumoth.yaml
```

### Skipping Steps

If features are already extracted from a previous run:

```bash
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-extract
```

If you only want to regenerate evaluation figures:

```bash
python experiments/run_pipeline.py --config configs/eumoth.yaml \
    --skip-extract --skip-cluster
```

### Skipping Specific Methods

```bash
# Skip all density methods (DBSCAN + HDBSCAN)
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-density

# Skip only HDBSCAN
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-hdbscan

# Skip only DBSCAN
python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-dbscan
```

### Output

Results are saved to the `output_dir` specified in the config (e.g., `results/eumoth/`):

```
results/eumoth/
├── features.npz                  # Extracted embeddings (N, 768)
├── features_2d.npy               # 2D UMAP projection for visualization
├── clusters_kmeans.npz           # Cluster assignments per method
├── clusters_hdbscan.npz
├── hdbscan_probabilities.npy     # HDBSCAN membership probabilities
├── metrics.json                  # ARI, NMI, Hungarian accuracy per method
├── metrics_summary.csv
└── figures/
    ├── embedding_true_labels.png
    ├── confusion_matrix_*.png
    └── method_comparison.png
```

---

## 4. Task 2: Species Annotation

**Goal:** Identify unknown biological specimens by clustering + zero-shot text matching.

The pipeline supports three ways to build the candidate species list:

| Mode | Flag | What it does | Internet |
|------|------|-------------|----------|
| GBIF | (default) | Queries GBIF API for species in your region | Required |
| Local | `--local` | Uses dataset folder names as candidates | Not needed |
| CSV | `--taxonomy file.csv` | Reads species + taxonomy from CSV | Not needed |

### GBIF Mode (default)

Best for truly unknown specimens where you only know the taxonomic order and region.

```bash
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml
```

Override region or candidate count:

```bash
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml \
    --region europe --n-candidates 300
```

### Local Mode

Best for testing annotation accuracy when you have labeled data (imagefolder layout).

Your images directory must have subfolders named by species:

```
images_dir/
├── abrostola_tripartita/
│   ├── img001.jpg
│   └── img002.jpg
├── aglia_tau/
│   └── img003.jpg
└── ...
```

```bash
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml --local
```

Local mode automatically:
- Uses folder names as candidate species
- Collects reference images from dataset folders
- Prints annotation accuracy against ground truth (folder labels)

### CSV Mode

Best when you have a pre-built taxonomy CSV (e.g., all European Lepidoptera).
Avoids slow GBIF API queries while providing full taxonomy for better prompts.

Your CSV file must have a `class_name` column. Optional: `genus`, `family`, `order`.

```csv
class_name,genus,family,order
Aglia tau,Aglia,Saturniidae,Lepidoptera
Tethea or,Tethea,Drepanidae,Lepidoptera
Abrostola tripartita,Abrostola,Noctuidae,Lepidoptera
```

```bash
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml \
    --taxonomy path/to/european_moths.csv
```

### HTML Report Options

By default, the HTML report uses local `file://` paths (small file, only works on same machine).

To create a self-contained HTML with embedded images (shareable, but large):

```bash
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml \
    --local --embed-images
```

### Resuming from a Specific Step

If earlier steps completed successfully and you want to regenerate only the report:

```bash
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml \
    --skip-to 7
```

### Pipeline Steps

The annotation pipeline runs 9-10 steps automatically:

```
Step 1:  Load images
Step 2:  Build candidate species list (GBIF / folders / CSV)
Step 3:  Collect reference images
Step 4:  Extract BioCLIP-2 embeddings
Step 5:  Cluster embeddings (HDBSCAN)
Step 6:  Annotate clusters via zero-shot text matching
Step 6b: Expand candidates for low-confidence clusters (GBIF mode only)
Step 7:  Generate uncertainty visualizations
Step 8:  Export review sheet with reference images
Step 9:  Generate interactive HTML species report
Step 10: Accuracy check against ground truth (local/CSV mode only)
```

### Output

```
results/unlabeled_test/
├── features.npz                  # Embeddings
├── clusters.npz                  # HDBSCAN cluster assignments
├── hdbscan_probabilities.npy     # Membership probabilities
├── annotations.csv               # Full annotation results with uncertainty
├── text_embeddings.npz            # Cached text embeddings
├── review_template.csv            # Template for expert review
├── review_images/                 # Uncertain images + references
├── species_report.html            # Interactive HTML report
├── gbif_candidates.csv            # Candidate species list (GBIF mode)
└── figures/
    ├── uncertainty_histogram.png
    ├── confidence_tiers.png
    └── umap_intervention_map.png
```

---

## 5. Task 3: Foundation Model Comparison

**Goal:** Compare clustering quality across BioCLIP-2, BioCLIP, CLIP, and DINOv2.

### Setup

First, extract features with each model. Each model needs its own config file:

```bash
python cli/extract.py --config configs/eumoth.yaml           # BioCLIP-2
python cli/extract.py --config configs/eumoth_bioclip.yaml    # BioCLIP
python cli/extract.py --config configs/eumoth_clip.yaml       # CLIP
python cli/extract.py --config configs/eumoth_dinov2.yaml     # DINOv2
```

### Run Comparison

```bash
python cli/compare_models.py --config configs/eumoth.yaml
```

This clusters each model's features and compares ARI, NMI, Hungarian accuracy.

### Note on DINOv2

DINOv2 has no text encoder. It can only participate in clustering comparison (Task 1 and Task 3), not annotation (Task 2). The pipeline handles this automatically.

---

## 6. Task 4: Dimensionality Reduction Analysis

**Goal:** Find the optimal dimensionality reduction for your specific dataset.

Tests Full-D (768D), PCA, UMAP, and PCA+UMAP on partition clustering methods.
Prints a recommendation for which reduction to use.

### Prerequisite

Features must be extracted first:

```bash
python cli/extract.py --config configs/eumoth.yaml
```

### Full Comparison

```bash
python cli/reduction_analysis.py --config configs/eumoth.yaml
```

Tests PCA, UMAP, and PCA+UMAP at dimensions [50, 100, 200] with KMeans, Agglomerative, Spectral, and GMM.

### Fast: PCA Only

```bash
python cli/reduction_analysis.py --config configs/eumoth.yaml \
    --skip-umap --skip-pca-umap
```

### Custom Dimensions

```bash
python cli/reduction_analysis.py --config configs/eumoth.yaml \
    --dims 20 50 100 200 300
```

### Include Density Methods

```bash
python cli/reduction_analysis.py --config configs/eumoth.yaml --include-density
```

### Skip Spectral (for large datasets)

Spectral clustering uses O(N^2) memory. For datasets with more than 5000 images:

```bash
python cli/reduction_analysis.py --config configs/eumoth.yaml --skip-spectral
```

### UMAP Hyperparameters

```bash
python cli/reduction_analysis.py --config configs/eumoth.yaml \
    --umap-neighbors 15 --umap-min-dist 0.0
```

### PCA+UMAP Settings

```bash
python cli/reduction_analysis.py --config configs/eumoth.yaml \
    --pca-pre 100 --pca-umap-targets 10 20 30 50
```

### Output

The tool prints a recommendation and saves:

```
results/eumoth/
├── reduction_comparison.csv           # Full results table
├── reduction_comparison.json          # Structured results
└── reduction_recommendation.txt       # Human-readable recommendation
```

Example output:

```
  RECOMMENDATION
  ============================================================

  Method           Reducer    Dim   Accuracy        ARI        NMI
  ------------------------------------------------------------------
  kmeans               pca     50     0.8234     0.8156     0.9102
  agglomerative        pca     50     0.8301     0.8245     0.9134
  spectral             pca    100     0.7856     0.7802     0.8901
  gmm                  pca     50     0.8123     0.8034     0.9045

  Partition methods: Use PCA-50D
```

---

## 7. Task 5: Expert Review Workflow

**Goal:** Improve annotation accuracy by having a domain expert verify uncertain predictions.

### Step 1: Generate Review Materials

Run the annotation pipeline (any mode):

```bash
python cli/annotate.py --config configs/mothDK_subsample.yaml --method hdbscan
```

This generates:
- `review_template.csv` -- a spreadsheet template
- `review_images/` -- copies of uncertain images with informative filenames

### Step 2: Expert Review

1. Open `review_template.csv` in Excel or Google Sheets
2. Examine images in `review_images/` folder
3. Fill the `Expert_Corrected_Label` column with the correct species name
4. Optionally add notes in the `Expert_Notes` column
5. Write "junk" or "delete" in `Expert_Notes` to flag invalid images
6. Save as `review_template_filled.csv`

### Step 3: Merge Corrections

```bash
python cli/annotate.py --config configs/mothDK_subsample.yaml \
    --merge results/mothDK_subsample/review_template_filled.csv
```

This regenerates annotations with expert corrections and produces a provenance pie chart showing AI vs Human contribution.

### Annotate a Specific Method

```bash
python cli/annotate.py --config configs/mothDK_subsample.yaml --method hdbscan
python cli/annotate.py --config configs/mothDK_subsample.yaml --method kmeans
```

---

## 8. Configuration Reference

### Available Configurations

| Config File | Dataset | Model | Classes |
|-------------|---------|-------|---------|
| `configs/eumoth.yaml` | EU-Moth | BioCLIP-2 | 200 |
| `configs/eumoth_bioclip.yaml` | EU-Moth | BioCLIP | 200 |
| `configs/eumoth_clip.yaml` | EU-Moth | CLIP | 200 |
| `configs/eumoth_dinov2.yaml` | EU-Moth | DINOv2 | 200 |
| `configs/mothDK.yaml` | MothDK (full) | BioCLIP-2 | 351 |
| `configs/mothDK_subsample.yaml` | MothDK (subset) | BioCLIP-2 | 119 |
| `configs/cub200.yaml` | CUB-200 | BioCLIP-2 | 200 |
| `experiments/unlabeled_pipeline/config_example.yaml` | Unlabeled | BioCLIP-2 | -- |

### Key Config Sections

```yaml
# Dataset
dataset:
  name: EU-Moth
  root: path/to/images
  format: imagefolder        # or "cub" for CUB-200
  num_classes: 200

# Model
model:
  name: bioclip2             # bioclip2, bioclip, clip, dinov2
  pretrained: hf-hub:imageomics/bioclip-2
  normalize_embeddings: true
  batch_size: 64
  prompt_templates:          # used for text encoding
    - "{name}"
    - "a photo of {name}"
    - "{name}, family {family}, order {order}"

# Clustering
clustering:
  methods:
    - name: kmeans
      n_clusters: 200
      n_init: 50
    - name: hdbscan
      min_cluster_size: 5
      min_samples: 2

# Reduction
reduction:
  partition:
    method: pca
    n_components: 50
  density:
    method: umap
    n_components: 100
    n_neighbors: 15

# Annotation
annotation:
  taxonomy_file: "path/to/taxonomy.csv"   # optional
  uncertainty:
    review_threshold: 0.55
    human_threshold: 0.70
    weights:
      text_margin: 0.35
      membership_probability: 0.25
      cluster_agreement: 0.25
      taxonomic_consistency: 0.15

# Unlabeled pipeline (config_example.yaml)
gbif:
  order_key: 797             # 797 = Lepidoptera
  region: europe
  query_mode: balanced
  per_family: 40
  rare_per_family: 10
```

---

## 9. Output Files Reference

### Clustering Pipeline Output

| File | Description |
|------|-------------|
| `features.npz` | Extracted embeddings (N, 768) + labels + paths |
| `features_2d.npy` | 2D UMAP projection for visualization |
| `clusters_<method>.npz` | Cluster assignments per method |
| `hdbscan_probabilities.npy` | HDBSCAN membership probabilities |
| `metrics.json` | ARI, NMI, Hungarian accuracy per method |
| `metrics_summary.csv` | Tabular metrics summary |
| `figures/*.png` | Confusion matrices, method comparison, embeddings |

### Annotation Pipeline Output

| File | Description |
|------|-------------|
| `annotations.csv` | Full results: species, uncertainty, confidence tier |
| `text_embeddings.npz` | Cached text embeddings (reused on re-runs) |
| `review_template.csv` | Template for expert review |
| `review_images/` | Copies of uncertain images |
| `species_report.html` | Interactive HTML report |
| `gbif_candidates.csv` | Candidate species list (GBIF mode) |
| `figures/uncertainty_histogram.png` | Distribution of uncertainty scores |
| `figures/confidence_tiers.png` | Breakdown by confidence tier |
| `figures/umap_intervention_map.png` | UMAP colored by uncertainty |

### Reduction Analysis Output

| File | Description |
|------|-------------|
| `reduction_comparison.csv` | Full results: method x reducer x dimension |
| `reduction_comparison.json` | Structured results |
| `reduction_recommendation.txt` | Human-readable recommendation |

---

## 10. Troubleshooting

### "The paging file is too small" (Windows OOM)

BioCLIP-2 ViT-L/14 needs ~2.5 GB RAM for model weights. Solutions:

1. Close other applications (browsers, VS Code extensions)
2. Increase Windows paging file: System Properties > Advanced > Performance > Virtual Memory > Custom size: 8192-16384 MB
3. Use `--skip-to 7` if features and text embeddings are already cached

### GBIF API Errors (503 / Timeout)

The pipeline uses 0.5s delay between requests with retry logic. If GBIF is down:

1. Wait and retry later
2. Use `--local` or `--taxonomy` mode instead
3. Use `--skip-gbif` to use cached candidates from a previous run

### "No images found"

Check that your `images_dir` path in the config points to a directory containing images. Supported formats: jpg, jpeg, png.

For local mode, images must be in subfolders: `images_dir/<species_name>/image.jpg`

### DINOv2 + Annotation

DINOv2 has no text encoder. It cannot be used for annotation (Task 2) or text-based comparison. Use BioCLIP-2, BioCLIP, or CLIP for annotation tasks.

### Spectral Clustering OOM

Spectral clustering uses O(N^2) memory. For datasets with more than 5000 images, skip it:

```bash
python cli/cluster.py --config configs/eumoth.yaml
# In config, remove spectral from clustering.methods

# Or for reduction analysis:
python cli/reduction_analysis.py --config configs/eumoth.yaml --skip-spectral
```

### Cached Results

The pipeline caches features, 2D projections, and text embeddings to disk. To force re-computation, delete the cached files:

```bash
# Re-extract features
rm results/eumoth/features.npz
python cli/extract.py --config configs/eumoth.yaml

# Re-encode text
rm results/unlabeled_test/text_embeddings.npz
```
