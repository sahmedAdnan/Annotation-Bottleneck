# BioCLIP Clustering + Annotation Pipeline

Unsupervised species discovery: extract BioCLIP-2 embeddings, cluster them,
and annotate clusters via zero-shot text matching with uncertainty signals.

## Setup

```bash
pip install -r requirements.txt
```

Point the configs at your image folders (folder-of-subfolders, one subfolder
per class). Edit the `root:` / `images_dir:` field in the relevant YAML:

- `configs/*.yaml`                         — labeled clustering evaluation
- `experiments/unlabeled_pipeline/*.yaml`  — unlabeled annotation

Datasets are **not** included; only the code and taxonomy CSVs are.

## Task 1 — Evaluate clustering methods (labeled data)

```bash
# full pipeline: extract -> cluster -> evaluate
python experiments/run_pipeline.py --config configs/eumoth.yaml

# or step by step
python cli/extract.py  --config configs/eumoth.yaml
python cli/cluster.py  --config configs/eumoth.yaml
python cli/evaluate.py --config configs/eumoth.yaml
```

Produces ARI, NMI, Hungarian accuracy across 6 clustering methods + figures.

## Task 2 — Annotate unknown specimens (unlabeled data)

```bash
# GBIF candidates (needs internet)
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml

# candidates from a taxonomy CSV (offline)
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/mothDK_subsample.yaml \
    --taxonomy mothDK_classes_with_family.csv

# candidates from folder names (offline, genus-only taxonomy)
python experiments/unlabeled_pipeline/run_unlabeled.py \
    --config experiments/unlabeled_pipeline/config_example.yaml --local
```

Produces per-cluster annotations, 4 uncertainty signals, confidence tiers,
and a self-contained interactive `review.html`.

## Task 3 — Compare foundation models

```bash
python cli/compare_models.py --config configs/eumoth.yaml
```

Compares BioCLIP-2, BioCLIP, CLIP, DINOv2 on clustering quality.

## Optional — Find the best dimensionality reduction

```bash
python cli/reduction_analysis.py            --config configs/eumoth.yaml   # labeled (ARI)
python cli/reduction_analysis_unlabeled.py  --config configs/eumoth.yaml   # unlabeled (silhouette)
```

Set the recommended `reduction.method` / `reduction.n_components` in the config
before running `cluster.py`.

## Taxonomy CSV format

```csv
class_name,genus,family,order
Aglia tau,Aglia,Saturniidae,Lepidoptera
```

Included: `mothDK_classes_with_family.csv`, `eumoth_taxonomy.csv`,
`cub200_taxonomy.csv`.

## Layout

```
src/          core library (data, models, reduction, clustering, evaluation,
              annotation, visualization, utils)
cli/          per-step entry points
experiments/  one-command pipelines (labeled + unlabeled)
configs/      dataset + model YAML configs
```

Notes:
- Text matching runs in the original 768/1024-D CLIP space (never reduced).
- Annotation requires a BioCLIP model (DINOv2 has no text encoder).
- All randomness is seeded via each config's `seed`.

