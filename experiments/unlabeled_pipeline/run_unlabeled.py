"""
experiments/unlabeled_pipeline/run_unlabeled.py
-----------------------------------------------
End-to-end pipeline for identifying unknown biological specimens.

Supports THREE candidate sources:

  1. GBIF mode (default): Query GBIF API for candidate species by taxonomy + region.
     Reference images fetched from GBIF.

  2. Local mode (--local): Use dataset folder names as candidates.
     Reference images taken from the dataset itself. No internet needed.
     Requires imagefolder layout: images_dir/<species_name>/image.jpg

  3. CSV mode (--taxonomy path.csv): Read candidate species + taxonomy from CSV.
     CSV must have columns: class_name, genus, family, order.
     Reference images can optionally come from a dataset (--images with subfolders).

Pipeline:
    1. Load images
    2. Build candidate species list (GBIF / folder names / CSV)
    3. Collect reference images (GBIF URLs / dataset samples / none)
    4. Extract BioCLIP-2 embeddings
    5. Cluster embeddings (HDBSCAN or KMeans)
    6. Annotate clusters via zero-shot text matching + 4 uncertainty signals
    7. Generate uncertainty visualizations
    8. Export review sheet + interactive HTML report

Usage:
    # GBIF mode (default):
    python experiments/unlabeled_pipeline/run_unlabeled.py --config experiments/unlabeled_pipeline/config_example.yaml

    # Local mode (no internet, folder names as candidates):
    python experiments/unlabeled_pipeline/run_unlabeled.py --config experiments/unlabeled_pipeline/config_example.yaml --local

    # CSV mode (taxonomy CSV as candidate list):
    python experiments/unlabeled_pipeline/run_unlabeled.py --config experiments/unlabeled_pipeline/config_example.yaml --taxonomy path/to/european_moths.csv
"""

import argparse
import csv
import os
import sys
import time
import glob
import shutil
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.utils.io import set_seed, load_config, ensure_dir, save_features
from src.utils.pretty import header, subheader, metrics_table, done, warning
from src.models.extractors import build_extractor
from src.annotation.annotator import run_annotation
from src.annotation.gallery import (plot_uncertainty_histogram,
                                     plot_annotation_map,
                                     plot_confidence_tiers)
# GBIF imports (only needed in GBIF mode, but import unconditionally for simplicity)
from gbif_candidates import (query_candidate_species, query_candidates_balanced,
                             download_reference_images,
                              collect_reference_urls,
                              candidates_to_taxonomy, candidates_to_class_names,
                              expand_candidates_for_gaps, CONTINENT_MAP,
                              describe_order_keys, KNOWN_ORDER_KEYS)


# -------------------------------------------------------
# Step 1: Load unlabeled images from a flat folder
# -------------------------------------------------------

def load_unlabeled_images(image_dir: str,
                          extensions: list = None) -> list:
    """
    Load all images from a flat directory (no subfolders needed).

    Args:
        image_dir:   Path to folder containing images.
        extensions:  List of valid extensions (default: jpg, jpeg, png).

    Returns:
        Sorted list of absolute image paths.
    """
    if extensions is None:
        extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]

    image_paths = []
    for ext in extensions:
        pattern = os.path.join(image_dir, f"*.{ext}")
        image_paths.extend(glob.glob(pattern))

    # Also check one level of subfolders (in case images are loosely organized)
    for ext in extensions:
        pattern = os.path.join(image_dir, "**", f"*.{ext}")
        for p in glob.glob(pattern, recursive=True):
            if p not in image_paths:
                image_paths.append(p)

    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"\n  ERROR: No images found in {image_dir}")
        print(f"  Supported extensions: {extensions}")
        sys.exit(1)

    print(f"    Found {len(image_paths)} images in {image_dir}")
    return image_paths


# -------------------------------------------------------
# Step 5: Cluster embeddings
# -------------------------------------------------------

def cluster_embeddings(features: np.ndarray,
                       config: dict) -> np.ndarray:
    """
    Cluster the embeddings. For unlabeled data we default to HDBSCAN
    (no need to specify n_clusters) but support KMeans if n_clusters given.
    """
    method = config.get("clustering", {}).get("method", "hdbscan")
    params = config.get("clustering", {}).get("params", {})

    if method == "hdbscan":
        import hdbscan as hdb

        min_cluster_size = params.get("min_cluster_size", 3)
        min_samples = params.get("min_samples", None)
        # cluster_selection_method honors the config (was hardcoded "eom").
        # eom keeps the legacy default for configs that do not set it.
        cluster_method = params.get("cluster_selection_method", "eom")

        # Reduce dimensionality for clustering
        from src.reduction import reduce_features
        features_reduced, _ = reduce_features(
            features, config, random_state=config.get("seed", 42)
        )

        clusterer = hdb.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=cluster_method,
        )
        labels = clusterer.fit_predict(features_reduced)
        probabilities = clusterer.probabilities_

        n_clusters = len(set(labels[labels >= 0]))
        n_noise = (labels < 0).sum()
        print(f"    HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

        return labels, probabilities

    elif method == "kmeans":
        from sklearn.cluster import KMeans

        n_clusters = params.get("n_clusters", 10)

        # Reduce dimensionality for clustering
        from src.reduction import reduce_features
        features_reduced, _ = reduce_features(
            features, config, random_state=config.get("seed", 42)
        )

        km = KMeans(n_clusters=n_clusters, n_init=30, random_state=42)
        labels = km.fit_predict(features_reduced)
        print(f"    KMeans: {n_clusters} clusters")

        return labels, None

    else:
        raise ValueError(f"Unknown clustering method: {method}")


# -------------------------------------------------------
# Local mode helpers
# -------------------------------------------------------

def normalize_folder_name(raw_name):
    """'abrostola_tripartita' -> 'Abrostola tripartita'"""
    name = raw_name.replace("_", " ").strip()
    parts = name.split()
    if parts:
        parts[0] = parts[0].capitalize()
    return " ".join(parts)


def build_candidates_from_folders(images_dir):
    """
    Build candidate species list from imagefolder directory structure.
    Each subfolder name becomes a candidate species.

    Returns:
        candidate_names: list of species name strings
        taxonomy_table:  dict mapping name -> {genus, family, order}
                         (genus parsed from name, family/order empty)
        reference_map:   dict mapping name -> [list of image paths]
    """
    candidate_names = []
    taxonomy_table = {}
    reference_map = {}

    for entry in sorted(os.listdir(images_dir)):
        subdir = os.path.join(images_dir, entry)
        if not os.path.isdir(subdir):
            continue

        name = normalize_folder_name(entry)
        candidate_names.append(name)

        # Parse genus from first word
        parts = name.split()
        genus = parts[0] if parts else ""
        taxonomy_table[name] = {
            "genus": genus,
            "family": "",
            "order": "",
            "match_type": "folder",
            "gbif_confidence": 0,
        }

        # Collect up to 3 reference images from the folder
        refs = []
        for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
            refs.extend(glob.glob(os.path.join(subdir, f"*.{ext}")))
        reference_map[name] = sorted(refs)[:3]

    return candidate_names, taxonomy_table, reference_map


def build_candidates_from_csv(csv_path, images_dir=None):
    """
    Build candidate species list from a taxonomy CSV file.

    CSV must have column: class_name
    Optional columns: genus, family, order

    If images_dir is provided and has subfolders matching class names,
    reference images are collected from those folders.

    Returns:
        candidate_names, taxonomy_table, reference_map
    """
    candidate_names = []
    taxonomy_table = {}
    reference_map = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("class_name", "").strip()
            if not name:
                continue
            candidate_names.append(name)

            parts = name.split()
            genus = row.get("genus", parts[0] if parts else "").strip()
            taxonomy_table[name] = {
                "genus":  genus,
                "family": row.get("family", "").strip(),
                "order":  row.get("order", "").strip(),
                "match_type": row.get("match_type", "CSV"),
                "gbif_confidence": int(row.get("gbif_confidence", 0) or 0),
            }

    # Try to find reference images from dataset folders
    if images_dir:
        for name in candidate_names:
            # Try both original name and underscore version as folder name
            for folder_name in [name, name.replace(" ", "_"),
                                name.lower().replace(" ", "_")]:
                subdir = os.path.join(images_dir, folder_name)
                if os.path.isdir(subdir):
                    refs = []
                    for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
                        refs.extend(glob.glob(os.path.join(subdir, f"*.{ext}")))
                    reference_map[name] = sorted(refs)[:3]
                    break

    print(f"[taxonomy] Loaded {len(candidate_names)} species from {csv_path}")
    n_with_family = sum(1 for t in taxonomy_table.values() if t["family"])
    print(f"    {n_with_family}/{len(candidate_names)} have family taxonomy")

    return candidate_names, taxonomy_table, reference_map



# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Identify unknown biological specimens using BioCLIP-2.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--images", default=None,
                        help="Override: path to folder with images.")

    # Candidate source mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--local", action="store_true",
                            help="Local mode: use folder names as candidates.\n"
                                 "No internet needed. Images dir must be imagefolder layout.")
    mode_group.add_argument("--taxonomy", default=None, metavar="CSV_PATH",
                            help="CSV mode: read candidates + taxonomy from CSV file.\n"
                                 "CSV must have column: class_name\n"
                                 "Optional columns: genus, family, order")

    # GBIF-specific options
    parser.add_argument("--region", default=None,
                        choices=list(CONTINENT_MAP.keys()),
                        help="Override: geographic region for GBIF query.")
    parser.add_argument("--n-candidates", type=int, default=None,
                        help="Override: number of GBIF candidate species.")
    parser.add_argument("--skip-gbif", action="store_true",
                        help="Skip GBIF query (use cached candidates).")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip reference image download.")
    parser.add_argument("--skip-to", type=int, default=0,
                        help="Skip to step N (use cached results for earlier steps).\n"
                             "E.g. --skip-to 7 to regenerate visualizations and report only.")
    parser.add_argument("--embed-images", action="store_true", default=True,
                        help="Embed images as base64 in HTML report (default: True).\n"
                             "Images are always visible. Use --no-embed for smaller file.")
    parser.add_argument("--no-embed", action="store_true", default=False,
                        help="Disable image embedding (file:// paths, may not render in browser).")
    parser.add_argument("--apply-decisions", default=None, metavar="CSV_PATH",
                        dest="apply_decisions",
                        help="Apply the single decisions.csv produced by the\n"
                             "interactive review.html (columns: file_path,\n"
                             "final_label,status). Produces the clean\n"
                             "imagefolder dataset, provenance.csv, and summary.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # ---- Apply-decisions mode: single decisions file from review.html ----
    if args.apply_decisions:
        from src.annotation.interactive_report import apply_decisions
        from src.annotation.review import export_sorted_species, generate_summary

        base_dir = config["output_dir"]
        candidates_dirs = [base_dir.rstrip("/\\") + "_local", base_dir,
                           base_dir.rstrip("/\\") + "_csv"]
        output_dir = next(
            (d for d in candidates_dirs
             if os.path.exists(os.path.join(d, "annotations.csv"))), None)
        if output_dir is None:
            print("[error] No annotations.csv found in any of:")
            for d in candidates_dirs:
                print(f"        {d}")
            sys.exit(1)
        if not os.path.exists(args.apply_decisions):
            print(f"[error] File not found: {args.apply_decisions}")
            sys.exit(1)

        ann_path = os.path.join(output_dir, "annotations.csv")
        print(f"[apply] annotations: {ann_path}")
        print(f"[apply] decisions  : {args.apply_decisions}")
        annotations_df = pd.read_csv(ann_path)
        final_df = apply_decisions(args.apply_decisions, annotations_df)

        export_sorted_species(final_df, output_dir)
        generate_summary(final_df, output_dir, config)
        final_df.to_csv(os.path.join(output_dir, "annotations_final.csv"),
                        index=False)
        print(f"\n{'='*60}")
        print("  APPLY-DECISIONS COMPLETE")
        print(f"{'='*60}")
        print(f"  annotations_final.csv  -- corrected annotations")
        print(f"  sorted_species/        -- clean imagefolder dataset")
        print(f"  species_list.csv, summary.txt")
        print(f"{'='*60}")
        sys.exit(0)

    # Determine candidate mode
    if args.local:
        candidate_mode = "local"
    elif args.taxonomy:
        candidate_mode = "csv"
    else:
        candidate_mode = "gbif"

    output_dir = config["output_dir"]
    if candidate_mode == "local":
        output_dir = output_dir.rstrip("/\\") + "_local"
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "figures"))

    cache_dir = os.path.join(output_dir, ".cache", "gbif")

    # ---- Apply overrides ----
    if args.images:
        config["images_dir"] = args.images
    if args.region and config.get("gbif"):
        config["gbif"]["region"] = args.region
    if args.n_candidates and config.get("gbif"):
        config["gbif"]["n_candidates"] = args.n_candidates

    # Image location: --images override, else top-level images_dir, else the
    # standard dataset.root used by every dataset config.
    images_dir = (config.get("images_dir")
                  or config.get("dataset", {}).get("root"))
    if not images_dir:
        print("\n  ERROR: No image directory. Set dataset.root in the config "
              "or pass --images <path>.")
        sys.exit(1)
    config["images_dir"] = images_dir

    header("UNLABELED SPECIMEN IDENTIFICATION PIPELINE")
    mode_labels = {"gbif": "GBIF API", "local": "Local (folder names)",
                   "csv": f"CSV ({args.taxonomy})"}
    print(f"\n  Images      : {images_dir}")
    print(f"  Mode        : {mode_labels[candidate_mode]}")
    print(f"  Model       : {config['model']['name']}")
    print(f"  Output      : {output_dir}")

    # =========================================================
    # STEP 1: Load images
    # =========================================================
    subheader("Step 1 - Loading images")
    t0 = time.time()

    extensions = config.get("dataset", {}).get(
        "image_extensions", ["jpg", "jpeg", "png"]
    )
    image_paths = load_unlabeled_images(images_dir, extensions)

    # If the dataset config defines an enabled subsample block, apply the
    # SAME seeded subsample as the Task 1 clustering experiments so the
    # annotation run operates on the identical data slice. Reuses the
    # tested load_imagefolder + subsample_dataset (no new logic). Fails
    # loud rather than silently processing the full dataset.
    sub_cfg = config.get("dataset", {}).get("subsample", {})
    if sub_cfg.get("enabled"):
        from src.data.dataset import load_imagefolder, subsample_dataset
        try:
            paths, labels, class_names = load_imagefolder(
                images_dir, extensions)
        except Exception as e:
            print(f"\n  ERROR: subsample is enabled but the dataset could "
                  f"not be loaded as an imagefolder ({images_dir}): {e}")
            sys.exit(1)
        if not paths:
            print(f"\n  ERROR: subsample is enabled but no class subfolders "
                  f"were found under {images_dir}.")
            sys.exit(1)
        before = len(paths)
        paths, labels, class_names = subsample_dataset(
            paths, labels, class_names, config)
        if not paths:
            print("\n  ERROR: subsample produced 0 images. Check n_classes / "
                  "min_images_per_class in the config.")
            sys.exit(1)
        image_paths = paths
        print(f"    Subsample applied: {before} -> {len(image_paths)} "
              f"images, {len(class_names)} classes "
              f"(seed {sub_cfg.get('random_seed', 42)})")

    n_images = len(image_paths)

    print(f"    Loaded {n_images} images in {time.time()-t0:.1f}s")

    # =========================================================
    # STEP 2: Build candidate species list
    # =========================================================
    subheader("Step 2 - Building candidate species list")
    t0 = time.time()

    reference_map = {}

    if candidate_mode == "local":
        # ---- LOCAL MODE: folder names as candidates ----
        candidate_names, taxonomy_table, reference_map = \
            build_candidates_from_folders(images_dir)

        if not candidate_names:
            print("\n  ERROR: No subfolders found in images_dir.")
            print("  Local mode requires imagefolder layout: images_dir/<species>/img.jpg")
            sys.exit(1)

        n_families = len(set(t["family"] for t in taxonomy_table.values() if t["family"]))
        print(f"    {len(candidate_names)} species from folder names")
        if n_families:
            print(f"    {n_families} families")
        else:
            print(f"    No family/order taxonomy (genus parsed from names)")
        n_refs = sum(1 for v in reference_map.values() if v)
        print(f"    {n_refs}/{len(candidate_names)} species have reference images")

    elif candidate_mode == "csv":
        # ---- CSV MODE: taxonomy file as candidate list ----
        candidate_names, taxonomy_table, reference_map = \
            build_candidates_from_csv(args.taxonomy, images_dir)

        if not candidate_names:
            print(f"\n  ERROR: No species found in {args.taxonomy}")
            sys.exit(1)

        n_families = len(set(t["family"] for t in taxonomy_table.values() if t["family"]))
        print(f"    {len(candidate_names)} species from CSV")
        print(f"    {n_families} families with taxonomy")

    else:
        # ---- GBIF MODE: query GBIF API ----
        # GBIF mode REQUIRES an explicit gbif: block with at least order_key
        # and region. Silently defaulting either of these would cause the
        # pipeline to query the wrong taxon or the wrong continent and produce
        # plausible-looking but incorrect candidate lists (e.g. moth species
        # returned for a bird dataset). Fail fast with a clear message instead.
        gbif_cfg = config.get("gbif")
        missing = []
        if not isinstance(gbif_cfg, dict):
            missing = ["order_key", "region"]
        else:
            if "order_key" not in gbif_cfg:
                missing.append("order_key")
            if "region" not in gbif_cfg:
                missing.append("region")
        if missing:
            known = ", ".join(f"{k}={v.split(' ')[0]}"
                              for k, v in KNOWN_ORDER_KEYS.items())
            print("\n  ERROR: GBIF mode requires an explicit 'gbif:' block in "
                  f"the config.\n         Missing: {', '.join(missing)}")
            print(f"         Config file: {args.config}")
            print("\n  Add a block like this to the config and re-run, or use\n"
                  "  --local (folder names as candidates) or\n"
                  "  --taxonomy <csv> (CSV candidate list):\n")
            print("    gbif:")
            print("      order_key: 797        # 797=Lepidoptera, 7192=Aves,")
            print(f"                            # known keys: {known}")
            print("      region: europe        # europe | asia | north_america |")
            print("                            # south_america | africa | oceania")
            print("      n_candidates: 400     # optional, default 200")
            print("      query_mode: balanced  # optional, balanced | top")
            sys.exit(1)

        region          = gbif_cfg["region"]
        order_key       = gbif_cfg["order_key"]       # int or list of ints
        n_candidates    = gbif_cfg.get("n_candidates", 200)
        query_mode      = gbif_cfg.get("query_mode", "balanced")
        per_family      = gbif_cfg.get("per_family", 40)
        rare_per_family = gbif_cfg.get("rare_per_family", 10)

        if region not in CONTINENT_MAP:
            print(f"\n  ERROR: Unknown region '{region}'. "
                  f"Valid: {', '.join(CONTINENT_MAP)}")
            sys.exit(1)
        continent = CONTINENT_MAP[region]

        # Visible "about to query GBIF" banner so a wrong order_key is caught
        # within the first second, before any slow API call. If the taxon name
        # does not match the dataset, the operator can Ctrl-C immediately.
        print("\n  ----------------------------------------------------------")
        print("    GBIF QUERY PLAN")
        print("  ----------------------------------------------------------")
        print(f"    Taxon : {describe_order_keys(order_key)}")
        print(f"    Region: {region} ({continent})")
        print(f"    Mode  : {query_mode}")
        print(f"    Config: {args.config}")
        print("  ----------------------------------------------------------")
        print("    If this does not match your dataset, press Ctrl+C now.")
        print("  ----------------------------------------------------------\n")

        if query_mode == "balanced":
            candidates = query_candidates_balanced(
                order_key=order_key,
                continent=continent,
                per_family=per_family,
                rare_per_family=rare_per_family,
                cache_dir=cache_dir,
            )
        else:
            include_rare = gbif_cfg.get("include_rare", True)
            candidates = query_candidate_species(
                order_key=order_key,
                continent=continent,
                limit=n_candidates,
                include_rare=include_rare,
                cache_dir=cache_dir,
            )

        if not candidates:
            print("\n  ERROR: No candidate species found from GBIF.")
            print("  Check your region and order_key settings.")
            sys.exit(1)

        candidate_names = candidates_to_class_names(candidates)
        taxonomy_table  = candidates_to_taxonomy(candidates)

        print(f"\n    {len(candidate_names)} candidate species, "
              f"{len(set(c['family'] for c in candidates))} families")

        for c in candidates[:5]:
            print(f"      {c['species']:40s}  {c['family']:20s}  "
                  f"({c['n_occurrences']} occurrences)")
        if len(candidates) > 5:
            print(f"      ... and {len(candidates)-5} more")

        # Save candidate list
        cand_path = os.path.join(output_dir, "gbif_candidates.csv")
        pd.DataFrame(candidates).to_csv(cand_path, index=False)
        print(f"    Candidate list saved to {cand_path}")

    print(f"    Completed in {time.time()-t0:.1f}s")

    # =========================================================
    # STEP 3: Collect reference images
    # =========================================================
    if candidate_mode == "gbif" and not args.skip_download:
        subheader("Step 3 - Collecting GBIF reference image URLs")
        t0 = time.time()

        gbif_cfg = config.get("gbif", {})
        images_per = gbif_cfg.get("images_per_species", 3)
        reference_map = collect_reference_urls(
            candidates,
            images_per_species=images_per,
            cache_dir=cache_dir,
        )

        n_with_refs = sum(1 for v in reference_map.values() if v)
        print(f"\n    {n_with_refs}/{len(candidates)} species have reference URLs")
        print(f"    URL collection completed in {time.time()-t0:.1f}s")
    elif candidate_mode != "gbif":
        subheader("Step 3 - Reference images from dataset")
        n_with_refs = sum(1 for v in reference_map.values() if v)
        print(f"    {n_with_refs}/{len(candidate_names)} species have reference images")

    # =========================================================
    # STEP 4: Extract BioCLIP-2 embeddings
    # =========================================================
    subheader("Step 4 - Extracting BioCLIP-2 embeddings")
    t0 = time.time()

    features_path = os.path.join(output_dir, "features.npz")

    if os.path.exists(features_path):
        from src.utils.io import load_features
        features, _, _, _ = load_features(features_path)
        print(f"    Loaded cached features: {features.shape}")
    else:
        extractor = build_extractor(config)
        features = extractor.extract(image_paths)
        # Save with dummy labels (all -1 since we don't know ground truth)
        dummy_labels = np.full(len(image_paths), -1, dtype=np.int32)
        save_features(features_path, features, dummy_labels,
                      image_paths, candidate_names)
        print(f"    Features: {features.shape}")

    print(f"    Embedding extraction completed in {time.time()-t0:.1f}s")

    # =========================================================
    # STEP 5: Cluster embeddings
    # =========================================================
    subheader("Step 5 - Clustering embeddings")
    t0 = time.time()

    cluster_result = cluster_embeddings(features, config)
    if isinstance(cluster_result, tuple):
        cluster_ids, probabilities = cluster_result
    else:
        cluster_ids = cluster_result
        probabilities = None

    n_clusters = len(set(cluster_ids[cluster_ids >= 0]))
    n_noise = (cluster_ids < 0).sum()

    metrics_table([
        {"label": "Clusters found",  "value": n_clusters},
        {"label": "Noise points",    "value": n_noise},
        {"label": "Images",          "value": len(cluster_ids)},
        {"label": "Time",            "value": f"{time.time()-t0:.1f}s"},
    ], title="Clustering results")

    # Save clusters
    from src.utils.io import save_clusters
    clusters_path = os.path.join(output_dir, "clusters.npz")
    save_clusters(clusters_path, cluster_ids, "hdbscan", {})

    if probabilities is not None:
        np.save(os.path.join(output_dir, "hdbscan_probabilities.npy"),
                probabilities)

    # =========================================================
    # STEP 6: Annotate clusters via zero-shot text matching
    # =========================================================
    ann_path = os.path.join(output_dir, "annotations.csv")
    text_cache_path = os.path.join(output_dir, "text_embeddings.npz")
    prompt_templates = config.get("model", {}).get("prompt_templates", None)

    if args.skip_to >= 7 and os.path.exists(ann_path):
        # Load cached annotations - skip model loading entirely
        subheader("Step 6 - Loading cached annotations (--skip-to >= 7)")
        annotations_df = pd.read_csv(ann_path)
        print(f"    Loaded {len(annotations_df)} annotations from {ann_path}")

        # Also load expanded candidates if available
        expanded_path = os.path.join(output_dir, "gbif_candidates_expanded.csv")
        if os.path.exists(expanded_path):
            expanded_df = pd.read_csv(expanded_path)
            candidate_names = list(expanded_df["species"].unique())
            print(f"    Using expanded candidate list: {len(candidate_names)} species")
    else:
        subheader("Step 6 - Annotating clusters with BioCLIP-2 text matching")
        t0 = time.time()

        # Encode text embeddings for all candidate species (cached to disk)
        extractor = None  # only load model if cache miss
        text_embs = None
        if os.path.exists(text_cache_path):
            try:
                cached = np.load(text_cache_path, allow_pickle=True)
                cached_names = list(cached["names"])
                if cached_names == candidate_names:
                    text_embs = cached["embeddings"]
                    print(f"    Loaded cached text embeddings: {text_embs.shape}")
                else:
                    print(f"    Candidate list changed ({len(cached_names)} -> "
                          f"{len(candidate_names)}), re-encoding...")
            except Exception:
                pass

        if text_embs is None:
            extractor = build_extractor(config)
            text_embs = extractor.encode_text(
                candidate_names,
                prompt_templates=prompt_templates,
                taxonomy_table=taxonomy_table,
            )
            # Cache for next run
            np.savez(text_cache_path,
                     embeddings=text_embs,
                     names=np.array(candidate_names, dtype=object))
            print(f"    Text embeddings: {text_embs.shape} "
                  f"({len(candidate_names)} candidates)")
            print(f"    Cached to {text_cache_path}")

        # Run annotation with uncertainty quantification
        annotations_df = run_annotation(
            features=features,
            cluster_ids=cluster_ids,
            text_embs=text_embs,
            species_names=candidate_names,
            taxonomy_table=taxonomy_table,
            config=config,
            probabilities=probabilities,
            image_paths=image_paths,
        )

        # Save annotations
        annotations_df.to_csv(ann_path, index=False)
        print(f"    Annotations saved to {ann_path}")

        # Print annotation summary
        tier_counts = annotations_df["confidence_tier"].value_counts()
        total = len(annotations_df)

        metrics_table([
            {"label": "Total images",     "value": total},
            {"label": "Auto-accept",      "value": tier_counts.get("auto_accept", 0),
             "note": f"({tier_counts.get('auto_accept', 0)/total*100:.1f}%)"},
            {"label": "Review",           "value": tier_counts.get("review", 0),
             "note": f"({tier_counts.get('review', 0)/total*100:.1f}%)"},
            {"label": "Human required",   "value": tier_counts.get("human_required", 0),
             "note": f"({tier_counts.get('human_required', 0)/total*100:.1f}%)"},
            {"label": "Unique species",   "value": annotations_df["ai_annotation"].nunique()},
            {"label": "Time",             "value": f"{time.time()-t0:.1f}s"},
        ], title="Annotation results")

        # Show per-cluster annotations
        print(f"\n    Per-cluster annotations:")
        for cid in sorted(annotations_df["cluster_id"].unique()):
            if cid < 0:
                continue
            cluster_df = annotations_df[annotations_df["cluster_id"] == cid]
            species = cluster_df["ai_annotation"].iloc[0]
            second = cluster_df["ai_second_choice"].iloc[0]
            margin = cluster_df["margin"].iloc[0]
            n = len(cluster_df)
            tier = cluster_df["confidence_tier"].iloc[0]
            has_ref = "yes" if reference_map.get(species) else "no"
            print(f"      Cluster {cid:2d}: {species:40s} "
                  f"({n} imgs, margin={margin:.3f}, {tier}, ref={has_ref})")
            if second and str(second) != "nan":
                print(f"                  runner-up: {second}")

        print(f"\n    Annotation completed in {time.time()-t0:.1f}s")

        # =========================================================
        # STEP 6b: Expand candidates for low-confidence clusters
        #          (GBIF mode only - requires API access)
        # =========================================================
        n_low_conf = (annotations_df["confidence_tier"].isin(
            ["review", "human_required"]
        )).sum()

        if n_low_conf > 0 and candidate_mode == "gbif" and not args.skip_gbif:
            subheader("Step 6b - Expanding candidates for low-confidence clusters")
            t0 = time.time()

            expanded = expand_candidates_for_gaps(
                annotations_df, candidates, config, cache_dir=cache_dir,
            )

            n_new = len(expanded) - len(candidates)
            if n_new > 0:
                print(f"    {n_new} new candidate species added")

                # Update candidate list and re-annotate
                candidates = expanded
                candidate_names = candidates_to_class_names(candidates)
                taxonomy_table = candidates_to_taxonomy(candidates)

                # Re-encode text with expanded list (reuse already-loaded model)
                if extractor is None:
                    extractor = build_extractor(config)
                text_embs = extractor.encode_text(
                    candidate_names,
                    prompt_templates=prompt_templates,
                    taxonomy_table=taxonomy_table,
                )
                print(f"    Re-encoded text embeddings: {text_embs.shape}")

                # Update text embedding cache
                np.savez(text_cache_path,
                         embeddings=text_embs,
                         names=np.array(candidate_names, dtype=object))

                # Re-run annotation
                annotations_df = run_annotation(
                    features=features,
                    cluster_ids=cluster_ids,
                    text_embs=text_embs,
                    species_names=candidate_names,
                    taxonomy_table=taxonomy_table,
                    config=config,
                    probabilities=probabilities,
                    image_paths=image_paths,
                )

                # Save updated annotations
                annotations_df.to_csv(ann_path, index=False)

                # Updated stats
                tier_counts = annotations_df["confidence_tier"].value_counts()
                total = len(annotations_df)
                n_low_new = tier_counts.get("review", 0) + tier_counts.get("human_required", 0)

                print(f"\n    After expansion: {n_low_new} flagged "
                      f"(was {n_low_conf} before)")
                print(f"    Expansion completed in {time.time()-t0:.1f}s")

                # Save expanded candidate list
                cand_path = os.path.join(output_dir, "gbif_candidates_expanded.csv")
                pd.DataFrame(candidates).to_csv(cand_path, index=False)

                # Collect reference URLs for new candidates
                new_only = [c for c in candidates
                            if c.get("source") == "stage2_expansion"]
                if new_only:
                    subheader("Collecting reference URLs for new candidates")
                    images_per = gbif_cfg.get("images_per_species", 3)
                    new_refs = collect_reference_urls(
                        new_only,
                        images_per_species=images_per,
                        cache_dir=cache_dir,
                    )
                    reference_map.update(new_refs)
            else:
                print("    No new candidates found. Keeping original annotation.")

    # Refresh tier_counts after potential re-annotation
    tier_counts = annotations_df["confidence_tier"].value_counts()
    total = len(annotations_df)

    # =========================================================
    # STEP 7: Generate uncertainty visualizations
    # =========================================================
    subheader("Step 7 - Uncertainty visualizations")
    fig_dir = os.path.join(output_dir, "figures")

    cluster_method = config.get("clustering", {}).get("method", "hdbscan")
    plot_uncertainty_histogram(annotations_df, fig_dir, config, method=cluster_method)
    plot_confidence_tiers(annotations_df, fig_dir, config, method=cluster_method)

    # UMAP annotation map (if features_2d available)
    try:
        from src.utils.io import get_2d_projection
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_jobs value.*",
                                    category=UserWarning)
            features_2d = get_2d_projection(features, config, output_dir)
        plot_annotation_map(annotations_df, features_2d, fig_dir, config,
                            method=cluster_method)
    except Exception as e:
        print(f"    [warn] Could not generate UMAP map: {e}")

    # =========================================================
    # STEP 8: Generate interactive review report
    # =========================================================
    subheader("Step 8 - Generating interactive review report")

    # One self-contained review.html for the ecologist (zero setup, offline
    # for local/csv runs). Finalize in the browser produces decisions.csv,
    # which is applied back with:
    #   run_unlabeled.py --config ... --apply-decisions decisions.csv
    from src.annotation.interactive_report import generate_interactive_report
    generate_interactive_report(
        annotations_df, reference_map,
        os.path.join(output_dir, "review.html"),
        config, candidate_mode=candidate_mode,
    )

    # =========================================================
    # STEP 10 (local/csv mode): Accuracy check against ground truth
    # =========================================================
    if candidate_mode in ("local", "csv"):
        subheader("Step 10 - Accuracy check against ground truth")

        # Extract ground truth label from image path (parent folder name)
        gt_labels = []
        for p in image_paths:
            folder = os.path.basename(os.path.dirname(p))
            gt_labels.append(normalize_folder_name(folder))

        ai_labels = annotations_df["ai_annotation"].tolist()

        def _canon(s):
            """Canonical form for name comparison: lowercase, spaces, stripped."""
            return s.lower().replace("_", " ").strip()

        correct = sum(1 for gt, ai in zip(gt_labels, ai_labels)
                      if _canon(gt) == _canon(ai))
        total_check = len(gt_labels)
        acc = correct / total_check * 100 if total_check > 0 else 0

        print(f"    Annotation accuracy: {correct}/{total_check} = {acc:.1f}%")

        # Per-cluster accuracy
        print(f"\n    Per-cluster breakdown:")
        for cid in sorted(annotations_df["cluster_id"].unique()):
            if cid < 0:
                continue
            mask = annotations_df["cluster_id"] == cid
            cluster_gt = [gt_labels[i] for i in range(len(gt_labels)) if mask.iloc[i]]
            cluster_ai = annotations_df.loc[mask, "ai_annotation"].iloc[0]
            gt_counter = Counter(cluster_gt)
            majority_gt, majority_n = gt_counter.most_common(1)[0]
            match_str = "OK" if _canon(cluster_ai) == _canon(majority_gt) else "WRONG"
            print(f"      Cluster {cid:2d}: AI={cluster_ai:35s} "
                  f"GT={majority_gt:35s} [{match_str}] "
                  f"({majority_n}/{sum(gt_counter.values())} pure)")

    # =========================================================
    # Summary
    # =========================================================
    header("PIPELINE COMPLETE")
    print(f"\n  Mode            : {mode_labels[candidate_mode]}")
    print(f"  Images processed: {n_images}")
    print(f"  Clusters found  : {n_clusters}")
    print(f"  Species matched : {annotations_df['ai_annotation'].nunique()}")
    print(f"  Auto-accepted   : {tier_counts.get('auto_accept', 0)} "
          f"({tier_counts.get('auto_accept', 0)/total*100:.1f}%)")
    print(f"  Need review     : {tier_counts.get('review', 0) + tier_counts.get('human_required', 0)} "
          f"({(tier_counts.get('review', 0) + tier_counts.get('human_required', 0))/total*100:.1f}%)")
    print(f"\n  Review report   : {os.path.join(output_dir, 'review.html')}")
    if candidate_mode == "gbif":
        print(f"\n  Next: Open review.html in a browser.")
        print(f"  (Requires internet - reference images load from GBIF on-the-fly)")
    else:
        print(f"\n  Next: Open review.html in a browser to review.")
    print(f"  Use buttons below each image, then export CSV.")

    done(output_dir)


if __name__ == "__main__":
    main()
