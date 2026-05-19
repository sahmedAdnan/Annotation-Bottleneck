"""
cli/reduction_analysis_unlabeled.py
-------------------------------------
Find the optimal dimensionality reduction for UNLABELED datasets.
No ground-truth labels required — uses fully unsupervised quality criteria.

Criteria used:
    Full-D    -> K-Means silhouette on raw features (baseline, no reduction)
    PCA path  -> Silhouette score after K-Means
                 + Cumulative Explained Variance (CEV) as reference
    UMAP path -> Trustworthiness (neighbourhood preservation, no clustering needed)
                 + Weighted Silhouette after HDBSCAN: silhouette x (1 - noise_fraction)
    PCA+UMAP  -> PCA to intermediate dim (--pca-components, default 100),
                 then UMAP to target dim; evaluated the same as UMAP path.

For the labeled benchmark version (uses ARI), see:
    cli/reduction_analysis.py

Requires:
    <output_dir>/features.npz  (produced by cli/extract.py)

Usage:
    # PCA only -- saves pca_unlabeled.csv
    python cli/reduction_analysis_unlabeled.py --config configs/eumoth.yaml --mode pca
    python cli/reduction_analysis_unlabeled.py --config configs/eumoth.yaml --mode pca --dims 50 100 200

    # UMAP only -- saves umap_unlabeled.csv
    python cli/reduction_analysis_unlabeled.py --config configs/eumoth.yaml --mode umap
    python cli/reduction_analysis_unlabeled.py --config configs/eumoth.yaml --mode umap --dims 5 10 15 20 50 100

    # PCA+UMAP -- saves pca_umap_unlabeled.csv
    python cli/reduction_analysis_unlabeled.py --config configs/eumoth.yaml --mode pca+umap
    python cli/reduction_analysis_unlabeled.py --config configs/eumoth.yaml --mode pca+umap --pca-components 100 --dims 5 10 15 20

    # All reducers + full-dim baseline -- saves unlabeled_comparison.csv (default)
    python cli/reduction_analysis_unlabeled.py --config configs/eumoth.yaml

Output files per mode:
    pca_unlabeled.csv / pca_unlabeled_recommendation.txt
    umap_unlabeled.csv / umap_unlabeled_recommendation.txt
    pca_umap_unlabeled.csv / pca_umap_unlabeled_recommendation.txt
    unlabeled_comparison.csv / unlabeled_recommendation.txt  (mode=all / default)

Notes:
    - num_classes in the config is used as the K-Means k for PCA silhouette and full-D baseline.
      For truly unknown datasets, set num_classes to your best estimate.
    - HDBSCAN on UMAP/PCA+UMAP path auto-determines k -- no estimate needed.
    - Trustworthiness is computed on a random sample (max 2000 points) for speed.
    - Silhouette is computed on a random sample (max 5000 points) for speed.
"""

import argparse
import os
import sys
import time
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_config, load_features, set_seed, ensure_dir
from src.utils.pretty import header, subheader, done

# -------------------------------------------------------
# Defaults
# -------------------------------------------------------

DEFAULT_PCA_DIMS      = [50, 100, 200]
DEFAULT_UMAP_DIMS     = [5, 10, 15, 20, 50, 100]
DEFAULT_PCA_UMAP_DIMS = [5, 10, 15, 20, 50, 100]   # UMAP target dims for pca+umap mode
DEFAULT_PCA_COMPONENTS = 100                          # intermediate PCA dim for pca+umap
DEFAULT_MCS_RANGE     = [3, 5, 7, 10, 15]
TRUST_SAMPLE          = 2000   # max points for trustworthiness (slow on large N)
SIL_SAMPLE            = 5000   # max points for silhouette score

MODE_OUTPUT = {
    "pca":      ("pca_unlabeled.csv",        "pca_unlabeled_recommendation.txt"),
    "umap":     ("umap_unlabeled.csv",       "umap_unlabeled_recommendation.txt"),
    "pca+umap": ("pca_umap_unlabeled.csv",   "pca_umap_unlabeled_recommendation.txt"),
    "all":      ("unlabeled_comparison.csv", "unlabeled_recommendation.txt"),
}


# -------------------------------------------------------
# Reduction helpers
# -------------------------------------------------------

def fit_pca(features, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    variance = float(pca.explained_variance_ratio_.sum())
    print(f"    [PCA]  {features.shape[1]}D -> {n_components}D  "
          f"(variance: {variance*100:.1f}%)")
    return reduced, variance


def fit_umap(features, n_components, n_neighbors=15, min_dist=0.1,
             metric="cosine", random_state=42):
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden")
        reducer = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        print(f"    [UMAP] {features.shape[1]}D -> {n_components}D "
              f"(nn={n_neighbors}, md={min_dist}) ...")
        t0 = time.time()
        reduced = reducer.fit_transform(features)
        print(f"           done in {time.time()-t0:.1f}s")
    return reduced


# -------------------------------------------------------
# Unsupervised quality criteria
# -------------------------------------------------------

def silhouette_sampled(features, labels, max_samples=SIL_SAMPLE, seed=42):
    """Silhouette score on a random sample for speed."""
    from sklearn.metrics import silhouette_score
    N = len(labels)
    if N > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, max_samples, replace=False)
        return silhouette_score(features[idx], labels[idx])
    return silhouette_score(features, labels)


def trustworthiness_sampled(features_orig, features_reduced,
                            n_neighbors=15, max_samples=TRUST_SAMPLE, seed=42):
    """
    Trustworthiness measures how well local neighbourhood structure
    from the original space is preserved in the reduced embedding.
    Score in [0, 1]: 1.0 = perfect preservation.
    """
    from sklearn.manifold import trustworthiness
    N = len(features_orig)
    if N > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, max_samples, replace=False)
        return trustworthiness(features_orig[idx], features_reduced[idx],
                               n_neighbors=n_neighbors)
    return trustworthiness(features_orig, features_reduced, n_neighbors=n_neighbors)


# -------------------------------------------------------
# Per-dimension evaluation
# -------------------------------------------------------

def evaluate_pca_dim(features_reduced, variance, n_clusters, seed=42):
    """
    K-Means silhouette on PCA-reduced features.
    n_clusters: user's estimate of the number of species.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init=20,
                max_iter=300, random_state=seed)
    labels = km.fit_predict(features_reduced)

    n_unique = len(set(labels))
    if n_unique < 2:
        return {"silhouette": None, "variance_retained": variance,
                "n_clusters": n_unique}

    sil = silhouette_sampled(features_reduced, labels, seed=seed)
    return {
        "silhouette":       round(float(sil), 4),
        "variance_retained": round(variance * 100, 1),
        "n_clusters":       n_unique,
    }


def evaluate_umap_dim(features_orig, features_reduced, mcs_range, seed=42):
    """
    Trustworthiness + best weighted silhouette (HDBSCAN) on UMAP-reduced features.
    weighted_silhouette = silhouette(non-noise) x (1 - noise_fraction)
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")

    trust = trustworthiness_sampled(features_orig, features_reduced,
                                    n_neighbors=15, seed=seed)

    best_weighted = -1.0
    best = None

    for mcs in mcs_range:
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=mcs, metric="euclidean"
        )
        labels = clusterer.fit_predict(features_reduced)
        n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac  = float((labels == -1).mean())
        mask        = labels >= 0

        if n_clusters < 2 or mask.sum() < 20:
            continue

        sil      = silhouette_sampled(features_reduced[mask], labels[mask], seed=seed)
        weighted = float(sil) * (1.0 - noise_frac)

        if weighted > best_weighted:
            best_weighted = weighted
            best = {
                "mcs":                  mcs,
                "n_clusters":           n_clusters,
                "noise_frac":           round(noise_frac * 100, 1),
                "silhouette":           round(float(sil), 4),
                "weighted_silhouette":  round(weighted, 4),
            }

    return {
        "trustworthiness": round(float(trust), 4),
        "best":            best,
    }


def evaluate_full_dim(features, n_clusters, mcs_range, seed=42):
    """
    Baseline evaluation on raw full-dimensional features (no reduction).
    Computes K-Means silhouette (PCA reference) and HDBSCAN weighted silhouette.
    Trustworthiness is 1.0 by definition (no reduction applied).
    """
    from sklearn.cluster import KMeans

    # K-Means silhouette
    km = KMeans(n_clusters=n_clusters, n_init=20, max_iter=300, random_state=seed)
    labels = km.fit_predict(features)
    n_unique = len(set(labels))
    km_sil = None
    if n_unique >= 2:
        km_sil = round(float(silhouette_sampled(features, labels, seed=seed)), 4)

    # HDBSCAN weighted silhouette
    try:
        import hdbscan as hdbscan_lib
        best_weighted = -1.0
        best_hdb = None
        for mcs in mcs_range:
            clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
            hlabels = clusterer.fit_predict(features)
            n_hclusters = len(set(hlabels)) - (1 if -1 in hlabels else 0)
            noise_frac   = float((hlabels == -1).mean())
            mask         = hlabels >= 0
            if n_hclusters < 2 or mask.sum() < 20:
                continue
            sil      = silhouette_sampled(features[mask], hlabels[mask], seed=seed)
            weighted = float(sil) * (1.0 - noise_frac)
            if weighted > best_weighted:
                best_weighted = weighted
                best_hdb = {
                    "mcs":               mcs,
                    "n_clusters":        n_hclusters,
                    "noise_frac":        round(noise_frac * 100, 1),
                    "silhouette":        round(float(sil), 4),
                    "weighted_silhouette": round(weighted, 4),
                }
    except ImportError:
        best_hdb = None

    return {
        "path":             "full-dim",
        "dim":              features.shape[1],
        "km_silhouette":    km_sil,
        "trustworthiness":  1.0,
        "best":             best_hdb,
    }


def evaluate_pca_umap_dim(features_orig, pca_components, umap_dim,
                           mcs_range, n_neighbors=15, min_dist=0.1,
                           metric="cosine", seed=42):
    """
    Two-stage PCA+UMAP: PCA to pca_components, then UMAP to umap_dim.
    Evaluated the same way as the pure UMAP path.
    Trustworthiness compares original features to the final PCA+UMAP embedding.
    """
    reduced_pca, _ = fit_pca(features_orig, pca_components)
    reduced_umap   = fit_umap(reduced_pca, umap_dim,
                              n_neighbors=n_neighbors, min_dist=min_dist,
                              metric=metric, random_state=seed)
    result = evaluate_umap_dim(features_orig, reduced_umap, mcs_range, seed=seed)
    return result


# -------------------------------------------------------
# Print helpers
# -------------------------------------------------------

def print_pca_table(rows):
    print(f"\n  {'PCA dim':>8}  {'CEV %':>7}  {'Silhouette':>12}  {'K found':>8}")
    print("  " + "-" * 44)
    for r in rows:
        sil_str = f"{r['silhouette']:>12.4f}" if r["silhouette"] is not None else f"{'--':>12}"
        print(f"  {r['pca_dim']:>8}  {r['variance_retained']:>6.1f}%  "
              f"{sil_str}  {r['n_clusters']:>8}")
    print("  " + "-" * 44)


def print_umap_table(rows, label="UMAP dim"):
    print(f"\n  {label:>12}  {'Trust':>7}  {'W.Silhouette':>14}  "
          f"{'Silhouette':>12}  {'Noise%':>7}  {'K found':>8}  {'MCS':>5}")
    print("  " + "-" * 75)
    for r in rows:
        b   = r.get("best") or {}
        dim_key = "umap_dim" if "umap_dim" in r else "pca_umap_dim"
        ws  = f"{b.get('weighted_silhouette', 0.0):>14.4f}" if b else f"{'--':>14}"
        sil = f"{b.get('silhouette', 0.0):>12.4f}"         if b else f"{'--':>12}"
        nf  = f"{b.get('noise_frac', 0.0):>6.1f}%"         if b else f"{'--':>7}"
        kf  = f"{b.get('n_clusters', '--'):>8}"
        mcs = f"{b.get('mcs', '--'):>5}"
        print(f"  {r[dim_key]:>12}  {r['trustworthiness']:>7.4f}  "
              f"{ws}  {sil}  {nf}  {kf}  {mcs}")
    print("  " + "-" * 75)


def print_recommendation(full_dim_row, pca_rows, umap_rows, pca_umap_rows,
                         n_clusters_est, output_dir, rec_filename):
    lines = []
    lines.append("\n  RECOMMENDATION (unsupervised)")
    lines.append("  " + "=" * 65)
    lines.append(f"  Estimated species count used for K-Means: {n_clusters_est}")

    # Full-dim baseline
    if full_dim_row:
        km_sil = full_dim_row.get("km_silhouette")
        km_str = f"{km_sil:.4f}" if km_sil is not None else "--"
        b = full_dim_row.get("best") or {}
        ws_str = f"{b.get('weighted_silhouette', 0.0):.4f}" if b else "--"
        lines.append(f"\n  Full-D baseline ({full_dim_row['dim']}D)  "
                     f"km_silhouette={km_str}  hdbscan_weighted_sil={ws_str}")

    # PCA
    if pca_rows:
        valid = [r for r in pca_rows if r["silhouette"] is not None]
        if valid:
            best = max(valid, key=lambda r: r["silhouette"])
            lines.append(f"\n  PCA path  -> {best['pca_dim']}D  "
                         f"(silhouette={best['silhouette']:.4f}, "
                         f"CEV={best['variance_retained']:.1f}%)")
            lines.append(f"    reduction:\n"
                         f"      method: pca\n"
                         f"      n_components: {best['pca_dim']}")

    # UMAP
    if umap_rows:
        valid = [r for r in umap_rows
                 if r.get("best") and r["best"]["weighted_silhouette"] > 0]
        if valid:
            best_w = max(valid, key=lambda r: r["best"]["weighted_silhouette"])
            best_t = max(umap_rows, key=lambda r: r["trustworthiness"])
            b = best_w["best"]
            lines.append(f"\n  UMAP path -> {best_w['umap_dim']}D  "
                         f"(weighted_silhouette={b['weighted_silhouette']:.4f}, "
                         f"trust={best_w['trustworthiness']:.4f}, "
                         f"noise={b['noise_frac']:.1f}%)")
            lines.append(f"    reduction:\n"
                         f"      method: umap\n"
                         f"      n_components: {best_w['umap_dim']}")
            if best_t["umap_dim"] != best_w["umap_dim"]:
                lines.append(f"    Note: highest trustworthiness at "
                             f"{best_t['umap_dim']}D "
                             f"(trust={best_t['trustworthiness']:.4f}) — "
                             f"consider if noise fraction is a concern.")

    # PCA+UMAP
    if pca_umap_rows:
        valid = [r for r in pca_umap_rows
                 if r.get("best") and r["best"]["weighted_silhouette"] > 0]
        if valid:
            best_w = max(valid, key=lambda r: r["best"]["weighted_silhouette"])
            best_t = max(pca_umap_rows, key=lambda r: r["trustworthiness"])
            b = best_w["best"]
            pca_c = best_w.get("pca_components", DEFAULT_PCA_COMPONENTS)
            lines.append(f"\n  PCA+UMAP  -> PCA {pca_c}D -> UMAP {best_w['pca_umap_dim']}D  "
                         f"(weighted_silhouette={b['weighted_silhouette']:.4f}, "
                         f"trust={best_w['trustworthiness']:.4f}, "
                         f"noise={b['noise_frac']:.1f}%)")
            lines.append(f"    reduction:\n"
                         f"      method: pca+umap\n"
                         f"      pca_components: {pca_c}\n"
                         f"      n_components: {best_w['pca_umap_dim']}")

    lines.append("")
    for line in lines:
        print(line)

    rec_path = os.path.join(output_dir, rec_filename)
    with open(rec_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved to {rec_path}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main(config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Unsupervised reduction analysis for unlabeled datasets.",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument("--config", required=True,
                            help="Path to YAML config (e.g. configs/eumoth.yaml)")
        parser.add_argument("--mode", default="all",
                            choices=["pca", "umap", "pca+umap", "all"],
                            help=(
                                "Which reducer(s) to test:\n"
                                "  pca      - PCA sweep via silhouette  -> pca_unlabeled.csv\n"
                                "  umap     - UMAP sweep via trustworthiness + weighted silhouette\n"
                                "             -> umap_unlabeled.csv\n"
                                "  pca+umap - PCA to --pca-components, then UMAP sweep\n"
                                "             -> pca_umap_unlabeled.csv\n"
                                "  all      - All paths + full-dim baseline (default)\n"
                                "             -> unlabeled_comparison.csv\n"
                            ))
        parser.add_argument("--dims", type=int, nargs="+", default=None,
                            help=(
                                "Target dimensions to test.\n"
                                f"  mode=pca:      default {DEFAULT_PCA_DIMS}\n"
                                f"  mode=umap:     default {DEFAULT_UMAP_DIMS}\n"
                                f"  mode=pca+umap: default {DEFAULT_PCA_UMAP_DIMS}\n"
                                f"  mode=all:      uses per-path defaults above"
                            ))
        parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS,
                            help=f"Intermediate PCA dimension for pca+umap mode "
                                 f"(default: {DEFAULT_PCA_COMPONENTS}).")
        parser.add_argument("--umap-neighbors", type=int, default=15,
                            help="UMAP n_neighbors (default: 15).")
        parser.add_argument("--umap-min-dist", type=float, default=0.1,
                            help="UMAP min_dist (default: 0.1).")
        parser.add_argument("--mcs-range", type=int, nargs="+",
                            default=DEFAULT_MCS_RANGE,
                            help="HDBSCAN min_cluster_size sweep range for UMAP/PCA+UMAP path "
                                 f"(default: {DEFAULT_MCS_RANGE}).")
        args = parser.parse_args()
        config_path = args.config
    else:
        class Args:
            mode = "all"; dims = None
            pca_components = DEFAULT_PCA_COMPONENTS
            umap_neighbors = 15; umap_min_dist = 0.1
            mcs_range = DEFAULT_MCS_RANGE
        args = Args()

    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    seed = config.get("seed", 42)

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    features_path = os.path.join(output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"\n  ERROR: {features_path} not found.")
        print("  Run `python cli/extract.py --config <config>` first.")
        sys.exit(1)

    # Load features — labels ignored even if present
    features, _, _, _ = load_features(features_path)
    N        = features.shape[0]
    full_dim = features.shape[1]

    n_clusters_est = config["dataset"].get("num_classes", 50)

    mode = args.mode
    csv_filename, rec_filename = MODE_OUTPUT[mode]

    run_pca      = mode in ("pca", "all")
    run_umap     = mode in ("umap", "all")
    run_pca_umap = mode in ("pca+umap", "all")
    run_full_dim = mode == "all"

    # Resolve dims — each path uses its own defaults when --dims is not given
    if args.dims:
        user_dims     = sorted(set(d for d in args.dims if d < full_dim))
        pca_dims      = user_dims if run_pca      else []
        umap_dims     = user_dims if run_umap     else []
        pca_umap_dims = user_dims if run_pca_umap else []
    else:
        pca_dims      = [d for d in DEFAULT_PCA_DIMS      if d < full_dim] if run_pca      else []
        umap_dims     = [d for d in DEFAULT_UMAP_DIMS     if d < full_dim] if run_umap     else []
        pca_umap_dims = [d for d in DEFAULT_PCA_UMAP_DIMS if d < full_dim] if run_pca_umap else []

    pca_components = args.pca_components

    # Banner
    header("DIMENSIONALITY REDUCTION ANALYSIS  [UNLABELED]")
    print(f"\n  Dataset        : {config['dataset']['name']}")
    print(f"  Model          : {config['model']['name']}")
    print(f"  Images         : {N}  |  Full dim: {full_dim}")
    print(f"  Mode           : {mode.upper()}")
    print(f"  Species est.   : {n_clusters_est}  (num_classes in config)")
    if run_full_dim: print(f"  Full-D baseline: yes ({full_dim}D)")
    if run_pca:      print(f"  PCA dims       : {pca_dims}")
    if run_umap:     print(f"  UMAP dims      : {umap_dims}  "
                           f"(nn={args.umap_neighbors}, md={args.umap_min_dist})")
    if run_pca_umap: print(f"  PCA+UMAP       : PCA {pca_components}D -> UMAP {pca_umap_dims}")
    if run_umap or run_pca_umap:
        print(f"  HDBSCAN MCS    : {args.mcs_range}")
    print(f"  Output         : {csv_filename}\n")

    full_dim_row = None
    pca_rows     = []
    umap_rows    = []
    pca_umap_rows = []
    all_rows     = []

    # ---- Full-dim baseline ----
    if run_full_dim:
        subheader(f"Full-D baseline  [{full_dim}D, no reduction]")
        full_dim_row = evaluate_full_dim(features, n_clusters_est, args.mcs_range, seed=seed)
        km_str = f"{full_dim_row['km_silhouette']:.4f}" if full_dim_row["km_silhouette"] else "--"
        b = full_dim_row.get("best") or {}
        ws_str = f"{b.get('weighted_silhouette', 0.0):.4f}" if b else "--"
        print(f"    km_silhouette={km_str}  hdbscan_weighted_sil={ws_str}")
        all_rows.append({**full_dim_row, "path": "full-dim"})

    # ---- PCA sweep ----
    if run_pca:
        subheader("PCA sweep  [silhouette]")
        for dim in pca_dims:
            reduced, variance = fit_pca(features, dim)
            result = evaluate_pca_dim(reduced, variance, n_clusters_est, seed=seed)
            row = {"path": "pca", "pca_dim": dim, **result}
            pca_rows.append(row)
            all_rows.append(row)
            sil_str = f"{result['silhouette']:.4f}" if result["silhouette"] else "--"
            print(f"    dim={dim:>4}  CEV={variance*100:.1f}%  silhouette={sil_str}")

        print_pca_table(pca_rows)

    # ---- UMAP sweep ----
    if run_umap:
        subheader("UMAP sweep  [trustworthiness + weighted silhouette]")
        for dim in umap_dims:
            reduced = fit_umap(features, dim,
                               n_neighbors=args.umap_neighbors,
                               min_dist=args.umap_min_dist,
                               random_state=seed)
            result = evaluate_umap_dim(features, reduced, args.mcs_range, seed=seed)
            b = result.get("best") or {}
            row = {
                "path":                "umap",
                "umap_dim":            dim,
                "trustworthiness":     result["trustworthiness"],
                "weighted_silhouette": b.get("weighted_silhouette"),
                "silhouette":          b.get("silhouette"),
                "noise_frac":          b.get("noise_frac"),
                "n_clusters":          b.get("n_clusters"),
                "best_mcs":            b.get("mcs"),
                "best":                b,
            }
            umap_rows.append(row)
            all_rows.append(row)

        print_umap_table(umap_rows, label="UMAP dim")

    # ---- PCA+UMAP sweep ----
    if run_pca_umap:
        subheader(f"PCA+UMAP sweep  [PCA {pca_components}D -> UMAP, "
                  f"trustworthiness + weighted silhouette]")
        for dim in pca_umap_dims:
            result = evaluate_pca_umap_dim(
                features, pca_components, dim, args.mcs_range,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric="cosine", seed=seed,
            )
            b = result.get("best") or {}
            row = {
                "path":                "pca+umap",
                "pca_umap_dim":        dim,
                "pca_components":      pca_components,
                "trustworthiness":     result["trustworthiness"],
                "weighted_silhouette": b.get("weighted_silhouette"),
                "silhouette":          b.get("silhouette"),
                "noise_frac":          b.get("noise_frac"),
                "n_clusters":          b.get("n_clusters"),
                "best_mcs":            b.get("mcs"),
                "best":                b,
            }
            pca_umap_rows.append(row)
            all_rows.append(row)

        print_umap_table(pca_umap_rows, label="PCA+UMAP dim")

    # ---- Recommendation ----
    header("RECOMMENDATION")
    print_recommendation(full_dim_row, pca_rows, umap_rows, pca_umap_rows,
                         n_clusters_est, output_dir, rec_filename)

    # ---- Save ----
    save_rows = [{k: v for k, v in r.items() if k != "best"} for r in all_rows]

    csv_path = os.path.join(output_dir, csv_filename)
    pd.DataFrame(save_rows).to_csv(csv_path, index=False)
    print(f"\n  [save] CSV  -> {csv_path}")

    json_path = csv_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "dataset":         config["dataset"]["name"],
            "model":           config["model"]["name"],
            "mode":            mode,
            "n_clusters_est":  n_clusters_est,
            "full_dim_row":    {k: v for k, v in full_dim_row.items() if k != "best"}
                               if full_dim_row else None,
            "pca_rows":        pca_rows,
            "umap_rows":       [{k: v for k, v in r.items() if k != "best"}
                                for r in umap_rows],
            "pca_umap_rows":   [{k: v for k, v in r.items() if k != "best"}
                                for r in pca_umap_rows],
        }, f, indent=2)
    print(f"  [save] JSON -> {json_path}")

    done(output_dir)


if __name__ == "__main__":
    main()
