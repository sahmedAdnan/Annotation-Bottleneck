"""
cli/reduction_analysis.py
--------------------------
Find the optimal dimensionality reduction for your dataset.

Tests Full-D, PCA, UMAP, and PCA+UMAP on partition clustering methods
(KMeans, Agglomerative, Spectral, GMM) and recommends the best reduction
configuration for both partition and density-based clustering.

Why this matters:
    PCA is linear, fast, maximizes variance -- good for partition methods.
    UMAP is non-linear, preserves local topology -- good for density methods.
    PCA+UMAP: PCA removes noise first, then UMAP separates non-linearly.
    This tool measures which approach works best on YOUR data.

Requires:
    <output_dir>/features.npz  (produced by cli/extract.py)

Usage:
    # Full comparison (PCA + UMAP + PCA+UMAP)
    python cli/reduction_analysis.py --config configs/eumoth.yaml

    # Custom dimensions to test
    python cli/reduction_analysis.py --config configs/eumoth.yaml --dims 20 50 100 200

    # PCA only (fast, no UMAP)
    python cli/reduction_analysis.py --config configs/eumoth.yaml --skip-umap --skip-pca-umap

    # Include density methods (HDBSCAN) in the comparison
    python cli/reduction_analysis.py --config configs/eumoth.yaml --include-density

    # Skip Spectral clustering (slow for large datasets)
    python cli/reduction_analysis.py --config configs/eumoth.yaml --skip-spectral

Produces:
    <output_dir>/reduction_comparison.csv     (full results table)
    <output_dir>/reduction_comparison.json    (structured results)
    <output_dir>/reduction_recommendation.txt (human-readable summary)
"""

import argparse
import os
import sys
import time
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_config, load_features, set_seed, ensure_dir
from src.utils.pretty import header, subheader, metrics_table, done
from src.clustering.methods import run_clustering
from src.evaluation.metrics import evaluate


# -------------------------------------------------------
# Default settings
# -------------------------------------------------------

PARTITION_METHODS = ["kmeans", "agglomerative", "spectral", "gmm"]
DENSITY_METHODS = ["hdbscan"]

DEFAULT_DIMS = [50, 100, 200]
DEFAULT_PCA_PRE = 50
DEFAULT_PCA_UMAP_TARGETS = [10, 20, 30, 40]


# -------------------------------------------------------
# Reduction functions
# -------------------------------------------------------

def fit_pca(features, n_components):
    """Fit PCA, return (reduced, pca_object, cumulative_variance)."""
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    cum_var = float(pca.explained_variance_ratio_.sum())
    print(f"    [PCA]  {features.shape[1]}D -> {n_components}D  "
          f"(variance: {cum_var*100:.1f}%)")
    return reduced, pca, cum_var


def fit_umap(features, n_components, n_neighbors=30, min_dist=0.0,
             metric="cosine", random_state=42):
    """Fit UMAP for intermediate reduction (not 2D visualization)."""
    import umap as umap_lib

    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    print(f"    [UMAP] {features.shape[1]}D -> {n_components}D  "
          f"(nn={n_neighbors}, md={min_dist}) ...")
    t0 = time.time()
    reduced = reducer.fit_transform(features)
    print(f"           done in {time.time()-t0:.1f}s")
    return reduced


# -------------------------------------------------------
# Run one clustering method and evaluate
# -------------------------------------------------------

def run_single(method_name, features, params, true_labels, class_names):
    """Run one clustering method, return metrics dict."""
    cluster_ids = run_clustering(method_name, features, params)
    metrics = evaluate(true_labels, cluster_ids, class_names, top_k=0)
    return {
        "accuracy": metrics["accuracy"],
        "ari":      metrics["ari"],
        "nmi":      metrics["nmi"],
        "n_clusters_found": metrics["n_clusters_found"],
    }


def get_method_params(config, method_name):
    """Extract params for a method from config, with sensible defaults."""
    for entry in config.get("clustering", {}).get("methods", []):
        if entry["name"] == method_name:
            return {k: v for k, v in entry.items() if k != "name"}

    n_classes = config["dataset"]["num_classes"]
    defaults = {
        "kmeans":        {"n_clusters": n_classes, "n_init": 50, "random_state": 42},
        "agglomerative": {"n_clusters": n_classes, "linkage": "ward"},
        "spectral":      {"n_clusters": n_classes, "n_init": 10, "random_state": 42,
                          "n_neighbors": 15},
        "gmm":           {"n_clusters": n_classes, "covariance_type": "diag",
                          "random_state": 42},
        "hdbscan":       {"min_cluster_size": 5, "min_samples": 2},
    }
    return defaults.get(method_name, {"n_clusters": n_classes})


# -------------------------------------------------------
# Print formatted results
# -------------------------------------------------------

def print_results_table(title, methods, columns, df, metric_key):
    """Print: rows = methods, columns = reduction configs."""
    col_width = 14
    row_width = 16
    total_width = row_width + col_width * len(columns)

    print(f"\n  {title}:")
    print("  " + "-" * total_width)

    hdr = f"  {'Method':<{row_width}}"
    for col_label, _, _ in columns:
        hdr += f"{col_label:>{col_width}}"
    print(hdr)
    print("  " + "-" * total_width)

    for method in methods:
        row = f"  {method:<{row_width}}"
        for _, reducer, dim_lbl in columns:
            match = df[(df["method"] == method) &
                       (df["reducer"] == reducer) &
                       (df["dim_label"] == dim_lbl)]
            if len(match) == 1 and match.iloc[0][metric_key] is not None:
                val = match.iloc[0][metric_key]
                row += f"{val:>{col_width}.4f}"
            else:
                row += f"{'--':>{col_width}}"
        print(row)

    print("  " + "-" * total_width)


def print_recommendation(df, methods, output_dir):
    """Print and save the recommended reduction for each method family."""
    lines = []

    lines.append("\n  RECOMMENDATION")
    lines.append("  " + "=" * 60)

    # Best per method
    lines.append(f"\n  {'Method':<16} {'Reducer':>12} {'Dim':>6} "
                 f"{'Accuracy':>10} {'ARI':>10} {'NMI':>10}")
    lines.append("  " + "-" * 66)

    for method in methods:
        method_df = df[df["method"] == method].dropna(subset=["accuracy"])
        if len(method_df) == 0:
            continue
        best_idx = method_df["accuracy"].idxmax()
        best = method_df.loc[best_idx]
        reducer_str = best["reducer"] if best["reducer"] != "none" else "full"
        dim_str = f"{int(best['dimensions'])}D" if best["dim_label"] != "full" else "full"
        line = (f"  {method:<16} {reducer_str:>12} {dim_str:>6} "
                f"{best['accuracy']:>10.4f} {best['ari']:>10.4f} {best['nmi']:>10.4f}")
        lines.append(line)

    lines.append("  " + "-" * 66)

    # Overall recommendation
    partition_methods = [m for m in methods if m in PARTITION_METHODS]
    density_methods = [m for m in methods if m in DENSITY_METHODS]

    if partition_methods:
        part_df = df[df["method"].isin(partition_methods)].dropna(subset=["accuracy"])
        if len(part_df) > 0:
            best = part_df.loc[part_df["accuracy"].idxmax()]
            reducer = best["reducer"] if best["reducer"] != "none" else "full"
            dim = int(best["dimensions"]) if best["dim_label"] != "full" else best["dimensions"]
            lines.append(f"\n  Partition methods: Use {reducer.upper()}-{dim}D")

    if density_methods:
        dens_df = df[df["method"].isin(density_methods)].dropna(subset=["accuracy"])
        if len(dens_df) > 0:
            best = dens_df.loc[dens_df["accuracy"].idxmax()]
            reducer = best["reducer"] if best["reducer"] != "none" else "full"
            dim = int(best["dimensions"]) if best["dim_label"] != "full" else best["dimensions"]
            lines.append(f"  Density methods:   Use {reducer.upper()}-{dim}D")

    lines.append("")

    for line in lines:
        print(line)

    # Save to file
    rec_path = os.path.join(output_dir, "reduction_recommendation.txt")
    with open(rec_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved to {rec_path}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main(config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Find optimal dimensionality reduction for your dataset.",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument("--config", required=True,
                            help="Path to YAML config (e.g. configs/eumoth.yaml)")
        parser.add_argument("--dims", type=int, nargs="+", default=None,
                            help="Dimension counts to test.\nDefault: [50, 100, 200]")
        parser.add_argument("--skip-pca", action="store_true",
                            help="Skip PCA-only reduction.")
        parser.add_argument("--skip-umap", action="store_true",
                            help="Skip UMAP-only reduction.")
        parser.add_argument("--skip-pca-umap", action="store_true",
                            help="Skip PCA+UMAP combined reduction.")
        parser.add_argument("--skip-spectral", action="store_true",
                            help="Skip Spectral clustering (slow for N > 5000).")
        parser.add_argument("--include-density", action="store_true",
                            help="Also test HDBSCAN on each reduction.")
        parser.add_argument("--umap-neighbors", type=int, default=30,
                            help="UMAP n_neighbors (default: 30).")
        parser.add_argument("--umap-min-dist", type=float, default=0.0,
                            help="UMAP min_dist (default: 0.0 for clustering).")
        parser.add_argument("--pca-pre", type=int, default=None,
                            help="PCA pre-reduction dim for PCA+UMAP (default: 50).")
        parser.add_argument("--pca-umap-targets", type=int, nargs="+", default=None,
                            help="UMAP target dims for PCA+UMAP.\n"
                                 "Default: [10, 20, 30, 40].")
        args = parser.parse_args()
        config_path = args.config
    else:
        # Called programmatically with defaults
        class Args:
            dims = None; skip_pca = False; skip_umap = False
            skip_pca_umap = False; skip_spectral = False
            include_density = False; umap_neighbors = 30
            umap_min_dist = 0.0; pca_pre = None; pca_umap_targets = None
        args = Args()

    config = load_config(config_path)
    seed = config.get("seed", 42)
    set_seed(seed)

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    # Load features
    features_path = os.path.join(output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"\n  ERROR: {features_path} not found.")
        print("  Run `python cli/extract.py --config <config>` first.")
        sys.exit(1)

    features, labels, _, class_names = load_features(features_path)
    true_labels = np.array(labels, dtype=np.int32)
    full_dim = features.shape[1]
    N = features.shape[0]

    # Settings
    dims = args.dims or DEFAULT_DIMS
    dims = sorted(set(d for d in dims if d < full_dim))
    if not dims:
        dims = [full_dim // 2]

    methods = [m for m in PARTITION_METHODS]
    if args.skip_spectral:
        methods = [m for m in methods if m != "spectral"]
    elif N > 10000:
        print(f"  WARNING: N={N}. Spectral uses O(N^2) memory. "
              f"Use --skip-spectral if needed.\n")

    if args.include_density:
        methods.extend(DENSITY_METHODS)

    run_pca = not args.skip_pca
    run_umap = not args.skip_umap
    run_pca_umap = not args.skip_pca_umap

    if not run_pca and not run_umap and not run_pca_umap:
        print("ERROR: All reducers skipped. Nothing to compare.")
        sys.exit(1)

    umap_nn = args.umap_neighbors
    umap_md = args.umap_min_dist
    pca_pre_dim = args.pca_pre or DEFAULT_PCA_PRE
    pca_umap_targets = args.pca_umap_targets or DEFAULT_PCA_UMAP_TARGETS
    pca_umap_targets = [d for d in pca_umap_targets if d < pca_pre_dim]
    if run_pca_umap and not pca_umap_targets:
        pca_umap_targets = [pca_pre_dim // 2]

    # Banner
    header("DIMENSIONALITY REDUCTION ANALYSIS")
    print(f"\n  Dataset : {config['dataset']['name']}")
    print(f"  Model   : {config['model']['name']}")
    print(f"  Images  : {N}  |  Classes: {len(class_names)}  |  Full dim: {full_dim}")
    print(f"  Methods : {', '.join(methods)}")
    print(f"  Dims    : {dims}")
    reducers = []
    if run_pca:      reducers.append("PCA")
    if run_umap:     reducers.append(f"UMAP(nn={umap_nn})")
    if run_pca_umap: reducers.append(f"PCA({pca_pre_dim})+UMAP->{pca_umap_targets}")
    print(f"  Reducers: {', '.join(reducers)} + full-D baseline\n")

    # ---- Pre-compute reductions ----
    reductions = {}
    pca_variance = {}

    if run_pca:
        subheader("Computing PCA reductions")
        for dim in dims:
            reduced, _, cum_var = fit_pca(features, dim)
            reductions[("pca", dim)] = reduced
            pca_variance[dim] = cum_var

    if run_umap:
        subheader("Computing UMAP reductions")
        try:
            for dim in dims:
                reduced = fit_umap(features, dim, n_neighbors=umap_nn,
                                   min_dist=umap_md, random_state=seed)
                reductions[("umap", dim)] = reduced
        except ImportError:
            print("  ERROR: umap-learn not installed. Use --skip-umap.")
            run_umap = False

    if run_pca_umap:
        subheader(f"Computing PCA({pca_pre_dim})+UMAP reductions")
        try:
            if ("pca", pca_pre_dim) in reductions:
                pca_pre_features = reductions[("pca", pca_pre_dim)]
                print(f"    Reusing PCA-{pca_pre_dim}D from earlier")
            else:
                pca_pre_features, _, _ = fit_pca(features, pca_pre_dim)

            for target_dim in pca_umap_targets:
                reduced = fit_umap(pca_pre_features, target_dim,
                                   n_neighbors=umap_nn, min_dist=umap_md,
                                   metric="euclidean", random_state=seed)
                reductions[(f"pca{pca_pre_dim}+umap", target_dim)] = reduced
        except ImportError:
            print("  ERROR: umap-learn not installed. Use --skip-pca-umap.")
            run_pca_umap = False

    # ---- Build experiment list ----
    experiments = [{"reducer": "none", "dim_label": "full",
                    "dim": full_dim, "features": features}]

    if run_pca:
        for dim in dims:
            experiments.append({"reducer": "pca", "dim_label": str(dim),
                                "dim": dim, "features": reductions[("pca", dim)]})

    if run_umap:
        for dim in dims:
            experiments.append({"reducer": "umap", "dim_label": str(dim),
                                "dim": dim, "features": reductions[("umap", dim)]})

    if run_pca_umap:
        for target_dim in pca_umap_targets:
            label = f"pca{pca_pre_dim}+umap"
            experiments.append({"reducer": label, "dim_label": str(target_dim),
                                "dim": target_dim,
                                "features": reductions[(label, target_dim)]})

    # ---- Run all experiments ----
    subheader("Running clustering experiments")
    results = []
    t_start = time.time()

    for exp in experiments:
        reducer_info = (f"{exp['reducer']}-{exp['dim_label']}D"
                        if exp['reducer'] != 'none' else f"full-{exp['dim']}D")

        for method_name in methods:
            params = get_method_params(config, method_name)
            print(f"    {method_name:<16s} @ {reducer_info}")
            t0 = time.time()

            try:
                metrics = run_single(method_name, exp["features"], params,
                                     true_labels, class_names)
                elapsed = time.time() - t0
                results.append({
                    "method":     method_name,
                    "reducer":    exp["reducer"],
                    "dimensions": exp["dim"],
                    "dim_label":  exp["dim_label"],
                    "accuracy":   round(metrics["accuracy"], 4),
                    "ari":        round(metrics["ari"], 4),
                    "nmi":        round(metrics["nmi"], 4),
                    "n_clusters": metrics["n_clusters_found"],
                    "time_sec":   round(elapsed, 1),
                })
            except Exception as e:
                print(f"      ERROR: {e}")
                results.append({
                    "method": method_name, "reducer": exp["reducer"],
                    "dimensions": exp["dim"], "dim_label": exp["dim_label"],
                    "accuracy": None, "ari": None, "nmi": None,
                    "n_clusters": None, "time_sec": None,
                })

    total_time = time.time() - t_start
    df = pd.DataFrame(results)

    # ---- Print results ----
    header("RESULTS")
    print(f"\n  Dataset: {config['dataset']['name']}  |  "
          f"Model: {config['model']['name']}  |  "
          f"Time: {int(total_time//60)}m {total_time%60:.0f}s\n")

    # PCA variance summary
    if run_pca:
        print("  PCA Explained Variance:")
        for dim in dims:
            print(f"    {dim:>4}D: {pca_variance[dim]*100:.1f}%")
        print()

    # Build column list
    columns = [("full", "none", "full")]
    if run_pca:
        for dim in dims:
            columns.append((f"PCA-{dim}", "pca", str(dim)))
    if run_umap:
        for dim in dims:
            columns.append((f"UMAP-{dim}", "umap", str(dim)))
    if run_pca_umap:
        for td in pca_umap_targets:
            columns.append((f"P{pca_pre_dim}U{td}", f"pca{pca_pre_dim}+umap", str(td)))

    print_results_table("Hungarian Accuracy", methods, columns, df, "accuracy")
    print_results_table("Adjusted Rand Index", methods, columns, df, "ari")
    print_results_table("Normalized Mutual Information", methods, columns, df, "nmi")

    # Recommendation
    print_recommendation(df, methods, output_dir)

    # ---- Save outputs ----
    csv_path = os.path.join(output_dir, "reduction_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  [save] Results CSV  -> {csv_path}")

    json_data = {
        "dataset":     config["dataset"]["name"],
        "model":       config["model"]["name"],
        "full_dim":    full_dim,
        "n_images":    N,
        "n_classes":   len(class_names),
        "dims_tested": dims,
        "results":     results,
    }
    json_path = os.path.join(output_dir, "reduction_comparison.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  [save] Results JSON -> {json_path}")

    done(output_dir)


if __name__ == "__main__":
    main()
