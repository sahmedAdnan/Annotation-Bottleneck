"""
cli/reduction_analysis.py
--------------------------
Find the optimal dimensionality reduction for your dataset.

Run a single reducer or all at once. Output files are named per mode
so results from different runs never overwrite each other.

Why this matters:
    PCA is linear, fast, maximizes variance -- good for partition methods.
    UMAP is non-linear, preserves local topology -- good for density methods.
    PCA+UMAP: PCA removes noise first, then UMAP separates non-linearly.

Requires:
    <output_dir>/features.npz  (produced by cli/extract.py)

Usage:
    # PCA only -- custom or default dims, saves pca_analysis.csv
    python cli/reduction_analysis.py --config configs/eumoth.yaml --mode pca
    python cli/reduction_analysis.py --config configs/eumoth.yaml --mode pca --dims 20 50 100 200

    # UMAP only -- saves umap_analysis.csv
    python cli/reduction_analysis.py --config configs/eumoth.yaml --mode umap
    python cli/reduction_analysis.py --config configs/eumoth.yaml --mode umap --dims 5 10 20 50 100

    # PCA+UMAP two-stage -- saves pca_umap_analysis.csv
    python cli/reduction_analysis.py --config configs/eumoth.yaml --mode pca+umap

    # All reducers + full-dim baseline -- saves reduction_comparison.csv (default)
    python cli/reduction_analysis.py --config configs/eumoth.yaml

    # Include HDBSCAN in any mode
    python cli/reduction_analysis.py --config configs/eumoth.yaml --mode umap --include-density

Output files per mode:
    pca_analysis.csv / pca_recommendation.txt
    umap_analysis.csv / umap_recommendation.txt
    pca_umap_analysis.csv / pca_umap_recommendation.txt
    reduction_comparison.csv / reduction_recommendation.txt  (mode=all / default)
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
from src.utils.pretty import header, subheader, done
from src.clustering.methods import run_clustering
from src.evaluation.metrics import evaluate


# -------------------------------------------------------
# Defaults
# -------------------------------------------------------

PARTITION_METHODS = ["kmeans", "agglomerative", "spectral", "gmm"]
DENSITY_METHODS   = ["hdbscan"]

DEFAULT_PCA_DIMS     = [20, 50, 75, 100, 150, 200]
DEFAULT_UMAP_DIMS    = [5, 10, 20, 50, 100]
DEFAULT_ALL_DIMS     = [50, 100, 200]
DEFAULT_PCA_PRE      = 50
DEFAULT_UMAP_TARGETS = [10, 20, 30, 40]

MODE_OUTPUT = {
    "pca":      ("pca_analysis.csv",       "pca_recommendation.txt"),
    "umap":     ("umap_analysis.csv",      "umap_recommendation.txt"),
    "pca+umap": ("pca_umap_analysis.csv",  "pca_umap_recommendation.txt"),
    "all":      ("reduction_comparison.csv", "reduction_recommendation.txt"),
}


# -------------------------------------------------------
# Reduction helpers
# -------------------------------------------------------

def fit_pca(features, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    cum_var = float(pca.explained_variance_ratio_.sum())
    print(f"    [PCA]  {features.shape[1]}D -> {n_components}D  "
          f"(variance: {cum_var*100:.1f}%)")
    return reduced, pca, cum_var


def fit_umap(features, n_components, n_neighbors=30, min_dist=0.1,
             metric="cosine", random_state=42):
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
# Clustering helpers
# -------------------------------------------------------

def run_single(method_name, features, params, true_labels, class_names):
    cluster_ids = run_clustering(method_name, features, params)
    metrics = evaluate(true_labels, cluster_ids, class_names, top_k=0)
    return {
        "accuracy":         metrics["accuracy"],
        "ari":              metrics["ari"],
        "nmi":              metrics["nmi"],
        "n_clusters_found": metrics["n_clusters_found"],
    }


def get_method_params(config, method_name):
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
# Print helpers
# -------------------------------------------------------

def print_results_table(title, methods, columns, df, metric_key):
    col_width  = 14
    row_width  = 16
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
                row += f"{match.iloc[0][metric_key]:>{col_width}.4f}"
            else:
                row += f"{'--':>{col_width}}"
        print(row)
    print("  " + "-" * total_width)


def print_recommendation(df, methods, output_dir, rec_filename):
    lines = []
    lines.append("\n  RECOMMENDATION")
    lines.append("  " + "=" * 60)
    lines.append(f"\n  {'Method':<16} {'Reducer':>12} {'Dim':>6} "
                 f"{'Accuracy':>10} {'ARI':>10} {'NMI':>10}")
    lines.append("  " + "-" * 66)

    for method in methods:
        sub = df[df["method"] == method].dropna(subset=["accuracy"])
        if sub.empty:
            continue
        best = sub.loc[sub["accuracy"].idxmax()]
        reducer_str = best["reducer"] if best["reducer"] != "none" else "full"
        dim_str = f"{int(best['dimensions'])}D" if best["dim_label"] != "full" else "full"
        lines.append(f"  {method:<16} {reducer_str:>12} {dim_str:>6} "
                     f"{best['accuracy']:>10.4f} {best['ari']:>10.4f} {best['nmi']:>10.4f}")

    lines.append("  " + "-" * 66)

    part_df = df[df["method"].isin(PARTITION_METHODS)].dropna(subset=["accuracy"])
    dens_df = df[df["method"].isin(DENSITY_METHODS)].dropna(subset=["accuracy"])

    if not part_df.empty:
        best = part_df.loc[part_df["accuracy"].idxmax()]
        reducer = best["reducer"] if best["reducer"] != "none" else "full"
        dim = int(best["dimensions"]) if best["dim_label"] != "full" else "full"
        lines.append(f"\n  Partition methods: Use {reducer.upper()}-{dim}D")

    if not dens_df.empty:
        best = dens_df.loc[dens_df["accuracy"].idxmax()]
        reducer = best["reducer"] if best["reducer"] != "none" else "full"
        dim = int(best["dimensions"]) if best["dim_label"] != "full" else "full"
        lines.append(f"  Density methods:   Use {reducer.upper()}-{dim}D")

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
            description="Find optimal dimensionality reduction for your dataset.",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument("--config", required=True,
                            help="Path to YAML config (e.g. configs/eumoth.yaml)")
        parser.add_argument("--mode", default="all",
                            choices=["pca", "umap", "pca+umap", "all"],
                            help=(
                                "Which reducer(s) to test:\n"
                                "  pca      - PCA only   -> pca_analysis.csv\n"
                                "  umap     - UMAP only  -> umap_analysis.csv\n"
                                "  pca+umap - Two-stage  -> pca_umap_analysis.csv\n"
                                "  all      - All + full-dim baseline (default)\n"
                            ))
        parser.add_argument("--dims", type=int, nargs="+", default=None,
                            help=(
                                "Target dimensions to test.\n"
                                "  mode=pca:      default [20, 50, 75, 100, 150, 200]\n"
                                "  mode=umap:     default [5, 10, 20, 50, 100]\n"
                                "  mode=pca+umap: UMAP target dims, default [10, 20, 30, 40]\n"
                                "  mode=all:      applies to both PCA and UMAP, default [50, 100, 200]"
                            ))
        parser.add_argument("--pca-pre", type=int, default=None,
                            help="PCA pre-reduction dim for pca+umap mode (default: 50).")
        parser.add_argument("--umap-neighbors", type=int, default=30,
                            help="UMAP n_neighbors (default: 30).")
        parser.add_argument("--umap-min-dist", type=float, default=0.1,
                            help="UMAP min_dist (default: 0.1, matches production pipeline).")
        parser.add_argument("--skip-spectral", action="store_true",
                            help="Skip Spectral clustering (slow for N > 5000).")
        parser.add_argument("--include-density", action="store_true",
                            help="Also test HDBSCAN alongside partition methods.")
        args = parser.parse_args()
        config_path = args.config
    else:
        class Args:
            mode = "all"; dims = None; pca_pre = None
            skip_spectral = False; include_density = False
            umap_neighbors = 30; umap_min_dist = 0.1
        args = Args()

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    features_path = os.path.join(output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"\n  ERROR: {features_path} not found.")
        print("  Run `python cli/extract.py --config <config>` first.")
        sys.exit(1)

    features, labels, _, class_names = load_features(features_path)
    true_labels = np.array(labels, dtype=np.int32)
    full_dim    = features.shape[1]
    N           = features.shape[0]

    mode = args.mode
    csv_filename, rec_filename = MODE_OUTPUT[mode]

    run_pca      = mode in ("pca", "all")
    run_umap     = mode in ("umap", "all")
    run_pca_umap = mode in ("pca+umap", "all")
    include_full = (mode == "all")

    pca_pre_dim = args.pca_pre or DEFAULT_PCA_PRE
    umap_nn     = args.umap_neighbors
    umap_md     = args.umap_min_dist

    # Resolve dims per mode
    if args.dims:
        user_dims    = sorted(set(d for d in args.dims if d < full_dim))
        pca_dims     = user_dims if run_pca else []
        umap_dims    = user_dims if run_umap else []
        umap_targets = sorted(d for d in user_dims if d < pca_pre_dim) if run_pca_umap else []
    else:
        pca_dims     = [d for d in DEFAULT_PCA_DIMS  if d < full_dim] if run_pca else []
        umap_dims    = [d for d in DEFAULT_UMAP_DIMS if d < full_dim] if run_umap else []
        umap_targets = list(DEFAULT_UMAP_TARGETS)                       if run_pca_umap else []
        if mode == "all":
            pca_dims  = [d for d in DEFAULT_ALL_DIMS if d < full_dim]
            umap_dims = [d for d in DEFAULT_ALL_DIMS if d < full_dim]

    umap_targets = [d for d in umap_targets if d < pca_pre_dim]
    if run_pca_umap and not umap_targets:
        umap_targets = [pca_pre_dim // 2]

    # Methods
    methods = list(PARTITION_METHODS)
    if args.skip_spectral:
        methods = [m for m in methods if m != "spectral"]
    elif N > 10000:
        print(f"  WARNING: N={N}. Spectral uses O(N²) memory. "
              f"Consider --skip-spectral.\n")
    if args.include_density:
        methods.extend(DENSITY_METHODS)

    # Banner
    header("DIMENSIONALITY REDUCTION ANALYSIS")
    print(f"\n  Dataset : {config['dataset']['name']}")
    print(f"  Model   : {config['model']['name']}")
    print(f"  Images  : {N}  |  Classes: {len(class_names)}  |  Full dim: {full_dim}")
    print(f"  Mode    : {mode.upper()}")
    print(f"  Methods : {', '.join(methods)}")
    if run_pca:      print(f"  PCA dims      : {pca_dims}")
    if run_umap:     print(f"  UMAP dims     : {umap_dims}  (nn={umap_nn}, md={umap_md})")
    if run_pca_umap: print(f"  PCA({pca_pre_dim})+UMAP  : {umap_targets}")
    print(f"  Output  : {csv_filename}\n")

    # ---- Compute reductions ----
    reductions   = {}
    pca_variance = {}

    if run_pca:
        subheader("PCA reductions")
        for dim in pca_dims:
            reduced, _, cum_var = fit_pca(features, dim)
            reductions[("pca", dim)] = reduced
            pca_variance[dim] = cum_var

    if run_umap:
        subheader("UMAP reductions")
        try:
            for dim in umap_dims:
                reduced = fit_umap(features, dim, n_neighbors=umap_nn,
                                   min_dist=umap_md, random_state=config.get("seed", 42))
                reductions[("umap", dim)] = reduced
        except ImportError:
            print("  ERROR: umap-learn not installed. pip install umap-learn")
            run_umap = False

    if run_pca_umap:
        subheader(f"PCA({pca_pre_dim})+UMAP reductions")
        try:
            if ("pca", pca_pre_dim) in reductions:
                pca_pre_features = reductions[("pca", pca_pre_dim)]
                print(f"    Reusing PCA-{pca_pre_dim}D")
            else:
                pca_pre_features, _, _ = fit_pca(features, pca_pre_dim)
            for td in umap_targets:
                reduced = fit_umap(pca_pre_features, td, n_neighbors=umap_nn,
                                   min_dist=umap_md, metric="euclidean",
                                   random_state=config.get("seed", 42))
                reductions[(f"pca{pca_pre_dim}+umap", td)] = reduced
        except ImportError:
            print("  ERROR: umap-learn not installed. pip install umap-learn")
            run_pca_umap = False

    # ---- Build experiment list ----
    experiments = []
    if include_full:
        experiments.append({"reducer": "none", "dim_label": "full",
                            "dim": full_dim, "features": features})
    if run_pca:
        for dim in pca_dims:
            experiments.append({"reducer": "pca", "dim_label": str(dim),
                                "dim": dim, "features": reductions[("pca", dim)]})
    if run_umap:
        for dim in umap_dims:
            experiments.append({"reducer": "umap", "dim_label": str(dim),
                                "dim": dim, "features": reductions[("umap", dim)]})
    if run_pca_umap:
        for td in umap_targets:
            label = f"pca{pca_pre_dim}+umap"
            experiments.append({"reducer": label, "dim_label": str(td),
                                "dim": td, "features": reductions[(label, td)]})

    # ---- Run clustering ----
    subheader("Clustering experiments")
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
                results.append({
                    "method":     method_name,
                    "reducer":    exp["reducer"],
                    "dimensions": exp["dim"],
                    "dim_label":  exp["dim_label"],
                    "accuracy":   round(metrics["accuracy"], 4),
                    "ari":        round(metrics["ari"], 4),
                    "nmi":        round(metrics["nmi"], 4),
                    "n_clusters": metrics["n_clusters_found"],
                    "time_sec":   round(time.time() - t0, 1),
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
          f"Mode: {mode.upper()}  |  "
          f"Time: {int(total_time//60)}m {total_time%60:.0f}s\n")

    if pca_variance:
        print("  PCA Explained Variance:")
        for dim, var in pca_variance.items():
            print(f"    {dim:>4}D: {var*100:.1f}%")
        print()

    columns = []
    if include_full:
        columns.append(("full", "none", "full"))
    if run_pca:
        for dim in pca_dims:
            columns.append((f"PCA-{dim}", "pca", str(dim)))
    if run_umap:
        for dim in umap_dims:
            columns.append((f"UMAP-{dim}", "umap", str(dim)))
    if run_pca_umap:
        for td in umap_targets:
            columns.append((f"P{pca_pre_dim}U{td}", f"pca{pca_pre_dim}+umap", str(td)))

    print_results_table("Hungarian Accuracy",             methods, columns, df, "accuracy")
    print_results_table("Adjusted Rand Index",            methods, columns, df, "ari")
    print_results_table("Normalized Mutual Information",  methods, columns, df, "nmi")

    print_recommendation(df, methods, output_dir, rec_filename)

    # ---- Save ----
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"\n  [save] CSV  -> {csv_path}")

    json_path = csv_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "dataset":   config["dataset"]["name"],
            "model":     config["model"]["name"],
            "mode":      mode,
            "full_dim":  full_dim,
            "n_images":  N,
            "n_classes": len(class_names),
            "pca_dims":  pca_dims,
            "umap_dims": umap_dims,
            "results":   results,
        }, f, indent=2)
    print(f"  [save] JSON -> {json_path}")

    done(output_dir)


if __name__ == "__main__":
    main()
