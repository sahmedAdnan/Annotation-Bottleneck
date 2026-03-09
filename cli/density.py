"""
cli/density.py
--------------
CLI command: Run the full DBSCAN + HDBSCAN density clustering pipeline.

This is a first-class CLI command, consistent with extract.py, cluster.py,
evaluate.py, and diagnose.py. It can be run standalone or as part of the
unified runner (experiments/run_pipeline.py).

Usage:
    python cli/density.py --config configs/eumoth.yaml

    # Skip HDBSCAN (if not installed)
    python cli/density.py --config configs/eumoth.yaml --skip-hdbscan

    # Skip DBSCAN (run HDBSCAN only)
    python cli/density.py --config configs/eumoth.yaml --skip-dbscan

Requires:
    <output_dir>/features.npz       (produced by cli/extract.py)

Produces:
    <output_dir>/dbscan_metrics.json
    <output_dir>/dbscan_sweep.csv
    <output_dir>/hdbscan_metrics.json
    <output_dir>/hdbscan_sweep.csv
    <output_dir>/density_metrics_combined.json
    <output_dir>/density_metrics_summary.csv
    <output_dir>/figures/dbscan_kdistance_plot.png
    <output_dir>/figures/dbscan_eps_sweep.png
    <output_dir>/figures/dbscan_umap_scatter.png
    <output_dir>/figures/dbscan_cluster_sizes.png
    <output_dir>/figures/dbscan_top_confusions.png
    <output_dir>/figures/hdbscan_sweep.png
    <output_dir>/figures/hdbscan_umap_scatter.png
    <output_dir>/figures/hdbscan_outlier_scores.png
    <output_dir>/figures/density_method_comparison.png
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

# Allow imports from project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import set_seed, load_config, load_features, ensure_dir
from src.clustering.density.dbscan_pipeline  import DBSCANPipeline
from src.clustering.density.hdbscan_pipeline import (
    HDBSCANPipeline,
    plot_method_comparison,
)


# -------------------------------------------------------
# 2D projection: shared between density and evaluate CLI
# -------------------------------------------------------

def get_2d_projection(features: np.ndarray, config: dict,
                      output_dir: str) -> np.ndarray:
    """
    Load a cached 2D UMAP/t-SNE projection or compute it fresh.

    The projection is used ONLY for visualization. DBSCAN and HDBSCAN
    receive PCA-reduced features as their actual input — never the 2D
    projection. Using UMAP-2D as clustering input distorts the density
    structure those algorithms depend on.

    The result is cached to disk so that evaluate.py, density.py and any
    future visualisation step all share the same projection without
    recomputing it.
    """
    cache_path = os.path.join(output_dir, "features_2d.npy")

    if os.path.exists(cache_path):
        print(f"[density] Loading cached 2D projection from {cache_path}")
        return np.load(cache_path)

    vis_method = config["visualization"].get("method", "umap")

    if vis_method == "umap":
        try:
            import umap as umap_lib
        except ImportError:
            print("[density] umap-learn not installed — falling back to t-SNE.")
            vis_method = "tsne"

    if vis_method == "umap":
        vis_cfg = config["visualization"]["umap"]
        import umap as umap_lib
        reducer = umap_lib.UMAP(
            n_neighbors  = vis_cfg.get("n_neighbors", 30),
            n_components = 2,
            # Enforce min_dist >= 0.05 for visualization.
            # min_dist=0.0 compresses neighborhoods artificially and
            # should never be used for 2D scatter plots.
            min_dist     = max(vis_cfg.get("min_dist", 0.1), 0.05),
            metric       = vis_cfg.get("metric", "cosine"),
            random_state = vis_cfg.get("random_state", 42),
        )
        print("[density] Computing UMAP 2D projection ...")
        features_2d = reducer.fit_transform(features)

    else:
        from sklearn.manifold import TSNE
        vis_cfg = config["visualization"].get("tsne", {})
        reducer = TSNE(
            n_components  = 2,
            perplexity    = vis_cfg.get("perplexity", 40),
            random_state  = vis_cfg.get("random_state", 42),
            init          = "pca",
            learning_rate = "auto",
        )
        print("[density] Computing t-SNE 2D projection ...")
        features_2d = reducer.fit_transform(features)

    np.save(cache_path, features_2d)
    print(f"[density] 2D projection saved to {cache_path}")

    return features_2d


# -------------------------------------------------------
# Comparison table and metrics save
# -------------------------------------------------------

def print_comparison_table(dbscan_res: dict, hdbscan_res: dict) -> None:
    """
    Print a clean side-by-side comparison of DBSCAN and HDBSCAN results.

    The verdict uses coverage-weighted ARI (ARI * coverage fraction) rather
    than raw ARI. A result with ARI=1.0 on 3% of the data is not better than
    ARI=0.96 on 98% of the data. The weighted score captures this correctly.
    """
    print("\n" + "=" * 68)
    print("DENSITY CLUSTERING COMPARISON")
    print("=" * 68)
    print(f"  {'Metric':<28s} {'DBSCAN':>14s} {'HDBSCAN':>14s}")
    print("-" * 68)

    rows = [
        ("ARI  (excl. noise)",     "ari"),
        ("NMI  (excl. noise)",     "nmi"),
        ("Acc  (excl. noise)",     "accuracy"),
        ("Clusters found",         "n_clusters"),
        ("Noise points",           "n_noise"),
        ("Noise %",                "noise_pct"),
    ]

    for label, key in rows:
        d_val = dbscan_res.get(key, "—")
        h_val = hdbscan_res.get(key, "—")

        if isinstance(d_val, float):
            if key == "noise_pct":
                print(f"  {label:<28s} {d_val:>13.1%} {h_val:>13.1%}")
            else:
                print(f"  {label:<28s} {d_val:>14.4f} {h_val:>14.4f}")
        else:
            print(f"  {label:<28s} {str(d_val):>14s} {str(h_val):>14s}")

    print("=" * 68)

    d_coverage = 1.0 - dbscan_res.get("noise_pct", 1.0)
    h_coverage = 1.0 - hdbscan_res.get("noise_pct", 1.0)
    d_weighted = dbscan_res.get("ari", 0.0)  * d_coverage
    h_weighted = hdbscan_res.get("ari", 0.0) * h_coverage

    print("\n  Coverage-weighted ARI  (ARI x coverage):")
    print(f"    DBSCAN  : {d_weighted:.4f}  "
          f"(ARI={dbscan_res.get('ari', 0):.4f},  "
          f"coverage={d_coverage:.1%})")
    print(f"    HDBSCAN : {h_weighted:.4f}  "
          f"(ARI={hdbscan_res.get('ari', 0):.4f},  "
          f"coverage={h_coverage:.1%})")

    winner = "DBSCAN" if d_weighted >= h_weighted else "HDBSCAN"
    print(f"\n  Verdict: {winner} is more effective when accounting "
          f"for data coverage.")

    if dbscan_res.get("noise_pct", 0) > 0.30:
        print(
            "\n  WARNING: DBSCAN noise fraction > 30%.\n"
            "  ARI is computed on a small portion of the data.\n"
            "  Consider increasing final_eps in the config,\n"
            "  reducing min_samples, or reporting HDBSCAN as primary."
        )

    print("=" * 68 + "\n")


def save_combined_metrics(dbscan_res: dict, hdbscan_res: dict,
                           output_dir: str) -> None:
    """Save combined JSON and CSV for both density methods."""
    safe = {}
    for method, res in [("dbscan", dbscan_res), ("hdbscan", hdbscan_res)]:
        safe[method] = {
            k: v for k, v in res.items()
            if not isinstance(v, (np.ndarray, pd.DataFrame))
            and k not in ("outlier_scores",)
        }

    json_path = os.path.join(output_dir, "density_metrics_combined.json")
    with open(json_path, "w") as f:
        json.dump(safe, f, indent=2)
    print(f"[save] Combined metrics : {json_path}")

    rows = []
    for method, res in safe.items():
        coverage = 1.0 - res.get("noise_pct", 0.0)
        rows.append({
            "method":          method,
            "ari":             round(res.get("ari", 0), 4),
            "nmi":             round(res.get("nmi", 0), 4),
            "accuracy":        round(res.get("accuracy", 0), 4),
            "coverage":        round(coverage, 4),
            "weighted_score":  round(res.get("ari", 0) * coverage, 4),
            "n_clusters":      res.get("n_clusters", 0),
            "n_noise":         res.get("n_noise", 0),
            "noise_pct":       round(res.get("noise_pct", 0), 4),
            "pca_components":  res.get("pca_components", "—"),
        })

    csv_path = os.path.join(output_dir, "density_metrics_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[save] Summary CSV      : {csv_path}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main(config_path: str = None,
         skip_dbscan: bool = False,
         skip_hdbscan: bool = False) -> None:
    """
    Run the full density clustering pipeline.

    Can be called from the command line (via argparse) or imported
    and called directly by experiments/run_pipeline.py.

    Args:
        config_path:  Path to YAML config. If None, reads from sys.argv.
        skip_dbscan:  If True, skip DBSCAN.
        skip_hdbscan: If True, skip HDBSCAN.
    """
    # Allow both direct call and CLI invocation
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Run DBSCAN + HDBSCAN density clustering pipeline."
        )
        parser.add_argument("--config", required=True,
                            help="Path to YAML config file.")
        parser.add_argument("--skip-dbscan",  action="store_true",
                            help="Skip DBSCAN.")
        parser.add_argument("--skip-hdbscan", action="store_true",
                            help="Skip HDBSCAN (if package not installed).")
        args        = parser.parse_args()
        config_path  = args.config
        skip_dbscan  = args.skip_dbscan
        skip_hdbscan = args.skip_hdbscan

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    # Guard: features must exist
    features_path = os.path.join(output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"\n[density] ERROR: features.npz not found at {features_path}")
        print("[density] Run first:  python cli/extract.py --config <config>")
        sys.exit(1)

    features, labels, _, class_names = load_features(features_path)
    true_label_ids = np.array(labels, dtype=np.int32)

    print(f"\n[density] {features.shape[0]} images | "
          f"{len(class_names)} classes | "
          f"dim={features.shape[1]}")

    # 2D projection — visualization only, not used as clustering input
    features_2d = get_2d_projection(features, config, output_dir)

    dbscan_results  = None
    hdbscan_results = None

    # ---- DBSCAN ----
    if not skip_dbscan:
        pipeline       = DBSCANPipeline(config)
        dbscan_results = pipeline.run(
            features, true_label_ids, class_names, features_2d, output_dir
        )

    # ---- HDBSCAN ----
    if not skip_hdbscan:
        try:
            pipeline        = HDBSCANPipeline(config)
            hdbscan_results = pipeline.run(
                features, true_label_ids, class_names, features_2d, output_dir
            )
        except ImportError as e:
            print(f"\n[density] HDBSCAN skipped: {e}")
            print("[density] Install with:  conda install -c conda-forge hdbscan")

    # ---- Comparison ----
    if dbscan_results is not None and hdbscan_results is not None:
        print_comparison_table(dbscan_results, hdbscan_results)

        fig_dir    = os.path.join(output_dir, "figures")
        export_pdf = config.get("visualization", {}).get("export_pdf", False)
        plot_method_comparison(dbscan_results, hdbscan_results,
                               fig_dir, export_pdf)
        save_combined_metrics(dbscan_results, hdbscan_results, output_dir)

    elif dbscan_results is not None:
        save_combined_metrics(
            dbscan_results, {"method": "hdbscan_skipped"}, output_dir
        )
    elif hdbscan_results is not None:
        save_combined_metrics(
            {"method": "dbscan_skipped"}, hdbscan_results, output_dir
        )

    print(f"\n[density] Complete. "
          f"Outputs: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
