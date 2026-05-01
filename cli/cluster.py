"""
cli/cluster.py
--------------
CLI command: Run ALL clustering methods defined in config.

Dual-path dimensionality reduction (empirically validated):

    Partition methods (kmeans, spectral, agglomerative, gmm)
        -> PCA-50D  (variance-maximization aligns with partition objectives)

    Density methods (dbscan, hdbscan)
        -> UMAP-100D  (topology-preservation aligns with density objectives)

Routing logic:
  - kmeans, spectral, agglomerative, gmm
        -> PCA-50D features via src/clustering/methods.py
  - dbscan
        -> UMAP-100D features via DBSCANPipeline
  - hdbscan
        -> UMAP-100D features via HDBSCANPipeline

Usage:
    python cli/cluster.py --config configs/eumoth.yaml

Requires:
    <output_dir>/features.npz    (produced by cli/extract.py)

Output:
    <output_dir>/clusters_<method>.npz        for each method
    <output_dir>/dbscan_metrics.json          if dbscan in config
    <output_dir>/hdbscan_metrics.json         if hdbscan in config
    <output_dir>/density_metrics_combined.json
    <output_dir>/figures/*                    density diagnostic plots
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.utils.io      import (set_seed, load_config, load_features,
                                save_clusters, ensure_dir, get_2d_projection)
from src.utils.pretty  import (header, subheader, metrics_table,
                                comparison_table, verdict, warning,
                                done, saved)
from src.clustering.methods import run_clustering
from src.reduction import reduce_for_partition, reduce_for_density, get_reduction_config


# -------------------------------------------------------
# Method classification
# -------------------------------------------------------

DENSITY_METHODS  = {"dbscan", "hdbscan"}
PARTITION_METHODS = {"kmeans", "spectral", "agglomerative", "gmm"}


# -------------------------------------------------------
# Run a density method with full diagnostics
# -------------------------------------------------------

def _run_density(method_name, method_cfg, features, true_label_ids,
                 class_names, features_2d, output_dir, config,
                 features_reduced=None, reduction_label=None):
    """
    Route a density method to its full diagnostic pipeline.

    Args:
        features_reduced: Pre-reduced features (UMAP-100D). If provided,
                          the pipeline skips its internal PCA.
        reduction_label:  Label for the reduction (e.g. "UMAP-100D").
    """
    merged_config = dict(config)
    if method_name == "dbscan":
        base = {k: v for k, v in method_cfg.items() if k != "name"}
        base.update(config.get("dbscan", {}))
        merged_config["dbscan"] = base
        from src.clustering.density.dbscan_pipeline import DBSCANPipeline
        pipeline = DBSCANPipeline(merged_config)

    elif method_name == "hdbscan":
        base = {k: v for k, v in method_cfg.items() if k != "name"}
        base.update(config.get("hdbscan", {}))
        merged_config["hdbscan"] = base
        from src.clustering.density.hdbscan_pipeline import HDBSCANPipeline
        pipeline = HDBSCANPipeline(merged_config)

    results = pipeline.run(
        features, true_label_ids, class_names, features_2d, output_dir,
        features_reduced=features_reduced,
        reduction_label=reduction_label,
    )

    cluster_path = os.path.join(output_dir, f"clusters_{method_name}.npz")
    save_clusters(cluster_path, results["labels"], method_name, method_cfg)

    # Save HDBSCAN membership probabilities for annotation pipeline
    if method_name == "hdbscan" and results.get("probabilities") is not None:
        probs_path = os.path.join(output_dir, "hdbscan_probabilities.npy")
        np.save(probs_path, results["probabilities"])
        print(f"    [save] HDBSCAN probabilities -> {probs_path}")

    return results


# -------------------------------------------------------
# Final summary table across all methods
# -------------------------------------------------------

def _print_summary(all_results: dict) -> None:
    """Print a unified comparison table after all methods complete."""
    if len(all_results) < 2:
        return

    methods = list(all_results.keys())

    def _get(res, key, default=0.0):
        return res.get(key, default)

    metrics = {
        "ARI":          {m: _get(all_results[m], "ari")      for m in methods},
        "NMI":          {m: _get(all_results[m], "nmi")      for m in methods},
        "Accuracy":     {m: _get(all_results[m], "accuracy") for m in methods},
        "Clusters":     {m: _get(all_results[m], "n_clusters", 0) for m in methods},
        "Noise %":      {m: f"{_get(all_results[m], 'noise_pct', 0.0):.1%}"
                         for m in methods},
    }

    comparison_table(methods, metrics, title="ALL METHODS — FINAL COMPARISON")

    # Identify best method by ARI (density methods use weighted ARI)
    def effective_ari(name):
        r = all_results[name]
        ari = r.get("ari", 0.0)
        if name in DENSITY_METHODS:
            return ari * (1.0 - r.get("noise_pct", 0.0))
        return ari

    best = max(methods, key=effective_ari)
    verdict(best, f"highest ARI={effective_ari(best):.4f}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main(config_path: str = None,
         skip_density: bool = False,
         skip_dbscan: bool = False,
         skip_hdbscan: bool = False) -> None:
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Run all clustering methods (partition + density)."
        )
        parser.add_argument("--config", required=True,
                            help="Path to YAML config file.")
        args        = parser.parse_args()
        config_path = args.config

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    output_dir = config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "figures"))

    # Load features
    features_path = os.path.join(output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"\n  ERROR: features.npz not found at {features_path}")
        print("  Run first:  python cli/extract.py --config <config>")
        sys.exit(1)

    features, labels, image_paths, class_names = load_features(features_path)
    true_label_ids = np.array(labels, dtype=np.int32)

    method_configs = config["clustering"]["methods"]
    method_names   = [m["name"] for m in method_configs]
    has_density    = any(n in DENSITY_METHODS for n in method_names)
    has_partition  = any(n in PARTITION_METHODS for n in method_names)

    header(f"CLUSTERING  —  {len(method_configs)} method(s)")
    print(f"\n  Dataset : {features.shape[0]} images | "
          f"{len(class_names)} classes | dim={features.shape[1]}")
    print(f"  Methods : {', '.join(method_names)}")

    # ---- Reduction config ----
    rcfg = get_reduction_config(config)
    print(f"\n  Reduction (partition) : {rcfg['partition']['method'].upper()}-"
          f"{rcfg['partition']['n_components']}D")
    print(f"  Reduction (density)  : {rcfg['density']['method'].upper()}-"
          f"{rcfg['density']['n_components']}D")

    # ---- PCA reduction for partition methods ----
    features_pca = None
    pca_object   = None
    if has_partition:
        subheader("Dimensionality reduction for partition methods")
        features_pca, pca_object, pca_var = reduce_for_partition(
            features, config, random_state=config.get("seed", 42)
        )

    # ---- UMAP reduction for density methods ----
    features_umap = None
    density_reduction_label = None
    if has_density:
        subheader("Dimensionality reduction for density methods")
        d_cfg = rcfg["density"]
        density_reduction_label = (f"{d_cfg['method'].upper()}-"
                                   f"{d_cfg['n_components']}D")
        features_umap, _ = reduce_for_density(
            features, config, random_state=config.get("seed", 42)
        )

    # ---- Pre-compute 2D projection for density visualization ----
    features_2d = None
    if has_density:
        subheader("2D projection for density visualization")
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                message="n_jobs value.*overridden",
                category=UserWarning)
            features_2d = get_2d_projection(features, config, output_dir)

    # ---- Run each method ----
    all_results  = {}
    total        = len(method_configs)

    for i, method_cfg in enumerate(method_configs, 1):
        name = method_cfg["name"]
        subheader(f"Method {i}/{total}:  {name.upper()}")
        t0 = time.time()

        if name in DENSITY_METHODS:
            print(f"    Reduction: {density_reduction_label or 'UMAP-100D'}")
            # Check skip flags
            if skip_density:
                print("    [skip] density methods (--skip-density)")
                continue
            if name == "dbscan" and skip_dbscan:
                print("    [skip] dbscan (--skip-dbscan)")
                continue
            if name == "hdbscan" and skip_hdbscan:
                print("    [skip] hdbscan (--skip-hdbscan)")
                continue

            # Full diagnostic pipeline with UMAP-reduced features
            results = _run_density(
                name, method_cfg,
                features, true_label_ids, class_names,
                features_2d, output_dir, config,
                features_reduced=features_umap,
                reduction_label=density_reduction_label,
            )
            all_results[name] = results

        else:
            # Partition methods: use PCA-reduced features
            reduction_info = (f"PCA-{features_pca.shape[1]}D"
                              if features_pca is not None else "Full-D")
            print(f"    Reduction: {reduction_info}")
            use_features = features_pca if features_pca is not None else features

            cluster_ids = run_clustering(
                method_name=name, features=use_features,
                params=method_cfg,
            )
            save_clusters(
                os.path.join(output_dir, f"clusters_{name}.npz"),
                cluster_ids, name, method_cfg,
            )

            # Quick inline metric preview
            from src.evaluation.metrics import hungarian_accuracy
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            valid = cluster_ids >= 0
            if valid.sum() > 0:
                ari = adjusted_rand_score(true_label_ids[valid], cluster_ids[valid])
                nmi = normalized_mutual_info_score(true_label_ids[valid], cluster_ids[valid])
                acc, _ = hungarian_accuracy(true_label_ids[valid], cluster_ids[valid])
            else:
                ari = nmi = acc = 0.0

            all_results[name] = {
                "ari": ari, "nmi": nmi, "accuracy": acc,
                "n_clusters": len(set(cluster_ids[cluster_ids >= 0])),
                "noise_pct": (cluster_ids < 0).mean(),
            }

            elapsed = time.time() - t0
            reduction_info = (f"PCA-{features_pca.shape[1]}D"
                              if features_pca is not None else "Full-D")
            metrics_table([
                {"label": "ARI",           "value": ari},
                {"label": "NMI",           "value": nmi},
                {"label": "Accuracy",      "value": acc},
                {"label": "Clusters",      "value": all_results[name]["n_clusters"],
                 "note": f"/ {len(class_names)} expected"},
                {"label": "Reduction",     "value": reduction_info},
                {"label": "Time",          "value": f"{elapsed:.1f}s"},
            ], title=f"{name} — quick metrics")

    # Density comparison figure (if both density methods ran)
    d_ran = {n: all_results[n] for n in DENSITY_METHODS if n in all_results}
    if len(d_ran) == 2:
        from src.clustering.density.hdbscan_pipeline import plot_method_comparison
        export_pdf = config.get("visualization", {}).get("export_pdf", False)
        ds         = config.get("dataset", {})
        ds_name = ds.get('name', '')
        ds_classes = ds.get('num_classes', '')
        ds_label = f"{ds_name}, {ds_classes} Species" if ds_name else f"{ds_classes} Species"
        plot_method_comparison(
            d_ran["dbscan"], d_ran["hdbscan"],
            os.path.join(output_dir, "figures"), export_pdf,
            dataset_label=ds_label,
            n_true_classes=len(class_names),
        )

    # ---- Final comparison ----
    _print_summary(all_results)
    done(output_dir)


if __name__ == "__main__":
    main()
