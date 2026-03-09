"""
cli/cluster.py
--------------
CLI command: Run ALL clustering methods defined in config.

This includes both partition-based methods (KMeans, Spectral, Agglomerative)
and density-based methods (DBSCAN, HDBSCAN). All clustering algorithms belong
here — they solve the same problem and should be configured and run from one
place.

Routing logic:
  - kmeans, kmeans_text_guided, spectral, agglomerative, gmm
        -> standard fit-predict via src/clustering/methods.py
  - dbscan
        -> full diagnostic pipeline: PCA reduction, k-distance plot,
           eps sweep, best-eps selection, final fit, figures
  - hdbscan
        -> full diagnostic pipeline: PCA reduction, mcs sweep,
           final fit, outlier scores, figures

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
                                save_clusters, ensure_dir)
from src.utils.pretty  import (header, subheader, metrics_table,
                                comparison_table, verdict, warning,
                                done, saved)
from src.clustering.methods import run_clustering


# -------------------------------------------------------
# Density method names
# -------------------------------------------------------

DENSITY_METHODS = {"dbscan", "hdbscan"}


# -------------------------------------------------------
# 2D projection helper (shared with evaluate.py via cache)
# -------------------------------------------------------

def _get_2d_projection(features, config, output_dir):
    """Load or compute 2D projection for density visualization."""
    cache = os.path.join(output_dir, "features_2d.npy")
    if os.path.exists(cache):
        print(f"  [proj] Loading cached 2D projection from {cache}")
        return np.load(cache)

    vis = config["visualization"]
    method = vis.get("method", "umap")

    if method == "umap":
        try:
            import umap as umap_lib
            cfg = vis["umap"]
            reducer = umap_lib.UMAP(
                n_neighbors  = cfg.get("n_neighbors", 30),
                n_components = 2,
                min_dist     = max(cfg.get("min_dist", 0.1), 0.05),
                metric       = cfg.get("metric", "cosine"),
                random_state = cfg.get("random_state", 42),
            )
            print("  [proj] Computing UMAP 2D projection ...")
            f2d = reducer.fit_transform(features)
        except ImportError:
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        cfg = vis.get("tsne", {})
        reducer = TSNE(n_components=2,
                       perplexity=cfg.get("perplexity", 40),
                       random_state=cfg.get("random_state", 42),
                       init="pca", learning_rate="auto")
        print("  [proj] Computing t-SNE 2D projection ...")
        f2d = reducer.fit_transform(features)

    np.save(cache, f2d)
    print(f"  [proj] Saved to {cache}")
    return f2d


# -------------------------------------------------------
# Run a density method with full diagnostics
# -------------------------------------------------------

def _run_density(method_name, method_cfg, features, true_label_ids,
                 class_names, features_2d, output_dir, config):
    """
    Route a density method to its full diagnostic pipeline.

    Merges method-level config params (from clustering.methods entry)
    with the top-level dbscan/hdbscan config block so both locations
    work correctly.
    """
    # Merge priority: top-level dbscan/hdbscan block WINS over clustering.methods entry.
    # Reason: the dedicated block (dbscan: pca_components, min_samples, etc.) is the
    # intentional, detailed configuration. The clustering.methods entry is just a routing
    # declaration — its parameters are only used as fallback when the dedicated block
    # does not exist at all.
    merged_config = dict(config)
    if method_name == "dbscan":
        # Start from method_cfg (routing entry), then let the dedicated block override
        base = {k: v for k, v in method_cfg.items() if k != "name"}
        base.update(config.get("dbscan", {}))   # top-level block wins
        merged_config["dbscan"] = base
        from src.clustering.density.dbscan_pipeline import DBSCANPipeline
        pipeline = DBSCANPipeline(merged_config)

    elif method_name == "hdbscan":
        base = {k: v for k, v in method_cfg.items() if k != "name"}
        base.update(config.get("hdbscan", {}))  # top-level block wins
        merged_config["hdbscan"] = base
        from src.clustering.density.hdbscan_pipeline import HDBSCANPipeline
        pipeline = HDBSCANPipeline(merged_config)

    results = pipeline.run(
        features, true_label_ids, class_names, features_2d, output_dir
    )

    # Save cluster assignments as .npz (same format as other methods)
    # so evaluate.py can load them for cross-method comparison
    cluster_path = os.path.join(output_dir, f"clusters_{method_name}.npz")
    save_clusters(cluster_path, results["labels"], method_name, method_cfg)

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

    header(f"CLUSTERING  —  {len(method_configs)} method(s)")
    print(f"\n  Dataset : {features.shape[0]} images | "
          f"{len(class_names)} classes | dim={features.shape[1]}")
    print(f"  Methods : {', '.join(method_names)}")

    # Pre-load text encoder if needed
    text_centroids = None
    if "kmeans_text_guided" in method_names:
        subheader("Loading text encoder for text-guided KMeans")
        from src.models.extractors import build_extractor
        extractor = build_extractor(config)
        templates = config["model"].get("prompt_templates", None)
        text_centroids = extractor.encode_text(class_names,
                                               prompt_templates=templates)

    # Pre-compute 2D projection if any density method is present
    features_2d = None
    if has_density:
        subheader("2D projection for density visualization")
        import warnings
        # Suppress UMAP's "n_jobs overridden" notice — it fires whenever
        # random_state is set (deterministic mode disables parallelism).
        # This is expected behavior, not a problem.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                message="n_jobs value.*overridden",
                category=UserWarning)
            features_2d = _get_2d_projection(features, config, output_dir)

    # ---- Run each method ----
    all_results  = {}
    total        = len(method_configs)

    for i, method_cfg in enumerate(method_configs, 1):
        name = method_cfg["name"]
        subheader(f"Method {i}/{total}:  {name.upper()}")
        t0 = time.time()

        if name in DENSITY_METHODS:
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

            # Full diagnostic pipeline for density methods
            results = _run_density(
                name, method_cfg,
                features, true_label_ids, class_names,
                features_2d, output_dir, config,
            )
            all_results[name] = results

        else:
            # Standard fit-predict for partition methods
            kwargs = {}
            if name == "kmeans_text_guided" and text_centroids is not None:
                kwargs["text_centroids"] = text_centroids

            cluster_ids = run_clustering(
                method_name = name,
                features    = features,
                params      = method_cfg,
                **kwargs,
            )
            save_clusters(
                os.path.join(output_dir, f"clusters_{name}.npz"),
                cluster_ids, name, method_cfg,
            )

            # Quick inline metric preview (full eval is in cli/evaluate.py)
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
            metrics_table([
                {"label": "ARI",           "value": ari},
                {"label": "NMI",           "value": nmi},
                {"label": "Accuracy",      "value": acc},
                {"label": "Clusters",      "value": all_results[name]["n_clusters"],
                 "note": f"/ {len(class_names)} expected"},
                {"label": "Time",          "value": f"{elapsed:.1f}s"},
            ], title=f"{name} — quick metrics")

    # Density comparison figure (if both density methods ran)
    d_ran = {n: all_results[n] for n in DENSITY_METHODS if n in all_results}
    if len(d_ran) == 2:
        from src.clustering.density.hdbscan_pipeline import plot_method_comparison
        export_pdf = config.get("visualization", {}).get("export_pdf", False)
        plot_method_comparison(
            d_ran["dbscan"], d_ran["hdbscan"],
            os.path.join(output_dir, "figures"), export_pdf,
        )

    # ---- Final comparison ----
    _print_summary(all_results)
    done(output_dir)


if __name__ == "__main__":
    main()
