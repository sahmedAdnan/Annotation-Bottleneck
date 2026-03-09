"""
cli/diagnose.py
---------------
CLI command: Run the diagnostic suite and produce a report.

Usage:
    python cli/diagnose.py --config configs/eumoth.yaml

Requires:
    <output_dir>/features.npz
    <output_dir>/features_2d.npy         (optional; computed if missing)
    <output_dir>/clusters_kmeans.npz     (optional; used for diagnostic overlay)

Output:
    <output_dir>/diagnostics_report.txt
    <output_dir>/figures/diagnostic_overview.png
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io            import set_seed, load_config, load_features, ensure_dir
from src.diagnostics.suite   import run_diagnostics


def main(config_path: str = None):
    if config_path is None:
        parser = argparse.ArgumentParser(description="Run diagnostics on extracted features.")
        parser.add_argument("--config", required=True, help="Path to YAML config file.")
        args = parser.parse_args()
        config_path = args.config

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    # Load features
    features_path = os.path.join(output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"[diagnose] ERROR: features not found at {features_path}")
        print("  Run: python cli/extract.py --config <config>  first.")
        sys.exit(1)

    features, labels, _, class_names = load_features(features_path)
    true_labels = np.array(labels)

    # Load 2D projection if available
    proj_path = os.path.join(output_dir, "features_2d.npy")
    if os.path.exists(proj_path):
        print(f"[diagnose] Loading cached 2D projection from {proj_path}")
        features_2d = np.load(proj_path)
    else:
        print("[diagnose] No 2D projection found. Run cli/evaluate.py first, "
              "or 2D-dependent diagnostics will be skipped.")
        features_2d = None

    # Load first available cluster file for diagnostic overlay
    cluster_ids = None
    for method_cfg in config["clustering"]["methods"]:
        cpath = os.path.join(output_dir, f"clusters_{method_cfg['name']}.npz")
        if os.path.exists(cpath):
            from src.utils.io import load_clusters
            cluster_ids, _, _ = load_clusters(cpath)
            print(f"[diagnose] Using cluster assignments from: {method_cfg['name']}")
            break

    # Run full diagnostic suite
    run_diagnostics(
        features    = features,
        features_2d = features_2d,
        true_labels = true_labels,
        cluster_ids = cluster_ids,
        class_names = class_names,
        config      = config,
        output_dir  = output_dir,
    )

    print("\n[diagnose] Done.")


if __name__ == "__main__":
    main()
