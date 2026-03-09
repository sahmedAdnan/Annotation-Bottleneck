"""
experiments/run_pipeline.py
----------------------------
Unified runner: execute the full BioCLIP clustering pipeline in one command.

4 steps, in order:
    1. extract   -- extract BioCLIP-2 image embeddings, save features.npz
    2. cluster   -- run ALL clustering methods (KMeans, text-guided KMeans,
                    Agglomerative, Spectral, DBSCAN, HDBSCAN).
                    Methods to run are defined in configs/eumoth.yaml under
                    clustering.methods. DBSCAN and HDBSCAN run their full
                    diagnostic pipelines (PCA, sweep, figures) from here.
    3. evaluate  -- evaluate all cluster results, generate publication figures
    4. diagnose  -- feature quality checks, 1-NN accuracy, silhouette scores

WHY 4 STEPS (NOT 5):
    Density clustering (DBSCAN, HDBSCAN) is clustering. It belongs in step 2
    alongside KMeans and Agglomerative. Separating it into a dedicated step
    implied it was a different kind of operation — it is not. All methods that
    assign images to groups run in step 2.

EACH STEP IS SKIPPABLE:
    Features and 2D projections are cached on disk, so re-runs are fast.
    Skip flags let you resume from any point.

Usage:
    # Run everything from scratch
    python experiments/run_pipeline.py --config configs/eumoth.yaml

    # Resume from clustering (features already extracted)
    python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-extract

    # Skip HDBSCAN only (if not installed)
    python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-hdbscan

    # Skip both density methods
    python experiments/run_pipeline.py --config configs/eumoth.yaml --skip-density

    # Re-run only evaluation figures (clustering already done)
    python experiments/run_pipeline.py --config configs/eumoth.yaml \\
        --skip-extract --skip-cluster --skip-diagnose

Each CLI step also works standalone:
    python cli/extract.py   --config configs/eumoth.yaml
    python cli/cluster.py   --config configs/eumoth.yaml
    python cli/evaluate.py  --config configs/eumoth.yaml
    python cli/diagnose.py  --config configs/eumoth.yaml
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli.extract  as step_extract
import cli.cluster  as step_cluster
import cli.evaluate as step_evaluate
import cli.diagnose as step_diagnose


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _banner(title: str) -> None:
    bar = "=" * 70
    print("\n" + bar)
    print("  " + title)
    print(bar)


def _run_step(label: str, fn, config_path: str, **kwargs) -> float:
    """Run one pipeline step. Print a header, time it, return elapsed seconds."""
    _banner("STEP: " + label)
    t0 = time.time()
    fn(config_path=config_path, **kwargs)
    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = elapsed % 60
    print("\n  [{label}] finished in {m}m {s:.0f}s".format(
        label=label, m=mins, s=secs))
    return elapsed


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full BioCLIP-2 clustering pipeline end to end.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g. configs/eumoth.yaml)."
    )

    # Step-level skip flags
    parser.add_argument("--skip-extract",  action="store_true",
                        help="Skip Step 1: feature extraction.\n"
                             "Use if features.npz already exists.")
    parser.add_argument("--skip-cluster",  action="store_true",
                        help="Skip Step 2: all clustering methods.")
    parser.add_argument("--skip-evaluate", action="store_true",
                        help="Skip Step 3: evaluation and figures.")
    parser.add_argument("--skip-diagnose", action="store_true",
                        help="Skip Step 4: diagnostics.")

    # Density method flags (only relevant when step 2 runs)
    parser.add_argument("--skip-density",  action="store_true",
                        help="Within Step 2: skip all density methods\n"
                             "(DBSCAN and HDBSCAN).")
    parser.add_argument("--skip-dbscan",   action="store_true",
                        help="Within Step 2: skip DBSCAN only.")
    parser.add_argument("--skip-hdbscan",  action="store_true",
                        help="Within Step 2: skip HDBSCAN only.\n"
                             "Use if hdbscan is not installed.")

    args = parser.parse_args()

    wall_start = time.time()
    timings    = {}

    _banner("PIPELINE START")
    steps_active = []
    if not args.skip_extract:  steps_active.append("extract")
    if not args.skip_cluster:  steps_active.append("cluster")
    if not args.skip_evaluate: steps_active.append("evaluate")
    if not args.skip_diagnose: steps_active.append("diagnose")

    print("  Config : " + args.config)
    print("  Steps  : " + "  >  ".join(steps_active) if steps_active
          else "  Steps  : (all skipped)")

    # ---- Step 1: Extract ----
    if not args.skip_extract:
        timings["extract"] = _run_step(
            "1/4  Feature Extraction",
            step_extract.main,
            args.config,
        )
    else:
        print("\n  [skip] Step 1: Feature Extraction")

    # ---- Step 2: Cluster (ALL methods, including density) ----
    if not args.skip_cluster:
        # Translate top-level skip flags into config overrides.
        # cluster.py reads skip_density, skip_dbscan, skip_hdbscan from kwargs.
        cluster_kwargs = {
            "skip_density":  args.skip_density,
            "skip_dbscan":   args.skip_dbscan  or args.skip_density,
            "skip_hdbscan":  args.skip_hdbscan or args.skip_density,
        }
        timings["cluster"] = _run_step(
            "2/4  Clustering  (KMeans / Agglomerative / Spectral / DBSCAN / HDBSCAN)",
            step_cluster.main,
            args.config,
            **cluster_kwargs,
        )
    else:
        print("\n  [skip] Step 2: Clustering")

    # ---- Step 3: Evaluate ----
    if not args.skip_evaluate:
        timings["evaluate"] = _run_step(
            "3/4  Evaluation + Publication Figures",
            step_evaluate.main,
            args.config,
        )
    else:
        print("\n  [skip] Step 3: Evaluation")

    # ---- Step 4: Diagnose ----
    if not args.skip_diagnose:
        timings["diagnose"] = _run_step(
            "4/4  Diagnostics",
            step_diagnose.main,
            args.config,
        )
    else:
        print("\n  [skip] Step 4: Diagnostics")

    # ---- Final summary ----
    total = time.time() - wall_start
    _banner("PIPELINE COMPLETE")
    if timings:
        print("  Timing breakdown:")
        for step_name, secs in timings.items():
            m = int(secs // 60)
            s = secs % 60
            print("    {:<12s}: {}m {:.0f}s".format(step_name, m, s))
    total_m = int(total // 60)
    total_s = total % 60
    print("\n  Total wall time : {}m {:.0f}s".format(total_m, total_s))
    print("  All outputs in the output_dir defined in your config.")


if __name__ == "__main__":
    main()
