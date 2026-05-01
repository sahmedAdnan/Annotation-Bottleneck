"""
cli/compare_models.py
---------------------
Compare clustering performance across multiple foundation models on one dataset.

Reads the metrics_summary.csv saved by cli/evaluate.py from each model's
result directory and produces a combined comparison table + figures.

Usage:
    python cli/compare_models.py \\
        --dataset "EU-Moth (200 classes)" \\
        --results results/eumoth:BioCLIP-2 \\
                  results/eumoth_bioclip:BioCLIP \\
                  results/eumoth_clip:CLIP \\
                  results/eumoth_dinov2:DINOv2 \\
        --output results/comparison_eumoth

    # Optional flags:
        --dpi 300          (default: 300)
        --pdf              (also save PDF versions of figures)

Output files:
    <output>/comparison_table.csv        — full method × model table
    <output>/heatmap_ari.png             — ARI heatmap (methods × models)
    <output>/heatmap_accuracy.png        — Accuracy heatmap
    <output>/heatmap_nmi.png             — NMI heatmap
    <output>/bar_ari_by_method.png       — grouped bar: models per method
    <output>/model_best_summary.png      — best method per model
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "font.family":    "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.bbox":   "tight",
})

# Canonical display order for clustering methods (add any new methods here)
METHOD_ORDER = [
    "kmeans",
    "agglomerative",
    "spectral",
    "dbscan",
    "hdbscan",
    "gmm",
]

# Human-readable labels for method names
METHOD_LABELS = {
    "kmeans":             "K-Means",
    "agglomerative":      "Agglomerative",
    "spectral":           "Spectral",
    "dbscan":             "DBSCAN",
    "hdbscan":            "HDBSCAN",
    "gmm":                "GMM",
}

# Bar colors per model (up to 6 models)
MODEL_COLORS = [
    "#4C72B0",  # blue    — BioCLIP-2
    "#55A868",  # green   — BioCLIP
    "#C44E52",  # red     — CLIP
    "#8172B2",  # purple  — DINOv2
    "#CCB974",  # yellow  — extra
    "#64B5CD",  # cyan    — extra
]


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _save(fig, output_dir, filename, dpi=300, pdf=False):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"    [fig]  {path}")
    if pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight",
                    facecolor="white")


def _load_results(result_dirs: list) -> pd.DataFrame:
    """
    Load metrics_summary.csv from each result directory.

    Returns a DataFrame with columns:
        model, method, accuracy, ari, nmi, n_clusters_found, n_noise_excluded
    """
    frames = []
    for entry in result_dirs:
        parts = entry.split(":", 1)
        if len(parts) != 2:
            print(f"  [warn] Skipping malformed entry: '{entry}'  "
                  f"(expected 'path:label')")
            continue
        result_dir, label = parts[0].strip(), parts[1].strip()
        csv_path = os.path.join(result_dir, "metrics_summary.csv")
        if not os.path.exists(csv_path):
            print(f"  [warn] Not found (run evaluate.py first): {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "model", label)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No metrics_summary.csv files found. "
            "Run cli/evaluate.py for each model first."
        )
    return pd.concat(frames, ignore_index=True)


def _pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to methods × models matrix for a given metric."""
    pv = df.pivot_table(index="method", columns="model", values=metric,
                        aggfunc="first")
    # Sort rows by METHOD_ORDER, keeping any unknowns at the end
    known  = [m for m in METHOD_ORDER if m in pv.index]
    others = [m for m in pv.index      if m not in METHOD_ORDER]
    pv = pv.loc[known + others]
    # Rename row index to readable labels
    pv.index = [METHOD_LABELS.get(m, m) for m in pv.index]
    return pv


# -------------------------------------------------------
# Figure 1: Heatmap (methods × models) for one metric
# -------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, metric: str, dataset_name: str,
                 output_dir: str, dpi: int, pdf: bool) -> None:
    """
    Annotation heatmap: rows = clustering methods, columns = models.
    NaN cells are shown in light gray.

    HOW TO READ:
        Darker blue = higher score.
        Gray cell   = method not available for that model.
        Each cell shows the exact numeric score.
    """
    pv = _pivot(df, metric)
    models   = list(pv.columns)
    n_rows   = len(pv)
    n_cols   = len(models)

    fig_w = max(7, n_cols * 2.0)
    fig_h = max(4, n_rows * 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Create a mask for NaN cells
    mask = pv.isna()

    # Draw the heatmap (NaN cells will appear as background)
    sns.heatmap(
        pv.astype(float),
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#cccccc",
        mask=mask,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": metric.upper()},
    )

    # Fill NaN cells with gray
    for r in range(n_rows):
        for c in range(n_cols):
            if mask.iloc[r, c]:
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1,
                    fill=True, color="#dddddd", zorder=2
                ))
                ax.text(
                    c + 0.5, r + 0.5, "N/A",
                    ha="center", va="center",
                    fontsize=8, color="#888888", zorder=3
                )

    metric_display = metric.upper().replace("_", " ")
    ax.set_title(
        f"{metric_display} — Model × Method Comparison\n{dataset_name}",
        pad=14
    )
    ax.set_xlabel("Model", labelpad=8)
    ax.set_ylabel("Clustering Method", labelpad=8)
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)

    fig.text(
        0.5, -0.04,
        f"Values are {metric_display} (higher = better, range 0–1).  "
        "Gray cells = method not applicable (model has no text encoder).  "
        "Density methods (DBSCAN, HDBSCAN) exclude noise points.",
        ha="center", fontsize=7, color="#555555", style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, f"heatmap_{metric}.png", dpi=dpi, pdf=pdf)
    plt.close(fig)


# -------------------------------------------------------
# Figure 2: Grouped bar chart — ARI by method, grouped by model
# -------------------------------------------------------

def plot_bar_by_method(df: pd.DataFrame, metric: str, dataset_name: str,
                       output_dir: str, dpi: int, pdf: bool) -> None:
    """
    Grouped bar chart: x-axis = clustering method, bar groups = models.

    HOW TO READ:
        Each cluster of bars = one clustering method.
        Each bar within a cluster = one foundation model.
        Height = metric value (ARI, Accuracy, or NMI).
        Directly compare models side-by-side on the same method.
    """
    pv     = _pivot(df, metric)
    models = list(pv.columns)
    methods = list(pv.index)

    n_methods = len(methods)
    n_models  = len(models)
    w  = 0.8 / n_models
    x  = np.arange(n_methods)

    fig, ax = plt.subplots(figsize=(max(10, n_methods * 1.8), 6))

    for i, model in enumerate(models):
        values = pv[model].values.astype(float)
        offset = (i - n_models / 2 + 0.5) * w
        bars = ax.bar(x + offset, values * 100, w,
                      label=model,
                      color=MODEL_COLORS[i % len(MODEL_COLORS)],
                      alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 2:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.6,
                    f"{h:.1f}",
                    ha="center", va="bottom", fontsize=6.5
                )

    metric_display = metric.upper().replace("_", " ")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(f"{metric_display} (%)")
    ax.set_ylim(0, 115)
    ax.set_title(
        f"{metric_display} by Clustering Method — Model Comparison\n{dataset_name}",
        pad=12
    )
    ax.legend(title="Model", loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.text(
        0.5, -0.05,
        f"Bars show {metric_display} (×100 = %). N/A bars omitted (model has no text encoder).  "
        "Density methods exclude noise points from scoring.",
        ha="center", fontsize=7, color="#555555", style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, f"bar_{metric}_by_method.png", dpi=dpi, pdf=pdf)
    plt.close(fig)


# -------------------------------------------------------
# Figure 3: Best method per model — summary bars
# -------------------------------------------------------

def plot_model_summary(df: pd.DataFrame, dataset_name: str,
                       output_dir: str, dpi: int, pdf: bool) -> None:
    """
    For each model: find its best method (by ARI) and show ARI / Accuracy / NMI.

    HOW TO READ:
        Each group of 3 bars = one model.
        Blue = Accuracy, Green = ARI, Red = NMI.
        The label above each group shows which clustering method achieved this score.
        Use this to quickly see which model is strongest overall.
    """
    models = df["model"].unique().tolist()
    best_rows = []
    for model in models:
        sub = df[df["model"] == model]
        best_idx = sub["ari"].idxmax()
        best_rows.append(sub.loc[best_idx])

    summary = pd.DataFrame(best_rows).reset_index(drop=True)

    x = np.arange(len(models))
    w = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), 6))

    b1 = ax.bar(x - w, summary["accuracy"] * 100, w,
                label="Accuracy (%)", color="#4C72B0", alpha=0.9)
    b2 = ax.bar(x,     summary["ari"]      * 100, w,
                label="ARI (%)",      color="#55A868", alpha=0.9)
    b3 = ax.bar(x + w, summary["nmi"]      * 100, w,
                label="NMI (%)",      color="#C44E52", alpha=0.9)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7
                )

    # Annotate best method name below each model
    for i, row in summary.iterrows():
        best_label = METHOD_LABELS.get(row["method"], row["method"])
        ax.text(x[i], -7, f"best: {best_label}",
                ha="center", va="top", fontsize=7, color="#444444",
                style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=0)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 120)
    ax.set_title(
        f"Best-Method Performance per Model\n{dataset_name}",
        pad=12
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.text(
        0.5, -0.08,
        "Each model's best method (by ARI) is selected automatically.  "
        "The method label below each group shows which method achieved these scores.  "
        "Higher is better for all three metrics.",
        ha="center", fontsize=7, color="#555555", style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, "model_best_summary.png", dpi=dpi, pdf=pdf)
    plt.close(fig)


# -------------------------------------------------------
# Console table
# -------------------------------------------------------

def print_comparison_table(df: pd.DataFrame, dataset_name: str) -> None:
    """Print a ranked summary table to the console."""
    print("\n" + "=" * 70)
    print(f"  MODEL COMPARISON  —  {dataset_name}")
    print("=" * 70)

    # Best ARI per (model, method)
    top = (df.sort_values("ari", ascending=False)
             .groupby("model", sort=False)
             .first()
             .reset_index()
             .sort_values("ari", ascending=False))

    print(f"\n  {'Rank':<5} {'Model':<20} {'Best Method':<24} "
          f"{'ARI':>6} {'NMI':>6} {'Acc':>6}")
    print("  " + "-" * 66)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        method_label = METHOD_LABELS.get(row["method"], row["method"])
        print(f"  {rank:<5} {row['model']:<20} {method_label:<24} "
              f"{row['ari']:>6.4f} {row['nmi']:>6.4f} {row['accuracy']:>6.4f}")

    print("\n  Full table (all methods):")
    print(f"  {'Model':<20} {'Method':<24} {'ARI':>6} {'NMI':>6} {'Acc':>6}")
    print("  " + "-" * 62)
    for _, row in df.sort_values(["model", "ari"], ascending=[True, False]).iterrows():
        method_label = METHOD_LABELS.get(row["method"], row["method"])
        print(f"  {row['model']:<20} {method_label:<24} "
              f"{row['ari']:>6.4f} {row['nmi']:>6.4f} {row['accuracy']:>6.4f}")
    print("=" * 70 + "\n")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare foundation models across clustering methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Display name for the dataset, e.g. 'EU-Moth (200 classes)'"
    )
    parser.add_argument(
        "--results", required=True, nargs="+",
        metavar="DIR:LABEL",
        help="Result directories paired with model labels. "
             "Format: 'results/eumoth:BioCLIP-2'. "
             "Repeat for each model."
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory where comparison figures and CSV are saved."
    )
    parser.add_argument("--dpi",  type=int,  default=300)
    parser.add_argument("--pdf",  action="store_true",
                        help="Also export PDF versions of all figures.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- Load ---
    print(f"\n[compare] Loading results for: {args.dataset}")
    df = _load_results(args.results)

    # Tidy method names
    df["method"] = df["method"].str.strip()

    print(f"  Models loaded  : {df['model'].unique().tolist()}")
    print(f"  Methods found  : {df['method'].unique().tolist()}")
    print(f"  Rows total     : {len(df)}")

    # --- Console summary ---
    print_comparison_table(df, args.dataset)

    # --- Save combined CSV ---
    csv_path = os.path.join(args.output, "comparison_table.csv")
    df_sorted = df.sort_values(["model", "ari"], ascending=[True, False])
    df_sorted.to_csv(csv_path, index=False)
    print(f"  [csv]  {csv_path}")

    # --- Figures ---
    print(f"\n[compare] Generating figures -> {args.output}/")

    for metric in ("ari", "accuracy", "nmi"):
        plot_heatmap(df, metric, args.dataset, args.output, args.dpi, args.pdf)
        plot_bar_by_method(df, metric, args.dataset, args.output, args.dpi, args.pdf)

    plot_model_summary(df, args.dataset, args.output, args.dpi, args.pdf)

    print(f"\n[compare] Done.  All files saved to: {args.output}")


if __name__ == "__main__":
    main()
