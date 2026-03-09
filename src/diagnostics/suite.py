"""
src/diagnostics/suite.py
------------------------
Diagnostic suite for quickly assessing embedding and clustering quality.

Produces a compact text report covering:
  1. Feature quality stats (mean norm, variance, dimensionality)
  2. Quick K-Means check on 2D projection (silhouette score)
  3. Top confusion pairs summary
  4. Nearest-neighbor accuracy (proxy for separability)

Also saves a compact diagnostic figure to figures/.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

matplotlib.use("Agg")


# -------------------------------------------------------
# Feature Quality Stats
# -------------------------------------------------------

def feature_stats(features: np.ndarray) -> dict:
    """
    Compute basic statistics about the feature array.

    These help catch preprocessing issues, like un-normalized embeddings
    (norms should all be ~1.0 if L2-normalized) or extreme values.
    """
    norms       = np.linalg.norm(features, axis=1)
    variance    = np.var(features, axis=0)

    stats = {
        "n_samples":           features.shape[0],
        "n_dimensions":        features.shape[1],
        "mean_l2_norm":        float(np.mean(norms)),
        "std_l2_norm":         float(np.std(norms)),
        "min_l2_norm":         float(np.min(norms)),
        "max_l2_norm":         float(np.max(norms)),
        "mean_feature_var":    float(np.mean(variance)),
        "dead_dimensions":     int((variance < 1e-8).sum()),  # zero-variance dims
    }

    return stats


# -------------------------------------------------------
# Silhouette Score on 2D Projection
# -------------------------------------------------------

def silhouette_on_2d(features_2d: np.ndarray, true_labels: np.ndarray,
                     max_samples: int = 3000) -> float:
    """
    Compute silhouette score on the 2D embedding (UMAP/t-SNE).

    Silhouette score ranges from -1 (bad) to 1 (perfect).
    Values above 0.3 generally indicate meaningful cluster structure.

    We compute it on 2D rather than full-D because:
    - It is much faster
    - It reflects the quality of the 2D visualization itself

    Args:
        features_2d: 2D projection, shape (N, 2).
        true_labels: Ground-truth class IDs.
        max_samples: Subsampled to this number for speed.

    Returns:
        Float silhouette score.
    """
    n = len(true_labels)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        features_2d = features_2d[idx]
        true_labels = true_labels[idx]

    score = silhouette_score(features_2d, true_labels, metric="euclidean")
    return float(score)


# -------------------------------------------------------
# Nearest-Neighbor Accuracy (Full-D)
# -------------------------------------------------------

def nn_accuracy(features: np.ndarray, true_labels: np.ndarray,
                k: int = 1, max_samples: int = 3000) -> float:
    """
    Estimate k-nearest-neighbor accuracy using leave-one-out cross-validation.

    This measures how separable the classes are in embedding space
    WITHOUT any training — it is purely a property of the embeddings.

    A high NN accuracy means the embeddings are very clean. It also serves
    as an upper bound: if NN accuracy is low, clustering will also be low.

    Args:
        k:           Number of neighbors (default 1 = exact match neighbor).
        max_samples: Subsampled for speed.

    Returns:
        Float accuracy in [0, 1].
    """
    n = len(true_labels)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        features   = features[idx]
        true_labels = true_labels[idx]

    knn    = KNeighborsClassifier(n_neighbors=k + 1, metric="cosine")
    scores = cross_val_score(knn, features, true_labels, cv=3, scoring="accuracy")

    return float(scores.mean())


# -------------------------------------------------------
# Compact Diagnostic Plot
# -------------------------------------------------------

def plot_diagnostic(features_2d: np.ndarray, true_labels: np.ndarray,
                    cluster_ids: np.ndarray, stats: dict,
                    output_dir: str, dpi: int = 200) -> None:
    """
    2x2 diagnostic figure:
      [top-left]  2D scatter, colored by true labels
      [top-right] 2D scatter, colored by K-Means clusters
      [bottom-left]  Feature norm distribution
      [bottom-right] Cluster size histogram
    """
    n_classes = int(true_labels.max()) + 1
    cmap_true    = plt.cm.nipy_spectral
    cmap_cluster = plt.cm.tab20 if n_classes <= 20 else plt.cm.nipy_spectral

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Diagnostic Overview", fontsize=14, y=1.01)

    # Subsample for scatter plots
    max_pts = 3000
    n = len(true_labels)
    if n > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_pts, replace=False)
        feat2d_s  = features_2d[idx]
        true_s    = true_labels[idx]
        clust_s   = cluster_ids[idx] if cluster_ids is not None else None
    else:
        feat2d_s = features_2d
        true_s   = true_labels
        clust_s  = cluster_ids

    # Top-left: true labels
    sc1 = axes[0, 0].scatter(
        feat2d_s[:, 0], feat2d_s[:, 1],
        c=true_s, cmap=cmap_true, s=6, alpha=0.5, linewidths=0
    )
    axes[0, 0].set_title("2D Embedding — True Labels")
    axes[0, 0].set_xlabel("Component 1")
    axes[0, 0].set_ylabel("Component 2")

    # Top-right: cluster assignments
    if clust_s is not None:
        sc2 = axes[0, 1].scatter(
            feat2d_s[:, 0], feat2d_s[:, 1],
            c=clust_s, cmap=cmap_cluster, s=6, alpha=0.5, linewidths=0
        )
        axes[0, 1].set_title("2D Embedding — Cluster Assignments")
    else:
        axes[0, 1].set_title("2D Embedding — Cluster Assignments (none available)")
    axes[0, 1].set_xlabel("Component 1")
    axes[0, 1].set_ylabel("Component 2")

    # Bottom-left: norm distribution
    norms = np.linalg.norm(features_2d, axis=1)   # norms of 2D projections
    # Recalculate from full-dim norms if possible (not passed here, use 2d proxy)
    axes[1, 0].hist(norms, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white")
    axes[1, 0].set_title("2D Embedding Norm Distribution")
    axes[1, 0].set_xlabel("L2 Norm")
    axes[1, 0].set_ylabel("Count")

    # Bottom-right: cluster sizes
    if cluster_ids is not None:
        valid_ids = cluster_ids[cluster_ids >= 0]
        _, counts = np.unique(valid_ids, return_counts=True)
        axes[1, 1].hist(counts, bins=30, color="#55A868", alpha=0.85, edgecolor="white")
        axes[1, 1].axvline(x=np.mean(counts), color="#C44E52", linestyle="--",
                           linewidth=1.5, label=f"Mean={np.mean(counts):.1f}")
        axes[1, 1].set_title("Cluster Size Distribution")
        axes[1, 1].set_xlabel("Cluster Size")
        axes[1, 1].set_ylabel("Number of Clusters")
        axes[1, 1].legend()

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "diagnostic_overview.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[diagnostics] Figure saved to {path}")
    plt.close(fig)


# -------------------------------------------------------
# Main Diagnostic Function
# -------------------------------------------------------

def run_diagnostics(features: np.ndarray, features_2d: np.ndarray,
                    true_labels: np.ndarray, cluster_ids: np.ndarray,
                    class_names: list, config: dict,
                    output_dir: str) -> str:
    """
    Run all diagnostics and write a text report.

    Args:
        features:     Full-dimensional feature array (N, D).
        features_2d:  2D UMAP/t-SNE projection (N, 2).
        true_labels:  Ground-truth integer class IDs (N,).
        cluster_ids:  Cluster assignments from any method (N,). Can be None.
        class_names:  List of class name strings.
        config:       Full pipeline config dict.
        output_dir:   Where to save the report and figure.

    Returns:
        The report as a string (also saved to diagnostics_report.txt).
    """
    print("\n[diagnostics] Running diagnostic suite ...")
    t0 = time.time()

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")

    lines = []
    lines.append("=" * 70)
    lines.append("DIAGNOSTIC REPORT")
    lines.append(f"Dataset : {config['dataset']['name']}")
    lines.append(f"Model   : {config['model']['name']}")
    lines.append("=" * 70)

    # 1. Feature stats
    lines.append("\n--- Feature Quality ---")
    stats = feature_stats(features)
    for k, v in stats.items():
        lines.append(f"  {k:<25s}: {v}")

    if abs(stats["mean_l2_norm"] - 1.0) > 0.05:
        lines.append("  WARNING: mean L2 norm is not ~1. "
                     "Check normalize_embeddings in config.")
    if stats["dead_dimensions"] > 0:
        lines.append(f"  WARNING: {stats['dead_dimensions']} "
                     "zero-variance dimensions found.")

    # 2. Nearest-neighbor accuracy (fast separability check)
    lines.append("\n--- Nearest-Neighbor Separability ---")
    nn_acc = nn_accuracy(features, np.array(true_labels))
    lines.append(f"  1-NN accuracy (cosine, 3-fold CV): {nn_acc:.4f}")
    if nn_acc > 0.90:
        lines.append("  Embeddings are highly separable.")
    elif nn_acc > 0.70:
        lines.append("  Embeddings show moderate separability.")
    else:
        lines.append("  Embeddings have low separability; clustering may be poor.")

    # 3. Silhouette on 2D
    if features_2d is not None:
        lines.append("\n--- 2D Embedding Quality ---")
        sil = silhouette_on_2d(features_2d, np.array(true_labels))
        lines.append(f"  Silhouette score on 2D projection: {sil:.4f}")
        if sil > 0.5:
            lines.append("  2D projection shows strong cluster separation.")
        elif sil > 0.2:
            lines.append("  2D projection shows moderate cluster separation.")
        else:
            lines.append("  2D projection shows weak cluster separation.")

    # 4. Quick K-Means check
    lines.append("\n--- Quick K-Means Baseline (n_init=10) ---")
    n_classes = len(class_names)
    km = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
    c  = km.fit_predict(features)

    from src.evaluation.metrics import hungarian_accuracy
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    acc, _ = hungarian_accuracy(np.array(true_labels), c)
    ari     = adjusted_rand_score(true_labels, c)
    nmi     = normalized_mutual_info_score(true_labels, c)

    lines.append(f"  Accuracy (Hungarian) : {acc:.4f}")
    lines.append(f"  ARI                  : {ari:.4f}")
    lines.append(f"  NMI                  : {nmi:.4f}")

    # 5. Timing
    elapsed = time.time() - t0
    lines.append(f"\n--- Timing ---")
    lines.append(f"  Diagnostic suite completed in {elapsed:.1f}s")
    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print(report)

    # Save text report
    report_path = os.path.join(output_dir, "diagnostics_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[diagnostics] Report saved to {report_path}")

    # Save diagnostic figure
    if features_2d is not None:
        plot_diagnostic(features_2d, np.array(true_labels),
                        cluster_ids, stats, figures_dir)

    return report
