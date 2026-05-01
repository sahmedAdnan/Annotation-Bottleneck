"""
src/reduction/reduction.py
--------------------------
Dimensionality reduction utilities for the clustering pipeline.

Two-path architecture (empirically validated on EU-Moth and CUB-200):

    Partition methods  ->  PCA-50D
        PCA's variance-maximization objective aligns with the variance-based
        objectives of K-Means, Spectral, Agglomerative, and GMM.
        PCA-50D retains 78-86% of variance across datasets and is the ONLY
        reduction that works for all 4 partition methods (Spectral and GMM
        fail on UMAP due to graph disconnection and singular covariances).

    Density methods    ->  UMAP-100D
        UMAP preserves local neighborhood structure, which aligns with the
        density/neighborhood objectives of DBSCAN and HDBSCAN.
        UMAP-100D achieves 5-10x less noise than PCA with comparable or
        better ARI on both EU-Moth (ARI=0.9798) and CUB-200 (ARI=0.8455).

References:
    Ding & He (2004) — PCA and K-Means optimize dual objectives
    McInnes et al. (2018) — UMAP preserves local topological structure
    Beyer et al. (1999) — distance concentration in high dimensions
"""

import warnings
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional


# -------------------------------------------------------
# PCA Reduction (for partition methods)
# -------------------------------------------------------

def fit_pca(features: np.ndarray,
            n_components: int = 50,
            random_state: int = 42) -> Tuple[np.ndarray, PCA, float]:
    """
    Reduce features with PCA for partition-based clustering.

    Args:
        features:      Raw embeddings, shape (N, D).
        n_components:  Target dimensionality (default 50).
        random_state:  Random seed for reproducibility.

    Returns:
        reduced:       PCA-reduced features, shape (N, n_components).
        pca:           Fitted PCA object (can project text centroids).
        variance:      Fraction of total variance retained.
    """
    n_components = min(n_components, features.shape[1], features.shape[0])

    print(f"  [reduction] PCA: {features.shape[1]}D -> {n_components}D ...")
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(features)

    variance = float(pca.explained_variance_ratio_.sum())
    print(f"  [reduction] PCA variance retained: {variance:.1%}")

    return reduced, pca, variance


# -------------------------------------------------------
# UMAP Reduction (for density methods)
# -------------------------------------------------------

def fit_umap(features: np.ndarray,
             n_components: int = 100,
             n_neighbors: int = 15,
             min_dist: float = 0.1,
             metric: str = "cosine",
             random_state: int = 42) -> Tuple[np.ndarray, object]:
    """
    Reduce features with UMAP for density-based clustering.

    UMAP preserves local neighborhood topology, which aligns with the
    local-density objectives of DBSCAN and HDBSCAN. Unlike PCA, UMAP
    does not distort the local density structure that density methods
    rely on for cluster boundary detection.

    Args:
        features:      Raw embeddings, shape (N, D).
        n_components:  Target dimensionality (default 100).
        n_neighbors:   Controls local vs global structure (default 15).
        min_dist:      Minimum distance in embedded space (default 0.1).
        metric:        Distance metric (default "cosine").
        random_state:  Random seed for reproducibility.

    Returns:
        reduced:       UMAP-reduced features, shape (N, n_components).
        reducer:       Fitted UMAP object.

    Raises:
        ImportError: If umap-learn is not installed.
    """
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError(
            "UMAP reduction requires: pip install umap-learn\n"
            "Falling back to PCA is not recommended for density methods."
        )

    print(f"  [reduction] UMAP: {features.shape[1]}D -> {n_components}D "
          f"(neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}) ...")

    # Suppress UMAP's "n_jobs overridden" notice when random_state is set
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
            message="n_jobs value.*overridden",
            category=UserWarning)

        reducer = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        reduced = reducer.fit_transform(features)

    print(f"  [reduction] UMAP complete: {reduced.shape}")

    return reduced, reducer


# -------------------------------------------------------
# Config-driven reduction
# -------------------------------------------------------

def get_reduction_config(config: dict) -> dict:
    """
    Extract reduction settings from config, with sensible defaults.

    Config structure (all optional — defaults apply if omitted):
        reduction:
          partition:
            method: pca
            n_components: 50
          density:
            method: umap
            n_components: 100
            n_neighbors: 15
            min_dist: 0.1
            metric: cosine

    Returns:
        Dict with 'partition' and 'density' sub-dicts.
    """
    reduction = config.get("reduction", {})

    partition = reduction.get("partition", {})
    density = reduction.get("density", {})

    return {
        "partition": {
            "method":       partition.get("method", "pca"),
            "n_components": partition.get("n_components", 50),
        },
        "density": {
            "method":       density.get("method", "umap"),
            "n_components": density.get("n_components", 100),
            "n_neighbors":  density.get("n_neighbors", 15),
            "min_dist":     density.get("min_dist", 0.1),
            "metric":       density.get("metric", "cosine"),
        },
    }


def reduce_for_partition(features: np.ndarray,
                         config: dict,
                         random_state: int = 42) -> Tuple[np.ndarray, Optional[PCA], float]:
    """
    Apply the configured reduction for partition methods.

    Returns:
        reduced, pca_object (or None), variance_retained
    """
    rcfg = get_reduction_config(config)["partition"]

    if rcfg["method"] == "pca":
        return fit_pca(features, rcfg["n_components"], random_state)
    else:
        raise ValueError(f"Unknown partition reduction: {rcfg['method']}")


def reduce_for_density(features: np.ndarray,
                       config: dict,
                       random_state: int = 42) -> Tuple[np.ndarray, object]:
    """
    Apply the configured reduction for density methods.

    Returns:
        reduced, reducer_object
    """
    rcfg = get_reduction_config(config)["density"]

    if rcfg["method"] == "umap":
        return fit_umap(
            features,
            n_components=rcfg["n_components"],
            n_neighbors=rcfg["n_neighbors"],
            min_dist=rcfg["min_dist"],
            metric=rcfg["metric"],
            random_state=random_state,
        )
    elif rcfg["method"] == "pca":
        reduced, pca, _ = fit_pca(features, rcfg["n_components"], random_state)
        return reduced, pca
    else:
        raise ValueError(f"Unknown density reduction: {rcfg['method']}")
