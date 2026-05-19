"""
src/reduction/reduction.py
--------------------------
Dimensionality reduction utilities for the clustering pipeline.

Single-path architecture
------------------------
One reduction method is applied uniformly to ALL clustering algorithms in a
pipeline run. The method is configured once in the YAML config and determines
the feature space that every algorithm receives.

Four modes are supported:

    none      -- no reduction; all algorithms receive the full 768D embeddings.
    pca       -- PCA: linear projection retaining the highest-variance axes.
    umap      -- UMAP: non-linear projection preserving local neighbourhood
                 topology.
    pca+umap  -- two-stage: PCA to an intermediate dimension first (set via
                 pca_components), then UMAP on the PCA output.

Choosing the best mode
----------------------
Run the appropriate analysis script before committing a value to the config:

    Labeled datasets   -> cli/reduction_analysis.py
                          Sweeps PCA/UMAP/PCA+UMAP dimensions and ranks them
                          by ARI, NMI, and Hungarian accuracy.

    Unlabeled datasets -> cli/reduction_analysis_unlabeled.py
                          Uses silhouette score (PCA path) and trustworthiness
                          + weighted silhouette (UMAP path) -- no ground-truth
                          labels needed.

Critical constraint
-------------------
Dimensionality reduction applies ONLY for clustering. Text-image cosine
similarity (annotation step) must always be computed in the original 768D
CLIP embedding space. Reducing embeddings before text matching destroys the
vision-language alignment learned during contrastive pretraining.

References
----------
Ding & He (2004)        -- PCA and K-Means optimize dual objectives
McInnes et al. (2018)   -- UMAP preserves local topological structure
Beyer et al. (1999)     -- distance concentration in high dimensions
"""

import warnings
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional


# -------------------------------------------------------
# PCA
# -------------------------------------------------------

def fit_pca(features: np.ndarray,
            n_components: int = 50,
            random_state: int = 42) -> Tuple[np.ndarray, PCA, float]:
    """
    Reduce features with PCA.

    Args:
        features:      Raw embeddings, shape (N, D).
        n_components:  Target dimensionality (default 50).
        random_state:  Random seed.

    Returns:
        reduced:   PCA-reduced features, shape (N, n_components).
        pca:       Fitted PCA object.
        variance:  Fraction of total variance retained.
    """
    n_components = min(n_components, features.shape[1], features.shape[0])

    print(f"  [reduction] PCA: {features.shape[1]}D -> {n_components}D ...")
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(features)

    variance = float(pca.explained_variance_ratio_.sum())
    print(f"  [reduction] PCA variance retained: {variance:.1%}")

    return reduced, pca, variance


# -------------------------------------------------------
# UMAP
# -------------------------------------------------------

def fit_umap(features: np.ndarray,
             n_components: int = 100,
             n_neighbors: int = 15,
             min_dist: float = 0.1,
             metric: str = "cosine",
             random_state: int = 42) -> Tuple[np.ndarray, object]:
    """
    Reduce features with UMAP.

    The n_components value here targets clustering quality, not visualization.
    Use larger values (e.g. 20-100) than typical 2D/3D visualization targets.

    Args:
        features:      Raw embeddings, shape (N, D).
        n_components:  Target dimensionality (default 100).
        n_neighbors:   Controls local vs. global structure balance (default 15).
        min_dist:      Minimum distance between points in embedded space
                       (default 0.1). Values near 0 collapse points tightly;
                       0.1 preserves small cluster structure.
        metric:        Distance metric in original space (default "cosine").
        random_state:  Random seed.

    Returns:
        reduced:   UMAP-reduced features, shape (N, n_components).
        reducer:   Fitted UMAP object.

    Raises:
        ImportError: If umap-learn is not installed.
    """
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError(
            "UMAP reduction requires: pip install umap-learn"
        )

    print(f"  [reduction] UMAP: {features.shape[1]}D -> {n_components}D "
          f"(neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}) ...")

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
# Config reader
# -------------------------------------------------------

def get_reduction_config(config: dict) -> dict:
    """
    Extract reduction settings from config, with sensible defaults.

    Expected config structure:

        reduction:
          method: pca           # none | pca | umap | pca+umap
          n_components: 50      # target dimension (ignored when method: none)
          n_neighbors: 15       # UMAP only
          min_dist: 0.1         # UMAP only
          metric: cosine        # UMAP only
          # pca_components: 100 # pca+umap only: intermediate PCA dimension

    All keys are optional. Defaults are applied when omitted.

    Returns:
        Flat dict with all reduction hyperparameters.
    """
    rcfg = config.get("reduction", {})

    return {
        "method":         rcfg.get("method", "pca"),
        "n_components":   rcfg.get("n_components", 50),
        "pca_components": rcfg.get("pca_components", 100),
        "n_neighbors":    rcfg.get("n_neighbors", 15),
        "min_dist":       rcfg.get("min_dist", 0.1),
        "metric":         rcfg.get("metric", "cosine"),
    }


# -------------------------------------------------------
# Unified reduction entry point
# -------------------------------------------------------

def reduce_features(features: np.ndarray,
                    config: dict,
                    random_state: int = 42) -> Tuple[np.ndarray, object]:
    """
    Apply the configured reduction to features.

    This is the single entry point used by all clustering methods. Every
    algorithm in the pipeline run receives the same reduced feature space.

    Supported methods (config key: reduction.method):
        none      -- return features unchanged (full 768D)
        pca       -- PCA to n_components dimensions
        umap      -- UMAP to n_components dimensions
        pca+umap  -- PCA to pca_components, then UMAP to n_components

    Args:
        features:      Raw L2-normalised embeddings, shape (N, D).
        config:        Loaded YAML config dict.
        random_state:  Random seed for reproducibility.

    Returns:
        reduced:   Reduced features, shape (N, d').
        reducer:   Fitted reducer object (PCA, UMAP reducer, or None when
                   method is 'none'). For pca+umap the UMAP reducer is returned.

    Raises:
        ValueError: If method is not one of the four supported values.
    """
    rcfg = get_reduction_config(config)
    method = rcfg["method"]

    if method == "none":
        print(f"  [reduction] No reduction — "
              f"using full {features.shape[1]}D embeddings.")
        return features, None

    elif method == "pca":
        reduced, pca, _ = fit_pca(
            features, rcfg["n_components"], random_state
        )
        return reduced, pca

    elif method == "umap":
        return fit_umap(
            features,
            n_components=rcfg["n_components"],
            n_neighbors=rcfg["n_neighbors"],
            min_dist=rcfg["min_dist"],
            metric=rcfg["metric"],
            random_state=random_state,
        )

    elif method == "pca+umap":
        pca_dim = rcfg["pca_components"]
        print(f"  [reduction] PCA+UMAP: "
              f"{features.shape[1]}D -> PCA-{pca_dim}D "
              f"-> UMAP-{rcfg['n_components']}D")
        pca_reduced, _, _ = fit_pca(features, pca_dim, random_state)
        return fit_umap(
            pca_reduced,
            n_components=rcfg["n_components"],
            n_neighbors=rcfg["n_neighbors"],
            min_dist=rcfg["min_dist"],
            metric=rcfg["metric"],
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unknown reduction method: '{method}'. "
            f"Supported: none, pca, umap, pca+umap"
        )


# -------------------------------------------------------
# Backward-compatibility shims
# -------------------------------------------------------
# Scripts that still import reduce_for_partition / reduce_for_density
# continue to work without modification. New code should call
# reduce_features() directly.

def reduce_for_partition(features: np.ndarray,
                         config: dict,
                         random_state: int = 42
                         ) -> Tuple[np.ndarray, Optional[PCA], float]:
    """Backward-compat shim. Use reduce_features() in new code."""
    reduced, reducer = reduce_features(features, config, random_state)
    return reduced, reducer, 1.0


def reduce_for_density(features: np.ndarray,
                       config: dict,
                       random_state: int = 42) -> Tuple[np.ndarray, object]:
    """Backward-compat shim. Use reduce_features() in new code."""
    return reduce_features(features, config, random_state)
