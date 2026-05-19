"""
sankey_flow.py
--------------
Generate a Sankey diagram showing how images flow through the annotation
pipeline: raw images -> clustering -> confidence tiers -> final disposition.

Reads the standard annotations CSV produced by run_unlabeled.py / annotate.py
and outputs an interactive HTML and a static PNG.

Usage
-----
    # Basic: read annotations CSV, write HTML beside it
    python experiments/unlabeled_pipeline/sankey_flow.py \\
        --annotations results/unlabeled_test_local/annotations.csv

    # Custom output paths
    python experiments/unlabeled_pipeline/sankey_flow.py \\
        --annotations results/unlabeled_test_local/annotations.csv \\
        --html results/unlabeled_test_local/sankey.html \\
        --png  results/unlabeled_test_local/sankey.png

    # If a merged review CSV exists, include the post-review outcome stage
    python experiments/unlabeled_pipeline/sankey_flow.py \\
        --annotations results/unlabeled_test_local/annotations.csv \\
        --merged     results/unlabeled_test_local/annotations_final.csv

PNG export requires the `kaleido` package: pip install kaleido
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("This script requires plotly: pip install plotly")


TIER_COLORS = {
    "auto_accept":     "#2ecc71",   # green
    "review":          "#f39c12",   # orange
    "human_required":  "#e74c3c",   # red
    "noise":           "#7f8c8d",   # grey
}

# Same colours but with 50% alpha, in rgba() form (plotly does not accept hex+alpha)
TIER_LINK_COLORS = {
    "auto_accept":     "rgba(46, 204, 113, 0.5)",
    "review":          "rgba(243, 156, 18, 0.5)",
    "human_required":  "rgba(231, 76,  60,  0.5)",
    "noise":           "rgba(127, 140, 141, 0.5)",
}

OUTCOME_COLORS = {
    "AI_kept":       "#27ae60",
    "human_verified":"#2980b9",
    "rejected":      "#c0392b",
}


def build_flow(annotations_path, merged_path=None):
    """
    Build the node and link lists for plotly Sankey.

    Returns dict with:
        labels        : list of node labels
        node_colors   : list of node colors
        sources       : list of source node indices
        targets       : list of target node indices
        values        : list of flow widths (image counts)
        link_colors   : list of link colors
        title         : suggested chart title
    """
    df = pd.read_csv(annotations_path)
    n_total = len(df)
    n_clusters = df["cluster_id"].nunique()
    n_noise = int((df["cluster_id"] == -1).sum())

    # ----- nodes -----
    # Layer 0: total images
    # Layer 1: clusters (aggregated) vs noise
    # Layer 2: confidence tiers
    # Layer 3 (optional): post-review outcome
    nodes = []
    colors = []

    def add_node(label, color):
        nodes.append(label)
        colors.append(color)
        return len(nodes) - 1

    n_total_idx = add_node(f"Input images\n({n_total})", "#34495e")
    n_clustered_idx = add_node(f"Clustered\n({n_total - n_noise} in {n_clusters} clusters)", "#3498db")
    n_noise_idx = add_node(f"Noise\n({n_noise})", TIER_COLORS["noise"]) if n_noise else None

    tier_counts = df["confidence_tier"].value_counts().to_dict()
    tier_indices = {}
    for tier in ("auto_accept", "review", "human_required"):
        count = tier_counts.get(tier, 0)
        tier_indices[tier] = add_node(
            f"{tier.replace('_', ' ').title()}\n({count})",
            TIER_COLORS[tier],
        )

    sources, targets, values, link_colors = [], [], [], []

    def add_link(src, tgt, val, color):
        if val <= 0:
            return
        sources.append(src)
        targets.append(tgt)
        values.append(int(val))
        link_colors.append(color)

    # Total -> Clustered / Noise
    add_link(n_total_idx, n_clustered_idx, n_total - n_noise, "#bdc3c7")
    if n_noise_idx is not None:
        add_link(n_total_idx, n_noise_idx, n_noise, TIER_LINK_COLORS["noise"])

    # Clustered -> tier (counted on non-noise rows)
    clustered_df = df[df["cluster_id"] != -1]
    for tier, t_idx in tier_indices.items():
        n = int((clustered_df["confidence_tier"] == tier).sum())
        add_link(n_clustered_idx, t_idx, n, TIER_LINK_COLORS[tier])

    # Noise -> human_required (noise points are always human-routed)
    if n_noise_idx is not None:
        n_noise_to_hr = int((df[df["cluster_id"] == -1]["confidence_tier"] == "human_required").sum())
        add_link(n_noise_idx, tier_indices["human_required"], n_noise_to_hr,
                 TIER_LINK_COLORS["noise"])
        # Any noise rows in other tiers (rare): also link them
        for tier, t_idx in tier_indices.items():
            if tier == "human_required":
                continue
            n = int((df[df["cluster_id"] == -1]["confidence_tier"] == tier).sum())
            add_link(n_noise_idx, t_idx, n, TIER_LINK_COLORS["noise"])

    # Optional: post-review outcomes (if merged CSV provided)
    if merged_path and Path(merged_path).exists():
        merged = pd.read_csv(merged_path)
        # Use annotation_source after merge to derive outcome
        outcome_indices = {}
        for outcome in ("AI_kept", "human_verified", "rejected"):
            n = int((merged["annotation_source"] == outcome).sum()) if "annotation_source" in merged else 0
            if n > 0:
                outcome_indices[outcome] = add_node(
                    f"{outcome.replace('_', ' ').title()}\n({n})",
                    OUTCOME_COLORS[outcome],
                )

        # Tier -> outcome (cross-tab)
        if "confidence_tier" in merged and "annotation_source" in merged:
            xt = pd.crosstab(merged["confidence_tier"], merged["annotation_source"])
            for tier, t_idx in tier_indices.items():
                if tier not in xt.index:
                    continue
                for outcome, o_idx in outcome_indices.items():
                    if outcome not in xt.columns:
                        continue
                    add_link(t_idx, o_idx, int(xt.loc[tier, outcome]),
                             TIER_LINK_COLORS[tier])

    return {
        "labels":      nodes,
        "node_colors": colors,
        "sources":     sources,
        "targets":     targets,
        "values":      values,
        "link_colors": link_colors,
        "title":       f"Annotation pipeline flow ({n_total} images, {n_clusters} clusters)",
    }


def render_sankey(flow, html_path=None, png_path=None, width=1100, height=600):
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=24,
            line=dict(color="black", width=0.5),
            label=flow["labels"],
            color=flow["node_colors"],
        ),
        link=dict(
            source=flow["sources"],
            target=flow["targets"],
            value=flow["values"],
            color=flow["link_colors"],
        ),
    )])
    fig.update_layout(
        title_text=flow["title"],
        font_size=12,
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    if html_path:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html_path)
        print(f"  HTML: {html_path}")

    if png_path:
        try:
            Path(png_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(png_path, scale=2)
            print(f"  PNG:  {png_path}")
        except Exception as e:
            print(f"  PNG export failed (install kaleido): {e}")

    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True,
                    help="Path to annotations.csv produced by run_unlabeled.py / annotate.py")
    ap.add_argument("--merged", default=None,
                    help="Optional path to annotations_final.csv (post-expert-review)")
    ap.add_argument("--html", default=None,
                    help="Output HTML path (default: alongside annotations.csv)")
    ap.add_argument("--png", default=None,
                    help="Output PNG path (default: alongside annotations.csv)")
    args = ap.parse_args()

    annotations = Path(args.annotations).resolve()
    if not annotations.exists():
        sys.exit(f"Not found: {annotations}")

    html_out = Path(args.html) if args.html else annotations.with_name("sankey.html")
    png_out  = Path(args.png)  if args.png  else annotations.with_name("sankey.png")

    print(f"Reading {annotations}")
    flow = build_flow(annotations, merged_path=args.merged)
    print(f"  Nodes: {len(flow['labels'])}, links: {len(flow['sources'])}")

    render_sankey(flow, html_path=str(html_out), png_path=str(png_out))


if __name__ == "__main__":
    main()
