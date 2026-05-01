"""
experiments/unlabeled_pipeline/species_report.py
-------------------------------------------------
Generate an interactive HTML species report for expert review.

For each cluster, shows:
  - All specimen images with per-image verdict controls
  - GBIF reference image(s) for the AI prediction and runner-up
  - Clustering confidence vs annotation confidence (separate scores)
  - Expert can mark individual images correct/wrong/move to different species

The expert opens ONE HTML file in a browser and reviews everything visually.
No CSV digging, no image folder hunting.

Usage (called from run_unlabeled.py, or standalone):
    python experiments/unlabeled_pipeline/species_report.py \
        --annotations results/unlabeled_test/annotations.csv \
        --references results/unlabeled_test/reference_images \
        --output results/unlabeled_test/species_report.html
"""

import os
import sys
import base64
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


# -------------------------------------------------------
# Image encoding (embed images as base64 in HTML)
# -------------------------------------------------------

def _img_to_base64(path: str, max_size: int = 400) -> str:
    """Convert an image file to a base64 data URI for embedding in HTML."""
    if not os.path.exists(path):
        return ""
    try:
        from PIL import Image
        import io
        img = Image.open(path)
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def _img_tag(path: str, alt: str = "", max_size: int = 300,
             css_class: str = "") -> str:
    """Generate an <img> tag with base64 embedded image."""
    data_uri = _img_to_base64(path, max_size)
    if not data_uri:
        return f'<div class="no-image">No image<br><small>{alt}</small></div>'
    cls = f' class="{css_class}"' if css_class else ""
    return f'<img src="{data_uri}" alt="{alt}"{cls} title="{alt}">'


# -------------------------------------------------------
# HTML Report Generator
# -------------------------------------------------------

def generate_species_report(
    annotations_df: pd.DataFrame,
    reference_map: dict,
    output_path: str,
    config: dict = None,
    max_images_per_cluster: int = 20,
    embed_images: bool = True,
) -> str:
    """
    Generate an interactive HTML species report.

    Args:
        annotations_df:  DataFrame from run_annotation().
        reference_map:   Dict mapping species_name -> [ref_image_path, ...].
        output_path:     Where to save the HTML file.
        config:          Pipeline config (for metadata display).
        max_images_per_cluster: Max specimen images to show per cluster.
        embed_images:    If True, embed images as base64 (portable HTML).
                         If False, use relative file:// paths (faster generation).

    Returns:
        Path to the generated HTML file.
    """
    # Group by cluster
    clusters = defaultdict(list)
    for _, row in annotations_df.iterrows():
        cid = row["cluster_id"]
        clusters[cid].append(row)

    # Sort clusters: human_required first, then review, then auto_accept
    tier_priority = {"human_required": 0, "review": 1, "auto_accept": 2}

    def cluster_sort_key(cid):
        rows = clusters[cid]
        tier = rows[0]["confidence_tier"]
        return (tier_priority.get(tier, 3), -len(rows))

    sorted_cids = sorted(
        [c for c in clusters.keys() if c >= 0],
        key=cluster_sort_key
    )

    # Noise points (cluster_id = -1)
    noise_rows = clusters.get(-1, [])

    # Stats
    total_images = len(annotations_df)
    n_clusters = len(sorted_cids)
    tier_counts = annotations_df["confidence_tier"].value_counts().to_dict()

    # Build HTML
    html_parts = [_html_header(n_clusters, total_images, tier_counts, config)]

    for cid in sorted_cids:
        rows = clusters[cid]
        html_parts.append(_render_cluster(
            cid, rows, reference_map, max_images_per_cluster, embed_images
        ))

    # Noise section
    if noise_rows:
        html_parts.append(_render_noise(noise_rows, embed_images))

    # Export controls
    html_parts.append(_html_export_section())
    html_parts.append(_html_footer())

    html = "\n".join(html_parts)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"    [report] Species report saved to {output_path}")
    return output_path


# -------------------------------------------------------
# HTML building blocks
# -------------------------------------------------------

def _html_header(n_clusters, total_images, tier_counts, config):
    region = ""
    n_candidates = ""
    if config:
        gbif_cfg = config.get("gbif", {})
        region = gbif_cfg.get("region", "unknown")
        n_candidates = gbif_cfg.get("n_candidates", "?")

    auto = tier_counts.get("auto_accept", 0)
    review = tier_counts.get("review", 0)
    human = tier_counts.get("human_required", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Species Identification Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
         background: #f5f5f5; color: #333; padding: 20px; }}

  /* Header */
  .header {{ background: linear-gradient(135deg, #2c3e50, #3498db);
             color: white; padding: 30px; border-radius: 12px;
             margin-bottom: 25px; }}
  .header h1 {{ font-size: 24px; margin-bottom: 10px; }}
  .header .stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px; }}
  .stat-box {{ background: rgba(255,255,255,0.15); padding: 12px 18px;
               border-radius: 8px; text-align: center; min-width: 120px; }}
  .stat-box .number {{ font-size: 28px; font-weight: bold; }}
  .stat-box .label {{ font-size: 12px; opacity: 0.9; margin-top: 4px; }}

  /* Filter bar */
  .filter-bar {{ background: white; padding: 15px 20px; border-radius: 10px;
                 margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                 display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .filter-btn {{ padding: 8px 16px; border: 2px solid #ddd; border-radius: 20px;
                 background: white; cursor: pointer; font-size: 13px;
                 transition: all 0.2s; }}
  .filter-btn:hover {{ border-color: #3498db; }}
  .filter-btn.active {{ background: #3498db; color: white; border-color: #3498db; }}

  /* Cluster cards */
  .cluster-card {{ background: white; border-radius: 12px; margin-bottom: 20px;
                   box-shadow: 0 2px 12px rgba(0,0,0,0.08); overflow: hidden;
                   transition: all 0.3s; }}
  .cluster-card.tier-human_required {{ border-left: 5px solid #e74c3c; }}
  .cluster-card.tier-review {{ border-left: 5px solid #f39c12; }}
  .cluster-card.tier-auto_accept {{ border-left: 5px solid #27ae60; }}

  .cluster-header {{ padding: 18px 22px; display: flex; justify-content: space-between;
                     align-items: center; cursor: pointer; }}
  .cluster-header:hover {{ background: #f8f9fa; }}
  .cluster-title {{ font-size: 18px; font-weight: 600; }}
  .cluster-meta {{ display: flex; gap: 12px; align-items: center; }}
  .badge {{ padding: 4px 12px; border-radius: 12px; font-size: 12px;
            font-weight: 600; text-transform: uppercase; }}
  .badge-accept {{ background: #d4edda; color: #155724; }}
  .badge-review {{ background: #fff3cd; color: #856404; }}
  .badge-human {{ background: #f8d7da; color: #721c24; }}

  .cluster-body {{ padding: 0 22px 22px; }}
  .collapsed .cluster-body {{ display: none; }}

  /* Two-column layout: specimens | references */
  .comparison {{ display: flex; gap: 25px; flex-wrap: wrap; }}
  .specimens {{ flex: 1; min-width: 300px; }}
  .reference {{ flex: 0 0 300px; }}
  .section-label {{ font-size: 13px; font-weight: 600; color: #666;
                    text-transform: uppercase; letter-spacing: 1px;
                    margin-bottom: 10px; border-bottom: 2px solid #eee;
                    padding-bottom: 5px; }}

  /* Per-image cards with individual controls */
  .image-grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .img-card {{ position: relative; display: inline-block; }}
  .img-card img {{ width: 140px; height: 140px; object-fit: cover;
                   border-radius: 8px; border: 3px solid #eee;
                   transition: all 0.2s; cursor: pointer; }}
  .img-card img:hover {{ transform: scale(1.8); z-index: 10;
                         position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
  .img-card.marked-correct img {{ border-color: #27ae60; }}
  .img-card.marked-wrong img {{ border-color: #e74c3c; }}
  .img-card.marked-moved img {{ border-color: #9b59b6; }}
  .img-card .img-controls {{ display: none; position: absolute; bottom: -30px;
                             left: 0; right: 0; text-align: center;
                             background: rgba(255,255,255,0.95); border-radius: 6px;
                             padding: 3px; z-index: 5; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
  .img-card:hover .img-controls {{ display: flex; gap: 2px; justify-content: center; }}
  .img-ctrl-btn {{ width: 24px; height: 24px; border: none; border-radius: 4px;
                   cursor: pointer; font-size: 11px; padding: 0; transition: all 0.15s; }}
  .img-ctrl-btn.btn-ok {{ background: #d4edda; color: #155724; }}
  .img-ctrl-btn.btn-ok:hover {{ background: #27ae60; color: white; }}
  .img-ctrl-btn.btn-no {{ background: #f8d7da; color: #721c24; }}
  .img-ctrl-btn.btn-no:hover {{ background: #e74c3c; color: white; }}
  .img-ctrl-btn.btn-mv {{ background: #e8daef; color: #6c3483; }}
  .img-ctrl-btn.btn-mv:hover {{ background: #9b59b6; color: white; }}
  .img-filename {{ font-size: 9px; color: #999; text-align: center;
                   max-width: 140px; overflow: hidden; text-overflow: ellipsis;
                   white-space: nowrap; margin-top: 2px; }}

  /* Reference images */
  .ref-images {{ display: flex; flex-direction: column; gap: 8px; }}
  .ref-images img {{ width: 220px; height: 220px; object-fit: cover;
                     border-radius: 8px; border: 3px solid #3498db; }}
  .ref-images img:hover {{ transform: scale(1.8); z-index: 10;
                           position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
  .ref-label {{ font-size: 11px; color: #666; font-style: italic; }}

  /* Confidence split display */
  .confidence-split {{ display: flex; gap: 15px; margin-top: 15px; flex-wrap: wrap; }}
  .conf-box {{ flex: 1; min-width: 200px; padding: 12px; border-radius: 8px; }}
  .conf-box.clustering {{ background: #eaf4fe; border: 1px solid #bee0fc; }}
  .conf-box.annotation {{ background: #fef5ea; border: 1px solid #fcd9a8; }}
  .conf-box h4 {{ font-size: 13px; margin-bottom: 8px; color: #444; }}
  .conf-meter {{ height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden;
                 margin: 4px 0; }}
  .conf-meter-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .conf-meter-fill.good {{ background: #27ae60; }}
  .conf-meter-fill.medium {{ background: #f39c12; }}
  .conf-meter-fill.bad {{ background: #e74c3c; }}
  .conf-detail {{ font-size: 12px; color: #666; margin: 3px 0; }}

  /* Verdict section */
  .verdict-section {{ margin-top: 15px; padding: 12px; background: #f0f7ff;
                      border-radius: 8px; }}
  .verdict-row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
                  margin-bottom: 8px; }}
  .verdict-btn {{ padding: 8px 20px; border: 2px solid #ddd; border-radius: 8px;
                  background: white; cursor: pointer; font-size: 14px;
                  transition: all 0.2s; }}
  .verdict-btn:hover {{ border-color: #3498db; }}
  .verdict-btn.correct {{ background: #d4edda; border-color: #27ae60; }}
  .verdict-btn.wrong {{ background: #f8d7da; border-color: #e74c3c; }}
  .verdict-btn.not-listed {{ background: #e8daef; border-color: #9b59b6; }}
  .correction-input {{ padding: 8px 12px; border: 2px solid #ddd; border-radius: 8px;
                       font-size: 14px; width: 280px; display: none; }}
  .correction-input.visible {{ display: inline-block; }}
  .notes-input {{ padding: 8px 12px; border: 2px solid #ddd; border-radius: 8px;
                  font-size: 13px; width: 220px; }}
  .img-count-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
                      font-size: 11px; margin-left: 6px; }}
  .img-count-badge.ok {{ background: #d4edda; color: #155724; }}
  .img-count-badge.wrong {{ background: #f8d7da; color: #721c24; }}
  .img-count-badge.moved {{ background: #e8daef; color: #6c3483; }}

  .no-image {{ width: 140px; height: 140px; background: #eee; border-radius: 8px;
               display: flex; align-items: center; justify-content: center;
               color: #999; font-size: 12px; text-align: center; }}

  /* Noise section */
  .noise-section {{ background: #fff5f5; border-radius: 12px; padding: 22px;
                    margin-bottom: 20px; border: 2px dashed #e74c3c; }}
  .noise-section h3 {{ color: #e74c3c; margin-bottom: 10px; }}

  /* Export */
  .export-section {{ background: white; padding: 20px; border-radius: 12px;
                     margin-top: 30px; text-align: center;
                     box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .export-btn {{ padding: 12px 30px; background: #27ae60; color: white;
                 border: none; border-radius: 8px; font-size: 16px;
                 cursor: pointer; transition: all 0.2s; margin: 5px; }}
  .export-btn:hover {{ background: #219a52; }}
  .export-btn.secondary {{ background: #3498db; }}
  .export-btn.secondary:hover {{ background: #2980b9; }}
</style>
</head>
<body>

<div class="header">
  <h1>Species Identification Report</h1>
  <p>BioCLIP-2 automated identification with uncertainty quantification</p>
  <div class="stats">
    <div class="stat-box">
      <div class="number">{total_images}</div>
      <div class="label">Total Images</div>
    </div>
    <div class="stat-box">
      <div class="number">{n_clusters}</div>
      <div class="label">Species Found</div>
    </div>
    <div class="stat-box">
      <div class="number" style="color: #a8e6cf">{auto}</div>
      <div class="label">Auto-accepted</div>
    </div>
    <div class="stat-box">
      <div class="number" style="color: #ffd93d">{review}</div>
      <div class="label">Need Review</div>
    </div>
    <div class="stat-box">
      <div class="number" style="color: #ff6b6b">{human}</div>
      <div class="label">Human Required</div>
    </div>
    <div class="stat-box">
      <div class="number">{region}</div>
      <div class="label">Region</div>
    </div>
    <div class="stat-box">
      <div class="number">{n_candidates}</div>
      <div class="label">GBIF Candidates</div>
    </div>
  </div>
</div>

<div class="filter-bar">
  <span style="font-weight:600; margin-right:8px;">Filter:</span>
  <button class="filter-btn active" onclick="filterClusters('all')">All</button>
  <button class="filter-btn" onclick="filterClusters('human_required')">Human Required</button>
  <button class="filter-btn" onclick="filterClusters('review')">Review</button>
  <button class="filter-btn" onclick="filterClusters('auto_accept')">Auto-accepted</button>
</div>
"""


def _render_cluster(cid, rows, reference_map, max_images, embed_images):
    row0 = rows[0]
    species = row0["ai_annotation"]
    second = row0.get("ai_second_choice", "")
    tier = row0["confidence_tier"]
    margin = row0.get("margin", 0)
    combined = row0.get("combined_uncertainty", 0)
    semantic = row0.get("semantic_uncertainty", 0)
    spatial = row0.get("spatial_uncertainty", 0)
    agreement = row0.get("cluster_agreement", 0)
    taxonomic = row0.get("taxonomic_consistency", 0)
    n_images = len(rows)

    # Badge
    if tier == "human_required":
        badge_cls = "badge-human"
        badge_text = "Human Required"
    elif tier == "review":
        badge_cls = "badge-review"
        badge_text = "Review"
    else:
        badge_cls = "badge-accept"
        badge_text = "Auto-accepted"

    # --- Per-image specimen cards with individual controls ---
    specimen_html = ""
    for i, r in enumerate(rows[:max_images]):
        path = r["file_path"]
        fname = os.path.basename(path)
        img_id = f"img-{cid}-{i}"

        if embed_images:
            img_html = _img_tag(path, fname, 300)
        else:
            img_html = (f'<img src="file:///{path}" alt="{fname}">')

        specimen_html += f"""
        <div class="img-card" id="{img_id}" data-cluster="{cid}" data-idx="{i}"
             data-filename="{fname}" data-filepath="{path}">
          {img_html}
          <div class="img-controls">
            <button class="img-ctrl-btn btn-ok" title="Correct"
                    onclick="markImage('{img_id}','correct')">&#10003;</button>
            <button class="img-ctrl-btn btn-no" title="Wrong species"
                    onclick="markImage('{img_id}','wrong')">&#10007;</button>
            <button class="img-ctrl-btn btn-mv" title="Move to different species"
                    onclick="markImage('{img_id}','move')">&#8594;</button>
          </div>
          <div class="img-filename">{fname}</div>
        </div>"""

    if n_images > max_images:
        specimen_html += (f'<div class="no-image">+{n_images - max_images} '
                          f'more images</div>')

    # --- Reference images ---
    ref_html = ""
    ref_paths = reference_map.get(species, [])
    if ref_paths:
        for rp in ref_paths[:2]:
            if embed_images:
                ref_html += _img_tag(rp, f"GBIF: {species}", 400, "ref-img")
            else:
                ref_html += (f'<img src="file:///{rp}" '
                             f'alt="GBIF: {species}" class="ref-img">')
        ref_html += f'<div class="ref-label">GBIF reference image</div>'
    else:
        ref_html = ('<div class="no-image" style="width:220px;height:220px;">'
                    'No GBIF reference<br>available<br><small>(species may not '
                    'be in candidate list)</small></div>')

    # Second choice reference
    ref2_html = ""
    second_display = second if second and str(second) != "nan" else "N/A"
    if second and str(second) != "nan":
        ref2_paths = reference_map.get(second, [])
        if ref2_paths:
            ref2_img = ""
            for rp in ref2_paths[:1]:
                if embed_images:
                    ref2_img += _img_tag(rp, f"GBIF: {second}", 400, "ref-img")
                else:
                    ref2_img += (f'<img src="file:///{rp}" '
                                 f'alt="GBIF: {second}" class="ref-img">')
            ref2_html = f"""
            <div style="margin-top:15px;">
              <div class="section-label">Runner-up: {second}</div>
              {ref2_img}
              <div class="ref-label">GBIF reference image</div>
            </div>"""
        else:
            ref2_html = f"""
            <div style="margin-top:15px;">
              <div class="section-label">Runner-up: {second}</div>
              <div class="no-image" style="width:150px;height:100px;font-size:10px;">
                No reference<br>for runner-up</div>
            </div>"""

    # --- Confidence split: Clustering vs Annotation ---
    # Clustering confidence: based on spatial (HDBSCAN probability) + agreement
    cluster_conf = 1.0 - (spatial * 0.5 + agreement * 0.5)
    cluster_conf_pct = max(0, min(100, cluster_conf * 100))
    cluster_conf_cls = ("good" if cluster_conf >= 0.7 else
                        "medium" if cluster_conf >= 0.4 else "bad")

    # Annotation confidence: based on semantic (text margin) + taxonomic
    annot_conf = 1.0 - (semantic * 0.6 + taxonomic * 0.4)
    annot_conf_pct = max(0, min(100, annot_conf * 100))
    annot_conf_cls = ("good" if annot_conf >= 0.7 else
                      "medium" if annot_conf >= 0.4 else "bad")

    # Determine if species might not be in candidate list
    no_ref_warning = ""
    if not ref_paths and annot_conf_pct < 40:
        no_ref_warning = """
        <div style="background:#fff3cd; border:1px solid #ffc107; border-radius:6px;
                    padding:8px 12px; margin-top:10px; font-size:13px;">
          <strong>Note:</strong> No GBIF reference found and annotation confidence
          is low. The true species may not be in the candidate list.
          Use "Species not listed" below to enter the correct name.
        </div>"""

    return f"""
<div class="cluster-card tier-{tier}" data-tier="{tier}" id="cluster-{cid}">
  <div class="cluster-header" onclick="toggleCluster({cid})">
    <div>
      <span class="cluster-title">Cluster {cid} &mdash; <em>{species}</em></span>
      <span style="color:#888; margin-left:10px;">({n_images} images)</span>
      <span class="img-count-badge ok" id="ok-count-{cid}" style="display:none"></span>
      <span class="img-count-badge wrong" id="wrong-count-{cid}" style="display:none"></span>
      <span class="img-count-badge moved" id="moved-count-{cid}" style="display:none"></span>
    </div>
    <div class="cluster-meta">
      <span class="badge {badge_cls}">{badge_text}</span>
      <span style="font-size:13px; color:#888;">margin: {margin:.3f}</span>
    </div>
  </div>
  <div class="cluster-body">
    <div class="comparison">
      <div class="specimens">
        <div class="section-label">Your Specimen Images
          <span style="font-weight:normal; font-size:11px; text-transform:none;">
            &mdash; hover for per-image controls
          </span>
        </div>
        <div class="image-grid">
          {specimen_html}
        </div>
      </div>
      <div class="reference">
        <div class="section-label">GBIF Reference: {species}</div>
        <div class="ref-images">
          {ref_html}
        </div>
        {ref2_html}
      </div>
    </div>

    <!-- Dual confidence display -->
    <div class="confidence-split">
      <div class="conf-box clustering">
        <h4>Clustering Confidence</h4>
        <div style="font-size:11px; color:#555; margin-bottom:4px;">
          "Do these images belong together?"</div>
        <div class="conf-meter">
          <div class="conf-meter-fill {cluster_conf_cls}"
               style="width:{cluster_conf_pct:.0f}%"></div>
        </div>
        <div class="conf-detail">Spatial cohesion: {1-spatial:.2f}</div>
        <div class="conf-detail">Cluster agreement: {1-agreement:.2f}</div>
        <div class="conf-detail"><strong>Overall: {cluster_conf:.0%}</strong></div>
      </div>
      <div class="conf-box annotation">
        <h4>Annotation Confidence</h4>
        <div style="font-size:11px; color:#555; margin-bottom:4px;">
          "Is the species name correct?"</div>
        <div class="conf-meter">
          <div class="conf-meter-fill {annot_conf_cls}"
               style="width:{annot_conf_pct:.0f}%"></div>
        </div>
        <div class="conf-detail">Text margin: {margin:.4f}</div>
        <div class="conf-detail">Taxonomic consistency: {1-taxonomic:.2f}</div>
        <div class="conf-detail"><strong>Overall: {annot_conf:.0%}</strong></div>
      </div>
    </div>

    {no_ref_warning}

    <!-- Expert verdict -->
    <div class="verdict-section">
      <div class="verdict-row">
        <strong>Cluster verdict:</strong>
        <button class="verdict-btn" onclick="setVerdict({cid},'correct',this)">
          All Correct</button>
        <button class="verdict-btn" onclick="setVerdict({cid},'wrong',this)">
          Wrong Name</button>
        <button class="verdict-btn" onclick="setVerdict({cid},'not_listed',this)">
          Species Not Listed</button>
        <input type="text" class="correction-input" id="correction-{cid}"
               placeholder="Enter correct species name..."
               onchange="saveCorrection({cid})">
      </div>
      <div class="verdict-row">
        <span style="font-size:12px; color:#888;">Notes:</span>
        <input type="text" class="notes-input" id="notes-{cid}"
               placeholder="Optional notes for this cluster...">
      </div>
    </div>
  </div>
</div>
"""


def _render_noise(noise_rows, embed_images):
    n = len(noise_rows)
    images_html = ""
    for i, r in enumerate(noise_rows[:30]):
        path = r["file_path"]
        fname = os.path.basename(path)
        img_id = f"img-noise-{i}"

        if embed_images:
            img_html = _img_tag(path, fname, 200)
        else:
            img_html = f'<img src="file:///{path}" alt="{fname}">'

        images_html += f"""
        <div class="img-card" id="{img_id}" data-cluster="-1" data-idx="{i}"
             data-filename="{fname}">
          {img_html}
          <div class="img-controls">
            <button class="img-ctrl-btn btn-mv" title="Assign species"
                    onclick="markImage('{img_id}','move')" style="width:auto; padding:0 8px;">
              Identify</button>
          </div>
          <div class="img-filename">{fname}</div>
        </div>"""

    extra = ""
    if n > 30:
        extra = f'<div class="no-image">+{n-30} more</div>'

    return f"""
<div class="noise-section" id="noise-section">
  <h3>Noise Points ({n} images)</h3>
  <p style="margin-bottom:10px; color:#666;">
    These images could not be assigned to any cluster. They may be unique species,
    poor-quality images, or outliers. Hover over each image to assign a species.
  </p>
  <div class="image-grid">
    {images_html}
    {extra}
  </div>
</div>
"""


def _html_export_section():
    return """
<div class="export-section">
  <h3 style="margin-bottom:15px;">Export Expert Review</h3>
  <p style="margin-bottom:15px; color:#666;">
    After reviewing clusters and individual images, export your verdicts.
    The CSV includes both cluster-level and per-image decisions.
  </p>
  <button class="export-btn" onclick="exportCSV()">Download Cluster Review CSV</button>
  <button class="export-btn secondary" onclick="exportImageCSV()">Download Per-Image CSV</button>
  <div style="margin-top:15px;">
    <span id="review-progress" style="font-size:14px; color:#666;"></span>
  </div>
</div>
"""


def _html_footer():
    return """
<script>
// ---- State ----
const verdicts = {};       // cluster-level: {cid: {verdict, correction, notes}}
const imageMarks = {};     // per-image: {imgId: {mark, species}}

// ---- Toggle cluster ----
function toggleCluster(cid) {
  const card = document.getElementById('cluster-' + cid);
  card.classList.toggle('collapsed');
}

// ---- Filter clusters ----
function filterClusters(tier) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.cluster-card').forEach(card => {
    if (tier === 'all' || card.dataset.tier === tier) {
      card.style.display = '';
    } else {
      card.style.display = 'none';
    }
  });
}

// ---- Cluster-level verdict ----
function setVerdict(cid, verdict, btn) {
  btn.parentElement.querySelectorAll('.verdict-btn').forEach(b => {
    b.classList.remove('correct', 'wrong', 'not-listed');
  });
  btn.classList.add(verdict === 'not_listed' ? 'not-listed' : verdict);

  verdicts[cid] = { verdict: verdict, correction: '', notes: '' };

  const input = document.getElementById('correction-' + cid);
  if (verdict === 'wrong' || verdict === 'not_listed') {
    input.classList.add('visible');
    input.placeholder = verdict === 'not_listed'
      ? 'Type the correct species name (not in GBIF list)...'
      : 'Enter correct species name...';
    input.focus();
  } else {
    input.classList.remove('visible');
  }
  updateProgress();
}

function saveCorrection(cid) {
  const input = document.getElementById('correction-' + cid);
  if (verdicts[cid]) {
    verdicts[cid].correction = input.value;
  }
}

// ---- Per-image marking ----
function markImage(imgId, mark) {
  const card = document.getElementById(imgId);
  card.classList.remove('marked-correct', 'marked-wrong', 'marked-moved');

  if (mark === 'correct') {
    card.classList.add('marked-correct');
    imageMarks[imgId] = { mark: 'correct', species: '' };
  } else if (mark === 'wrong') {
    card.classList.add('marked-wrong');
    imageMarks[imgId] = { mark: 'wrong', species: '' };
  } else if (mark === 'move') {
    const species = prompt('Enter the correct species name for this image:');
    if (species) {
      card.classList.add('marked-moved');
      imageMarks[imgId] = { mark: 'move', species: species };
    }
  }
  updateImageCounts(card.dataset.cluster);
  updateProgress();
}

function updateImageCounts(cid) {
  let ok = 0, wrong = 0, moved = 0;
  document.querySelectorAll('.img-card[data-cluster="' + cid + '"]').forEach(card => {
    const m = imageMarks[card.id];
    if (m) {
      if (m.mark === 'correct') ok++;
      else if (m.mark === 'wrong') wrong++;
      else if (m.mark === 'move') moved++;
    }
  });
  const showBadge = (id, count, label) => {
    const el = document.getElementById(id);
    if (el) {
      if (count > 0) { el.textContent = count + ' ' + label; el.style.display = ''; }
      else { el.style.display = 'none'; }
    }
  };
  showBadge('ok-count-' + cid, ok, 'ok');
  showBadge('wrong-count-' + cid, wrong, 'wrong');
  showBadge('moved-count-' + cid, moved, 'moved');
}

function updateProgress() {
  const totalClusters = document.querySelectorAll('.cluster-card').length;
  const reviewed = Object.keys(verdicts).length;
  const totalImages = document.querySelectorAll('.img-card').length;
  const markedImages = Object.keys(imageMarks).length;
  const el = document.getElementById('review-progress');
  if (el) {
    el.textContent = `Progress: ${reviewed}/${totalClusters} clusters reviewed, `
                   + `${markedImages}/${totalImages} images marked`;
  }
}

// ---- Export cluster-level CSV ----
function exportCSV() {
  const rows = [['Cluster_ID', 'AI_Prediction', 'Expert_Verdict',
                 'Expert_Corrected_Label', 'Expert_Notes',
                 'Images_Correct', 'Images_Wrong', 'Images_Moved']];

  document.querySelectorAll('.cluster-card').forEach(card => {
    const cid = card.id.replace('cluster-', '');
    const title = card.querySelector('.cluster-title em');
    const species = title ? title.textContent : '';
    const v = verdicts[cid] || {};
    const notes = document.getElementById('notes-' + cid);

    // Count per-image marks
    let ok = 0, wrong = 0, moved = 0;
    document.querySelectorAll('.img-card[data-cluster="' + cid + '"]').forEach(ic => {
      const m = imageMarks[ic.id];
      if (m) {
        if (m.mark === 'correct') ok++;
        else if (m.mark === 'wrong') wrong++;
        else if (m.mark === 'move') moved++;
      }
    });

    rows.push([
      cid, species, v.verdict || 'not_reviewed',
      v.correction || '', notes ? notes.value : '',
      ok, wrong, moved
    ]);
  });

  downloadCSV(rows, 'expert_review_clusters.csv');
}

// ---- Export per-image CSV ----
function exportImageCSV() {
  const rows = [['Filename', 'Cluster_ID', 'AI_Prediction',
                 'Image_Verdict', 'Corrected_Species']];

  document.querySelectorAll('.img-card').forEach(card => {
    const filename = card.dataset.filename || '';
    const cid = card.dataset.cluster || '';
    const m = imageMarks[card.id] || {};

    // Get cluster's AI prediction
    let aiPred = '';
    const clusterCard = document.getElementById('cluster-' + cid);
    if (clusterCard) {
      const em = clusterCard.querySelector('.cluster-title em');
      if (em) aiPred = em.textContent;
    }

    if (m.mark) {
      rows.push([filename, cid, aiPred, m.mark, m.species || '']);
    }
  });

  if (rows.length <= 1) {
    alert('No individual images have been marked yet. Use the hover controls on each image first.');
    return;
  }
  downloadCSV(rows, 'expert_review_images.csv');
}

function downloadCSV(rows, filename) {
  const csv = rows.map(r =>
    r.map(c => '"' + String(c).replace(/"/g, '""') + '"').join(',')
  ).join('\\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Initialize progress
updateProgress();
</script>

<div style="text-align:center; padding:30px; color:#999; font-size:12px;">
  Generated by BioCLIP-2 Clustering Pipeline &mdash; Species Identification Report
</div>
</body>
</html>
"""


# -------------------------------------------------------
# Standalone usage
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate HTML species report from annotations."
    )
    parser.add_argument("--annotations", required=True,
                        help="Path to annotations CSV.")
    parser.add_argument("--references", default=None,
                        help="Path to reference_images directory.")
    parser.add_argument("--output", default="species_report.html",
                        help="Output HTML file path.")
    parser.add_argument("--no-embed", action="store_true",
                        help="Use file:// paths instead of base64 embedding.")
    args = parser.parse_args()

    df = pd.read_csv(args.annotations)

    # Build reference map from directory structure
    ref_map = {}
    if args.references and os.path.exists(args.references):
        for species_dir in os.listdir(args.references):
            sp_path = os.path.join(args.references, species_dir)
            if os.path.isdir(sp_path) and species_dir != "_references":
                name = species_dir.replace("_", " ")
                imgs = [os.path.join(sp_path, f) for f in sorted(os.listdir(sp_path))
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if imgs:
                    ref_map[name] = imgs

    path = generate_species_report(
        df, ref_map, args.output, embed_images=not args.no_embed
    )
    print(f"\nReport generated: {path}")
    print("Open in a web browser to review.")
