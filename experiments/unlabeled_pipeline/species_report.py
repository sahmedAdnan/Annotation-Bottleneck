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


def _is_url(s):
    """Check if a string is an HTTP(S) URL."""
    return isinstance(s, str) and s.startswith("http")


def _ref_img_tag(ref, alt="", css_class="ref-img", embed_images=True):
    """Generate an <img> tag for a reference image (URL, local path, or base64)."""
    if _is_url(ref):
        return (f'<img src="{ref}" alt="{alt}" class="{css_class}" '
                f'title="{alt}" loading="lazy" '
                f'onerror="this.parentElement.innerHTML=\'<div class=no-image '
                f'style=width:220px;height:220px>Unavailable</div>\'">')
    if embed_images:
        return _img_tag(ref, alt, 400, css_class)
    return f'<img src="file:///{ref}" alt="{alt}" class="{css_class}">'


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
    all_candidate_names: list = None,
    candidate_mode: str = "gbif",
) -> str:
    """
    Generate an interactive HTML species report.

    Args:
        annotations_df:  DataFrame from run_annotation().
        reference_map:   Dict mapping species_name -> [ref_image_path_or_url, ...].
        output_path:     Where to save the HTML file.
        config:          Pipeline config (for metadata display).
        max_images_per_cluster: Max specimen images to show per cluster.
        embed_images:    If True, embed images as base64 (portable HTML).
                         If False, use relative file:// paths (faster generation).
        all_candidate_names: Full list of candidate species names.
        candidate_mode:  "gbif", "local", or "csv" - determines what
                         metadata is shown in the report header.

    Returns:
        Path to the generated HTML file.
    """
    # Group by cluster
    clusters = defaultdict(list)
    for _, row in annotations_df.iterrows():
        cid = row["cluster_id"]
        clusters[cid].append(row)

    # Sort clusters: human_required first, then review, then auto_accept.
    # Within each tier, sort by ascending margin so the most confusable
    # clusters (smallest gap between top-1 and top-2 species) appear first.
    tier_priority = {"human_required": 0, "review": 1, "auto_accept": 2}

    def cluster_sort_key(cid):
        rows = clusters[cid]
        tier = rows[0]["confidence_tier"]
        margin = rows[0].get("margin", 0.0)
        return (tier_priority.get(tier, 3), margin)

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

    # Determine ref label based on mode
    ref_source = {"gbif": "GBIF Reference", "local": "Dataset Reference",
                  "csv": "CSV Reference"}.get(candidate_mode, "Reference")

    # Build HTML
    html_parts = [_html_header(n_clusters, total_images, tier_counts, config,
                               candidate_mode)]

    for cid in sorted_cids:
        rows = clusters[cid]
        html_parts.append(_render_cluster(
            cid, rows, reference_map, max_images_per_cluster, embed_images,
            ref_source
        ))

    # Noise section
    if noise_rows:
        html_parts.append(_render_noise(noise_rows, embed_images))

    # Collect all species names for autocomplete (cluster names + candidates)
    all_species = sorted(set(
        [rows[0]["ai_annotation"] for rows in clusters.values() if rows]
    ))
    if all_candidate_names:
        all_species = sorted(set(all_species) | set(all_candidate_names))

    # Build lightweight annotations JSON for finalize export
    ann_records = []
    cols_to_keep = ["file_path", "cluster_id", "ai_annotation",
                    "ai_second_choice", "confidence", "margin",
                    "semantic_uncertainty", "spatial_uncertainty",
                    "cluster_agreement", "taxonomic_consistency",
                    "combined_uncertainty", "confidence_tier",
                    "annotation_source"]
    for _, row in annotations_df.iterrows():
        rec = {}
        for c in cols_to_keep:
            val = row.get(c, "")
            if pd.isna(val):
                val = ""
            elif isinstance(val, float):
                val = round(val, 6)
            rec[c] = val
        ann_records.append(rec)

    # Export controls
    html_parts.append(_html_export_section())
    html_parts.append(_html_footer(all_species, ann_records))

    html = "\n".join(html_parts)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"    [report] Species report saved to {output_path}")
    return output_path


# -------------------------------------------------------
# HTML building blocks
# -------------------------------------------------------

def _build_metadata(config, candidate_mode):
    """Extract display metadata based on candidate mode."""
    meta = {
        "source_label": {"gbif": "GBIF API", "local": "Local Folders",
                         "csv": "CSV Taxonomy"}.get(candidate_mode, candidate_mode),
        "region": "",
        "n_candidates": "",
        "model": "",
        "clustering": "",
    }
    if not config:
        return meta

    model_cfg = config.get("model", {})
    meta["model"] = model_cfg.get("name", "bioclip2").upper()

    clust_cfg = config.get("clustering", {})
    meta["clustering"] = clust_cfg.get("method", "hdbscan").upper()

    if candidate_mode == "gbif":
        gbif_cfg = config.get("gbif", {})
        meta["region"] = gbif_cfg.get("region", "")
        meta["n_candidates"] = gbif_cfg.get("n_candidates", "")
    elif candidate_mode == "local":
        meta["region"] = "Dataset folders"
    elif candidate_mode == "csv":
        meta["region"] = "CSV file"

    return meta


def _html_header(n_clusters, total_images, tier_counts, config,
                 candidate_mode="gbif"):
    meta = _build_metadata(config, candidate_mode)

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
  .search-input {{ padding: 8px 14px; border: 2px solid #ddd; border-radius: 20px;
                   font-size: 13px; width: 220px; outline: none;
                   margin-left: auto; }}
  .search-input:focus {{ border-color: #3498db; }}

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
  .image-grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .img-card {{ display: inline-flex; flex-direction: column; align-items: center; }}
  .img-card img {{ width: 180px; height: 180px; object-fit: cover;
                   border-radius: 8px; border: 3px solid #eee;
                   transition: all 0.2s; cursor: pointer; }}
  .img-card img:hover {{ transform: scale(1.8); z-index: 10;
                         position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
  .img-card.marked-correct img {{ border-color: #27ae60; }}
  .img-card.marked-wrong img {{ border-color: #e74c3c; }}
  .img-card.marked-moved {{ display: none; }}
  .img-card .img-controls {{ display: flex; gap: 4px; justify-content: center;
                             margin-top: 6px; }}
  .img-ctrl-btn {{ width: 36px; height: 28px; border: none; border-radius: 5px;
                   cursor: pointer; font-size: 15px; padding: 0; transition: all 0.15s; }}
  .img-ctrl-btn.btn-ok {{ background: #d4edda; color: #155724; }}
  .img-ctrl-btn.btn-ok:hover {{ background: #27ae60; color: white; }}
  .img-ctrl-btn.btn-no {{ background: #f8d7da; color: #721c24; }}
  .img-ctrl-btn.btn-no:hover {{ background: #e74c3c; color: white; }}
  .img-ctrl-btn.btn-mv {{ background: #e8daef; color: #6c3483; }}
  .img-ctrl-btn.btn-mv:hover {{ background: #9b59b6; color: white; }}
  .img-filename {{ font-size: 9px; color: #999; text-align: center;
                   max-width: 180px; overflow: hidden; text-overflow: ellipsis;
                   white-space: nowrap; margin-top: 2px; }}

  /* Reference images */
  .ref-images {{ display: flex; flex-direction: column; gap: 8px; }}
  .ref-images img {{ width: 180px; height: 180px; object-fit: cover;
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

  .no-image {{ width: 180px; height: 180px; background: #eee; border-radius: 8px;
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

  /* Move modal */
  .modal-overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
                    z-index: 1000; justify-content: center; align-items: center; }}
  .modal-overlay.open {{ display: flex; }}
  .modal {{ background: white; border-radius: 12px; width: 480px; max-height: 80vh;
            box-shadow: 0 8px 40px rgba(0,0,0,0.25); overflow: hidden;
            display: flex; flex-direction: column; }}
  .modal-head {{ padding: 18px 22px; border-bottom: 1px solid #eee;
                 display: flex; justify-content: space-between; align-items: center; }}
  .modal-head h3 {{ font-size: 16px; margin: 0; }}
  .modal-close {{ background: none; border: none; font-size: 22px;
                  cursor: pointer; color: #888; padding: 0 4px; }}
  .modal-close:hover {{ color: #333; }}
  .modal-body {{ padding: 16px 22px; overflow-y: auto; flex: 1; }}
  .modal-search {{ width: 100%; padding: 10px 14px; border: 2px solid #ddd;
                   border-radius: 8px; font-size: 14px; outline: none;
                   margin-bottom: 10px; }}
  .modal-search:focus {{ border-color: #3498db; }}
  .modal-list {{ max-height: 320px; overflow-y: auto; }}
  .modal-item {{ padding: 10px 14px; cursor: pointer; border-radius: 6px;
                 font-size: 14px; display: flex; justify-content: space-between;
                 align-items: center; }}
  .modal-item:hover {{ background: #eef5ff; }}
  .modal-item .item-source {{ font-size: 11px; color: #999; }}
  .modal-item.noise {{ color: #e74c3c; font-weight: 600; border-top: 1px solid #eee;
                       margin-top: 4px; padding-top: 12px; }}
  .modal-preview {{ width: 60px; height: 60px; object-fit: cover; border-radius: 6px;
                    border: 2px solid #ddd; margin-right: 14px; flex-shrink: 0; }}

  /* Undo bar */
  .undo-bar {{ position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
               background: #333; color: white; padding: 10px 20px; border-radius: 10px;
               display: none; align-items: center; gap: 14px; z-index: 900;
               box-shadow: 0 4px 20px rgba(0,0,0,0.3); font-size: 14px; }}
  .undo-bar.visible {{ display: flex; }}
  .undo-bar button {{ background: rgba(255,255,255,0.2); border: none; color: white;
                      padding: 6px 16px; border-radius: 6px; cursor: pointer;
                      font-size: 13px; transition: background 0.15s; }}
  .undo-bar button:hover {{ background: rgba(255,255,255,0.35); }}
  .undo-bar button:disabled {{ opacity: 0.3; cursor: default; }}
  .undo-bar button:disabled:hover {{ background: rgba(255,255,255,0.2); }}
  .undo-bar .undo-info {{ max-width: 350px; overflow: hidden; text-overflow: ellipsis;
                          white-space: nowrap; }}
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
      <div class="number" style="font-size:16px;">{meta['source_label']}</div>
      <div class="label">Source</div>
    </div>
    <div class="stat-box">
      <div class="number" style="font-size:16px;">{meta['model']}</div>
      <div class="label">Model</div>
    </div>
    <div class="stat-box">
      <div class="number" style="font-size:16px;">{meta['clustering']}</div>
      <div class="label">Clustering</div>
    </div>
    <div class="stat-box">
      <div class="number" style="font-size:16px;">{meta['region']}</div>
      <div class="label">Region</div>
    </div>
  </div>
</div>

<div class="filter-bar">
  <span style="font-weight:600; margin-right:8px;">Filter:</span>
  <button class="filter-btn active" onclick="filterClusters('all')">All</button>
  <button class="filter-btn" onclick="filterClusters('human_required')">Human Required</button>
  <button class="filter-btn" onclick="filterClusters('review')">Review</button>
  <button class="filter-btn" onclick="filterClusters('auto_accept')">Auto-accepted</button>
  <input type="text" class="search-input" placeholder="Search species..."
         oninput="searchClusters(this.value)">
</div>
"""


def _render_cluster(cid, rows, reference_map, max_images, embed_images,
                    ref_source="GBIF Reference"):
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
            esc = path.replace("\\", "/")
            img_html = f'<img src="file:///{esc}" alt="{fname}">'

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

    # --- Reference images (supports URLs for GBIF mode, local paths otherwise) ---
    ref_html = ""
    ref_paths = reference_map.get(species, [])
    if ref_paths:
        for rp in ref_paths[:2]:
            ref_html += _ref_img_tag(rp, f"{ref_source}: {species}",
                                     "ref-img", embed_images)
        ref_html += (f'<div class="ref-label">{ref_source} image</div>')
    else:
        ref_html = (f'<div class="no-image" style="width:180px;height:180px;">'
                    f'No reference<br>available<br><small>(species may not '
                    f'be in candidate list)</small></div>')

    # Second choice reference
    ref2_html = ""
    second_display = second if second and str(second) != "nan" else "N/A"
    if second and str(second) != "nan":
        ref2_paths = reference_map.get(second, [])
        if ref2_paths:
            ref2_img = _ref_img_tag(ref2_paths[0], f"Runner-up: {second}",
                                    "ref-img", embed_images)
            ref2_html = f"""
            <div style="margin-top:15px;">
              <div class="section-label">Runner-up: {second}</div>
              {ref2_img}
              <div class="ref-label">{ref_source} image</div>
            </div>"""
        else:
            ref2_html = f"""
            <div style="margin-top:15px;">
              <div class="section-label">Runner-up: {second}</div>
              <div class="no-image" style="width:150px;height:100px;font-size:10px;">
                No reference<br>for runner-up</div>
            </div>"""

    # --- Confidence split: Clustering vs Annotation ---
    cluster_conf = 1.0 - (spatial * 0.5 + agreement * 0.5)
    cluster_conf_pct = max(0, min(100, cluster_conf * 100))
    cluster_conf_cls = ("good" if cluster_conf >= 0.7 else
                        "medium" if cluster_conf >= 0.4 else "bad")

    annot_conf = 1.0 - (semantic * 0.6 + taxonomic * 0.4)
    annot_conf_pct = max(0, min(100, annot_conf * 100))
    annot_conf_cls = ("good" if annot_conf >= 0.7 else
                      "medium" if annot_conf >= 0.4 else "bad")

    # Warning if species might not be in candidate list
    no_ref_warning = ""
    if not ref_paths and annot_conf_pct < 40:
        no_ref_warning = """
        <div style="background:#fff3cd; border:1px solid #ffc107; border-radius:6px;
                    padding:8px 12px; margin-top:10px; font-size:13px;">
          <strong>Note:</strong> No reference found and annotation confidence
          is low. The true species may not be in the candidate list.
          Use "Species not listed" below to enter the correct name.
        </div>"""

    return f"""
<div class="cluster-card tier-{tier}" data-tier="{tier}" data-species="{species}" id="cluster-{cid}">
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
        <div class="section-label">{ref_source}: {species}</div>
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
            esc = path.replace("\\", "/")
            img_html = f'<img src="file:///{esc}" alt="{fname}">'

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
  <h3 style="margin-bottom:15px;">Finalize & Export</h3>
  <div style="margin-bottom:15px;">
    <span id="review-progress" style="font-size:15px; color:#444;"></span>
  </div>
  <p style="margin-bottom:15px; color:#666; font-size:13px;">
    Your review is auto-saved. Close and reopen this file anytime &mdash; your progress is preserved.
  </p>
  <button class="export-btn" onclick="finalizeReview()" style="font-size:18px; padding:14px 40px;">
    Finalize Review &amp; Download</button>
  <p style="margin-top:10px; color:#888; font-size:12px;">
    Downloads a complete <strong>annotations_final.csv</strong> with all images,
    corrected labels, and provenance (AI / Confirmed / Corrected).
  </p>
  <div style="margin-top:20px; border-top:1px solid #eee; padding-top:15px;">
    <span style="font-size:13px; color:#888;">Additional exports:</span>
    <button class="export-btn secondary" onclick="exportCSV()"
            style="font-size:13px; padding:8px 20px;">Cluster CSV</button>
    <button class="export-btn secondary" onclick="exportImageCSV()"
            style="font-size:13px; padding:8px 20px;">Per-Image CSV</button>
  </div>
</div>
"""


def _html_footer(all_species=None, ann_records=None):
    import json as _json
    from datetime import datetime
    species_json = _json.dumps(all_species or [])
    ann_json = _json.dumps(ann_records or [])
    generation_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"""
<!-- Move modal -->
<div class="modal-overlay" id="move-modal">
  <div class="modal">
    <div class="modal-head">
      <h3 id="modal-title">Move image to...</h3>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div id="modal-img-preview" style="display:flex; align-items:center; margin-bottom:12px;
           padding:10px; background:#f8f9fa; border-radius:8px;">
      </div>
      <input type="text" class="modal-search" id="modal-search"
             placeholder="Search species name..." oninput="filterModalList()">
      <div class="modal-list" id="modal-list"></div>
    </div>
  </div>
</div>

<!-- Undo bar -->
<div class="undo-bar" id="undo-bar">
  <span class="undo-info" id="undo-info"></span>
  <button id="undo-btn" onclick="undo()">Undo (Ctrl+Z)</button>
  <button id="redo-btn" onclick="redo()">Redo (Ctrl+Y)</button>
  <button onclick="clearHistory()" style="opacity:0.7;">Clear</button>
</div>

<script>
// ---- State ----
const verdicts = {{}};
const imageMarks = {{}};
const allSpecies = {species_json};
const allAnnotations = {ann_json};
const STORAGE_KEY = 'bioclip_review_{generation_ts}';

// ---- localStorage: auto-save & restore ----
function saveState() {{
  try {{
    const state = {{
      verdicts: verdicts,
      imageMarks: imageMarks,
      corrections: {{}},
      notes: {{}},
      timestamp: new Date().toISOString()
    }};
    // Save correction inputs and notes
    document.querySelectorAll('.correction-input').forEach(el => {{
      if (el.value) state.corrections[el.id] = el.value;
    }});
    document.querySelectorAll('.notes-input').forEach(el => {{
      if (el.value) state.notes[el.id] = el.value;
    }});
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }} catch(e) {{}}
}}

function loadState() {{
  try {{
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const state = JSON.parse(raw);

    // Restore verdicts
    if (state.verdicts) {{
      Object.assign(verdicts, state.verdicts);
      for (const [cid, v] of Object.entries(state.verdicts)) {{
        const card = document.getElementById('cluster-' + cid);
        if (!card) continue;
        const btns = card.querySelectorAll('.verdict-btn');
        btns.forEach(b => {{
          const txt = b.textContent.trim();
          if ((v.verdict === 'correct' && txt === 'All Correct') ||
              (v.verdict === 'wrong' && txt === 'Wrong Name') ||
              (v.verdict === 'not_listed' && txt === 'Species Not Listed')) {{
            b.classList.add(v.verdict === 'not_listed' ? 'not-listed' : v.verdict);
          }}
        }});
        if (v.verdict === 'wrong' || v.verdict === 'not_listed') {{
          const inp = document.getElementById('correction-' + cid);
          if (inp) inp.classList.add('visible');
        }}
      }}
    }}

    // Restore image marks
    if (state.imageMarks) {{
      for (const [imgId, m] of Object.entries(state.imageMarks)) {{
        const card = document.getElementById(imgId);
        if (!card) continue;
        imageMarks[imgId] = m;
        card.classList.remove('marked-correct', 'marked-wrong', 'marked-moved');
        if (m.mark === 'correct') card.classList.add('marked-correct');
        else if (m.mark === 'wrong') card.classList.add('marked-wrong');
        else if (m.mark === 'move') card.classList.add('marked-moved');
      }}
      // Update all cluster badge counts
      const clusters = new Set();
      document.querySelectorAll('.img-card').forEach(c => clusters.add(c.dataset.cluster));
      clusters.forEach(cid => updateImageCounts(cid));
    }}

    // Restore corrections and notes text
    if (state.corrections) {{
      for (const [id, val] of Object.entries(state.corrections)) {{
        const el = document.getElementById(id);
        if (el) el.value = val;
      }}
    }}
    if (state.notes) {{
      for (const [id, val] of Object.entries(state.notes)) {{
        const el = document.getElementById(id);
        if (el) el.value = val;
      }}
    }}

    console.log('Review state restored from ' + state.timestamp);
  }} catch(e) {{ console.warn('Could not restore state:', e); }}
}}

// ---- Undo/redo history ----
const history = [];
let historyIdx = -1;

function pushAction(action) {{
  history.splice(historyIdx + 1);
  history.push(action);
  historyIdx = history.length - 1;
  updateUndoBar();
  saveState();
}}

function undo() {{
  if (historyIdx < 0) return;
  const action = history[historyIdx];
  historyIdx--;
  applyAction(action, true);
  updateUndoBar();
  saveState();
}}

function redo() {{
  if (historyIdx >= history.length - 1) return;
  historyIdx++;
  const action = history[historyIdx];
  applyAction(action, false);
  updateUndoBar();
  saveState();
}}

function applyAction(action, isUndo) {{
  const card = document.getElementById(action.imgId);
  if (!card) return;
  if (action.type === 'move') {{
    if (isUndo) {{
      card.classList.remove('marked-moved');
      delete imageMarks[action.imgId];
    }} else {{
      card.classList.add('marked-moved');
      imageMarks[action.imgId] = {{ mark: 'move', species: action.target === 'Noise' ? '' : action.target }};
    }}
  }} else if (action.type === 'mark') {{
    if (isUndo) {{
      card.classList.remove('marked-correct', 'marked-wrong');
      delete imageMarks[action.imgId];
    }} else {{
      card.classList.remove('marked-correct', 'marked-wrong');
      card.classList.add('marked-' + action.mark);
      imageMarks[action.imgId] = {{ mark: action.mark, species: '' }};
    }}
  }}
  updateImageCounts(card.dataset.cluster);
  updateProgress();
}}

function updateUndoBar() {{
  const bar = document.getElementById('undo-bar');
  const info = document.getElementById('undo-info');
  const undoBtn = document.getElementById('undo-btn');
  const redoBtn = document.getElementById('redo-btn');
  if (history.length === 0) {{ bar.classList.remove('visible'); return; }}
  bar.classList.add('visible');
  undoBtn.disabled = historyIdx < 0;
  redoBtn.disabled = historyIdx >= history.length - 1;
  if (historyIdx >= 0) {{
    const a = history[historyIdx];
    const fname = (document.getElementById(a.imgId) || {{}}).dataset?.filename || a.imgId;
    info.textContent = a.type === 'move'
      ? 'Moved "' + fname + '" to ' + a.target
      : 'Marked "' + fname + '" as ' + a.mark;
  }} else {{
    info.textContent = history.length + ' action(s)';
  }}
}}

function clearHistory() {{
  history.length = 0;
  historyIdx = -1;
  updateUndoBar();
}}

// ---- Toggle cluster ----
function toggleCluster(cid) {{
  const card = document.getElementById('cluster-' + cid);
  card.classList.toggle('collapsed');
}}

// ---- Filter clusters ----
function filterClusters(tier) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.cluster-card').forEach(card => {{
    if (tier === 'all' || card.dataset.tier === tier) card.style.display = '';
    else card.style.display = 'none';
  }});
}}

// ---- Search clusters ----
function searchClusters(q) {{
  q = q.toLowerCase().trim();
  document.querySelectorAll('.cluster-card').forEach(card => {{
    const sp = (card.dataset.species || '').toLowerCase();
    card.style.display = (!q || sp.includes(q)) ? '' : 'none';
  }});
}}

// ---- Cluster-level verdict ----
function setVerdict(cid, verdict, btn) {{
  btn.parentElement.querySelectorAll('.verdict-btn').forEach(b => {{
    b.classList.remove('correct', 'wrong', 'not-listed');
  }});
  btn.classList.add(verdict === 'not_listed' ? 'not-listed' : verdict);
  verdicts[cid] = {{ verdict: verdict, correction: '', notes: '' }};
  const input = document.getElementById('correction-' + cid);
  if (verdict === 'wrong' || verdict === 'not_listed') {{
    input.classList.add('visible');
    input.placeholder = verdict === 'not_listed'
      ? 'Type the correct species name (not in candidate list)...'
      : 'Enter correct species name...';
    input.focus();
  }} else {{
    input.classList.remove('visible');
  }}
  updateProgress();
  saveState();
}}

function saveCorrection(cid) {{
  const input = document.getElementById('correction-' + cid);
  if (verdicts[cid]) verdicts[cid].correction = input.value;
  saveState();
}}

// ---- Per-image marking ----
let pendingMoveImgId = null;

function markImage(imgId, mark) {{
  const card = document.getElementById(imgId);
  if (!card) return;
  if (mark === 'correct' || mark === 'wrong') {{
    card.classList.remove('marked-correct', 'marked-wrong', 'marked-moved');
    card.classList.add('marked-' + mark);
    imageMarks[imgId] = {{ mark: mark, species: '' }};
    pushAction({{ type: 'mark', imgId: imgId, mark: mark }});
    updateImageCounts(card.dataset.cluster);
    updateProgress();
  }} else if (mark === 'move') {{
    openMoveModal(imgId);
  }}
}}

// ---- Move modal ----
function openMoveModal(imgId) {{
  pendingMoveImgId = imgId;
  const card = document.getElementById(imgId);
  const fname = card?.dataset?.filename || imgId;
  const fromCluster = card?.dataset?.cluster || '?';
  let fromSpecies = '';
  const cc = document.getElementById('cluster-' + fromCluster);
  if (cc) {{ const em = cc.querySelector('.cluster-title em'); if (em) fromSpecies = em.textContent; }}

  const preview = document.getElementById('modal-img-preview');
  const img = card?.querySelector('img');
  const imgSrc = img ? img.src : '';
  preview.innerHTML = (imgSrc ? '<img class="modal-preview" src="' + imgSrc + '">' : '')
    + '<div><strong>' + fname + '</strong><br>'
    + '<span style="font-size:12px;color:#888;">Currently in: Cluster ' + fromCluster
    + ' (' + fromSpecies + ')</span></div>';
  document.getElementById('modal-title').textContent = 'Move image to...';
  document.getElementById('modal-search').value = '';
  buildModalList('');
  document.getElementById('move-modal').classList.add('open');
  document.getElementById('modal-search').focus();
}}

function closeModal() {{
  document.getElementById('move-modal').classList.remove('open');
  pendingMoveImgId = null;
}}

function filterModalList() {{ buildModalList(document.getElementById('modal-search').value); }}

function buildModalList(query) {{
  const list = document.getElementById('modal-list');
  const q = (query || '').toLowerCase().trim();
  const clusterSpecies = [];
  document.querySelectorAll('.cluster-card').forEach(cc => {{
    const em = cc.querySelector('.cluster-title em');
    if (em) clusterSpecies.push({{ name: em.textContent, source: 'Cluster ' + cc.id.replace('cluster-','') }});
  }});
  const candidateOnly = allSpecies.filter(sp => !clusterSpecies.some(cs => cs.name === sp))
    .map(sp => ({{ name: sp, source: 'Candidate' }}));
  let items = [...clusterSpecies, ...candidateOnly];
  if (q) items = items.filter(it => it.name.toLowerCase().includes(q));
  let html = '';
  items.forEach(it => {{
    html += '<div class="modal-item" onclick="doMove(\\'' + it.name.replace(/'/g, "\\\\'") + '\\')">'
          + '<span>' + it.name + '</span><span class="item-source">' + it.source + '</span></div>';
  }});
  html += '<div class="modal-item noise" onclick="doMove(\\'__noise__\\')">Mark as noise (unassigned)</div>';
  list.innerHTML = html;
}}

function doMove(target) {{
  if (!pendingMoveImgId) return;
  const card = document.getElementById(pendingMoveImgId);
  if (!card) return;
  const displayTarget = target === '__noise__' ? 'Noise' : target;
  card.classList.remove('marked-correct', 'marked-wrong');
  card.classList.add('marked-moved');
  imageMarks[pendingMoveImgId] = {{ mark: 'move', species: target === '__noise__' ? '' : target }};
  pushAction({{ type: 'move', imgId: pendingMoveImgId, target: displayTarget }});
  updateImageCounts(card.dataset.cluster);
  updateProgress();
  closeModal();
}}

// ---- Keyboard shortcuts ----
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeModal();
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') {{ e.preventDefault(); undo(); }}
  if ((e.ctrlKey || e.metaKey) && e.key === 'y') {{ e.preventDefault(); redo(); }}
}});

// ---- Image counts ----
function updateImageCounts(cid) {{
  let ok = 0, wrong = 0, moved = 0;
  document.querySelectorAll('.img-card[data-cluster="' + cid + '"]').forEach(card => {{
    const m = imageMarks[card.id];
    if (m) {{
      if (m.mark === 'correct') ok++;
      else if (m.mark === 'wrong') wrong++;
      else if (m.mark === 'move') moved++;
    }}
  }});
  const showBadge = (id, count, label) => {{
    const el = document.getElementById(id);
    if (el) {{
      if (count > 0) {{ el.textContent = count + ' ' + label; el.style.display = ''; }}
      else {{ el.style.display = 'none'; }}
    }}
  }};
  showBadge('ok-count-' + cid, ok, 'ok');
  showBadge('wrong-count-' + cid, wrong, 'wrong');
  showBadge('moved-count-' + cid, moved, 'moved');
}}

function updateProgress() {{
  const totalClusters = document.querySelectorAll('.cluster-card').length;
  const reviewed = Object.keys(verdicts).length;
  const totalImages = document.querySelectorAll('.img-card').length;
  const markedImages = Object.keys(imageMarks).length;
  const movedImages = Object.values(imageMarks).filter(m => m.mark === 'move').length;
  const el = document.getElementById('review-progress');
  if (el) {{
    el.textContent = 'Progress: ' + reviewed + '/' + totalClusters + ' clusters reviewed, '
                   + markedImages + '/' + totalImages + ' images marked'
                   + (movedImages > 0 ? ' (' + movedImages + ' moved)' : '');
  }}
}}

// ---- FINALIZE: produce complete annotations_final.csv ----
function finalizeReview() {{
  // Build filename -> expert decision lookup
  const imgLookup = {{}};  // filename -> {{mark, species}}
  document.querySelectorAll('.img-card').forEach(card => {{
    const fname = card.dataset.filename;
    const m = imageMarks[card.id];
    if (fname && m) imgLookup[fname] = m;
  }});

  // Build cluster -> verdict lookup
  const clusterLookup = {{}};  // cid -> {{verdict, correction}}
  for (const [cid, v] of Object.entries(verdicts)) {{
    const corr = document.getElementById('correction-' + cid);
    const notes = document.getElementById('notes-' + cid);
    clusterLookup[cid] = {{
      verdict: v.verdict,
      correction: corr ? corr.value : (v.correction || ''),
      notes: notes ? notes.value : ''
    }};
  }}

  // Process every annotation row
  const header = ['file_path','cluster_id','ai_annotation','ai_second_choice',
    'confidence','margin','semantic_uncertainty','spatial_uncertainty',
    'cluster_agreement','taxonomic_consistency','combined_uncertainty',
    'confidence_tier','annotation_source','expert_notes'];
  const rows = [header];

  allAnnotations.forEach(ann => {{
    const row = {{ ...ann }};
    const fname = row.file_path.split(/[/\\\\]/).pop();
    const cid = String(row.cluster_id);

    // 1. Check per-image decision (highest priority)
    const imgDecision = imgLookup[fname];
    if (imgDecision) {{
      if (imgDecision.mark === 'correct') {{
        row.annotation_source = 'Human_Confirmed';
        row.confidence_tier = 'human_verified';
        row.combined_uncertainty = 0;
      }} else if (imgDecision.mark === 'wrong') {{
        row.annotation_source = 'Flagged_Wrong';
        row.confidence_tier = 'human_required';
      }} else if (imgDecision.mark === 'move') {{
        if (imgDecision.species) {{
          row.ai_annotation = imgDecision.species;
          row.annotation_source = 'Human_Expert';
          row.confidence_tier = 'human_verified';
          row.combined_uncertainty = 0;
          row.semantic_uncertainty = 0;
          row.spatial_uncertainty = 0;
          row.cluster_agreement = 0;
          row.taxonomic_consistency = 0;
        }} else {{
          // Moved to noise
          row.cluster_id = -1;
          row.annotation_source = 'Human_Noise';
          row.confidence_tier = 'human_verified';
        }}
      }}
    }}
    // 2. Check cluster-level verdict (if no per-image decision)
    else if (clusterLookup[cid]) {{
      const cv = clusterLookup[cid];
      if (cv.verdict === 'correct') {{
        row.annotation_source = 'Human_Confirmed';
        row.confidence_tier = 'human_verified';
        row.combined_uncertainty = 0;
      }} else if (cv.verdict === 'wrong' || cv.verdict === 'not_listed') {{
        if (cv.correction) {{
          row.ai_annotation = cv.correction;
          row.annotation_source = 'Human_Expert';
          row.confidence_tier = 'human_verified';
          row.combined_uncertainty = 0;
          row.semantic_uncertainty = 0;
          row.spatial_uncertainty = 0;
          row.cluster_agreement = 0;
          row.taxonomic_consistency = 0;
        }} else {{
          row.annotation_source = 'Flagged_Wrong';
          row.confidence_tier = 'human_required';
        }}
      }}
    }}

    // Add expert notes if any
    const clNotes = clusterLookup[cid]?.notes || '';
    row.expert_notes = clNotes;

    rows.push(header.map(h => row[h] !== undefined ? row[h] : ''));
  }});

  downloadCSV(rows, 'annotations_final.csv');

  // Show summary
  let confirmed = 0, corrected = 0, flagged = 0, ai = 0;
  rows.slice(1).forEach(r => {{
    const src = r[header.indexOf('annotation_source')];
    if (src === 'Human_Confirmed') confirmed++;
    else if (src === 'Human_Expert') corrected++;
    else if (src === 'Flagged_Wrong' || src === 'Human_Noise') flagged++;
    else ai++;
  }});
  alert('annotations_final.csv downloaded!\\n\\n'
    + 'Total images: ' + (rows.length - 1) + '\\n'
    + 'AI accepted: ' + ai + '\\n'
    + 'Human confirmed: ' + confirmed + '\\n'
    + 'Human corrected: ' + corrected + '\\n'
    + 'Flagged/noise: ' + flagged + '\\n\\n'
    + 'Use --merge with this CSV to generate sorted species folders and summary.');
}}

// ---- Export cluster CSV (secondary) ----
function exportCSV() {{
  const rows = [['Cluster_ID', 'AI_Prediction', 'Expert_Verdict',
                 'Expert_Corrected_Label', 'Expert_Notes',
                 'Images_Correct', 'Images_Wrong', 'Images_Moved']];
  document.querySelectorAll('.cluster-card').forEach(card => {{
    const cid = card.id.replace('cluster-', '');
    const title = card.querySelector('.cluster-title em');
    const species = title ? title.textContent : '';
    const v = verdicts[cid] || {{}};
    const notes = document.getElementById('notes-' + cid);
    let ok = 0, wrong = 0, moved = 0;
    document.querySelectorAll('.img-card[data-cluster="' + cid + '"]').forEach(ic => {{
      const m = imageMarks[ic.id];
      if (m) {{ if (m.mark === 'correct') ok++; else if (m.mark === 'wrong') wrong++; else if (m.mark === 'move') moved++; }}
    }});
    rows.push([cid, species, v.verdict || 'not_reviewed',
      v.correction || '', notes ? notes.value : '', ok, wrong, moved]);
  }});
  downloadCSV(rows, 'expert_review_clusters.csv');
}}

// ---- Export per-image CSV (secondary) ----
function exportImageCSV() {{
  const rows = [['Filename', 'Cluster_ID', 'AI_Prediction', 'Image_Verdict', 'Moved_To_Species']];
  document.querySelectorAll('.img-card').forEach(card => {{
    const m = imageMarks[card.id] || {{}};
    if (m.mark) {{
      let aiPred = '';
      const cc = document.getElementById('cluster-' + card.dataset.cluster);
      if (cc) {{ const em = cc.querySelector('.cluster-title em'); if (em) aiPred = em.textContent; }}
      rows.push([card.dataset.filename || '', card.dataset.cluster || '', aiPred, m.mark, m.species || '']);
    }}
  }});
  if (rows.length <= 1) {{ alert('No images marked yet.'); return; }}
  downloadCSV(rows, 'expert_review_images.csv');
}}

function downloadCSV(rows, filename) {{
  const csv = rows.map(r => r.map(c => '"' + String(c).replace(/"/g, '""') + '"').join(',')).join('\\n');
  const blob = new Blob([csv], {{ type: 'text/csv' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}}

// ---- Initialize ----
loadState();
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
    parser.add_argument("--mode", default="gbif",
                        choices=["gbif", "local", "csv"],
                        help="Candidate source mode (gbif/local/csv).")
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
        df, ref_map, args.output,
        embed_images=not args.no_embed,
        candidate_mode=args.mode
    )
    print(f"\nReport generated: {path}")
    print("Open in a web browser to review.")
