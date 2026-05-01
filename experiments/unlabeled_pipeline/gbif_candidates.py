"""
experiments/unlabeled_pipeline/gbif_candidates.py
-------------------------------------------------
Query GBIF for candidate Lepidoptera species in a given region,
then download 1-2 reference images per species for expert comparison.

Two-step process:
  1. query_candidate_species()  - GBIF Occurrence API -> species list + taxonomy
  2. download_reference_images() - GBIF Occurrence API (media) -> reference photos

Usage (standalone test):
    python experiments/unlabeled_pipeline/gbif_candidates.py \
        --region europe --limit 100 --output results/unlabeled_test
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import List, Dict, Optional, Tuple, Union
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.utils.pretty import header, subheader, done


# -------------------------------------------------------
# Constants
# -------------------------------------------------------

GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_SPECIES_URL    = "https://api.gbif.org/v1/species/search"
GBIF_SPECIES_MATCH  = "https://api.gbif.org/v1/species/match"

LEPIDOPTERA_KEY = 797   # GBIF taxon key for order Lepidoptera

# GBIF continent codes
CONTINENT_MAP = {
    "europe":        "EUROPE",
    "asia":          "ASIA",
    "africa":        "AFRICA",
    "north_america": "NORTH_AMERICA",
    "south_america": "SOUTH_AMERICA",
    "oceania":       "OCEANIA",
    "antarctica":    "ANTARCTICA",
}

# Rate limiting
GBIF_DELAY       = 0.5    # seconds between requests (be gentle to GBIF API)
GBIF_RETRIES     = 3
GBIF_RETRY_DELAY = 10.0   # longer backoff on failure


# -------------------------------------------------------
# 1. Query candidate species from GBIF
# -------------------------------------------------------

def query_candidate_species(
    order_key: Union[int, List[int]] = LEPIDOPTERA_KEY,
    continent: str = "EUROPE",
    limit: int = 200,
    include_rare: bool = True,
    rare_fraction: float = 0.1,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Query GBIF occurrence API for species observed in a region.

    Returns a list of dicts, each with:
        species, genus, family, order, taxon_key, n_occurrences

    Args:
        order_key:     GBIF taxon key(s) for the target order(s).
                       Single int (e.g. 797 for Lepidoptera) or list of ints
                       (e.g. [797, 7192] for Lepidoptera + Aves).
        continent:     GBIF continent code (e.g., "EUROPE", "ASIA").
        limit:         Max number of candidate species to return.
                       When multiple order_keys, limit is split proportionally.
        include_rare:  If True, include some rare species (fewer occurrences).
        rare_fraction: Fraction of the limit reserved for rare species.
        cache_dir:     Directory to cache results (None = no caching).
    """
    # Normalize order_key to list
    order_keys = [order_key] if isinstance(order_key, int) else list(order_key)
    cache_key_str = "_".join(str(k) for k in sorted(order_keys))

    # Check cache
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            f"candidates_{continent.lower()}_{cache_key_str}_{limit}.json"
        )
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"[gbif] Loaded {len(cached)} candidates from cache")
            return cached

    # Split limit across order keys
    per_key_limit = max(1, limit // len(order_keys))

    all_species = {}

    for okey in order_keys:
        subheader(f"Querying GBIF for {continent} species (orderKey={okey})")

        # GBIF species facet: get species list with occurrence counts
        offset = 0
        page_size = 300

        print(f"    Fetching species facets from GBIF occurrences ...")

        key_species = {}
        while len(key_species) < per_key_limit * 3:
            params = {
                "orderKey":   okey,
                "continent":  continent,
                "hasCoordinate": "true",
                "mediaType":  "StillImage",
                "facet":      "speciesKey",
                "facetLimit": page_size,
                "facetOffset": offset,
                "limit":      0,
            }

            r = _gbif_request(GBIF_OCCURRENCE_URL, params)
            if r is None:
                break

            data = r.json()
            facets = data.get("facets", [])
            if not facets:
                break

            species_facet = facets[0].get("counts", [])
            if not species_facet:
                break

            for item in species_facet:
                key = int(item["name"])
                count = item["count"]
                key_species[key] = count

            print(f"    ... {len(key_species)} species found so far")

            if len(species_facet) < page_size:
                break
            offset += page_size
            time.sleep(GBIF_DELAY)

        all_species.update(key_species)
        print(f"    orderKey={okey}: {len(key_species)} species")

    if not all_species:
        print("    [warn] No species found! Check region and taxon key.")
        return []

    print(f"    Total unique species with images: {len(all_species)}")

    # Sort by occurrence count (most common first)
    sorted_species = sorted(all_species.items(), key=lambda x: -x[1])

    # Select candidates: mostly common, some rare
    if include_rare and len(sorted_species) > limit:
        n_rare = max(1, int(limit * rare_fraction))
        n_common = limit - n_rare
        common = sorted_species[:n_common]
        # Rare = from the bottom quartile
        rare_pool = sorted_species[len(sorted_species) * 3 // 4:]
        import random
        random.shuffle(rare_pool)
        rare = rare_pool[:n_rare]
        selected = common + rare
    else:
        selected = sorted_species[:limit]

    selected_keys = {k for k, _ in selected}
    selected_counts = {k: c for k, c in selected}

    # Resolve taxonomy for each species key
    subheader(f"Resolving taxonomy for {len(selected)} candidates")
    candidates = []

    for i, (species_key, count) in enumerate(selected):
        pct = (i + 1) / len(selected) * 100
        print(f"    [{i+1:3d}/{len(selected)}] {pct:5.1f}%  "
              f"key={species_key} ({count} occ)", end=" ... ", flush=True)

        info = _resolve_species(species_key)
        if info and info.get("species"):
            info["n_occurrences"] = count
            info["taxon_key"] = species_key
            candidates.append(info)
            print(f"{info['species']}  ({info.get('family', '??')})")
        else:
            print("skip (no species-level match)")

        time.sleep(GBIF_DELAY)

    # Deduplicate by species name
    seen = set()
    unique = []
    for c in candidates:
        if c["species"] not in seen:
            seen.add(c["species"])
            unique.append(c)
    candidates = unique

    print(f"\n    Final candidates: {len(candidates)} species")

    # Cache
    if cache_file:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)
        print(f"    Cached to {cache_file}")

    return candidates


def query_candidates_balanced(
    order_key: Union[int, List[int]] = LEPIDOPTERA_KEY,
    continent: str = "EUROPE",
    per_family: int = 40,
    rare_per_family: int = 10,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Balanced per-family candidate query.

    Instead of fetching the top-N species globally (which over-represents
    common families like Noctuidae/Geometridae), this queries each family
    separately and takes the top `per_family` species + `rare_per_family`
    rare species from each.

    This ensures small families (Saturniidae, Drepanidae) get proper
    coverage -- critical for annotation accuracy.

    Args:
        order_key:        GBIF taxon key(s). Single int (e.g. 797) or list
                          of ints (e.g. [797, 7192]) for mixed taxa.
        continent:        GBIF continent code.
        per_family:       Common species to take per family.
        rare_per_family:  Rare species to sample per family (from bottom quartile).
        cache_dir:        Cache directory.
    """
    import random

    # Normalize order_key to list
    order_keys = [order_key] if isinstance(order_key, int) else list(order_key)
    cache_key_str = "_".join(str(k) for k in sorted(order_keys))

    # Check cache
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            f"candidates_balanced_{continent.lower()}_{cache_key_str}.json"
        )
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"[gbif] Loaded {len(cached)} balanced candidates from cache")
            return cached

    order_label = " + ".join(str(k) for k in order_keys)
    subheader(f"Balanced per-family query: {continent} (orderKeys: {order_label})")

    # Step 1: Discover families across all order keys
    family_keys = []

    for okey in order_keys:
        print(f"    Discovering families for orderKey={okey} in {continent} ...")
        params = {
            "orderKey":      okey,
            "continent":     continent,
            "hasCoordinate": "true",
            "facet":         "familyKey",
            "facetLimit":    100,
            "limit":         0,
        }

        r = _gbif_request(GBIF_OCCURRENCE_URL, params)
        if r is None:
            print(f"    [warn] Could not query families for orderKey={okey}")
            continue

        facets = r.json().get("facets", [])
        if not facets:
            print(f"    [warn] No family facets for orderKey={okey}")
            continue

        for item in facets[0].get("counts", []):
            family_keys.append((int(item["name"]), item["count"]))

        time.sleep(GBIF_DELAY)

    if not family_keys:
        print("    [fail] No families found. Falling back to flat query.")
        return query_candidate_species(order_keys, continent, 400, True, 0.1, cache_dir)

    # Deduplicate family keys (same family might appear under different orders)
    seen_fam_keys = {}
    for fk, count in family_keys:
        seen_fam_keys[fk] = seen_fam_keys.get(fk, 0) + count
    family_keys = list(seen_fam_keys.items())

    print(f"    Found {len(family_keys)} families")
    time.sleep(GBIF_DELAY)

    # Step 2: Resolve family names
    print(f"    Resolving family names ...")
    families = []  # list of (family_name, family_key, n_occurrences)

    for fam_key, n_occ in family_keys:
        url = f"https://api.gbif.org/v1/species/{fam_key}"
        r = _gbif_request(url)
        if r is not None:
            data = r.json()
            fam_name = data.get("family", data.get("canonicalName", f"key_{fam_key}"))
            families.append((fam_name, fam_key, n_occ))
        time.sleep(GBIF_DELAY * 0.5)

    families.sort(key=lambda x: -x[2])
    print(f"    Families (by occurrence count):")
    for fam_name, fam_key, n_occ in families[:15]:
        print(f"      {fam_name:25s}  key={fam_key}  ({n_occ:,} occurrences)")
    if len(families) > 15:
        print(f"      ... and {len(families)-15} more")

    # Step 3: Query species per family
    all_candidates = []
    seen_names = set()

    for fam_name, fam_key, n_occ in families:
        print(f"    {fam_name:25s}", end=" ... ", flush=True)

        # Get species in this family for this continent
        family_species = {}
        offset = 0

        while len(family_species) < (per_family + rare_per_family) * 3:
            params = {
                "familyKey":     fam_key,
                "continent":     continent,
                "hasCoordinate": "true",
                "facet":         "speciesKey",
                "facetLimit":    200,
                "facetOffset":   offset,
                "limit":         0,
            }

            r = _gbif_request(GBIF_OCCURRENCE_URL, params)
            if r is None:
                break

            facets = r.json().get("facets", [])
            if not facets:
                break

            counts = facets[0].get("counts", [])
            if not counts:
                break

            for item in counts:
                family_species[int(item["name"])] = item["count"]

            if len(counts) < 200:
                break
            offset += 200
            time.sleep(GBIF_DELAY)

        if not family_species:
            print("0 species (query failed)")
            continue

        # Sort by occurrence count
        sorted_sp = sorted(family_species.items(), key=lambda x: -x[1])
        total_available = len(sorted_sp)

        # Select: top per_family + rare tail
        n_common = min(per_family, len(sorted_sp))
        selected = sorted_sp[:n_common]

        if rare_per_family > 0 and len(sorted_sp) > per_family:
            tail = sorted_sp[len(sorted_sp) * 3 // 4:]
            # Don't re-pick species already in common
            common_keys = {k for k, _ in selected}
            tail = [(k, c) for k, c in tail if k not in common_keys]
            random.shuffle(tail)
            selected += tail[:rare_per_family]

        # Resolve taxonomy
        family_candidates = []
        for species_key, count in selected:
            info = _resolve_species(species_key)
            if info and info.get("species") and info["species"] not in seen_names:
                seen_names.add(info["species"])
                info["n_occurrences"] = count
                info["taxon_key"] = species_key
                family_candidates.append(info)
            time.sleep(GBIF_DELAY * 0.3)

        all_candidates.extend(family_candidates)
        print(f"{len(family_candidates)} species (of {total_available} available)")
        time.sleep(GBIF_DELAY)

    print(f"\n    Total balanced candidates: {len(all_candidates)} species "
          f"from {len(families)} families")

    # Cache
    if cache_file:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(all_candidates, f, indent=2)
        print(f"    Cached to {cache_file}")

    return all_candidates


def _resolve_species(species_key: int) -> Optional[Dict]:
    """Look up taxonomy for a GBIF species key."""
    url = f"https://api.gbif.org/v1/species/{species_key}"
    r = _gbif_request(url)
    if r is None:
        return None

    data = r.json()
    rank = data.get("rank", "")
    if rank not in ("SPECIES", "SUBSPECIES"):
        return None

    return {
        "species": data.get("species", data.get("canonicalName", "")),
        "genus":   data.get("genus", ""),
        "family":  data.get("family", ""),
        "order":   data.get("order", ""),
    }


# -------------------------------------------------------
# 2a. Collect reference image URLs (no download)
# -------------------------------------------------------

def collect_reference_urls(
    candidates: List[Dict],
    images_per_species: int = 3,
    cache_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Collect GBIF image URLs for each candidate species WITHOUT downloading.

    The HTML report can use these URLs directly as <img src="...">,
    letting the browser load them on-the-fly. Much faster than downloading
    hundreds of images to disk.

    Args:
        candidates:        List of species dicts from query_candidate_species().
        images_per_species: Number of reference image URLs per species.
        cache_dir:         Directory for caching URL lookups.

    Returns:
        Dict mapping species_name -> [url1, url2, ...].
    """
    subheader(f"Collecting GBIF reference image URLs ({images_per_species}/species)")

    # Load URL cache if available
    url_cache_path = None
    url_cache = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        url_cache_path = os.path.join(cache_dir, "reference_urls_cache.json")
        if os.path.exists(url_cache_path):
            with open(url_cache_path, "r", encoding="utf-8") as f:
                url_cache = json.load(f)

    reference_urls = {}
    total = len(candidates)
    cached_count = 0

    for i, species_info in enumerate(candidates):
        name = species_info["species"]
        taxon_key = species_info.get("taxon_key")

        # Check URL cache first
        if name in url_cache and len(url_cache[name]) >= images_per_species:
            reference_urls[name] = url_cache[name][:images_per_species]
            cached_count += 1
            continue

        pct = (i + 1) / total * 100
        print(f"    [{i+1:3d}/{total}] {pct:5.1f}%  {name}", end=" ... ", flush=True)

        image_urls = _get_species_image_urls(
            taxon_key, max_images=images_per_species
        )
        url_cache[name] = image_urls
        reference_urls[name] = image_urls
        print(f"{len(image_urls)} URLs")
        time.sleep(GBIF_DELAY)

    # Save URL cache
    if url_cache_path:
        with open(url_cache_path, "w", encoding="utf-8") as f:
            json.dump(url_cache, f, indent=2)

    n_with_urls = sum(1 for v in reference_urls.values() if v)
    total_urls = sum(len(v) for v in reference_urls.values())
    print(f"\n    {n_with_urls}/{total} species have reference URLs "
          f"({total_urls} total)")
    if cached_count:
        print(f"    {cached_count} loaded from cache")

    return reference_urls


# -------------------------------------------------------
# 2b. Download reference images from GBIF (optional)
# -------------------------------------------------------

def download_reference_images(
    candidates: List[Dict],
    output_dir: str,
    images_per_species: int = 2,
    max_retries: int = 2,
    cache_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Download reference images from GBIF for each candidate species.

    Args:
        candidates:        List of species dicts from query_candidate_species().
        output_dir:        Directory to save images to.
        images_per_species: Number of reference images per species (1-3).
        max_retries:       Max download retries per image.
        cache_dir:         Directory for caching image URL lookups.

    Returns:
        Dict mapping species_name -> [local_image_path, ...]
    """
    ref_dir = os.path.join(output_dir, "reference_images")
    os.makedirs(ref_dir, exist_ok=True)

    subheader(f"Downloading reference images ({images_per_species}/species)")

    # Load URL cache if available
    url_cache_path = None
    url_cache = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        url_cache_path = os.path.join(cache_dir, "reference_urls_cache.json")
        if os.path.exists(url_cache_path):
            with open(url_cache_path, "r", encoding="utf-8") as f:
                url_cache = json.load(f)

    reference_map = {}
    total = len(candidates)
    downloaded = 0
    skipped = 0

    for i, species_info in enumerate(candidates):
        name = species_info["species"]
        taxon_key = species_info.get("taxon_key")
        pct = (i + 1) / total * 100

        # Sanitize folder name
        safe_name = name.replace(" ", "_").replace("/", "_")
        species_dir = os.path.join(ref_dir, safe_name)

        # Check if already downloaded
        if os.path.exists(species_dir):
            existing = [f for f in os.listdir(species_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if len(existing) >= images_per_species:
                reference_map[name] = [
                    os.path.join(species_dir, f) for f in existing[:images_per_species]
                ]
                skipped += 1
                continue

        print(f"    [{i+1:3d}/{total}] {pct:5.1f}%  {name}", end=" ... ", flush=True)

        # Get image URLs from GBIF
        if name in url_cache:
            image_urls = url_cache[name]
        else:
            image_urls = _get_species_image_urls(
                taxon_key, max_images=images_per_species * 2
            )
            url_cache[name] = image_urls
            time.sleep(GBIF_DELAY)

        if not image_urls:
            print("no images available")
            reference_map[name] = []
            continue

        # Download images
        os.makedirs(species_dir, exist_ok=True)
        local_paths = []

        for j, url in enumerate(image_urls[:images_per_species]):
            ext = _guess_extension(url)
            local_path = os.path.join(species_dir, f"ref_{j+1}{ext}")

            if os.path.exists(local_path):
                local_paths.append(local_path)
                continue

            success = _download_image(url, local_path, max_retries)
            if success:
                local_paths.append(local_path)

        reference_map[name] = local_paths
        downloaded += 1
        print(f"{len(local_paths)} images")

    # Save URL cache
    if url_cache_path:
        with open(url_cache_path, "w", encoding="utf-8") as f:
            json.dump(url_cache, f, indent=2)

    total_images = sum(len(v) for v in reference_map.values())
    print(f"\n    Downloaded: {downloaded} species, {total_images} images total")
    if skipped:
        print(f"    Skipped (already exists): {skipped} species")
    print(f"    Reference images saved to: {ref_dir}")

    return reference_map


def _get_species_image_urls(taxon_key: int, max_images: int = 4) -> List[str]:
    """Get image URLs from GBIF occurrence records for a species."""
    params = {
        "speciesKey": taxon_key,
        "mediaType":  "StillImage",
        "limit":      max_images * 2,  # fetch extra in case some fail
    }

    r = _gbif_request(GBIF_OCCURRENCE_URL, params)
    if r is None:
        return []

    urls = []
    for occ in r.json().get("results", []):
        for media in occ.get("media", []):
            if media.get("type") == "StillImage":
                url = media.get("identifier", "")
                if url and url.startswith("http"):
                    urls.append(url)
                    if len(urls) >= max_images:
                        return urls
    return urls


def _download_image(url: str, local_path: str, max_retries: int = 2) -> bool:
    """Download a single image from URL to local path."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=30, stream=True,
                             headers={"User-Agent": "BioCLIP-Pipeline/1.0"})
            r.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return False

    return False


def _guess_extension(url: str) -> str:
    """Guess image file extension from URL."""
    url_lower = url.lower().split("?")[0]
    if url_lower.endswith(".png"):
        return ".png"
    if url_lower.endswith(".jpeg"):
        return ".jpeg"
    return ".jpg"


# -------------------------------------------------------
# Helper: GBIF request with retry
# -------------------------------------------------------

def _gbif_request(url: str, params: dict = None) -> Optional[requests.Response]:
    """Make a GBIF API request with retries."""
    for attempt in range(1, GBIF_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=15,
                             headers={"User-Agent": "BioCLIP-Pipeline/1.0"})
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if attempt < GBIF_RETRIES:
                time.sleep(GBIF_RETRY_DELAY)
            else:
                print(f"    [fail] {url}: {e}")
                return None


# -------------------------------------------------------
# Stage 2: Expand candidates for low-confidence clusters
# -------------------------------------------------------

def expand_candidates_for_gaps(
    annotations_df,
    existing_candidates: List[Dict],
    config: dict,
    cache_dir: Optional[str] = None,
    max_per_family: int = 80,
) -> List[Dict]:
    """
    Second-stage GBIF query: find additional species for clusters where
    annotation confidence is low (the true species may not be in the
    original candidate list).

    Strategy:
      1. Identify clusters with high uncertainty or low margin
      2. Look at the AI's best guess genus/family for those clusters
      3. Query GBIF OCCURRENCE API (region-filtered!) for more species
         in those families that actually occur in the target region
      4. Return expanded candidate list (original + new)

    Args:
        annotations_df:      DataFrame from first-stage annotation.
        existing_candidates: Original candidate list from stage 1.
        config:              Pipeline config.
        cache_dir:           Cache directory for GBIF queries.
        max_per_family:      Max new species to add per family.

    Returns:
        Extended list of candidate dicts (original + new).
    """
    from src.utils.pretty import subheader

    subheader("Stage 2 - Expanding candidate list for low-confidence clusters")

    # Get region from config (critical: only fetch species from the right region)
    gbif_cfg = config.get("gbif", {})
    region = gbif_cfg.get("region", "europe")
    continent = CONTINENT_MAP.get(region, "EUROPE")

    # Check expansion cache first
    expansion_cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        expansion_cache_path = os.path.join(
            cache_dir, f"expansion_{continent.lower()}.json"
        )
        if os.path.exists(expansion_cache_path):
            with open(expansion_cache_path, "r", encoding="utf-8") as f:
                cached_new = json.load(f)
            print(f"    Loaded {len(cached_new)} expansion candidates from cache")
            return existing_candidates + cached_new

    # Existing species names (to avoid duplicates)
    existing_names = {c["species"] for c in existing_candidates}

    # Get cluster-level info (one row per cluster)
    cluster_info = []
    for cid in annotations_df["cluster_id"].unique():
        if cid < 0:
            continue
        cdf = annotations_df[annotations_df["cluster_id"] == cid]
        row = cdf.iloc[0]
        margin = row.get("margin", 0)
        species = row["ai_annotation"]
        tier = row["confidence_tier"]

        if tier in ("review", "human_required") or margin < 0.02:
            cluster_info.append({
                "cluster_id": cid,
                "species": species,
                "margin": margin,
                "tier": tier,
                "n_images": len(cdf),
            })

    if not cluster_info:
        print("    No low-confidence clusters found. Skipping expansion.")
        return existing_candidates

    print(f"    {len(cluster_info)} low-confidence clusters identified")
    print(f"    Region filter: {continent}")

    # Collect target families to expand
    target_families = {}  # family -> family_key (to be resolved)

    for ci in cluster_info:
        sp = ci["species"]
        for ec in existing_candidates:
            if ec["species"] == sp:
                fam = ec.get("family", "")
                if fam and fam not in target_families:
                    target_families[fam] = None
                break

    if not target_families:
        print("    Could not determine families for expansion.")
        return existing_candidates

    print(f"    Target families: {', '.join(sorted(target_families.keys()))}")

    # Resolve family keys from GBIF
    for family in list(target_families.keys()):
        r = _gbif_request(GBIF_SPECIES_MATCH, {"name": family, "rank": "FAMILY",
                                                 "kingdom": "Animalia"})
        if r is not None:
            target_families[family] = r.json().get("usageKey")
        time.sleep(GBIF_DELAY)

    # Query GBIF OCCURRENCE API for species in each family, filtered by continent
    # This ensures we only get species that actually occur in the target region
    new_candidates = []

    for family, family_key in target_families.items():
        if not family_key:
            continue

        print(f"    Querying {continent} occurrences for {family} ...",
              end=" ", flush=True)

        # Use occurrence facet (same approach as Stage 1) but filtered by family
        family_species = {}
        offset = 0

        while len(family_species) < max_per_family:
            params = {
                "familyKey":     family_key,
                "continent":     continent,
                "hasCoordinate": "true",
                "facet":         "speciesKey",
                "facetLimit":    200,
                "facetOffset":   offset,
                "limit":         0,  # only need facets
            }

            r = _gbif_request(GBIF_OCCURRENCE_URL, params)
            if r is None:
                break

            facets = r.json().get("facets", [])
            if not facets:
                break

            counts = facets[0].get("counts", [])
            if not counts:
                break

            for item in counts:
                key = int(item["name"])
                count = item["count"]
                family_species[key] = count

            if len(counts) < 200:
                break
            offset += 200
            time.sleep(GBIF_DELAY)

        # Resolve species names for new keys only
        family_new = 0
        for species_key, count in sorted(family_species.items(),
                                          key=lambda x: -x[1]):
            info = _resolve_species(species_key)
            if info and info.get("species"):
                sp_name = info["species"]
                if sp_name not in existing_names:
                    existing_names.add(sp_name)
                    new_candidates.append({
                        "species":       sp_name,
                        "genus":         info.get("genus", ""),
                        "family":        info.get("family", family),
                        "order":         info.get("order", "Lepidoptera"),
                        "taxon_key":     species_key,
                        "n_occurrences": count,
                        "source":        "stage2_expansion",
                    })
                    family_new += 1
                    if family_new >= max_per_family:
                        break
            time.sleep(GBIF_DELAY * 0.5)

        print(f"+{family_new} species (from {continent})")
        time.sleep(GBIF_DELAY)

    if new_candidates:
        print(f"\n    Stage 2 added {len(new_candidates)} region-filtered species")
        # Cache expansion results
        if expansion_cache_path:
            with open(expansion_cache_path, "w", encoding="utf-8") as f:
                json.dump(new_candidates, f, indent=2)
            print(f"    Cached to {expansion_cache_path}")
    else:
        print("\n    No new candidates found in expansion.")

    return existing_candidates + new_candidates


# -------------------------------------------------------
# Build taxonomy table from candidates
# -------------------------------------------------------

def candidates_to_taxonomy(candidates: List[Dict]) -> Dict[str, Dict]:
    """
    Convert candidate species list into a TaxonomyTable
    compatible with the annotation pipeline.
    """
    table = {}
    for c in candidates:
        name = c["species"]
        table[name] = {
            "genus":            c.get("genus", ""),
            "family":           c.get("family", ""),
            "order":            c.get("order", ""),
            "match_type":       "GBIF",
            "gbif_confidence":  100,
        }
    return table


def candidates_to_class_names(candidates: List[Dict]) -> List[str]:
    """Extract sorted, unique species names from candidates."""
    return sorted(set(c["species"] for c in candidates))


# -------------------------------------------------------
# Standalone test
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query GBIF for candidate species + reference images."
    )
    parser.add_argument("--region", default="europe",
                        choices=list(CONTINENT_MAP.keys()),
                        help="Geographic region to query.")
    parser.add_argument("--order-key", type=int, nargs="+", default=[797],
                        help="GBIF order key(s). E.g. 797 (Lepidoptera) "
                             "7192 (Aves). Multiple keys for mixed taxa.")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of candidate species.")
    parser.add_argument("--images", type=int, default=2,
                        help="Reference images per species.")
    parser.add_argument("--output", default="results/unlabeled_test",
                        help="Output directory.")
    parser.add_argument("--include-rare", action="store_true", default=True,
                        help="Include rare species in candidates.")
    args = parser.parse_args()

    continent = CONTINENT_MAP[args.region]
    cache_dir = os.path.join(args.output, ".cache", "gbif")

    # Single key -> int, multiple -> list
    order_key = args.order_key[0] if len(args.order_key) == 1 else args.order_key

    header("GBIF CANDIDATE SPECIES QUERY")

    candidates = query_candidate_species(
        order_key=order_key,
        continent=continent,
        limit=args.limit,
        include_rare=args.include_rare,
        cache_dir=cache_dir,
    )

    if candidates:
        print(f"\n  Top 10 candidates:")
        for c in candidates[:10]:
            print(f"    {c['species']:40s}  {c['family']:20s}  "
                  f"({c['n_occurrences']} occ)")

        print(f"\n  Downloading reference images ...")
        ref_map = download_reference_images(
            candidates, args.output,
            images_per_species=args.images,
            cache_dir=cache_dir,
        )

        done(args.output)
