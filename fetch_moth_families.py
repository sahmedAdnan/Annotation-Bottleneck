"""
fetch_moth_families.py
----------------------
Queries GBIF species/match API for every species in mothDK_classes.csv
and writes mothDK_classes_with_family.csv with full taxonomic columns.

Output columns:
  index, class_name, family, order, genus, match_type, gbif_confidence

Usage:
  python fetch_moth_families.py
"""

import time
import requests
import pandas as pd
from pathlib import Path

# -------------------------------------------------------
# Config
# -------------------------------------------------------
INPUT_CSV  = Path(__file__).parent / "mothDK_classes.csv"
OUTPUT_CSV = Path(__file__).parent / "mothDK_classes_with_family.csv"

GBIF_URL       = "https://api.gbif.org/v1/species/match"
DELAY_SEC      = 0.15   
MAX_RETRIES    = 3
RETRY_DELAY    = 5.0    


# -------------------------------------------------------
# Single-species lookup
# -------------------------------------------------------
def lookup_gbif(species_name: str) -> dict:
    """
    Query GBIF species/match for one name.
    Returns dict with family, order, genus, match_type, confidence.
    Falls back to 'Unknown' on failure.
    """
    params = {
        "name":    species_name,
        "kingdom": "Animalia",
        "rank":    "SPECIES",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(GBIF_URL, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            # matchType: EXACT, FUZZY, HIGHERRANK, NONE
            match_type = data.get("matchType", "NONE")
            confidence = data.get("confidence", 0)

            # Warn on low-confidence or no-match
            if match_type == "NONE" or confidence < 70:
                print(f"  [WARN] Low match for '{species_name}': "
                      f"matchType={match_type}, confidence={confidence}")

            return {
                "family":          data.get("family",  "Unknown"),
                "order":           data.get("order",   "Unknown"),
                "genus":           data.get("genus",   "Unknown"),
                "match_type":      match_type,
                "gbif_confidence": confidence,
            }

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] '{species_name}': {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  [FAIL] '{species_name}' after {MAX_RETRIES} attempts: {e}")
                return {
                    "family":          "Unknown",
                    "order":           "Unknown",
                    "genus":           "Unknown",
                    "match_type":      "ERROR",
                    "gbif_confidence": 0,
                }


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    df = pd.read_csv(INPUT_CSV)
    n  = len(df)
    print(f"Loaded {n} species from {INPUT_CSV.name}")
    print(f"Querying GBIF API... (estimated time: ~{n * DELAY_SEC / 60:.1f} min)\n")

    results = []
    unknown_count = 0

    for i, row in df.iterrows():
        species = row["class_name"]
        pct     = (i + 1) / n * 100
        print(f"[{i+1:3d}/{n}] {pct:5.1f}%  {species}", end=" ... ", flush=True)

        info = lookup_gbif(species)
        results.append(info)

        status = info["family"] if info["family"] != "Unknown" else "??"
        print(f"{status}  (conf={info['gbif_confidence']})")

        if info["family"] == "Unknown":
            unknown_count += 1

        time.sleep(DELAY_SEC)

    # Merge results back
    result_df = pd.DataFrame(results)
    out_df    = pd.concat([df.reset_index(drop=True), result_df], axis=1)

    out_df.to_csv(OUTPUT_CSV, index=False)

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Output : {OUTPUT_CSV}")
    print(f"  Total  : {n} species")
    print(f"  Resolved: {n - unknown_count} ({(n - unknown_count)/n*100:.1f}%)")
    print(f"  Unknown : {unknown_count}")

    if unknown_count > 0:
        print("\n  Unresolved species:")
        mask = out_df["family"] == "Unknown"
        for name in out_df.loc[mask, "class_name"]:
            print(f"    - {name}")

    print("\n  Family distribution (top 15):")
    fam_counts = out_df["family"].value_counts().head(15)
    for fam, cnt in fam_counts.items():
        bar = "#" * int(cnt / n * 40)
        print(f"    {fam:<25s} {cnt:3d}  {bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()
