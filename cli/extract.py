"""
cli/extract.py
--------------
CLI command: extract BioCLIP-2 embeddings and save them to disk.

Usage:
    python cli/extract.py --config configs/eumoth.yaml

Output:
    <output_dir>/features.npz   -- embeddings, labels, class names
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io     import set_seed, load_config, save_features, ensure_dir
from src.data.dataset import load_dataset
from src.models.extractors import build_extractor
from src.utils.pretty import header, subheader, metrics_table, done, saved


def main(config_path: str = None) -> None:
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Extract BioCLIP-2 image embeddings."
        )
        parser.add_argument("--config", required=True,
                            help="Path to YAML config file.")
        args        = parser.parse_args()
        config_path = args.config

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    header("STEP: Feature Extraction")

    # Step 1: Discover images
    subheader("Loading dataset")
    t0 = time.time()
    image_paths, labels, class_names = load_dataset(config)
    print("    Dataset    : " + config["dataset"]["name"])
    print("    Images     : " + str(len(image_paths)))
    print("    Classes    : " + str(len(class_names)))

    # Step 2: Load model
    subheader("Loading model")
    extractor = build_extractor(config)
    print("    Model      : " + config["model"]["name"])
    print("    Pretrained : " + config["model"]["pretrained"])

    # Step 3: Extract
    subheader("Extracting embeddings")
    features = extractor.extract(image_paths)

    elapsed = time.time() - t0
    metrics_table([
        {"label": "Feature shape",  "value": str(features.shape)},
        {"label": "Dim per image",  "value": features.shape[1]},
        {"label": "Time",           "value": "{:.1f}s".format(elapsed)},
    ], title="Extraction complete")

    # Step 4: Save
    out_path = os.path.join(output_dir, "features.npz")
    save_features(out_path, features, labels, image_paths, class_names)
    saved("features.npz", out_path)
    done(output_dir)


if __name__ == "__main__":
    main()
