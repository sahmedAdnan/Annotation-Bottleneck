"""
src/models/extractors.py
------------------------
Embedding extraction using foundation models.

Currently supported:
  - bioclip2  (ViT-L/14, hf-hub:imageomics/bioclip-2)
  - bioclip   (ViT-B/16, hf-hub:imageomics/bioclip)
  - clip      (ViT-L/14, openai/clip-vit-large-patch14)
  - dinov2    (facebook/dinov2-large via timm or transformers)

All extractors follow the same interface:
    extractor = build_extractor(config)
    features  = extractor.extract(image_paths)

The returned features are a NumPy array of shape (N, D),
optionally L2-normalized based on config.model.normalize_embeddings.
"""

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing import List


# -------------------------------------------------------
# Base Interface
# -------------------------------------------------------

class BaseExtractor:
    """
    Abstract base class. All extractors must implement:
      - load_model()
      - extract(image_paths) -> np.ndarray
    """

    def __init__(self, config: dict):
        self.config    = config
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.normalize = config["model"].get("normalize_embeddings", True)
        self.batch_size = config["model"].get("batch_size", 32)
        self.num_workers = config["model"].get("num_workers", 0)
        self.model      = None
        self.preprocess = None

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """L2-normalize along the feature dimension (safe for zero vectors)."""
        norms = features.norm(dim=-1, keepdim=True)
        # Avoid division by zero: replace zero norms with 1.0 so the
        # zero vector stays zero rather than becoming NaN.
        norms = norms.clamp(min=1e-8)
        return features / norms

    def extract(self, image_paths: List[str]) -> np.ndarray:
        raise NotImplementedError


# -------------------------------------------------------
# BioCLIP / BioCLIP-2 (open_clip based)
# -------------------------------------------------------

class OpenCLIPExtractor(BaseExtractor):
    """
    Extractor for any model available through the open_clip library.
    This covers BioCLIP, BioCLIP-2, and standard CLIP models.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._load_model()

    def _load_model(self):
        import open_clip

        name       = self.config["model"]["name"]
        pretrained = self.config["model"]["pretrained"]
        arch       = self.config["model"].get("arch", None)

        print(f"[model] Loading {name} from '{pretrained}' ...")

        if arch is not None:
            # Standard open_clip model with separate arch + pretrained.
            # Example: arch='ViT-L-14', pretrained='openai'  (CLIP)
            self.model, _, self.preprocess = \
                open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            self.tokenizer = open_clip.get_tokenizer(arch)
        else:
            # HF-Hub model: pretrained='hf-hub:imageomics/bioclip-2'
            self.model, _, self.preprocess = \
                open_clip.create_model_and_transforms(pretrained)
            self.tokenizer = open_clip.get_tokenizer(pretrained)

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[model] Ready on device: {self.device}")

    def extract(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract image embeddings in batches.

        Args:
            image_paths: List of full image file paths.

        Returns:
            NumPy array of shape (N, D), optionally L2-normalized.
        """
        all_features = []
        n_failed = 0
        n_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        print(f"[extract] {len(image_paths)} images, "
              f"batch_size={self.batch_size}, {n_batches} batches")

        # Determine the correct input size from the model's preprocessing
        # by running the preprocess on a dummy image once.
        dummy = Image.new("RGB", (256, 256))
        input_shape = tuple(self.preprocess(dummy).shape)  # (C, H, W)

        for i in tqdm(range(0, len(image_paths), self.batch_size),
                      desc="Extracting features"):

            batch_paths = image_paths[i : i + self.batch_size]

            # Load and preprocess each image in the batch
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    n_failed += 1
                    print(f"[warn] Could not load {path}: {e} - using zeros")
                    # Use a zero tensor matching the model's expected input size
                    images.append(torch.zeros(input_shape))

            batch_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                if self.normalize:
                    features = self._normalize(features)

            all_features.append(features.cpu().numpy())

        if n_failed > 0:
            print(f"[warn] {n_failed}/{len(image_paths)} images failed to load")

        result = np.vstack(all_features)
        print(f"[extract] Done: {result.shape[0]} embeddings, dim={result.shape[1]}")

        # Validate: no NaN values should be present in the output
        n_nan_rows = int(np.any(np.isnan(result), axis=1).sum())
        if n_nan_rows > 0:
            raise RuntimeError(
                f"NaN detected in {n_nan_rows} feature vectors after extraction. "
                f"This indicates corrupted images or model issues. "
                f"Run filter_valid_images() before extraction to remove bad files."
            )

        return result

    def encode_text(self, class_names: List[str],
                    prompt_templates: List[str] = None,
                    taxonomy_table: dict = None) -> np.ndarray:
        """
        Encode text prompts and return normalized text embeddings.
        Used by the annotation pipeline for zero-shot text matching.

        WHY ENSEMBLING:
            No single prompt perfectly represents a biological species in
            embedding space. Averaging over a small set of complementary
            prompts produces a more stable, central estimate — this is the
            same technique used in the original CLIP paper and has been
            shown to improve zero-shot alignment consistently.

        WHY TAXONOMIC TEMPLATES:
            BioCLIP-2 was trained on TreeOfLife-200M, where the text paired
            with each image is structured taxonomic text, not image captions.
            Prompts that include taxonomic context (order, family, genus) are
            therefore more aligned with the model's training distribution than
            generic "a photo of X" prompts.

            The genus is always the first word of a binomial scientific name,
            so it is parsed directly from the folder name without any external
            lookup (e.g. "abrostola_tripartita" -> genus="abrostola").

            Family and order are looked up from a taxonomy table (pre-computed
            CSV or GBIF query). If a template uses {family} or {order} but the
            taxonomy entry is missing, that template is skipped for that class.

        Args:
            class_names:       List of class folder name strings.
            prompt_templates:  List of template strings with {name}, {genus},
                               {family}, {order} placeholders.
                               If None, uses the default set below.
            taxonomy_table:    Optional dict mapping class_name -> taxonomy entry
                               with keys: genus, family, order.
                               Built by src/annotation/taxonomy.py.

        Returns:
            NumPy array of shape (num_classes, D), L2-normalized.
            Each row is the mean of all template embeddings for that class,
            re-normalized to unit length.
        """
        # Default templates. These follow the format closest to BioCLIP-2's
        # training data (taxonomic hierarchy first), plus a few descriptive
        # alternatives to reduce variance through ensembling.
        #
        # {name}   -> full scientific name with spaces  (e.g. "abrostola tripartita")
        # {genus}  -> genus only, first word             (e.g. "abrostola")
        # {family} -> taxonomic family                   (e.g. "Noctuidae")
        # {order}  -> taxonomic order                    (e.g. "Lepidoptera")
        #
        # Override these in your YAML config under model.prompt_templates.
        if prompt_templates is None:
            prompt_templates = [
                "{name}",
                "a photo of {name}",
                "a photo of {name}, a species",
                "a species of genus {genus}",
            ]

        if taxonomy_table is None:
            taxonomy_table = {}

        print(f"[text] Encoding {len(class_names)} classes "
              f"with {len(prompt_templates)} prompt templates (ensemble) ...")
        if taxonomy_table:
            n_with_family = sum(1 for n in class_names
                                if taxonomy_table.get(n, {}).get("family"))
            print(f"[text] Taxonomy available for {n_with_family}/{len(class_names)} "
                  f"classes (family-level)")

        all_class_embeddings = []

        for name in class_names:
            # Parse the scientific name and genus from the folder name.
            # Strip numeric prefixes like "001." (CUB-200 format).
            clean = name
            if clean and clean.split(".")[0].isdigit():
                clean = clean.split(".", 1)[1]
            readable_name = clean.replace("_", " ").strip()
            genus         = readable_name.split()[0] if " " in readable_name \
                            else readable_name

            # Look up family and order from taxonomy table
            tax = taxonomy_table.get(name, {})
            family = tax.get("family", "")
            order  = tax.get("order", "")

            # Fill in the templates, skipping any that require missing fields
            filled = []
            for t in prompt_templates:
                if "{family}" in t and not family:
                    continue
                if "{order}" in t and not order:
                    continue
                try:
                    filled.append(t.format(
                        name=readable_name, genus=genus,
                        family=family, order=order,
                    ))
                except KeyError:
                    continue

            if not filled:
                # Fallback: use just the readable name
                filled = [readable_name]

            # Encode all templates for this class
            tokens   = self.tokenizer(filled).to(self.device)
            with torch.no_grad():
                embeds = self.model.encode_text(tokens)       # (n_templates, D)
                embeds = self._normalize(embeds)              # normalize each

            # Average across templates, then re-normalize.
            # Re-normalizing after averaging is important: the mean of unit
            # vectors is not itself a unit vector.
            mean_embed = embeds.mean(dim=0)                   # (D,)
            mean_embed = mean_embed / mean_embed.norm()       # back to unit sphere

            all_class_embeddings.append(mean_embed.cpu().numpy())

        result = np.stack(all_class_embeddings, axis=0)       # (num_classes, D)
        print(f"[text] Text centroids shape: {result.shape}")
        return result

    def get_logit_scale(self) -> float:
        """Return the learned temperature scaling factor."""
        return self.model.logit_scale.exp().item()


# -------------------------------------------------------
# DINOv2 Extractor (via timm)
# -------------------------------------------------------

class DINOv2Extractor(BaseExtractor):
    """
    Extractor for DINOv2 models via the timm library.

    Install: pip install timm
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._load_model()

    def _load_model(self):
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError:
            raise ImportError("DINOv2 extractor requires: pip install timm")

        # Default to the large model
        model_name = self.config["model"].get("pretrained", "vit_large_patch14_dinov2.lvd142m")
        print(f"[model] Loading DINOv2: {model_name}")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Let timm derive the correct input size and normalization from the model.
        # vit_large_patch14_dinov2.lvd142m expects 518x518, not 224x224.
        # Hardcoding 224 causes an AssertionError inside patch_embed.
        data_config     = resolve_data_config({}, model=self.model)
        self.preprocess = create_transform(**data_config)

        # Store input spatial size for the error-fallback placeholder in extract().
        # data_config["input_size"] is (C, H, W), e.g. (3, 518, 518).
        self._input_size = tuple(data_config["input_size"])

        input_size = data_config.get("input_size", "unknown")
        print(f"[model] DINOv2 ready on device: {self.device}  "
              f"(input_size={input_size})")

    def extract(self, image_paths: List[str]) -> np.ndarray:
        all_features = []
        n_failed = 0
        n_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        print(f"[extract] {len(image_paths)} images, "
              f"batch_size={self.batch_size}, {n_batches} batches")

        for i in tqdm(range(0, len(image_paths), self.batch_size),
                      desc="Extracting DINOv2 features"):
            batch_paths = image_paths[i : i + self.batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    n_failed += 1
                    print(f"[warn] Could not load {path}: {e} - using zeros")
                    images.append(torch.zeros(self._input_size))

            batch_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor)
                if self.normalize:
                    features = self._normalize(features)

            all_features.append(features.cpu().numpy())

        if n_failed > 0:
            print(f"[warn] {n_failed}/{len(image_paths)} images failed to load")

        result = np.vstack(all_features)
        print(f"[extract] Done: {result.shape[0]} embeddings, dim={result.shape[1]}")

        # Validate: no NaN values should be present
        n_nan_rows = int(np.any(np.isnan(result), axis=1).sum())
        if n_nan_rows > 0:
            raise RuntimeError(
                f"NaN detected in {n_nan_rows} feature vectors after extraction. "
                f"Run filter_valid_images() before extraction to remove bad files."
            )

        return result


# -------------------------------------------------------
# Factory: Build the Right Extractor from Config
# -------------------------------------------------------

def build_extractor(config: dict) -> BaseExtractor:
    """
    Instantiate and return the correct extractor based on config.model.name.

    Supported values:
        bioclip2, bioclip, clip  ->  OpenCLIPExtractor
        dinov2                   ->  DINOv2Extractor
    """
    name = config["model"]["name"].lower()

    if name in ("bioclip2", "bioclip", "clip"):
        return OpenCLIPExtractor(config)
    elif name == "dinov2":
        return DINOv2Extractor(config)
    else:
        raise ValueError(
            f"Unknown model: '{name}'. "
            "Supported: bioclip2, bioclip, clip, dinov2"
        )
