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
        """L2-normalize along the feature dimension."""
        return features / features.norm(dim=-1, keepdim=True)

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

        print(f"[model] Loading {name} from '{pretrained}' ...")

        try:
            self.model, _, self.preprocess = \
                open_clip.create_model_and_transforms(pretrained)
            self.tokenizer = open_clip.get_tokenizer(pretrained)
        except Exception as e:
            # Fallback to bioclip if bioclip-2 is unavailable
            print(f"[model] Failed to load '{pretrained}': {e}")
            print("[model] Falling back to hf-hub:imageomics/bioclip")
            fallback = "hf-hub:imageomics/bioclip"
            self.model, _, self.preprocess = \
                open_clip.create_model_and_transforms(fallback)
            self.tokenizer = open_clip.get_tokenizer(fallback)

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
                    print(f"[warn] Could not load {path}: {e} — using zeros")
                    # Use a zero tensor as a placeholder so batch size stays fixed
                    images.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                if self.normalize:
                    features = self._normalize(features)

            all_features.append(features.cpu().numpy())

        return np.vstack(all_features)

    def encode_text(self, class_names: List[str],
                    prompt_templates: List[str] = None) -> np.ndarray:
        """
        Encode text prompts and return normalized text embeddings.
        Used by text-guided K-Means initialization.

        WHY ENSEMBLING:
            No single prompt perfectly represents a biological species in
            embedding space. Averaging over a small set of complementary
            prompts produces a more stable, central estimate — this is the
            same technique used in the original CLIP paper and has been
            shown to improve zero-shot alignment consistently.

        WHY TAXONOMIC TEMPLATES:
            BioCLIP-2 was trained on TreeOfLife-200M, where the text paired
            with each image is structured taxonomic text, not image captions.
            Prompts that include taxonomic context (order, class, etc.) are
            therefore more aligned with the model's training distribution than
            generic "a photo of X" prompts.

            The genus is always the first word of a binomial scientific name,
            so it is parsed directly from the folder name without any external
            lookup (e.g. "abrostola_tripartita" -> genus="abrostola").

        Args:
            class_names:       List of class folder name strings.
            prompt_templates:  List of template strings with {name} and {genus}
                               placeholders. If None, uses the default set below.

        Returns:
            NumPy array of shape (num_classes, D), L2-normalized.
            Each row is the mean of all template embeddings for that class,
            re-normalized to unit length.
        """
        # Default templates. These follow the format closest to BioCLIP-2's
        # training data (taxonomic hierarchy first), plus a few descriptive
        # alternatives to reduce variance through ensembling.
        #
        # {name}  -> full scientific name with spaces  (e.g. "abrostola tripartita")
        # {genus} -> genus only, first word             (e.g. "abrostola")
        #
        # Override these in your YAML config under model.prompt_templates.
        if prompt_templates is None:
            prompt_templates = [
                "{name}",
                "a photo of {name}",
                "a photo of {name}, a species",
                "{genus} {name}",
            ]

        print(f"[text] Encoding {len(class_names)} classes "
              f"with {len(prompt_templates)} prompt templates (ensemble) ...")

        all_class_embeddings = []

        for name in class_names:
            # Parse the scientific name and genus from the folder name
            readable_name = name.replace("_", " ").strip()
            genus         = readable_name.split()[0] if " " in readable_name \
                            else readable_name

            # Fill in the templates for this class
            filled = [
                t.format(name=readable_name, genus=genus)
                for t in prompt_templates
            ]

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
            from torchvision import transforms
        except ImportError:
            raise ImportError("DINOv2 extractor requires: pip install timm torchvision")

        # Default to the large model
        model_name = self.config["model"].get("pretrained", "vit_large_patch14_dinov2.lvd142m")
        print(f"[model] Loading DINOv2: {model_name}")

        self.model     = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model     = self.model.to(self.device)
        self.model.eval()

        # Standard DINOv2 preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"[model] DINOv2 ready on device: {self.device}")

    def extract(self, image_paths: List[str]) -> np.ndarray:
        all_features = []

        for i in tqdm(range(0, len(image_paths), self.batch_size),
                      desc="Extracting DINOv2 features"):
            batch_paths = image_paths[i : i + self.batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception:
                    images.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor)
                if self.normalize:
                    features = self._normalize(features)

            all_features.append(features.cpu().numpy())

        return np.vstack(all_features)


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
