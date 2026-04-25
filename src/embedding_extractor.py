"""
embedding_extractor.py — ResNet18 Appearance Embedding Extractor

Produces a 512-dimensional feature vector for each detected bounding-box crop.
This is the Python/PyTorch port of the getEmbedding() function from
the C++ utilities.cpp used in CS 5330.

Usage:
    extractor = EmbeddingExtractor(device="cuda")
    crop = frame[y1:y2, x1:x2]          # BGR numpy array
    emb  = extractor.extract(crop)       # (512,) numpy vector, L2-normalised
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class EmbeddingExtractor:
    """Extract 512-d appearance embeddings from image crops using ResNet18."""

    def __init__(self, device=None):
        """
        Parameters
        ----------
        device : str or None
            'cuda', 'cpu', or None (auto-select).
        """
        if device is None:
            device = "cpu"
        self.device = torch.device(device)

        # Load ResNet18 pre-trained on ImageNet; remove the final FC layer
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Keep everything up to (and including) the global average pool
        self.model = nn.Sequential(*list(base.children())[:-1])  # output: (B, 512, 1, 1)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalisation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),   # common Re-ID crop size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract(self, crop_bgr):
        """
        Extract a single 512-d embedding from a BGR numpy crop.

        Parameters
        ----------
        crop_bgr : np.ndarray, shape (H, W, 3), dtype uint8
            Cropped detection region in BGR format (OpenCV convention).

        Returns
        -------
        embedding : np.ndarray, shape (512,)
            L2-normalised feature vector.
        """
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
            return np.zeros(512, dtype=np.float32)

        # BGR → RGB
        crop_rgb = crop_bgr[:, :, ::-1].copy()
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)  # (1, 3, 128, 64)
        feat = self.model(tensor)           # (1, 512, 1, 1)
        feat = feat.squeeze().cpu().numpy()  # (512,)

        # L2 normalise
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat /= norm
        return feat.astype(np.float32)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_batch(self, crops_bgr):
        """
        Extract embeddings for a list of BGR crops.

        Returns
        -------
        embeddings : np.ndarray, shape (N, 512)
        """
        if len(crops_bgr) == 0:
            return np.empty((0, 512), dtype=np.float32)

        tensors = []
        valid_indices = []
        for i, crop in enumerate(crops_bgr):
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            crop_rgb = crop[:, :, ::-1].copy()
            tensors.append(self.transform(crop_rgb))
            valid_indices.append(i)

        results = np.zeros((len(crops_bgr), 512), dtype=np.float32)
        if len(tensors) == 0:
            return results

        batch = torch.stack(tensors).to(self.device)
        feats = self.model(batch).squeeze(-1).squeeze(-1).cpu().numpy()

        # L2 normalise each row
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        feats = feats / norms

        for idx, vi in enumerate(valid_indices):
            results[vi] = feats[idx]
        return results


def cosine_distance(emb_a, emb_b):
    """Cosine distance between two L2-normalised vectors: 1 − dot(a, b)."""
    return 1.0 - np.dot(emb_a, emb_b)