"""Given a card photo, find its nearest match in the local DINOv2 embedding index.

Handles rotation robustness (our captured photos are sometimes upside-down or
sideways) by embedding the query at all 4 rotations and keeping the best match.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(DIRECTORY, "embeddings.npy")
METADATA_PATH = os.path.join(DIRECTORY, "embeddings_meta.jsonl")

MODEL_NAME = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_processor = None
_matrix = None
_meta = None


def _load():
    global _model, _processor, _matrix, _meta
    if _model is not None:
        return
    _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    _matrix = np.load(EMBEDDINGS_PATH)  # (N, D), L2-normalized
    _meta = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                _meta.append(json.loads(line))
    assert len(_meta) == _matrix.shape[0], "metadata/embeddings count mismatch"


def _embed_image(img: Image.Image) -> np.ndarray:
    inputs = _processor(images=[img.convert("RGB")], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = _model(**inputs)
    vec = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
    vec = vec / np.linalg.norm(vec)
    return vec.astype(np.float32)


def identify(image_path, top_k=3, try_rotations=True):
    """Returns a list of top_k (similarity, metadata_dict, rotation_degrees) tuples,
    best match first, searched across all 4 rotations of the query image."""
    _load()
    img = Image.open(image_path)

    rotations = [0, 90, 180, 270] if try_rotations else [0]
    best_overall = []

    for deg in rotations:
        rotated = img.rotate(-deg, expand=True) if deg else img
        vec = _embed_image(rotated)
        sims = _matrix @ vec  # cosine similarity since both are L2-normalized
        top_idx = np.argpartition(-sims, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        for i in top_idx:
            best_overall.append((float(sims[i]), _meta[i], deg))

    best_overall.sort(key=lambda t: -t[0])
    # De-dup by card id, keep best rotation's score for each.
    seen = set()
    deduped = []
    for sim, meta, deg in best_overall:
        if meta["id"] in seen:
            continue
        seen.add(meta["id"])
        deduped.append((sim, meta, deg))
        if len(deduped) >= top_k:
            break
    return deduped


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python dino_lookup.py <image_path>")
        sys.exit(1)
    results = identify(sys.argv[1])
    for sim, meta, deg in results:
        print("sim=%.4f rot=%d  %s (%s #%s)" % (
            sim, deg, meta["name"], meta["set"].upper(), meta["collector_number"]
        ))
