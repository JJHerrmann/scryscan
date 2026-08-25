"""Compute DINOv2 embeddings for every downloaded card image and save as a single
numpy matrix + parallel metadata list, for fast local nearest-neighbor search."""
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(DIRECTORY, "corpus.jsonl")
IMAGES_DIR = os.path.join(DIRECTORY, "images")
EMBEDDINGS_PATH = os.path.join(DIRECTORY, "embeddings.npy")
METADATA_PATH = os.path.join(DIRECTORY, "embeddings_meta.jsonl")

MODEL_NAME = "facebook/dinov2-base"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_records():
    records = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    print("device:", DEVICE)
    print("loading model:", MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    records = load_records()
    print("corpus records:", len(records))

    # Only embed records whose image actually downloaded successfully.
    usable = []
    for rec in records:
        path = os.path.join(IMAGES_DIR, rec["id"] + ".jpg")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            rec["_image_path"] = path
            usable.append(rec)
    print("usable (image present):", len(usable))

    all_embeddings = []
    meta_f = open(METADATA_PATH, "w", encoding="utf-8")

    start = time.time()
    processed = 0
    failed = 0

    with torch.no_grad():
        for batch_start in range(0, len(usable), BATCH_SIZE):
            batch_records = usable[batch_start:batch_start + BATCH_SIZE]
            images = []
            good_records = []
            for rec in batch_records:
                try:
                    img = Image.open(rec["_image_path"]).convert("RGB")
                    images.append(img)
                    good_records.append(rec)
                except Exception as e:
                    failed += 1
                    print("  failed to open", rec["_image_path"], type(e).__name__)

            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            # CLS token pooled output as the card's embedding vector.
            embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)  # L2-normalize for cosine sim via dot product

            all_embeddings.append(embeds.astype(np.float32))
            for rec in good_records:
                meta = {k: v for k, v in rec.items() if k != "_image_path"}
                meta_f.write(json.dumps(meta) + "\n")

            processed += len(good_records)
            if processed % (BATCH_SIZE * 20) == 0 or batch_start + BATCH_SIZE >= len(usable):
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                print("progress: %d/%d  %.1f/sec  elapsed=%.0fs  eta=%.0fs" % (
                    processed, len(usable), rate, elapsed,
                    (len(usable) - processed) / rate if rate > 0 else -1
                ))

    meta_f.close()
    matrix = np.concatenate(all_embeddings, axis=0)
    np.save(EMBEDDINGS_PATH, matrix)
    print("DONE. embeddings shape:", matrix.shape, "failed:", failed)
    print("saved to:", EMBEDDINGS_PATH)
    print("metadata saved to:", METADATA_PATH)


if __name__ == "__main__":
    main()
