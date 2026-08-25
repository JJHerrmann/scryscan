"""Concurrently download the small-resolution card images for the filtered corpus.
Resumable: skips any file that already exists on disk."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(DIRECTORY, "corpus.jsonl")
IMAGES_DIR = os.path.join(DIRECTORY, "images")
UA = "RookworksDinoBuild/1.0 (jaherrmann@gmail.com)"
WORKERS = 24

os.makedirs(IMAGES_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": UA})


def load_records():
    records = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def download_one(rec):
    dest = os.path.join(IMAGES_DIR, rec["id"] + ".jpg")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    try:
        resp = session.get(rec["image_url"], timeout=20)
        if resp.status_code == 200 and resp.content:
            tmp = dest + ".tmp"
            with open(tmp, "wb") as f:
                f.write(resp.content)
            os.replace(tmp, dest)
            return "ok"
        return "http_%d" % resp.status_code
    except Exception as e:
        return "err_%s" % type(e).__name__


def main():
    records = load_records()
    print("corpus size:", len(records))

    counts = {"ok": 0, "skip": 0}
    errors = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, rec): rec for rec in records}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result in ("ok", "skip"):
                counts[result] += 1
            else:
                errors += 1
                if errors <= 20:
                    print("  error:", result, futures[fut]["id"], futures[fut]["name"])
            if done % 2000 == 0:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                print("progress: %d/%d (ok=%d skip=%d err=%d) %.1f/sec elapsed=%.0fs" % (
                    done, len(records), counts["ok"], counts["skip"], errors, rate, elapsed
                ))

    elapsed = time.time() - start
    print("DONE. ok=%d skip=%d err=%d elapsed=%.0fs" % (counts["ok"], counts["skip"], errors, elapsed))


if __name__ == "__main__":
    main()
