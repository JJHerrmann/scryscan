"""Run the DINOv2 lookup against every real captured card_NNNN.png event photo
from this session, and save the results for accuracy comparison."""
import csv
import json
import os
import time

from dino_lookup import identify

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SCRATCHPAD = os.path.dirname(DIRECTORY)
EVENTS_DIR = os.path.join(SCRATCHPAD, "events")
EVENTS_LOG = os.path.join(EVENTS_DIR, "events.jsonl")
OUT_CSV = os.path.join(DIRECTORY, "dino_results.csv")


def load_events():
    events = []
    with open(EVENTS_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    events.sort(key=lambda r: r["index"])
    return events


def main():
    events = load_events()
    print("total events:", len(events))

    fields = [
        "event_index", "file",
        "top1_name", "top1_set", "top1_cn", "top1_sim", "top1_rotation",
        "top2_name", "top2_set", "top2_cn", "top2_sim",
        "top3_name", "top3_set", "top3_cn", "top3_sim",
        "query_seconds",
    ]

    start = time.time()
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for i, event in enumerate(events):
            path = os.path.join(EVENTS_DIR, event["file"])
            if not os.path.exists(path):
                continue
            t0 = time.time()
            try:
                results = identify(path, top_k=3)
            except Exception as e:
                print("  FAILED on", event["file"], type(e).__name__, e)
                continue
            dt = time.time() - t0

            row = {"event_index": event["index"], "file": event["file"], "query_seconds": round(dt, 3)}
            for rank, key in enumerate(["top1", "top2", "top3"]):
                if rank < len(results):
                    sim, meta, rot = results[rank]
                    row[key + "_name"] = meta["name"]
                    row[key + "_set"] = meta["set"]
                    row[key + "_cn"] = meta["collector_number"]
                    row[key + "_sim"] = round(sim, 4)
                    if key == "top1":
                        row["top1_rotation"] = rot
            writer.writerow(row)

            if (i + 1) % 25 == 0:
                elapsed = time.time() - start
                print("progress: %d/%d  elapsed=%.0fs  avg=%.2fs/query" % (
                    i + 1, len(events), elapsed, elapsed / (i + 1)
                ))

    total_elapsed = time.time() - start
    print("DONE. total=%.0fs avg=%.2fs/query. Results at %s" % (
        total_elapsed, total_elapsed / len(events), OUT_CSV
    ))


if __name__ == "__main__":
    main()
