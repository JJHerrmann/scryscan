"""Run DINOv2 lookup against the 40 manually-verified ground-truth events and
report real accuracy: name match, exact printing match, similarity scores,
and per-query timing."""
import json
import os
import time

from dino_lookup import identify

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SCRATCHPAD = os.path.dirname(DIRECTORY)
EVENTS_DIR = os.path.join(SCRATCHPAD, "events")
GT_PATH = os.path.join(DIRECTORY, "ground_truth.json")


def norm_name(s):
    return "".join(ch.lower() for ch in s if ch.isalnum())


def main():
    with open(GT_PATH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    name_correct = 0
    exact_correct = 0
    total = 0
    timings = []
    failures = []

    for idx_str, gt in sorted(ground_truth.items(), key=lambda kv: int(kv[0])):
        idx = int(idx_str)
        path = os.path.join(EVENTS_DIR, "card_%04d.png" % idx)
        if not os.path.exists(path):
            print("MISSING FILE for event", idx)
            continue

        t0 = time.time()
        results = identify(path, top_k=3)
        dt = time.time() - t0
        timings.append(dt)
        total += 1

        top_sim, top_meta, top_rot = results[0]
        name_ok = norm_name(top_meta["name"]) == norm_name(gt["name"])
        exact_ok = name_ok and top_meta["set"] == gt["set"] and str(top_meta["collector_number"]) == str(gt["collector_number"])

        if name_ok:
            name_correct += 1
        if exact_ok:
            exact_correct += 1

        status = "EXACT" if exact_ok else ("NAME_ONLY" if name_ok else "WRONG")
        print("event %3d [%s] sim=%.4f rot=%3d  got=%r (%s/%s)  expected=%r (%s/%s)" % (
            idx, status, top_sim, top_rot,
            top_meta["name"], top_meta["set"], top_meta["collector_number"],
            gt["name"], gt["set"], gt["collector_number"],
        ))

        if not name_ok:
            failures.append({
                "event": idx, "got": top_meta["name"], "got_set_cn": "%s/%s" % (top_meta["set"], top_meta["collector_number"]),
                "expected": gt["name"], "expected_set_cn": "%s/%s" % (gt["set"], gt["collector_number"]),
                "sim": top_sim,
                "top3": [(m["name"], m["set"], m["collector_number"], round(s, 4)) for s, m, r in results],
            })

    print()
    print("=== SUMMARY ===")
    print("total samples:", total)
    print("name-correct (top-1): %d/%d = %.1f%%" % (name_correct, total, 100 * name_correct / total))
    print("exact-printing-correct (top-1): %d/%d = %.1f%%" % (exact_correct, total, 100 * exact_correct / total))
    print("avg query time: %.3fs (searches 4 rotations x %d candidates)" % (sum(timings) / len(timings), 98456))
    print()
    if failures:
        print("=== FAILURES (name mismatch) ===")
        for f in failures:
            print(json.dumps(f, indent=2))


if __name__ == "__main__":
    main()
