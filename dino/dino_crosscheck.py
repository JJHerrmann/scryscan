"""Cross-check utility: given a captured card photo, return DINOv2's top-3
visual-similarity candidates alongside their confidence. Given the measured
accuracy (see ACCURACY_REPORT.md), this is NOT reliable enough to use as a
standalone/primary identifier - it's wired in here as a secondary signal:

- If DINOv2's top-1 candidate agrees with the LLM-OCR read, that's a useful
  extra confidence signal (both independent methods agreeing).
- If they disagree, surface both to a human/LLM for disambiguation rather
  than trusting either blindly.
- Low similarity scores (roughly <0.65 based on the validation set) should
  be treated as "no confident visual match" and ignored.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dino_lookup import identify  # noqa: E402

LOW_CONFIDENCE_THRESHOLD = 0.65  # below this, treat as "no match" per validation set observations


def cross_check(image_path, llm_name=None, llm_set=None, llm_collector_number=None):
    """Returns a dict summarizing DINOv2's opinion and whether it agrees with
    an LLM-OCR read (if provided)."""
    results = identify(image_path, top_k=3)
    top_sim, top_meta, top_rot = results[0]

    confident = top_sim >= LOW_CONFIDENCE_THRESHOLD
    agrees_with_llm = None
    if llm_name:
        norm = lambda s: "".join(ch.lower() for ch in s if ch.isalnum())
        agrees_with_llm = norm(top_meta["name"]) == norm(llm_name)

    return {
        "dino_top1_name": top_meta["name"],
        "dino_top1_set": top_meta["set"],
        "dino_top1_collector_number": top_meta["collector_number"],
        "dino_top1_similarity": round(top_sim, 4),
        "dino_confident": confident,
        "agrees_with_llm": agrees_with_llm,
        "candidates": [
            {"name": m["name"], "set": m["set"], "collector_number": m["collector_number"], "similarity": round(s, 4)}
            for s, m, r in results
        ],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python dino_crosscheck.py <image_path> [llm_name]")
        sys.exit(1)
    llm_name = sys.argv[2] if len(sys.argv) > 2 else None
    result = cross_check(sys.argv[1], llm_name=llm_name)
    import json
    print(json.dumps(result, indent=2))
