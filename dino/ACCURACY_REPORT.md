# DINOv2 local card recognition - accuracy report

## What was built
- Downloaded Scryfall's full `default_cards` bulk export, filtered to English/paper
  printings, excluding tokens/art-series/emblems: **98,456 unique printings**.
- Downloaded each printing's "small" reference image (98,456 images, 0 failures).
- Computed DINOv2 (`facebook/dinov2-base`) CLS-token embeddings for the full corpus
  on GPU (RTX 3060): **98,456 x 768 float32**, ~23 min.
- Built a rotation-robust lookup (`dino_lookup.py`): embeds the query at 0/90/180/270
  degrees and returns the best match across all four, since captured photos are
  sometimes upside-down or sideways.
- Query time: ~0.22s per lookup (4 rotations x 98,456-candidate cosine search).

## Accuracy (40 manually-verified real captures, spread across the whole session)
- **Name-correct (top-1): 23/40 = 57.5%** (one apparent miss was actually a
  ground-truth naming mismatch on a split card - the exact printing matched).
- **Exact-printing-correct (top-1): 15/40 = 37.5%**.

This is meaningfully worse than the existing LLM-OCR pipeline's accuracy when it
successfully reads a collector number (near-100% via exact Scryfall lookup in that
case). It is **not accurate enough to use as a standalone/primary identifier**.

## Diagnosis
Tested the "reference images are too low-res" hypothesis directly: re-fetched 4
failed cases' correct answer + wrong top-1 answer at Scryfall's "normal" (488x680)
resolution instead of "small" (146x204) and re-compared similarity to the query.
**Only 1 of 4 flipped to correct.** Resolution is a minor contributing factor, not
the dominant cause.

More likely causes, based on the failure pattern (wrong matches are usually
generically similar - same border color, similar art darkness/composition, similar
layout - rather than random):
- `dinov2-base`'s single CLS-token embedding may not carry enough fine-grained
  detail to distinguish visually-similar cards (same frame style, similar art mood)
  under real-world webcam photo conditions (blur, off-angle, JPEG artifacts, mixed
  lighting) - as opposed to clean, well-lit product photography.
- Whole-card embedding conflates border/frame/layout similarity with the actual
  distinguishing content (name, specific art, collector info), which is exactly
  why the one comparable open project found (scryglass) does "dual-zone
  verification" (art box + full card separately) rather than a single embedding.

## What this is actually useful for right now
Not a replacement for the LLM-OCR pipeline. Wired in as `dino_crosscheck.py`: a
secondary signal that runs alongside the existing identify step and returns its
top-3 visual candidates + similarity score. Useful for:
- Extra confidence when it agrees with the LLM's read.
- A candidate list to disambiguate when the LLM's OCR read fails or looks wrong
  (occasionally the correct card IS in DINOv2's top-3 even when top-1 is wrong).
- Similarity scores below ~0.65 (observed in this validation set) correlate with
  "no confident match" and should be treated as noise.

## If this is worth pursuing further, in priority order
1. **Try `dinov2-large` or `dinov2-giant`** - more capacity for fine-grained
   detail, same pipeline otherwise. Straightforward re-embed (~30-45 min at this
   corpus size), worth testing given how cheap it is relative to the other options.
2. **Dual-zone matching (art crop + full card separately)**, matching what
   scryglass does - meaningfully more implementation work (needs to segment/crop
   the art box specifically) but addresses the actual failure pattern observed.
3. **Improve query image quality at capture time** (better/more even lighting,
   less blur) - free, doesn't touch the model, but only helps as much as photo
   quality was actually the bottleneck (untested how much this alone would move
   the numbers here).
4. **Fine-tune** on MTG-specific data (e.g. the `acidtib/tcg-mtg-cards` or
   `gabraken/mtg-detection` HuggingFace datasets found during earlier research) -
   highest expected payoff, substantially more effort than the above.
