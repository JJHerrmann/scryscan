# scryscan

A fully local Magic: The Gathering card scanner. Point a webcam at your cards,
and it identifies them, confirms the exact printing against Scryfall, and
keeps a running tally with quantities — no cloud SaaS, no subscription, no
sending your collection to a third party.

Built for a "just point a camera at the pile" workflow rather than a scan-and-feed
rig: a browser page watches your webcam, detects when a new card is placed in
frame, and hands the photo off to a local vision model for identification.

## How it works

```
webcam (browser) -> capture_server.py -> local_card_scanner.py -> scanned_cards_local.csv
                          |                        |
                     events/*.png            LM Studio (local LLM)
                                                    |
                                              Scryfall API (confirm exact printing + price)
```

1. **`webcam_viewer.html`** — served locally, shows your webcam feed, lets you
   pick a camera, crop the view to just the card, and adjust camera focus/zoom
   if your hardware supports it. A frame-diffing "auto-detect" mode notices
   when a new card settles into frame and saves a snapshot.
2. **`capture_server.py`** — a small local HTTP server (no external deps
   beyond the standard library) that receives those snapshots, logs them, and
   serves the live "scanned cards" panel in the browser page.
3. **`local_card_scanner.py`** — polls for new snapshots, sends each one to a
   local vision-capable LLM running in [LM Studio](https://lmstudio.ai/) and
   asks it to read the card's name, collector number, set code, and whether
   it looks foil. It then confirms the exact printing against the
   [Scryfall API](https://scryfall.com/docs/api) (using the collector
   number/set when legible, falling back to a fuzzy name search otherwise),
   and maintains `scanned_cards_local.csv` — deduped by exact printing, with
   a running quantity count.

Everything after the initial LM Studio setup runs on your machine. The only
network calls at runtime are to Scryfall's free public API for card data.

## Setup

Requires Python 3.10+ and [LM Studio](https://lmstudio.ai/) with a
vision-capable local model loaded (this was built and tested against
`google/gemma-4-e4b`, i.e. Gemma 3n E4B — any multimodal model LM Studio can
serve should work, quality will vary by model).

```bash
git clone https://github.com/<you>/scryscan.git
cd scryscan
pip install -r requirements.txt
```

1. In LM Studio, start the local server (Developer tab), load a vision model,
   and enable "require API key" in the server settings. Copy the token.
2. Create `lmstudio_token.txt` in the project root containing just that token.
3. Start the capture server:
   ```bash
   python capture_server.py
   ```
4. Open **`http://127.0.0.1:8743/webcam_viewer.html`** in your browser (it
   must be loaded over `http://localhost`/`127.0.0.1`, not opened as a
   `file://` path — browsers won't grant camera access to local files
   reliably).
5. Pick your camera, click **Start**, optionally use **Adjust Crop** to frame
   just the card, and check **Auto-detect new card**.
6. In a second terminal, run the identifier:
   ```bash
   python local_card_scanner.py
   ```
   It'll pick up new captures, identify them, and grow
   `scanned_cards_local.csv` as you show it cards.

### Tuning notes

- `local_card_scanner.py` has a `SKIP_CONSECUTIVE_DUPLICATES` flag near the
  top. Leave it `False` if you're flipping through a stack that might have
  several copies of the same card back-to-back (each gets counted). Set it
  `True` if a single still card sitting in frame keeps re-triggering from
  camera jitter and inflating its count.
- Camera focus/zoom controls only appear if your webcam driver reports those
  capabilities to the browser — not all hardware does.

## The DINOv2 experiment (`dino/`)

An attempt at replacing the LLM-OCR step with local, open-weight visual
similarity search (DINOv2 embeddings + nearest-neighbor lookup against a
locally-indexed copy of every Scryfall card image) instead of asking a
vision-language model to read text off the card.

**Honest result: it doesn't beat the OCR approach.** Tested against 40
manually-verified real captures: 57.5% name-accuracy, 37.5% exact-printing
accuracy (top-1) — worse than the LLM-OCR pipeline gets when it can read a
collector number clearly. Full writeup, including what was tried to fix it
and what's likely actually needed, is in [`dino/ACCURACY_REPORT.md`](dino/ACCURACY_REPORT.md).

It's kept in the repo as `dino/dino_crosscheck.py` — a secondary signal you
can run alongside the LLM identification for extra confidence or a candidate
list when OCR fails, not a standalone replacement. Building the index
yourself requires downloading Scryfall's full bulk card image set (~1.5GB) —
scripts for that are in `dino/`, nothing pre-built is shipped in this repo.

## Known limitations

- Card-name OCR accuracy depends entirely on the local vision model you load
  in LM Studio and your capture lighting/focus — expect occasional misreads,
  especially on older/foreign-influenced card frames.
- No automated card-boundary detection or perspective correction yet —
  framing is manual (the crop tool) rather than an auto-detected, deskewed
  crop. This is probably the single biggest lever for improving accuracy
  further; see `dino/ACCURACY_REPORT.md` for more on why.
- Foil detection is a vision-model guess from a static photo, not a real
  reflectivity/holographic measurement — treat it as a hint, not a fact.
- Tested on Windows with an external USB webcam. Camera selection/permission
  behavior may differ on other platforms.

## Support

If this is useful to you, tips are welcome: **[ko-fi.com/mindpalacegarden](https://ko-fi.com/mindpalacegarden)**

## License

MIT — see [LICENSE](LICENSE).
