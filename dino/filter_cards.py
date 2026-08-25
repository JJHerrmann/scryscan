"""Filter Scryfall default_cards bulk data down to the corpus we'll embed."""
import gzip
import json
import os

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(DIRECTORY, "default_cards.jsonl.gz")
OUT_PATH = os.path.join(DIRECTORY, "corpus.jsonl")

EXCLUDE_LAYOUTS = {"art_series", "token", "double_faced_token", "emblem", "planar", "scheme", "vanguard", "reversible_card"}

kept = 0
skipped_lang = 0
skipped_digital = 0
skipped_layout = 0
skipped_no_image = 0
total = 0

with gzip.open(IN_PATH, "rt", encoding="utf-8") as f, open(OUT_PATH, "w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        card = json.loads(line)
        if card.get("lang") != "en":
            skipped_lang += 1
            continue
        if "paper" not in (card.get("games") or []):
            skipped_digital += 1
            continue
        if card.get("layout") in EXCLUDE_LAYOUTS:
            skipped_layout += 1
            continue

        # Handle both single-faced (image_uris) and double-faced (card_faces[i].image_uris) cards.
        faces = card.get("card_faces")
        image_url = None
        if card.get("image_uris"):
            image_url = card["image_uris"].get("small") or card["image_uris"].get("normal")
        elif faces and faces[0].get("image_uris"):
            image_url = faces[0]["image_uris"].get("small") or faces[0]["image_uris"].get("normal")

        if not image_url:
            skipped_no_image += 1
            continue

        record = {
            "id": card["id"],
            "oracle_id": card.get("oracle_id"),
            "name": card["name"],
            "set": card["set"],
            "set_name": card.get("set_name"),
            "collector_number": card["collector_number"],
            "type_line": card.get("type_line"),
            "mana_cost": card.get("mana_cost"),
            "oracle_text": card.get("oracle_text") or (faces[0].get("oracle_text") if faces else None),
            "prices": card.get("prices"),
            "scryfall_uri": card.get("scryfall_uri"),
            "finishes": card.get("finishes"),
            "image_url": image_url,
        }
        out.write(json.dumps(record) + "\n")
        kept += 1

print("total in bulk file:", total)
print("kept:", kept)
print("skipped (non-english):", skipped_lang)
print("skipped (digital-only):", skipped_digital)
print("skipped (excluded layout):", skipped_layout)
print("skipped (no image):", skipped_no_image)
