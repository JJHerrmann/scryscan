"""
Fully local, autonomous MTG card scanner.

Watches events/ (populated by webcam_viewer.html's auto-detect) for new
card_NNNN.png captures, asks a local LM Studio vision model for the card
name, confirms/enriches via the Scryfall API, and maintains a running CSV
of everything scanned with quantities. No cloud LLM involved.
"""
import base64
import csv
import json
import os
import time
import urllib.parse
import urllib.request

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EVENTS_DIR = os.path.join(DIRECTORY, "events")
EVENTS_LOG = os.path.join(EVENTS_DIR, "events.jsonl")
STATE_FILE = os.path.join(EVENTS_DIR, "local_llm_last_index.txt")
CSV_PATH = os.path.join(DIRECTORY, "scanned_cards_local.csv")
TOKEN_FILE = os.path.join(DIRECTORY, "lmstudio_token.txt")

LM_STUDIO_URL = "http://localhost:42069/api/v1/chat"
LM_STUDIO_MODEL = "google/gemma-4-e4b"
POLL_INTERVAL_SECONDS = 3

# When True, a card that matches the immediately-previous frame's name is treated as
# "still the same physical card sitting in view" and skipped (protects against camera
# jitter/lighting noise re-triggering the same card). Turn this OFF once you're flipping
# through already-sorted stacks where several consecutive frames legitimately ARE
# separate copies of the same card (e.g. a pile of 4x the same card) - otherwise those
# get undercounted as one.
SKIP_CONSECUTIVE_DUPLICATES = False

SCRYFALL_UA = "RookworksLocalCardScanner/1.0 (jaherrmann@gmail.com)"

CSV_FIELDS = [
    "name", "mana_cost", "type_line", "oracle_text", "set", "set_name",
    "collector_number", "price_usd", "price_usd_foil", "foil_guess",
    "scryfall_uri", "match_kind", "oracle_id", "quantity", "first_seen_at", "last_seen_at",
]


def load_token():
    with open(TOKEN_FILE) as f:
        return f.read().strip()


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            content = f.read().strip()
            return int(content) if content else 0
    return 0


def save_state(idx):
    with open(STATE_FILE, "w") as f:
        f.write(str(idx))


def read_new_events(last_index):
    events = []
    if not os.path.exists(EVENTS_LOG):
        return events
    with open(EVENTS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["index"] > last_index:
                events.append(record)
    events.sort(key=lambda r: r["index"])
    return events


def identify_card(image_path, token):
    """Returns dict with name, collector_number, set_code (any of the latter two may be None)."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": LM_STUDIO_MODEL,
        "system_prompt": (
            "You identify Magic: The Gathering cards from photos of the card face. "
            "Look carefully at the small print along the bottom edge of the card, which "
            "usually shows a collector number and set code, e.g. '82/301' or 'U 0082 TDM' "
            "or '112/274 R BFZ'. "
            "Also judge whether the card is a foil (holographic, reflective, rainbow-sheen "
            "surface) or nonfoil (flat matte cardstock). Foil cards often show a glare, "
            "sparkle, or rainbow diffraction pattern somewhere on the card face even under "
            "normal photo lighting; if the surface just looks like plain flat cardstock with "
            "no shine, it is nonfoil. If you genuinely cannot tell from the glare/lighting in "
            "the photo, say UNSURE rather than guessing. "
            "Reply with EXACTLY one line in this format and nothing else:\n"
            "NAME | COLLECTOR_NUMBER | SET_CODE | FOIL\n"
            "Use the literal text NONE for COLLECTOR_NUMBER or SET_CODE if you cannot read "
            "them clearly. Do not guess a set code you cannot actually see. "
            "SET_CODE should be the 2-5 letter/digit code only (e.g. TDM, BFZ, KTK), not the "
            "full set name. FOIL must be exactly one of: YES, NO, UNSURE. "
            "Do not add explanations, quotes, or extra lines."
        ),
        "input": [
            {"type": "text", "content": "Identify this card: name, collector number, set code, and whether it is foil."},
            {"type": "image", "data_url": "data:image/png;base64," + b64},
        ],
    }
    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    raw = None
    for item in data.get("output", []):
        if item.get("type") == "message":
            raw = item["content"].strip()
            break
    if not raw:
        return None

    parts = [p.strip() for p in raw.split("|")]
    name = parts[0] if parts else None
    if not name or name.upper() in ("NONE", "UNKNOWN", "N/A"):
        return None

    collector_number = None
    set_code = None
    if len(parts) > 1 and parts[1] and parts[1].upper() != "NONE":
        # Model may echo "82/301" or "0082" or "82" - keep only the leading digits/letters,
        # strip leading zeros (Scryfall's collector_number field is usually unpadded).
        raw_cn = parts[1].split("/")[0].strip()
        stripped = raw_cn.lstrip("0")
        collector_number = stripped if stripped else raw_cn
    if len(parts) > 2 and parts[2] and parts[2].upper() != "NONE":
        set_code = "".join(ch for ch in parts[2] if ch.isalnum()).lower()

    foil_guess = "unsure"
    if len(parts) > 3:
        foil_raw = parts[3].strip().upper()
        if foil_raw == "YES":
            foil_guess = "yes"
        elif foil_raw == "NO":
            foil_guess = "no"

    return {
        "name": name,
        "collector_number": collector_number,
        "set_code": set_code,
        "foil_guess": foil_guess,
    }


def scryfall_exact(set_code, collector_number):
    url = "https://api.scryfall.com/cards/%s/%s" % (
        urllib.parse.quote(set_code), urllib.parse.quote(collector_number)
    )
    req = urllib.request.Request(url, headers={"User-Agent": SCRYFALL_UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def scryfall_fuzzy(name):
    url = "https://api.scryfall.com/cards/named?fuzzy=" + urllib.parse.quote(name)
    req = urllib.request.Request(url, headers={"User-Agent": SCRYFALL_UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print("  Scryfall fuzzy lookup failed for %r: %s" % (name, e))
        return None


def scryfall_lookup(identified):
    """Try the exact printing first (if the model read a collector number/set code),
    falling back to fuzzy name search. Also sanity-checks the exact hit's name against
    what the model read, in case it misread the collector number onto the wrong card."""
    name = identified["name"]
    set_code = identified.get("set_code")
    collector_number = identified.get("collector_number")

    if set_code and collector_number:
        exact = scryfall_exact(set_code, collector_number)
        if exact and exact.get("object") == "card":
            # Guard against a misread collector number landing on an unrelated card.
            if _names_roughly_match(exact["name"], name):
                return exact, "exact"
            else:
                print("  exact match %s/%s = %r doesn't match read name %r, falling back to fuzzy" % (
                    set_code, collector_number, exact["name"], name
                ))

    fuzzy = scryfall_fuzzy(name)
    return fuzzy, "fuzzy"


def _names_roughly_match(a, b):
    norm = lambda s: "".join(ch.lower() for ch in s if ch.isalnum())
    a, b = norm(a), norm(b)
    return a == b or a in b or b in a


def load_csv_store():
    store = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["set"], row["collector_number"])
                row["quantity"] = int(row["quantity"])
                store[key] = row
    return store


def save_csv_store(store):
    rows = sorted(store.values(), key=lambda r: float(r["last_seen_at"]))
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def warn_if_likely_duplicate_printing(store, card, match_kind):
    """A fuzzy match landing on a different printing of a card we already have
    stored (same oracle_id, different set/collector_number) usually means the
    model failed to read the collector info and Scryfall's fuzzy default picked
    the wrong printing of the SAME physical card - not a genuine second copy in
    a different printing. We don't auto-merge (that could be wrong the other
    way, if you really do own two different printings), but we flag it loudly
    so it can be reviewed/merged."""
    if match_kind != "fuzzy":
        return
    oracle_id = card.get("oracle_id")
    if not oracle_id:
        return
    key = (card["set"], card["collector_number"])
    for existing_key, row in store.items():
        if existing_key != key and row.get("oracle_id") == oracle_id:
            print("  ** LIKELY DUPLICATE PRINTING: %r matched %s/%s but is already stored as %s/%s "
                  "(qty %s). This was a fuzzy match (collector info unreadable) - probably the same "
                  "physical card, wrongly split. Consider merging manually. **" % (
                      card["name"], card["set"], card["collector_number"],
                      existing_key[0], existing_key[1], row.get("quantity", "?")
                  ))
            return


def upsert_card(store, card, match_kind, foil_guess):
    warn_if_likely_duplicate_printing(store, card, match_kind)
    now = time.time()
    key = (card["set"], card["collector_number"])
    if key in store:
        store[key]["quantity"] += 1
        store[key]["last_seen_at"] = now
        store[key]["match_kind"] = match_kind
        # Reflects only the most-recently-scanned physical copy - if you have
        # multiple copies of this printing, they may not all match this foil guess.
        store[key]["foil_guess"] = foil_guess
    else:
        store[key] = {
            "name": card["name"],
            "mana_cost": card.get("mana_cost") or "",
            "type_line": card.get("type_line") or "",
            "oracle_text": (card.get("oracle_text") or "").replace("\n", " "),
            "set": card["set"],
            "set_name": card.get("set_name") or "",
            "collector_number": card["collector_number"],
            "price_usd": card.get("prices", {}).get("usd") or "",
            "price_usd_foil": card.get("prices", {}).get("usd_foil") or "",
            "foil_guess": foil_guess,
            "scryfall_uri": (card.get("scryfall_uri") or "").split("?")[0],
            "match_kind": match_kind,
            "oracle_id": card.get("oracle_id") or "",
            "quantity": 1,
            "first_seen_at": now,
            "last_seen_at": now,
        }
    return store[key]["quantity"]


def main():
    token = load_token()
    last_index = load_state()
    last_name = None
    last_event_index = None
    store = load_csv_store()

    print("Local card scanner running. Watching:", EVENTS_DIR)
    print("Writing to:", CSV_PATH)
    print("Starting from event index:", last_index)

    while True:
        events = read_new_events(last_index)
        for event in events:
            idx = event["index"]
            image_path = os.path.join(EVENTS_DIR, event["file"])
            if not os.path.exists(image_path):
                last_index = idx
                save_state(last_index)
                continue

            try:
                identified = identify_card(image_path, token)
            except Exception as e:
                print("Event %d: LLM identification failed: %s" % (idx, e))
                last_index = idx
                save_state(last_index)
                continue

            if not identified:
                print("Event %d: no name returned" % idx)
                last_index = idx
                save_state(last_index)
                continue

            name = identified["name"]
            is_consecutive_same = (
                SKIP_CONSECUTIVE_DUPLICATES
                and last_name is not None
                and name.lower() == last_name.lower()
                and last_event_index is not None
                and idx == last_event_index + 1
            )

            if is_consecutive_same:
                print("Event %d: same as previous frame (%s), skipping" % (idx, name))
            else:
                card, match_kind = scryfall_lookup(identified)
                if card and card.get("object") == "card":
                    qty = upsert_card(store, card, match_kind, identified["foil_guess"])
                    save_csv_store(store)
                    print("Event %d: %s (%s #%s, %s match, foil=%s) -> qty %d" % (
                        idx, card["name"], card["set"].upper(), card["collector_number"],
                        match_kind, identified["foil_guess"], qty
                    ))
                else:
                    print("Event %d: could not confirm '%s' on Scryfall" % (idx, name))

            last_name = name
            last_event_index = idx
            last_index = idx
            save_state(last_index)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
