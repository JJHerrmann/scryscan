"""Targeted test: for a few failed cases, download the CORRECT card's reference
image at 'normal' resolution instead of 'small', re-embed both the correct
answer and the (wrong) top-1 answer at normal res, and see if similarity to
the query flips in favor of the correct card."""
import json
import os

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SCRATCHPAD = os.path.dirname(DIRECTORY)
EVENTS_DIR = os.path.join(SCRATCHPAD, "events")
UA = "RookworksDinoBuild/1.0 (jaherrmann@gmail.com)"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()


def embed(img):
    inputs = processor(images=[img.convert("RGB")], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    vec = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return vec / np.linalg.norm(vec)


def fetch_normal_image(set_code, collector_number):
    card = requests.get(
        "https://api.scryfall.com/cards/%s/%s" % (set_code, collector_number),
        headers={"User-Agent": UA}, timeout=15
    ).json()
    url = card.get("image_uris", {}).get("normal")
    if not url and card.get("card_faces"):
        url = card["card_faces"][0].get("image_uris", {}).get("normal")
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=15)
    from io import BytesIO
    return Image.open(BytesIO(resp.content)), card["name"]


# (event_index, expected_set, expected_cn, wrong_top1_set, wrong_top1_cn, expected_name, wrong_name)
CASES = [
    (1, "fdn", "61", "pone", "218s", "High-Society Hunter", "Tyvar, Jubilant Brawler"),
    (25, "tdm", "72", "one", "106", "Alesha's Legacy", "Ravenous Necrotitan"),
    (129, "tdm", "89", "one", "132", "Sandskitter Outrider", "Furnace Punisher"),
    (241, "m12", "115", "cma", "56", "Vampire Outcasts", "Dread Summons"),
]

for idx, e_set, e_cn, w_set, w_cn, e_name, w_name in CASES:
    query_path = os.path.join(EVENTS_DIR, "card_%04d.png" % idx)
    query_img = Image.open(query_path)
    query_vec = embed(query_img)

    correct_img, correct_name = fetch_normal_image(e_set, e_cn)
    correct_vec = embed(correct_img)
    correct_sim = float(np.dot(query_vec, correct_vec))

    wrong_img, wrong_name = fetch_normal_image(w_set, w_cn)
    wrong_vec = embed(wrong_img)
    wrong_sim = float(np.dot(query_vec, wrong_vec))

    winner = "CORRECT now wins" if correct_sim > wrong_sim else "still WRONG"
    print("event %d: correct(%s)=%.4f  wrong(%s)=%.4f  -> %s" % (
        idx, correct_name, correct_sim, wrong_name, wrong_sim, winner
    ))
