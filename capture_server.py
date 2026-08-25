import csv
import http.server
import json
import os
import time

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(DIRECTORY, "capture.png")
EVENTS_DIR = os.path.join(DIRECTORY, "events")
EVENTS_LOG = os.path.join(EVENTS_DIR, "events.jsonl")
IDENTIFIED_STORE = os.path.join(EVENTS_DIR, "identified_store.json")
LOCAL_SCAN_CSV = os.path.join(DIRECTORY, "scanned_cards_local.csv")

os.makedirs(EVENTS_DIR, exist_ok=True)


def _identified_key(record):
    # Prefer set+collector_number (identifies the exact printing); fall back to name.
    set_code = (record.get("set") or "").strip().lower()
    collector = (record.get("collector_number") or "").strip().lower()
    if set_code and collector:
        return f"{set_code}/{collector}"
    return (record.get("name") or "unknown").strip().lower()


def _load_identified_store():
    if os.path.exists(IDENTIFIED_STORE):
        with open(IDENTIFIED_STORE) as f:
            return json.load(f)
    return {}


def _save_identified_store(store):
    with open(IDENTIFIED_STORE, "w") as f:
        json.dump(store, f)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == "/upload":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            with open(CAPTURE_PATH, "wb") as f:
                f.write(body)
            self._ok(b"ok")

        elif self.path == "/event":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            ts = time.time()
            idx = self._next_index()
            filename = f"card_{idx:04d}.png"
            filepath = os.path.join(EVENTS_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(body)
            record = {"index": idx, "file": filename, "ts": ts}
            with open(EVENTS_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._ok(json.dumps(record).encode())

        elif self.path == "/identified":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                record = json.loads(body.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_response(400)
                self.end_headers()
                return
            now = time.time()
            store = _load_identified_store()
            key = _identified_key(record)
            if key in store:
                # Use the freshly posted details (in case they're more accurate),
                # but carry the running quantity/first-seen forward.
                quantity = store[key].get("quantity", 1) + 1
                first_seen_at = store[key].get("first_seen_at", now)
                record["quantity"] = quantity
                record["first_seen_at"] = first_seen_at
                record["last_seen_at"] = now
                store[key] = record
            else:
                record["quantity"] = 1
                record["first_seen_at"] = now
                record["last_seen_at"] = now
                store[key] = record
            _save_identified_store(store)
            self._ok(json.dumps(store[key]).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/events.json":
            self._serve_jsonl(EVENTS_LOG)
        elif self.path == "/identified.json":
            store = _load_identified_store()
            records = sorted(store.values(), key=lambda r: r.get("last_seen_at", 0))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(records).encode())
        elif self.path == "/local_scan.json":
            records = []
            if os.path.exists(LOCAL_SCAN_CSV):
                with open(LOCAL_SCAN_CSV, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        try:
                            row["quantity"] = int(row["quantity"])
                        except (ValueError, KeyError):
                            row["quantity"] = 1
                        try:
                            row["last_seen_at"] = float(row["last_seen_at"])
                        except (ValueError, KeyError):
                            row["last_seen_at"] = 0
                        records.append(row)
            records.sort(key=lambda r: r["last_seen_at"])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(records).encode())
        else:
            super().do_GET()

    def _serve_jsonl(self, path):
        records = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(records).encode())

    def _next_index(self):
        existing = [f for f in os.listdir(EVENTS_DIR) if f.startswith("card_") and f.endswith(".png")]
        nums = []
        for f in existing:
            try:
                nums.append(int(f[len("card_"):-len(".png")]))
            except ValueError:
                pass
        return (max(nums) + 1) if nums else 1

    def _ok(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 8743), Handler)
    print("Serving on http://127.0.0.1:8743")
    print("capture.png:", CAPTURE_PATH)
    print("events dir:", EVENTS_DIR)
    server.serve_forever()
