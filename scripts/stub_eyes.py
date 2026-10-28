"""Rehearsal stand-in for Cosmos Reason: a tiny OpenAI-compatible /v1/chat/completions server (stdlib) that reports
'disease' for side-camera frames whose prompt metadata says metre within --spot ±0.6, 'healthy' otherwise.
Lets the whole loop (robot → hub → eyes → HERMES revisit → dashboard) run without a GPU. Not a model.

  python3 scripts/stub_eyes.py --port 8001 --spot 15   # then EYES_BASE_URL=http://127.0.0.1:8001/v1"""
import argparse, json, re, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def answer(prompt: str, spot: float) -> dict:
    m = re.search(r'metre"?\s*[:=]\s*(-?[\d.]+)', prompt)
    metre = float(m.group(1)) if m else None
    if "front" in prompt and "mast_" not in prompt:
        return {"status": "not_plant", "issues": [], "summary": "aisle clear",
                "nav": {"aisle_clear": True, "obstacle": None, "appears_centered": True}}
    if metre is not None and abs(metre - spot) <= 0.6:
        return {"status": "disease", "summary": "dark concentric lesions on lower leaves",
                "issues": [{"label": "early blight", "confidence": 0.82, "bbox_norm": [0.42, 0.30, 0.71, 0.62],
                            "evidence": "brown target-like spots with yellow halo (stub)"}]}
    return {"status": "healthy", "issues": [], "summary": "foliage looks normal (stub)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8001); ap.add_argument("--spot", type=float, default=15.0)
    a = ap.parse_args()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self._send({"object": "list", "data": [{"id": "stub-eyes", "object": "model"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            text = " ".join(p.get("text", "") if isinstance(p, dict) else str(p)
                            for m in req.get("messages", []) for p in (m["content"] if isinstance(m["content"], list) else [m["content"]]))
            content = json.dumps(answer(text, a.spot))
            self._send({"id": "stub", "object": "chat.completion", "created": int(time.time()), "model": req.get("model", "stub"),
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    print(f"stub eyes on http://127.0.0.1:{a.port}/v1 (disease at metre {a.spot})")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
