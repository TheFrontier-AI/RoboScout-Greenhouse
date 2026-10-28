"""Run the eyes over a folder of real greenhouse photos and write one JSON line per image (finding or error).
Usage: python -m cloud.eyes.batch_real_photos [folder] --camera mast_left_mid --out cloud/eyes/out/real_photos.jsonl"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

from .inspect import inspect_image, make_client

DEFAULT_DIR = Path("/home/passin/Documents/Blender/TheFarm/refs/02_tomato_greenhouses")
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "real_photos.jsonl"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def run(folder: Path, out: Path, camera: str, limit: int | None = None) -> dict:
    client = make_client()
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with out.open("w") as f:
        for i, p in enumerate(iter_images(folder)[:limit]):
            meta = {"ts": time.time(), "house": None, "aisle": None, "metre": None, "heading_deg": None,
                    "camera": camera, "frame_id": i, "pose_confidence": 0.0, "source": str(p)}
            t0 = time.time()
            try:
                rec = {"file": p.name, "finding": inspect_image(p.read_bytes(), meta, client=client).model_dump()}
                key = rec["finding"]["status"]
            except Exception as e:                       # keep going; one bad photo must not stop the batch
                rec, key = {"file": p.name, "meta": meta, "error": f"{type(e).__name__}: {e}"}, "error"
            rec["latency_s"] = round(time.time() - t0, 2)
            counts[key] = counts.get(key, 0) + 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{p.name}: {key} ({rec['latency_s']} s)", file=sys.stderr)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", nargs="?", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--camera", default="mast_left_mid")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.folder, a.out, a.camera, a.limit)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
