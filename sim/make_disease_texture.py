"""Procedural test texture for the sim 'diseased plant' fixture: tomato-like leaflets with early-blight style lesions
(brown concentric target spots with yellow chlorotic halos). It is a labelled test fixture, like the obstacle box,
so the inspection → revisit loop has something to find in the digital twin. Not real disease data.

  python3 sim/make_disease_texture.py out.png [--seed 3] [--healthy]"""
from __future__ import annotations
import argparse, math, random
from PIL import Image, ImageDraw, ImageFilter


def leaflet(d: ImageDraw.ImageDraw, cx, cy, length, width, angle, fill):
    pts = []
    for k in range(41):
        t = k / 40 * 2 * math.pi
        x, y = length / 2 * math.cos(t), width / 2 * math.sin(t) * (1 + 0.25 * math.cos(t))
        x += 0.06 * length * math.sin(5 * t)                     # serrated edge
        ca, sa = math.cos(angle), math.sin(angle)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    d.polygon(pts, fill=fill)
    ca, sa = math.cos(angle), math.sin(angle)
    d.line([(cx - length / 2 * ca, cy - length / 2 * sa), (cx + length / 2 * ca, cy + length / 2 * sa)], fill=(70, 120, 45), width=3)


def make(w=700, h=1000, seed=3, healthy=False) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h), (34, 70, 30))
    d = ImageDraw.Draw(img)
    leaves = []
    for _ in range(38):
        cx, cy = rnd.uniform(0, w), rnd.uniform(0, h)
        L, W = rnd.uniform(140, 240), rnd.uniform(70, 120)
        g = rnd.randint(105, 150)
        leaflet(d, cx, cy, L, W, rnd.uniform(0, math.pi), (rnd.randint(45, 75), g, rnd.randint(35, 60)))
        leaves.append((cx, cy, L, W))
    if not healthy:
        def blob(x, y, r, fill):                                  # irregular lesion outline
            pts = [(x + r * (1 + 0.25 * rnd.uniform(-1, 1)) * math.cos(t), y + r * (1 + 0.25 * rnd.uniform(-1, 1)) * math.sin(t))
                   for t in (k / 14 * 2 * math.pi for k in range(14))]
            d.polygon(pts, fill=fill)
        for cx, cy, L, W in rnd.sample(leaves, 18):
            for _ in range(rnd.randint(1, 4)):
                x, y, r = cx + rnd.uniform(-L / 3, L / 3), cy + rnd.uniform(-W / 4, W / 4), rnd.uniform(6, 16)
                blob(x, y, r * 1.35, (150, 150, 62))                                                   # thin chlorotic halo
                for k, c in enumerate([(88, 60, 32), (58, 40, 24), (98, 70, 38), (48, 33, 20)]):       # target rings
                    blob(x, y, r * (1 - k * 0.22), c)
        for cx, cy, L, W in rnd.sample(leaves, 6):                                                      # yellowing lower leaves
            leaflet(d, cx, cy, L * 0.8, W * 0.8, rnd.uniform(0, math.pi), (176, 168, 64))
    return img.filter(ImageFilter.GaussianBlur(0.8))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--healthy", action="store_true")
    a = ap.parse_args()
    make(seed=a.seed, healthy=a.healthy).save(a.out)
    print(a.out)
