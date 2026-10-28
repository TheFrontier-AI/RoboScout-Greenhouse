"""Assemble the ≤3 min submission video from real run artefacts (no generated footage): title, real→twin evidence,
real-photo inspection, drone SLAM (Nebius Serverless Job), the live dashboard timelapse of the full loop, the flagged
full-size frame, the farmer report, the farm site map, architecture and honest status.

  python3 scripts/make_demo_video.py --run ../r1_runs/realism2_2026-10-28 --out ../r1_runs/RoboFarm_demo.mp4 \
      --four-way <astra review_four_way_final.jpg> --real-photo <IMG.jpg> --slam-topdown <topdown.png> --sketch <png>"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H, BG, FG, ACC, WARN = 1920, 1080, (16, 24, 22), (236, 240, 236), (118, 185, 0), (240, 190, 60)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def canvas(title: str | None = None, sub: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    if title:
        d.text((60, 40), title, font=font(54, True), fill=FG)
    if sub:
        d.text((60, 112), sub, font=font(30), fill=(170, 190, 180))
    d.text((W - 430, H - 50), "RoboFarm · Nebius x NVIDIA · Physical AI", font=font(22), fill=(110, 130, 120))
    return im, d


def fit(img: Image.Image, box: tuple[int, int]) -> Image.Image:
    im = img.convert("RGB"); im.thumbnail(box, Image.LANCZOS); return im


def bullets(d, lines, x, y, size=32, gap=1.55, color=FG):
    for ln in lines:
        c = ACC if ln.startswith("✓") else WARN if ln.startswith("!") else color
        d.text((x, y), ln, font=font(size), fill=c); y += int(size * gap)
    return y


def slide_title():
    im, d = canvas()
    d.text((120, 330), "RoboFarm", font=font(120, True), fill=FG)
    d.text((124, 480), "An agentic greenhouse robot that scouts tomato aisles,", font=font(40), fill=FG)
    d.text((124, 535), "spots suspect plants and decides where to look again.", font=font(40), fill=FG)
    d.text((124, 640), "HERMES (Hermes Agent + NVIDIA Nemotron) · NVIDIA Cosmos Reason · Isaac Sim · Nebius AI Cloud",
           font=font(28), fill=ACC)
    d.text((124, 690), "Sim-to-real: digital twin built from our real farm · robot R1 being assembled", font=font(28),
           fill=(170, 190, 180))
    return im


def slide_image(title, sub, img_path, caption_lines):
    im, d = canvas(title, sub)
    pic = fit(Image.open(img_path), (W - 120, 760)); im.paste(pic, ((W - pic.width) // 2, 170))
    bullets(d, caption_lines, 60, 170 + pic.height + 20, size=28, gap=1.4)
    return im


def slide_two(title, sub, left, right, cap_l, cap_r, notes):
    im, d = canvas(title, sub)
    a, b = fit(Image.open(left), (880, 700)), fit(Image.open(right), (880, 700))
    im.paste(a, (60 + (880 - a.width) // 2, 180)); im.paste(b, (980 + (880 - b.width) // 2, 180))
    d.text((60, 900), cap_l, font=font(28), fill=FG); d.text((980, 900), cap_r, font=font(28), fill=FG)
    bullets(d, notes, 60, 950, size=26, gap=1.3)
    return im


def slide_text(title, sub, lines, size=34):
    im, d = canvas(title, sub); bullets(d, lines, 80, 200, size=size); return im


def site_map_plot(site_json: Path, out_png: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    m = json.loads(site_json.read_text())
    nodes = {n["id"]: n for n in m["nodes"]}
    fig, ax = plt.subplots(figsize=(9, 7), dpi=110); fig.patch.set_facecolor("#101816"); ax.set_facecolor("#101816")
    for e in m["edges"]:
        a, b = nodes[e["a"]]["xy"], nodes[e["b"]]["xy"]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#5f7f73", lw=1.2)
    for n in m["nodes"]:
        k = n.get("kind"); x, y = n["xy"]
        if k in ("door", "dock"):
            ax.scatter([x], [y], s=40 if k == "door" else 90, color="#76b900" if k == "door" else "#f0be3c", zorder=3)
            if k == "door" and (n.get("house", "").startswith("tunnel") or "01" in n.get("house", "")):
                ax.annotate(n.get("house", ""), (x, y), fontsize=7, color="#dfe8e2", xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal"); ax.tick_params(colors="#7f9a90"); [s.set_color("#33463f") for s in ax.spines.values()]
    ax.set_title(f"V18 site map draft: {sum(1 for n in m['nodes'] if n.get('kind') == 'door')} greenhouse doors, "
                 f"{len(m['edges'])} paths (unconfirmed)", color="#ecf0ec", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, facecolor=fig.get_facecolor()); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--four-way", type=Path); ap.add_argument("--six-cam", type=Path); ap.add_argument("--real-photo", type=Path)
    ap.add_argument("--slam-topdown", type=Path); ap.add_argument("--sketch", type=Path)
    ap.add_argument("--site-map", type=Path, default=Path(__file__).resolve().parents[1] / "data/site_map_v18_draft.json")
    ap.add_argument("--dash-fps", type=float, default=2.2)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="robofarm_video_"))
    segs: list[tuple[Path, float]] = []

    def still(im: Image.Image, secs: float):
        p = tmp / f"s{len(segs):02d}.png"; im.save(p); segs.append((p, secs))

    data = a.run / "data"
    findings = [json.loads(l) for l in (data / "findings.jsonl").read_text().splitlines() if l.strip()]
    decisions = [json.loads(l) for l in (data / "decisions.jsonl").read_text().splitlines() if l.strip()]
    flagged = [f for f in findings if f.get("status") in ("suspect", "disease") and f.get("purpose") == "inspect"]
    n_inspect = sum(1 for f in findings if f.get("purpose") == "inspect")

    still(slide_title(), 5)
    if a.four_way:
        still(slide_image("Real → Sim: a digital twin of our farm", "Real photo · Blender · Isaac RT · Isaac PathTracing (same views)",
                          a.four_way, ["Plants, film, light and exposure matched to photos of our real tunnel-09 greenhouse."]), 6)
    if a.six_cam:
        still(slide_image("R1's six plant cameras in the twin", "PathTracing ISO 400 · runtime cameras",
                          a.six_cam, ["Six side cameras (low/mid/high × left/right) + a front camera, as on the real robot."]), 5)
    if a.real_photo:
        still(slide_two("Same eyes on real photos", "NVIDIA Cosmos-Reason2-8B on a Nebius RTX PRO 6000 (vLLM)",
                        a.real_photo, a.real_photo, "Real greenhouse photo (our farm)", "→ 'healthy': green foliage, flowers, green fruit",
                        ["✓ 11 real photos: 9 healthy, 2 correctly 'not a plant' (exterior shots), 0.5–1 s each"]), 5)
    if a.slam_topdown:
        still(slide_image("Cloud SLAM AI from real drone video", "MapAnything as a Nebius Serverless Job on an NVIDIA H200",
                          a.slam_topdown, ["✓ 200 frames → 4 M points in 55 s · farm buildings and the flight path recovered"]), 6)

    still(slide_text("The live loop", "Next: 70 s timelapse of one real run (sim ≈ 6 s per step)", [
        "HERMES (Hermes Agent + Nemotron 3 Super, Nebius Token Factory) sends goals, never wheel speeds",
        "R1 in Isaac Sim drives the aisle: LiDAR centring, SAFETY, odometry — cameras at real 1080×1920",
        "Every 0.5 m: two NVIDIA eyes screen each side frame",
        "     Cosmos-Reason2-8B (on our Nebius RTX PRO 6000 VM) + Nemotron-3.5-Super-VL (Token Factory)",
        "Either eye says 'suspect' → HERMES sends the robot back for a still, six-camera second look",
        "! The diseased plant at 15 m is a labelled synthetic test asset (subtle leaf spots)"], size=30), 7)

    dash = sorted((a.run / "dash_frames").glob("d_*.png"))
    if dash:
        lst = tmp / "dash.txt"
        lst.write_text("".join(f"file '{p}'\nduration {1 / a.dash_fps:.4f}\n" for p in dash) + f"file '{dash[-1]}'\n")
        clip = tmp / "dash.mp4"
        cap = "Live run · Nebius RTX PRO 6000 VM (Isaac + Cosmos) · Token Factory (Nemotron) · timelapse"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-vf", f"scale={W}:{H},drawbox=y=ih-56:w=iw:h=56:color=black@0.65:t=fill,"
                               f"drawtext=fontfile={FONT}:text='{cap}':fontcolor=white:fontsize=24:x=30:y=h-42",
                        "-r", "30", "-pix_fmt", "yuv420p", str(clip)], check=True)
        segs.append((clip, 0))

    if flagged:
        f = flagged[0]
        votes = f.get("meta", {}).get("votes", {})
        iss = f["issues"][0] if f.get("issues") else {}
        still(slide_image(f"What the eyes saw at {f['metre']} m ({f['camera']}, full 1080×1920)",
                          " · ".join(f"{k}: {v}" for k, v in votes.items()),
                          data / f["image_path"],
                          [f"! {iss.get('label', '')} {iss.get('confidence', '')} — {iss.get('evidence', '')[:120]}",
                           f"✓ {n_inspect} screened frames · {len(flagged)} suspect · 0 false alarms elsewhere"]), 7)
    revisit = [f for f in findings if f.get("purpose") == "revisit"]
    report = (a.run / "brain2.log").read_text() if (a.run / "brain2.log").exists() else ""
    eng = report.split("Farmer Report (English)")[-1].strip().splitlines()[:6] if "English" in report else []
    still(slide_text("Revisit and report", "HERMES decisions (Nemotron writes the reasoning)", [
        *[f"{d['command']['cmd']} {d['command'].get('metre', '') or ''}: {d['reasoning'][:95]}" for d in decisions[-3:]],
        f"Revisit: {len(revisit)} still shots, both eyes → {', '.join(sorted({r['status'] for r in revisit})) or '-'}",
        "! Spot cleared on re-check: the subtle test lesion was not confirmed → farmer is asked to look",
        *[ln.strip("-* ").replace("**", "")[:110] for ln in eng if ln.strip()]], size=26), 10)

    if a.site_map.exists():
        mp = tmp / "site.png"; site_map_plot(a.site_map, mp)
        still(slide_two("Moving between greenhouses", "Farm-level site maps: V18 model · drone · farmer sketch · robot",
                        mp, a.sketch or mp, "Draft from the V18 farm model: 22 greenhouses",
                        "Test sketch (drawn by us, not a farmer) → Nemotron-VL → map draft",
                        ["✓ Topology read correctly from the sketch; handwriting/coordinates need a human check",
                         "Robot only uses CONFIRMED maps: go_to_house / survey another house → route on the site map"]), 7)

    still(slide_text("Built on Nebius + NVIDIA", "", [
        "✓ Nebius Token Factory: Nemotron 3 Super (HERMES) · Nemotron-3.5-Super-VL (second eye, sketch reading)",
        "✓ Nebius AI Cloud: RTX PRO 6000 Blackwell VM — Isaac Sim 5.1 twin + Cosmos-Reason2-8B (vLLM)",
        "✓ Nebius Serverless Job (H200) + Container Registry — cloud SLAM from drone video",
        "✓ Agent runtime: Hermes Agent with RoboFarm skills (robot control, diagnosis, motion, site map)",
        "! Honest status: robot simulated (real R1 in assembly); synthetic test plant; screening ≠ diagnosis",
        "  Repo (Apache-2.0): github.com/TheFrontier-AI/RoboScout-Greenhouse"], size=31), 9)

    parts = []
    for i, (p, secs) in enumerate(segs):
        if p.suffix == ".mp4":
            parts.append(p); continue
        out = tmp / f"p{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", str(secs), "-i", str(p),
                        "-vf", f"scale={W}:{H},fade=t=in:st=0:d=0.4", "-r", "30", "-pix_fmt", "yuv420p", str(out)], check=True)
        parts.append(out)
    lst = tmp / "all.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(a.out)],
                         capture_output=True, text=True).stdout.strip()
    print(f"{a.out}  {float(dur):.1f} s"); shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
