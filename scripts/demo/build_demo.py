#!/usr/bin/env python3
"""RoboFarm demo film assembler.

One continuous film: the chase camera is the spine, overlays (PIL) are drawn per frame,
cut-ins and insets come from the other raw renders, narration is per-segment TTS
(see make_tts.py). Each shot is rendered by its own worker into an mp4, the shots are
concatenated, and the narration is muxed at the end.

Usage:
  build_demo.py --tts DIR [--draft] [--out docs/demo/robofarm_demo.mp4]
"""
import argparse, json, math, os, subprocess, sys, textwrap
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN = "/home/passin/Documents/Blender/TheFarm/r1_runs/farm_2026-10-29/render_v2_astra_scene"
VID = RUN + "/videos"
DEC = "/home/passin/Documents/Blender/TheFarm/r1_runs/farm_2026-10-29/vm_runs/farm_full3/data/decisions.jsonl"
FIG = ROOT + "/docs/media/figures"
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"

GREEN = (118, 185, 0)
WHITE = (255, 255, 255)
GREY = (190, 190, 190)
YELLOW = (255, 210, 63)
RED = (255, 59, 59)
MAGENTA = (255, 47, 214)
BOX_COL = {"fruit_ripe": GREEN, "fruit_turning": YELLOW, "fruit_green": YELLOW,
           "fruit_lesion": RED, "leaf_chlorosis": MAGENTA, "leaf_spots": MAGENTA}

CHLOROSIS_NS = 107116672000      # stop 25, tunnel-09/aisle-02, metre 11.85
CHLOROSIS_T = CHLOROSIS_NS / 1e9
ROUTE_LEN = 647.8

# ----------------------------------------------------------------------------- config
class Cfg:
    W = 1920; H = 1080; FPS = 30; SC = 1.0
    CRF = 18; PRESET = "medium"

CFG = Cfg()


def px(v):
    return int(round(v * CFG.SC))


_FONTS = {}
def font(size, bold=False, mono=False):
    size = max(8, px(size))
    key = (size, bold, mono)
    if key not in _FONTS:
        name = ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf") if mono else \
               ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        _FONTS[key] = ImageFont.truetype(FONT_DIR + name, size)
    return _FONTS[key]


# ----------------------------------------------------------------------------- helpers
def ease(t, t0, dur):
    """smoothstep 0..1 from t0 over dur"""
    if dur <= 0:
        return 1.0 if t >= t0 else 0.0
    x = (t - t0) / dur
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade_in_out(t, t_in, d_in, t_out, d_out):
    return ease(t, t_in, d_in) * (1 - ease(t, t_out, d_out))


def cue(text, phrase, dur, lead=0.15):
    """time (s) within a narration of duration dur at which `phrase` is spoken, by char position"""
    i = text.find(phrase)
    if i < 0:
        raise KeyError(phrase)
    return max(0.0, dur * i / len(text) - lead)


def mmss(t):
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"


def grab(path, t, size=None):
    """one frame from a video at time t as RGB PIL"""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
           "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]
    if size:
        cmd[-1:-1] = ["-vf", f"scale={size[0]}:{size[1]}"]
    w, h = size if size else probe_size(path)
    raw = subprocess.run(cmd, capture_output=True).stdout
    if len(raw) < w * h * 3:   # past the end: take the last frame
        return grab(path, max(0, t - 0.5), size)
    return Image.frombytes("RGB", (w, h), raw[: w * h * 3])


_SIZES = {}
def probe_size(path):
    if path not in _SIZES:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height", "-of", "csv=p=0", path],
                             capture_output=True, text=True).stdout.strip().split(",")
        _SIZES[path] = (int(out[0]), int(out[1]))
    return _SIZES[path]


class Clip:
    """Streams frames of a video from `start` for `span` seconds of source time, played at `speed`,
    resampled to the output fps and scaled to `size`. Repeats the last frame on EOF."""

    def __init__(self, path, start, span, speed, size, fps=None):
        fps = fps or CFG.FPS
        self.size = size
        self.n = 0
        vf = f"setpts=PTS/{speed:.6f},fps={fps},scale={size[0]}:{size[1]}"
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{span + 1.0:.3f}", "-i", path,
               "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.nbytes = size[0] * size[1] * 3
        self.last = None

    def next(self):
        raw = self.p.stdout.read(self.nbytes) if self.p else b""
        if len(raw) == self.nbytes:
            self.last = Image.frombytes("RGB", self.size, raw)
        elif self.last is None:
            self.last = Image.new("RGB", self.size, (0, 0, 0))
        self.n += 1
        return self.last

    def close(self):
        if self.p:
            try:
                self.p.stdout.close(); self.p.kill(); self.p.wait()
            except Exception:
                pass
            self.p = None


class Encoder:
    def __init__(self, path):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{CFG.W}x{CFG.H}", "-r", str(CFG.FPS), "-i", "-",
               "-c:v", "libx264", "-preset", CFG.PRESET, "-crf", str(CFG.CRF), "-pix_fmt", "yuv420p",
               "-r", str(CFG.FPS), "-movflags", "+faststart", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n = 0

    def write(self, im):
        self.p.stdin.write(im.tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        err = self.p.stderr.read().decode()
        self.p.wait()
        if self.p.returncode != 0:
            raise RuntimeError(err)


# ----------------------------------------------------------------------------- drawing
def rounded(draw, box, r, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def card(layer, x, y, w, h, alpha=1.0, accent=True, fill=(10, 12, 14, 205)):
    """dark translucent card with a thin green accent bar on the left"""
    d = ImageDraw.Draw(layer)
    f = (fill[0], fill[1], fill[2], int(fill[3] * alpha))
    rounded(d, [x, y, x + w, y + h], px(6), f)
    if accent:
        d.rectangle([x, y + px(10), x + px(4), y + h - px(10)], fill=GREEN + (int(255 * alpha),))


def text(layer, xy, s, size=24, bold=False, col=WHITE, alpha=1.0, mono=False, anchor=None):
    d = ImageDraw.Draw(layer)
    d.text((xy[0], xy[1]), s, font=font(size, bold, mono), fill=col + (int(255 * alpha),), anchor=anchor)


def text_w(s, size=24, bold=False, mono=False):
    f = font(size, bold, mono)
    return f.getlength(s)


def wrap(s, size, width_px, bold=False):
    f = font(size, bold)
    words = s.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t) <= width_px:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def para(layer, x, y, s, size, width_px, col=WHITE, alpha=1.0, bold=False, lh=1.35):
    """wrapped paragraph; returns the y after the last line"""
    for ln in wrap(s, size, width_px, bold):
        text(layer, (x, y), ln, size, bold, col, alpha)
        y += px(size * lh)
    return y


def typed(layer, x, y, s, n_chars, size, width_px, col=WHITE, alpha=1.0, lh=1.35, cursor=True):
    """typing effect: first n_chars of s, wrapped like the full text so lines don't jump"""
    lines = wrap(s, size, width_px)
    shown = 0
    for ln in lines:
        if shown >= n_chars:
            break
        part = ln[: max(0, n_chars - shown)]
        if cursor and n_chars < len(s) and shown + len(ln) >= n_chars:
            part += "_"
        text(layer, (x, y), part, size, False, col, alpha)
        shown += len(ln) + 1
        y += px(size * lh)
    return y


def label_tag(layer, x, y, s, size=18, alpha=1.0, col=WHITE):
    """small dark tag with text, used for inset labels"""
    d = ImageDraw.Draw(layer)
    w = text_w(s, size, True) + px(16)
    h = px(size * 1.6)
    d.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, int(170 * alpha)))
    d.rectangle([x, y, x + px(3), y + h], fill=GREEN + (int(255 * alpha),))
    text(layer, (x + px(10), y + px(size * 0.25)), s, size, True, col, alpha)
    return w, h


def inset(base, layer, im, x, y, w, h, label=None, alpha=1.0, border=WHITE, flash=0.0):
    """picture-in-picture: 1-px border + small label. `im` is RGB PIL, scaled to w x h."""
    if alpha <= 0:
        return
    im = im.resize((w, h), Image.BILINEAR)
    if alpha < 1:
        base.paste(Image.blend(base.crop((x, y, x + w, y + h)), im, alpha), (x, y))
    else:
        base.paste(im, (x, y))
    d = ImageDraw.Draw(layer)
    bw = 1 if flash <= 0 else max(1, px(3))
    bc = tuple(int(c * (1 - flash) + 255 * flash) for c in border) + (int(230 * alpha),)
    d.rectangle([x - 1, y - 1, x + w, y + h], outline=bc, width=bw)
    if label:
        label_tag(layer, x, y, label, 16, alpha)


def draw_boxes(layer, x, y, w, h, dets, frac=1.0, min_score=0.5):
    """RT-DETR boxes on an inset at (x,y,w,h); `frac` reveals boxes progressively (drawn on)"""
    d = ImageDraw.Draw(layer)
    dets = [dd for dd in dets if dd["score"] >= min_score and dd["label"] in BOX_COL]
    n = int(math.ceil(len(dets) * frac))
    for dd in dets[:n]:
        bx = dd["bbox_norm"]
        x0 = x + max(0, bx[0]) * w; y0 = y + max(0, bx[1]) * h
        x1 = x + min(1, bx[2]) * w; y1 = y + min(1, bx[3]) * h
        col = BOX_COL[dd["label"]]
        lw = px(3) if dd["label"].startswith("leaf") else max(1, px(1.5))
        d.rectangle([x0, y0, x1, y1], outline=col + (255,), width=lw)


def hud(layer, route_t, speed, alpha=1.0, note=None):
    """persistent route clock + progress bar, bottom-left"""
    x, y = px(40), CFG.H - px(58)
    w = px(400)
    d = ImageDraw.Draw(layer)
    d.rectangle([x, y, x + w, y + px(36)], fill=(0, 0, 0, int(150 * alpha)))
    text(layer, (x + px(12), y + px(9)), "ROUTE", 15, True, GREY, alpha)
    text(layer, (x + px(95), y + px(5)), mmss(route_t), 22, True, WHITE, alpha, mono=True)
    sp = f"x{speed:g}" if speed else "PAUSED"
    text(layer, (x + px(205), y + px(9)), sp, 15, True, GREEN if speed else YELLOW, alpha)
    if note:
        text(layer, (x + px(290), y + px(9)), note, 15, False, GREY, alpha)
    # progress bar
    by = y + px(36) + px(4)
    d.rectangle([x, by, x + w, by + px(3)], fill=(255, 255, 255, int(60 * alpha)))
    d.rectangle([x, by, x + int(w * min(1, route_t / ROUTE_LEN)), by + px(3)], fill=GREEN + (int(255 * alpha),))


def view_label(layer, s, alpha=1.0):
    label_tag(layer, px(40), px(36), s, 17, alpha)


def composite(base, layer):
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


def darken(im, amount):
    if amount <= 0:
        return im
    return Image.blend(im, Image.new("RGB", im.size, (0, 0, 0)), amount)


def letterbox(im, w, h, bg=(8, 8, 8)):
    iw, ih = im.size
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    out = Image.new("RGB", (w, h), bg)
    out.paste(im.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
    return out


# ----------------------------------------------------------------------------- data
def load_data():
    d = {}
    d["stops"] = [json.loads(l) for l in open(RUN + "/vlm_nemotron_detailed/vlm_stops.jsonl")]
    d["stop_ns"] = [s["ns"] / 1e9 for s in d["stops"]]
    d["chl"] = next(s for s in d["stops"] if s["ns"] == CHLOROSIS_NS)
    fs = [json.loads(l) for l in open(RUN + "/detection/findings.jsonl")]
    d["chl_find"] = {f["camera"]: f for f in fs if f["image"].startswith(f"side/{CHLOROSIS_NS}_")}
    d["decisions"] = [json.loads(l) for l in open(DEC)]
    d["det_summary"] = json.load(open(RUN + "/detection/summary.json"))
    d["vlm_summary"] = json.load(open(RUN + "/vlm_nemotron_detailed/summary.json"))
    return d


def stop_index(stop_ns, t):
    """index of the last plant stop captured at or before route time t (-1 if none)"""
    k = -1
    for i, s in enumerate(stop_ns):
        if s <= t + 0.2:
            k = i
    return k


def side_png(ns, cam):
    return f"{RUN}/raw/side/{ns}_plant_{cam}.png"


# ----------------------------------------------------------------------------- shots
# Each shot renderer: render_<name>(shot, enc) writes shot["frames"] frames.
# shot dict: id, dur, audio (segment id or None), text, plus chase start/speed.

def chase_speed(span, dur, lo=1.0, hi=2.0):
    return max(lo, min(hi, span / dur))


def plan_shots(durs, plan):
    """Build the ordered shot list; chase time is monotonic through the film."""
    seg = {s["id"]: s for s in plan["segments"]}
    pad = 0.7  # breathing room after each narration
    shots = []
    D = lambda k: durs[k] + pad

    # s0: CCTV overview with title, then chase 0 -> ~30
    d0 = D("s0_open"); a = 11.0
    sp = chase_speed(30.0, d0 - a, 1.0, 2.0)
    shots.append(dict(id="s0_open", dur=d0, audio="s0_open", text=seg["s0_open"]["text"],
                      cctv_dur=a, chase0=0.0, speed=sp))
    c = sp * (d0 - a)
    # s1: edge stack, chase c -> ~62
    d1 = D("s1_edge"); sp = round(chase_speed(62 - c, d1, 1.0, 1.5), 2)
    shots.append(dict(id="s1_edge", dur=d1, audio="s1_edge", text=seg["s1_edge"]["text"], chase0=c, speed=sp))
    c += sp * d1
    # s2: plant stops, chase c -> CHLOROSIS_T exactly
    d2 = D("s2_stops"); sp = (CHLOROSIS_T - c) / d2
    if sp > 2.6:   # start a bit later rather than racing
        sp = 2.6; c = CHLOROSIS_T - sp * d2
    shots.append(dict(id="s2_stops", dur=d2, audio="s2_stops", text=seg["s2_stops"]["text"], chase0=c, speed=round(sp, 2)))
    c = CHLOROSIS_T
    # s3: freeze
    d3 = D("s3_disease_freeze")
    shots.append(dict(id="s3_disease_freeze", dur=d3, audio="s3_disease_freeze",
                      text=seg["s3_disease_freeze"]["text"], chase0=c, speed=0))
    # s4: HERMES, chase 107 -> ~185 at 3x
    d4 = D("s4_hermes"); sp = 3.0
    shots.append(dict(id="s4_hermes", dur=d4, audio="s4_hermes", text=seg["s4_hermes"]["text"], chase0=c, speed=sp))
    c += sp * d4
    # bridge: fast-forward to the back door (245)
    ff_speed = 16.0; ff_dur = (245.0 - c) / ff_speed
    shots.append(dict(id="ff_t09", dur=ff_dur, audio=None, text="", chase0=c, speed=ff_speed,
                      note="rest of tunnel-09"))
    c = 245.0
    # s5: yard, chase 245 -> ~300 (the robot reaches tunnel-10's door ~330)
    d5 = D("s5_yard_safety"); sp = chase_speed(85 - c + 245, d5, 1.5, 2.5)
    sp = round(chase_speed(330 - 245, d5, 1.5, 3.0), 2)
    shots.append(dict(id="s5_yard_safety", dur=d5, audio="s5_yard_safety", text=seg["s5_yard_safety"]["text"], chase0=c, speed=sp))
    c += sp * d5
    # s6: tunnel-10 with ZED, 1x
    d6 = D("s6_r181s"); sp = 1.0
    shots.append(dict(id="s6_r181s", dur=d6, audio="s6_r181s", text=seg["s6_r181s"]["text"], chase0=c, speed=sp))
    c += sp * d6
    # s7: 3-D maps full screen, chase PiP fast-forwards to 600
    d7 = D("s7_maps"); sp = round((600 - c) / d7, 2)
    shots.append(dict(id="s7_maps", dur=d7, audio="s7_maps", text=seg["s7_maps"]["text"], chase0=c, speed=sp))
    c = 600.0
    # s8: home to dock 600 -> 647.8
    d8 = D("s8_results"); sp = round((ROUTE_LEN - 0.2 - c) / d8, 3)
    shots.append(dict(id="s8_results", dur=d8, audio="s8_results", text=seg["s8_results"]["text"], chase0=c, speed=sp))
    # s9: closing card over the final CCTV frame
    d9 = D("s9_pipeline") + 1.5
    shots.append(dict(id="s9_pipeline", dur=d9, audio="s9_pipeline", text=seg["s9_pipeline"]["text"], chase0=ROUTE_LEN, speed=0))
    # frame counts + absolute times
    t = 0.0
    for s in shots:
        s["frames"] = int(round(s["dur"] * CFG.FPS))
        s["dur"] = s["frames"] / CFG.FPS
        s["start"] = t; t += s["dur"]; s["end"] = t
        s["chase_end"] = s["chase0"] + s["speed"] * (s["dur"] - s.get("cctv_dur", 0.0))
    return shots


# --- common per-frame finishing ------------------------------------------------------------------
def finish(base, layer, enc, shot, t, route_t, speed, hud_alpha=1.0, note=None, fade=None):
    if hud_alpha > 0:
        hud(layer, route_t, speed, hud_alpha, note)
    im = composite(base, layer)
    if fade is not None and fade < 1:
        im = darken(im, 1 - fade)
    enc.write(im)


def seg_caption(layer, title, lines, t, t0, alpha_out=1.0, x=None, y=None, w=None, size=22):
    """bottom-right caption card: title + lines; slides in at t0"""
    e = ease(t, t0, 0.5) * alpha_out
    if e <= 0:
        return
    w = w or px(620)
    lines = [ln for s_ in lines for ln in wrap(s_, size, w - px(44))]
    h = px(28 + 44) + px(size * 1.45) * len(lines)
    x = (CFG.W - w - px(40)) if x is None else x
    y = (CFG.H - h - px(48)) if y is None else y
    x += int(px(40) * (1 - e))
    card(layer, x, y, w, h, e)
    text(layer, (x + px(22), y + px(16)), title, 20, True, GREEN, e)
    yy = y + px(50)
    for ln in lines:
        text(layer, (x + px(22), yy), ln, size, False, WHITE, e)
        yy += px(size * 1.45)


# --- s0 -------------------------------------------------------------------------------------------
def render_s0(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]; a = shot["cctv_dur"]
    cctv = Clip(VID + "/shots/cctv7_front_overview_1.mp4", 0.0, a, 1.0, (W, H))
    chase = Clip(VID + "/chase.mp4", shot["chase0"], shot["speed"] * (D - a), shot["speed"], (W, H))
    c_twin = cue(T, "digital twin", D); c_script = cue(T, "Nothing in this film", D)
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if t < a:
            base = cctv.next()
            # title card: fade in 0.8..1.8, out over last 1.2 s of the cctv part
            e = fade_in_out(t, 0.8, 1.0, a - 1.4, 1.0)
            base = darken(base, 0.35 * e)
            if e > 0:
                cx = W // 2
                text(layer, (cx, px(400)), "RoboFarm", 120, True, WHITE, e, anchor="mm")
                d = ImageDraw.Draw(layer)
                d.rectangle([cx - px(140), px(478), cx + px(140), px(482)], fill=GREEN + (int(255 * e),))
                text(layer, (cx, px(530)), "R181-S  ·  autonomous greenhouse scout", 34, False, WHITE, e, anchor="mm")
                text(layer, (cx, px(590)), "Nebius x NVIDIA Global AI Hackathon 2026", 28, False, GREEN, e, anchor="mm")
            view_label(layer, "CCTV-7 · farm overview · NVIDIA Isaac Sim digital twin", 1 - ease(t, a - 0.5, 0.5))
            fade = ease(t, 0.0, 1.0)
            finish(base, layer, enc, shot, t, 0.0, 1, hud_alpha=0.0, fade=fade)
        else:
            base = chase.next()
            tt = t - a
            route_t = shot["chase0"] + shot["speed"] * tt
            view_label(layer, "CHASE CAM · robot third-person view", ease(t, a, 0.4))
            lines = ["Digital twin of our tomato farm · NVIDIA Isaac Sim"]
            if t >= c_script:
                lines.append("Nothing scripted: every turn and stop is the robot's own decision")
            seg_caption(layer, "ROBOFARM · R1", lines, t, max(a + 0.3, c_twin), 1 - ease(t, D - 0.6, 0.5), w=px(760))
            finish(base, layer, enc, shot, t, route_t, shot["speed"], hud_alpha=ease(t, a, 0.5))
    cctv.close(); chase.close()


# --- s1 -------------------------------------------------------------------------------------------
def render_s1(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]; sp = shot["speed"]
    chase = Clip(VID + "/chase.mp4", shot["chase0"], sp * D, sp, (W, H))
    lid_sz = px(330)
    lidar = Clip(VID + "/lidar.mp4", shot["chase0"], sp * D, sp, (lid_sz, lid_sz))
    fig = Image.open(FIG + "/fig_run3_route_vs_estimate.png").convert("RGB").crop((60, 60, 990, 1760))
    fig_h = px(560); fig_w = int(fig_h * fig.width / fig.height)
    fig = fig.resize((fig_w, fig_h), Image.LANCZOS)
    c_book = cue(T, "A rule-based driver", D); c_safe = cue(T, "A SAFETY layer", D)
    c_loc = cue(T, "And localisation", D); c_err = cue(T, "Over this hundred", D)
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        base = chase.next(); lf = lidar.next()
        route_t = shot["chase0"] + sp * t
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        view_label(layer, "CHASE CAM · robot third-person view")
        # LiDAR inset top-right
        e = ease(t, 0.3, 0.5)
        inset(base, layer, lf, W - lid_sz - px(40), px(36) + int(px(30) * (1 - e)), lid_sz, lid_sz,
              "LiDAR scan · robot frame", e)
        # edge-stack card (left-bottom, above HUD)
        lines = []
        if t >= c_book: lines.append(("BOOK", "rule-based aisle driver · follows the farm map"))
        if t >= c_safe: lines.append(("SAFETY", "LiDAR obstacle stop · cuts the wheels"))
        if t >= c_loc: lines.append(("LOCALISATION", "wheels + gyro + LiDAR scan-to-map · no fiducial tags"))
        if lines:
            e2 = ease(t, c_book, 0.5) * (1 - ease(t, D - 0.6, 0.5))
            w = px(840); h = px(70) + px(40) * len(lines)
            x = px(40) + int(px(40) * (1 - e2)); y = H - h - px(120)
            card(layer, x, y, w, h, e2)
            text(layer, (x + px(22), y + px(16)), "ON THE ROBOT · no cloud needed", 20, True, GREEN, e2)
            yy = y + px(52)
            for k, (nm, desc) in enumerate(lines):
                ee = ease(t, [c_book, c_safe, c_loc][k], 0.4) * e2
                text(layer, (x + px(22), yy), nm, 22, True, WHITE, ee)
                text(layer, (x + px(250), yy + px(2)), desc, 20, False, GREY, ee)
                yy += px(40)
        # localisation figure, right side, after the error cue
        if t >= c_err:
            e3 = ease(t, c_err, 0.6) * (1 - ease(t, D - 0.6, 0.5))
            fx = W - fig_w - px(40); fy = px(36) + lid_sz + px(24)
            inset(base, layer, fig, fx + int(px(40) * (1 - e3)), fy, fig_w, fig_h,
                  "robot vs. truth · run 3", e3)
        finish(base, layer, enc, shot, t, route_t, sp)
    chase.close(); lidar.close()


# --- s2 -------------------------------------------------------------------------------------------
def render_s2(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]; sp = shot["speed"]
    chase = Clip(VID + "/chase.mp4", shot["chase0"], sp * D, sp, (W, H))
    sw, sh = px(220), px(391)
    cache = {}
    stop_ns = data["stop_ns"]; stops = data["stops"]
    c_six = cue(T, "Six cameras", D); c_hub = cue(T, "The pictures stream", D)
    last_k = -1; flash_t = -10
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        base = chase.next()
        route_t = shot["chase0"] + sp * t
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        view_label(layer, "CHASE CAM · tunnel-09 · aisle-02")
        k = stop_index(stop_ns, route_t)
        if k != last_k:
            last_k = k; flash_t = t
        if k >= 0:
            ns = stops[k]["ns"]
            if ns not in cache:
                cache[ns] = Image.open(side_png(ns, "right_mid")).convert("RGB").resize((sw, sh), Image.BILINEAR)
            fl = max(0.0, 1 - (t - flash_t) / 0.35)
            inset(base, layer, cache[ns], W - sw - px(40), px(36), sw, sh,
                  f"right-mid · stop {k}", 1.0, flash=fl)
            # counter card
            e = ease(t, 0.2, 0.5) * (1 - ease(t, D - 0.6, 0.5))
            w, h = px(660), px(150)
            x = W - w - px(40) + int(px(40) * (1 - e)); y = px(36) + sh + px(24)
            card(layer, x, y, w, h, e)
            text(layer, (x + px(22), y + px(14)), "PLANT STOPS", 20, True, GREEN, e)
            text(layer, (x + px(22), y + px(44)), f"{k + 1:3d}", 56, True, WHITE, e, mono=True)
            text(layer, (x + px(150), y + px(52)), f"metre {stops[k]['metre']:.2f} · 45 cm apart", 22, False, WHITE, e)
            l2 = "6 cameras per stop · L/R x low/mid/high" if t >= c_six else ""
            l3 = "frames -> Nebius hub · robot moves on" if t >= c_hub else ""
            text(layer, (x + px(150), y + px(86)), l2, 20, False, GREY, e * ease(t, c_six, 0.3))
            text(layer, (x + px(150), y + px(114)), l3, 20, False, GREY, e * ease(t, c_hub, 0.3))
        finish(base, layer, enc, shot, t, route_t, sp)
    chase.close()


# --- s3 freeze ------------------------------------------------------------------------------------
def render_s3(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]
    frozen = grab(VID + "/chase.mp4", CHLOROSIS_T, (W, H))
    chl = data["chl"]; finds = data["chl_find"]
    cams = [("left_high", "L high"), ("left_mid", "L mid"), ("left_low", "L low"),
            ("right_high", "R high"), ("right_mid", "R mid"), ("right_low", "R low")]
    tw, th = px(225), px(400); gap = px(10)
    gx0, gy0 = px(50), px(150)
    tiles = {c: Image.open(side_png(CHLOROSIS_NS, c)).convert("RGB").resize((tw, th), Image.LANCZOS) for c, _ in cams}
    dets = {c: finds["mast_" + c]["detections"] for c, _ in cams}
    # harvest totals for this stop (all six views)
    hv = {"pass": 0, "not_ready": 0, "reject": 0}
    for c, _ in cams:
        for k in hv:
            hv[k] += finds["mast_" + c]["harvest"][k]
    n_chl_right = sum(1 for c in ("right_high", "right_mid", "right_low") for d in dets[c] if d["label"] == "leaf_chlorosis")
    right = chl["right"]
    desc = right["description"]; action = right["action"]
    issue = right["issues"][0]
    c_eyes = cue(T, "On the server", D)
    c_boxes = cue(T, "draws the boxes", D)
    c_fruit = cue(T, "and, separately, every fruit", D)
    c_nemo = cue(T, "Then NVIDIA Nemotron", D)
    c_write = cue(T, "and writes what", D)
    c_advice = cue(T, "Its advice", D)
    c_apart = cue(T, "Health and harvest", D)
    type_rate = len(desc) / max(4.0, (c_advice - c_write - 0.8))
    cx = px(860); cw = W - cx - px(40)
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        base = frozen.copy()
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        e_dark = ease(t, 0.8, 0.8)
        base = darken(base, 0.55 * e_dark)
        # stop tag on the frozen frame
        e_tag = ease(t, 0.15, 0.3)
        label_tag(layer, px(40), px(36), "CHASE CAM · FROZEN", 17, 1.0)
        pulse = 0.5 + 0.5 * math.sin(t * 6)
        w, h = label_tag(layer, px(40), px(72), f"STOP 25 · tunnel-09 / aisle-02 · metre {chl['metre']:.2f} · right plant", 19,
                         e_tag, col=tuple(int(255 * (1 - 0.4 * pulse) + GREEN[j] * 0.4 * pulse) for j in range(3)))
        # six side views slide in from the left
        e_grid = ease(t, 1.0, 0.7)
        if e_grid > 0:
            off = int(px(250) * (1 - e_grid))
            text(layer, (gx0 - off, gy0 - px(24)), "SIX SIDE CAMERAS AT THIS STOP · left plant (top row) · right plant (bottom row)", 17, True, GREY, e_grid)
            for j, (c, nm) in enumerate(cams):
                r, q = divmod(j, 3)
                x = gx0 + q * (tw + gap) - off; y = gy0 + r * (th + gap)
                col = GREEN if c.startswith("right") else WHITE
                inset(base, layer, tiles[c], x, y, tw, th, nm, e_grid, border=col)
                # boxes draw on: fruit after c_boxes, leaf boxes first
                leaf = [d for d in dets[c] if d["label"].startswith("leaf")]
                fruit = [d for d in dets[c] if d["label"].startswith("fruit")]
                draw_boxes(layer, x, y, tw, th, leaf, ease(t, c_boxes, 1.2))
                draw_boxes(layer, x, y, tw, th, fruit, ease(t, c_fruit, 1.5))
            # legend under the grid
            ly = gy0 + 2 * th + gap + px(10)
            if t >= c_boxes:
                e_l = ease(t, c_boxes, 0.4)
                items = [("chlorosis / leaf spots", MAGENTA), ("fruit lesion", RED), ("ripe", GREEN), ("turning / green", YELLOW)]
                lx = gx0
                d = ImageDraw.Draw(layer)
                for nm, col in items:
                    d.rectangle([lx, ly + px(4), lx + px(14), ly + px(18)], outline=col + (int(255 * e_l),), width=px(2))
                    text(layer, (lx + px(22), ly), nm, 17, False, WHITE, e_l)
                    lx += text_w(nm, 17) + px(50)
        # PLANT HEALTH card
        e_ph = ease(t, c_eyes, 0.6)
        if e_ph > 0:
            x = cx + int(px(60) * (1 - e_ph)); y = px(92); h = px(600)
            card(layer, x, y, cw, h, e_ph)
            text(layer, (x + px(22), y + px(16)), "PLANT HEALTH", 22, True, GREEN, e_ph)
            text(layer, (x + px(240), y + px(20)), "separate from harvest", 17, False, GREY, e_ph)
            yy = y + px(56)
            e_b = ease(t, c_boxes, 0.4)
            text(layer, (x + px(22), yy), "RT-DETRv2", 20, True, WHITE, e_b)
            text(layer, (x + px(170), yy + px(2)), "trained only on synthetic renders · leaf anomaly boxes", 19, False, GREY, e_b)
            yy += px(30)
            text(layer, (x + px(170), yy), f"right plant: {n_chl_right} chlorosis regions in 3 views · left plant: none", 19, False, MAGENTA, e_b)
            yy += px(44)
            e_n = ease(t, c_nemo, 0.4)
            text(layer, (x + px(22), yy), "NVIDIA Nemotron 3.5 Super VL", 20, True, WHITE, e_n)
            text(layer, (x + px(430), yy + px(2)), "all six views, one request", 19, False, GREY, e_n)
            yy += px(30)
            text(layer, (x + px(22), yy), f"status: {right['status'].upper()}   issue: {issue['label']}   confidence {issue['confidence']:.2f}", 19, False, YELLOW, e_n)
            yy += px(38)
            if t >= c_write:
                n = int((t - c_write) * type_rate)
                yy = typed(layer, x + px(22), yy, desc, n, 21, cw - px(44), WHITE, e_ph, lh=1.4)
            if t >= c_advice:
                e_a = ease(t, c_advice, 0.4)
                ay = y + h - px(86)
                d = ImageDraw.Draw(layer)
                d.rectangle([x + px(22), ay - px(8), x + cw - px(22), ay - px(7)], fill=(255, 255, 255, int(60 * e_a)))
                text(layer, (x + px(22), ay), "ACTION", 18, True, GREEN, e_a)
                para(layer, x + px(22), ay + px(26), action, 21, cw - px(44), WHITE, e_a)
        # HARVEST card
        e_hv = ease(t, c_fruit, 0.6)
        if e_hv > 0:
            x = cx + int(px(60) * (1 - e_hv)); y = px(92) + px(600) + px(24); h = H - y - px(56)
            card(layer, x, y, cw, h, e_hv)
            text(layer, (x + px(22), y + px(16)), "HARVEST", 22, True, GREEN, e_hv)
            text(layer, (x + px(160), y + px(20)), "RT-DETRv2 fruit grading · this stop, six views", 17, False, GREY, e_hv)
            yy = y + px(62)
            for k, (nm, val, col) in enumerate([("ripe", hv["pass"], GREEN), ("not ready", hv["not_ready"], YELLOW), ("reject", hv["reject"], RED)]):
                xx = x + px(22) + k * px(300)
                text(layer, (xx, yy), f"{val}", 48, True, col, e_hv, mono=True)
                text(layer, (xx + px(90), yy + px(22)), nm, 20, False, WHITE, e_hv)
            if t >= c_apart:
                e_ap = ease(t, c_apart, 0.4)
                text(layer, (x + px(22), y + h - px(40)), "health and harvest are always reported apart", 19, False, GREY, e_ap)
        fade = 1 - ease(t, D - 0.4, 0.4) * 0.0
        finish(base, layer, enc, shot, t, CHLOROSIS_T, 0, note="disease stop")


# --- s4 HERMES ------------------------------------------------------------------------------------
def render_s4(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]; sp = shot["speed"]
    chase = Clip(VID + "/chase.mp4", shot["chase0"], sp * D, sp, (W, H))
    decs = data["decisions"]
    c_go = cue(T, "Those findings go", D); c_goals = cue(T, "It chooses goals", D)
    c_two = cue(T, "On this run it ordered", D); c_clear = cue(T, "Both came back healthy", D)
    def cmd_str(d):
        c = d["command"]
        if c["cmd"] == "survey_farm":
            return "survey_farm"
        return f"revisit  {c['aisle']}  metre {c['metre']:.1f}"
    entries = [(c_goals, decs[0]), (c_two, decs[1]), (c_two + 2.5, decs[2])]
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        base = chase.next()
        route_t = shot["chase0"] + sp * t
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        view_label(layer, "CHASE CAM · tunnel-09 · aisle-02")
        e = ease(t, c_go, 0.6) * (1 - ease(t, D - 0.6, 0.5))
        if e > 0:
            w = px(880); h = px(680)
            x = W - w - px(40) + int(px(60) * (1 - e)); y = px(36)
            card(layer, x, y, w, h, e)
            text(layer, (x + px(22), y + px(16)), "HERMES PLANNER", 22, True, GREEN, e)
            text(layer, (x + px(260), y + px(20)), "NVIDIA Nemotron 3 · Nebius Token Factory", 18, False, GREY, e)
            text(layer, (x + px(22), y + px(50)), "picks goals only · never touches the wheels", 19, False, WHITE, e)
            yy = y + px(92)
            for (t0, d) in entries:
                if t < t0:
                    break
                ee = ease(t, t0, 0.4) * e
                text(layer, (x + px(22), yy), "> " + cmd_str(d), 20, True, WHITE, ee, mono=True)
                yy += px(30)
                n = int((t - t0) * 60)
                yy = typed(layer, x + px(22), yy, d["reasoning"].replace(" ", " ").replace("‑", "-"), n, 19,
                           w - px(44), GREY, ee, lh=1.35)
                tag = "accepted" if d["reply"] == "accepted" else d["reply"]
                text(layer, (x + px(22), yy), f"reply: {tag}", 17, False, GREEN, ee)
                yy += px(40)
            if t >= c_clear:
                ea = ease(t, c_clear, 0.4) * e
                d = ImageDraw.Draw(layer)
                ay = y + h - px(70)
                d.rectangle([x + px(22), ay - px(10), x + w - px(22), ay - px(9)], fill=(255, 255, 255, int(60 * ea)))
                para(layer, x + px(22), ay, "both revisits came back healthy -> two false alarms cleared before they reached the farmer", 19, w - px(44), WHITE, ea)
        finish(base, layer, enc, shot, t, route_t, sp)
    chase.close()


# --- fast-forward bridge --------------------------------------------------------------------------
def render_ff(shot, enc, data):
    W, H = CFG.W, CFG.H
    sp = shot["speed"]; D = shot["dur"]
    chase = Clip(VID + "/chase.mp4", shot["chase0"], sp * D, sp, (W, H))
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        base = chase.next()
        route_t = shot["chase0"] + sp * t
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        view_label(layer, "CHASE CAM · tunnel-09 · aisle-02")
        e = ease(t, 0, 0.3) * (1 - ease(t, D - 0.3, 0.3))
        s_ = f">>  x{sp:g}  ·  {shot['note']}"
        label_tag(layer, W - px(40) - int(text_w(s_, 19, True)) - px(16), px(36), s_, 19, e)
        finish(base, layer, enc, shot, t, route_t, sp)
    chase.close()


# --- s5 yard --------------------------------------------------------------------------------------
def render_s5(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]; sp = shot["speed"]
    chase = Clip(VID + "/chase.mp4", shot["chase0"], sp * D, sp, (W, H))
    fw, fh = px(384), px(216)
    front = Clip(VID + "/front.mp4", shot["chase0"], sp * D, sp, (fw, fh))
    # cut-ins: (segment time, duration, clip, label)
    c_out = cue(T, "across the yard", D)
    c_if = cue(T, "If SAFETY ever", D); c_cos = cue(T, "NVIDIA Cosmos Reason", D)
    c_herm = cue(T, "HERMES then waits", D); c_safe = cue(T, "The robot stays safe", D)
    cut1 = dict(t0=c_out + 0.5, dur=6.0, clip=Clip(VID + "/shots/cctv4_yard_1.mp4", 14.0, 6.0, 1.0, (W, H)),
                label="CCTV-4 · yard")
    cut2 = dict(t0=c_if + 0.3, dur=7.0, clip=Clip(VID + "/shots/person2_yard_1.mp4", 24.0, 7.0, 1.0, (W, H)),
                label="BYSTANDER VIEW · yard")
    cuts = [cut1, cut2]
    steps = [("SAFETY stop", "LiDAR sees an obstacle", c_if), ("front frame", "ZED / front camera", c_if + 1.2),
             ("Cosmos Reason 2", "says what is there, in farm terms", c_cos),
             ("HERMES", "wait (person, animal) / notify farmer", c_herm)]
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        base = chase.next(); ff = front.next()
        route_t = shot["chase0"] + sp * t
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        cut = None
        for c in cuts:
            if c["t0"] <= t < c["t0"] + c["dur"]:
                cut = c
                base = c["clip"].next()
        if cut:
            view_label(layer, cut["label"] + " · the robot keeps driving")
        else:
            view_label(layer, "CHASE CAM · back door -> yard -> tunnel-10")
            inset(base, layer, ff, W - fw - px(40), px(36), fw, fh, "front camera", ease(t, 0.3, 0.5))
        # flow diagram card, bottom-right
        e = ease(t, c_if, 0.6) * (1 - ease(t, D - 0.6, 0.5))
        if e > 0:
            w = px(1120); h = px(172)
            x = W - w - px(40) + int(px(60) * (1 - e)); y = H - h - px(48)
            card(layer, x, y, w, h, e)
            text(layer, (x + px(22), y + px(14)), "IF THE ROBOT HAS TO STOP", 20, True, GREEN, e)
            bw = px(250); bx = x + px(22); by = y + px(48); bh = px(100)
            d = ImageDraw.Draw(layer)
            for k, (nm, desc, t0) in enumerate(steps):
                ee = ease(t, t0, 0.4) * e
                xx = bx + k * (bw + px(24))
                d.rounded_rectangle([xx, by, xx + bw, by + bh], radius=px(5), outline=GREEN + (int(255 * ee),), width=px(2))
                text(layer, (xx + px(12), by + px(12)), nm, 21, True, WHITE, ee)
                para(layer, xx + px(12), by + px(42), desc, 16, bw - px(24), GREY, ee)
                if k < len(steps) - 1:
                    ax = xx + bw + px(4)
                    text(layer, (ax, by + px(26)), ">", 24, True, GREEN, ee)
            if t >= c_safe:
                pass
        finish(base, layer, enc, shot, t, route_t, sp, note="yard" if not cut else None)
    chase.close(); front.close()
    for c in cuts:
        c["clip"].close()


# --- s6 R181-S ------------------------------------------------------------------------------------
def render_s6(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]; sp = shot["speed"]
    chase = Clip(VID + "/chase.mp4", shot["chase0"], sp * D, sp, (W, H))
    zw, zh = px(400), px(225)
    zl = Clip(VID + "/zed_left_540p.mp4", shot["chase0"], sp * D, sp, (zw, zh))
    zd = Clip(VID + "/zed_depth.mp4", shot["chase0"], sp * D, sp, (zw, zh))
    fig = Image.open(FIG + "/fig_slam_paths_astra_scene.png").convert("RGB").crop((70, 40, 545, 1300))
    fig_h = px(640); fig_w = int(fig_h * fig.width / fig.height)
    fig = fig.resize((fig_w, fig_h), Image.LANCZOS)
    c_zed = cue(T, "a ZED 2i stereo", D); c_cuv = cue(T, "NVIDIA cuVSLAM tracks", D)
    c_route = cue(T, "and across the whole", D); c_jet = cue(T, "The same code runs", D)
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        base = chase.next(); l = zl.next(); dpt = zd.next()
        route_t = shot["chase0"] + sp * t
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        view_label(layer, "CHASE CAM · tunnel-10 · aisle-02")
        e = ease(t, c_zed, 0.5)
        inset(base, layer, l, W - zw - px(40), px(36) + int(px(30) * (1 - e)), zw, zh, "ZED 2i · left", e)
        inset(base, layer, dpt, W - zw - px(40), px(36) + zh + px(16) + int(px(30) * (1 - e)), zw, zh, "ZED 2i · stereo depth", e)
        e2 = ease(t, c_cuv, 0.6) * (1 - ease(t, D - 0.6, 0.5))
        if e2 > 0:
            inset(base, layer, fig, px(40) - int(px(40) * (1 - e2)), px(80), fig_w, fig_h,
                  "cuVSLAM vs. truth · 111 m", e2)
        lines = ["Stereolabs ZED 2i replaces the single front camera"]
        if t >= c_cuv: lines.append("NVIDIA cuVSLAM · 526 fps on one GPU")
        if t >= c_route: lines.append("6.4 cm error over the 111 m route · true metric scale")
        if t >= c_jet: lines.append("same code runs on a Jetson inside the robot")
        seg_caption(layer, "R181-S · THE NEW SENSOR", lines, t, c_zed, 1 - ease(t, D - 0.6, 0.5), w=px(700))
        finish(base, layer, enc, shot, t, route_t, sp)
    chase.close(); zl.close(); zd.close()


# --- s7 maps --------------------------------------------------------------------------------------
def render_s7(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]; sp = shot["speed"]
    c_bench = cue(T, "We also benchmarked", D)
    c_ling = cue(T, "Only lingbot-map", D); c_stereo = cue(T, "Stereo, or LiDAR", D)
    t_sw = c_bench + 0.3            # switch from cuVSLAM map to lingbot map
    d_a = t_sw; d_b = D - t_sw
    sp_a = 30.0 / d_a; sp_b = 20.0 / d_b
    cuv = Clip(VID + "/cuvslam_3d.mp4", 0.0, 30.0, sp_a, (W, H))
    lin = Clip(VID + "/lingbot_3d.mp4", 28.0, 20.0, sp_b, (W, H))
    pw, ph = px(448), px(252)
    chase = Clip(VID + "/chase.mp4", shot["chase0"], sp * D, sp, (pw, ph))
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        route_t = shot["chase0"] + sp * t
        ch = chase.next()
        if t < t_sw:
            base = cuv.next()
            lbl = "3-D map: ZED depth placed by cuVSLAM's own poses · no ground truth"
        else:
            base = lin.next()
            lbl = "lingbot-map: 3-D map from the front camera only"
        if abs(t - t_sw) < 0.25:
            base = darken(base, 1 - abs(t - t_sw) / 0.25)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        inset(base, layer, ch, W - pw - px(40), H - ph - px(48), pw, ph, f"chase cam · x{sp:g}", ease(t, 0.3, 0.5))
        lines = [lbl]
        if t < t_sw: lines.append("built on a Nebius GPU · poses from cuVSLAM")
        if t >= c_bench: lines.append("7 camera-only mapping methods benchmarked on this route")
        if t >= c_ling: lines.append("only lingbot-map kept the two identical greenhouses apart (~1 m)")
        if t >= c_stereo: lines.append("stereo or LiDAR for localisation · lingbot-map for the map")
        cw_ = px(800)
        hh = px(72) + px(22 * 1.45) * sum(len(wrap(l_, 22, cw_ - px(44))) for l_ in lines)
        seg_caption(layer, "SERVER-SIDE MAPPING · NEBIUS", lines, t, 0.5, 1 - ease(t, D - 0.6, 0.5),
                    x=px(40), y=H - hh - px(120), w=cw_)
        finish(base, layer, enc, shot, t, route_t, sp, note="chase inset")
    cuv.close(); lin.close(); chase.close()


# --- s8 results -----------------------------------------------------------------------------------
def render_s8(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]; sp = shot["speed"]
    chase = Clip(VID + "/chase.mp4", shot["chase0"], sp * D, sp, (W, H))
    hv = data["det_summary"]["1_harvest_fruit_grading"]["tunnel-09/aisle-02"]
    vs = data["vlm_summary"]
    c_rep = cue(T, "the report is already", D); c_t9 = cue(T, "Tunnel nine", D)
    c_nr = cue(T, "two thousand", D); c_rej = cue(T, "fifty-three", D)
    c_dis = cue(T, "Of nine diseased", D); c_fa = cue(T, "Nemotron raised", D); c_miss = cue(T, "And we are honest", D)
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        base = chase.next()
        route_t = shot["chase0"] + sp * t
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        view_label(layer, "CHASE CAM · yard -> dock")
        e = ease(t, c_rep, 0.6) * (1 - ease(t, D - 0.5, 0.4))
        if e > 0:
            w = px(760); h = px(760)
            x = px(40) - int(px(60) * (1 - e)); y = px(80)
            card(layer, x, y, w, h, e)
            text(layer, (x + px(22), y + px(16)), "SURVEY REPORT · 2026-10-29", 22, True, GREEN, e)
            text(layer, (x + px(430), y + px(20)), "written before the robot is home", 17, False, GREY, e)
            yy = y + px(64)
            # harvest
            eh = ease(t, c_t9, 0.4) * e
            text(layer, (x + px(22), yy), "HARVEST · tunnel-09", 20, True, WHITE, eh); yy += px(40)
            rows = [("ripe, ready to pick", hv["pass"], GREEN, c_t9 + 0.6), ("not ready", hv["not_ready"], YELLOW, c_nr),
                    ("reject", hv["reject"], RED, c_rej)]
            for nm, v, col, t0 in rows:
                ee = ease(t, t0, 0.4) * e
                text(layer, (x + px(22), yy), f"{v:>6,}", 36, True, col, ee, mono=True)
                text(layer, (x + px(190), yy + px(12)), nm, 21, False, WHITE, ee)
                yy += px(50)
            yy += px(14)
            d = ImageDraw.Draw(layer)
            d.rectangle([x + px(22), yy, x + w - px(22), yy + 1], fill=(255, 255, 255, int(60 * e)))
            yy += px(18)
            ep = ease(t, c_dis, 0.4) * e
            text(layer, (x + px(22), yy), "PLANT HEALTH · both greenhouses", 20, True, WHITE, ep); yy += px(40)
            rows2 = [("6 / 9", "diseased plants hidden in the scene were found (RT-DETR + Nemotron)", GREEN, c_dis + 0.5),
                     (f"{vs['flags_not_near_any_anomaly_plant']}", f"Nemotron false alarms over {vs['plants_judged']} plants judged", YELLOW, c_fa)]
            for v, nm, col, t0 in rows2:
                ee = ease(t, t0, 0.4) * e
                text(layer, (x + px(22), yy), v, 36, True, col, ee, mono=True)
                yy = para(layer, x + px(190), yy + px(6), nm, 20, w - px(220), WHITE, ee) + px(14)
            em = ease(t, c_miss, 0.4) * e
            text(layer, (x + px(22), yy + px(4)), "NOT YET DETECTED", 18, True, RED, em)
            para(layer, x + px(22), yy + px(32), "wilting and defoliation: whole-plant shapes the close side cameras cannot see yet - next to fix",
                 20, w - px(44), GREY, em)
        finish(base, layer, enc, shot, t, route_t, sp)
    chase.close()


# --- s9 closing -----------------------------------------------------------------------------------
def render_s9(shot, enc, data):
    W, H = CFG.W, CFG.H
    T = shot["text"]; D = shot["dur"]
    src = VID + "/shots/cctv7_front_overview_2.mp4"
    cctv = Clip(src, 76.0, 11.5, 1.0, (W, H))
    c_one = cue(T, "the robot posts one message", D); c_rest = cue(T, "the Nebius server does", D)
    c_next = cue(T, "Next time", D); c_end = cue(T, "RoboFarm, built", D)
    stages = ["detect", "vlm_stops", "lidar_map", "cuvslam", "lingbot", "cosmos", "report"]
    for i in range(shot["frames"]):
        t = i / CFG.FPS
        base = cctv.next()
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        e_dark = ease(t, 0.3, 1.0)
        base = darken(base, 0.6 * e_dark)
        view_label(layer, "CCTV-7 · farm overview · robot back at the dock", 1 - ease(t, c_end, 0.5))
        e = ease(t, 0.5, 0.6)
        # pipeline diagram
        cxm = W // 2
        text(layer, (cxm, px(150)), "ONE SYSTEM", 30, True, GREEN, e, anchor="mm")
        text(layer, (cxm, px(200)), "robot  ->  POST /run_complete  ->  Nebius server pipeline", 26, False, WHITE, e, anchor="mm")
        d = ImageDraw.Draw(layer)
        # robot and message boxes
        bw, bh = px(214), px(70); gap = px(20)
        total = len(stages) * bw + (len(stages) - 1) * gap
        x0 = cxm - total // 2; by = px(330)
        e1 = ease(t, c_one, 0.5)
        rounded(d, [cxm - px(420), px(250), cxm - px(150), px(250) + px(50)], px(5), (0, 0, 0, 0), GREEN + (int(255 * e1),), px(2))
        text(layer, (cxm - px(285), px(275)), "robot: survey done", 20, True, WHITE, e1, anchor="mm")
        text(layer, (cxm - px(125), px(275)), ">", 26, True, GREEN, e1, anchor="mm")
        rounded(d, [cxm - px(100), px(250), cxm + px(150), px(250) + px(50)], px(5), (0, 0, 0, 0), GREEN + (int(255 * e1),), px(2))
        text(layer, (cxm + px(25), px(275)), "POST /run_complete", 20, True, WHITE, e1, anchor="mm", )
        for k, s in enumerate(stages):
            t0 = c_rest + 0.25 * k
            ee = ease(t, t0, 0.4)
            xx = x0 + k * (bw + gap)
            fill = (118, 185, 0, int(40 * ee))
            rounded(d, [xx, by, xx + bw, by + bh], px(5), fill, GREEN + (int(255 * ee),), px(2))
            text(layer, (xx + bw // 2, by + bh // 2), s, 22, True, WHITE, ee, anchor="mm")
            if k < len(stages) - 1:
                text(layer, (xx + bw + gap // 2, by + bh // 2), ">", 22, True, GREEN, ee, anchor="mm")
        subs = ["RT-DETRv2 boxes", "Nemotron VL · 6 views", "2-D LiDAR map", "stereo poses", "camera-only map", "Cosmos Reason", "TH + EN report"]
        for k, s in enumerate(subs):
            ee = ease(t, c_rest + 0.25 * k, 0.4)
            xx = x0 + k * (bw + gap)
            text(layer, (xx + bw // 2, by + bh + px(22)), s, 15, False, GREY, ee, anchor="mm")
        e2 = ease(t, c_next, 0.5)
        text(layer, (cxm, px(520)), "next run: nobody runs a single step by hand", 24, False, WHITE, e2, anchor="mm")
        e3 = ease(t, c_end, 0.6)
        text(layer, (cxm, px(720)), "RoboFarm", 90, True, WHITE, e3, anchor="mm")
        d.rectangle([cxm - px(120), px(782), cxm + px(120), px(786)], fill=GREEN + (int(255 * e3),))
        text(layer, (cxm, px(830)), "R181-S  ·  Nebius x NVIDIA Global AI Hackathon 2026", 30, False, WHITE, e3, anchor="mm")
        text(layer, (cxm, px(880)), "NVIDIA Isaac Sim · cuVSLAM · Nemotron 3 / 3.5 Super VL · Cosmos Reason · Nebius Token Factory", 19, False, GREY, e3, anchor="mm")
        fade = 1 - ease(t, D - 1.5, 1.3)
        finish(base, layer, enc, shot, t, ROUTE_LEN, 0, hud_alpha=0.0, fade=fade)
    cctv.close()


RENDERERS = {"s0_open": render_s0, "s1_edge": render_s1, "s2_stops": render_s2, "s3_disease_freeze": render_s3,
             "s4_hermes": render_s4, "ff_t09": render_ff, "s5_yard_safety": render_s5, "s6_r181s": render_s6,
             "s7_maps": render_s7, "s8_results": render_s8, "s9_pipeline": render_s9}


def render_shot(args):
    shot, cfg, out = args
    CFG.W, CFG.H, CFG.FPS, CFG.SC, CFG.CRF, CFG.PRESET = cfg
    data = load_data()
    enc = Encoder(out)
    try:
        RENDERERS[shot["id"]](shot, enc, data)
    finally:
        enc.close()
    return shot["id"], enc.n


# ----------------------------------------------------------------------------- audio
def build_audio(shots, tts_dir, out_wav, total):
    sr = 48000
    buf = np.zeros(int(total * sr) + sr, dtype=np.float32)
    lead = 0.35
    for s in shots:
        if not s["audio"]:
            continue
        wav = os.path.join(tts_dir, s["audio"] + ".wav")
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                             capture_output=True).stdout
        a = np.frombuffer(raw, dtype=np.float32)
        i0 = int((s["start"] + lead) * sr)
        buf[i0:i0 + len(a)] += a
        s["audio_start"] = s["start"] + lead
        s["audio_end"] = s["start"] + lead + len(a) / sr
    buf = buf[: int(total * sr)]
    pcm = np.clip(buf, -1, 1)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-c:a", "pcm_s16le", out_wav], input=pcm.tobytes(), check=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", required=True)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/demo/robofarm_demo.mp4"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--only", default=None, help="comma-separated shot ids to render (debug)")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.draft:
        CFG.W, CFG.H, CFG.FPS, CFG.SC, CFG.CRF, CFG.PRESET = 640, 360, 10, 640 / 1920, 23, "veryfast"
    work = a.work or os.path.join(os.path.dirname(a.out), "_work" + ("_draft" if a.draft else ""))
    os.makedirs(work, exist_ok=True)
    plan = json.load(open(os.path.join(ROOT, "docs/demo/narration.json")))
    durs = json.load(open(os.path.join(a.tts, "durations.json")))["durations"]
    shots = plan_shots(durs, plan)
    total = shots[-1]["end"]
    for s in shots:
        print(f"{s['id']:18s} {s['start']:7.2f} -> {s['end']:7.2f}  dur {s['dur']:6.2f}  chase {s['chase0']:6.1f} -> {s['chase_end']:6.1f}  x{s['speed']}")
    print(f"total {total:.1f}s")
    cfg = (CFG.W, CFG.H, CFG.FPS, CFG.SC, CFG.CRF, CFG.PRESET)
    todo = [s for s in shots if not a.only or s["id"] in a.only.split(",")]
    jobs = [(s, cfg, os.path.join(work, s["id"] + ".mp4")) for s in todo]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for sid, n in ex.map(render_shot, jobs):
            print("rendered", sid, n, "frames", flush=True)
    # concat
    lst = os.path.join(work, "concat.txt")
    with open(lst, "w") as f:
        for s in shots:
            f.write(f"file '{os.path.join(work, s['id'] + '.mp4')}'\n")
    silent = os.path.join(work, "video_silent.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent], check=True)
    wav = os.path.join(work, "narration_full.wav")
    build_audio(shots, a.tts, wav, total)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", a.out], check=True)
    # timeline for music alignment
    tl = {"fps": CFG.FPS, "size": [CFG.W, CFG.H], "total_s": round(total, 3),
          "voice": json.load(open(os.path.join(a.tts, "durations.json")))["voice"],
          "narration_lufs": json.load(open(os.path.join(a.tts, "durations.json")))["lufs"],
          "disease_freeze": None, "segments": []}
    for s in shots:
        row = {"id": s["id"], "start_s": round(s["start"], 3), "end_s": round(s["end"], 3),
               "chase_route_time_s": [round(s["chase0"], 2), round(s["chase_end"], 2)], "chase_speed": s["speed"]}
        if s["audio"]:
            row["narration_start_s"] = round(s["audio_start"], 3); row["narration_end_s"] = round(s["audio_end"], 3)
        tl["segments"].append(row)
        if s["id"] == "s3_disease_freeze":
            tl["disease_freeze"] = {"start_s": round(s["start"], 3), "end_s": round(s["end"], 3),
                                    "route_time_s": CHLOROSIS_T, "stop": 25, "aisle": "tunnel-09/aisle-02", "metre": 11.85}
    if not a.draft:
        json.dump(tl, open(os.path.join(ROOT, "docs/demo/timeline.json"), "w"), indent=1)
    else:
        json.dump(tl, open(os.path.join(work, "timeline.json"), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
