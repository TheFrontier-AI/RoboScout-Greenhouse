#!/usr/bin/env python3
"""Generate per-segment narration with edge-tts, loudness-normalise to -16 LUFS,
and write durations.json. Usage: make_tts.py OUT_DIR [--voice VOICE]"""
import json, os, subprocess, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EDGE = ("/tmp/claude-1000/-home-passin-Documents-Blender-TheFarm/"
        "53dbb844-b15f-438d-a6e2-8e4a8daeb73c/scratchpad/vlmvenv/bin/edge-tts")

# spoken-form fixes only (facts and numbers unchanged)
SPEAK = [("RT-DETR", "R T DETR"), ("cuVSLAM", "cu V SLAM"), ("R181-S", "R one eighty-one S"),
         ("R1,", "R one,"), ("lingbot-map", "lingbot map"), ("HERMES", "Hermes"),
         ("tunnel-09/aisle-02", "tunnel nine aisle two")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--voice", default="en-AU-NatashaNeural")
    ap.add_argument("--lufs", type=float, default=-16.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plan = json.load(open(os.path.join(ROOT, "docs/demo/narration.json")))
    durs = {}
    for s in plan["segments"]:
        sid = s["id"]
        mp3 = os.path.join(a.out, sid + ".mp3")
        wav = os.path.join(a.out, sid + ".wav")
        txt = s["text"]
        for k, v in SPEAK:
            txt = txt.replace(k, v)
        if not (os.path.exists(mp3) and os.path.getsize(mp3) > 1000):
            for attempt in range(3):
                r = subprocess.run([EDGE, "--voice", a.voice, "--text", txt, "--write-media", mp3],
                                   capture_output=True, text=True)
                if r.returncode == 0 and os.path.getsize(mp3) > 1000:
                    break
                print(sid, "retry", attempt, r.stderr[-300:], flush=True)
        # loudness normalise (single pass, linear-ish), 48 kHz mono, trim leading/trailing silence
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3,
                        "-af", f"silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.15,"
                               f"areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.25,areverse,"
                               f"loudnorm=I={a.lufs}:TP=-1.5:LRA=11",
                        "-ar", "48000", "-ac", "1", wav], check=True)
        d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "csv=p=0", wav], capture_output=True, text=True).stdout.strip())
        durs[sid] = d
        print(sid, f"{d:.2f}s", flush=True)
    json.dump({"voice": a.voice, "lufs": a.lufs, "durations": durs},
              open(os.path.join(a.out, "durations.json"), "w"), indent=1)
    print("total", sum(durs.values()))


if __name__ == "__main__":
    main()
