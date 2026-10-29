#!/usr/bin/env python3
"""Drop externally produced narration clips (e.g. CapCut TTS) into a make_tts.py output dir, replacing the
edge-tts take for those segments: same trim + loudness normalisation, and durations.json updated.
Usage: import_voice.py TTS_DIR s0_open=/path/clip1.mp3 s1_edge=/path/clip3.mp3 ..."""
import json, os, subprocess, sys

tts_dir = sys.argv[1]
durs_path = os.path.join(tts_dir, "durations.json")
meta = json.load(open(durs_path))
for arg in sys.argv[2:]:
    sid, src = arg.split("=", 1)
    wav = os.path.join(tts_dir, sid + ".wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                    "-af", f"silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.15,"
                           f"areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.25,areverse,"
                           f"loudnorm=I={meta['lufs']}:TP=-1.5:LRA=11",
                    "-ar", "48000", "-ac", "1", wav], check=True)
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav],
                             capture_output=True, text=True).stdout.strip())
    meta["durations"][sid] = d
    meta.setdefault("external", {})[sid] = os.path.basename(src)
    print(sid, f"{d:.2f}s", "<-", os.path.basename(src))
json.dump(meta, open(durs_path, "w"), indent=1)
print("total", round(sum(meta["durations"].values()), 1))
