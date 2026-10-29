# RoboFarm demo — music brief (for the AI music tool)

**Length:** 5:30 (a little longer than the film; I will trim and fade).  
**Role:** an under-score, not a song. The narration (female, Australian, calm) sits on top the whole time; the music must never compete with it. I will duck the music by about 14 dB under speech automatically, so write it so that it still reads when quiet.

## Prompt to paste

> Cinematic neoclassical underscore for a technology launch film, 5 minutes 30 seconds, instrumental, no vocals, no drums. Strings and piano led, warm and natural, hopeful and unhurried, like a morning in a greenhouse. Sparse felt piano and sustained string pads underneath spoken narration; nothing busy in the 200 Hz to 4 kHz range while the narration plays. Slow build with one clear lift around 3:45, a brief quiet moment around 1:30 to 2:15, and a resolved, confident ending at 5:20. Tempo 68 bpm, key D major, soft dynamics, long reverb, no sudden hits.

## Shape, aligned to the film

| Time | What is on screen | Music |
|---|---|---|
| 0:00–0:30 | Title over the greenhouses, robot leaves the dock | Intro swell: piano motif enters, strings rise and settle |
| 0:30–1:30 | Robot drives into the aisle, stops at plants | Steady, low, almost a pad. This is where most narration is |
| 1:30–2:15 | **Freeze** on the sick plant, AI text types in | Quietest part: a single sustained chord, piano only |
| 2:15–3:10 | Planner decides, robot crosses the yard | Gentle motion returns, light pulse (no drums) |
| 3:10–3:45 | Stereo camera, cuVSLAM path | Warmth builds, strings widen |
| 3:45–4:25 | 3-D maps grow and orbit | **The lift**: fullest moment, still soft |
| 4:25–5:10 | Robot docks, results card | Resolve, pull back under the numbers |
| 5:10–5:25 | Pipeline card, closing line | Final cadence, hold, fade by 5:30 |

(Exact times will move by a few seconds once the narration audio is final; I will send the final `timeline.json`.)

## Delivery
- WAV or MP3 at 48 kHz, stereo.
- If the tool can export stems, send **piano** and **strings** separately; otherwise the mix is fine.
- No fade-in at 0:00 longer than 2 s; leave the ending ring out so I can fade it.
