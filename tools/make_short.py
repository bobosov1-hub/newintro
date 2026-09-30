#!/usr/bin/env python3
"""Episode script (shorts/*.json) -> timed storyboard -> vertical short.

    python tools/make_short.py shorts/jjajang_1970.json --tts          # write the TTS script to paste
    python tools/make_short.py shorts/jjajang_1970.json                # estimated timing (no voice yet)
    python tools/make_short.py shorts/jjajang_1970.json --audio audio/jjajang_1970.mp3
    python tools/make_short.py shorts/jjajang_1970.json --audio ... --render --seed 1970

Timing
  With --audio the narration is cut at its pauses (ffmpeg silencedetect) and the pauses that best
  match the script's line breaks become the line boundaries - read the lines with a clear pause
  between them (Typecast does this by default when each line is its own paragraph).
  Without audio every line gets a syllable-based estimate so the cut can be previewed.

Scene times
  Inside a scene, a number is seconds from the scene start; "L3" is the start of line 3,
  "L3@0.5" halfway through line 3, "L3+0.2" 0.2s after it starts.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

SYL_PER_SEC = 7.0      # Typecast default speed, Korean
PAUSE = 0.35           # between lines
COMMA = 0.14           # per comma / full stop inside a line


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def syllables(text):
    return len(re.findall(r"[가-힣0-9A-Za-z]", text))


def speech_len(text):
    return syllables(text) / SYL_PER_SEC + COMMA * len(re.findall(r"[,.?!]", text.rstrip(".?! ")))


def estimate(lines, lead=0.4):
    t, out = lead, []
    for ln in lines:
        d = speech_len(ln["tts"])
        out.append((t, t + d))
        t += d + PAUSE
    return out, t + 0.5


def audio_duration(path):
    r = subprocess.run([ffmpeg_bin(), "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    m = re.findall(r"time=(\d+):(\d+):([\d.]+)", r.stderr)
    h, mi, s = m[-1]
    return int(h) * 3600 + int(mi) * 60 + float(s)


def speech_segments(path, noise_db=-35, min_sil=0.22):
    r = subprocess.run([ffmpeg_bin(), "-i", path, "-af", f"silencedetect=noise={noise_db}dB:d={min_sil}",
                        "-f", "null", "-"], capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    dur = audio_duration(path)
    sil = list(zip(starts, ends + [dur] * (len(starts) - len(ends))))
    segs, t = [], 0.0
    for s0, s1 in sil:
        if s0 > t + 0.05:
            segs.append((t, s0))
        t = s1
    if dur > t + 0.05:
        segs.append((t, dur))
    return segs, dur


def align(lines, segs):
    """Choose len(lines)-1 of the pauses between speech segments as line boundaries (DP on the
    distance to the syllable-proportional expectation)."""
    n, m = len(lines), len(segs)
    if m < n:
        return None
    t0, t1 = segs[0][0], segs[-1][1]
    w = [speech_len(ln["tts"]) for ln in lines]
    tot = sum(w)
    exp, acc = [], 0.0
    for k in range(n - 1):
        acc += w[k]
        exp.append(t0 + (t1 - t0) * acc / tot)
    gaps = [(segs[j][1] + segs[j + 1][0]) / 2 for j in range(m - 1)]
    INF = float("inf")
    # dp[k][j]: boundary k placed at gap j
    dp = [[INF] * (m - 1) for _ in range(n - 1)]
    bk = [[-1] * (m - 1) for _ in range(n - 1)]
    for j in range(m - 1):
        dp[0][j] = abs(gaps[j] - exp[0])
    for k in range(1, n - 1):
        best, arg = INF, -1
        for j in range(m - 1):
            if j - 1 >= 0 and dp[k - 1][j - 1] < best:
                best, arg = dp[k - 1][j - 1], j - 1
            if best < INF:
                dp[k][j] = best + abs(gaps[j] - exp[k])
                bk[k][j] = arg
    j = min(range(m - 1), key=lambda q: dp[n - 2][q]) if n > 1 else None
    cuts = []
    for k in range(n - 2, -1, -1):
        cuts.append(j)
        j = bk[k][j]
    cuts = cuts[::-1]
    out, first = [], 0
    for c in cuts + [m - 1]:
        out.append((segs[first][0], segs[c][1]))
        first = c + 1
    return out


def resolve(v, t0, spans):
    if isinstance(v, (int, float)):
        return t0 + float(v)
    m = re.fullmatch(r"L(\d+)(?:@([\d.]+))?([+-][\d.]+)?", str(v).strip())
    if not m:
        raise SystemExit(f"bad time reference {v!r}")
    a, b = spans[int(m.group(1)) - 1]
    t = a + (b - a) * float(m.group(2) or 0.0)
    return t + float(m.group(3) or 0.0)


def build(ep, spans, duration, audio):
    starts = []
    for i, sc in enumerate(ep["scenes"]):
        li = int(sc["from_line"]) - 1
        starts.append(0.0 if i == 0 else spans[li][0] - 0.12)
    scenes = []
    for i, sc in enumerate(ep["scenes"]):
        t0 = starts[i]
        t1 = starts[i + 1] if i + 1 < len(starts) else duration
        s = json.loads(json.dumps(sc))
        s.pop("from_line", None)
        s["start"], s["end"] = round(t0, 3), round(t1, 3)

        def R(v):
            return round(min(max(resolve(v, t0, spans), t0), t1 - 0.05), 3)
        for key in ("hero", "lockup"):
            if key in s and "enter" in s[key]:
                s[key]["enter"] = R(s[key]["enter"])
        if "hero" in s and "exit" in s["hero"]:
            s["hero"]["exit"]["t"] = R(s["hero"]["exit"]["t"])
        for sp in s.get("support", []):
            if "enter" in sp:
                sp["enter"] = R(sp["enter"])
        for b in s.get("beats", []):
            b["t"] = R(b["t"])
        for d in s.get("diagrams", []):
            d["t"] = R(d["t"])
        scenes.append(s)
    caps = []
    for i, ln in enumerate(ep["lines"]):
        a, b = spans[i]
        nxt = spans[i + 1][0] if i + 1 < len(spans) else duration
        chunks = [c.strip() for c in ln.get("caption", ln["tts"]).split(" / ") if c.strip()]
        weights = [max(1, syllables(c)) for c in chunks]
        tot, t = sum(weights), a
        for c, wgt in zip(chunks, weights):
            e = t + (b - a) * wgt / tot
            caps.append({"start": round(t, 3), "end": round(e, 3), "text": c})
            t = e
        caps[-1]["end"] = round(nxt, 3)
    sb = {"title": ep["title"], "duration": round(duration, 3), "fps": ep.get("fps", 30),
          "step_fps": ep.get("step_fps", 12), "size": ep["size"], "header": ep.get("header"),
          "captions_style": ep.get("captions_style", {}), "captions": caps, "scenes": scenes}
    if audio:
        sb["audio"] = os.path.relpath(os.path.abspath(audio), os.path.join(HERE, "storyboards"))
    return sb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--audio")
    ap.add_argument("--tts", action="store_true", help="write the narration script for the TTS tool and exit")
    ap.add_argument("--noise", type=float, default=-35, help="silence threshold in dB")
    ap.add_argument("--render", action="store_true", help="run render.py afterwards")
    a, rest = ap.parse_known_args()
    ep = json.load(open(a.episode, encoding="utf-8"))
    lines = ep["lines"]
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    if a.tts:
        path = os.path.join(HERE, "output", f"{ep['id']}_tts.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(ln["tts"] for ln in lines) + "\n")
        est, dur = estimate(lines)
        print(f"[tts] {path}  ({len(lines)} lines, ~{dur:.1f}s at {SYL_PER_SEC} syllables/s)")
        return
    if a.audio:
        segs, dur = speech_segments(a.audio, a.noise)
        spans = align(lines, segs)
        if spans is None:
            print(f"[timing] only {len(segs)} pauses-separated phrases for {len(lines)} lines - "
                  f"using proportional timing inside the narration")
            est, edur = estimate(lines, lead=0.0)
            k0, k1 = segs[0][0], segs[-1][1]
            sc = (k1 - k0) / (edur - 0.5)
            spans = [(k0 + x * sc, k0 + y * sc) for x, y in est]
        duration = dur + 0.3
    else:
        spans, duration = estimate(lines)
    for i, (x, y) in enumerate(spans):
        print(f"  L{i + 1:<2} {x:6.2f}-{y:6.2f}  {lines[i]['tts']}")
    sb = build(ep, spans, duration, a.audio)
    out = os.path.join(HERE, "storyboards", f"{ep['id']}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(sb, fh, ensure_ascii=False, indent=1)
    print(f"[storyboard] {out}  duration {duration:.2f}s, {len(sb['scenes'])} scenes, {len(sb['captions'])} captions")
    if a.render:
        subprocess.run([sys.executable, os.path.join(HERE, "render.py"), out] + rest, check=True)


if __name__ == "__main__":
    main()
