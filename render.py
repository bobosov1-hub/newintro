#!/usr/bin/env python3
"""Render a storyboard to video (or stills).

    python render.py storyboards/gimje_1997.json                    # fresh seed, chaos 0.4
    python render.py storyboards/gimje_1997.json --seed 4817 --chaos 0.55
    python render.py storyboards/gimje_1997.json --stills 3.0 9.5 31.4 --scale 0.5
    python render.py storyboards/gimje_1997.json --start 40 --end 50 --scale 0.5

The same seed and chaos always produce the identical frames. Every build writes a manifest
(seed, chaos and the choices the seed made per scene) next to the video.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from collage import palette as P  # noqa: E402
from collage.compositor import Compositor  # noqa: E402
from collage.easing import in_out_sine, out_expo, out_expo_settle, stepped  # noqa: E402
from collage.finish import Finish  # noqa: E402
from collage.newspaper import Pool  # noqa: E402
from collage.scene import build_scene  # noqa: E402
from collage.seeding import Build  # noqa: E402


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


# ------------------------------------------------------------------------------ timeline
class Timeline:
    """Scenes + transitions. Transition specs live on the INCOMING scene."""

    PUSH_OUT, PUSH_IN, SLIDE = 0.30, 0.55, 0.42

    def __init__(self, scenes, duration):
        self.scenes = scenes
        self.duration = duration
        for i in range(1, len(scenes)):
            prev, cur = scenes[i - 1], scenes[i]
            b = cur.t0
            if cur.transition.get("type") == "push":
                prev.camera.extra.append(self._push_out(b))
                cur.camera.extra.append(self._push_in(b))

    def _push_out(self, b):
        def fn(t):
            u = (t - (b - self.PUSH_OUT)) / self.PUSH_OUT
            if u <= 0:
                return 0.0, 0.0, 0.0, 0.0
            k = 2.0 ** (10.0 * min(u, 1.2) - 10.0)
            return 0.0, 0.0, 0.34 * k, 0.0
        return fn

    def _push_in(self, b):
        def fn(t):
            u = (t - b) / self.PUSH_IN
            if u >= 1:
                return 0.0, 0.0, 0.0, 0.0
            k = 1.0 - out_expo(max(u, 0.0))
            return 0.0, 0.0, -0.24 * k, 0.0
        return fn

    def index(self, t):
        for i, s in enumerate(self.scenes):
            if t < s.t1 or i == len(self.scenes) - 1:
                if t >= s.t0 or i == 0:
                    return i
        return len(self.scenes) - 1

    def plan(self, t):
        """-> list of (scene, weight/offset info), leak amount, leak id."""
        i = self.index(t)
        cur = self.scenes[i]
        nxt = self.scenes[i + 1] if i + 1 < len(self.scenes) else None
        leak, leak_id, shift = 0.0, 0, 0.0
        # leaving `cur` into `nxt`
        if nxt is not None:
            b = nxt.t0
            tr = nxt.transition
            if tr.get("type") == "push" and t >= b - self.PUSH_OUT:
                if tr.get("leak"):
                    u = (t - (b - self.PUSH_OUT)) / (self.PUSH_OUT + 0.3)
                    leak, leak_id, shift = 0.55 * math.sin(math.pi * min(max(u, 0), 1)), i, u
                return [("full", cur)], leak, leak_id, shift
        # arriving in `cur`
        prev = self.scenes[i - 1] if i > 0 else None
        tr = cur.transition
        typ = tr.get("type", "cut")
        dt_in = t - cur.t0
        if prev is not None and typ == "push" and dt_in < 0.08:
            a = (dt_in + 0.0) / 0.08
            if tr.get("leak"):
                u = (t - (cur.t0 - self.PUSH_OUT)) / (self.PUSH_OUT + 0.3)
                leak, leak_id, shift = 0.55 * math.sin(math.pi * min(max(u, 0), 1)), i - 1, u
            return [("mix", prev, cur, a)], leak, leak_id, shift
        if prev is not None and typ == "push" and tr.get("leak") and dt_in < 0.3:
            u = (t - (cur.t0 - self.PUSH_OUT)) / (self.PUSH_OUT + 0.3)
            leak, leak_id, shift = 0.55 * math.sin(math.pi * min(max(u, 0), 1)), i - 1, u
        if prev is not None and typ == "slide" and dt_in < self.SLIDE:
            if tr.get("leak"):
                u = dt_in / (self.SLIDE + 0.2)
                leak, leak_id, shift = 0.45 * math.sin(math.pi * min(u, 1)), i, u
            return [("slide", prev, cur, dt_in, tr.get("dir", "left"))], leak, leak_id, shift
        if prev is not None and typ == "cut" and tr.get("leak") and dt_in < 0.35:
            leak, leak_id, shift = 0.45 * math.sin(math.pi * dt_in / 0.35), i, dt_in
        return [("full", cur)], leak, leak_id, shift


# ------------------------------------------------------------------------------ frame
G = {}


def render_frame(fi):
    tl, comp, fin, fps = G["tl"], G["comp"], G["fin"], G["fps"]
    t = fi / fps
    plan, leak, leak_id, shift = tl.plan(t)
    kind = plan[0][0]
    if kind == "full":
        rgb, acc = comp.render(plan[0][1].layers, plan[0][1].camera, t, diagrams=plan[0][1].diagrams)
    elif kind == "mix":
        _, a_s, b_s, w = plan[0]
        r1, c1 = comp.render(a_s.layers, a_s.camera, t, diagrams=a_s.diagrams)
        r2, c2 = comp.render(b_s.layers, b_s.camera, t, diagrams=b_s.diagrams)
        w = in_out_sine(w)
        rgb, acc = r1 * (1 - w) + r2 * w, c1 * (1 - w) + c2 * w
    else:  # slide: the incoming sheet is laid over the outgoing one (stepped, it is an entrance)
        _, a_s, b_s, dt_in, direction = plan[0]
        r1, c1 = comp.render(a_s.layers, a_s.camera, t, diagrams=a_s.diagrams)
        r2, c2 = comp.render(b_s.layers, b_s.camera, t, diagrams=b_s.diagrams)
        W, H = comp.W, comp.H
        te = stepped(dt_in, comp.step_fps)
        te_prev = stepped(max(dt_in - 1.0 / fps, 0.0), comp.step_fps)
        k = 1.0 - out_expo_settle(te / tl.SLIDE, 0.02)
        kp = 1.0 - out_expo_settle(te_prev / tl.SLIDE, 0.02)
        vec = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[direction]
        span = W if vec[0] else H
        off = k * span * 1.04
        vel = abs(kp - k) * span * 0.45
        sx, sy = int(round(vec[0] * off)), int(round(vec[1] * off))
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        r2s = cv2.warpAffine(r2, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        c2s = cv2.warpAffine(c2, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.ones((H, W), np.float32), M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if vel > 1.5:
            ksz = int(vel) | 1
            kern = (ksz, 1) if vec[0] else (1, ksz)
            r2s = cv2.blur(r2s, kern)
            c2s = cv2.blur(c2s, kern)
            mask = cv2.blur(mask, kern)
        sh = cv2.GaussianBlur(cv2.warpAffine(mask, np.float32([[1, 0, 18 * comp.vs], [0, 1, 26 * comp.vs]]), (W, H)),
                              (0, 0), 28 * comp.vs)
        r1 = r1 * (1 - 0.45 * sh * (1 - mask))[..., None]
        rgb = r1 * (1 - mask[..., None]) + r2s * mask[..., None]
        acc = c1 * (1 - mask) + c2s * mask
    last = tl.scenes[-1]
    fo = float(G.get("fade_out", 0.0))
    if fo > 0 and t > tl.duration - fo:
        k = in_out_sine((t - (tl.duration - fo)) / fo)
        rgb = rgb * (1 - k) + P.INK * k
    return fin.apply(rgb, acc, fi, leak=leak, leak_id=leak_id, leak_shift=shift, grain_step=G.get("grain_step", 1))


def _worker_encode(args):
    f0, f1, path = args
    W, H, fps, crf = G["comp"].W, G["comp"].H, G["fps"], G["crf"]
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", G["preset"], "-crf", str(crf)]
    if G.get("tune", "none") != "none":
        cmd += ["-tune", G["tune"]]
    cmd += ["-pix_fmt", "yuv420p", "-g", str(int(fps * 2)), path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t_start = time.time()
    for fi in range(f0, f1):
        img = render_frame(fi)
        p.stdin.write((np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8).tobytes())
        if (fi - f0) % 60 == 0:
            el = time.time() - t_start
            print(f"  [{os.path.basename(path)}] frame {fi - f0 + 1}/{f1 - f0}  {el:.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    return path


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storyboard")
    ap.add_argument("--seed", type=int, default=None, help="default: a fresh seed for every build")
    ap.add_argument("--chaos", type=float, default=0.4, help="0 = clean and registered, 1 = barely holding together")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="0.5 for quick previews")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--stills", type=float, nargs="*", help="render these times to PNG instead of video")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--grain-step", type=int, default=2, help="frames per grain refresh (1 = every frame)")
    ap.add_argument("--tune", default="none", help="x264 tune, e.g. grain (bigger files)")
    a = ap.parse_args()

    sb = json.load(open(a.storyboard, encoding="utf-8"))
    build = Build(a.seed, a.chaos)
    fps = a.fps or float(sb.get("fps", 30))
    W0, H0 = sb.get("size", [1920, 1080])
    W, H = int(W0 * a.scale), int(H0 * a.scale)
    duration = float(sb["duration"])
    print(f"[render] {sb.get('title', '')}  {build.tag()}  {W}x{H}@{fps}", flush=True)

    t = time.time()
    pool = Pool(build)
    specs = sb["scenes"]
    if a.stills:
        # stills only need the scenes around the requested times
        specs = [s for s in specs if any(s["start"] - 0.6 <= ts <= s["end"] + 0.6 for ts in a.stills)]
    elif a.start > 0 or a.end is not None:
        e = a.end if a.end is not None else duration
        specs = [s for s in specs if s["end"] + 0.6 >= a.start and s["start"] - 0.6 <= e]
    scenes = [build_scene(s, build, pool, W0, H0) for s in specs]
    tl = Timeline(scenes, duration)
    print(f"[render] built {len(scenes)} scenes in {time.time() - t:.1f}s", flush=True)
    comp = Compositor(W, H, fps=fps, step_fps=float(sb.get("step_fps", 12)), vs=a.scale)
    fin = Finish(W, H, build)
    G.update(tl=tl, comp=comp, fin=fin, fps=fps, crf=a.crf, preset=a.preset, grain_step=a.grain_step, tune=a.tune,
             fade_out=float(sb["scenes"][-1].get("fade_out", 0.0)))

    out_dir = os.path.join(HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.storyboard))[0]
    manifest = {"storyboard": a.storyboard, "seed": build.seed, "chaos": build.chaos, "fps": fps, "size": [W, H],
                "palette": P.HEX, "scenes": [dict(s.info, start=s.t0, end=s.t1, transition=s.transition) for s in scenes]}

    if a.stills:
        for ts in a.stills:
            img = render_frame(int(round(ts * fps)))
            path = os.path.join(out_dir, f"{stem}_{build.tag()}_t{ts:06.2f}.png")
            cv2.imwrite(path, cv2.cvtColor((img * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            print("[still]", path, flush=True)
        return

    f0 = int(round(a.start * fps))
    f1 = int(round((a.end if a.end is not None else duration) * fps))
    n = f1 - f0
    workers = max(1, min(a.workers, n // 30 or 1))
    tmp = os.path.join(out_dir, "_segments")
    os.makedirs(tmp, exist_ok=True)
    chunks = []
    for k in range(workers):
        s0 = f0 + n * k // workers
        s1 = f0 + n * (k + 1) // workers
        chunks.append((s0, s1, os.path.join(tmp, f"seg_{k:02d}.mp4")))
    t = time.time()
    if workers == 1:
        segs = [_worker_encode(chunks[0])]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as p:
            segs = p.map(_worker_encode, chunks, chunksize=1)
    print(f"[render] {n} frames in {time.time() - t:.0f}s", flush=True)

    listfile = os.path.join(tmp, "list.txt")
    with open(listfile, "w") as fh:
        for s in segs:
            fh.write(f"file '{s}'\n")
    out = a.out or os.path.join(out_dir, f"{stem}_{build.tag()}.mp4")
    audio = a.audio or sb.get("audio")
    if audio and not os.path.isabs(audio):
        audio = os.path.join(os.path.dirname(os.path.abspath(a.storyboard)), audio)
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listfile]
    if audio and os.path.exists(audio):
        cmd += ["-ss", f"{a.start:.3f}", "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print("[render] wrote", out, flush=True)


if __name__ == "__main__":
    main()
