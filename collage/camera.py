"""2.5D camera over layered planes. Always evaluated at full frame rate (never stepped).

World: the plane at depth 1.0 maps 1:1 to screen pixels when the camera is at rest.
A point (x, y) on a plane at depth d projects with scale 1 / (d - cz).
"""
import math

import numpy as np

from .easing import in_out_cubic, in_out_expo, in_out_sine
from .noise import noise1d


class Camera:
    def __init__(self, t0, t1, build, key, drift=None, beats=(), focus=1.0, rack=None, start=(0.0, 0.0, 0.0)):
        rng = build.rng("camera", key)
        self.t0, self.t1 = t0, t1
        dur = max(t1 - t0, 0.1)
        # slow continuous drift: constant velocity, no dead holds
        if drift is None:
            ang = float(rng.uniform(0, 2 * np.pi))
            speed = float(rng.uniform(9.0, 16.0))          # world px / second
            drift = (math.cos(ang) * speed * dur, math.sin(ang) * speed * dur * 0.6,
                     float(rng.uniform(0.030, 0.050)) * dur / 6.0)
        self.p0 = np.array(start, np.float64)
        self.p1 = self.p0 + np.array(drift, np.float64)
        self.beats = []
        for b in beats:
            b = dict(b)
            b.setdefault("dur", 0.65)
            b.setdefault("amount", 0.10 if b.get("type", "push") == "push" else -0.08)
            b.setdefault("pull", 0.75)
            b.setdefault("roll", 0.0)
            self.beats.append(b)
        self.focus_depth = focus
        self.rack = rack                                    # (t_start, dur, from_depth)
        n = int(dur * 30) + 60
        self.hx = noise1d(n, rng, 45.0, 3) * build.amt(0.6, 2.2)
        self.hy = noise1d(n, rng, 45.0, 3) * build.amt(0.5, 1.8)
        self.hr = noise1d(n, rng, 60.0, 3) * build.amt(0.03, 0.18)
        self.extra = []                                     # (fn(t) -> dx, dy, dz, droll) for transitions

    def _hand(self, t):
        i = (t - self.t0) * 30.0 + 30
        i0 = int(np.clip(math.floor(i), 0, len(self.hx) - 2))
        f = float(np.clip(i - i0, 0, 1))
        lerp = lambda arr: float(arr[i0] * (1 - f) + arr[i0 + 1] * f)
        return lerp(self.hx), lerp(self.hy), lerp(self.hr)

    def pose(self, t):
        u = (t - self.t0) / max(self.t1 - self.t0, 1e-6)
        p = self.p0 + (self.p1 - self.p0) * u
        cx, cy, cz = float(p[0]), float(p[1]), float(p[2])
        roll = 0.0
        for b in self.beats:
            k = in_out_expo((t - b["t"]) / b["dur"]) if b.get("ease", "expo") == "expo" else in_out_cubic((t - b["t"]) / b["dur"])
            if k <= 0.0:
                continue
            cz += b["amount"] * k
            tx, ty = b.get("target", (cx, cy))
            cx += (tx - cx) * b["pull"] * k
            cy += (ty - cy) * b["pull"] * k
            roll += b["roll"] * k
        hx, hy, hr = self._hand(t)
        cx, cy, roll = cx + hx, cy + hy, roll + hr
        for fn in self.extra:
            dx, dy, dz, dr = fn(t)
            cx, cy, cz, roll = cx + dx, cy + dy, cz + dz, roll + dr
        return cx, cy, cz, roll

    def focus(self, t):
        if self.rack:
            ts, dur, frm = self.rack
            k = in_out_sine((t - ts) / dur)
            return frm + (self.focus_depth - frm) * k
        return self.focus_depth
