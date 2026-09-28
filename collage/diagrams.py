"""Hand-drawn diagrams (underlines, rings, crosses, timelines, arrows, dashed outlines).

Drawn diagrams always run at full frame rate - they are never stepped.
"""
import math

import cv2
import numpy as np

from . import palette as P
from .compositor import layer_matrix
from .easing import out_expo


def wobble(pts, rng, amp=1.5, cell=18.0):
    pts = np.asarray(pts, np.float32)
    n = len(pts)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    k = max(2, int(seg[-1] / cell) + 2)
    g = rng.standard_normal((k, 2)).astype(np.float32) * amp
    idx = seg / max(seg[-1], 1e-6) * (k - 1)
    off = np.stack([np.interp(idx, np.arange(k), g[:, 0]), np.interp(idx, np.arange(k), g[:, 1])], 1)
    return pts + off


def resample(pts, step=4.0):
    pts = np.asarray(pts, np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(2, int(s[-1] / step))
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], 1).astype(np.float32)


def partial(pts, p):
    if p >= 1.0:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    lim = s[-1] * max(p, 0.0)
    k = int(np.searchsorted(s, lim))
    if k <= 0:
        return pts[:1]
    out = pts[:k].copy()
    if k < len(pts):
        f = (lim - s[k - 1]) / max(seg[k - 1], 1e-6)
        out = np.vstack([out, pts[k - 1] + (pts[k] - pts[k - 1]) * f])
    return out


class Diagram:
    """strokes: list of polylines. Coordinates are world units on a plane at `depth`, or sprite
    pixels of the layer named `attach` (then they follow that layer, entrance included)."""

    def __init__(self, strokes, t_in, dur=0.7, depth=1.0, width=5.0, color=None, attach=None, stagger=0.0,
                 dashed=None, t_out=None, dur_out=0.25, name="", z=None):
        self.strokes = [resample(s, 3.0) for s in strokes]
        self.t_in, self.dur, self.depth, self.width = t_in, dur, depth, width
        self.color = P.INK if color is None else color
        self.attach, self.stagger, self.dashed = attach, stagger, dashed
        self.t_out, self.dur_out, self.name = t_out, dur_out, name
        self.z = z                     # draw order among layers (None = on top of everything)

    def _to_screen(self, pts, cam, comp, layers, t):
        if self.attach:
            lay = next((l for l in layers if l.name == self.attach), None)
            if lay is None:
                return None, 1.0
            st = lay.state(t, comp.step_fps)
            if st is None:
                return None, 1.0
            M = layer_matrix(lay, st, cam, comp.W, comp.H, comp.vs)
            q = (M[:2, :2] @ pts.T).T + M[:2, 2]
            return q, math.sqrt(abs(np.linalg.det(M[:2, :2])))
        cx, cy, cz, roll = cam
        s = 1.0 / max(self.depth - cz, 0.05)
        q = (pts - np.array([cx, cy], np.float32)) * s
        rr = math.radians(roll)
        R = np.array([[math.cos(rr), -math.sin(rr)], [math.sin(rr), math.cos(rr)]], np.float32)
        q = (q @ R.T) * comp.vs + np.array([comp.W / 2, comp.H / 2], np.float32)
        return q, s * comp.vs

    def draw(self, canvas, acc, t, cam, comp, layers):
        if t < self.t_in:
            return
        fade = 1.0
        if self.t_out is not None and t > self.t_out:
            fade = 1.0 - min(1.0, (t - self.t_out) / self.dur_out)
            if fade <= 0:
                return
        H, W = canvas.shape[:2]
        mask = None
        for i, s in enumerate(self.strokes):
            p = out_expo((t - self.t_in - i * self.stagger) / self.dur)
            if p <= 0:
                continue
            part = partial(s, p)
            if len(part) < 2:
                continue
            q, sc = self._to_screen(part, cam, comp, layers, t)
            if q is None:
                continue
            if mask is None:
                mask = np.zeros((H, W), np.uint8)
            th = max(1, int(round(self.width * sc)))
            if self.dashed:
                on, off = self.dashed
                seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
                acc_len = np.r_[0, np.cumsum(seg)]
                period = (on + off) * sc
                phase = (acc_len % period) < on * sc
                start = None
                for j in range(len(q)):
                    if phase[j] and start is None:
                        start = j
                    if (not phase[j] or j == len(q) - 1) and start is not None:
                        if j - start >= 1:
                            cv2.polylines(mask, [(q[start:j + 1] * 16).astype(np.int32)], False, 255, th, cv2.LINE_AA, shift=4)
                        start = None
            else:
                cv2.polylines(mask, [(q * 16).astype(np.int32)], False, 255, th, cv2.LINE_AA, shift=4)
        if mask is None:
            return
        m = (mask.astype(np.float32) / 255.0) * 0.94 * fade
        canvas *= (1.0 - m)[..., None]
        canvas += (self.color * m[..., None]).astype(np.float32)
        acc *= (1.0 - m)


# ---------------------------------------------------------------------------- shapes
def underline(x0, x1, y, rng, curl=True):
    xs = np.linspace(x0, x1, 40)
    ys = y + 3.0 * np.sin(np.linspace(0, math.pi, 40)) + np.linspace(0, rng.uniform(-6, 6), 40)
    pts = np.stack([xs, ys], 1)
    if curl:
        tail = np.array([[x1 + 10, y - 2], [x1 + 2, y - 14], [x1 - 28, y + 6]])
        pts = np.vstack([pts, tail])
    return [wobble(pts, rng, 1.2)]


def ring(cx, cy, rx, ry, rng, turns=1.12, tilt=None):
    tilt = rng.uniform(-12, 12) if tilt is None else tilt
    n = 120
    a0 = rng.uniform(0, 2 * math.pi)
    a = a0 + np.linspace(0, 2 * math.pi * turns, n)
    rr = 1.0 + 0.06 * np.sin(np.linspace(0, 3.0, n) + rng.uniform(0, 6))
    x = np.cos(a) * rx * rr
    y = np.sin(a) * ry * rr
    tr = math.radians(tilt)
    pts = np.stack([cx + x * math.cos(tr) - y * math.sin(tr), cy + x * math.sin(tr) + y * math.cos(tr)], 1)
    return [wobble(pts, rng, 1.5)]


def cross(cx, cy, size, rng):
    s = size / 2
    a = wobble(np.array([[cx - s, cy - s], [cx + s, cy + s * 0.9]]), rng, 1.0)
    b = wobble(np.array([[cx + s * 0.95, cy - s], [cx - s, cy + s]]), rng, 1.0)
    return [resample(a, 3), resample(b, 3)]


def arrow(p0, p1, rng, head=26.0, bend=0.15):
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    mid = (p0 + p1) / 2 + np.array([-(p1 - p0)[1], (p1 - p0)[0]]) * bend
    t = np.linspace(0, 1, 40)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * mid + t ** 2 * p1
    d = pts[-1] - pts[-4]
    d /= np.linalg.norm(d) + 1e-6
    n = np.array([-d[1], d[0]])
    h1 = np.array([p1 - d * head + n * head * 0.55, p1, p1 - d * head - n * head * 0.55])
    return [wobble(pts, rng, 1.0), wobble(h1, rng, 0.6)]


def timeline(x0, x1, y, ticks, rng):
    main = wobble(np.array([[x0, y], [x1, y]]), rng, 0.8)
    strokes = [resample(main, 3)]
    for i in range(ticks + 1):
        x = x0 + (x1 - x0) * i / ticks
        hgt = 18 if i in (0, ticks) else 10
        strokes.append(np.array([[x, y - hgt], [x, y + hgt]], np.float32))
    return strokes


def outline_from_alpha(alpha, offset_xy, res, step=6):
    """Contour of an alpha mask -> polyline in world units (offset = top-left in world)."""
    m = (alpha > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    c = c[::step]
    c = np.vstack([c, c[:1]])
    return [c / res + np.array(offset_xy, np.float32)]
