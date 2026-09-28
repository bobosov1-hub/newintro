"""Paper stock: the organic, slightly warm base everything sits on."""
import cv2
import numpy as np

from . import palette as P
from .noise import blur, fbm


def fibres(h, w, rng, per_mpx=1100, length=(4, 26)):
    """Random short fibres, lighter and darker than the stock."""
    canvas = np.zeros((h, w), np.float32)
    n = int(per_mpx * h * w / 1e6)
    xs = rng.uniform(0, w, n)
    ys = rng.uniform(0, h, n)
    ang = rng.uniform(0, np.pi, n)
    ln = rng.uniform(length[0], length[1], n)
    val = rng.choice([-1.0, 1.0], n) * rng.uniform(0.25, 1.0, n)
    for x, y, a, l, v in zip(xs, ys, ang, ln, val):
        dx, dy = np.cos(a) * l / 2, np.sin(a) * l / 2
        bend = rng.uniform(-0.25, 0.25) * l
        p0 = (int((x - dx) * 4), int((y - dy) * 4))
        pm = (int((x + bend * np.sin(a)) * 4), int((y - bend * np.cos(a)) * 4))
        p1 = (int((x + dx) * 4), int((y + dy) * 4))
        cv2.polylines(canvas, [np.array([p0, pm, p1], np.int32)], False, float(v), 1, cv2.LINE_AA, shift=2)
    return blur(canvas, 0.45)


def paper_luma(h, w, rng, amount=1.0):
    """Multiplicative paper structure around 1.0: mottling, cloudiness, fibres, grain."""
    mott = fbm(h, w, rng, cell=max(h, w) / 3.0, octaves=3) * 0.016
    cloud = fbm(h, w, rng, cell=70, octaves=3) * 0.009
    fib = fibres(h, w, rng) * 0.035
    grain = blur(rng.standard_normal((h, w)).astype(np.float32), 0.55) * 0.010
    return (1.0 + amount * (mott + cloud + fib + grain)).astype(np.float32)


def sheet(h, w, rng, base=None, amount=1.0):
    """An RGB sheet of paper (or of ink-black stock if base=INK)."""
    base = P.PAPER if base is None else base
    tex = paper_luma(h, w, rng, amount)
    if base is P.INK:
        # black stock: fibres show as slightly lighter flecks
        rgb = base + (P.GREY - base) * np.clip((tex - 1.0) * 2.2 + 0.035, 0, 1)[..., None]
    else:
        rgb = base * tex[..., None]
    return np.clip(rgb, 0, 1).astype(np.float32)


def stain(h, w, rng, strength=0.05):
    """A faint, large tea/age stain (multiplicative, towards grey)."""
    n = fbm(h, w, rng, cell=max(h, w) / 2.5, octaves=3)
    m = np.clip((n - 0.9) * 0.9, 0, 1)
    ring = np.clip(1.0 - np.abs(n - 0.95) * 12.0, 0, 1) * 0.6
    return (1.0 - strength * (m + ring)).astype(np.float32)
