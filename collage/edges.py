"""Edge treatments, chosen per element by the seed and weighted toward the clean options.

sticker : clean cut with a thin paper border
hand    : roughened, hand-trimmed edge (scissor facets, uneven margin)
torn    : torn edge with the white fibre core showing and a fuzzy fringe
print   : no cutout - a rectangular print with a narrow margin
"""
import cv2
import numpy as np

from . import palette as P
from .noise import blur, dist_outside, fbm, streaks, value_noise
from .paper import paper_luma

KINDS = ("sticker", "print", "hand", "torn")


def choose(build, rng, hero=False, photo=False):
    """Hero gets the cleanest edge available; torn stays an accent, never a texture."""
    c = build.chaos
    if hero:
        # a clean cut keeps the hero's silhouette crisp; a plain print only for real photographs
        w = [0.86 - 0.12 * c, 0.14 if photo else 0.0, 0.08 * c, 0.04 * c]
    else:
        w = [0.40 - 0.10 * c, 0.30 - 0.10 * c, 0.18 + 0.10 * c, 0.12 + 0.10 * c]
    return build.pick(rng, KINDS, w)


def _pad(rgb, a, acc, m):
    m = int(m)
    rgb = cv2.copyMakeBorder(rgb, m, m, m, m, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    a = cv2.copyMakeBorder(a, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    if acc is not None:
        acc = cv2.copyMakeBorder(acc, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    return rgb, a, acc


def _stock(h, w, rng, base, amount=0.8):
    return (base * paper_luma(h, w, rng, amount)[..., None]).astype(np.float32)


def _over(rgb, a, under_rgb):
    return rgb * a[..., None] + under_rgb * (1.0 - a[..., None])


def sticker(rgb, a, acc, rng, bw):
    rgb, a, acc = _pad(rgb, a, acc, bw * 2 + 6)
    h, w = a.shape
    d = dist_outside(blur(a, bw * 0.35) > 0.3)
    outer = np.clip(bw - d + 0.5, 0, 1)
    stock = _stock(h, w, rng, P.PAPER)
    rim = np.clip(1.0 - np.abs(d - (bw - 0.7)) / 1.2, 0, 1) * 0.10
    stock *= (1.0 - rim)[..., None]
    return _over(rgb, a, stock), np.maximum(outer, a), acc


def hand(rgb, a, acc, rng, bw, chaos):
    rgb, a, acc = _pad(rgb, a, acc, bw * 3 + 8)
    h, w = a.shape
    d = dist_outside(blur(a, bw * 0.5) > 0.25)
    n = value_noise(h, w, rng, cell=max(h, w) / 6.0)
    thr = np.maximum(bw * (1.15 + (0.35 + 0.45 * chaos) * n), bw * 0.4)
    m1 = (d < thr).astype(np.uint8)
    cnts, _ = cv2.findContours(m1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    eps = (0.0035 + 0.006 * chaos) * max(h, w)
    canvas = np.zeros((h, w), np.uint8)
    for c in cnts:
        if cv2.contourArea(c) < 30:
            continue
        ap = cv2.approxPolyDP(c, eps, True)
        cv2.fillPoly(canvas, [(ap * 4).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=2)
    outer = np.maximum(canvas.astype(np.float32) / 255.0, np.clip(a * 1.5, 0, 1))
    stock = _stock(h, w, rng, P.PAPER)
    do = dist_outside(outer > 0.5)
    di = dist_outside(outer < 0.5)
    rim = np.clip(1.0 - di / 1.4, 0, 1) * (do == 0) * 0.12
    stock *= (1.0 - rim)[..., None]
    return _over(rgb, a, stock), outer, acc


def torn(rgb, a, acc, rng, bw, chaos):
    rgb, a, acc = _pad(rgb, a, acc, bw * 4 + 10)
    h, w = a.shape
    d = dist_outside(blur(a, bw * 0.6) > 0.25)
    n_lo = value_noise(h, w, rng, cell=max(h, w) / 6.0)
    n_hi = fbm(h, w, rng, cell=16.0, octaves=4)
    thr = np.maximum(bw * (1.6 + 0.6 * n_lo) + bw * (0.35 + 0.25 * chaos) * n_hi, bw * 0.5)
    outer = np.clip(thr - d + 0.5, 0, 1)
    rimw = 2.0 + 2.5 * np.clip(value_noise(h, w, rng, cell=28.0) * 0.5 + 0.5, 0, 1)
    core = np.clip((d - (thr - rimw)) / 1.3, 0, 1) * outer
    zone = np.clip(1.0 - (d - thr) / 3.5, 0, 1) * (d > thr)
    fib = streaks(h, w, rng, length=7, angle_deg=float(rng.uniform(0, 180)))
    fuzz = np.clip(fib - 1.1, 0, 1) * zone * 0.85
    margin = _stock(h, w, rng, P.mix(P.PAPER, P.GREY, 0.08))
    core_rgb = _stock(h, w, rng, P.PAPER, 1.2)
    under = _over(core_rgb, core, margin)
    out_a = np.maximum(np.maximum(outer, fuzz), a)
    return _over(rgb, a, under), out_a, acc


def print_margin(rgb, a, acc, rng, margin):
    """`a` is already the rectangular print; add a narrow paper margin around it."""
    rgb, a, acc = _pad(rgb, a, acc, margin + 4)
    h, w = a.shape
    ys, xs = np.where(a > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    outer = np.zeros((h, w), np.float32)
    outer[max(0, y0 - margin):y1 + margin, max(0, x0 - margin):x1 + margin] = 1.0
    stock = _stock(h, w, rng, P.PAPER)
    edge = np.zeros((h, w), np.float32)
    edge[max(0, y0 - margin):y1 + margin, max(0, x0 - margin):x1 + margin] = 1.0
    edge = 1.0 - blur(edge, 0.8)
    stock *= (1.0 - 0.10 * np.clip(edge * 2.0, 0, 1))[..., None]
    return _over(rgb, a, stock), np.maximum(outer, a), acc


def apply(kind, rgb, a, acc, rng, bw, chaos):
    if kind == "sticker":
        return sticker(rgb, a, acc, rng, bw)
    if kind == "hand":
        return hand(rgb, a, acc, rng, bw, chaos)
    if kind == "torn":
        return torn(rgb, a, acc, rng, bw, chaos)
    if kind == "print":
        return print_margin(rgb, a, acc, rng, int(round(bw * 1.3)))
    raise ValueError(kind)


def torn_mask(h, w, rng, rough=1.0, inset=6.0):
    """A torn-paper rectangle mask (for newspaper fragments, tape, scraps)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    n = fbm(h, w, rng, cell=max(10.0, min(h, w) / 5.0), octaves=5)
    thr = inset + rough * (inset * 0.8) * n
    return np.clip(d - thr + 0.5, 0, 1).astype(np.float32)
