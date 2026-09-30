"""Burned-in captions and a series header for vertical shorts.

Captions are big, heavy type on torn paper strips (the collage language), placed in the
lower-middle band of a 9:16 frame - above the YouTube Shorts title/description overlay and
clear of the right-hand buttons. `**word**` marks a word in the accent colour.

Each caption pops in on the 12fps step (a tiny scale settle) and holds; nothing fades.
"""
import math
import re

import cv2
import numpy as np

from . import palette as P
from .edges import torn_mask
from .easing import out_expo_settle, stepped
from .paper import paper_luma
from .typography import font


def _parse(text):
    """'바로 **짜장면**이죠' -> [(run, accent?)]"""
    out = []
    for i, part in enumerate(re.split(r"\*\*", text)):
        if part:
            out.append((part, i % 2 == 1))
    return out


def _line_image(runs, size, weight):
    f = font(size, weight)
    asc, desc = f.getmetrics()
    total = sum(f.getlength(r) for r, _ in runs)
    pad = int(size * 0.2)
    W, H = int(math.ceil(total)) + 2 * pad, asc + desc + 2 * pad
    from PIL import Image, ImageDraw
    ink = Image.new("L", (W, H), 0)
    acc = Image.new("L", (W, H), 0)
    x = float(pad)
    for r, is_acc in runs:
        d = ImageDraw.Draw(acc if is_acc else ink)
        d.text((x, pad), r, font=f, fill=255)
        x += f.getlength(r)
    return np.asarray(ink, np.float32) / 255.0, np.asarray(acc, np.float32) / 255.0


def _wrap(line):
    """Split one caption line in two at the space closest to its middle (never inside **...**)."""
    plain = line.replace("**", "")
    best, arg, depth = 1e9, None, 0
    i = 0
    while i < len(line):
        if line.startswith("**", i):
            depth ^= 1
            i += 2
            continue
        if line[i] == " " and not depth:
            left = len(line[:i].replace("**", ""))
            d = abs(left - len(plain) / 2)
            if d < best:
                best, arg = d, i
        i += 1
    if arg is None:
        return [line]
    return [line[:arg].strip(), line[arg + 1:].strip()]


class _Card:
    """One caption: premultiplied RGBA image, laid out once."""

    def __init__(self, text, W, size, weight, rng, max_w):
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        inner = max_w - int(size * 1.6)
        wrapped = []
        for ln in lines:
            ink, _ = _line_image(_parse(ln), size, weight)
            if ink.shape[1] > inner and " " in ln:
                wrapped.extend(_wrap(ln))
            else:
                wrapped.append(ln)
        imgs = []
        for ln in wrapped:
            s = size
            ink, acc = _line_image(_parse(ln), s, weight)
            while ink.shape[1] > inner and s > size * 0.55:
                s *= 0.93
                ink, acc = _line_image(_parse(ln), s, weight)
            imgs.append((ink, acc))
        gap = int(size * 0.12)
        cw = max(i.shape[1] for i, _ in imgs)
        ch = sum(i.shape[0] for i, _ in imgs) + gap * (len(imgs) - 1)
        padx, pady = int(size * 0.45), int(size * 0.28)
        Wc, Hc = cw + 2 * padx, ch + 2 * pady
        ink_all = np.zeros((Hc, Wc), np.float32)
        acc_all = np.zeros((Hc, Wc), np.float32)
        strip = np.zeros((Hc, Wc), np.float32)
        y = pady
        for ink, acc in imgs:
            x = (Wc - ink.shape[1]) // 2
            ink_all[y:y + ink.shape[0], x:x + ink.shape[1]] = ink
            acc_all[y:y + acc.shape[0], x:x + acc.shape[1]] = acc
            # one torn paper strip per line, slightly wider than the words
            lh = ink.shape[0]
            sx0 = max(0, x - int(size * 0.35))
            sx1 = min(Wc, x + ink.shape[1] + int(size * 0.35))
            sh, sw = lh + int(size * 0.10), sx1 - sx0
            m = torn_mask(sh, sw, rng, 0.8, max(2.0, 5.0 * size / 76.0))
            y0 = max(0, y - int(size * 0.05))
            hh = min(sh, Hc - y0)
            strip[y0:y0 + hh, sx0:sx0 + sw] = np.maximum(strip[y0:y0 + hh, sx0:sx0 + sw], m[:hh])
            y += lh + gap
        tex = paper_luma(Hc, Wc, rng, 1.2)
        paper = P.PAPER * tex[..., None]
        rgb = paper * (1 - ink_all[..., None]) + P.INK * ink_all[..., None]
        rgb = rgb * (1 - acc_all[..., None]) + P.ACCENT * acc_all[..., None]
        a = np.clip(np.maximum(strip, np.maximum(ink_all, acc_all)), 0, 1)
        # soft contact shadow under the strip
        sh = cv2.GaussianBlur(np.roll(a, (int(size * 0.08), int(size * 0.05)), axis=(0, 1)), (0, 0), size * 0.12)
        self.shadow = (sh * 0.45).astype(np.float32)
        self.rgb = rgb.astype(np.float32)
        self.a = a.astype(np.float32)
        self.rot = float(rng.uniform(-1.6, 1.6))


def _warp(img, M, W, H, border=0):
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


class Overlay:
    def __init__(self, W, H, spec, build, scale=1.0, step_fps=12.0):
        self.W, self.H, self.vs, self.step_fps = W, H, scale, step_fps
        cs = spec.get("captions_style", {})
        self.y = float(cs.get("y", 0.69))              # centre of the caption band, fraction of H
        size = float(cs.get("size", 78)) * scale
        weight = int(cs.get("weight", 800))
        max_w = int(W * float(cs.get("max_width", 0.84)))
        rng = build.rng("captions")
        self.caps = []
        for c in spec.get("captions", []):
            self.caps.append((float(c["start"]), float(c["end"]), _Card(c["text"], W, size, weight, rng, max_w)))
        self.header = None
        hd = spec.get("header")
        if hd:
            self.header = self._header(hd, build.rng("header"), scale)

    def _header(self, hd, rng, s):
        """Series tag on an ink strip + an episode line under it, top of frame."""
        from PIL import Image, ImageDraw
        f1 = font(58 * s, 900)
        f2 = font(36 * s, 500)
        t1, t2 = hd.get("series", ""), hd.get("episode", "")
        w1 = int(f1.getlength(t1)) + int(80 * s)
        h1 = int(96 * s)
        w2 = int(f2.getlength(t2)) + int(20 * s)
        Wc, Hc = max(w1, w2) + int(40 * s), h1 + int(70 * s)
        tag = np.zeros((Hc, Wc), np.float32)
        x1 = (Wc - w1) // 2
        tag[:h1, x1:x1 + w1] = torn_mask(h1, w1, rng, 0.5, max(2.0, 5.0 * s))
        im1 = Image.new("L", (Wc, Hc), 0)
        ImageDraw.Draw(im1).text((Wc / 2, h1 / 2), t1, font=f1, fill=255, anchor="mm")
        im2 = Image.new("L", (Wc, Hc), 0)
        ImageDraw.Draw(im2).text((Wc / 2, h1 + 34 * s), t2, font=f2, fill=255, anchor="mm")
        txt1 = np.asarray(im1, np.float32) / 255.0
        txt2 = np.asarray(im2, np.float32) / 255.0
        rgb = np.empty((Hc, Wc, 3), np.float32)
        rgb[:] = P.INK
        rgb = rgb * (1 - txt1[..., None]) + P.PAPER * txt1[..., None]
        a = np.clip(np.maximum(tag, txt2), 0, 1)
        full_rgb = np.zeros((self.H, self.W, 3), np.float32)
        full_a = np.zeros((self.H, self.W), np.float32)
        x0 = (self.W - Wc) // 2
        y0 = int(float(hd.get("y", 0.085)) * self.H)
        full_rgb[y0:y0 + Hc, x0:x0 + Wc] = rgb
        full_a[y0:y0 + Hc, x0:x0 + Wc] = a
        return full_rgb, full_a

    def apply(self, rgb, t):
        if self.header is not None:
            hr, ha = self.header
            rgb = rgb * (1 - ha[..., None]) + hr * ha[..., None]
        for t0, t1, card in self.caps:
            if not (t0 <= t < t1):
                continue
            te = stepped(t - t0, self.step_fps)
            k = out_expo_settle(min(te / 0.25, 1.0), 0.06)
            sc = 0.92 + 0.08 * k
            h, w = card.a.shape
            cx, cy = self.W / 2.0, self.y * self.H
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), card.rot, sc)
            M[0, 2] += cx - w / 2.0
            M[1, 2] += cy - h / 2.0
            sh = _warp(card.shadow, M, self.W, self.H)
            a = _warp(card.a, M, self.W, self.H)
            c = _warp(card.rgb, M, self.W, self.H)
            rgb = rgb * (1 - sh[..., None])
            rgb = rgb * (1 - a[..., None]) + c * a[..., None]
        return rgb
