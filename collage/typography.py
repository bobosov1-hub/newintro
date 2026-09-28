"""Title lockups: a thin descriptor line against a heavy editorial serif title.

The lockup is rendered as ONE sprite and animated as one unit. Chaos adds small per-word
baseline and rotation offsets - never enough to break the reading line.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import FONTS
from . import palette as P
from .noise import blur, smoothstep, value_noise
from .sprite import Sprite

FAMILIES = {
    "hahmlet": "Hahmlet-VF.ttf",            # variable 100..900, editorial Korean serif
    "myeongjo": "NanumMyeongjo-Regular.ttf",
    "myeongjo-xb": "NanumMyeongjo-ExtraBold.ttf",
    "oldstd": "OldStandard-Regular.ttf",
    "oldstd-b": "OldStandard-Bold.ttf",
}
_cache = {}


def font(size, weight=900, family="hahmlet"):
    key = (family, int(round(size)), int(weight))
    if key not in _cache:
        f = ImageFont.truetype(os.path.join(FONTS, FAMILIES[family]), int(round(size)))
        if family == "hahmlet":
            f.set_variation_by_axes([int(weight)])
        _cache[key] = f
    return _cache[key]


def text_mask(text, fnt, tracking=0.0):
    """Render text to a float mask. Returns (mask, baseline_y, advance_width)."""
    asc, desc = fnt.getmetrics()
    size = fnt.size
    widths = [fnt.getlength(ch) for ch in text]
    track = tracking * size
    total = sum(widths) + track * max(0, len(text) - 1)
    pad = int(size * 0.25) + 2
    W = int(np.ceil(total)) + 2 * pad
    H = asc + desc + 2 * pad
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    if tracking == 0.0:
        d.text((pad, pad), text, font=fnt, fill=255)
    else:
        x = float(pad)
        for ch, wch in zip(text, widths):
            d.text((x, pad), ch, font=fnt, fill=255)
            x += wch + track
    m = np.asarray(img, np.float32) / 255.0
    return m, pad + asc, total, pad


def _rotate(mask, deg):
    if abs(deg) < 1e-3:
        return mask
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.rotate(deg, resample=Image.BICUBIC, expand=False)
    return np.asarray(im, np.float32) / 255.0


def _paste_max(canvas, m, x, y):
    h, w = m.shape
    H, W = canvas.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], m[y0 - y:y1 - y, x0 - x:x1 - x])


def ink_texture(mask, rng, amount=1.0):
    """Printed-ink feel: density variation and a very slightly irregular edge."""
    h, w = mask.shape
    n = value_noise(h, w, rng, cell=5.0)
    soft = blur(mask, 0.6)
    edge = smoothstep(0.5 - 0.22, 0.5 + 0.22, soft + 0.06 * amount * n)
    dens = 0.93 + 0.07 * np.clip(value_noise(h, w, rng, cell=40.0) * 0.5 + 0.5, 0, 1)
    return (np.maximum(edge, mask * 0.0) * dens).astype(np.float32)


def render_words(line, fnt, build, rng, jit_scale=1.0, tracking=0.0, space_em=0.26):
    """Lay out one line word by word with chaos-driven baseline/rotation jitter."""
    words = line.split(" ")
    pieces = []
    size = fnt.size
    for wd in words:
        m, base, adv, pad = text_mask(wd, fnt, tracking)
        dy = build.jit(rng, 0.004, 0.07) * size * jit_scale
        rot = build.jit(rng, 0.15, 3.4) * jit_scale
        pieces.append((_rotate(m, rot), base, adv, pad, dy))
    total = sum(p[2] for p in pieces) + space_em * size * (len(pieces) - 1)
    return pieces, total


def lockup(descriptor, title, build, rng, title_size=150, desc_size=None, color=None, desc_color=None,
           align="center", res=1.5, rule=None, title_weight=900, desc_weight=220, leading=1.06):
    """Returns (Sprite, metrics). Metrics are in sprite pixels (divide by res for world units)."""
    color = P.INK if color is None else color
    desc_color = color if desc_color is None else desc_color
    tfont = font(title_size * res, title_weight)
    dsize = desc_size or max(30, title_size * 0.29)
    dfont = font(dsize * res, desc_weight)
    lines = title.split("\n")
    laid = [render_words(ln, tfont, build, rng, 1.0, tracking=-0.015) for ln in lines]
    desc = render_words(descriptor, dfont, build, rng, 0.5, tracking=0.10, space_em=0.45) if descriptor else None
    use_rule = (rng.random() < 0.5) if rule is None else rule

    tsz = tfont.size
    widths = [w for _, w in laid] + ([desc[1]] if desc else [])
    W = int(max(widths) + tsz * 1.2)
    line_h = tsz * leading
    top_pad = int(tsz * 0.35)
    desc_h = int(dfont.size * 1.9) if desc else 0
    rule_h = int(dfont.size * 0.55) if (desc and use_rule) else 0
    H = int(top_pad + desc_h + rule_h + line_h * len(lines) + tsz * 0.55)
    title_m = np.zeros((H, W), np.float32)
    desc_m = np.zeros((H, W), np.float32)

    def x_start(total):
        if align == "left":
            return int(tsz * 0.6)
        if align == "right":
            return int(W - tsz * 0.6 - total)
        return int((W - total) / 2)

    y = top_pad
    metrics = {}
    if desc:
        pieces, total = desc
        x = x_start(total)
        base = y + int(dfont.size * 1.2)
        for m, b, adv, pad, dy in pieces:
            _paste_max(desc_m, m, int(x - pad), int(base - b + dy))
            x += adv + 0.45 * dfont.size
        metrics["desc_box"] = (x_start(total), y, x_start(total) + total, base)
        y += desc_h
        if use_rule:
            rw = int(min(total * 1.15, W * 0.9))
            rx = x_start(rw)
            th = max(1, int(round(1.3 * res)))
            desc_m[y + rule_h // 3: y + rule_h // 3 + th, rx:rx + rw] = 1.0
            y += rule_h
    title_top = None
    for i, (pieces, total) in enumerate(laid):
        x = x_start(total)
        base = int(y + line_h * i + tsz * 0.86)
        for m, b, adv, pad, dy in pieces:
            _paste_max(title_m, m, int(x - pad), int(base - b + dy))
            x += adv + 0.26 * tsz
        if title_top is None:
            title_top = base - int(tsz * 0.78)
    title_bottom = int(y + line_h * (len(lines) - 1) + tsz * 0.86)
    metrics.update({"title_top": title_top, "title_baseline": title_bottom, "cap": tsz * 0.74,
                    "width": W, "height": H, "res": res})
    ta = ink_texture(title_m, rng)
    da = ink_texture(desc_m, rng, 0.6)
    a = np.maximum(ta, da)
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = color
    if desc_color is not color:
        w_desc = np.where(a > 0, da / np.maximum(a, 1e-4), 0.0)[..., None]
        rgb = color * (1 - w_desc) + desc_color * w_desc
    acc = ta if (color is P.ACCENT) else None
    return Sprite(rgb, a, acc, res=res, meta=metrics), metrics


def label(text, size, weight=700, color=None, res=1.0, tracking=0.0, family="hahmlet"):
    """Single-line type sprite (stickers, captions, diagram labels)."""
    color = P.INK if color is None else color
    fnt = font(size * res, weight, family)
    m, base, adv, pad = text_mask(text, fnt, tracking)
    rgb = np.empty(m.shape + (3,), np.float32)
    rgb[:] = color
    return Sprite(rgb, m, None, res=res, meta={"baseline": base, "advance": adv, "pad": pad})
