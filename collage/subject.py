"""Turn a prepared subject into printed collage pieces.

* supporting elements: black and white only, never the accent
* hero: black and white + exactly one accent method, chosen by the seed:
    duotone   - shadows -> ink, highlights -> red-orange
    backing   - a flat red-orange paper shape directly behind the hero
    selective - recolour only the single part that defines it (needs NAME.part.png)
* people: sliced into 3-6 horizontal bands, each with its own offset/rotation/timing,
  all sharing one edge treatment
* light subjects get a mid-grey card so they don't dissolve into the page
"""
import json
import os

import cv2
import numpy as np

from . import ROOT
from . import edges as E
from . import palette as P
from .noise import blur, smoothstep, value_noise
from .paper import paper_luma
from .print_fx import print_luma
from .sprite import Sprite, crop_to_content

ACCENT_METHODS = ("duotone", "backing", "selective")


class Asset:
    _cache = {}

    def __init__(self, name, folder=None):
        folder = folder or os.path.join(ROOT, "assets")
        base = os.path.join(folder, name)
        if not os.path.exists(base + ".png"):
            raise FileNotFoundError(f"asset '{name}' not prepared - run prep_subject.py (looked in {folder})")
        rgba = cv2.imread(base + ".png", cv2.IMREAD_UNCHANGED)
        self.name = name
        self.L = rgba[..., 0].astype(np.float32) / 255.0
        self.a = rgba[..., 3].astype(np.float32) / 255.0
        full = cv2.imread(base + ".full.png", cv2.IMREAD_GRAYSCALE)
        self.full = None if full is None else full.astype(np.float32) / 255.0
        part = cv2.imread(base + ".part.png", cv2.IMREAD_GRAYSCALE) if os.path.exists(base + ".part.png") else None
        self.part = None if part is None else part.astype(np.float32) / 255.0
        with open(base + ".json", encoding="utf-8") as fh:
            self.meta = json.load(fh)

    @classmethod
    def get(cls, name, folder=None):
        key = (name, folder)
        if key not in cls._cache:
            cls._cache[key] = Asset(name, folder)
        return cls._cache[key]

    @property
    def is_photo(self):
        return self.meta.get("method") in ("isnet", "u2net", "key", "none")


class Piece:
    """One sprite of an element, with its rest offset (world units) from the element centre."""

    def __init__(self, sprite, offset, rot=0.0, role="body", index=0):
        self.sprite = sprite
        self.offset = offset
        self.rot = rot
        self.role = role          # body | band | backing | card
        self.index = index


class Element:
    def __init__(self, name, pieces, meta):
        self.name = name
        self.pieces = pieces
        self.meta = meta          # world_size, top_xs/top_ys (world, relative to centre), edge, accent


def choose_accent(build, rng, asset):
    avail = ["duotone", "backing"] + (["selective"] if asset.part is not None and asset.part.max() > 0.2 else [])
    return build.pick(rng, avail, [1.0] * len(avail))


def _resize(img, size):
    if img is None:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if size[0] < img.shape[1] else cv2.INTER_CUBIC)


def _studio_backdrop(h, w, rng, light):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w * 0.5) / w) ** 2 + ((yy - h * 0.42) / h) ** 2)
    base = (0.46 if light else 0.72) + 0.10 * (0.6 - r) - 0.06 * (yy / h)
    return np.clip(base + 0.02 * value_noise(h, w, rng, max(h, w) / 5), 0, 1).astype(np.float32)


def build_element(asset, build, key, role="support", height=500.0, res=1.25, accent=None, edge=None,
                  crop=None, bands=0, pitch=4.2, xerox=None, card=None):
    """Returns an Element. `height` is the world height of the (cropped) subject."""
    rng = build.rng("element", key)
    c = build.chaos
    x0, y0, x1, y1 = asset.meta["bbox"]
    if asset.meta.get("print_only"):
        x0, y0, x1, y1 = 0, 0, asset.a.shape[1], asset.a.shape[0]
    bw_, bh_ = x1 - x0, y1 - y0
    # crop: fractions of the bbox; chaos nudges the crop edges
    cx0, cy0, cx1, cy1 = crop if crop is not None else (0.0, 0.0, 1.0, 1.0)
    jit = [build.jit(rng, 0.0, 0.05) for _ in range(4)]
    if crop is not None:
        cx0, cy0 = np.clip(cx0 + jit[0] * (cx0 > 0), 0, 1), np.clip(cy0 + jit[1] * (cy0 > 0), 0, 1)
        cx1, cy1 = np.clip(cx1 + jit[2] * (cx1 < 1), 0, 1), np.clip(cy1 + jit[3] * (cy1 < 1), 0, 1)
    sx0, sy0 = int(x0 + cx0 * bw_), int(y0 + cy0 * bh_)
    sx1, sy1 = int(x0 + cx1 * bw_), int(y0 + cy1 * bh_)
    L = asset.L[sy0:sy1, sx0:sx1]
    a = asset.a[sy0:sy1, sx0:sx1]
    full = None if asset.full is None else asset.full[sy0:sy1, sx0:sx1]
    part = None if asset.part is None else asset.part[sy0:sy1, sx0:sx1]

    hero = role == "hero"
    kind = edge or (E.choose(build, rng, hero=hero, photo=asset.is_photo) if not asset.meta.get("print_only") else "print")
    if asset.meta.get("print_only"):
        kind = "print"
    method = (accent or choose_accent(build, rng, asset)) if hero else None
    if method == "selective" and part is None:
        method = "duotone"

    # scale to the requested world height at `res` sprite px per world unit
    s = height * res / float(L.shape[0])
    size = (max(2, int(L.shape[1] * s)), max(2, int(L.shape[0] * s)))
    L, a = _resize(L, size), _resize(a, size)
    full, part = _resize(full, size), _resize(part, size)
    h, w = a.shape
    a = np.clip(a, 0, 1)

    if kind == "print":
        # rectangular print: the subject on its own photographic backdrop, no cutout
        pad = int((0.05 + 0.03 * float(rng.random())) * max(h, w))
        H2, W2 = h + 2 * pad, w + 2 * pad
        if asset.is_photo and full is not None:
            ex0, ey0 = max(0, sx0 - int(pad / s)), max(0, sy0 - int(pad / s))
            ex1 = min(asset.full.shape[1], sx1 + int(pad / s))
            ey1 = min(asset.full.shape[0], sy1 + int(pad / s))
            bg = cv2.resize(asset.full[ey0:ey1, ex0:ex1], (W2, H2), interpolation=cv2.INTER_AREA)
            Lp = bg.copy()
            ox, oy = int((sx0 - ex0) * s), int((sy0 - ey0) * s)
            Lp[oy:oy + h, ox:ox + w] = L * a + Lp[oy:oy + h, ox:ox + w] * (1 - a)
        else:
            Lp = _studio_backdrop(H2, W2, rng, asset.meta.get("mean_luma", 0.5) > 0.55)
            ox, oy = pad, pad
            Lp[oy:oy + h, ox:ox + w] = L * a + Lp[oy:oy + h, ox:ox + w] * (1 - a)
        ap = np.ones((H2, W2), np.float32)
        pp = None
        if part is not None:
            pp = np.zeros((H2, W2), np.float32)
            pp[oy:oy + h, ox:ox + w] = part
        L, a, part = Lp, ap, pp
        subj_off = (ox, oy, w, h)
    else:
        subj_off = (0, 0, w, h)

    # print process (object space). The accent area is kept clean.
    protect = None
    ht = 0.82
    if method == "duotone":
        protect = smoothstep(0.35, 0.8, L)
        ht = 0.60
    elif method == "selective":
        protect = part
    xa = build.amt(0.28, 0.78) * (0.7 if hero else 1.0) if xerox is None else xerox
    angle = float(rng.choice([15.0, 45.0, 75.0])) + build.jit(rng, 0.0, 8.0)
    Lp = print_luma(L, rng, pitch=pitch * res, angle=angle, ht_strength=ht, xerox_amount=xa, protect=protect)

    acc = None
    if method == "duotone":
        rgb = P.duotone(np.clip(Lp * 1.04, 0, 1))
        acc = np.clip(Lp, 0, 1)
    elif method == "selective":
        # keep the part's own shading but at full accent strength, whatever its original tone
        inside = part > 0.5
        lo, hi = (np.percentile(Lp[inside], [4, 96]) if inside.any() else (0.0, 1.0))
        Ln = np.clip((Lp - lo) / max(hi - lo, 0.05), 0, 1)
        Ls = 0.50 + 0.50 * Ln
        rgb = P.tone(Lp) * (1 - part[..., None]) + (P.INK + (P.ACCENT - P.INK) * Ls[..., None]) * part[..., None]
        acc = part * Ls
    else:
        rgb = P.tone(Lp)

    # geometry helpers (world units, relative to the element centre = centre of the subject box)
    ox, oy, sw, sh = subj_off
    ecx, ecy = ox + sw / 2.0, oy + sh / 2.0
    pieces = []
    bw = max(3.0, (0.010 + 0.004 * float(rng.random())) * max(h, w))

    person = bool(asset.meta.get("person")) and bands != 0
    if person:
        nb = int(bands) if bands > 0 else int(np.clip(round(3 + 3 * (sh / max(sw, 1)) / 1.6 + float(rng.random())), 3, 6))
        cuts = _band_cuts(build, rng, nb, oy, oy + sh, a.shape[1])
        for i in range(nb):
            ya = cuts[i]
            yb = cuts[i + 1]
            yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
            top_line = ya[0] + (ya[1] - ya[0]) * xx / a.shape[1]
            bot_line = yb[0] + (yb[1] - yb[0]) * xx / a.shape[1]
            m = np.clip(yy - top_line + 0.5, 0, 1) * np.clip(bot_line - yy + 0.5, 0, 1)
            if i == 0:
                m = np.clip(bot_line - yy + 0.5, 0, 1)
            if i == nb - 1:
                m = np.clip(yy - top_line + 0.5, 0, 1)
            pa = a * m
            if pa.max() < 0.05:
                continue
            prgb, pa2, pacc, (px, py) = _crop_piece(rgb, pa, None if acc is None else acc * m)
            if kind != "print":
                prgb, pa2, pacc = E.apply(kind, prgb, pa2, pacc, build.rng("edge", key, i), bw, c)
                padm = _pad_amount(kind, bw)
                px, py = px - padm, py - padm
            sp = Sprite(prgb, pa2, pacc, res=res)
            off = ((px + sp.w / 2.0 - ecx) / res, (py + sp.h / 2.0 - ecy) / res)
            pieces.append(Piece(sp, off, role="band", index=i))
    else:
        prgb, pa2, pacc, (px, py) = _crop_piece(rgb, a, acc)
        prgb, pa2, pacc = E.apply(kind, prgb, pa2, pacc, build.rng("edge", key), bw, c)
        padm = _pad_amount(kind, bw)
        px, py = px - padm, py - padm
        sp = Sprite(prgb, pa2, pacc, res=res)
        pieces.append(Piece(sp, ((px + sp.w / 2.0 - ecx) / res, (py + sp.h / 2.0 - ecy) / res), role="body"))

    # grey card behind light subjects (not needed when the accent already separates it)
    light = asset.meta.get("reads_light", False) and kind != "print"
    need_card = card if card is not None else (light and method not in ("duotone", "backing"))
    if need_card:
        pieces.insert(0, _card(build, key, a, subj_off, res, ecx, ecy, color=P.GREY, rng=rng))
    if method == "backing":
        pieces.insert(0, _backing(build, key, a, subj_off, res, ecx, ecy, kind, rng))

    top_x, top_y = top_contour([p for p in pieces if p.role in ("body", "band")])
    meta = {"src_origin": (sx0, sy0), "src_scale": s, "sub_px": (sw, sh), "res": res,
            "world_w": a.shape[1] / res, "world_h": a.shape[0] / res, "subject_w": sw / res, "subject_h": sh / res,
            "edge": kind, "accent": method, "top_x": top_x, "top_y": top_y, "person": person,
            "bands": sum(1 for p in pieces if p.role == "band")}
    return Element(asset.name, pieces, meta)


def _pad_amount(kind, bw):
    return int({"sticker": bw * 2 + 6, "hand": bw * 3 + 8, "torn": bw * 4 + 10,
                "print": int(round(bw * 1.3)) + 4}[kind])


def _crop_piece(rgb, a, acc):
    ys, xs = np.where(a > 0.003)
    y0, y1 = max(0, ys.min() - 2), min(a.shape[0], ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(a.shape[1], xs.max() + 3)
    return (rgb[y0:y1, x0:x1], a[y0:y1, x0:x1], None if acc is None else acc[y0:y1, x0:x1], (x0, y0))


def _band_cuts(build, rng, n, top, bottom, width):
    """n bands -> n+1 cut lines, each (y_left, y_right); slightly tilted, spacing jittered."""
    span = bottom - top
    cuts = [(top - 5.0, top - 5.0)]
    # the first band holds the head: make it a bit taller than the rest
    weights = np.array([1.35] + [1.0] * (n - 1))
    edges = top + span * np.cumsum(weights) / weights.sum()
    for i in range(n - 1):
        y = edges[i] + build.jit(rng, 0.0, 0.08) * span / n
        tilt = np.tan(np.deg2rad(build.jit(rng, 0.3, 2.4))) * width
        cuts.append((y - tilt / 2, y + tilt / 2))
    cuts.append((bottom + 5.0, bottom + 5.0))
    return cuts


def top_contour(pieces, n=96):
    """Visible top edge of the union of pieces: (xs, ys) in world units from the element centre."""
    boxes = []
    for p in pieces:
        sp = p.sprite
        ww, hh = sp.world_size()
        boxes.append((p.offset[0] - ww / 2, p.offset[1] - hh / 2, ww, hh, sp))
    xmin = min(b[0] for b in boxes)
    xmax = max(b[0] + b[2] for b in boxes)
    xs = np.linspace(xmin, xmax, n).astype(np.float32)
    ys = np.full(n, np.inf, np.float32)
    for (bx, by, ww, hh, sp) in boxes:
        for i, x in enumerate(xs):
            if bx <= x < bx + ww:
                col = np.where(sp.a[:, min(sp.w - 1, int((x - bx) * sp.res))] > 0.5)[0]
                if len(col):
                    ys[i] = min(ys[i], by + col[0] / sp.res)
    return xs, ys


def _card(build, key, a, subj_off, res, ecx, ecy, color, rng):
    ox, oy, sw, sh = subj_off
    grow = 1.10 + 0.08 * float(rng.random())
    w, h = int(sw * grow), int(sh * grow)
    tex = paper_luma(h, w, build.rng("card", key), 0.8)
    rgb = color * tex[..., None]
    alpha = np.ones((h, w), np.float32)
    rgb, alpha, _ = E.sticker(rgb.astype(np.float32), alpha, None, build.rng("card-edge", key), 0.0 + 1.0)
    sp = Sprite(rgb, alpha, None, res=res)
    dx = build.jit(rng, 0.02, 0.06) * sw / res
    dy = build.jit(rng, 0.01, 0.04) * sh / res
    rot = build.jit(rng, 0.5, 5.0)
    return Piece(sp, (dx, dy + 0.02 * sh / res), rot=rot, role="card")


def _backing(build, key, a, subj_off, res, ecx, ecy, kind, rng):
    """A flat red-orange paper shape set directly behind the hero."""
    ox, oy, sw, sh = subj_off
    shape = build.pick(rng, ["rect", "circle", "silhouette", "arch"], [0.3, 0.25, 0.25, 0.2])
    brng = build.rng("backing", key)
    if shape == "silhouette" and kind != "print":
        pad = int(0.08 * max(sw, sh))
        m = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        d = cv2.distanceTransform((m < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        grow = 0.045 * max(sw, sh)
        alpha = np.clip(grow - blur(d, 2.0) + 0.5, 0, 1)
        off0 = (-pad, -pad)
        w, h = alpha.shape[1], alpha.shape[0]
        base_cx, base_cy = off0[0] + w / 2.0, off0[1] + h / 2.0
    else:
        scale = 1.08 + 0.18 * float(rng.random())
        if shape == "circle":
            dia = int(max(sw, sh) * 0.92 * scale)
            w = h = dia
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            alpha = np.clip(dia / 2.0 - np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) + 0.5, 0, 1)
        else:
            w, h = int(sw * scale * 0.92), int(sh * min(1.0, scale * 0.86))
            alpha = np.ones((h, w), np.float32)
            if shape == "arch":
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                r = w / 2.0
                arch = np.clip(r - np.sqrt((xx - r) ** 2 + (yy - r) ** 2) + 0.5, 0, 1)
                alpha = np.where(yy < r, arch, 1.0).astype(np.float32)
        base_cx, base_cy = ox + sw / 2.0, oy + sh * 0.46
    tex = paper_luma(alpha.shape[0], alpha.shape[1], brng, 0.7)
    rgb = (P.ACCENT * (0.985 + 0.015 * (tex - 1.0) * 40)[..., None]).astype(np.float32)
    rgb = np.clip(rgb, 0, 1)
    sp = Sprite(rgb, alpha.astype(np.float32), alpha.astype(np.float32).copy(), res=res)
    dx = (base_cx - ecx) / res + build.jit(rng, 0.02, 0.07) * sw / res
    dy = (base_cy - ecy) / res + build.jit(rng, 0.01, 0.05) * sh / res - (0.03 * sh / res if shape != "silhouette" else 0)
    rot = build.jit(rng, 0.8, 7.0)
    return Piece(sp, (dx, dy), rot=rot, role="backing")
