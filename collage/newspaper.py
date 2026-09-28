"""Background scatter: torn newspaper fragments at low contrast.

A pool of fake broadsheet pages is generated per build (random syllables - nothing to read),
and fragments are cut from it and placed by the seed, so no two builds repeat.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import palette as P
from .edges import torn_mask
from .noise import blur, fbm
from .paper import paper_luma
from .print_fx import halftone
from .procedural.docs import SYL, fake_text
from .sprite import Sprite
from .typography import font


def _vertical_column(d, x, y0, y1, size, rng, fnt):
    y = y0
    while y < y1 - size:
        d.text((x, y), SYL[int(rng.integers(0, len(SYL)))], font=fnt, fill=0)
        y += int(size * 1.08)


def page(rng, w=1500, h=2100):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    body = font(17, 400, "myeongjo")
    ncol = int(rng.integers(5, 8))
    m, gut = 44, 16
    cw = (w - 2 * m - gut * (ncol - 1)) / ncol
    photos = []
    c = 0
    while c < ncol:
        span = int(min(ncol - c, rng.choice([1, 1, 2, 2, 3])))
        x0 = m + c * (cw + gut)
        x1 = x0 + span * cw + (span - 1) * gut
        y = m + int(rng.integers(0, 30))
        while y < h - m - 60:
            kind = rng.choice(["head", "body", "body", "body", "photo", "rule", "vert"], p=[.16, .22, .22, .14, .12, .08, .06])
            if kind == "head":
                size = int(rng.integers(30, 64))
                hf = font(size, int(rng.choice([700, 800, 900])))
                n = max(2, int((x1 - x0) / (size * 1.02)))
                for _ in range(int(rng.integers(1, 3))):
                    d.text((x0, y), fake_text(rng, n, (2, 5)), font=hf, fill=0)
                    y += int(size * 1.18)
                y += 10
            elif kind == "body":
                n = max(4, int((x1 - x0) / 17.6))
                for _ in range(int(rng.integers(6, 22))):
                    d.text((x0, y), fake_text(rng, n), font=body, fill=0)
                    y += 25
                y += 12
            elif kind == "photo":
                ph = int(rng.uniform(0.5, 1.0) * (x1 - x0))
                photos.append((int(x0), y, int(x1), y + ph))
                y += ph + 30
            elif kind == "rule":
                d.line([(x0, y), (x1, y)], fill=0, width=int(rng.integers(1, 4)))
                y += 14
            else:
                vf = font(19, 500, "hahmlet")
                yy1 = min(h - m, y + int(rng.integers(260, 600)))
                x = x1 - 24
                while x > x0:
                    _vertical_column(d, x, y, yy1, 19, rng, vf)
                    x -= 26
                y = yy1 + 16
        c += span
    L = np.asarray(img, np.float32) / 255.0
    for (x0, y0, x1, y1) in photos:
        y1 = min(y1, h - m)
        if y1 - y0 < 20:
            continue
        p = np.clip(0.5 + 0.28 * fbm(y1 - y0, x1 - x0, rng, float(rng.uniform(25, 90)), 4), 0, 1)
        L[y0:y1, x0:x1] = halftone(p, pitch=5.0, angle=45, strength=0.95, gate=(0.0, 1.01))
    return L * paper_luma(h, w, rng, 1.2)


class Pool:
    def __init__(self, build, n_pages=4):
        self.build = build
        self.pages = [page(build.rng("news-page", i)) for i in range(n_pages)]

    def fragment(self, key, scale=1.0, contrast=None):
        """Torn fragment Sprite (straight RGB in palette values, low contrast)."""
        b = self.build
        rng = b.rng("news-frag", key)
        pg = self.pages[int(rng.integers(0, len(self.pages)))]
        fw = int(rng.uniform(260, 720) * scale)
        fh = int(rng.uniform(200, 560) * scale)
        fw, fh = min(fw, pg.shape[1] - 10), min(fh, pg.shape[0] - 10)
        x = int(rng.integers(0, pg.shape[1] - fw))
        y = int(rng.integers(0, pg.shape[0] - fh))
        L = pg[y:y + fh, x:x + fw]
        L = blur(L, 0.7)  # keep it illegible: ambience, not content
        con = contrast if contrast is not None else b.amt(0.30, 0.46) * float(rng.uniform(0.8, 1.15))
        stock = P.mix(P.PAPER, P.GREY, float(rng.uniform(0.05, 0.16)))
        ink = P.mix(P.GREY, P.INK, 0.35)
        rgb = stock + (ink - stock) * ((1.0 - L) * con)[..., None]
        rgb *= paper_luma(fh, fw, rng, 1.0)[..., None]
        a = torn_mask(fh, fw, rng, rough=b.amt(0.8, 1.6), inset=float(rng.uniform(5, 10)))
        return Sprite(np.clip(rgb, 0, 1), a, None, res=1.0)
