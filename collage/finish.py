"""Frame finishing: print grain, xerox unevenness, restrained edge chromatic aberration,
a soft focusing vignette, and an occasional discoloured light leak (transitions only).
The accent area is protected from the heavy degradation."""
import cv2
import numpy as np

from . import palette as P
from .noise import blur, fbm, smoothstep


class Finish:
    def __init__(self, W, H, build, n_grain=6):
        rng = build.rng("finish")
        self.W, self.H = W, H
        self.grain = [blur(rng.standard_normal((H, W)).astype(np.float32), 0.55).astype(np.float16)
                      for _ in range(n_grain)]
        blot = fbm(H // 4, W // 4, rng, 70.0, 3)
        self.blotch = cv2.resize(blot, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float16)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 * 0.85 + ((yy - H / 2) / (H / 2)) ** 2 * 0.75)
        self.vignette = (1.0 - build.amt(0.16, 0.26) * smoothstep(0.42, 1.30, r)).astype(np.float32)
        self.grain_amp = build.amt(0.040, 0.085)
        self.blotch_amt = build.amt(0.012, 0.035)
        self.ca = build.amt(0.00065, 0.0015)
        self.leaks = [self._leak(build.rng("leak", i)) for i in range(4)]

    def _leak(self, rng):
        h, w = self.H // 4, self.W // 4
        img = np.zeros((h, w, 3), np.float32)
        warm = P.mix(P.PAPER, P.ACCENT, 0.45)
        for _ in range(int(rng.integers(2, 4))):
            side = rng.integers(0, 4)
            cx = rng.uniform(-0.1, 0.35) * w if side == 0 else rng.uniform(0.65, 1.1) * w if side == 1 else rng.uniform(0, w)
            cy = rng.uniform(0, h) if side < 2 else (rng.uniform(-0.1, 0.3) * h if side == 2 else rng.uniform(0.7, 1.1) * h)
            rx, ry = rng.uniform(0.18, 0.45) * w, rng.uniform(0.25, 0.6) * h
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
            col = P.ACCENT if rng.random() < 0.55 else warm
            img += g[..., None] * col * rng.uniform(0.6, 1.0)
        return np.clip(img, 0, 1)

    def apply(self, rgb, acc, frame, leak=0.0, leak_id=0, leak_shift=0.0, grain_step=1):
        # grain boils on the stop-motion step, not every frame (kinder to the encoder too)
        g = self.grain[(frame // max(1, grain_step)) % len(self.grain)].astype(np.float32)
        protect = 1.0 - 0.85 * np.clip(acc, 0, 1)
        amp = (self.grain_amp * protect)[..., None]
        # overlay-style grain, strongest in the mid tones
        rgb = rgb + g[..., None] * amp * (rgb * (1.0 - rgb)) * 4.0
        rgb = rgb * (1.0 - (self.blotch_amt * self.blotch.astype(np.float32) * protect)[..., None])
        # restrained chromatic aberration growing towards the frame edges
        if self.ca > 0:
            out = np.empty_like(rgb)
            for ch, k in ((0, 1.0 + self.ca), (1, 1.0), (2, 1.0 - self.ca)):
                if k == 1.0:
                    out[..., ch] = rgb[..., ch]
                    continue
                M = np.float32([[k, 0, (1 - k) * self.W / 2], [0, k, (1 - k) * self.H / 2]])
                out[..., ch] = cv2.warpAffine(rgb[..., ch], M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE)
            rgb = out
        rgb = rgb * self.vignette[..., None]
        if leak > 0.0:
            lk = self.leaks[leak_id % len(self.leaks)]
            dx = int(leak_shift * lk.shape[1] * 0.25)
            lk = np.roll(lk, dx, axis=1)
            lk = cv2.resize(lk, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            rgb = 1.0 - (1.0 - rgb) * (1.0 - lk * leak)
        # the palette never reaches pure white (or pure black): clamp to the paper / ink stock
        return np.clip(rgb, P.INK * 0.55, np.minimum(P.PAPER + 0.035, 0.985))
