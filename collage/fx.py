"""Small living-picture effects drawn over a scene in screen space: rising steam, floating dust.

Deterministic per (seed, scene): every frame is a pure function of time.
"""
import math

import cv2
import numpy as np


class Steam:
    """Soft wisps rising from an emitter line (screen px at rest, origin = frame centre)."""

    def __init__(self, spec, W, H, rng, vs=1.0):
        self.W, self.H, self.vs = W, H, vs
        self.x, self.y = spec.get("at", [0, 0])
        self.w = float(spec.get("w", 260))
        self.rise = float(spec.get("rise", 520))
        self.amount = float(spec.get("amount", 0.22))
        n = int(spec.get("n", 26))
        self.puffs = [(float(rng.uniform(-0.5, 0.5)), float(rng.uniform(0, 1)), float(rng.uniform(0.7, 1.3)),
                       float(rng.uniform(0, 2 * math.pi))) for _ in range(n)]
        self.life = float(spec.get("life", 2.6))
        self.f = 4                                                   # render at 1/4 res

    def draw(self, rgb, t):
        f = self.f
        h, w = self.H // f, self.W // f
        m = np.zeros((h, w), np.float32)
        cx0 = (self.W / 2 + self.x * self.vs) / f
        cy0 = (self.H / 2 + self.y * self.vs) / f
        for (u, ph, sp, wob) in self.puffs:
            k = ((t / self.life) * sp + ph) % 1.0
            px = cx0 + u * self.w * self.vs / f + math.sin(wob + t * 1.3 + k * 4) * 18 * k * self.vs / f
            py = cy0 - k * self.rise * self.vs / f
            r = (10 + 60 * k) * self.vs / f
            a = math.sin(math.pi * k) ** 1.5
            cv2.circle(m, (int(px), int(py)), max(1, int(r)), a, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 6.0 * self.vs * 4 / f)
        m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        k = np.clip(m * self.amount, 0, 0.6)[..., None]
        return rgb + (np.array([0.96, 0.94, 0.90], np.float32) - rgb) * k


class Dust:
    """Specks drifting slowly through a light beam."""

    def __init__(self, spec, W, H, rng, vs=1.0):
        self.W, self.H, self.vs = W, H, vs
        n = int(spec.get("n", 70))
        self.p = rng.uniform(0, 1, (n, 2)).astype(np.float32)
        self.v = rng.normal(0, 1, (n, 2)).astype(np.float32) * np.array([0.010, 0.006], np.float32) + \
            np.array([0.004, -0.008], np.float32)
        self.s = rng.uniform(1.0, 2.8, n).astype(np.float32)
        self.ph = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.amount = float(spec.get("amount", 0.5))

    def draw(self, rgb, t):
        m = np.zeros((self.H, self.W), np.float32)
        pos = (self.p + self.v * t) % 1.0
        for (x, y), s, ph in zip(pos, self.s, self.ph):
            a = 0.5 + 0.5 * math.sin(t * 1.7 + ph)
            cv2.circle(m, (int(x * self.W), int(y * self.H)), max(1, int(s * self.vs)), a, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 1.2 * self.vs)
        k = np.clip(m * self.amount, 0, 0.8)[..., None]
        return rgb + (np.array([0.98, 0.95, 0.88], np.float32) - rgb) * k


KINDS = {"steam": Steam, "dust": Dust}


def build(specs, W, H, rng, vs=1.0):
    return [KINDS[s["type"]](s, W, H, rng, vs) for s in specs]
