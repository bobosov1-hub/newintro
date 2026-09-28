"""Seed + chaos.

One chaos value (0..1) scales every variable at once. Random draws never depend on chaos,
only their amplitude does, so the same seed at a different chaos gives the same composition,
just more (or less) out of register. Same seed + same chaos => identical frame.
"""
import hashlib
import os

import numpy as np


def new_seed():
    return int.from_bytes(os.urandom(4), "little") % 100_000_000


class Build:
    def __init__(self, seed=None, chaos=0.4):
        self.seed = new_seed() if seed is None else int(seed)
        self.chaos = float(np.clip(chaos, 0.0, 1.0))

    def rng(self, *keys):
        h = hashlib.blake2b(repr((self.seed,) + tuple(keys)).encode("utf-8"), digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(h, "little"))

    # amplitude helpers -------------------------------------------------------------------
    def amt(self, at0, at1):
        """Linear amplitude between the clean value (chaos 0) and the wild value (chaos 1)."""
        return at0 + (at1 - at0) * self.chaos

    def jit(self, rng, at0, at1):
        """Symmetric jitter in [-amt, amt]; always consumes exactly one draw."""
        return float(rng.uniform(-1.0, 1.0)) * self.amt(at0, at1)

    def pick(self, rng, options, weights):
        """Weighted choice; always consumes exactly one draw."""
        w = np.asarray(weights, np.float64)
        w = w / w.sum()
        u = float(rng.random())
        return options[int(np.searchsorted(np.cumsum(w), u, side="right").clip(0, len(options) - 1))]

    def tag(self):
        return f"seed{self.seed}_chaos{self.chaos:.2f}"
