"""Print processes baked into each printed element (object space, so the dots move with the paper).

* halftone: an AM dot screen gated to the shadow/midtone range and combined by MULTIPLY with a
  continuous highlight layer - highlights stay clean, shadows read as print.
* xerox: toner blotching, specks, drop-outs and drum streaks, combined by OVERLAY/MULTIPLY.
"""
import numpy as np

from .noise import blur, fbm, smoothstep, value_noise


def overlay(a, b):
    return np.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))


def spot_function(h, w, pitch, angle_deg, phase=(0.0, 0.0)):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle_deg)
    u = (xx * np.cos(a) + yy * np.sin(a)) / pitch + phase[0]
    v = (-xx * np.sin(a) + yy * np.cos(a)) / pitch + phase[1]
    return (0.5 + 0.25 * (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v))).astype(np.float32)


def halftone(L, pitch=4.6, angle=45.0, strength=0.85, gate=(0.30, 0.82), phase=(0.0, 0.0)):
    """Gated halftone. L in 0..1 (1 = paper). Returns luminance of the same shape."""
    h, w = L.shape
    D = 1.0 - L
    g = smoothstep(gate[1], gate[0], L)              # 1 in shadows/mids, 0 in highlights
    L_hi = 1.0 - D * (1.0 - g)                        # continuous highlight component
    L_sh = 1.0 - D * g                                # shadow component, to be screened
    spot = spot_function(h, w, pitch, angle, phase)
    wid = 1.8 / pitch
    ink = np.clip((spot - L_sh) / wid + 0.5, 0.0, 1.0)
    screened = blur(1.0 - ink, 0.35)
    H = L_sh + (screened - L_sh) * strength
    return np.clip(L_hi * H, 0.0, 1.0)               # multiply


def xerox(L, rng, amount=0.5, protect=None):
    """Photocopy degradation. `protect` (0..1) marks areas (the accent) to keep clean."""
    if amount <= 0.0:
        return L
    h, w = L.shape
    keep = 1.0 if protect is None else (1.0 - 0.85 * protect)
    a = amount * keep
    D = 1.0 - L
    # toner density blotching (overlay-like modulation of darkness)
    blot = fbm(h, w, rng, cell=max(24.0, min(h, w) / 9.0), octaves=3)
    D = D * (1.0 + 0.22 * a * blot)
    # contrast crush of the copier
    out = 1.0 - np.clip(D, 0, 1)
    crushed = smoothstep(0.10, 0.90, out)
    out = out + (crushed - out) * (0.55 * a)
    # specks (dust on the glass) in the light areas, drop-outs in the dark ones
    r = rng.random((h, w)).astype(np.float32)
    specks = blur((r < 0.0009 * amount).astype(np.float32), 0.7) * 3.0
    drops = blur((r > 1.0 - 0.0012 * amount).astype(np.float32), 0.8) * 2.5
    out = out * (1.0 - np.clip(specks, 0, 1) * 0.75 * keep * smoothstep(0.3, 0.8, out))
    out = out + (1.0 - out) * np.clip(drops, 0, 1) * 0.6 * keep * smoothstep(0.6, 0.2, out)
    # drum streaks (multiply)
    cols = value_noise(1, w, rng, cell=3.0)[0]
    lines = np.clip((np.abs(cols) - 2.2) * 0.6, 0, 1)
    along = 0.6 + 0.4 * value_noise(h, 1, rng, cell=max(8.0, h / 6.0))[:, :1]
    out = out * (1.0 - 0.10 * a * lines[None, :] * along)
    # fine toner grain (overlay)
    grain = 0.5 + 0.06 * a * blur(rng.standard_normal((h, w)).astype(np.float32), 0.6)
    out = overlay(np.clip(out, 0, 1), grain)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def autolevel(L, alpha=None, lo=1.0, hi=99.3, gamma=1.0):
    """Percentile stretch (inside alpha) so every source prints with a full ink-to-paper range."""
    vals = L[alpha > 0.5] if alpha is not None and (alpha > 0.5).any() else L.ravel()
    a, b = np.percentile(vals, [lo, hi])
    # cap the gain so near-uniform stock (a page with little ink) is not blown into blotches
    a = min(a, b - 0.62)
    out = np.clip((L - a) / max(b - a, 1e-3), 0, 1)
    return out ** gamma if gamma != 1.0 else out


def print_luma(L, rng, pitch, angle, ht_strength, xerox_amount, protect=None):
    phase = (float(rng.random()), float(rng.random()))
    out = halftone(L, pitch=pitch, angle=angle, strength=ht_strength, phase=phase)
    return xerox(out, rng, xerox_amount, protect)
