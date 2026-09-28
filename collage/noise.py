"""Small, fast procedural-noise helpers (numpy + OpenCV)."""
import cv2
import numpy as np


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def value_noise(h, w, rng, cell):
    """Smooth noise with feature size ~cell px, roughly unit variance."""
    cell = max(float(cell), 1.0)
    if cell > 16.0:
        # large features: synthesise at low resolution, then upsample (same look, far cheaper)
        r = cell / 8.0
        small = value_noise(max(2, int(np.ceil(h / r))), max(2, int(np.ceil(w / r))), rng, 8.0)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    gh, gw = int(np.ceil(h / cell)) + 3, int(np.ceil(w / cell)) + 3
    g = rng.standard_normal((gh, gw)).astype(np.float32)
    big = cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)
    oy = int(rng.integers(0, max(1, big.shape[0] - h)))
    ox = int(rng.integers(0, max(1, big.shape[1] - w)))
    out = big[oy:oy + h, ox:ox + w]
    return out / (out.std() + 1e-6)


def fbm(h, w, rng, cell, octaves=4, gain=0.5):
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    for i in range(octaves):
        out += amp * value_noise(h, w, rng, cell / (2.0 ** i))
        amp *= gain
    return out / (out.std() + 1e-6)


def noise1d(n, rng, cell, octaves=4, gain=0.5):
    """1-D fractal noise of length n (unit variance)."""
    out = np.zeros(n, np.float32)
    amp = 1.0
    for i in range(octaves):
        c = max(cell / (2.0 ** i), 1.0)
        m = int(np.ceil(n / c)) + 4
        g = rng.standard_normal(m).astype(np.float32)
        x = np.arange(n) / c + 1.0
        out += amp * np.interp(x, np.arange(m), g).astype(np.float32)
        amp *= gain
    out = cv2.GaussianBlur(out.reshape(1, -1), (0, 0), sigmaX=max(cell / 16.0, 0.6)).ravel()
    return out / (out.std() + 1e-6)


def streaks(h, w, rng, length, angle_deg=90.0, density=1.0):
    """Anisotropic noise (fibres, hair, brushed metal): white noise smeared along a direction."""
    n = rng.standard_normal((h, w)).astype(np.float32) * density
    k = int(max(3, length)) | 1
    kern = np.zeros((k, k), np.float32)
    c = k // 2
    a = np.deg2rad(angle_deg)
    for i in range(k):
        t = i - c
        x, y = int(round(c + t * np.cos(a))), int(round(c + t * np.sin(a)))
        kern[y, x] = 1.0
    kern /= kern.sum()
    out = cv2.filter2D(n, -1, kern, borderType=cv2.BORDER_REFLECT)
    return out / (out.std() + 1e-6)


def blur(img, sigma):
    if sigma <= 0.05:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)


def dist_outside(mask):
    """Distance (px) from each pixel to the mask (0 inside)."""
    m = (mask < 0.5).astype(np.uint8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)


def dist_inside(mask):
    m = (mask >= 0.5).astype(np.uint8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
