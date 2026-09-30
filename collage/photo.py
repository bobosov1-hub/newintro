"""Full-bleed photographs for shorts: film grade + a 2.5D split (subject plane in front of a
background plane) so the camera's drift and pushes produce real parallax.

The subject is cut out with the local ISNet matte; the hole it leaves in the background is
in-painted so nothing doubles when the planes slide past each other.
"""
import hashlib
import os

import cv2
import numpy as np

from . import ROOT
from . import segment
from .sprite import Sprite

CACHE = os.path.join(ROOT, "output", "_photo_cache")


def grade(rgb, look="70s"):
    """Film looks. '70s': faded, warm, lifted blacks, softened saturation. 'modern': a light touch."""
    x = np.clip(rgb.astype(np.float32), 0, 1)
    lum = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    if look == "70s":
        x = lum + (x - lum) * 0.80                                   # softer colour
        x = x * np.array([1.05, 1.00, 0.88], np.float32) + np.array([0.012, 0.006, 0.0], np.float32)
        x = 0.07 + 0.88 * x                                          # lifted blacks, rolled highlights
        x = x + 0.06 * (x - 0.5) * (1 - np.abs(2 * x - 1))           # gentle S in the mid tones
        # warm the highlights, keep shadows a touch green-grey (print stock)
        hi = np.clip((lum - 0.55) / 0.45, 0, 1)
        x = x + hi * np.array([0.020, 0.008, -0.020], np.float32)
    else:
        x = lum + (x - lum) * 0.95
        x = 0.03 + 0.95 * x
    return np.clip(x, 0, 1).astype(np.float32)


def _upscale(rgb, width):
    h, w = rgb.shape[:2]
    if w >= width:
        return rgb
    s = width / w
    up = cv2.resize(rgb, (width, int(round(h * s))), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(up, (0, 0), 1.2)
    return np.clip(up + 0.45 * (up - blur), 0, 1).astype(np.float32)


def load(path, look="70s", cutout=True, width=1500):
    """-> dict(bg=rgb, fg=rgb or None, a=matte or None). Cached per file + options."""
    key = hashlib.md5(f"{os.path.abspath(path)}|{os.path.getmtime(path)}|{look}|{cutout}|{width}".encode()).hexdigest()[:16]
    cp = os.path.join(CACHE, key + ".npz")
    if os.path.exists(cp):
        z = np.load(cp)
        return {"bg": z["bg"], "fg": z["fg"] if "fg" in z else None, "a": z["a"] if "a" in z else None}
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb, _ = segment.fix_corner_watermark(rgb)
    out = {"fg": None, "a": None}
    if cutout:
        m = segment.refine(segment.predict(rgb, "isnet"), lo=0.35, hi=0.65, keep_frac=0.05, fill_holes=True)
        hole = cv2.dilate((m > 0.2).astype(np.uint8), np.ones((15, 15), np.uint8))
        bg8 = cv2.inpaint((rgb * 255).astype(np.uint8), hole * 255, 12, cv2.INPAINT_TELEA)
        bg = bg8.astype(np.float32) / 255.0
        a = cv2.GaussianBlur(m, (0, 0), 0.8)
        out["fg"] = grade(_upscale(rgb, width), look)
        out["a"] = cv2.resize(a, (out["fg"].shape[1], out["fg"].shape[0]), interpolation=cv2.INTER_LINEAR)
        out["bg"] = grade(_upscale(bg, width), look)
    else:
        out["bg"] = grade(_upscale(rgb, width), look)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cp, **{k: v for k, v in out.items() if v is not None})
    return out


def layers(spec, W, H, d_bg, d_fg, Layer):
    """Build the photo planes for a scene spec {"src", "look", "cutout", "cover", "offset"}."""
    path = spec["src"]
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    ph = load(path, spec.get("look", "70s"), bool(spec.get("cutout", False)))
    bg = ph["bg"]
    h, w = bg.shape[:2]
    cover = float(spec.get("cover", 1.16))              # screen coverage at rest (margin for camera moves)
    scr_w = max(W / w, H / h) * cover * w               # on-screen width at rest
    ox, oy = spec.get("offset", [0, 0])
    out = []
    res_bg = w / (scr_w * d_bg)
    out.append(Layer(Sprite(bg, np.ones((h, w), np.float32), None, res=res_bg), d_bg, ox * d_bg, oy * d_bg,
                     shadow=0, name="photo_bg"))
    if ph["fg"] is not None:
        res_fg = w / (scr_w * d_fg)
        out.append(Layer(Sprite(ph["fg"], ph["a"], None, res=res_fg), d_fg, ox * d_fg, oy * d_fg,
                         shadow=0.35, page=d_bg, name="photo_fg"))
    return out


class VideoLayer:
    """A full-bleed moving picture: decodes its clip lazily (per process) and swaps the layer's
    sprite every frame. Clip time = scene time - t0 + `start`, played at `speed`."""

    def __init__(self, path, t0, W, H, depth, Layer, look="70s", cover=1.10, start=0.0, speed=1.0, offset=(0, 0)):
        import cv2 as _cv
        self.path, self.t0, self.look = path, float(t0), look
        self.start, self.speed = float(start), float(speed)
        cap = _cv.VideoCapture(path)
        self.fps = cap.get(_cv.CAP_PROP_FPS) or 24.0
        self.n = int(cap.get(_cv.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(_cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(_cv.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        scr_w = max(W / w, H / h) * cover * w
        self.res = w / (scr_w * depth)
        self._cap, self._pid, self._idx, self._frame = None, None, -1, None
        blank = np.zeros((h, w, 3), np.float32)
        self.layer = Layer(Sprite(blank, np.ones((h, w), np.float32), None, res=self.res), depth,
                           offset[0] * depth, offset[1] * depth, shadow=0, name="video")
        self.layer.update = self.update

    def _read(self, idx):
        import cv2 as _cv
        if self._cap is None or self._pid != os.getpid():
            self._cap, self._pid, self._idx = _cv.VideoCapture(self.path), os.getpid(), -1
        if idx < self._idx or idx > self._idx + 48:
            self._cap.set(_cv.CAP_PROP_POS_FRAMES, idx)
            self._idx = idx - 1
        while self._idx < idx:
            ok, fr = self._cap.read()
            if not ok:
                break
            self._idx += 1
            self._frame = fr
        return self._frame

    def update(self, t):
        idx = int(np.clip((self.start + max(t - self.t0, 0.0) * self.speed) * self.fps, 0, self.n - 1))
        fr = self._read(idx)
        if fr is None:
            return
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = grade(rgb, self.look)
        h, w = rgb.shape[:2]
        self.layer.sprite = Sprite(rgb, np.ones((h, w), np.float32), None, res=self.res)
