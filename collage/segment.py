"""Local background removal (no UI, no network at run time once the model is cached).

Models are the public rembg ONNX exports (ISNet general-use, U^2-Net). They are downloaded
once from the rembg GitHub release into ~/.cache/collage-models.
"""
import os
import urllib.request

import cv2
import numpy as np

from .noise import smoothstep

MODEL_DIR = os.environ.get("COLLAGE_MODEL_DIR", os.path.expanduser("~/.cache/collage-models"))
MODELS = {
    # name: (file, input size, mean, std)
    "isnet": ("isnet-general-use.onnx", 1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "u2net": ("u2net.onnx", 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
_sessions = {}


def model_path(name):
    fn = MODELS[name][0]
    path = os.path.join(MODEL_DIR, fn)
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"[segment] downloading {fn} ...", flush=True)
        urllib.request.urlretrieve(URL + fn, path + ".part")
        os.replace(path + ".part", path)
    return path


def _session(name):
    if name not in _sessions:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2))
        _sessions[name] = ort.InferenceSession(model_path(name), so, providers=["CPUExecutionProvider"])
    return _sessions[name]


def predict(rgb, name="isnet"):
    """rgb: HxWx3 float 0..1 -> soft matte HxW 0..1."""
    fn, size, mean, std = MODELS[name]
    sess = _session(name)
    im = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    im = im / max(float(im.max()), 1e-6)
    x = ((im - np.array(mean, np.float32)) / np.array(std, np.float32)).transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    return cv2.resize(out, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def key_matte(rgb, tol=0.10, soft=0.06):
    """Plain-backdrop key: distance from the colour sampled along the image border."""
    h, w = rgb.shape[:2]
    b = max(4, min(h, w) // 40)
    border = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                             rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # allow a gentle backdrop gradient: compare with a heavily blurred, background-only estimate
    d = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    return smoothstep(tol, tol + soft, d).astype(np.float32)


def refine(matte, rgb=None, lo=0.30, hi=0.70, keep_frac=0.02, fill_holes=False):
    """Crisp up a soft matte, drop small islands, optionally fill holes."""
    a = smoothstep(lo, hi, matte)
    m = (a > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = areas >= keep_frac * areas.max()
        # keep the big regions plus the soft edge pixels touching them
        grown = cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8))
        a = np.where(grown > 0, smoothstep(lo, hi, matte), 0.0).astype(np.float32)
    if fill_holes:
        inv = (a < 0.5).astype(np.uint8)
        n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, 4)
        h, w = a.shape
        for i in range(1, n2):
            x, y, bw, bh, area = stats2[i]
            if x > 0 and y > 0 and x + bw < w and y + bh < h:
                a[lab2 == i] = 1.0
    return a.astype(np.float32)


def fix_corner_watermark(rgb):
    """Remove a small bright generator watermark (e.g. a sparkle) from the bottom-right corner."""
    h, w = rgb.shape[:2]
    y0, x0 = int(h * 0.78), int(w * 0.86)
    patch = rgb[y0:, x0:]
    L = patch.mean(axis=2)
    hp = L - cv2.medianBlur((L * 255).astype(np.uint8), 31).astype(np.float32) / 255.0
    m = (hp > 0.035).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return rgb, False
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[i, cv2.CC_STAT_AREA]
    bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    if not (40 <= area <= 0.02 * h * w and 0.5 < bw / max(bh, 1) < 2.0):
        return rgb, False
    mask = np.zeros((h, w), np.uint8)
    x, y = stats[i, cv2.CC_STAT_LEFT] + x0, stats[i, cv2.CC_STAT_TOP] + y0
    pad = int(max(bw, bh) * 0.35) + 4
    mask[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad] = 255
    out = cv2.inpaint((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mask, 9, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255.0, True
