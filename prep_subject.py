#!/usr/bin/env python3
"""Subject preparation: background removal + black-and-white conversion + metadata.

Runs as a plain script (no UI) so the same operation can be batched:

    python prep_subject.py photo.jpg --name sluice_gate --method isnet
    python prep_subject.py man.jpg --name man_back --person --method key
    python prep_subject.py --batch sources/            # uses sources/sources.json if present

For every subject it writes to --out (default assets/):
    NAME.png       RGBA: RGB = black-and-white luminance, A = matte (background-free)
    NAME.full.png  the whole frame in black and white (for the 'print' edge treatment)
    NAME.part.png  optional mask of the single defining part (selective-recolour accent)
    NAME.json      metadata: bbox, mean luminance, reads_light flag, top contour, person flag

Methods: auto (alpha if present, else isnet) | alpha | isnet | u2net | key | none (print only)
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from collage import segment  # noqa: E402
from collage.noise import blur, smoothstep  # noqa: E402
from collage.print_fx import autolevel  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def read_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = None
    if img.shape[2] == 4:
        alpha = img[..., 3].astype(np.float32) / 255.0
        img = img[..., :3]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, alpha


def parse_crop(crop, w, h):
    if crop is None:
        return None
    v = [float(x) for x in (crop.split(",") if isinstance(crop, str) else crop)]
    if all(0.0 <= x <= 1.0 for x in v):
        v = [v[0] * w, v[1] * h, v[2] * w, v[3] * h]
    x0, y0, x1, y1 = [int(round(x)) for x in v]
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def luminance(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def top_profile(a, n=64):
    ys, xs = np.where(a > 0.5)
    if len(ys) == 0:
        return [1.0] * n, [0, 0, a.shape[1], a.shape[0]]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    prof = []
    for i in range(n):
        cx = int(x0 + (i + 0.5) * (x1 - x0) / n)
        col = np.where(a[y0:y1, cx] > 0.5)[0]
        prof.append(round(float(col[0]) / max(1, y1 - y0), 4) if len(col) else 1.0)
    return prof, [int(x0), int(y0), int(x1), int(y1)]


def prep(path, out_dir, name=None, method="auto", person=False, part=None, crop=None,
         max_size=1800, fill_holes=False, watermark=True, lo=0.30, hi=0.70, gamma=1.0):
    name = name or os.path.splitext(os.path.basename(path))[0]
    rgb, alpha = read_rgba(path)
    h, w = rgb.shape[:2]
    part_img = None
    part_path = part or os.path.splitext(path)[0] + ".part.png"
    if os.path.exists(part_path) and part_path != path:
        part_img = cv2.imread(part_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if part_img.shape != (h, w):
            part_img = cv2.resize(part_img, (w, h), interpolation=cv2.INTER_LINEAR)
    c = parse_crop(crop, w, h)
    if c:
        x0, y0, x1, y1 = c
        rgb = rgb[y0:y1, x0:x1]
        alpha = None if alpha is None else alpha[y0:y1, x0:x1]
        part_img = None if part_img is None else part_img[y0:y1, x0:x1]
    s = min(1.0, max_size / max(rgb.shape[:2]))
    if s < 1.0:
        size = (int(rgb.shape[1] * s), int(rgb.shape[0] * s))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        alpha = None if alpha is None else cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        part_img = None if part_img is None else cv2.resize(part_img, size, interpolation=cv2.INTER_AREA)
    fixed = False
    if watermark and alpha is None:
        rgb, fixed = segment.fix_corner_watermark(rgb)

    used = method
    if method == "auto":
        used = "alpha" if (alpha is not None and (alpha < 0.98).mean() > 0.01) else "isnet"
    if used == "alpha":
        if alpha is None:
            raise SystemExit(f"{path}: no alpha channel")
        a = alpha.astype(np.float32)
    elif used in ("isnet", "u2net"):
        a = segment.refine(segment.predict(rgb, used), rgb, lo=lo, hi=hi, fill_holes=fill_holes)
    elif used == "key":
        a = segment.refine(segment.key_matte(rgb), rgb, lo=0.2, hi=0.8, fill_holes=fill_holes)
    elif used == "none":
        a = np.ones(rgb.shape[:2], np.float32)
    else:
        raise SystemExit(f"unknown method {method}")

    L0 = luminance(rgb)
    L = autolevel(L0, a, lo=0.8, hi=99.5, gamma=gamma)
    full = autolevel(L0, None, lo=0.5, hi=99.6, gamma=gamma)
    # gentle print S-curve
    L = np.clip(L + 0.10 * np.sin((L - 0.5) * np.pi) * 0.5, 0, 1)
    full = np.clip(full + 0.10 * np.sin((full - 0.5) * np.pi) * 0.5, 0, 1)

    inside = a > 0.5
    mean_l = float(L[inside].mean()) if inside.any() else 0.5
    edge = (blur(a, 3.0) > 0.05) & (blur(a, 3.0) < 0.95)
    edge_light = float((L[edge] > 0.72).mean()) if edge.any() else 0.0
    prof, bbox = top_profile(a)
    meta = {
        "name": name, "source": os.path.relpath(path, HERE) if path.startswith(HERE) else path,
        "method": used, "size": [int(a.shape[1]), int(a.shape[0])], "bbox": bbox,
        "mean_luma": round(mean_l, 4), "edge_light": round(edge_light, 4),
        "reads_light": bool(mean_l > 0.60 or edge_light > 0.45),
        "person": bool(person), "has_part": part_img is not None, "print_only": used == "none",
        "top_profile": prof, "watermark_removed": fixed,
    }
    os.makedirs(out_dir, exist_ok=True)
    rgba = np.dstack([L, L, L, a])
    cv2.imwrite(os.path.join(out_dir, name + ".png"),
                cv2.cvtColor((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(os.path.join(out_dir, name + ".full.png"), (np.clip(full, 0, 1) * 255 + 0.5).astype(np.uint8))
    if part_img is not None:
        cv2.imwrite(os.path.join(out_dir, name + ".part.png"), (np.clip(part_img, 0, 1) * 255 + 0.5).astype(np.uint8))
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(f"[prep] {name:18s} method={used:6s} size={meta['size']} mean={mean_l:.2f} "
          f"reads_light={meta['reads_light']} part={meta['has_part']} watermark_fix={fixed}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--batch", help="folder of source images (reads sources.json there for per-file options)")
    ap.add_argument("--out", default=os.path.join(HERE, "assets"))
    ap.add_argument("--name")
    ap.add_argument("--method", default="auto", choices=["auto", "alpha", "isnet", "u2net", "key", "none"])
    ap.add_argument("--person", action="store_true")
    ap.add_argument("--part")
    ap.add_argument("--crop", help="x0,y0,x1,y1 in px or 0..1 fractions")
    ap.add_argument("--max-size", type=int, default=1800)
    ap.add_argument("--fill-holes", action="store_true")
    ap.add_argument("--no-watermark-fix", action="store_true")
    a = ap.parse_args()

    jobs = []
    if a.batch:
        manifest = {}
        mp = os.path.join(a.batch, "sources.json")
        if os.path.exists(mp):
            manifest = json.load(open(mp, encoding="utf-8"))
        for p in sorted(glob.glob(os.path.join(a.batch, "*"))):
            if not p.lower().endswith(IMG_EXT) or ".part." in os.path.basename(p):
                continue
            nm = os.path.splitext(os.path.basename(p))[0]
            opts = dict(manifest.get(nm, {}))
            opts.pop("generated", None)
            if opts.pop("skip", False):
                continue
            jobs.append((p, opts))
    for p in a.images:
        jobs.append((p, {"name": a.name, "method": a.method, "person": a.person, "part": a.part,
                         "crop": a.crop, "fill_holes": a.fill_holes}))
    if not jobs:
        ap.error("no input images")
    for p, opts in jobs:
        opts.setdefault("max_size", a.max_size)
        opts.setdefault("watermark", not a.no_watermark_fix)
        prep(p, a.out, **{k: v for k, v in opts.items() if v is not None})


if __name__ == "__main__":
    main()
