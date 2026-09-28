#!/usr/bin/env python3
"""Make NAME.part.png - the single part that defines a subject (for the selective-recolour accent).

The part = subject matte  AND  inside a region  AND  within a luminance range, then cleaned up
(largest components, holes filled). Coordinates are fractions of the image (0..1).

    # the jacket on a figure: everything dark below the neck
    python tools/part_mask.py sources/man_back.jpg --region 0,0.30,1,1 --luma 0,0.38
    # a gate: the dark steel plate between two piers
    python tools/part_mask.py sources/sluice_gate.jpg --region 0.40,0.30,0.60,0.82 --luma 0,0.45 --matte none
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from collage import segment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--region", default="0,0,1,1", help="x0,y0,x1,y1 as fractions")
    ap.add_argument("--luma", default="0,1", help="min,max luminance (0..1) of the part")
    ap.add_argument("--matte", default="isnet", choices=["isnet", "u2net", "key", "none"])
    ap.add_argument("--keep", type=int, default=2, help="keep the N largest components")
    ap.add_argument("--out")
    a = ap.parse_args()
    bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    if a.matte in ("isnet", "u2net"):
        m = segment.refine(segment.predict(rgb, a.matte)) > 0.5
    elif a.matte == "key":
        m = segment.refine(segment.key_matte(rgb), lo=0.2, hi=0.8) > 0.5
    else:
        m = np.ones((h, w), bool)
    x0, y0, x1, y1 = [float(v) for v in a.region.split(",")]
    reg = np.zeros((h, w), bool)
    reg[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    lo, hi = [float(v) for v in a.luma.split(",")]
    L = cv2.GaussianBlur(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], (0, 0), 2.0)
    part = (m & reg & (L >= lo) & (L <= hi)).astype(np.uint8)
    part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    part = cv2.morphologyEx(part, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(part, 8)
    if n > 1:
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:a.keep] + 1
        part = np.isin(lab, order).astype(np.uint8)
    inv = (1 - part).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n2):
        x, y, bw, bh, area = st2[i]
        if x > 0 and y > 0 and x + bw < w and y + bh < h and area < 0.02 * h * w:
            part[lab2 == i] = 1
    soft = cv2.GaussianBlur(part.astype(np.float32), (0, 0), 1.2) * m
    out = a.out or os.path.splitext(a.image)[0] + ".part.png"
    cv2.imwrite(out, (np.clip(soft, 0, 1) * 255).astype(np.uint8))
    print(f"[part] {out}  coverage={soft.mean():.3f}")


if __name__ == "__main__":
    main()
