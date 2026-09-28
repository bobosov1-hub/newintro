"""Printed maps from the KOSTAT boundaries in data/korea_map.json."""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import DATA
from ..noise import blur, fbm
from ..paper import paper_luma
from ..typography import font

_DATA = None


def data():
    global _DATA
    if _DATA is None:
        with open(os.path.join(DATA, "korea_map.json"), encoding="utf-8") as fh:
            _DATA = json.load(fh)
    return _DATA


def _fit(polys, width, pad):
    pts = np.concatenate([np.asarray(p, np.float32) for p in polys])
    mn, mx = pts.min(0), pts.max(0)
    s = (width - 2 * pad) / float(mx[0] - mn[0])
    height = int((mx[1] - mn[1]) * s + 2 * pad)
    return (lambda p: (np.asarray(p, np.float32) - mn) * s + pad), height, s


def _fill(shape, polys, tf, ss=4):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [(tf(p) * ss).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    return m.astype(np.float32) / 255.0


def _stroke(shape, polys, tf, width):
    m = np.zeros(shape, np.float32)
    for p in polys:
        cv2.polylines(m, [(tf(p) * 4).astype(np.int32)], True, 1.0, int(width), cv2.LINE_AA, shift=2)
    return m


def _dots(shape, pitch=7.0, r=1.1):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + yy * 0.5) / pitch
    v = yy / (pitch * 0.866)
    du = u - np.round(u)
    dv = v - np.round(v)
    d = np.sqrt((du * pitch) ** 2 + (dv * pitch * 0.866) ** 2)
    return np.clip(r + 0.5 - d, 0, 1)


def _labels(shape, items):
    img = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for text, (x, y), size, weight in items:
        f = font(size, weight)
        d.text((x, y), text, font=f, fill=255, anchor="mm", stroke_width=0)
    return np.asarray(img, np.float32) / 255.0


def _centroid(polys, tf):
    best = max(polys, key=lambda p: abs(cv2.contourArea(np.asarray(p, np.float32))))
    M = cv2.moments(tf(best).astype(np.float32))
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def jeonbuk(seed=21, width=1100):
    """Jeonbuk municipalities, Gimje as the defining part."""
    rng = np.random.default_rng(seed)
    regions = data()["jeonbuk"]
    allp = [p for r in regions for p in r["polys"]]
    tf, H, _ = _fit(allp, width, 40)
    W = width
    land = _fill((H, W), allp, tf)
    gim = next(r for r in regions if r["name"] == "김제시")
    gm = _fill((H, W), gim["polys"], tf)
    borders = _stroke((H, W), [p for r in regions for p in r["polys"]], tf, 2)
    coast = _stroke((H, W), allp, tf, 4) * (1 - blur(land, 0.1) * 0)
    tex = paper_luma(H, W, rng, 1.6)
    L = 0.86 * tex - 0.10 * _dots((H, W), 6.0, 0.9)
    L = L * (1 - gm) + (0.46 * tex + 0.04 * fbm(H, W, rng, 40.0, 3)) * gm
    hatch = np.zeros((H, W), np.float32)
    for k in range(-H, W, 11):
        cv2.line(hatch, (k, 0), (k + H, H), 1.0, 2, cv2.LINE_AA)
    L = L * (1 - hatch * gm * 0.25)
    L = L * (1 - np.clip(borders * 0.75 + coast * 0.2, 0, 1))
    jeonju = [p for r in regions if r["name"].startswith("전주") for p in r["polys"]]
    names = [("김제", _centroid(gim["polys"], tf), 44, 800)]
    for nm, key in [("전주", None), ("군산", "군산시"), ("익산", "익산시"), ("정읍", "정읍시"), ("부안", "부안군"),
                    ("완주", "완주군"), ("남원", "남원시"), ("고창", "고창군"), ("임실", "임실군"), ("진안", "진안군"),
                    ("무주", "무주군"), ("장수", "장수군"), ("순창", "순창군")]:
        polys = jeonju if key is None else next(r["polys"] for r in regions if r["name"] == key)
        names.append((nm, _centroid(polys, tf), 26, 600))
    lab = _labels((H, W), names)
    L = L * (1 - lab * 0.88) + 0.10 * lab * 0.88
    # the defining part: Gimje without its label
    part = np.clip(gm * (1 - lab), 0, 1)
    return np.clip(L, 0, 1).astype(np.float32), np.clip(land, 0, 1), part


def korea_px(lon, lat, width=900):
    """Longitude/latitude -> pixel position on the korea() map of the given width."""
    import math
    lat0 = data()["projection"]["lat0"]
    kx = 111.32 * math.cos(math.radians(lat0))
    pt = [(lon - 127.5) * kx, -(lat - lat0) * 110.57]
    allp = [p for r in data()["provinces"] for p in r["polys"]]
    tf, _, _ = _fit(allp, width, 30)
    q = tf([pt])[0]
    return float(q[0]), float(q[1])


def korea(seed=22, width=900, mark="김제시"):
    """South Korea by province with the case location ringed."""
    rng = np.random.default_rng(seed)
    provs = data()["provinces"]
    allp = [p for r in provs for p in r["polys"]]
    tf, H, _ = _fit(allp, width, 30)
    W = width
    land = _fill((H, W), allp, tf)
    tex = paper_luma(H, W, rng, 1.6)
    L = 0.84 * tex - 0.09 * _dots((H, W), 6.0, 0.9)
    shade = np.zeros((H, W), np.float32)
    for i, r in enumerate(provs):
        if i % 3 == 0:
            shade = np.maximum(shade, _fill((H, W), r["polys"], tf) * 0.06)
    L = L - shade
    borders = _stroke((H, W), allp, tf, 2)
    L = L * (1 - borders * 0.7)
    if mark:
        gim = next(r for r in data()["jeonbuk"] if r["name"] == mark)
        cx, cy = _centroid(gim["polys"], tf)
        dot = np.zeros((H, W), np.float32)
        cv2.circle(dot, (int(cx * 4), int(cy * 4)), 7 * 4, 1.0, -1, cv2.LINE_AA, shift=2)
        L = L * (1 - dot) + 0.1 * dot
    return np.clip(L, 0, 1).astype(np.float32), np.clip(land, 0, 1), None
