"""Scene builder: one storyboard entry -> layers, diagrams and a camera.

Spec coordinates are SCREEN pixels at rest (origin = frame centre); they are converted to world
units per plane depth so the composition matches the storyboard when the camera is at rest.
"""
import math

import numpy as np

from . import palette as P
from .camera import Camera
from .compositor import Layer
from .diagrams import Diagram, arrow, cross, outline_from_alpha, ring, timeline, underline
from .edges import torn_mask
from .paper import paper_luma, sheet, stain
from .sprite import Sprite
from .subject import Asset, build_element
from .typography import label, lockup

D_PAGE = 1.12
D_LOCK = 1.03
D_HERO = 1.0


class Scene:
    def __init__(self, sid, t0, t1, layers, diagrams, camera, transition, info):
        self.id, self.t0, self.t1 = sid, t0, t1
        self.layers, self.diagrams, self.camera = layers, diagrams, camera
        self.transition = transition
        self.info = info


def _rot(v, deg):
    r = math.radians(deg)
    return (v[0] * math.cos(r) - v[1] * math.sin(r), v[0] * math.sin(r) + v[1] * math.cos(r))


def _enter(build, rng, kind=None, scale=1.0, side=None):
    """Entrance: ease-out expo + small overshoot, stepped. 'drop' = placed by hand onto the page,
    'slide' = slid in from a side."""
    c = build.chaos
    kind = kind or build.pick(rng, ["drop", "slide"], [0.55, 0.45])
    if kind == "drop":
        return {"dx": build.jit(rng, 6, 40) * scale, "dy": build.jit(rng, 6, 40) * scale - 12 * scale,
                "drot": build.jit(rng, 2.5, 11.0), "dscale": 0.10 + 0.08 * c, "lift": 0.10, "overshoot": 0.05}
    side = side if side is not None else int(rng.integers(0, 4))
    dist = (260 + 340 * float(rng.random())) * scale
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dx, dy = dirs[side]
    return {"dx": dx * dist, "dy": dy * dist * 0.7, "drot": build.jit(rng, 2.0, 10.0), "dscale": 0.03,
            "lift": 0.05, "overshoot": 0.07}


def _local_point(meta, sx, sy):
    """Asset pixel -> element-local world units."""
    ox, oy = meta["src_origin"]
    s = meta["src_scale"]
    return ((sx - ox) * s - meta["sub_px"][0] / 2.0) / meta["res"], ((sy - oy) * s - meta["sub_px"][1] / 2.0) / meta["res"]


def _asset(spec):
    """Asset.get with an optional 'fallback' name; None when neither is prepared."""
    for name in (spec.get("asset"), spec.get("fallback")):
        if not name:
            continue
        try:
            return Asset.get(name)
        except FileNotFoundError:
            continue
    return None


def build_scene(spec, build, pool, W=1920, H=1080):
    sid = spec["id"]
    t0, t1 = float(spec["start"]), float(spec["end"])
    rng = build.rng("scene", sid)
    c = build.chaos
    layers, diagrams = [], []
    counter = [0]
    info = {"id": sid}

    def add(layer):
        layer.order = counter[0]
        counter[0] += 1
        layers.append(layer)
        return layer

    ink_page = spec.get("page") == "ink"
    # 1. the page ------------------------------------------------------------------
    pw, ph = int(W * D_PAGE * 1.36), int(H * D_PAGE * 1.36)
    prng = build.rng("page", sid)
    page_rgb = sheet(ph, pw, prng, base=P.INK if ink_page else P.PAPER, amount=1.0)
    if not ink_page:
        page_rgb *= stain(ph, pw, prng, 0.045)[..., None]
    add(Layer(Sprite(page_rgb, np.ones((ph, pw), np.float32)), D_PAGE, 0, 0, shadow=0, name="page"))

    # 2. background scatter: torn newspaper fragments, low contrast, placed by the seed
    n_sc = spec.get("scatter")
    n_sc = int(round(build.amt(5, 11))) if n_sc is None else int(n_sc)
    if ink_page:
        n_sc = 0
    for i in range(n_sc):
        frag = pool.fragment((sid, i), scale=float(rng.uniform(0.8, 1.35)))
        x = float(rng.uniform(-0.60, 0.60)) * W * D_PAGE
        y = float(rng.uniform(-0.58, 0.58)) * H * D_PAGE
        rot = build.jit(rng, 4.0, 38.0) + (90.0 if rng.random() < 0.12 else 0.0)
        add(Layer(frag, D_PAGE - 0.0006 * (i + 1), x, y, rot, shadow=0.30, page=D_PAGE, name=f"scatter{i}"))

    # 3. supporting elements (black and white only) ----------------------------------------
    sup_meta = []
    sup_times = []
    for j, ss in enumerate(spec.get("support", [])):
        srng = build.rng("support", sid, j)
        asset = _asset(ss)
        if asset is None:
            print(f"[scene {sid}] support '{ss.get('asset')}' not prepared - skipped")
            continue
        el = build_element(asset, build, (sid, "sup", j), role="support", height=float(ss.get("height", 400)),
                           res=float(ss.get("res", 1.0)), edge=ss.get("edge"), crop=ss.get("crop"), bands=0,
                           card=ss.get("card"))
        depth = float(ss.get("depth", 1.05))
        ax, ay = ss.get("at", [0, 0])
        sx = ax + build.jit(srng, 4, 70)
        sy = ay + build.jit(srng, 4, 55)
        srot = float(ss.get("rot", 0.0)) + build.jit(srng, 0.6, 9.0)
        ssc = 1.0 + build.jit(srng, 0.01, 0.10)
        t_in = float(ss.get("enter", t0 + 0.15 + 0.14 * j))
        sup_times.append(t_in)
        ent = _enter(build, srng, ss.get("enter_kind"))
        name = ss.get("name", f"sup{j}")
        for k, pc in enumerate(el.pieces):
            off = _rot((pc.offset[0] * ssc, pc.offset[1] * ssc), srot)
            dd = depth + (0.006 if pc.role in ("card", "backing") else 0.0)
            add(Layer(pc.sprite, dd, (sx + off[0]) * dd, (sy + off[1]) * dd, srot + pc.rot, ssc * dd,
                      t_in=t_in, dur_in=float(ss.get("dur", 0.6)), enter=ent, shadow=1.0, page=D_PAGE,
                      name=name if pc.role == "body" else f"{name}_{pc.role}", t_out=ss.get("exit")))
        sup_meta.append({"el": el, "x": sx, "y": sy, "rot": srot, "scale": ssc, "depth": depth, "name": name,
                         "asset": asset})

    # 4. hero element --------------------------------------------------------------------
    hs = spec.get("hero") or {}
    hero = None
    hx = hy = 0.0
    asset = _asset(hs) if hs.get("asset") else None
    if hs.get("asset") and asset is None:
        print(f"[scene {sid}] hero '{hs.get('asset')}' not prepared - lockup only")
    if asset is not None:
        hrng = build.rng("hero", sid)
        bands = hs.get("bands", -1 if asset.meta.get("person") else 0)
        el = build_element(asset, build, (sid, "hero"), role="hero", height=float(hs.get("height", 720)),
                           res=float(hs.get("res", 1.2)), accent=hs.get("accent"), edge=hs.get("edge"),
                           crop=hs.get("crop"), bands=bands, card=hs.get("card"))
        hscale = 1.0 + build.jit(hrng, 0.0, 0.07)
        hrot = float(hs.get("rot", 0.0)) + build.jit(hrng, 0.3, 4.0)
        hx = float(hs.get("x", 0.0)) + build.jit(hrng, 4, 60)
        tx, ty = el.meta["top_x"], el.meta["top_y"]
        finite = np.isfinite(ty)
        if "top" in hs:
            hy = float(hs["top"]) - float(np.min(ty[finite])) * hscale + build.jit(hrng, 2, 30)
        else:
            hy = float(hs.get("y", 0.0)) + build.jit(hrng, 2, 30)
        hero = {"el": el, "x": hx, "y": hy, "rot": hrot, "scale": hscale, "asset": asset}
        info["hero"] = {"asset": asset.name, "accent": el.meta["accent"], "edge": el.meta["edge"],
                        "bands": el.meta["bands"]}

    t_hero = None
    if hero is not None:
        # the hero arrives last within its entrance group (elements entering up to its cue);
        # later beats may still add type or supporting pieces
        req = float(hs.get("enter", t0 + 0.9))
        t_hero = req
        pre = [ts for ts in sup_times if ts <= req + 0.2]
        if pre:
            t_hero = max(t_hero, max(pre) + 0.25)
        lsp = spec.get("lockup")
        if lsp and float(lsp.get("enter", t0 + 0.4)) <= req + 0.2:
            t_hero = max(t_hero, float(lsp.get("enter", t0 + 0.4)) + 0.35)
        brng0 = build.rng("backing-in", sid)
        for pc in hero["el"].pieces:
            if pc.role not in ("backing", "card"):
                continue
            off = _rot((pc.offset[0] * hero["scale"], pc.offset[1] * hero["scale"]), hero["rot"])
            depth = D_HERO + 0.010
            add(Layer(pc.sprite, depth, (hx + off[0]) * depth, (hy + off[1]) * depth, hero["rot"] + pc.rot,
                      hero["scale"] * depth, t_in=t_hero - 0.22, dur_in=0.6, enter=_enter(build, brng0, "drop", 0.8),
                      shadow=1.0, page=D_PAGE, name=f"hero_{pc.role}"))

    # 5. lockup (thin descriptor + heavy title), one unit, behind the hero's top edge ----------
    ls = spec.get("lockup")
    lock_layer = None
    lock_meta = None
    if ls:
        lrng = build.rng("lockup", sid)
        title_is_hero = bool(ls.get("hero"))
        tcol = P.ACCENT if title_is_hero else (P.PAPER if ink_page else P.INK)
        dcol = P.PAPER if ink_page else P.INK
        spr, m = lockup(ls.get("descriptor", ""), ls["title"], build, lrng, title_size=float(ls.get("size", 138)),
                        color=tcol, desc_color=dcol, align=ls.get("align", "center"), rule=ls.get("rule"))
        k_fit = min(1.0, float(ls.get("max_width", 1560)) / (spr.w / spr.res))
        cap = m["cap"] / spr.res * k_fit
        base_off = (m["title_baseline"] - spr.h / 2.0) / spr.res * k_fit
        if "at" in ls or hero is None:
            lx, ly = ls.get("at", [0, -80])
            lx += build.jit(lrng, 3, 30)
            ly += build.jit(lrng, 2, 20)
        else:
            el = hero["el"]
            lx = hx + float(ls.get("dx", 0.0)) + build.jit(lrng, 4, 36)
            tw = (m["width"] / spr.res) * k_fit * 0.42
            # top contour in screen units, including the hero's scale and rotation
            r = math.radians(hero["rot"])
            tx0, ty0 = el.meta["top_x"] * hero["scale"], el.meta["top_y"] * hero["scale"]
            xs = tx0 * math.cos(r) - ty0 * math.sin(r) + hx
            ys = tx0 * math.sin(r) + ty0 * math.cos(r) + hy
            ok = np.isfinite(ys) & (np.abs(xs - lx) < tw)
            cand = ys[ok] if ok.any() else ys[np.isfinite(ys)]
            # a person: over the head. An object: over its upper contour (robust to stray bits)
            y_top = float(np.min(cand)) if el.meta["person"] else float(np.percentile(cand, 3))
            ov = float(ls.get("overlap", 0.30 if el.meta["person"] else -0.02)) + build.jit(lrng, 0.0, 0.04)
            ly = y_top + ov * cap - base_off
        lrot = build.jit(lrng, 0.2, 2.5)
        ent = _enter(build, lrng, ls.get("enter_kind", "drop"), 0.6)
        ent["dscale"] = 0.06 + 0.05 * c
        ent["lift"] = 0.0
        lock_layer = add(Layer(spr, D_LOCK, lx * D_LOCK, ly * D_LOCK, lrot, k_fit * D_LOCK,
                               t_in=float(ls.get("enter", t0 + 0.4)), dur_in=0.6, enter=ent, shadow=0.0,
                               page=D_PAGE, name="lockup", t_out=ls.get("exit")))
        lock_meta = {"x": lx, "y": ly, "k": k_fit, "m": m, "spr": spr, "base_off": base_off, "cap": cap}
        info["lockup"] = ls["title"].replace("\n", " ")

    # hero layers go on top of the lockup (the hero's top edge overlaps the title);
    # its paper backing / grey card were already laid down underneath the lockup
    if hero is not None:
        el = hero["el"]
        brng = build.rng("bands", sid)
        n_b = el.meta["bands"]
        stagger = 0.075 + 0.05 * c
        ex = hs.get("exit")
        for pc in el.pieces:
            if pc.role in ("backing", "card"):
                continue
            off = _rot((pc.offset[0] * hero["scale"], pc.offset[1] * hero["scale"]), hero["rot"])
            rot = hero["rot"] + pc.rot
            px, py = hx + off[0], hy + off[1]
            depth = D_HERO
            t_in = t_hero
            dur = 0.62
            t_out = None
            exit_spec = None
            if pc.role == "band":
                i = pc.index
                side = 1 if (i % 2 == 0) == (brng.random() < 0.5) else 0
                px += build.jit(brng, 0.004, 0.045) * el.meta["subject_w"]
                rot += build.jit(brng, 0.25, 3.2)
                depth = D_HERO - 0.002 * i
                t_in = t_hero + i * stagger + build.jit(brng, 0.0, 0.05)
                ent = _enter(build, brng, "slide", 0.9, side=side)
                ent["dy"] = build.jit(brng, 5, 40)
                dur = 0.55
                if ex:
                    t_out = float(ex["t"]) + i * 0.08 + build.jit(brng, 0.0, 0.05)
                    sgn = -1 if side == 0 else 1
                    exit_spec = {"dx": sgn * float(brng.uniform(1000, 1500)), "dy": float(brng.uniform(-180, 180)),
                                 "drot": sgn * float(brng.uniform(10, 28)), "lift": 0.10}
            else:
                ent = _enter(build, brng, hs.get("enter_kind"), 1.0)
                if ex and ex.get("all"):
                    t_out = float(ex["t"])
                    exit_spec = {"dx": float(brng.choice([-1, 1])) * 1500.0, "drot": 20.0, "lift": 0.1}
            add(Layer(pc.sprite, depth, px * depth, py * depth, rot, hero["scale"] * depth, t_in=t_in, dur_in=dur,
                      enter=ent, shadow=1.0, page=D_PAGE, name=f"hero_{pc.role}{pc.index}", t_out=t_out,
                      dur_out=0.5, exit=exit_spec))
        hero["t_in"] = t_hero
        if ex and ex.get("ghost", True) and n_b:
            # a dashed outline drawn where the figure used to be
            ghost = _ghost_outline(el, hero)
            if ghost:
                diagrams.append(Diagram(ghost, float(ex["t"]) + 0.45, dur=1.3, depth=D_HERO + 0.005, width=3.2,
                                        dashed=(16, 11), color=P.INK, name="ghost"))

    # 6. foreground scraps (close to camera: strong parallax + focal defocus) -----------------
    n_fg = spec.get("fg")
    n_fg = int(rng.integers(1, 3)) if n_fg is None else int(n_fg)
    for k in range(n_fg):
        frng = build.rng("fg", sid, k)
        kind = build.pick(frng, ["news", "tape", "paper"], [0.45, 0.30, 0.25])
        if kind == "news":
            spr = pool.fragment((sid, "fg", k), scale=1.1, contrast=0.32)
        elif kind == "tape":
            try:
                ta = Asset.get("tape")
                L = ta.L
                rgb = P.tone(np.clip(L, 0, 1))
                spr = Sprite(rgb, ta.a * 0.85, None, res=0.6)
            except FileNotFoundError:
                spr = pool.fragment((sid, "fg", k), scale=0.8)
        else:
            fh, fw = int(frng.uniform(220, 420)), int(frng.uniform(300, 620))
            rgb = P.PAPER * paper_luma(fh, fw, frng, 1.0)[..., None]
            spr = Sprite(rgb, torn_mask(fh, fw, frng, 1.3, 7.0), None, res=1.0)
        depth = float(frng.uniform(0.58, 0.78))
        side = build.pick(frng, ["l", "r", "t", "b"], [0.4, 0.4, 0.2, 0.0])
        if side in ("l", "r"):
            sx = (-1 if side == "l" else 1) * float(frng.uniform(0.46, 0.58)) * W
            sy = float(frng.uniform(-0.42, 0.30)) * H
        else:
            sx = float(frng.uniform(-0.45, 0.45)) * W
            sy = (-1 if side == "t" else 1) * float(frng.uniform(0.46, 0.56)) * H
        ent = _enter(build, frng, "slide", 1.2, side={"l": 0, "r": 1, "t": 2, "b": 3}[side])
        add(Layer(spr, depth, sx * depth, sy * depth, float(frng.uniform(-40, 40)), depth, t_in=t0 + float(frng.uniform(0.0, 0.5)),
                  dur_in=0.55, enter=ent, shadow=0.6, page=D_PAGE, name=f"fg{k}"))

    # 7. diagrams (drawn at full frame rate) -----------------------------------------------
    for q, ds in enumerate(spec.get("diagrams", [])):
        drng = build.rng("diagram", sid, q)
        diagrams.extend(_diagram(ds, drng, hero, lock_meta, sup_meta, layers, add, build))

    # 8. camera ---------------------------------------------------------------------------
    beats = []
    for b in spec.get("beats", []):
        b = dict(b)
        tgt = b.get("target", "hero")
        if tgt == "part" and hero is not None and hero["asset"].part is not None:
            ys_, xs_ = np.nonzero(hero["asset"].part > 0.5)
            lx, ly = _local_point(hero["el"].meta, float(xs_.mean()), float(ys_.mean()))
            lx, ly = _rot((lx * hero["scale"], ly * hero["scale"]), hero["rot"])
            b["target"] = (hx + lx, hy + ly)
        elif tgt in ("hero", "part") and hero is not None:
            # land on the upper part of the hero (head / upper contour) so the lockup stays in frame
            ys_top = hero["el"].meta["top_y"]
            top = hy + float(np.min(ys_top[np.isfinite(ys_top)])) * hero["scale"]
            span = min(0.30 * hero["el"].meta["subject_h"] * hero["scale"], 240.0)
            b["target"] = (hx + float(b.get("dx", 0.0)), top + span + float(b.get("dy", 0.0)))
            b.setdefault("pull", 0.45)
        elif tgt == "lockup" and lock_meta is not None:
            b["target"] = (lock_meta["x"] * D_LOCK, lock_meta["y"] * D_LOCK)
        elif isinstance(tgt, (list, tuple)):
            b["target"] = tuple(tgt)
        else:
            b.pop("target", None)
        if b.get("type") == "pull":
            b["amount"] = -abs(float(b.get("amount", 0.08)))
        beats.append(b)
    start = spec.get("camera_start", [build.jit(rng, 5, 40), build.jit(rng, 5, 30), 0.0])
    rack = (t0 + 0.1, 1.0, 0.62) if spec.get("rack") else None
    focus = D_LOCK if (hero is None and lock_meta is not None) else D_HERO
    cam = Camera(t0, t1 + 1.0, build, sid, drift=spec.get("drift"), beats=beats, focus=focus, rack=rack,
                 start=tuple(start))
    trans = dict(spec.get("transition", {"type": "cut"}))
    if trans.get("leak") is None:
        trans["leak"] = bool(build.rng("leak?", sid).random() < 0.5) and trans.get("type") != "cut"
    return Scene(sid, t0, t1, layers, diagrams, cam, trans, info)


def _ghost_outline(el, hero):
    bands = [p for p in el.pieces if p.role == "band"]
    if not bands:
        return None
    res = bands[0].sprite.res
    xs = [p.offset[0] - p.sprite.w / res / 2 for p in bands]
    ys = [p.offset[1] - p.sprite.h / res / 2 for p in bands]
    x0, y0 = min(xs), min(ys)
    x1 = max(p.offset[0] + p.sprite.w / res / 2 for p in bands)
    y1 = max(p.offset[1] + p.sprite.h / res / 2 for p in bands)
    Wc, Hc = int((x1 - x0) * res) + 4, int((y1 - y0) * res) + 4
    canvas = np.zeros((Hc, Wc), np.float32)
    for p in bands:
        px = int((p.offset[0] - p.sprite.w / res / 2 - x0) * res)
        py = int((p.offset[1] - p.sprite.h / res / 2 - y0) * res)
        h, w = p.sprite.a.shape
        canvas[py:py + h, px:px + w] = np.maximum(canvas[py:py + h, px:px + w], p.sprite.a[:Hc - py, :Wc - px])
    import cv2
    canvas = cv2.GaussianBlur(canvas, (0, 0), 6.0)
    strokes = outline_from_alpha(canvas, (x0, y0), res, step=5)
    sc = hero["scale"]
    out = []
    for s in strokes:
        s = np.array([_rot((p[0] * sc, p[1] * sc), hero["rot"]) for p in s], np.float32)
        s[:, 0] += hero["x"]
        s[:, 1] += hero["y"]
        out.append(s * D_HERO)
    return out


def _diagram(ds, rng, hero, lock_meta, sup_meta, layers, add, build):
    kind = ds["type"]
    t_in = float(ds["t"])
    dur = float(ds.get("dur", 0.7))
    width = float(ds.get("width", 5.0))
    out = []
    if kind == "underline_title" and lock_meta:
        m, k = lock_meta["m"], lock_meta["k"]
        spr = lock_meta["spr"]
        wv = m["width"] / spr.res * k
        x0 = lock_meta["x"] - wv * float(ds.get("span", 0.34))
        x1 = lock_meta["x"] + wv * float(ds.get("span", 0.34))
        y = lock_meta["y"] + lock_meta["base_off"] + lock_meta["cap"] * 0.18 + float(ds.get("dy", 0))
        strokes = [s * D_LOCK for s in underline(x0, x1, y, rng)]
        lk = next((l for l in layers if l.name == "lockup"), None)
        out.append(Diagram(strokes, t_in, dur, depth=D_LOCK, width=width, z=(lk.order + 0.5) if lk else None))
    elif kind in ("ring", "cross_days", "arrow", "timeline"):
        # anchor: hero or a named support
        tgt = ds.get("on", "hero")
        if tgt == "hero":
            base = hero
            depth = D_HERO
        else:
            base = next(s for s in sup_meta if s["name"] == tgt)
            depth = base["depth"]
        if kind == "ring":
            if ds.get("src") == "part" and base is not None:
                part = base["asset"].part
                ys_, xs_ = np.nonzero(part > 0.5)
                lx, ly = _local_point(base["el"].meta, float(xs_.mean()), float(ys_.mean()))
            elif "lonlat" in ds and base is not None:
                from .procedural.maps import korea_px
                lx, ly = _local_point(base["el"].meta, *korea_px(*ds["lonlat"], width=base["asset"].L.shape[1]))
            elif "src" in ds and base is not None:
                lx, ly = _local_point(base["el"].meta, *ds["src"])
            else:
                lx, ly = ds.get("rel", [0, 0])
            lx, ly = _rot((lx * base["scale"], ly * base["scale"]), base["rot"])
            cx, cy = base["x"] + lx, base["y"] + ly
            rx, ry = ds.get("r", [120, 80])
            strokes = [s * depth for s in ring(cx, cy, rx, ry, rng)]
            out.append(Diagram(strokes, t_in, dur, depth=depth - 0.004, width=width))
        elif kind == "cross_days":
            from .procedural.docs import calendar
            cells = calendar()[3]
            strokes = []
            for d in ds["days"]:
                lx, ly = _local_point(base["el"].meta, *cells[d])
                lx, ly = _rot((lx * base["scale"], ly * base["scale"]), base["rot"])
                size = float(ds.get("size", 52)) * base["scale"]
                strokes.extend(cross(base["x"] + lx, base["y"] + ly, size, rng))
            strokes = [s * depth for s in strokes]
            out.append(Diagram(strokes, t_in, float(ds.get("dur", 0.22)), depth=depth - 0.004, width=width,
                               stagger=float(ds.get("stagger", 0.11))))
        elif kind == "arrow":
            p0 = np.array(ds["from"], np.float32) + [base["x"], base["y"]]
            p1 = np.array(ds["to"], np.float32) + [base["x"], base["y"]]
            strokes = [s * depth for s in arrow(p0, p1, rng)]
            out.append(Diagram(strokes, t_in, dur, depth=depth - 0.004, width=width, stagger=0.25))
        elif kind == "timeline":
            x0, x1, y = ds["x0"], ds["x1"], ds["y"]
            strokes = [s * D_LOCK for s in timeline(x0, x1, y, int(ds.get("ticks", 12)), rng)]
            out.append(Diagram(strokes, t_in, dur, depth=D_LOCK, width=width, stagger=0.05))
            for text, x in ((ds.get("left", ""), x0), (ds.get("right", ""), x1)):
                if not text:
                    continue
                lab = label(text, 44, 800, res=1.5)
                ent = _enter(build, rng, "drop", 0.4)
                ent["lift"] = 0.0
                add(Layer(lab, D_LOCK, x * D_LOCK, (y + 58) * D_LOCK, 0.0, D_LOCK, t_in=t_in + (0.0 if x == x0 else dur * 0.8),
                          dur_in=0.45, enter=ent, shadow=0.0, page=D_PAGE, name="tl_label"))
    return out
