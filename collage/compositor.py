"""Layer compositing with real parallax, separated blurs and contact shadows.

* focal defocus: gaussian, from the circle of confusion between layer depth and focus depth
* directional blur: accumulation of sub-frame warps along the actual motion (velocity only)
  - the two never stand in for each other
* stop-motion cadence: layer entrances/exits step at `step_fps`; the camera never does
"""
import math

import cv2
import numpy as np

from . import palette as P
from .easing import out_expo_settle, stepped

LIGHT = np.array([0.52, 0.85], np.float32)          # shadows fall down-right (key light up-left)
LIGHT /= np.linalg.norm(LIGHT)


def in_expo(t):
    t = min(max(t, 0.0), 1.0)
    return 0.0 if t <= 0 else 2.0 ** (10.0 * t - 10.0)


class Layer:
    def __init__(self, sprite, depth, x=0.0, y=0.0, rot=0.0, scale=1.0, t_in=-1e9, dur_in=0.0, enter=None,
                 stepped_in=True, t_out=None, dur_out=0.4, exit=None, stepped_out=True, shadow=1.0,
                 name="", order=0, opacity=1.0, page=None):
        self.sprite, self.depth = sprite, float(depth)
        self.x, self.y, self.rot, self.scale = float(x), float(y), float(rot), float(scale)
        self.t_in, self.dur_in, self.enter, self.stepped_in = t_in, dur_in, enter, stepped_in
        self.t_out, self.dur_out, self.exit, self.stepped_out = t_out, dur_out, exit, stepped_out
        self.shadow, self.name, self.order, self.opacity = shadow, name, order, opacity
        self.page = page                                 # depth of the surface it rests on

    def state(self, t, step_fps):
        if t < self.t_in:
            return None
        x, y, rot, sc, lift, op = self.x, self.y, self.rot, self.scale, 0.0, self.opacity
        moving = False
        if self.enter and self.dur_in > 0:
            te = t - self.t_in
            if te < self.dur_in:
                moving = True
                if self.stepped_in:
                    te = stepped(te, step_fps)
                p = out_expo_settle(te / self.dur_in, self.enter.get("overshoot", 0.06))
                k = 1.0 - p
                x += self.enter.get("dx", 0.0) * k
                y += self.enter.get("dy", 0.0) * k
                rot += self.enter.get("drot", 0.0) * k
                sc *= 1.0 + self.enter.get("dscale", 0.0) * k
                lift += self.enter.get("lift", 0.0) * max(k, 0.0)
        if self.t_out is not None and t >= self.t_out:
            te = t - self.t_out
            if self.stepped_out:
                te = stepped(te, step_fps)
            q = in_expo(te / self.dur_out) if self.dur_out > 0 else 1.0
            if te >= self.dur_out:
                return None
            moving = True
            ex = self.exit or {}
            x += ex.get("dx", 0.0) * q
            y += ex.get("dy", 0.0) * q
            rot += ex.get("drot", 0.0) * q
            sc *= 1.0 + ex.get("dscale", 0.0) * q
            lift += ex.get("lift", 0.0) * q
            op *= 1.0 - ex.get("fade", 0.0) * q
        return x, y, rot, sc, lift, op, moving


def layer_matrix(layer, st, cam, W, H, vs=1.0):
    """3x3 matrix: sprite pixels -> screen pixels (vs = view scale for low-res previews)."""
    x, y, rot, sc, lift = st[:5]
    cx, cy, cz, roll = cam
    sp = layer.sprite
    d = layer.depth - cz - lift * 0.0
    s_p = 1.0 / max(d, 0.05)
    ax, ay = sp.anchor
    k = sc / sp.res
    r = math.radians(rot)
    cr, sr = math.cos(r), math.sin(r)
    # local = R * k * (p - anchor) + (x, y)
    A = np.array([[cr * k, -sr * k, x - (cr * k * ax - sr * k * ay)],
                  [sr * k, cr * k, y - (sr * k * ax + cr * k * ay)],
                  [0, 0, 1]], np.float64)
    Cm = np.array([[s_p, 0, -cx * s_p], [0, s_p, -cy * s_p], [0, 0, 1]], np.float64)
    rr = math.radians(roll)
    c2, s2 = math.cos(rr), math.sin(rr)
    R = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]], np.float64)
    T = np.array([[vs, 0, W / 2.0], [0, vs, H / 2.0], [0, 0, 1]], np.float64)
    return T @ R @ Cm @ A


def _corners(M, w, h):
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
    q = M @ pts
    return q[:2].T


class Compositor:
    def __init__(self, W=1920, H=1080, fps=30.0, step_fps=12.0, dof=9.0, shutter=0.5, vs=1.0):
        self.W, self.H, self.fps, self.step_fps = W, H, fps, step_fps
        self.dof, self.shutter, self.vs = dof * vs, shutter, vs

    # ------------------------------------------------------------------ warping
    def _warp(self, sprite, mats, pad):
        s_tot = math.sqrt(abs(np.linalg.det(mats[0][:2, :2])))
        k = 0
        while s_tot * (2 ** k) < 0.62 and k < 5:
            k += 1
        pm, pacc = sprite.level(k)
        S = np.diag([2.0 ** k, 2.0 ** k, 1.0])
        mats = [M @ S for M in mats]
        h, w = pm.shape[:2]
        pts = np.concatenate([_corners(M, w, h) for M in mats])
        x0 = int(math.floor(pts[:, 0].min())) - pad
        y0 = int(math.floor(pts[:, 1].min())) - pad
        x1 = int(math.ceil(pts[:, 0].max())) + pad
        y1 = int(math.ceil(pts[:, 1].max())) + pad
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.W), min(y1, self.H)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        rw, rh = x1 - x0, y1 - y0
        acc_pm = None
        acc_c = None
        for M in mats:
            Mr = M.copy()
            Mr[0, 2] -= x0
            Mr[1, 2] -= y0
            o = cv2.warpAffine(pm, Mr[:2], (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            acc_pm = o if acc_pm is None else acc_pm + o
            if pacc is not None:
                oc = cv2.warpAffine(pacc, Mr[:2], (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                acc_c = oc if acc_c is None else acc_c + oc
        n = float(len(mats))
        if n > 1:
            acc_pm /= n
            if acc_c is not None:
                acc_c /= n
        return acc_pm, acc_c, (x0, y0, x1, y1)

    # ------------------------------------------------------------------ shadows
    def _shadow(self, canvas, a, box, height, strength):
        if strength <= 0 or height <= -0.5:
            return
        x0, y0, x1, y1 = box
        vs = self.vs
        for (off, sig, op, f) in ((vs * (6.0 + 110.0 * height), vs * (5.0 + 75.0 * height), 0.30, 4),
                                  (vs * (1.5 + 8.0 * height), vs * (1.2 + 5.0 * height), 0.24, 2)):
            op *= strength
            h, w = a.shape
            sw, sh = max(1, w // f), max(1, h // f)
            small = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
            pad = int((off + 3 * sig) / f) + 2
            small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            M = np.float32([[1, 0, off * LIGHT[0] / f], [0, 1, off * LIGHT[1] / f]])
            small = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]))
            small = cv2.GaussianBlur(small, (0, 0), max(sig / f, 0.3))
            big = cv2.resize(small, (small.shape[1] * f, small.shape[0] * f), interpolation=cv2.INTER_LINEAR)
            gx0, gy0 = x0 - pad * f, y0 - pad * f
            cx0, cy0 = max(gx0, 0), max(gy0, 0)
            cx1, cy1 = min(gx0 + big.shape[1], self.W), min(gy0 + big.shape[0], self.H)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            sh_ = big[cy0 - gy0:cy1 - gy0, cx0 - gx0:cx1 - gx0]
            canvas[cy0:cy1, cx0:cx1] *= (1.0 - op * np.clip(sh_, 0, 1))[..., None]

    # ------------------------------------------------------------------ main
    def render(self, layers, camera, t, canvas=None, acc=None, diagrams=()):
        W, H = self.W, self.H
        if canvas is None:
            canvas = np.zeros((H, W, 3), np.float32)
            canvas[:] = P.PAPER
        if acc is None:
            acc = np.zeros((H, W), np.float32)
        cam = camera.pose(t)
        focus = camera.focus(t)
        dt = self.shutter / self.fps
        cam_prev = camera.pose(t - dt)
        under = sorted([d for d in diagrams if d.z is not None], key=lambda d: d.z)
        for layer in sorted(layers, key=lambda l: l.order):
            while under and under[0].z < layer.order:
                under.pop(0).draw(canvas, acc, t, cam, self, layers)
            st = layer.state(t, self.step_fps)
            if st is None:
                continue
            if hasattr(layer, "update"):
                layer.update(t)
            M = layer_matrix(layer, st, cam, W, H, self.vs)
            st_prev = layer.state(t - dt, self.step_fps) or st
            Mp = layer_matrix(layer, st_prev, cam_prev, W, H, self.vs)
            sp = layer.sprite
            c_now = _corners(M, sp.w, sp.h)
            disp = float(np.abs(c_now - _corners(Mp, sp.w, sp.h)).max())
            mats = [M]
            if disp > 1.5:
                n = int(min(9, 2 + disp / 3.0))
                if st[6] and layer.stepped_in:
                    # stop-motion smear frame: a short streak along the jump
                    mats = [M + (Mp - M) * (0.30 * j / (n - 1)) for j in range(n)]
                else:
                    mats = []
                    for j in range(n):
                        tj = t - dt * j / (n - 1)
                        sj = layer.state(tj, self.step_fps) or st
                        mats.append(layer_matrix(layer, sj, camera.pose(tj), W, H, self.vs))
            d_eff = layer.depth - cam[2]
            f_eff = focus - cam[2]
            sigma = self.dof * abs(1.0 / max(d_eff, 0.05) - 1.0 / max(f_eff, 0.05))
            pad = int(3 * sigma) + 2
            res = self._warp(sp, mats, pad)
            if res is None:
                continue
            pm, pacc, box = res
            if sigma > 0.35:
                pm = cv2.GaussianBlur(pm, (0, 0), sigma)
                if pacc is not None:
                    pacc = cv2.GaussianBlur(pacc, (0, 0), sigma)
            if st[5] < 1.0:
                pm = pm * st[5]
                if pacc is not None:
                    pacc = pacc * st[5]
            x0, y0, x1, y1 = box
            a = pm[..., 3]
            if layer.shadow > 0 and layer.page is not None:
                height = max(0.0, layer.page - layer.depth) + st[4]
                self._shadow(canvas, a, box, height, layer.shadow)
            region = canvas[y0:y1, x0:x1]
            region *= (1.0 - a)[..., None]
            region += pm[..., :3]
            ar = acc[y0:y1, x0:x1]
            ar *= (1.0 - a)
            if pacc is not None:
                ar += pacc
        for dg in under:
            dg.draw(canvas, acc, t, cam, self, layers)
        for dg in diagrams:
            if dg.z is None:
                dg.draw(canvas, acc, t, cam, self, layers)
        return canvas, acc
