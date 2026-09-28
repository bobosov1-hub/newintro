"""Depth-primitive renderer: build a height field from simple solids, then light it."""
import cv2
import numpy as np

from ..noise import blur

NEG = -1e6


class Depth:
    """Front-most-wins depth buffer with material ids, rendered supersampled."""

    def __init__(self, w, h, ss=2):
        self.w, self.h, self.ss = w, h, ss
        H, W = h * ss, w * ss
        self.yy, self.xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.yy /= ss
        self.xx /= ss
        self.z = np.full((H, W), NEG, np.float32)
        self.mat = np.zeros((H, W), np.int16)

    def add(self, z, mat):
        m = z > self.z
        self.z[m] = z[m]
        self.mat[m] = mat
        return m

    # primitives (all coordinates in output pixels) --------------------------------------
    def ellipsoid(self, cx, cy, rx, ry, rz, z0=0.0, clip=None):
        u = ((self.xx - cx) / rx) ** 2 + ((self.yy - cy) / ry) ** 2
        z = np.where(u < 1.0, z0 + rz * np.sqrt(np.clip(1.0 - u, 0, 1)), NEG)
        if clip is not None:
            z = np.where(clip, z, NEG)
        return z.astype(np.float32)

    def capsule(self, p0, p1, r, z0=0.0, rz=None):
        rz = r if rz is None else rz
        p0 = np.asarray(p0, np.float32)
        p1 = np.asarray(p1, np.float32)
        d = p1 - p0
        L2 = float((d ** 2).sum())
        t = np.clip(((self.xx - p0[0]) * d[0] + (self.yy - p0[1]) * d[1]) / L2, 0, 1)
        px, py = p0[0] + t * d[0], p0[1] + t * d[1]
        dist = np.sqrt((self.xx - px) ** 2 + (self.yy - py) ** 2)
        return np.where(dist < r, z0 + rz * np.sqrt(np.clip(1 - (dist / r) ** 2, 0, 1)), NEG).astype(np.float32)

    def profile_solid(self, cx, halfwidth_of_y, depth_of_y, z0=0.0, power=0.5):
        """A body of revolution-ish solid: half width and depth given per row (1-D arrays over output rows)."""
        rows = self.yy[:, 0]
        hw = np.interp(rows, np.arange(len(halfwidth_of_y)), halfwidth_of_y).astype(np.float32)[:, None]
        dz = np.interp(rows, np.arange(len(depth_of_y)), depth_of_y).astype(np.float32)[:, None]
        u = np.clip(1.0 - ((self.xx - cx) / np.maximum(hw, 1e-3)) ** 2, 0, 1)
        return np.where((np.abs(self.xx - cx) < hw) & (hw > 1.0), z0 + dz * u ** power, NEG).astype(np.float32)

    def slab(self, poly, z0, bevel=6.0, height=10.0):
        """Flat extruded polygon with a rounded bevel."""
        mask = np.zeros(self.z.shape, np.uint8)
        pts = (np.asarray(poly, np.float32) * self.ss * 4).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1, lineType=cv2.LINE_8, shift=2)
        d = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32) / self.ss
        prof = np.sqrt(np.clip(1 - (1 - np.clip(d / bevel, 0, 1)) ** 2, 0, 1))
        return np.where(mask > 0, z0 + height * prof, NEG).astype(np.float32)

    # output ---------------------------------------------------------------------------------
    def mask(self, mat=None):
        m = (self.mat > 0) if mat is None else np.isin(self.mat, np.atleast_1d(mat))
        return m.astype(np.float32)

    def down(self, img):
        return cv2.resize(img.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_AREA)


def light(dep, albedo, L=(-0.50, -0.62, 0.60), ambient=0.30, wrap=0.3, spec=None, shin=None,
          rim=0.0, ao_sigma=16.0, ao_depth=40.0, slope_clip=3.5):
    """Lambert + wrap, Blinn spec, right-side rim light, cheap ambient occlusion."""
    ss = dep.ss
    inside = dep.mat > 0
    zmin = dep.z[inside].min() if inside.any() else 0.0
    z = np.where(inside, dep.z, zmin - 30.0).astype(np.float32)
    zs = blur(z, 0.8 * ss)
    gx = cv2.Sobel(zs, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * ss
    gy = cv2.Sobel(zs, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * ss
    gx = np.clip(gx, -slope_clip, slope_clip)
    gy = np.clip(gy, -slope_clip, slope_clip)
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    Lv = np.asarray(L, np.float32)
    Lv /= np.linalg.norm(Lv)
    ndl = n @ Lv
    diff = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
    zb = blur(z, ao_sigma * ss)
    ao = 1.0 - 0.65 * np.clip((zb - z) / ao_depth, 0, 1)
    lum = albedo * (ambient * ao + (1 - ambient) * diff * np.sqrt(ao))
    if spec is not None:
        Hv = Lv + np.array([0, 0, 1], np.float32)
        Hv /= np.linalg.norm(Hv)
        ndh = np.clip(n @ Hv, 0, 1)
        lum = lum + spec * ndh ** (shin if shin is not None else 20.0)
    if rim:
        lum = lum + rim * np.clip(1 - n[..., 2], 0, 1) ** 2 * np.clip(n[..., 0], 0, 1) * np.clip(albedo * 2.5, 0.25, 1)
    return lum.astype(np.float32), n


def finish(dep, lum, part_mat=None, gamma=0.85):
    """Downsample to output size; returns (L, alpha, part)."""
    a = dep.down(dep.mask())
    L = dep.down(np.clip(lum, 0, 1.4) * (dep.mat > 0))
    L = np.where(a > 1e-3, L / np.maximum(a, 1e-3), 0.0)
    L = np.clip(L, 0, 1) ** gamma
    part = None if part_mat is None else dep.down(dep.mask(part_mat))
    return L.astype(np.float32), a.astype(np.float32), part
