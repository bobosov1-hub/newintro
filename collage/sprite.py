"""A printed element ready for the compositor: straight RGB + alpha (+ accent coverage)."""
import cv2
import numpy as np


class Sprite:
    def __init__(self, rgb, a, acc=None, res=1.0, anchor=None, meta=None):
        self.rgb = np.ascontiguousarray(rgb, np.float32)
        self.a = np.ascontiguousarray(a, np.float32)
        self.acc = None if acc is None else np.ascontiguousarray(acc, np.float32)
        self.res = float(res)                     # sprite pixels per world unit
        h, w = self.a.shape
        self.anchor = anchor if anchor is not None else (w / 2.0, h / 2.0)
        self.meta = meta or {}
        self._mips = {}

    @property
    def w(self):
        return self.a.shape[1]

    @property
    def h(self):
        return self.a.shape[0]

    def level(self, k):
        """Premultiplied (rgb*a, a) float32 + premultiplied accent, at mip level k (1/2**k)."""
        if k not in self._mips:
            if k == 0:
                pm = np.dstack([self.rgb * self.a[..., None], self.a])
                pacc = None if self.acc is None else self.acc * self.a
            else:
                pm0, pacc0 = self.level(k - 1)
                size = (max(1, pm0.shape[1] // 2), max(1, pm0.shape[0] // 2))
                pm = cv2.resize(pm0, size, interpolation=cv2.INTER_AREA)
                pacc = None if pacc0 is None else cv2.resize(pacc0, size, interpolation=cv2.INTER_AREA)
            self._mips[k] = (np.ascontiguousarray(pm, np.float32),
                             None if pacc is None else np.ascontiguousarray(pacc, np.float32))
        return self._mips[k]

    def world_size(self):
        return self.w / self.res, self.h / self.res


def crop_to_content(rgb, a, acc=None, pad=2):
    ys, xs = np.where(a > 0.003)
    if len(ys) == 0:
        return rgb, a, acc, (0, 0)
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad + 1)
    return (rgb[y0:y1, x0:x1], a[y0:y1, x0:x1], None if acc is None else acc[y0:y1, x0:x1], (x0, y0))
