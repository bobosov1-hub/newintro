"""The four locked values. Every pixel the system produces is a mixture of these."""
import numpy as np

INK = np.array([23, 21, 19], np.float32) / 255.0        # near-black ink
PAPER = np.array([236, 230, 216], np.float32) / 255.0   # warm off-white paper (never #FFFFFF)
GREY = np.array([139, 136, 131], np.float32) / 255.0    # mid neutral grey
ACCENT = np.array([226, 78, 42], np.float32) / 255.0    # the single red-orange accent

HEX = {"ink": "#171513", "paper": "#ECE6D8", "grey": "#8B8883", "accent": "#E24E2A"}


def tone(L):
    """Map a 0..1 luminance map onto the ink->paper ramp (black-and-white print)."""
    return INK + (PAPER - INK) * L[..., None]


def duotone(L):
    """Hero duotone: shadows -> ink, highlights -> red-orange."""
    return INK + (ACCENT - INK) * L[..., None]


def mix(a, b, t):
    return a + (b - a) * t
