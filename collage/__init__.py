"""Analog editorial-collage title system (Vox-style explainer interstitials).

Everything that looks random is drawn from `Build.rng(...)`, so a (seed, chaos) pair always
reproduces the identical frame.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.path.join(ROOT, "fonts")
DATA = os.path.join(ROOT, "data")
