"""Easing curves and the selective stop-motion cadence."""
import math


def clamp01(t):
    return 0.0 if t <= 0.0 else 1.0 if t >= 1.0 else t


def out_expo(t):
    t = clamp01(t)
    return 1.0 if t >= 1.0 else (1.0 - 2.0 ** (-10.0 * t)) / (1.0 - 2.0 ** -10.0)


def in_out_expo(t):
    t = clamp01(t)
    if t <= 0.0 or t >= 1.0:
        return t
    if t < 0.5:
        return (2.0 ** (20.0 * t - 10.0)) / 2.0
    return (2.0 - 2.0 ** (-20.0 * t + 10.0)) / 2.0


def in_out_sine(t):
    t = clamp01(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def in_out_cubic(t):
    t = clamp01(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def out_expo_settle(t, overshoot=0.06, split=0.58):
    """Ease-out expo that runs slightly past the target, then settles back (C1-continuous)."""
    t = clamp01(t)
    if t < split:
        return (1.0 + overshoot) * out_expo(t / split)
    u = (t - split) / (1.0 - split)
    return 1.0 + overshoot * (1.0 - in_out_sine(u))


def stepped(local_t, step_fps):
    """Quantise an entrance's local time to stop-motion steps (e.g. 12 fps)."""
    if local_t <= 0.0:
        return local_t
    return math.floor(local_t * step_fps) / step_fps
