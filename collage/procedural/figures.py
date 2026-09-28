"""Back-view figures (no faces): the village head and the young clerk.

Rendered as a lit height field so that, after the black-and-white print pass, they read as
photographs of real people cut out of a page.
"""
import numpy as np

from ..noise import blur, fbm, noise1d, smoothstep, streaks
from .shading import NEG, Depth, finish, light

SKIN, HAIR, JACKET, SHIRT, CAP, BLOUSE, VEST, BUTTON = 1, 2, 3, 4, 5, 6, 7, 8


def _folds(dep, rng, region, n, amp, width, angle_range=(-40, 40), center=None, spread=(300, 400)):
    """Soft fabric folds: gaussian ridges/valleys along random line segments."""
    add = np.zeros_like(dep.z)
    cx, cy = center if center is not None else (dep.w / 2, dep.h * 0.7)
    for _ in range(n):
        x = cx + rng.uniform(-spread[0], spread[0])
        y = cy + rng.uniform(-spread[1], spread[1])
        a = np.deg2rad(rng.uniform(*angle_range))
        ln = rng.uniform(80, 260)
        wdt = rng.uniform(width * 0.6, width * 1.4)
        s = rng.choice([-1, 1]) * rng.uniform(0.5, 1.0) * amp
        dx, dy = np.cos(a), np.sin(a)
        t = (dep.xx - x) * dx + (dep.yy - y) * dy
        d = -(dep.xx - x) * dy + (dep.yy - y) * dx
        along = np.clip(1 - (t / (ln / 2)) ** 2, 0, 1)
        add += s * np.exp(-(d / wdt) ** 2) * along
    dep.z += np.where(region, add, 0).astype(np.float32)


def man_back(variant="suit", seed=1, w=900, h=1150, ss=2, grey_hair=0.45):
    """Middle-aged man seen from behind, waist-up. variant: 'suit' | 'jumper' (with cap)."""
    rng = np.random.default_rng(seed)
    dep = Depth(w, h, ss)
    X, Y = dep.xx, dep.yy
    x0 = w / 2

    rows = np.arange(h, dtype=np.float32)
    s = np.clip((rows - 398) / 125.0, 0, 1)
    hw = 66 + 190 * (1 - (1 - s) ** 3) - 18 * smoothstep(600, 1150, rows)
    hw[rows < 392] = 0
    dz = (40 + 100 * smoothstep(392, 500, rows) ** 0.7).astype(np.float32)
    torso = dep.profile_solid(x0, hw, dz, z0=30.0, power=0.55)
    dep.add(torso, JACKET)
    for sgn in (-1, 1):
        dep.add(dep.ellipsoid(x0 + sgn * 232, 548, 74, 64, 62, z0=38), JACKET)
        dep.add(dep.capsule((x0 + sgn * 258, 575), (x0 + sgn * 286, 1250), 60, z0=24, rz=58), JACKET)
    # neck, ears, head
    dep.add(dep.capsule((x0, 300), (x0, 440), 56, z0=46), SKIN)
    for sgn in (-1, 1):
        dep.add(dep.ellipsoid(x0 + sgn * 95, 262, 19, 40, 16, z0=58), SKIN)
    head = dep.ellipsoid(x0, 240, 98, 124, 92, z0=70)
    dep.add(head, SKIN)
    # hair: nape hairline lower at the centre, rising towards the ears
    dx = X - x0
    hairline = 334 - 44 * (dx / 92.0) ** 2
    hair = dep.ellipsoid(x0, 238, 104, 129, 97, z0=70, clip=(Y < hairline))
    dep.add(hair, HAIR)
    # collar: shirt band above a jacket band, curving down to the sides
    yc = 398 + 0.0032 * dx ** 2
    band_j = (np.abs(Y - yc) < 19) & (np.abs(dx) < 104) & (Y > 360)
    band_s = (Y > yc - 36) & (Y < yc - 17) & (np.abs(dx) < 74)
    base = np.where(dep.mat > 0, dep.z, 60.0)
    dep.add(np.where(band_s, np.maximum(base, 130.0) + 6, NEG).astype(np.float32), SHIRT)
    dep.add(np.where(band_j, np.maximum(base, 132.0) + 14 * np.sqrt(np.clip(1 - ((Y - yc) / 19) ** 2, 0, 1)), NEG).astype(np.float32), JACKET)

    if variant == "jumper":
        cap_clip = (Y < 262 - 0.002 * dx ** 2)
        cap = dep.ellipsoid(x0, 212, 110, 108, 100, z0=74, clip=cap_clip)
        opening = (np.abs(dx) < 30) & (Y > 214) & (Y < 262)
        cap = np.where(opening & ~((Y > 240) & (Y < 250)), NEG, cap)
        dep.add(cap.astype(np.float32), CAP)

    jacket = dep.mat == JACKET
    # seams and folds
    dep.z -= np.where(jacket & (Y > 430), 3.5 * np.exp(-(dx / 2.2) ** 2), 0).astype(np.float32)
    if variant == "jumper":
        yoke = 520 + 0.0006 * dx ** 2
        dep.z -= np.where(jacket, 3.0 * np.exp(-((Y - yoke) / 2.5) ** 2), 0).astype(np.float32)
        _folds(dep, rng, jacket, 16, 12.0, 20.0, (-35, 35), (x0, 820), (300, 330))
        _folds(dep, rng, jacket, 8, 8.0, 14.0, (60, 120), (x0, 900), (320, 250))
    else:
        _folds(dep, rng, jacket, 10, 6.0, 22.0, (-30, 30), (x0, 850), (260, 280))
        _folds(dep, rng, jacket, 6, 5.0, 12.0, (70, 110), (x0, 950), (320, 200))
    # hair strands (displacement) + fanning direction
    left = streaks(dep.z.shape[0], dep.z.shape[1], rng, 9 * ss, angle_deg=112)
    right = streaks(dep.z.shape[0], dep.z.shape[1], rng, 9 * ss, angle_deg=68)
    wmix = smoothstep(-40, 40, dx)
    hs = left * (1 - wmix) + right * wmix
    hairm = dep.mat == HAIR
    dep.z += np.where(hairm, hs * 1.6, 0).astype(np.float32)

    # albedo
    alb = np.zeros_like(dep.z)
    fine = blur(rng.standard_normal(dep.z.shape).astype(np.float32), 0.6 * ss)
    alb[dep.mat == SKIN] = 0.60
    grey = np.clip((hs - 0.9) * 0.9, 0, 1) * smoothstep(40, 95, np.abs(dx)) * grey_hair
    alb = np.where(hairm, 0.085 + 0.33 * grey + 0.02 * fine, alb)
    jac_alb = 0.24 if variant == "suit" else 0.40
    alb = np.where(jacket, jac_alb * (1 + 0.05 * fine) * (1 + 0.04 * fbm(*dep.z.shape, rng, 60 * ss, 3)), alb)
    alb[dep.mat == SHIRT] = 0.86
    alb[dep.mat == CAP] = 0.27

    spec = np.zeros_like(dep.z)
    shin = np.full_like(dep.z, 20.0)
    spec = np.where(hairm, 0.22 * np.clip(0.6 + 0.5 * hs, 0, 1.5), spec)
    shin = np.where(hairm, 28.0, shin)
    if variant == "jumper":
        spec = np.where(jacket, 0.10, spec)
        shin = np.where(jacket, 9.0, shin)
    else:
        spec = np.where(jacket, 0.03, spec)
    lum, _ = light(dep, alb, spec=spec, shin=shin, rim=0.28, ao_sigma=14, ao_depth=30)
    return finish(dep, lum, part_mat=[JACKET, CAP] if variant == "jumper" else [JACKET])


def woman_back(seed=2, w=900, h=1150, ss=2):
    """Young woman seen from behind: shoulder-length hair, blouse and uniform vest."""
    rng = np.random.default_rng(seed)
    dep = Depth(w, h, ss)
    X, Y = dep.xx, dep.yy
    x0 = w / 2
    dx = X - x0

    rows = np.arange(h, dtype=np.float32)
    s = np.clip((rows - 428) / 115.0, 0, 1)
    hw = 62 + 158 * (1 - (1 - s) ** 3) - 22 * smoothstep(620, 940, rows) + 24 * smoothstep(960, 1150, rows)
    hw[rows < 422] = 0
    dz = (36 + 88 * smoothstep(422, 520, rows) ** 0.7).astype(np.float32)
    dep.add(dep.profile_solid(x0, hw, dz, z0=28.0, power=0.55), BLOUSE)
    for sgn in (-1, 1):
        dep.add(dep.ellipsoid(x0 + sgn * 200, 560, 58, 54, 50, z0=34), BLOUSE)
        dep.add(dep.capsule((x0 + sgn * 220, 585), (x0 + sgn * 240, 1250), 47, z0=22, rz=46), BLOUSE)
    dep.add(dep.capsule((x0, 330), (x0, 470), 44, z0=40), SKIN)
    # vest over the back: narrow straps at the shoulders widening under the arms
    rowsY = Y
    hwY = np.interp(rowsY[:, 0], rows, hw).astype(np.float32)[:, None]
    xv = 112 + (hwY - 8 - 112) * smoothstep(470, 720, rowsY)
    vest = (np.abs(dx) < xv) & (dep.mat == BLOUSE) & (Y > 440) & (np.abs(dx) < hwY - 4)
    dep.mat[vest] = VEST
    dep.z += np.where(vest, 3.0, 0).astype(np.float32)
    # half belt with two buttons
    belt = vest & (Y > 905) & (Y < 942) & (np.abs(dx) < 118)
    dep.z += np.where(belt, 4.0 * np.sqrt(np.clip(1 - ((Y - 923.5) / 18.5) ** 2, 0, 1)), 0).astype(np.float32)
    for sgn in (-1, 1):
        b = dep.ellipsoid(x0 + sgn * 100, 923, 9, 9, 5, z0=0)
        b = np.where(b > NEG / 2, dep.z + b, NEG)
        dep.add(b.astype(np.float32), BUTTON)
    dep.z -= np.where(vest & (Y > 470), 3.0 * np.exp(-(dx / 2.0) ** 2), 0).astype(np.float32)
    # head + hair volume (covers the neck)
    dep.add(dep.ellipsoid(x0, 252, 90, 114, 86, z0=72), SKIN)
    hb = 552 + 18 * noise1d(dep.z.shape[1], rng, 60 * ss)[None, :] + 9 * np.sin(X / 13.0) \
        + 14 * np.clip(noise1d(dep.z.shape[1], rng, 6 * ss)[None, :], -2, 2)
    hwh = 100 + 34 * smoothstep(250, 540, Y)
    curtain = np.where((Y > 245) & (Y < hb) & (np.abs(dx) < hwh),
                       74 + 70 * np.sqrt(np.clip(1 - (dx / hwh) ** 2, 0, 1)) - 12 * smoothstep(420, 560, Y), NEG)
    dep.add(curtain.astype(np.float32), HAIR)
    dep.add(dep.ellipsoid(x0, 246, 101, 126, 94, z0=74), HAIR)
    hairm = dep.mat == HAIR
    wave = 4.0 * np.sin(Y / 38.0 + dx / 60.0)
    st = streaks(dep.z.shape[0], dep.z.shape[1], rng, 26 * ss, angle_deg=90)
    st = np.roll(st, 3, axis=1) * 0.5 + st * 0.5
    dep.z += np.where(hairm, st * 1.8 + wave, 0).astype(np.float32)

    blouse = np.isin(dep.mat, [BLOUSE])
    _folds(dep, rng, blouse, 10, 5.0, 12.0, (60, 120), (x0, 850), (260, 250))
    _folds(dep, rng, vest, 7, 4.0, 18.0, (-25, 25), (x0, 800), (170, 180))

    fine = blur(rng.standard_normal(dep.z.shape).astype(np.float32), 0.6 * ss)
    alb = np.zeros_like(dep.z)
    alb[dep.mat == SKIN] = 0.62
    alb = np.where(hairm, 0.075 + 0.03 * np.clip(st, -1, 2), alb)
    alb = np.where(blouse, 0.84 * (1 + 0.03 * fine), alb)
    alb = np.where(dep.mat == VEST, 0.23 * (1 + 0.06 * fine), alb)
    alb[dep.mat == BUTTON] = 0.45
    spec = np.where(hairm, 0.30 * np.clip(0.55 + 0.45 * st, 0, 1.6), 0.02).astype(np.float32)
    spec = np.where(dep.mat == BUTTON, 0.4, spec)
    shin = np.where(hairm, 34.0, 16.0).astype(np.float32)
    lum, _ = light(dep, alb, spec=spec, shin=shin, rim=0.25, ao_sigma=14, ao_depth=28)
    return finish(dep, lum, part_mat=[VEST])
