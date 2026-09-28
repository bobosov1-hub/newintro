"""Fallback objects and places (used only when no photograph with the same name is supplied):
sluice gate, broken concrete, soban meal table, rural house, mountain ridges."""
import cv2
import numpy as np

from ..noise import blur, fbm, noise1d, smoothstep, streaks, value_noise
from .shading import NEG, Depth, finish, light


def _poly(shape, pts, ss=4):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [(np.asarray(pts, np.float32) * ss).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    return m.astype(np.float32) / 255.0


def _rect(shape, x0, y0, x1, y1):
    return _poly(shape, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _concrete_tex(h, w, rng, base=0.62):
    t = base + 0.06 * fbm(h, w, rng, 60.0, 5) + 0.03 * fbm(h, w, rng, 8.0, 3)
    pits = (rng.random((h, w)) < 0.004).astype(np.float32)
    t -= blur(pits, 0.9) * 0.9
    drips = streaks(h, w, rng, 60, 90) * 0.035
    return t + drips


def _cracks(shape, rng, n, x_range, y_range, length=(60, 200)):
    m = np.zeros(shape, np.float32)
    for _ in range(n):
        x, y = rng.uniform(*x_range), rng.uniform(*y_range)
        a = rng.uniform(0, np.pi)
        pts = [(x, y)]
        for _ in range(int(rng.uniform(*length) / 8)):
            a += rng.normal(0, 0.35)
            x, y = x + 8 * np.cos(a), y + 8 * np.sin(a)
            pts.append((x, y))
        cv2.polylines(m, [(np.array(pts) * 4).astype(np.int32)], False, 1.0, 2, cv2.LINE_AA, shift=2)
    return blur(m, 0.6)


def sluice_gate(seed=31, W=900, H=1000):
    """Reservoir sluice gate, straight-on (part = the steel gate, spindle and hand wheel)."""
    rng = np.random.default_rng(seed)
    shp = (H, W)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    L = np.zeros(shp, np.float32)
    a = np.zeros(shp, np.float32)
    part = np.zeros(shp, np.float32)
    conc = _concrete_tex(H, W, rng, 0.64)
    # piers
    for x0, x1 in ((90, 255), (645, 810)):
        m = _rect(shp, x0, 300, x1, 880)
        shade = 1.0 - 0.18 * smoothstep(x0, x1, xx) if x0 < 400 else 0.88 + 0.12 * smoothstep(x0, x1, xx)
        moss = smoothstep(700, 860, yy) * 0.22 * (0.7 + 0.3 * value_noise(H, W, rng, 20.0))
        L = L * (1 - m) + (conc * shade - moss) * m
        a = np.maximum(a, m)
    cr = _cracks(shp, rng, 5, (100, 800), (320, 820))
    # gate plate with channels, ribs and rivets
    g = _rect(shp, 255, 330, 645, 870)
    steel = 0.42 + 0.05 * fbm(H, W, rng, 30.0, 4) - 0.10 * np.clip(streaks(H, W, rng, 90, 90), 0, 3) * 0.3
    ribs = np.zeros(shp, np.float32)
    for yr in range(390, 860, 78):
        top = smoothstep(yr - 1, yr + 1, yy) * (1 - smoothstep(yr + 14, yr + 16, yy))
        ribs += top * (0.20 * (1 - smoothstep(yr, yr + 6, yy)) - 0.12 * smoothstep(yr + 9, yr + 15, yy))
        for xr in range(280, 630, 44):
            cv2.circle(ribs, (xr, yr + 7), 3, 0.25, -1, cv2.LINE_AA)
    chan = (_rect(shp, 255, 330, 272, 870) + _rect(shp, 628, 330, 645, 870)) * -0.18
    Lg = steel + ribs + chan
    L = L * (1 - g) + Lg * g
    a = np.maximum(a, g)
    part = np.maximum(part, g)
    # deck + top face + railing
    deck = _rect(shp, 40, 232, 860, 305)
    L = L * (1 - deck) + (conc * (0.95 - 0.10 * smoothstep(232, 305, yy))) * deck
    topf = _rect(shp, 50, 214, 850, 234)
    L = L * (1 - topf) + (conc * 1.12) * topf
    a = np.maximum(a, np.maximum(deck, topf))
    rail = np.zeros(shp, np.float32)
    for xp in range(70, 860, 110):
        cv2.line(rail, (xp, 214), (xp, 112), 1.0, 8, cv2.LINE_AA)
    for yr in (118, 164):
        cv2.line(rail, (62, yr), (838, yr), 1.0, 6, cv2.LINE_AA)
    rail = blur(rail, 0.6)
    L = L * (1 - rail) + (0.30 + 0.10 * (1 - smoothstep(0, 6, np.abs(yy - 118)))) * rail
    a = np.maximum(a, rail)
    # hoist: gearbox, spindle, hand wheel
    box = _rect(shp, 392, 170, 508, 232)
    L = L * (1 - box) + (0.36 + 0.10 * (1 - smoothstep(170, 232, yy))) * box
    spin = _rect(shp, 444, 18, 456, 232)
    L = L * (1 - spin) + (0.30 + 0.25 * (1 - np.abs(xx - 450) / 6)) * spin
    wheel = np.zeros(shp, np.float32)
    cv2.ellipse(wheel, (450 * 4, 118 * 4), (112 * 4, 34 * 4), 0, 0, 360, 1.0, 40, cv2.LINE_AA, shift=2)
    for k in range(6):
        ang = np.deg2rad(k * 60 + 15)
        cv2.line(wheel, (450 * 4, 118 * 4), (int((450 + 108 * np.cos(ang)) * 4), int((118 + 32 * np.sin(ang)) * 4)),
                 1.0, 20, cv2.LINE_AA, shift=2)
    cv2.ellipse(wheel, (450 * 4, 118 * 4), (22 * 4, 12 * 4), 0, 0, 360, 1.0, -1, cv2.LINE_AA, shift=2)
    wheel = blur(wheel, 0.5)
    L = L * (1 - wheel) + (0.26 + 0.18 * (yy < 112)) * wheel
    hoist = np.maximum(np.maximum(box, spin), wheel)
    a = np.maximum(a, hoist)
    part = np.maximum(part, np.maximum(spin, wheel))
    # water in the channel below
    wat = _rect(shp, 40, 862, 860, 1000)
    rip = 0.5 + 0.5 * np.sin(yy * 0.9 + 3 * value_noise(H, W, rng, 25.0))
    Lw = 0.12 + 0.10 * rip * smoothstep(862, 1000, yy) + 0.06 * value_noise(H, W, rng, 12.0)
    L = L * (1 - wat) + Lw * wat
    a = np.maximum(a, wat)
    L = L * (1 - cr * 0.5 * a)
    return np.clip(L, 0, 1), np.clip(a, 0, 1), np.clip(part, 0, 1)


def concrete(seed=32, W=900, H=700):
    """A broken chunk of old concrete with exposed rebar."""
    rng = np.random.default_rng(seed)
    dep = Depth(W, H, 2)
    X, Y = dep.xx, dep.yy
    cx, cy = 420.0, 400.0
    ang = np.arctan2(Y - cy, X - cx)
    n = 64
    base_r = 290 + 70 * noise1d(n + 1, rng, 6.0, 3)
    base_r[-1] = base_r[0]
    theta = np.linspace(-np.pi, np.pi, n + 1)
    r = np.interp(ang, theta, base_r)
    r = r * (1.0 - 0.28 * np.clip(np.sin(ang), 0, 1) ** 3)          # flatter underside
    jag = 10 * fbm(dep.z.shape[0], dep.z.shape[1], rng, 30.0, 4)
    d = np.sqrt((X - cx) ** 2 + ((Y - cy) * 1.35) ** 2)
    inside = d < (r + jag)
    facets = fbm(dep.z.shape[0], dep.z.shape[1], rng, 120.0, 2)
    z = np.where(inside, 60 + 170 * np.sqrt(np.clip(1 - (d / (r + jag)) ** 2, 0, 1))
                 + 26 * np.round(facets * 1.6) / 1.6 + 6 * fbm(dep.z.shape[0], dep.z.shape[1], rng, 10.0, 3), NEG)
    dep.add(z.astype(np.float32), 1)
    # aggregate and rebar
    alb = np.where(dep.mat == 1, 0.55 + 0.05 * fbm(dep.z.shape[0], dep.z.shape[1], rng, 40.0, 3), 0).astype(np.float32)
    stones = np.zeros(dep.z.shape, np.float32)
    for _ in range(260):
        sx, sy = rng.uniform(0, dep.z.shape[1]), rng.uniform(0, dep.z.shape[0])
        cv2.ellipse(stones, (int(sx), int(sy)), (int(rng.uniform(4, 16)), int(rng.uniform(3, 11))),
                    float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(-1, 1)), -1, cv2.LINE_AA)
    alb = alb + 0.16 * blur(stones, 1.0) * (dep.mat == 1)
    for (p0, p1, rr) in (((620, 330), (880, 150), 16), ((560, 260), (760, 230), 12), ((600, 470), (800, 420), 13)):
        cap = dep.capsule(p0, p1, rr, z0=150, rz=rr)
        m = dep.add(cap, 2)
        t = ((X - p0[0]) * (p1[0] - p0[0]) + (Y - p0[1]) * (p1[1] - p0[1])) / float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        ribs = 0.5 + 0.5 * np.sin(t * 0.55)
        alb = np.where(dep.mat == 2, 0.20 + 0.10 * ribs + 0.05 * fbm(dep.z.shape[0], dep.z.shape[1], rng, 8.0, 2), alb)
    lum, _ = light(dep, alb, spec=np.where(dep.mat == 2, 0.08, 0.02).astype(np.float32), shin=12.0, rim=0.15,
                   ao_sigma=10, ao_depth=25)
    cr = _cracks(dep.z.shape, rng, 9, (200, 1500), (300, 1100), (80, 260))
    lum = lum * (1 - cr * 0.6 * (dep.mat == 1))
    return finish(dep, lum, None)


def soban(seed=33, W=1000, H=760):
    """Round low table with a home-cooked meal, three-quarter view (part = the table top)."""
    rng = np.random.default_rng(seed)
    dep = Depth(W, H, 2)
    X, Y = dep.xx, dep.yy
    cx, cy, rx, ry = 500.0, 330.0, 420.0, 168.0
    u = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
    top = np.where(u < 1, 100.0, NEG).astype(np.float32)
    dep.add(top, 1)
    # thickness band + apron
    band = (((X - cx) / rx) ** 2 + ((Y - cy - 36) / ry) ** 2 < 1) & (Y > cy) & (u >= 1)
    dep.add(np.where(band, 90.0 - 0.2 * (Y - cy), NEG).astype(np.float32), 2)
    for (x0, y0, x1, y1, bend) in ((175, 470, 150, 720, -40), (825, 470, 850, 720, 40), (300, 470, 310, 620, -20), (700, 470, 690, 620, 20)):
        t = np.linspace(0, 1, 20)
        xs = x0 + (x1 - x0) * t + bend * np.sin(np.pi * t)
        ys = y0 + (y1 - y0) * t
        for i in range(19):
            dep.add(dep.capsule((xs[i], ys[i]), (xs[i + 1], ys[i + 1]), 22 - 8 * t[i], z0=40 if x0 in (300, 700) else 70), 2)
    # dishes: (x, y, rx, ry, height, kind)
    dishes = [(390, 300, 74, 30, 58, "rice"), (575, 318, 84, 33, 52, "soup"), (250, 245, 58, 23, 20, "kimchi"),
              (455, 215, 56, 22, 20, "namul"), (660, 232, 58, 23, 20, "egg"), (780, 290, 55, 22, 20, "fish")]
    for (x, y, bx, by, hgt, kind) in dishes:
        body = dep.ellipsoid(x, y + hgt * 0.35, bx * 0.92, by * 1.0 + hgt * 0.45, 40, z0=110)
        body = np.where(Y > y, body, NEG).astype(np.float32)
        dep.add(body, 3)
        rim = dep.ellipsoid(x, y, bx, by, 6, z0=150)
        dep.add(rim, 4)
        fill = dep.ellipsoid(x, y - (10 if kind == "rice" else 2), bx * 0.84, by * 0.78, 28 if kind == "rice" else 8, z0=152)
        dep.add(fill, {"rice": 5, "soup": 6}.get(kind, 7))
    # spoon + chopsticks
    dep.add(dep.capsule((810, 385), (905, 330), 6, z0=110), 8)
    dep.add(dep.ellipsoid(800, 392, 20, 12, 5, z0=110), 8)
    dep.add(dep.capsule((835, 400), (935, 345), 4, z0=110), 8)
    dep.add(dep.capsule((848, 404), (945, 352), 4, z0=110), 8)
    grain = streaks(dep.z.shape[0], dep.z.shape[1], rng, 70, 3.0)
    alb = np.zeros(dep.z.shape, np.float32)
    alb[dep.mat == 1] = 0.30
    alb = np.where(dep.mat == 1, 0.30 + 0.05 * grain + 0.04 * fbm(dep.z.shape[0], dep.z.shape[1], rng, 90.0, 3), alb)
    alb = np.where(dep.mat == 2, 0.22 + 0.04 * grain, alb)
    alb = np.where(dep.mat == 3, 0.62, alb)
    alb = np.where(dep.mat == 4, 0.78, alb)
    alb = np.where(dep.mat == 5, 0.95 + 0.04 * blur(rng.standard_normal(dep.z.shape).astype(np.float32), 1.2), alb)
    alb = np.where(dep.mat == 6, 0.30 + 0.12 * value_noise(dep.z.shape[0], dep.z.shape[1], rng, 8.0), alb)
    alb = np.where(dep.mat == 7, 0.35 + 0.18 * value_noise(dep.z.shape[0], dep.z.shape[1], rng, 5.0), alb)
    alb = np.where(dep.mat == 8, 0.70, alb)
    dep.z += np.where(dep.mat == 5, 3 * blur(rng.standard_normal(dep.z.shape).astype(np.float32), 1.5), 0).astype(np.float32)
    spec = np.where(np.isin(dep.mat, [3, 4, 8]), 0.35, 0.03).astype(np.float32)
    lum, _ = light(dep, alb, spec=spec, shin=np.where(np.isin(dep.mat, [3, 4, 8]), 40.0, 10.0).astype(np.float32),
                   rim=0.12, ao_sigma=12, ao_depth=30)
    return finish(dep, lum, [1])


def rural_house(seed=34, W=1000, H=620):
    """Old rural house with a slate roof, straight-on."""
    rng = np.random.default_rng(seed)
    shp = (H, W)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    L = np.zeros(shp, np.float32)
    a = np.zeros(shp, np.float32)
    wall = _rect(shp, 110, 240, 890, 520)
    plaster = 0.78 + 0.05 * fbm(H, W, rng, 50.0, 4) - 0.06 * np.clip(streaks(H, W, rng, 50, 90), 0, 3) * 0.3
    L = plaster * wall
    a = np.maximum(a, wall)
    for xp in (110, 330, 670, 890):
        post = _rect(shp, xp - 11, 240, xp + 11, 525)
        L = L * (1 - post) + (0.26 + 0.05 * streaks(H, W, rng, 30, 90)) * post
    door = _rect(shp, 355, 285, 645, 505)
    lat = ((np.mod(xx - 355, 36) < 3) | (np.mod(yy - 285, 30) < 3)).astype(np.float32)
    L = L * (1 - door) + (0.86 - 0.55 * lat) * door
    for (x0, x1) in ((170, 290), (710, 830)):
        win = _rect(shp, x0, 300, x1, 390)
        latw = ((np.mod(xx - x0, 20) < 3) | (np.mod(yy - 300, 20) < 3)).astype(np.float32)
        L = L * (1 - win) + (0.80 - 0.5 * latw) * win
    found = _rect(shp, 80, 518, 920, 575)
    stones = 0.52 + 0.10 * value_noise(H, W, rng, 22.0) - 0.25 * (np.abs(np.mod(xx - 80, 70) - 35) > 33)
    L = L * (1 - found) + stones * found
    a = np.maximum(a, found)
    step = _rect(shp, 440, 572, 560, 600)
    L = L * (1 - step) + 0.60 * step
    a = np.maximum(a, step)
    roof = _poly(shp, [(190, 110), (810, 110), (975, 255), (25, 255)])
    corr = 0.5 + 0.5 * np.sin((xx - 500) / np.maximum(0.2 + (yy - 110) / 145.0, 0.2) * 0.42)
    Lr = 0.20 + 0.10 * corr + 0.05 * fbm(H, W, rng, 40.0, 3) - 0.06 * smoothstep(110, 255, yy)
    L = L * (1 - roof) + Lr * roof
    a = np.maximum(a, roof)
    eave_shadow = smoothstep(290, 255, yy) * wall * 0.35
    L = L * (1 - eave_shadow)
    ridge = _rect(shp, 180, 100, 820, 116)
    L = L * (1 - ridge) + 0.16 * ridge
    a = np.maximum(a, ridge)
    return np.clip(L, 0, 1), np.clip(a, 0, 1), None


def mountains(seed=35, W=1600, H=620):
    """Layered ridges in autumn haze, far = light, near = dark."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    L = np.full((H, W), 0.93, np.float32) - 0.05 * (yy / H)
    a = np.zeros((H, W), np.float32)
    layers = [(170, 0.80, 120), (260, 0.64, 110), (360, 0.46, 90), (460, 0.28, 70)]
    for i, (base, tone, amp) in enumerate(layers):
        ridge = base - amp * (0.55 + 0.45 * noise1d(W, rng, 260.0 / (i + 1), 5))
        m = smoothstep(ridge[None, :] - 1.0, ridge[None, :] + 1.0, yy)
        haze = smoothstep(ridge[None, :] + 140, ridge[None, :], yy) * 0.10
        trees = 0.05 * value_noise(H, W, rng, 3.0 + i) * (i >= 2)
        Li = tone + haze + trees + 0.03 * fbm(H, W, rng, 60.0, 3)
        L = L * (1 - m) + Li * m
        a = np.maximum(a, m)
    return np.clip(L, 0, 1), np.clip(a, 0, 1), None
