"""Props for the '그때 그 돈' shorts, episode 1 (짜장면 price controls in the 1970s).

Every printed word restates the narration or a sourced figure:
  - 1970s early: 짜장면 held under 100원 by the 가격협정요금 system (경향신문 2022-05-26)
  - consumer price inflation 1970 16.0%, 1971 13.5%, 1972 11.7%, 1973 3.2% (same article)
  - price rises answered with hygiene inspections and tax audits; restaurants answered with
    new dishes outside the controlled item: 삼선짜장, 유니짜장 (same article)
Prices that are not sourced are never printed.

All return (L, alpha, part) like the other generators.
"""
import cv2
import numpy as np

from ..noise import blur, fbm, smoothstep, streaks, value_noise
from ..paper import paper_luma
from .docs import Doc, _paste, fake_text, stamp_mask
from .shading import NEG, Depth, finish, light


# ------------------------------------------------------------------------------ bowl
def bowl(seed=41, W=1000, H=820):
    """A bowl of 짜장면 with chopsticks, three-quarter view (part = the bowl's patterned rim band)."""
    rng = np.random.default_rng(seed)
    dep = Depth(W, H, 2)
    X, Y = dep.xx, dep.yy
    cx, cy, rx, ry = 500.0, 300.0, 400.0, 150.0
    # outer body: lower half-ellipse bulging down to a foot
    body_h = 300.0
    rows = np.arange(H, dtype=np.float32)
    t = np.clip((rows - cy) / body_h, 0, 1)
    hw = np.where(rows >= cy - 2, rx * np.sqrt(np.clip(1 - t ** 2.2, 0, 1)) * (1 - 0.18 * t), 0.0)
    hw = np.where(rows > cy + body_h, 0.0, hw)
    body = dep.profile_solid(cx, hw, np.full_like(hw, 90.0), z0=40.0, power=0.5)
    dep.add(body, 1)
    foot = (((X - cx) / 170.0) ** 2 + ((Y - cy - body_h + 4) / 26.0) ** 2 < 1) | \
        ((np.abs(X - cx) < 170) & (Y > cy + body_h - 30) & (Y < cy + body_h + 4))
    dep.add(np.where(foot, 60.0 - 0.1 * (Y - cy), NEG).astype(np.float32), 1)
    # rim + inside wall
    u = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
    rim = (u < 1.0) & (u > 0.86)
    dep.add(np.where(rim, 150.0 + 4 * np.sqrt(np.clip(1 - np.abs(u - 0.93) / 0.07, 0, 1)), NEG).astype(np.float32), 2)
    inner = u <= 0.86
    dep.add(np.where(inner, 100.0 + 30 * u, NEG).astype(np.float32), 3)
    # noodles: a mound of long wavy capsules, sauce poured over the top half
    mound_cx, mound_cy = cx + 10, cy + 8
    for k in range(150):
        a0 = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(0.05, 0.78)
        x0 = mound_cx + np.cos(a0) * r0 * rx * 0.78
        y0 = mound_cy + np.sin(a0) * r0 * ry * 0.72
        ang = rng.uniform(0, np.pi)
        pts = []
        for s in range(9):
            ang += rng.normal(0, 0.45)
            x0 += np.cos(ang) * 20
            y0 += np.sin(ang) * 8
            if ((x0 - cx) / (rx * 0.84)) ** 2 + ((y0 - cy) / (ry * 0.80)) ** 2 > 1:
                break
            pts.append((x0, y0))
        for i in range(len(pts) - 1):
            d = np.hypot(pts[i][0] - mound_cx, (pts[i][1] - mound_cy) * 2.6) / (rx * 0.8)
            z = 150 + 38 * np.sqrt(max(0.0, 1 - d * d)) + rng.uniform(0, 4)
            dep.add(dep.capsule(pts[i], pts[i + 1], 5.0, z0=z, rz=4), 4)
    # sauce: a lumpy cap over the mound centre
    sauce_u = ((X - mound_cx + 30) / (rx * 0.60)) ** 2 + ((Y - mound_cy + 8) / (ry * 0.58)) ** 2
    lumps = 10 * fbm(dep.z.shape[0], dep.z.shape[1], rng, 30.0, 3)
    sauce = np.where(sauce_u < 1, 176 + 26 * np.sqrt(np.clip(1 - sauce_u, 0, 1)) + lumps, NEG).astype(np.float32)
    dep.add(sauce, 5)
    for _ in range(26):  # diced onion / pork in the sauce
        px, py = mound_cx - 30 + rng.normal(0, 70), mound_cy - 8 + rng.normal(0, 24)
        dep.add(dep.ellipsoid(px, py, rng.uniform(6, 11), rng.uniform(4, 7), 5, z0=200), 6)
    for _ in range(9):  # cucumber julienne on top
        px, py = mound_cx - 20 + rng.normal(0, 40), mound_cy - 20 + rng.normal(0, 12)
        ang = rng.uniform(-0.6, 0.6)
        dep.add(dep.capsule((px, py), (px + 46 * np.cos(ang), py + 16 * np.sin(ang)), 3.5, z0=214), 7)
    # chopsticks across the rim
    for dy in (0, 16):
        dep.add(dep.capsule((140, 150 + dy), (930, 108 + dy), 6.5 - dy * 0.05, z0=240), 8)
    shp = dep.z.shape
    grain = streaks(shp[0], shp[1], rng, 60, 12.0)
    alb = np.zeros(shp, np.float32)
    alb = np.where(dep.mat == 1, 0.90, alb)
    alb = np.where(dep.mat == 2, 0.93, alb)
    alb = np.where(dep.mat == 3, 0.86, alb)
    alb = np.where(dep.mat == 4, 0.80 + 0.06 * value_noise(shp[0], shp[1], rng, 6.0), alb)
    alb = np.where(dep.mat == 5, 0.12 + 0.05 * value_noise(shp[0], shp[1], rng, 5.0), alb)
    alb = np.where(dep.mat == 6, 0.36, alb)
    alb = np.where(dep.mat == 7, 0.55, alb)
    alb = np.where(dep.mat == 8, 0.66 + 0.06 * grain, alb)
    # the classic patterned band on the outside of the bowl (thunder-scroll border)
    band = (dep.mat == 1) & (Y > cy + 26) & (Y < cy + 78)
    pat = (np.sin(X * 0.16) * np.sin((Y - cy) * 0.22) > 0.15) | (np.abs(Y - cy - 30) < 3) | (np.abs(Y - cy - 74) < 3)
    alb = np.where(band & pat, 0.30, alb)
    part_mat_mask = band.astype(np.float32)
    spec = np.where(np.isin(dep.mat, [1, 2, 3, 5]), 0.45, 0.08).astype(np.float32)
    shin = np.where(np.isin(dep.mat, [1, 2, 3]), 60.0, 25.0).astype(np.float32)
    lum, _ = light(dep, alb, spec=spec, shin=shin, rim=0.10, ao_sigma=10, ao_depth=30)
    L, a, _ = finish(dep, lum, None)
    part = dep.down(part_mat_mask)
    return L, a, np.clip(part, 0, 1)


# ------------------------------------------------------------------------------ menus
def _menu_board(rng, W, H, title, rows, tone=0.90):
    """Paper menu (차림표) pinned on a wall. rows: [(name, price or None, is_part, value)]."""
    doc = Doc(W, H, rng, tone=tone, light=0.08)
    doc.rect(34, 34, W - 34, H - 34, width=6)
    doc.rect(50, 50, W - 50, H - 50, width=2)
    doc.text(title, (W / 2, 150), 92, 900, anchor="mm", tracking=0.5)
    doc.line((120, 230), (W - 120, 230), 3)
    part = np.zeros((H, W), np.float32)
    n = len(rows)
    top, bottom = 300, H - 150
    step = (bottom - top) / max(n, 1)
    for i, (name, price, is_part, value) in enumerate(rows):
        y = top + step * (i + 0.5)
        tgt = part if is_part else None
        m = doc.text(name, (120, y), 70, 800, anchor="lm", value=value, target=tgt)
        if is_part:
            np.maximum(doc.ink, m, out=doc.ink)
        name_w = float(np.max(np.nonzero(m.max(axis=0) > 0.3)[0])) if (m > 0.3).any() else 300.0
        # dotted leader
        x0 = name_w + 30
        x1 = W - 330 if price else W - 120
        for x in np.arange(x0, x1, 22):
            cv2.circle(doc.ink, (int(x), int(y + 18)), 3, float(value * 0.8), -1, cv2.LINE_AA)
        if price:
            pm = doc.text(price, (W - 110, y), 76, 900, anchor="rm", value=value, target=tgt)
            if is_part:
                np.maximum(doc.ink, pm, out=doc.ink)
    doc.typed(0.6)
    L, a = doc.result(0.11)
    # pin holes + wall-worn corners
    for (px, py) in ((W / 2, 22),):
        cv2.circle(L, (int(px), int(py)), 9, 0.25, -1, cv2.LINE_AA)
    wear = smoothstep(0.9, 2.4, fbm(H, W, rng, 60.0, 4)) * 0.10
    L = L * (1 - wear)
    return L, a, part, doc


def menu_1970(seed=42):
    """1970s 차림표 - only the sourced price is printed (part = '짜장면 100원')."""
    rng = np.random.default_rng(seed)
    W, H = 900, 1180
    rows = [("짜 장 면", "100원", True, 1.0), ("우   동", None, False, 0.85), ("짬   뽕", None, False, 0.85),
            ("볶 음 밥", None, False, 0.85), ("탕 수 육", None, False, 0.85)]
    L, a, part, doc = _menu_board(rng, W, H, "차 림 표", rows)
    return np.clip(L, 0, 1).astype(np.float32), a, part


def menu_new(seed=43):
    """The same menu with the two new dishes written in (part = the new dishes)."""
    rng = np.random.default_rng(seed)
    W, H = 900, 1180
    rows = [("짜 장 면", "100원", False, 0.9), ("삼선짜장", None, True, 1.0), ("유니짜장", None, True, 1.0),
            ("우   동", None, False, 0.8), ("짬   뽕", None, False, 0.8)]
    L, a, part, doc = _menu_board(rng, W, H, "차 림 표", rows)
    # the new lines are written on paper strips pasted over the old board
    strips = np.zeros((H, W), np.float32)
    top, bottom = 300, H - 150
    step = (bottom - top) / 5
    for i in (1, 2):
        y = top + step * (i + 0.5)
        cv2.rectangle(strips, (96, int(y - 58)), (W - 96, int(y + 58)), 1.0, -1)
    strips = blur(strips, 1.0)
    tex = 0.96 * paper_luma(H, W, rng, 1.0)
    shadow = blur(np.roll(strips, (6, 4), axis=(0, 1)), 5.0) * (1 - strips) * 0.25
    ink = part
    L = L * (1 - shadow)
    L = L * (1 - strips) + (tex * (1 - ink) + 0.10 * ink) * strips
    return np.clip(L, 0, 1).astype(np.float32), a, part * strips


def menu_today(seed=44):
    """Today's long list of 짜장 dishes - names only, no prices (part = every '짜장' dish)."""
    rng = np.random.default_rng(seed)
    W, H = 900, 1320
    rows = [("짜 장 면", None, True, 1.0), ("간 짜 장", None, True, 1.0), ("삼선짜장", None, True, 1.0),
            ("유니짜장", None, True, 1.0), ("쟁반짜장", None, True, 1.0), ("사천짜장", None, True, 1.0)]
    L, a, part, _ = _menu_board(rng, W, H, "메 뉴", rows, tone=0.93)
    return np.clip(L, 0, 1).astype(np.float32), a, part


# ------------------------------------------------------------------------------ chart
def inflation_chart(seed=45):
    """Bar chart of consumer price inflation 1970-73 on graph paper (part = the bars)."""
    rng = np.random.default_rng(seed)
    W, H = 980, 900
    doc = Doc(W, H, rng, tone=0.95)
    grid = np.zeros((H, W), np.float32)
    for x in range(60, W - 40, 30):
        cv2.line(grid, (x, 150), (x, H - 120), 0.18 if (x - 60) % 150 else 0.35, 1)
    for y in range(150, H - 119, 30):
        cv2.line(grid, (60, y), (W - 60, y), 0.18 if (y - 150) % 150 else 0.35, 1)
    np.maximum(doc.ink, grid, out=doc.ink)
    doc.text("소비자물가 상승률", (70, 60), 54, 800)
    base = H - 150
    data = [("1970", 16.0), ("1971", 13.5), ("1972", 11.7), ("1973", 3.2)]
    part = np.zeros((H, W), np.float32)
    bw = 130
    for i, (yr, v) in enumerate(data):
        x0 = 140 + i * 200
        top = base - v * 34.0
        cv2.rectangle(part, (x0, int(top)), (x0 + bw, base), 1.0, -1)
        doc.text(f"{v:.1f}%", (x0 + bw / 2, top - 44), 44, 900, anchor="mm")
        doc.text(yr, (x0 + bw / 2, base + 50), 40, 600, anchor="mm")
    doc.line((60, base), (W - 60, base), 4)
    doc.typed(0.5)
    L, a = doc.result(0.12)
    hatch = (np.sin((np.mgrid[0:H, 0:W][1] + np.mgrid[0:H, 0:W][0]) * 0.5) > 0.2).astype(np.float32)
    L = L * (1 - part) + (0.22 + 0.10 * hatch) * part
    return np.clip(L, 0, 1).astype(np.float32), a, part


# ------------------------------------------------------------------------------ notices
def notice(seed=46, title="위 생 검 사", stamp="위생검사", reason="위생 점검"):
    """Official notice to a Chinese restaurant (part = the stamp)."""
    rng = np.random.default_rng(seed)
    W, H = 820, 1100
    doc = Doc(W, H, rng, tone=0.93)
    doc.text("통  지  서", (W / 2, 120), 70, 800, anchor="mm", tracking=0.2)
    doc.text(title, (W / 2, 210), 40, 500, "myeongjo", anchor="mm", tracking=0.3)
    rows = [("업      종", "중 화 요 리"), ("상      호", "○ ○ 반 점"), ("사      유", reason), ("일      시", "197 .     .     .")]
    x0, x1, y = 70, 750, 290
    rh = 92
    doc.rect(x0, y, x1, y + rh * len(rows), width=3)
    for i, (k, v) in enumerate(rows):
        yy = y + i * rh
        if i:
            doc.line((x0, yy), (x1, yy), 2)
        doc.text(k, (x0 + 22, yy + rh / 2), 28, 500, anchor="lm")
        doc.text(v, (x0 + 230, yy + rh / 2), 32, 500, "myeongjo", anchor="lm")
    doc.line((x0 + 200, y), (x0 + 200, y + rh * len(rows)), 2)
    for k in range(5):
        doc.text(fake_text(rng, 24), (x0, y + rh * len(rows) + 60 + k * 44), 22, 400, "myeongjo", value=0.75)
    doc.typed()
    part = np.zeros((H, W), np.float32)
    st = stamp_mask(stamp, 80, rng)
    _paste(part, st, 520, 920, angle=float(rng.uniform(-12, -5)))
    L, a = doc.result()
    L = L * (1 - part * 0.82) + 0.16 * part * 0.82
    return L.astype(np.float32), a, part
