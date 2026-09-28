"""Printed props whose text must be exact: forms, posters, folders, clippings, signs, calendars.

All return (L, alpha, part) like the other generators; L is lit like a photographed document.
Personal details are masked (박 ○ ○) - the props only restate what the narration says.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..noise import blur, fbm, smoothstep, streaks, value_noise
from ..paper import paper_luma
from ..typography import font

SYL = ("가각간갈감강개거건걸검게겨결경계고곡공과관광교구국군권귀규그근금기길김나남내너년노농누뉴는"
       "다단달담당대더도동두드들등라란람랑래러레려력련령로록론뢰료루류르른를리린림마만말망매머면명"
       "모목몰무문물미민바박반발방배백번범법변별병보복본부북분불비빈사산살삼상새생서석선설성세소속"
       "손송수숙순술시식신실심아악안알암압앙애야약양어억언얼엄업에여역연열염영예오옥온올완왕외요용우"
       "운원월위유육윤은을음의이익인일임입자작잔장재저적전절점정제조족존종주죽준중지직진질집차착찬참"
       "창채책처천철청체초촌총최추축출충취측치친칠침카타탄태터토통투특파판패편평포표품풍프피필하학한"
       "할함합항해행향허험혁현협형호혹혼화확환활황회획효후훈휴흥희")


def fake_text(rng, n_chars, space_every=(2, 5)):
    out = []
    k = int(rng.integers(*space_every))
    for _ in range(n_chars):
        if k == 0:
            out.append(" ")
            k = int(rng.integers(*space_every))
        else:
            out.append(SYL[int(rng.integers(0, len(SYL)))])
            k -= 1
    return "".join(out)


class Doc:
    """A sheet being printed on. Luminance canvas (1 = paper white) + ink helpers."""

    def __init__(self, w, h, rng, tone=0.95, light=0.06):
        self.w, self.h, self.rng = w, h, rng
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = 1.0 - light * ((xx / w) * 0.6 + (yy / h) * 0.8)
        self.L = (tone * paper_luma(h, w, rng, 1.4) * grad).astype(np.float32)
        self.ink = np.zeros((h, w), np.float32)
        self.part = np.zeros((h, w), np.float32)

    def text(self, s, xy, size, weight=400, family="hahmlet", anchor="la", value=1.0, tracking=0.0, target=None):
        img = Image.new("L", (self.w, self.h), 0)
        d = ImageDraw.Draw(img)
        f = font(size, weight, family)
        if tracking:
            x, y = xy
            total = sum(f.getlength(c) for c in s) + tracking * size * (len(s) - 1)
            if anchor[0] == "m":
                x -= total / 2
            elif anchor[0] == "r":
                x -= total
            for c in s:
                d.text((x, y), c, font=f, fill=255, anchor="l" + anchor[1])
                x += f.getlength(c) + tracking * size
        else:
            d.text(xy, s, font=f, fill=255, anchor=anchor)
        m = np.asarray(img, np.float32) / 255.0 * value
        tgt = self.ink if target is None else target
        np.maximum(tgt, m, out=tgt)
        return m

    def rect(self, x0, y0, x1, y1, value=1.0, width=0, target=None):
        img = np.zeros((self.h, self.w), np.float32)
        if width <= 0:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), float(value), -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), float(value), int(width), cv2.LINE_AA)
        tgt = self.ink if target is None else target
        np.maximum(tgt, img, out=tgt)
        return img

    def line(self, p0, p1, width=2, value=1.0):
        img = np.zeros((self.h, self.w), np.float32)
        cv2.line(img, (int(p0[0] * 4), int(p0[1] * 4)), (int(p1[0] * 4), int(p1[1] * 4)), float(value), int(width),
                 cv2.LINE_AA, shift=2)
        np.maximum(self.ink, img, out=self.ink)

    def typed(self, mask_amount=0.9):
        """Typewriter/print irregularity for everything inked so far."""
        n = value_noise(self.h, self.w, self.rng, cell=3.0)
        dens = np.clip(0.82 + 0.18 * n, 0.55, 1.0)
        self.ink = np.clip(blur(self.ink, 0.45) * dens * mask_amount + self.ink * (1 - mask_amount), 0, 1)

    def result(self, ink_value=0.10, alpha=None):
        L = self.L * (1 - self.ink) + ink_value * self.ink
        a = np.ones((self.h, self.w), np.float32) if alpha is None else alpha
        return np.clip(L, 0, 1).astype(np.float32), a.astype(np.float32)


def stamp_mask(text, size, rng, w=None, h=None, double=True, circle=False, weight=800):
    """Rubber-stamp impression: bordered text with patchy ink."""
    f = font(size, weight)
    tw = f.getlength(text)
    w = int(w or tw + size * 1.2)
    h = int(h or size * 1.9)
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    lw = max(3, int(size * 0.09))
    if circle:
        d.ellipse([lw, lw, w - lw, h - lw], outline=255, width=lw)
        if double:
            d.ellipse([lw * 3, lw * 3, w - lw * 3, h - lw * 3], outline=255, width=max(2, lw // 2))
    else:
        d.rectangle([lw, lw, w - lw, h - lw], outline=255, width=lw)
        if double:
            d.rectangle([lw * 3, lw * 3, w - lw * 3, h - lw * 3], outline=255, width=max(2, lw // 2))
    d.text((w / 2, h / 2), text, font=f, fill=255, anchor="mm")
    m = np.asarray(img, np.float32) / 255.0
    patch = np.clip(fbm(h, w, rng, cell=size * 0.6, octaves=4) * 0.5 + 0.85, 0, 1)
    speck = (rng.random((h, w)) > 0.06).astype(np.float32)
    m = m * smoothstep(0.35, 0.75, patch) * blur(speck, 0.5)
    return np.clip(blur(m, 0.6) * 1.15, 0, 1)


def _paste(dst, src, x, y, angle=0.0, mode="max"):
    if angle:
        h, w = src.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        c, s = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * s + w * c), int(h * c + w * s)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        src = cv2.warpAffine(src, M, (nw, nh), flags=cv2.INTER_LINEAR)
    h, w = src.shape
    x, y = int(x - w / 2), int(y - h / 2)
    H, W = dst.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 > x0 and y1 > y0:
        seg = src[y0 - y:y1 - y, x0 - x:x1 - x]
        if mode == "max":
            dst[y0:y1, x0:x1] = np.maximum(dst[y0:y1, x0:x1], seg)
        else:
            dst[y0:y1, x0:x1] = seg
    return src


def report_form(seed=11):
    """Police report receipt with a '단순가출' stamp (part = the stamp)."""
    rng = np.random.default_rng(seed)
    W, H = 820, 1160
    doc = Doc(W, H, rng, tone=0.94)
    doc.text("제 97 -          호", (70, 70), 24, 400, "myeongjo")
    doc.text("1997 .      .      .", (750, 70), 24, 400, "myeongjo", anchor="ra")
    doc.text("신고 접수서", (W / 2, 150), 58, 700, anchor="ma", tracking=0.35)
    rows = [("성      명", "박  ○  ○"), ("연      령", "27 세"), ("직      업", "농협 직원"),
            ("주      소", "전북 김제시"), ("신  고  인", "가  족"), ("신고 내용", "연락 두절 · 귀가하지 않음"),
            ("처리 결과", "")]
    x0, x1, y = 70, 750, 270
    rh = 96
    doc.rect(x0, y, x1, y + rh * len(rows), width=3)
    for i, (k, v) in enumerate(rows):
        yy = y + i * rh
        if i:
            doc.line((x0, yy), (x1, yy), 2)
        doc.text(k, (x0 + 22, yy + rh / 2), 28, 500, anchor="lm")
        doc.text(v, (x0 + 230, yy + rh / 2), 30, 400, "myeongjo", anchor="lm")
    doc.line((x0 + 200, y), (x0 + 200, y + rh * len(rows)), 2)
    for k in range(3):
        doc.text(fake_text(rng, 26), (x0, y + rh * len(rows) + 60 + k * 44), 22, 400, "myeongjo", value=0.8)
    doc.text("담 당 :  ______________   ( 인 )", (750, 1080), 26, 400, "myeongjo", anchor="ra")
    doc.typed()
    stamp = stamp_mask("단순가출", 78, rng)
    part = np.zeros((H, W), np.float32)
    _paste(part, stamp, 470, y + rh * 6.45, angle=float(rng.uniform(5, 9)))
    L, a = doc.result()
    L = L * (1 - part * 0.82) + 0.16 * part * 0.82
    return L.astype(np.float32), a, part


def missing_poster(seed=12):
    """'사람을 찾습니다' flyer with tear-off tabs (part = the header band)."""
    rng = np.random.default_rng(seed)
    W, H = 820, 1200
    doc = Doc(W, H, rng, tone=0.95)
    band = np.zeros((H, W), np.float32)
    band[40:230, 40:W - 40] = 1.0
    white = np.zeros((H, W), np.float32)
    doc.text("사람을 찾습니다", (W / 2, 136), 96, 900, anchor="mm", target=white)
    # photo box with an anonymous silhouette
    px0, py0, px1, py1 = 250, 280, 570, 680
    photo = np.zeros((H, W), np.float32)
    photo[py0:py1, px0:px1] = 1.0
    sil = np.zeros((H, W), np.float32)
    cv2.ellipse(sil, (410, 450), (70, 88), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.ellipse(sil, (410, 690), (150, 150), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
    sil *= photo
    doc.rect(px0, py0, px1, py1, width=3)
    doc.text("박 ○ ○  (27세 · 여)", (W / 2, 745), 46, 800, anchor="mm")
    doc.text("1997년 가을  전북 김제에서 실종", (W / 2, 815), 34, 500, anchor="mm")
    doc.text("보신 분은 가까운 경찰서로 연락 바랍니다", (W / 2, 875), 28, 400, "myeongjo", anchor="mm")
    # tear-off tabs
    tabs_y = 960
    n_tabs = 9
    tw = (W - 80) / n_tabs
    for i in range(n_tabs + 1):
        x = 40 + i * tw
        for yy in range(tabs_y, H - 30, 14):
            doc.line((x, yy), (x, yy + 7), 2, 0.8)
    for i in range(n_tabs):
        cx = 40 + (i + 0.5) * tw
        doc.text("박○○ 찾습니다", (cx, tabs_y + 105), 20, 500, anchor="mm")
    doc.typed(0.6)
    L, _ = doc.result()
    L = L * (1 - band) + (0.14 + 0.04 * value_noise(H, W, rng, 30.0)) * band
    L = L * (1 - white * band) + 0.93 * white * band
    L = L * (1 - photo) + (0.70 + 0.03 * value_noise(H, W, rng, 25.0)) * photo
    L = L * (1 - sil) + 0.36 * sil
    # alpha: tabs partly torn off (two missing)
    a = np.ones((H, W), np.float32)
    for i in rng.choice(n_tabs, 2, replace=False):
        x = int(40 + i * tw)
        a[tabs_y + 4:, x + 2:int(x + tw) - 1] = 0.0
    return np.clip(L, 0, 1).astype(np.float32), a, band.astype(np.float32)


def paperclip(canvas_shape, x, y, scale=1.0, angle=0.0):
    """A wire paper clip mask + its highlight (drawn)."""
    h, w = canvas_shape
    pts = np.array([[0, 0], [0, 150], [26, 176], [52, 150], [52, 30], [34, 12], [16, 30], [16, 130]], np.float32)
    pts = pts * scale
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    pts = pts @ R.T + np.array([x, y], np.float32)
    m = np.zeros((h, w), np.float32)
    cv2.polylines(m, [(pts * 4).astype(np.int32)], False, 1.0, int(7 * scale), cv2.LINE_AA, shift=2)
    hi = np.zeros((h, w), np.float32)
    cv2.polylines(hi, [((pts + [-1.5, -1.5]) * 4).astype(np.int32)], False, 1.0, max(1, int(2 * scale)), cv2.LINE_AA, shift=2)
    return m, hi


def case_file(seed=13, label="수 사 기 록", stamp="종결", big_x=False):
    """Worn cardboard case folder with papers inside (part = the front cover)."""
    rng = np.random.default_rng(seed)
    W, H = 1000, 780
    doc = Doc(W, H, rng, tone=0.95)
    a = np.zeros((H, W), np.float32)
    L = np.zeros((H, W), np.float32)
    # back board + tab
    back = np.zeros((H, W), np.float32)
    cv2.rectangle(back, (40, 70), (940, 740), 1.0, -1)
    cv2.rectangle(back, (70, 30), (330, 80), 1.0, -1)
    back = blur(back, 1.0)
    board_tex = paper_luma(H, W, rng, 2.2) * (0.60 + 0.03 * fbm(H, W, rng, 90.0, 3))
    L = board_tex * back
    a = np.maximum(a, back)
    # papers sticking out
    for i, (dx, dy, ang) in enumerate([(30, -26, -1.2), (58, -12, 0.8)]):
        sheet = np.zeros((H, W), np.float32)
        box = cv2.boxPoints(((500 + dx, 400 + dy), (860, 640), ang)).astype(np.int32)
        cv2.fillPoly(sheet, [box], 1.0, cv2.LINE_AA)
        sheet_L = 0.93 * paper_luma(H, W, rng, 1.0)
        lines = np.zeros((H, W), np.float32)
        for k in range(14):
            yy = 110 + dy + k * 34
            x0 = 130 + dx
            cv2.line(lines, (x0, yy), (x0 + int(rng.uniform(420, 700)), yy), 1.0, 5)
        lines = blur(lines, 1.2) * 0.35 * sheet
        L = L * (1 - sheet) + (sheet_L * (1 - lines)) * sheet
        a = np.maximum(a, sheet)
    # front cover (slightly smaller, offset)
    cover = np.zeros((H, W), np.float32)
    pts = np.array([[52, 104], [930, 96], [944, 752], [46, 758]], np.int32)
    cv2.fillPoly(cover, [pts * 4], 1.0, cv2.LINE_AA, shift=2)
    cover = blur(cover, 0.8)
    wear = smoothstep(0.6, 2.2, fbm(H, W, rng, 40.0, 4))
    edge_d = cv2.distanceTransform((cover > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    worn = np.clip(1.0 - edge_d / 18.0, 0, 1) * 0.10 + wear * 0.06
    cover_L = paper_luma(H, W, rng, 2.4) * (0.64 + 0.03 * fbm(H, W, rng, 120.0, 3)) + worn
    ring = np.zeros((H, W), np.float32)
    cv2.ellipse(ring, (int(rng.uniform(650, 820)), int(rng.uniform(560, 660))), (70, 64), 0, 0, 360, 1.0, 6, cv2.LINE_AA)
    cover_L *= 1 - blur(ring, 2.0) * 0.12 * np.clip(fbm(H, W, rng, 30.0, 3) * 0.5 + 0.7, 0, 1)
    L = L * (1 - cover) + cover_L * cover
    a = np.maximum(a, cover)
    part = cover.copy()
    # label
    lab = np.zeros((H, W), np.float32)
    cv2.rectangle(lab, (310, 220), (690, 430), 1.0, -1)
    lab = blur(lab, 0.7)
    L = L * (1 - lab) + 0.94 * paper_luma(H, W, rng, 1.0) * lab
    part = part * (1 - lab)
    ink = np.zeros((H, W), np.float32)
    d2 = Doc(W, H, rng)
    d2.ink = ink
    d2.text(label, (500, 300), 56, 700, anchor="mm", tracking=0.1)
    d2.text("1997 · 김 제", (500, 370), 30, 400, "myeongjo", anchor="mm")
    d2.line((340, 400), (660, 400), 2, 0.8)
    d2.typed()
    L = L * (1 - d2.ink) + 0.12 * d2.ink
    if stamp:
        st = stamp_mask(stamp, 74, rng, circle=True, w=210, h=210)
        sm = np.zeros((H, W), np.float32)
        _paste(sm, st, 760, 250, angle=float(rng.uniform(-14, -6)))
        L = L * (1 - sm * 0.85) + 0.18 * sm * 0.85
    if big_x:
        xm = np.zeros((H, W), np.float32)
        for (p0, p1) in [((360, 470), (650, 700)), ((655, 468), (372, 712))]:
            n = 40
            for i in range(n):
                t = i / (n - 1)
                x = p0[0] + (p1[0] - p0[0]) * t + rng.normal(0, 1.2)
                y = p0[1] + (p1[1] - p0[1]) * t + rng.normal(0, 1.2)
                r = 17 * (0.75 + 0.35 * np.sin(np.pi * t)) * (0.9 + 0.2 * rng.random())
                cv2.circle(xm, (int(x * 4), int(y * 4)), int(r * 4), 1.0, -1, cv2.LINE_AA, shift=2)
        brush = streaks(H, W, rng, 15, angle_deg=40)
        xm = np.clip(xm * (0.85 + 0.15 * np.clip(brush, -1, 1)), 0, 1)
        L = L * (1 - xm * 0.9) + 0.08 * xm * 0.9
        part = part * (1 - xm)
    clip_m, clip_hi = paperclip((H, W), 850, 40, 1.05, -8)
    L = L * (1 - clip_m) + 0.55 * clip_m + clip_hi * 0.35
    a = np.maximum(a, clip_m)
    L = L * (1.0 - 0.05 * np.mgrid[0:H, 0:W][1] / W)
    return np.clip(L, 0, 1).astype(np.float32), np.clip(a, 0, 1), np.clip(part, 0, 1)


def clipping(seed=14, photo=None):
    """A newspaper clipping restating the narration (part = none)."""
    rng = np.random.default_rng(seed)
    W, H = 900, 1060
    doc = Doc(W, H, rng, tone=0.88, light=0.04)
    doc.text("수문 보수공사 중", (40, 40), 78, 850)
    doc.text("백골 시신 발견", (40, 135), 78, 850)
    doc.rect(40, 245, 860, 249)
    doc.text("12년 전 실종된 농협 여직원 추정 … 마을 저수지 수문 하부서", (40, 268), 30, 500)
    px0, py0, px1, py1 = 40, 330, 520, 690
    ph = np.zeros((H, W), np.float32)
    ph[py0:py1, px0:px1] = 1.0
    if photo is not None:
        p = cv2.resize(photo, (px1 - px0, py1 - py0), interpolation=cv2.INTER_AREA)
    else:
        p = np.clip(0.55 + 0.25 * fbm(py1 - py0, px1 - px0, rng, 60.0, 4), 0, 1)
    from ..print_fx import halftone
    p = halftone(p, pitch=4.2, angle=45, strength=0.95, gate=(0.0, 1.01))
    full = np.zeros((H, W), np.float32)
    full[py0:py1, px0:px1] = p
    p = full
    doc.text("▲ 백골이 발견된 저수지 수문", (px0, py1 + 14), 20, 500)
    # body text columns
    cols = [(545, 330, 860, 1030), (40, 730, 290, 1030), (310, 730, 520, 1030)]
    for (x0, y0, x1, y1) in cols:
        n = int((x1 - x0) / 18.5)
        yy = y0
        while yy < y1 - 20:
            doc.text(fake_text(rng, n), (x0, yy), 17, 400, "myeongjo", value=0.9)
            yy += 27
    doc.typed(0.7)
    L, _ = doc.result(0.12)
    L = L * (1 - ph) + p * ph
    # scissor-cut outline (slightly irregular straight cuts)
    a = np.zeros((H, W), np.float32)
    pts = np.array([[8 + rng.uniform(-4, 4), 6], [W - 6, 10 + rng.uniform(-4, 4)],
                    [W - 10 + rng.uniform(-4, 4), H - 8], [6, H - 4 + rng.uniform(-4, 0)]], np.float32)
    cv2.fillPoly(a, [(pts * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
    return np.clip(L, 0, 1).astype(np.float32), a, None


def warning_sign(seed=15):
    """Rusty '위험 / 출입금지' plate on a pipe post (part = the plate)."""
    rng = np.random.default_rng(seed)
    W, H = 760, 1080
    L = np.zeros((H, W), np.float32)
    a = np.zeros((H, W), np.float32)
    # post
    post = np.zeros((H, W), np.float32)
    cv2.rectangle(post, (350, 480), (410, H), 1.0, -1)
    xx = np.mgrid[0:H, 0:W][1].astype(np.float32)
    cyl = np.clip(1 - ((xx - 380) / 30.0) ** 2, 0, 1) ** 0.5
    L += post * (0.28 + 0.30 * cyl + 0.05 * streaks(H, W, rng, 25, 90))
    a = np.maximum(a, post)
    # plate
    plate = np.zeros((H, W), np.float32)
    cv2.rectangle(plate, (40, 60), (720, 560), 1.0, -1)
    plate = blur(plate, 1.2)
    plate = (plate > 0.5).astype(np.float32)
    plate = blur(plate, 0.8)
    doc = Doc(W, H, rng, tone=0.90, light=0.10)
    doc.rect(62, 82, 698, 538, width=10)
    doc.text("위  험", (380, 250), 190, 900, anchor="mm")
    doc.text("출 입 금 지", (380, 440), 92, 800, anchor="mm")
    doc.typed(0.5)
    PL, _ = doc.result(0.12)
    rust = np.zeros((H, W), np.float32)
    for bx, by in [(90, 110), (670, 110), (90, 510), (670, 510)]:
        cv2.circle(rust, (bx, by), 10, 1.0, -1, cv2.LINE_AA)
        drip = np.zeros((H, W), np.float32)
        cv2.line(drip, (bx, by), (bx + int(rng.uniform(-6, 6)), by + int(rng.uniform(60, 200))), 1.0, int(rng.uniform(4, 9)))
        rust = np.maximum(rust, blur(drip, 3.0) * rng.uniform(0.3, 0.6))
    stains = np.clip(smoothstep(0.8, 2.2, fbm(H, W, rng, 50.0, 4)), 0, 1) * 0.35
    PL = PL * (1 - 0.45 * np.clip(rust + stains, 0, 1))
    L = L * (1 - plate) + PL * plate
    a = np.maximum(a, plate)
    return np.clip(L, 0, 1).astype(np.float32), a, plate


def calendar(seed=16, first_weekday=3, days=31):
    """A 1997 month page. Returns (L, a, None, cells) - cells[d] = (x, y) of day d in px."""
    rng = np.random.default_rng(seed)
    W, H = 760, 860
    doc = Doc(W, H, rng, tone=0.95)
    doc.text("1997", (60, 50), 86, 800)
    doc.rect(60, 160, 700, 164)
    names = "일월화수목금토"
    cw, ch = 640 / 7, 110
    for i, n in enumerate(names):
        doc.text(n, (60 + cw * (i + 0.5), 200), 30, 600, anchor="mm", value=0.9)
    cells = {}
    for d in range(1, days + 1):
        k = first_weekday + d - 1
        r, c = divmod(k, 7)
        x, y = 60 + cw * (c + 0.5), 260 + ch * r + 20
        cells[d] = (x, y)
        doc.text(str(d), (x, y), 40, 500, anchor="mm", value=0.85)
    for r in range(6):
        doc.line((60, 238 + ch * r), (700, 238 + ch * r), 1, 0.35)
    doc.typed(0.5)
    L, a = doc.result(0.13)
    # binding perforation at the top
    for x in range(40, W - 30, 34):
        cv2.circle(a, (x, 14), 7, 0.0, -1, cv2.LINE_AA)
    return L, a, None, cells


def sticker(text, seed=17, w=560, h=230, filled=True):
    """A button-shaped sticker ('구독', '좋아요'); part = the button body."""
    rng = np.random.default_rng(seed)
    body = np.zeros((h, w), np.float32)
    r = h // 2 - 8
    cv2.rectangle(body, (8 + r, 8), (w - 8 - r, h - 8), 1.0, -1, cv2.LINE_AA)
    cv2.circle(body, (8 + r, h // 2), r, 1.0, -1, cv2.LINE_AA)
    cv2.circle(body, (w - 8 - r, h // 2), r, 1.0, -1, cv2.LINE_AA)
    doc = Doc(w, h, rng, tone=0.95)
    txt = np.zeros((h, w), np.float32)
    doc.text(text, (w / 2, h / 2 + 4), int(h * 0.46), 900, anchor="mm", target=txt)
    tex = paper_luma(h, w, rng, 1.2)
    if filled:
        L = (0.16 * tex) * (1 - txt) + 0.93 * txt
    else:
        ring = body - blur((cv2.erode(body, np.ones((13, 13), np.uint8)) > 0.5).astype(np.float32), 0.8)
        L = 0.93 * tex * (1 - np.maximum(txt, ring)) + 0.12 * np.maximum(txt, ring)
    return np.clip(L, 0, 1).astype(np.float32), body, (body * (1 - txt) if filled else None)


def tape(seed=18, length=420, width=110):
    """Masking tape strip: translucent, crepe texture, torn/zig-zag ends."""
    rng = np.random.default_rng(seed)
    h, w = width + 20, length + 20
    a = np.zeros((h, w), np.float32)
    a[10:10 + width, 10:10 + length] = 1.0
    for side in (0, 1):
        x = 10 if side == 0 else 10 + length
        for y in range(10, 10 + width):
            dxx = int(abs(np.sin(y * 0.55)) * 7 + rng.uniform(0, 4))
            if side == 0:
                a[y, x:x + dxx] = 0
            else:
                a[y, x - dxx:x] = 0
    a = blur(a, 0.6)
    L = 0.86 + 0.03 * streaks(h, w, rng, 9, 90) + 0.02 * value_noise(h, w, rng, 30.0)
    return np.clip(L, 0, 1).astype(np.float32), (a * 0.82).astype(np.float32), None
