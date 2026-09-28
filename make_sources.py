#!/usr/bin/env python3
"""Generate the procedural source images into sources/ (RGBA + optional .part.png).

Real photographs can sit next to them in sources/; `python prep_subject.py --batch sources/`
then prepares everything the same way. A photo with the same name as a procedural source
wins (the generator skips names that already exist unless --force).

    python make_sources.py            # writes only missing sources
    python make_sources.py --force    # regenerate all
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from collage.procedural import docs, figures, maps, objects  # noqa: E402

GENERATORS = {
    # name: (callable, options for prep_subject)
    "map_jeonbuk": (lambda: maps.jeonbuk(), {}),
    "map_korea": (lambda: maps.korea(), {}),
    "report_form": (lambda: docs.report_form(), {}),
    "missing_poster": (lambda: docs.missing_poster(), {}),
    "case_file": (lambda: docs.case_file(), {}),
    "case_file_x": (lambda: docs.case_file(seed=19, stamp=None, big_x=True, label="X 파 일"), {}),
    "clipping": (lambda: docs.clipping(), {}),
    "warning_sign": (lambda: docs.warning_sign(), {}),
    "calendar": (lambda: docs.calendar()[:3], {}),
    "sticker_sub": (lambda: docs.sticker("구독"), {}),
    "sticker_like": (lambda: docs.sticker("좋아요", seed=23, w=620, filled=False), {}),
    "tape": (lambda: docs.tape(), {}),
    # fallbacks, used only when no photograph with this name exists
    "man_back": (lambda: figures.man_back("jumper"), {"person": True}),
    "woman_back": (lambda: figures.woman_back(), {"person": True}),
    "sluice_gate": (lambda: objects.sluice_gate(), {}),
    "concrete": (lambda: objects.concrete(), {}),
    "soban": (lambda: objects.soban(), {}),
    "rural_house": (lambda: objects.rural_house(), {}),
    "mountains": (lambda: objects.mountains(), {}),
}
PHOTO_EXT = (".jpg", ".jpeg", ".webp", ".png")


def save(name, L, a, part, out):
    rgba = np.dstack([L, L, L, a])
    cv2.imwrite(os.path.join(out, name + ".png"),
                cv2.cvtColor((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    pp = os.path.join(out, name + ".part.png")
    if part is not None:
        cv2.imwrite(pp, (np.clip(part, 0, 1) * 255 + 0.5).astype(np.uint8))
    elif os.path.exists(pp):
        os.remove(pp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "sources"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "sources.json")
    manifest = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {}
    for name, (fn, opts) in GENERATORS.items():
        if a.only and name not in a.only:
            continue
        photo = [e for e in PHOTO_EXT if os.path.exists(os.path.join(a.out, name + e))
                 and manifest.get(name, {}).get("generated") is not True]
        if photo and not a.force:
            print(f"[sources] {name}: using supplied image {name}{photo[0]}")
            continue
        L, al, part = fn()
        save(name, L, al, part, a.out)
        manifest[name] = dict(opts, method="alpha", generated=True)
        print(f"[sources] {name}: generated {L.shape[1]}x{L.shape[0]} part={part is not None}")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
