"""Build data/korea_map.json from the public KOSTAT 2013 boundaries (southkorea/southkorea-maps).

    python tools/build_map_data.py            # downloads the GeoJSON from GitHub
    python tools/build_map_data.py --src DIR  # or reads muni.json / prov.json from DIR

Coordinates are projected to kilometres (equirectangular around 36N) and simplified,
so the renderer never needs the 80MB source files.
"""
import argparse, json, math, os, urllib.request
import numpy as np
import cv2

BASE = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/"
LAT0 = 36.0
KX = 111.32 * math.cos(math.radians(LAT0))
KY = 110.57


def load(src, name, url_name):
    if src:
        return json.load(open(os.path.join(src, name), encoding="utf-8"))
    with urllib.request.urlopen(BASE + url_name) as r:
        return json.loads(r.read().decode("utf-8"))


def rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        yield poly[0]  # outer ring only; holes are irrelevant at this scale


def project(ring):
    a = np.asarray(ring, np.float64)
    return np.stack([(a[:, 0] - 127.5) * KX, -(a[:, 1] - LAT0) * KY], 1)


def simplify(pts, eps_km, min_area_km2):
    area = abs(cv2.contourArea(pts.astype(np.float32)))
    if area < min_area_km2:
        return None
    s = cv2.approxPolyDP((pts * 100).astype(np.int32).reshape(-1, 1, 2), eps_km * 100, True)
    s = s.reshape(-1, 2) / 100.0
    return [[round(float(x), 2), round(float(y), 2)] for x, y in s] if len(s) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "korea_map.json"))
    a = ap.parse_args()
    prov = load(a.src, "prov.json", "skorea_provinces_geo.json")
    muni = load(a.src, "muni.json", "skorea_municipalities_geo.json")
    out = {"projection": {"lat0": LAT0, "lon0": 127.5, "unit": "km"}, "provinces": [], "jeonbuk": []}
    for f in prov["features"]:
        p = f["properties"]
        polys = [q for q in (simplify(project(r), 0.35, 4.0) for r in rings(f["geometry"])) if q]
        out["provinces"].append({"code": p["code"], "name": p["name"], "polys": polys})
    for f in muni["features"]:
        p = f["properties"]
        if not p["code"].startswith("35"):
            continue
        polys = [q for q in (simplify(project(r), 0.12, 0.3) for r in rings(f["geometry"])) if q]
        out["jeonbuk"].append({"code": p["code"], "name": p["name"], "polys": polys})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print("wrote", a.out, os.path.getsize(a.out), "bytes;",
          sum(len(q) for p in out["provinces"] for q in p["polys"]), "province pts,",
          sum(len(q) for p in out["jeonbuk"] for q in p["polys"]), "jeonbuk pts")


if __name__ == "__main__":
    main()
