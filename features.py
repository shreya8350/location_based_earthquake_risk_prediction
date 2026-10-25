import json, os
import numpy as np, pandas as pd

R, HORIZON = 50, 10

def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def load_catalog():
    cat = pd.read_csv("data/usgs_india.csv", parse_dates=["time"])
    t = (cat["time"].dt.year + (cat["time"].dt.dayofyear - 1) / 365.25).values
    return cat.latitude.values, cat.longitude.values, t, cat.mag.values

def load_faults(path="data/faults.geojson"):
    if not os.path.exists(path):
        return None
    gj = json.load(open(path, encoding="utf-8"))
    pts = []
    def add(line):
        arr = np.array([[c[1], c[0]] for c in line], dtype=float)   # lat, lon
        for a, b in zip(arr[:-1], arr[1:]):
            n = max(1, int(np.hypot(*(b - a)) / 0.05))
            for k in range(n + 1):
                p = a + (b - a) * k / n
                if 19 <= p[0] <= 33 and 85 <= p[1] <= 101:
                    pts.append(p)
    for f in gj["features"]:
        g = f.get("geometry")
        if not g:
            continue
        if g["type"] == "LineString":
            add(g["coordinates"])
        elif g["type"] == "MultiLineString":
            for line in g["coordinates"]:
                add(line)
    return np.array(pts) if pts else None

_fault_cache = {}
def fault_distance(la, lo, faults):
    if faults is None:
        return np.nan
    key = (round(la, 3), round(lo, 3))
    if key not in _fault_cache:
        _fault_cache[key] = float(hav(la, lo, faults[:, 0], faults[:, 1]).min())
    return _fault_cache[key]

def features(la, lo, cutoff, ev, faults=None):
    ev_lat, ev_lon, ev_t, ev_m = ev
    d = hav(la, lo, ev_lat, ev_lon)
    past = ev_t < cutoff
    n50, n100, n200 = d <= 50, d <= 100, d <= 200

    p50 = past & n50
    m5 = p50 & (ev_m >= 5.0)
    r10 = p50 & (ev_t >= cutoff - 10)
    last5 = ev_t[m5]

    p100 = past & n100 & (ev_m >= 5.0)
    p200 = past & n200 & (ev_m >= 5.0)
    r10_100 = past & n100 & (ev_t >= cutoff - 10)
    last5_100 = ev_t[p100]

    return {
        "n5_5y": int((m5 & (ev_t >= cutoff - 5)).sum()),
        "n5_10y": int((m5 & (ev_t >= cutoff - 10)).sum()),
        "n5_20y": int((m5 & (ev_t >= cutoff - 20)).sum()),
        "n5_all": int(m5.sum()),
        "n45_5y": int((p50 & (ev_m >= 4.5) & (ev_t >= cutoff - 5)).sum()),
        "max_mag_10y": ev_m[r10].max() if r10.any() else 0.0,
        "years_since_m5": cutoff - last5.max() if len(last5) else 50.0,
        "baseline_p": 1 - np.exp(-((m5 & (ev_t >= cutoff - 20)).sum() / 20) * HORIZON),
        "n5_100_10y": int((p100 & (ev_t >= cutoff - 10)).sum()),
        "n5_100_all": int(p100.sum()),
        "n5_200_10y": int((p200 & (ev_t >= cutoff - 10)).sum()),
        "n5_200_all": int(p200.sum()),
        "max_mag_100_10y": ev_m[r10_100].max() if r10_100.any() else 0.0,
        "years_since_m5_100": cutoff - last5_100.max() if len(last5_100) else 50.0,
        "dist_fault_km": fault_distance(la, lo, faults),
    }

def label(la, lo, cutoff, ev):
    ev_lat, ev_lon, ev_t, ev_m = ev
    d = hav(la, lo, ev_lat, ev_lon)
    return int(((d <= R) & (ev_t >= cutoff) & (ev_t < cutoff + HORIZON) & (ev_m >= 5.0)).any())