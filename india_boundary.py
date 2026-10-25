import json, os
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep

_HERE = os.path.dirname(os.path.abspath(__file__))
_gj = json.load(open(os.path.join(_HERE, "data", "india.geojson"), encoding="utf-8"))
_INDIA = prep(unary_union([shape(f["geometry"]) for f in _gj["features"]]))

def inside_india(lat, lon):
    """True if the point (lat, lon) lies inside India's boundary."""
    return _INDIA.contains(Point(lon, lat))