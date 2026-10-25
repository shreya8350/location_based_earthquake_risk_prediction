import os, time, joblib
import numpy as np, pandas as pd, streamlit as st
import folium
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from features import *
from india_boundary import inside_india, _INDIA

st.set_page_config(page_title="India Seismic Activity Estimator", page_icon=None, layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
MODEL_PATH = os.path.join(HERE, "final_model_india.joblib")
BOUNDARY_PATH = os.path.join(HERE, "data", "india.geojson")
NOW = 2026.0
DEFAULT_LAT, DEFAULT_LON = 25.57, 91.88   # Shillong
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 7.0, 36.0, 68.0, 97.5

COLORS = {"Lower": "#2E7D32", "Moderate": "#ED8B00", "Higher": "#C62828"}
DESCRIPTION = {
    "Lower": "Fewer strong earthquakes nearby than most of India. This is a relative comparison, "
             "not a guarantee of safety.",
    "Moderate": "About as active as a typical part of India. Strong earthquakes can still occur.",
    "Higher": "Among the most active 20% of areas in India. This does not mean an earthquake is imminent.",
}

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.9rem !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    .level-badge { display:inline-block; padding:4px 12px; border-radius:4px;
                   color:white; font-weight:600; font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)

def prob(la, lo, ev, art):
    x = pd.DataFrame([features(la, lo, NOW, ev, None)])[art["features"]]
    raw = art["model"].predict_proba(x)[:, 1]
    if art["method"] == "isotonic":
        return float(art["cal"].predict(raw)[0])
    return float(art["cal"].predict_proba(logit(raw))[0, 1])

@st.cache_resource(show_spinner="Loading model and drawing the India map (first load only)...")
def build(model_stamp):
    ev = load_catalog()
    faults = load_faults()
    hist = pd.read_csv("data/historical_great.csv")
    art = joblib.load(MODEL_PATH)
    q40, q80 = float(art["q40"]), float(art["q80"])

    rows = []
    for la, lo in art["cells"]:
        p = prob(la, lo, ev, art)
        level = "Lower" if p < q40 else ("Moderate" if p < q80 else "Higher")
        rows.append((la, lo, p, level))
    grid = pd.DataFrame(rows, columns=["lat", "lon", "p", "level"])

    # Build one polygon per level from the nearest-cell Voronoi regions, then clip to India.
    # Voronoi cells tile the plane, so the coloured area covers India edge to edge.
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon, MultiPoint
    from shapely.ops import voronoi_diagram
    pts = MultiPoint([(lo, la) for la, lo in zip(grid.lat, grid.lon)])
    regions = voronoi_diagram(pts, envelope=Polygon([(LON_MIN - 1, LAT_MIN - 1), (LON_MAX + 1, LAT_MIN - 1),
                                                     (LON_MAX + 1, LAT_MAX + 1), (LON_MIN - 1, LAT_MAX + 1)]))
    india = _INDIA.context
    india_shape = unary_union([shape(f["geometry"]) for f in
                               __import__("json").load(open(BOUNDARY_PATH, encoding="utf-8"))["features"]])

    level_polys = {"Lower": [], "Moderate": [], "Higher": []}
    for region in regions.geoms:
        # find the grid cell this Voronoi region belongs to
        c = region.representative_point()
        d2 = (grid.lon.values - c.x) ** 2 + (grid.lat.values - c.y) ** 2
        lvl = grid.level.values[int(np.argmin(d2))]
        clipped = region.intersection(india_shape)
        if not clipped.is_empty:
            level_polys[lvl].append(clipped)

    layers = {k: unary_union(v) for k, v in level_polys.items() if v}
    return ev, faults, hist, art, q40, q80, layers, india_shape

stamp = os.path.getmtime(MODEL_PATH)
ev, faults, hist, art, q40, q80, layers, india_shape = build(stamp)

# ---------- session state ----------
if "sel" not in st.session_state:
    st.session_state.sel = (DEFAULT_LAT, DEFAULT_LON)
    st.session_state.ver = 0
st.session_state.setdefault("last_click", None)
st.session_state.setdefault("last_geo", None)

def set_selection(la, lo):
    st.session_state.sel = (la, lo)
    st.session_state.ver += 1

# ---------- sidebar: inputs ----------
with st.sidebar:
    st.header("Location")
    st.caption("Choose a point in India using any of the methods below.")

    st.subheader("1. Type coordinates")
    ver = st.session_state.ver
    with st.form("location_form"):
        cur_lat, cur_lon = st.session_state.sel
        f_lat = st.number_input("Latitude (°N)", LAT_MIN, LAT_MAX, value=float(cur_lat),
                                step=0.01, format="%.4f", key=f"lat_{ver}")
        f_lon = st.number_input("Longitude (°E)", LON_MIN, LON_MAX, value=float(cur_lon),
                                step=0.01, format="%.4f", key=f"lon_{ver}")
        submitted = st.form_submit_button("Show result", type="primary", use_container_width=True)
    if submitted:
        st.session_state.sel = (round(f_lat, 4), round(f_lon, 4))

    st.subheader("2. Use my current location")
    geo = streamlit_geolocation()
    st.caption("Your browser will ask for permission. Accuracy depends on your device.")
    if geo and geo.get("latitude") is not None:
        new = (round(geo["latitude"], 4), round(geo["longitude"], 4))
        if new != st.session_state.last_geo:
            st.session_state.last_geo = new
            set_selection(*new)
            st.rerun()

    st.subheader("3. Click the map")
    st.caption("Click any point on the map to select it.")

    st.divider()
    st.caption("This tool estimates recent seismic activity. It is not an earthquake prediction.")

lat, lon = st.session_state.sel

# ---------- header ----------
st.title("India Seismic Activity Estimator")
st.write("Estimated chance of at least one magnitude 5.0 or stronger earthquake within 50 km "
         "over the next 10 years, based on earthquake records from 1970 to 2026.")

tab_map, tab_result, tab_method = st.tabs(["Map", "Result", "Method and limitations"])

# ---------- map tab ----------
with tab_map:
    st.caption("Colours show the activity level of each area relative to India. "
               "Click a point to select it.")
    m = folium.Map(location=[22.5, 82.5], zoom_start=4, tiles="OpenStreetMap")

    # filled vector layers, one per level, clipped exactly to the India boundary
    for level in ("Lower", "Moderate", "Higher"):
        if level in layers:
            folium.GeoJson(
                mapping(layers[level]),
                name=level,
                style_function=lambda _, c=COLORS[level]: {"fillColor": c, "color": c,
                                                           "weight": 0, "fillOpacity": 0.6},
                interactive=False,
            ).add_to(m)

    # India outline on top
    folium.GeoJson(
        mapping(india_shape),
        name="India boundary",
        style_function=lambda _: {"color": "#333333", "weight": 1.2, "fillOpacity": 0},
        interactive=False,
    ).add_to(m)

    folium.Marker([lat, lon], tooltip="Selected location",
                  icon=folium.Icon(color="black", icon="info-sign")).add_to(m)

    out = st_folium(m, width=None, height=560, returned_objects=["last_clicked"])
    if out and out.get("last_clicked"):
        c = out["last_clicked"]
        new = (round(c["lat"], 4), round(c["lng"], 4))
        if new != st.session_state.last_click:
            st.session_state.last_click = new
            set_selection(*new)
            st.rerun()

    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown(f'<span class="level-badge" style="background:{COLORS["Lower"]}">Lower</span> quieter than most of India', unsafe_allow_html=True)
    lc2.markdown(f'<span class="level-badge" style="background:{COLORS["Moderate"]}">Moderate</span> about average', unsafe_allow_html=True)
    lc3.markdown(f'<span class="level-badge" style="background:{COLORS["Higher"]}">Higher</span> more active than most of India', unsafe_allow_html=True)

# ---------- result tab ----------
with tab_result:
    st.subheader(f"Location: {lat:.4f}°N, {lon:.4f}°E")

    if not inside_india(lat, lon):
        st.warning("This location is outside India. The tool gives estimates only for places inside India. "
                   "Select another point on the map or enter different coordinates in the sidebar.")
    else:
        p = prob(lat, lon, ev, art)
        level = "Lower" if p < q40 else ("Moderate" if p < q80 else "Higher")

        k1, k2, k3 = st.columns(3)
        k1.metric("Activity level", level)
        k2.metric("Probability over 10 years", f"{p*100:.1f}%")
        k3.metric("Equivalent frequency", f"about 1 in {int(round(1 / max(p, 0.001), -1))}")

        st.markdown(f"**Interpretation.** {DESCRIPTION[level]}")
        st.write(f"Of 100 locations with similar recent activity, about {p*100:.0f} would experience "
                 "at least one M5.0+ earthquake within 50 km during the next 10 years.")

        st.caption(f"Level thresholds across India: Lower below {q40*100:.1f}%; "
                   f"Moderate {q40*100:.1f}% to {q80*100:.1f}%; Higher above {q80*100:.1f}%.")

        with st.expander("Regional context not captured by the model"):
            st.write("India's BIS seismic zoning (IS 1893) places the Himalayan and Northeast regions in "
                     "Zone V, the highest hazard category. This model uses recent earthquake records only and "
                     "can understate hazard where very large earthquakes are rare but possible. "
                     "Consult the official BIS map for site-specific decisions.")
            dist = hav(lat, lon, hist.latitude.values, hist.longitude.values)
            near = hist.assign(distance_km=dist.round(0))[dist <= 150].sort_values("distance_km")
            if len(near):
                st.write("Major earthquakes (magnitude 6.5 or more) before 1970 within 150 km:")
                st.dataframe(near[["time", "mag", "place", "distance_km"]], hide_index=True)
            else:
                st.write("No major earthquakes (magnitude 6.5 or more) before 1970 within 150 km in the dataset.")

# ---------- method tab ----------
with tab_method:
    st.subheader("How the estimate is produced")
    st.write("The model divides India into grid cells of 0.5° and, for each cell, counts past M5+ earthquakes "
             "within 50, 100 and 200 km, their recency and their maximum magnitude. A LightGBM gradient-boosting "
             "model learns from 1990 to 2025 how often such patterns were followed by an M5+ earthquake within "
             "50 km over the next 10 years. Probabilities are then calibrated with Platt scaling. "
             "The map colours each area by its nearest scored cell.")

    st.subheader("Performance on held-out years")
    s1, s2, s3 = st.columns(3)
    s1.metric("ROC-AUC (ranking ability)", "0.89", "baseline 0.78", delta_color="off")
    s2.metric("Brier score (lower is better)", "0.065", delta_color="off")
    s3.metric("Average positive rate in test", "8%", delta_color="off")
    st.caption("Tested on five out-of-time years (2006 to 2015), each predicted using only earlier data.")

    st.subheader("Limitations")
    st.markdown(
        "- Uses only the 1970 to 2026 earthquake catalogue. Regions with major historical earthquakes "
        "but few recent ones may be underestimated.\n"
        "- Early catalogue years are incomplete for small earthquakes.\n"
        "- Peninsular India has few M5+ events, so estimates there rest on limited data.\n"
        "- Estimates can shift by several percentage points between model versions.\n"
        "- The India boundary is approximate and may differ from official maps in disputed areas.\n"
        "- This is an activity estimate. It does not predict when or where an earthquake will occur."
    )

    with st.expander("Technical details"):
        st.write(f"Model: LightGBM (200 trees, learning rate 0.03) · Calibration: {art['method']} · "
                 f"Model file saved: {time.ctime(stamp)}")