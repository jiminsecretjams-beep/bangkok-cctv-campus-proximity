"""Bangkok School Zone: an offline-first dashboard of verified CCTV observations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# Keep the portable app importable when invoked from another working directory
# or by Streamlit's isolated AppTest runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard_data import (  # noqa: E402
    PERIOD_NAMES,
    VEHICLES,
    add_segments,
    anchor_summary,
    csv_bytes,
    hour_profile,
    load_bundle,
    paired_periods,
    period_summary,
    select_context,
    select_rows,
    timeline,
)

HERE = Path(os.environ.get("BKK_DASHBOARD_DATA", Path(__file__).resolve().parents[1] / "results/dashboard"))
BLUE, TEAL, CORAL, GRAY = "#176B87", "#238D80", "#CB735D", "#AAB7BF"
VIEWS = ["Overview", "Camera explorer", "School context", "Coverage & sources"]
st.set_page_config(page_title="Bangkok · School Zone", page_icon="◉", layout="wide")
st.markdown(
    """
<style>
 .block-container {max-width:1480px; padding-top:2rem; padding-bottom:3rem;}
 h1 {font-size:2.55rem!important; letter-spacing:-1.5px; font-weight:750!important;}
 h2 {font-size:1.5rem!important; letter-spacing:-.45px;}
 h3 {font-size:1.1rem!important;}
 .kicker {font-size:.71rem;letter-spacing:2.2px;font-weight:750;color:#176B87;margin-bottom:.25rem;}
 .study-tag {display:inline-block;border:1px solid #d4e0e5;border-radius:30px;padding:7px 13px;
 font-size:.72rem;color:#36586a;letter-spacing:.5px;background:#fff;}
 .brand {font-size:1.5rem;font-weight:750;letter-spacing:-.7px;line-height:1.2;}
 .brand-small {font-size:.66rem;letter-spacing:2px;opacity:.65;margin-bottom:.65rem;}
 .side-note {font-size:.78rem;opacity:.72;line-height:1.55;}
 [data-testid="stMetricValue"] {font-size:2.25rem;font-weight:720;letter-spacing:-1px;}
 [data-testid="stMetricLabel"] {font-weight:650;}
 [class*="st-key-kpi_"] {background:white;border-radius:12px;}
 [data-testid="stSidebar"] .block-container {padding-top:2rem;}
 div[data-testid="stAlert"] {border-radius:10px;}
 footer {visibility:hidden;}
 @media (max-width:1100px) {
 .st-key-overview_kpis [data-testid="stHorizontalBlock"] {flex-wrap:wrap!important;}
 .st-key-overview_kpis [data-testid="stColumn"] {width:calc(50% - 1rem)!important;
 min-width:calc(50% - 1rem)!important;flex:1 1 calc(50% - 1rem)!important;}
 .st-key-overview_visuals [data-testid="stHorizontalBlock"],
 .st-key-explorer_secondary [data-testid="stHorizontalBlock"] {flex-direction:column!important;}
 .st-key-overview_visuals [data-testid="stColumn"],
 .st-key-explorer_secondary [data-testid="stColumn"] {width:100%!important;flex:1 1 100%!important;}
 [data-testid="stMetricValue"] {font-size:1.9rem!important;}
 }
 @media (max-width:760px) {.block-container {padding:1rem;} h1 {font-size:2rem!important;}}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_data(version):
    return load_bundle(HERE)


try:
    version = (HERE / "manifest.json").stat().st_mtime_ns
    data = cached_data(version)
except (OSError, ValueError, KeyError) as error:
    st.error("The dashboard data is missing or has changed. Restore the complete dashboard folder.")
    st.caption(str(error))
    st.stop()

DATES = data["manifest"]["dates"]


def reset_filters():
    for key in [
        "vehicle",
        "date_range",
        "day_type",
        "distance",
        "education",
        "choose_cameras",
        "specific_cameras",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def pick_camera():
    st.session_state["active_camera"] = st.session_state["camera_picker"]


def map_selected():
    event = st.session_state.get("camera_map", {})
    objects = event.get("selection", {}).get("objects", {}).get("camera-views", [])
    if objects:
        st.session_state["active_camera"] = str(objects[0]["camera_id"])
        st.session_state["camera_picker"] = str(objects[0]["camera_id"])
        st.session_state["view"] = "Camera explorer"


with st.sidebar:
    st.markdown(
        '<div class="brand-small">BANGKOK / DATA OBSERVATORY</div><div class="brand">School<br>Zone.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="side-note">Explore visible traffic around educational institutions.</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    vehicle = st.selectbox("Vehicle class", list(VEHICLES), key="vehicle")
    date_range = st.date_input(
        "Study dates",
        value=tuple(pd.to_datetime([DATES[0], DATES[-1]]).date),
        min_value=pd.Timestamp(DATES[0]).date(),
        max_value=pd.Timestamp(DATES[-1]).date(),
        format="DD/MM/YYYY",
        key="date_range",
    )
    day_type = st.selectbox("Day type", ["Both day types", "Weekdays only", "Weekend only"], key="day_type")
    with st.expander("School proximity & cameras"):
        band = st.selectbox(
            "Nearest school candidate", ["All distances", "0-250m", "250-500m", "500-1500m"], key="distance"
        )
        education = st.selectbox(
            "University / college context",
            [
                "All education contexts",
                "Also near a university / college",
                "No university / college catalogue match",
            ],
            key="education",
        )
        choose = st.toggle("Choose specific cameras", key="choose_cameras")
        specific = st.multiselect(
            "Camera IDs",
            sorted(data["context"].camera_id, key=int),
            default=["139", "150", "239", "1060"],
            disabled=not choose,
            key="specific_cameras",
        )
        st.caption("Distances use screened catalogue reference points, not verified school gates.")
    st.button("Reset filters", on_click=reset_filters, width="stretch")
    st.divider()
    st.markdown(
        '<div class="side-note">Whole-image counts, including parked vehicles.<br>'
        "Bangkok collection time · UTC+7<br>Counts are sightings per snapshot.</div>",
        unsafe_allow_html=True,
    )

if not isinstance(date_range, (tuple, list)):
    date_range = (date_range, date_range)
if len(date_range) < 2:
    st.info("Choose an end date to complete the date range.")
    st.stop()
dates = [x for x in DATES if str(date_range[0]) <= x <= str(date_range[1])]
if day_type != "Both day types":
    dates = [x for x in dates if (pd.Timestamp(x).weekday() >= 5) == (day_type == "Weekend only")]
context = select_context(data["context"], band, education, specific if choose else None)
cameras = context.camera_id.tolist()
metric = VEHICLES[vehicle]
hours = select_rows(data["hourly"], cameras, dates)
frames = select_rows(data["frames"], cameras, dates)
pairs = paired_periods(data["daily"], cameras, dates, metric)
summary = period_summary(pairs)

left, right = st.columns([3, 1], vertical_alignment="center")
with left:
    st.markdown('<div class="kicker">THE SCHOOL ZONE COMMUTE TRACKER</div>', unsafe_allow_html=True)
    st.title("A closer look at the commute.")
with right:
    st.markdown('<div class="study-tag">25–31 AUG 2026 · BANGKOK</div>', unsafe_allow_html=True)
st.caption(
    f"{len(cameras)} camera views · {len(frames):,} captures in selection · "
    f"{int(frames.eligible_temporal.sum()):,} usable · {vehicle.lower()}"
)
view = st.segmented_control("Dashboard view", VIEWS, default="Overview", key="view", width="stretch")
st.write("")
if not cameras or not dates:
    st.info("No observations match these filters. Change the selection or reset the filters.")
    st.stop()


def styled(chart, height=270):
    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="#E6EDF1",
            labelColor="#49616F",
            titleColor="#49616F",
            titleFontWeight=400,
            domain=False,
            tickColor="#D7E2E8",
        )
        .configure_legend(title=None, labelColor="#49616F")
    )


def show_chart(chart, height=270):
    st.altair_chart(styled(chart, height), width="stretch", theme=None)


def pct(value):
    return "—" if pd.isna(value) else f"{value:+.1f}%"


def camera_selector():
    selected = st.session_state.get("active_camera", "139")
    if selected not in cameras:
        selected = cameras[0]
    if st.session_state.get("camera_picker") not in cameras:
        st.session_state["camera_picker"] = selected
    result = st.selectbox(
        "Camera view",
        cameras,
        index=None,
        key="camera_picker",
        on_change=pick_camera,
        format_func=lambda x: f"{x} · " + str(context.set_index("camera_id").loc[x, "location"])[:75],
    )
    if result is None:
        st.info("Choose a camera view to continue.")
        st.stop()
    st.session_state["active_camera"] = result
    if context.set_index("camera_id").loc[result, "location_caution"] == "known_metadata_label_discrepancy":
        st.warning(
            "The supplied location labels disagree for this view. Treat its map position and nearby "
            "institution links as provisional; this does not invalidate its temporal counts."
        )
    return result


def draw_map(table, selected=None, schools=None, clickable=False):
    q = table.copy()
    if "change" not in q:
        q["change"] = np.nan
    q["change_label"] = q.change.map(lambda v: "Not estimable" if pd.isna(v) else f"{v:+.1f}%")
    q["point_label"] = "Camera " + q.camera_id
    q["marker_color"] = q.change.map(
        lambda v: (
            [170, 183, 191, 220] if pd.isna(v) else [23, 107, 135, 220] if v >= 0 else [203, 115, 93, 220]
        )
    )
    if selected is not None:
        q.loc[q.camera_id.eq(selected), "marker_color"] = pd.Series(
            [[238, 158, 75, 255]], index=q.index[q.camera_id.eq(selected)]
        )
    layer = pdk.Layer(
        "ScatterplotLayer",
        q,
        id="camera-views",
        get_position=["longitude", "latitude"],
        get_fill_color="marker_color",
        get_radius=100,
        radius_min_pixels=5,
        radius_max_pixels=12,
        pickable=True,
        auto_highlight=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )
    layers = [layer]
    if schools is not None and not schools.empty:
        locations = schools.rename(columns={"institution_name": "location"}).copy()
        locations["camera_id"] = "Institution reference"
        locations["point_label"] = "Institution reference"
        locations["change_label"] = "Calendar context only"
        layers.insert(
            0,
            pdk.Layer(
                "ScatterplotLayer",
                locations,
                id="institution-points",
                get_position=["longitude", "latitude"],
                get_fill_color=[203, 115, 93, 150],
                get_radius=65,
                radius_min_pixels=4,
                pickable=True,
            ),
        )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=float(q.latitude.median()),
            longitude=float(q.longitude.median()),
            zoom=13 if len(q) == 1 else 10.4,
            pitch=0,
        ),
        map_provider="carto",
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{point_label}\n{location}\n{change_label}"},
    )
    if clickable:
        st.pydeck_chart(
            deck, height=330, key="camera_map", on_select=map_selected, selection_mode="single-object"
        )
    else:
        st.pydeck_chart(deck, height=300)


def overview():
    st.subheader("When does the weekday difference appear?")
    st.caption(
        "Matched views, matched clock hours. Each comparison needs at least three complete weekdays "
        "and both weekend days. Calendar labels do not establish actual school attendance."
    )
    with st.container(key="overview_kpis"):
        columns = st.columns(4)
    for column, ident, title in zip(
        columns[:3], PERIOD_NAMES, ["Morning · 07–09", "Midday · 10–14", "Afternoon · 15–17"]
    ):
        r = summary[summary.period.eq(ident)].iloc[0]
        with column.container(border=True, key="kpi_" + ident):
            st.metric(
                title,
                pct(r.median_pct),
                help="Median within-camera percentage change from weekend to weekday.",
            )
            st.caption(f"{int(r.ratio_cameras)} stable camera ratios · weekday vs weekend")
    with columns[3].container(border=True, key="kpi_views"):
        st.metric("Balanced camera views", pairs.camera_id.nunique())
        st.caption(f"Of {len(cameras)} selected views · same accepted days")
    if summary.status.eq("insufficient").any():
        st.info(
            "A network percentage needs at least 10 stable camera ratios. Keep both weekend dates and "
            "at least three weekdays, broaden the camera selection, or inspect an individual view."
        )
    with st.container(key="overview_visuals"):
        a, b = st.columns([1, 1.15], gap="large")
    with a.container(border=True):
        st.subheader("A morning signature, a quieter midday")
        q = summary[summary.status.eq("estimated")].copy()
        if q.empty:
            st.caption("No balanced network comparison is available for this selection.")
        else:
            q["display"] = q.median_pct.map(pct)
            base = alt.Chart(q).encode(
                x=alt.X("label:N", sort=list(PERIOD_NAMES.values()), title=None, axis=alt.Axis(labelAngle=0)),
                tooltip=[
                    alt.Tooltip("label:N", title="Window"),
                    alt.Tooltip("median_pct:Q", title="Median change (%)", format="+.1f"),
                    alt.Tooltip("ratio_cameras:Q", title="Camera ratios"),
                    alt.Tooltip("positive_cameras:Q", title="Higher on weekdays"),
                ],
            )
            bars = base.mark_bar(size=58, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                y=alt.Y("median_pct:Q", title="Weekday vs weekend change (%)"),
                color=alt.condition(alt.datum.median_pct >= 0, alt.value(BLUE), alt.value(CORAL)),
            )
            bounds = base.mark_rule(color="#183B4A", strokeWidth=2).encode(y="q25:Q", y2="q75:Q")
            show_chart(bars + bounds, 285)
            st.caption(
                "Bars: median change. Lines: middle 50% of camera differences, not a confidence interval."
            )
    with b.container(border=True):
        st.subheader("Explore the camera views")
        period = st.selectbox(
            "Colour by time window",
            list(PERIOD_NAMES),
            format_func=PERIOD_NAMES.get,
            label_visibility="collapsed",
            key="map_period",
        )
        effects = pairs[pairs.period.eq(period)][["camera_id", "relative_pct"]].rename(
            columns={"relative_pct": "change"}
        )
        points = context.merge(effects, on="camera_id", how="left", validate="one_to_one")
        offline = st.toggle("Coordinate view · works offline", key="offline_map")
        if offline:
            available = points[points.change.notna()]
            missing = points[points.change.isna()].copy()
            missing["Comparison"] = "Not estimable"
            chart = (
                alt.Chart(available)
                .mark_circle(size=90, opacity=0.8)
                .encode(
                    x=alt.X("longitude:Q", scale=alt.Scale(zero=False), title="Longitude"),
                    y=alt.Y("latitude:Q", scale=alt.Scale(zero=False), title="Latitude"),
                    color=alt.Color(
                        "change:Q",
                        scale=alt.Scale(range=[CORAL, "#E5EEF2", BLUE], domainMid=0),
                        title="Change (%)",
                    ),
                    tooltip=[
                        "camera_id:N",
                        "location:N",
                        alt.Tooltip("change:Q", format="+.1f", title="Weekday change (%)"),
                    ],
                )
            )
            # A separate layer avoids Vega dropping null colour values before
            # the conditional colour expression is evaluated.
            unknown = (
                alt.Chart(missing)
                .mark_circle(size=90, color=GRAY, opacity=0.8)
                .encode(
                    x=alt.X("longitude:Q", scale=alt.Scale(zero=False), title="Longitude"),
                    y=alt.Y("latitude:Q", scale=alt.Scale(zero=False), title="Latitude"),
                    tooltip=["camera_id:N", "location:N", "Comparison:N"],
                )
            )
            show_chart((chart + unknown).interactive(), 285)
        else:
            draw_map(points, clickable=True)
        st.caption(
            "Blue: higher weekdays · coral: lower · grey: not estimable. "
            + (
                "Hover to inspect; choose a view in Camera explorer. "
                if offline
                else "Click a camera to explore. "
            )
            + "Positions and school links are provisional."
        )
    st.subheader("The shape across the day")
    profile = hour_profile(data["hourly"], cameras, dates, metric)
    valid = profile.dropna(subset=["median_pct"]).copy()
    if valid.empty:
        st.caption("Not enough matched camera/hour ratios for this selection.")
    else:
        valid["profile_segment"] = valid.hour_local.diff().ne(1).cumsum().astype(str)
        base = alt.Chart(valid).encode(
            x=alt.X("hour_local:Q", title="Bangkok collection hour", axis=alt.Axis(tickMinStep=1)),
            detail="profile_segment:N",
        )
        area = base.mark_area(color=BLUE, opacity=0.12).encode(
            y=alt.Y("q25:Q", title="Weekday vs weekend change (%)"), y2="q75:Q"
        )
        line = base.mark_line(point=True, color=BLUE, strokeWidth=3).encode(
            y="median_pct:Q",
            tooltip=[
                alt.Tooltip("hour_local:Q", title="Hour"),
                alt.Tooltip("median_pct:Q", title="Median change (%)", format="+.1f"),
                "cameras:Q",
            ],
        )
        zero = (
            alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(strokeDash=[5, 5], color=GRAY).encode(y="zero:Q")
        )
        show_chart(area + line + zero, 240)
        st.caption(
            "This flexible matched-hour panel has its own camera count at each hour. It is separate "
            "from the stricter balanced-day cards above. Shading shows camera variation."
        )
    st.download_button(
        "Download this comparison", csv_bytes(pairs), "camera_period_comparison.csv", "text/csv"
    )


def coverage_chart(q, height=210):
    if q.empty:
        st.caption("No hourly coverage cells in the selection.")
        return
    chart = (
        alt.Chart(q)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("hour_local:O", title="Bangkok collection hour"),
            y=alt.Y("date_local:O", title=None, sort="ascending"),
            color=alt.condition(
                alt.datum.coverage_pass,
                alt.Color(
                    "usable_coverage:Q",
                    scale=alt.Scale(domain=[0, 1], range=["#DFECF0", BLUE]),
                    title="Cadence coverage",
                ),
                alt.value("#DCE3E8"),
            ),
            tooltip=[
                alt.Tooltip("date_local:N", title="Date"),
                alt.Tooltip("hour_local:O", title="Hour"),
                alt.Tooltip("n_acquired:Q", title="Acquired"),
                alt.Tooltip("n_usable:Q", title="Usable"),
                alt.Tooltip("coverage_pass:N", title="Accepted"),
            ],
        )
    )
    show_chart(chart, height)
    st.caption("Grey cells fail the coverage rule. They mean unavailable, not zero traffic.")


def camera_explorer():
    camera = camera_selector()
    meta = context.set_index("camera_id").loc[camera]
    q = frames[frames.camera_id.eq(camera)].copy()
    good = q[q.eligible_temporal]
    a, b, c = st.columns(3)
    a.metric("Usable captures", f"{len(good):,}")
    b.metric("Mean per usable snapshot", "—" if good.empty else f"{good[metric + '_count'].mean():.2f}")
    c.metric("Excluded captures", f"{len(q) - len(good):,}")
    st.caption(
        f"Nearest screened school candidate: {meta.screened_nearest_school_name} · "
        f"{meta.screened_nearest_school_distance_m:,.0f} m to a reference point. School-gate interaction is unverified."
    )
    mode = st.radio(
        "Time-series detail", ["Hourly means", "Individual snapshots"], horizontal=True, key="detail"
    )
    if mode == "Hourly means":
        series = timeline(data["hourly"], camera, dates, metric)
        if series.empty:
            st.info("No hours pass the quality and coverage rules in this view.")
        else:
            chart = (
                alt.Chart(series)
                .mark_line(color=BLUE, point=alt.OverlayMarkDef(size=18))
                .encode(
                    x=alt.X("time:T", title="Bangkok collection time"),
                    y=alt.Y(
                        "value:Q", title="Mean sightings per usable snapshot", scale=alt.Scale(zero=True)
                    ),
                    detail="segment:N",
                    tooltip=[
                        alt.Tooltip("time:T", title="Bangkok time", format="%d %b %H:%M"),
                        alt.Tooltip("value:Q", title="Mean sightings", format=".2f"),
                        alt.Tooltip("n_usable:Q", title="Usable captures"),
                    ],
                )
                .interactive(bind_y=False)
            )
            show_chart(chart, 290)
        st.caption(
            "Lines break across excluded hours and overnight gaps. Missing counts are never drawn as zero."
        )
    else:
        q["time"] = q.observed_at_utc.dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
        q["value"] = q[metric + "_count"]
        series = add_segments(q, gap_minutes=20)
        if not series.empty:
            chart = (
                alt.Chart(series)
                .mark_line(color=BLUE, point=alt.OverlayMarkDef(size=12))
                .encode(
                    x=alt.X("time:T", title="Bangkok collection time"),
                    y=alt.Y("value:Q", title="Sightings per snapshot"),
                    detail="segment:N",
                    tooltip=[
                        alt.Tooltip("time:T", format="%d %b %H:%M", title="Bangkok time"),
                        alt.Tooltip("value:Q", title="Sightings"),
                        "quality_status:N",
                    ],
                )
                .interactive(bind_y=False)
            )
            show_chart(chart, 290)
        else:
            st.info("No usable snapshot counts for this selection.")
        if st.toggle("Show raw predictions for quality-flagged captures", key="raw_diagnostic"):
            raw = q[~q.eligible_temporal].copy()
            if raw.empty:
                st.caption("There are no excluded captures in this selection.")
            else:
                raw["raw_prediction"] = raw["raw_" + metric + "_count"]
                st.caption("Diagnostic only: these predictions are excluded from all analytical comparisons.")
                show_chart(
                    alt.Chart(raw)
                    .mark_point(color=CORAL, size=45)
                    .encode(
                        x=alt.X("time:T", title="Bangkok collection time"),
                        y=alt.Y("raw_prediction:Q", title="Excluded raw prediction"),
                        tooltip=[
                            alt.Tooltip("time:T", format="%d %b %H:%M"),
                            "raw_prediction:Q",
                            "quality_status:N",
                            "quality_reason:N",
                        ],
                    ),
                    150,
                )
    with st.container(key="explorer_secondary"):
        a, b = st.columns([1.5, 1], gap="large")
    with a:
        st.subheader("Can we compare these hours?")
        coverage_chart(hours[hours.camera_id.eq(camera)])
    with b:
        st.subheader("Where is this view?")
        draw_map(context[context.camera_id.eq(camera)], selected=camera)
    st.subheader("Weekdays and weekends in this view")
    selected = pairs[pairs.camera_id.eq(camera)]
    if selected.empty:
        st.info(
            "This selection does not contain three complete weekdays and both weekend days. "
            "The time series remains available, but a balanced comparison is not estimated."
        )
    else:
        st.dataframe(
            selected.assign(Window=selected.period.map(PERIOD_NAMES))[
                ["Window", "weekday_mean", "weekend_mean", "relative_pct", "weekday_days", "weekend_days"]
            ].rename(
                columns={
                    "weekday_mean": "Weekday mean",
                    "weekend_mean": "Weekend mean",
                    "relative_pct": "Change (%)",
                    "weekday_days": "Weekday dates",
                    "weekend_days": "Weekend dates",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    st.download_button(
        "Download camera observations", csv_bytes(q), f"camera_{camera}_observations.csv", "text/csv"
    )
    st.subheader("Vehicle mix in this view")
    st.caption(
        "All five classes · share of sightings in selected usable captures. This descriptive mix "
        "uses the selected captures, not the balanced-clock comparison above."
    )
    if not good.empty:
        count_cols = [x + "_count" for x in ["car", "motorcycle", "bus", "truck", "bicycle"]]
        mix = good.groupby("is_weekend")[count_cols].sum().reset_index()
        mix = mix.melt("is_weekend", var_name="class", value_name="sightings")
        mix["Day type"] = mix.is_weekend.map({False: "Weekdays", True: "Weekend"})
        mix["Vehicle class"] = mix["class"].str.replace("_count", "", regex=False).str.title()
        if mix.sightings.sum() > 0:
            chart = (
                alt.Chart(mix)
                .mark_bar(size=45)
                .encode(
                    x=alt.X(
                        "sightings:Q",
                        stack="normalize",
                        title="Share of visible vehicle sightings",
                        axis=alt.Axis(format="%"),
                    ),
                    y=alt.Y("Day type:N", title=None, sort=["Weekdays", "Weekend"]),
                    color=alt.Color(
                        "Vehicle class:N",
                        scale=alt.Scale(
                            domain=["Car", "Motorcycle", "Bus", "Truck", "Bicycle"],
                            range=[BLUE, TEAL, "#E5AF56", CORAL, "#8B87A4"],
                        ),
                    ),
                    tooltip=[
                        "Day type:N",
                        "Vehicle class:N",
                        alt.Tooltip("sightings:Q", title="Repeated snapshot sightings"),
                    ],
                )
            )
            show_chart(chart, 140)


CASE_NAMES = {
    "hw_assembly": "Horwang · arrival reference",
    "sam_m1_1_start": "Samsen M1/1 · first-period reference",
    "sam_m1_1_end_1640": "Samsen M1/1 · 16:40 class end",
    "sam_m1_1_end_1500": "Samsen M1/1 · Thursday 15:00 class end",
}


def school_context():
    camera = camera_selector()
    st.subheader("Nearby institutions are context, not proof of a school trip")
    kind = st.selectbox(
        "Institution type", ["Any institution", "School", "University", "College"], key="institution_type"
    )
    links = data["links"][data["links"].camera_id.eq(camera)].copy()
    if kind != "Any institution":
        links = links[links.institution_type.eq(kind.lower())]
    links = links.sort_values("distance_m")
    if links.empty:
        st.info("No institution catalogue links match this camera and type.")
    else:
        options = links.institution_id.tolist()
        ident = st.selectbox(
            "Institution reference",
            options,
            format_func=lambda x: str(links.set_index("institution_id").loc[x, "institution_name"]),
            key="institution_reference",
        )
        calendar = data["calendar"][
            data["calendar"].institution_id.eq(ident) & data["calendar"].date_local.isin(dates)
        ]
        reference = links.set_index("institution_id").loc[ident]
        st.caption(
            f"{reference.distance_m:,.0f} m from the supplied camera location to an institution reference point. "
            "Proximity does not establish a gate view or travel direction."
        )
        with st.expander("Map nearby institution reference points"):
            school_points = data["institutions"][
                data["institutions"].institution_id.isin(links.institution_id)
            ]
            draw_map(context[context.camera_id.eq(camera)], selected=camera, schools=school_points)
            st.caption(
                "Orange: selected camera · coral: institution reference points. These are not verified entrances."
            )
        display = calendar[
            [
                "date_local",
                "scope_name",
                "scope_kind",
                "term_status",
                "closure_status",
                "instruction_day_status",
            ]
        ].rename(
            columns={
                "date_local": "Date",
                "scope_name": "Calendar scope",
                "scope_kind": "Scope type",
                "term_status": "Term context",
                "closure_status": "Closure",
                "instruction_day_status": "Instruction day",
            }
        )
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption(
            "Unknown means no supported label. Programme and cohort calendars remain separate from the institution."
        )
        rule_ids = set()
        for value in calendar.rule_ids_json:
            rule_ids.update(json.loads(value))
        for rule in data["rules"]:
            if rule["rule_id"] in rule_ids:
                with st.expander(rule["label"].replace("_", " ")):
                    st.write(rule["evidence_basis"])
                    st.caption(rule["note"])
                    source = next(x for x in data["sources"] if x["source_id"] == rule["source_id"])
                    st.link_button("Open calendar source", source["source_url"])
        st.download_button(
            "Download institution calendar", csv_bytes(calendar), "institution_calendar.csv", "text/csv"
        )
    st.divider()
    st.subheader("Does the baseline change the timetable story?")
    st.caption(
        "Reference clock windows for documented schedules. Daily attendance and exact exposure times are unknown."
    )
    options = [
        x
        for x in CASE_NAMES
        if data["anchors"].loc[data["anchors"].schedule_id.eq(x), "camera_id"].isin(cameras).any()
    ]
    if not options:
        st.info(
            "No documented timetable case is available among the selected cameras. Broaden the camera filters to see Horwang or Samsen."
        )
        return
    case = st.selectbox("School timetable case", options, format_func=CASE_NAMES.get, key="school_case")
    a, b = st.columns(2)
    width = a.select_slider("Window duration (minutes)", [30, 60, 90], value=60, key="window_width")
    shift = b.select_slider(
        "Hypothetical clock shift (minutes)", [-30, -15, 0, 15, 30], value=0, key="clock_shift"
    )
    result = anchor_summary(data["anchors"], dates, cameras, case, width, shift, metric, day_type)
    scope = data["anchors"][data["anchors"].schedule_id.eq(case)].iloc[0]
    st.caption(
        "Horwang: calendar-supported term context; activities do not imply a whole-school closure."
        if case == "hw_assembly"
        else "Samsen: one M1/1 class only. The exact August effective interval is unverified; this is a timetable reference, not a school-wide dismissal."
    )
    if result.empty:
        st.info(
            "Insufficient paired observations for this case, date selection and window. No effect is estimated."
        )
    else:
        chart = (
            alt.Chart(result)
            .mark_bar(color=BLUE, cornerRadiusEnd=4)
            .encode(
                x=alt.X("difference:Q", title="Window minus baseline · sightings per snapshot"),
                y=alt.Y("baseline:N", title=None, sort=["earlier", "later"]),
                tooltip=[
                    "baseline:N",
                    "days:Q",
                    alt.Tooltip("event_mean:Q", title="Window mean", format=".2f"),
                    alt.Tooltip("baseline_mean:Q", title="Baseline mean", format=".2f"),
                    alt.Tooltip("difference:Q", title="Difference", format="+.2f"),
                ],
            )
        )
        show_chart(chart, 170)
        st.dataframe(
            result.rename(
                columns={
                    "days": "Accepted dates",
                    "event_mean": "Window mean",
                    "baseline_mean": "Baseline mean",
                    "difference": "Difference",
                    "before_18": "All pairs end before 18:00",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        if not result.before_18.all():
            st.caption(
                "One or more later baselines extend after 18:00; changing light may affect vehicle detection."
            )
        if len(result) < 2:
            st.caption(
                "One adjacent baseline is unavailable because it lies outside the acquisition span or fails coverage."
            )
    st.link_button("Open timetable source", scope.source_url)
    st.download_button(
        "Download this window comparison", csv_bytes(result), "school_window_comparison.csv", "text/csv"
    )


def coverage_sources():
    st.subheader("Make missingness visible")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accepted camera-hours", f"{int(hours.coverage_pass.sum()):,}")
    c2.metric("Camera-hours in selection", f"{len(hours):,}")
    c3.metric("Usable captures", f"{int(frames.eligible_temporal.sum()):,}")
    grouped = (
        hours.groupby(["date_local", "hour_local"])
        .agg(accepted=("coverage_pass", "sum"), views=("camera_id", "nunique"))
        .reset_index()
    )
    grouped["accepted_fraction"] = grouped.accepted / grouped.views
    chart = (
        alt.Chart(grouped)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("hour_local:O", title="Bangkok collection hour"),
            y=alt.Y("date_local:O", title=None),
            color=alt.Color(
                "accepted_fraction:Q",
                scale=alt.Scale(domain=[0, 1], range=["#EEF2F4", BLUE]),
                title="Views accepted",
                legend=alt.Legend(format=".0%"),
            ),
            tooltip=[
                "date_local:N",
                "hour_local:O",
                alt.Tooltip("accepted:Q", title="Accepted views"),
                alt.Tooltip("views:Q", title="Selected views"),
            ],
        )
    )
    show_chart(chart, 240)
    st.caption(
        "On 27 August, the full study has no accepted camera-hour at 06:00, 07:00 or 08:00. "
        "This is unavailable data, not a traffic collapse. The sparse 21:00 endpoint is not a full hour."
    )
    with st.expander("Comparison rules & interpretation", expanded=True):
        st.write(
            "Hourly means require at least three usable captures, 50% expected cadence coverage and observations "
            "near both ends of the hour. The primary comparison uses all eight required hours on the same accepted days, "
            "at least three weekdays and both weekend days. Each date has equal weight; cameras are summarized by their median percentage change."
        )
        st.write(
            "Counts describe visible vehicle instances in the whole image, including parked vehicles. "
            "They do not measure flow, unique trips, congestion delay or traffic caused by schools. "
            "The week has no observed public holiday; unknown school calendars are not treated as open."
        )
        st.write(
            "Independent model accuracy is provisional. Rain, road layout and uncertain collection/exposure clock alignment "
            "limit interpretation. No school-free control group is present in the 1.5 km sample."
        )
    with st.expander("Full-study robustness · all vehicles, full week"):
        st.caption("Reference results for the original study, independent of the current sidebar filters.")
        table = data["sensitivity"].copy()
        labels = {
            "primary_balanced_days": "Primary comparison",
            "stricter_hour_coverage": "Stricter hourly coverage",
            "exclude_audited_clock_discrepancy_locations": "Exclude audited clock discrepancies",
            "dry_only_model_rain_le_0.1mm": "Dry-only comparison (unavailable)",
        }
        table = table[table.variant.isin(labels) & table.period.eq("morning_07_09")].copy()
        table["Check"] = table.variant.map(labels)
        st.dataframe(
            table[["Check", "n_cameras", "median_relative_pct", "status"]].rename(
                columns={
                    "n_cameras": "Cameras",
                    "median_relative_pct": "Morning change (%)",
                    "status": "Status",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Saturday morning is wet in the model rain input. A matched dry-only comparison lacks enough dates; its estimate remains missing."
        )
    with st.expander("Calendar sources"):
        for source in data["sources"]:
            st.markdown(f"**{source['source_kind'].replace('_', ' ').title()}**")
            st.caption(source["note"])
            st.link_button("Open source · " + source["source_id"], source["source_url"])
    st.subheader("Take the evidence with you")
    a, b, c = st.columns(3)
    a.download_button(
        "Filtered hourly table", csv_bytes(hours), "filtered_camera_hours.csv", "text/csv", width="stretch"
    )
    b.download_button(
        "Selected camera context", csv_bytes(context), "camera_context.csv", "text/csv", width="stretch"
    )
    settings = dict(
        dataset=data["manifest"]["dataset_id"],
        cameras=cameras,
        dates=dates,
        vehicle_class=metric,
        day_type=day_type,
        comparison_protocol="balanced-eight-clock-hours-v1",
        interpretation="Visible sightings per snapshot; observational association, not school causation",
    )
    c.download_button(
        "Selection & methods",
        json.dumps(settings, indent=2).encode(),
        "selection.json",
        "application/json",
        width="stretch",
    )
    a, b = st.columns(2)
    a.download_button(
        "Full findings report", (HERE / "FINDINGS.md").read_bytes(), "FINDINGS.md", "text/markdown"
    )
    b.download_button(
        "Full methods report", (HERE / "METHODS.md").read_bytes(), "METHODS.md", "text/markdown"
    )
    st.caption(
        "Data bundle checksums verified on load. Original study: 61,319 images · 100 cameras · "
        "59,834 usable observations. AI assisted the analysis and dashboard implementation; it is not human accuracy validation."
    )


if view == "Camera explorer":
    camera_explorer()
elif view == "School context":
    school_context()
elif view == "Coverage & sources":
    coverage_sources()
else:
    overview()
st.divider()
st.caption(
    "BANGKOK SCHOOL ZONE · 2190513 DATA SCIENCE · 25–31 AUGUST 2026 · "
    "OBSERVATIONAL STUDY · MODEL ACCURACY PROVISIONAL"
)
