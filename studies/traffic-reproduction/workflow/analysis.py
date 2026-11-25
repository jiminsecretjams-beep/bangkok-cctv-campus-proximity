"""Execute the school-zone commute analysis from frozen completed image counts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.traffic import (
    CLASSES,
    PERIODS,
    balanced_periods,
    camera_effects,
    holdout_peaks,
    hourly_cube,
    location_clusters,
    matched_hours,
    school_anchor_comparisons,
)

ROOT = Path(__file__).resolve().parents[1]
SCHOOL = ROOT / "inputs/context"
CAMPUS = ROOT / "inputs/campus"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def describe(periods, label, primary_ids=None):
    rows = []
    if periods.empty:
        return [
            dict(
                variant=label,
                period=name,
                n_cameras=0,
                median_relative_pct=None,
                q25_pct=None,
                q75_pct=None,
                positive_cameras=0,
                common_primary_cameras=0,
                common_primary_median_pct=None,
                status="insufficient_matched_days",
            )
            for name in PERIODS
        ]
    for name, group in periods.groupby("period"):
        values = group.vehicle_relative_pct.dropna()
        common = group[group.camera_id.isin(primary_ids)] if primary_ids is not None else group
        rows.append(
            dict(
                variant=label,
                period=name,
                n_cameras=len(values),
                median_relative_pct=float(values.median()),
                q25_pct=float(values.quantile(0.25)),
                q75_pct=float(values.quantile(0.75)),
                positive_cameras=int(values.gt(0).sum()),
                common_primary_cameras=len(common),
                common_primary_median_pct=float(common.vehicle_relative_pct.median()),
                status="estimated" if len(values) else "insufficient_matched_days",
            )
        )
    return rows


def run(output, calendar):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; existing results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    source = calendar / "observations_with_calendar.parquet"
    inputs = [
        source,
        calendar / "rules.json",
        calendar / "date_labels.csv",
        SCHOOL / "rainfall.parquet",
        SCHOOL / "school_schedules.csv",
        SCHOOL / "camera_road_context.csv",
        SCHOOL / "camera_timestamp_audit.csv",
        CAMPUS / "cctv_campus_flags.csv",
        CAMPUS / "camera_institution_matches.csv",
        CAMPUS / "institutions_clean.csv",
    ]
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    d = pd.read_parquet(source)
    assert len(d) == 61319 and d.image_id.is_unique and d.camera_id.nunique() == 100
    for raw in [False, True]:
        prefix = "raw_" if raw else ""
        eligible = d if raw else d[d.eligible_temporal]
        assert (
            eligible[[prefix + c + "_count" for c in CLASSES]]
            .sum(axis=1)
            .eq(eligible[prefix + "vehicle_count"])
            .all()
        )
    weather = pd.read_parquet(SCHOOL / "rainfall.parquet", columns=["image_id", "precipitation_mm"])
    assert set(weather.image_id) == set(d.image_id)
    d = d.merge(weather, on="image_id", validate="one_to_one", how="left")
    flags = pd.read_csv(CAMPUS / "cctv_campus_flags.csv", dtype={"camera_id": str})
    feature_cols = [
        "camera_id",
        "location",
        "location_en",
        "latitude",
        "longitude",
        "near_school",
        "near_university_college",
        "schools_within_radius",
        "universities_colleges_within_radius",
        "nearest_school_name",
        "nearest_school_distance_m",
        "nearest_institution_type",
    ]
    roads = pd.read_csv(SCHOOL / "camera_road_context.csv", dtype={"camera_id": str})
    features = flags[feature_cols].merge(
        roads[["camera_id", "road_class", "road_name"]], on="camera_id", validate="one_to_one"
    )
    assert set(features.camera_id) == set(d.camera_id)
    clusters = location_clusters(features)
    features["location_cluster"] = features.camera_id.map(clusters)
    institutions = pd.read_csv(CAMPUS / "institutions_clean.csv", keep_default_na=False)
    schools = institutions[institutions.institution_type.eq("school")].copy()
    # Explicit non-general-school screen, matching the existing historical OSM
    # work. Preserve original catalogue types; this does not verify school gates.
    schools["excluded_from_general_school_screen"] = schools.institution_name.str.contains(
        r"yoga|kumon|the brain|tutor|language|กวดวิชา|โยคะ|สอนภาษา|ศูนย์ฝึก", case=False, regex=True
    )
    schools.to_csv(output / "school_type_screen.csv", index=False)
    links = pd.read_csv(CAMPUS / "camera_institution_matches.csv", dtype={"camera_id": str})
    links = links[
        links.institution_id.isin(schools.loc[~schools.excluded_from_general_school_screen, "institution_id"])
    ]
    nearest = links.sort_values(["camera_id", "distance_m", "institution_id"]).drop_duplicates("camera_id")
    nearest = nearest[["camera_id", "institution_name", "distance_m"]].rename(
        columns={
            "institution_name": "screened_nearest_school_name",
            "distance_m": "screened_nearest_school_distance_m",
        }
    )
    features = features.merge(nearest, on="camera_id", how="left", validate="one_to_one")
    features["nearest_school_band"] = pd.cut(
        features.screened_nearest_school_distance_m,
        [-1, 250, 500, 1500],
        labels=["0-250m", "250-500m", "500-1500m"],
    ).astype(str)
    features["location_caution"] = np.where(
        features.camera_id.isin(["1188", "529"]),
        "known_metadata_label_discrepancy",
        "supplied_location_provisional",
    )
    features.to_csv(output / "camera_context.csv", index=False)
    h = hourly_cube(d)
    h.to_parquet(output / "camera_day_hour.parquet", index=False)
    h.to_csv(output / "camera_day_hour.csv", index=False)
    coverage = (
        h.groupby(["date_local", "hour_local"])
        .agg(
            cameras_passing=("coverage_pass", "sum"),
            acquired_images=("n_acquired", "sum"),
            usable_images=("n_usable", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(output / "coverage_by_date_hour.csv", index=False)
    mh = matched_hours(h)
    mh.to_csv(output / "same_clock_hour_comparisons.csv", index=False)
    hp = (
        mh.groupby("hour_local")
        .vehicle_ratio.agg(
            n="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
        )
        .reset_index()
    )
    hp.to_csv(output / "matched_hourly_profile.csv", index=False)
    daily, periods = balanced_periods(h)
    effects = camera_effects(periods).merge(features, on="camera_id", validate="one_to_one")
    periods.to_csv(output / "camera_period_comparisons.csv", index=False)
    daily.to_csv(output / "balanced_camera_day_period.csv", index=False)
    effects.to_csv(output / "camera_commute_effects.csv", index=False)
    # Count complete dates from the same eight-hour panel, including ineligible cameras.
    complete_days = daily.groupby(["camera_id", "is_weekend"]).date_local.nunique().unstack(fill_value=0)
    eligibility = (
        complete_days.reindex(features.camera_id, fill_value=0)
        .rename(columns={False: "complete_weekdays", True: "complete_weekend_days"})
        .reset_index()
    )
    eligibility["primary_coverage_eligible"] = eligibility.complete_weekdays.ge(
        3
    ) & eligibility.complete_weekend_days.ge(2)
    eligibility.to_csv(output / "camera_eligibility.csv", index=False)
    ids = set(effects.camera_id)
    sensitivity = describe(periods, "primary_balanced_days", ids)

    def variant(label, cube, weekdays=3, weekends=2):
        _, p = balanced_periods(cube, minimum_weekdays=weekdays, minimum_weekend=weekends)
        sensitivity.extend(describe(p, label, ids))

    variant("stricter_hour_coverage", hourly_cube(d, strict=True))
    variant("raw_inference_including_quality_flagged", hourly_cube(d, raw=True))
    dry_hours = hourly_cube(d, dry=True)
    variant("dry_only_model_rain_le_0.1mm", dry_hours)
    for period, hours in PERIODS.items():
        _, dry_period = balanced_periods(dry_hours, period_hours={period: hours})
        records = describe(dry_period, "dry_only_separate_period_days", ids)
        sensitivity.extend([r for r in records if r["period"] == period])
    rain_coverage = (
        d.groupby(["date_local", "hour_local"])
        .agg(
            usable_frames=("eligible_temporal", "sum"),
            mean_model_rain_mm=("precipitation_mm", "mean"),
            model_dry_fraction=("precipitation_mm", lambda x: x.le(0.1).mean()),
        )
        .reset_index()
    )
    rain_coverage.to_csv(output / "model_rain_coverage.csv", index=False)
    for day in sorted(d.date_local.unique()):
        is_weekend = pd.Timestamp(day).weekday() >= 5
        variant("leave_out_" + day, h[~h.date_local.eq(day)], weekends=1 if is_weekend else 2)
    for shift in [-30, -15, 15, 30]:
        shifted = d.copy()
        shifted["observed_at_utc"] = shifted.observed_at_utc + pd.Timedelta(minutes=shift)
        variant(f"hypothetical_collection_clock_shift_{shift:+d}min", hourly_cube(shifted))
    audit = pd.read_csv(SCHOOL / "camera_timestamp_audit.csv", dtype={"camera_id": str})
    suspect = set(audit.loc[audit.overlay_minus_collection_minutes.abs().gt(5), "camera_id"])
    suspect_locations = set(features.loc[features.camera_id.isin(suspect), "location_cluster"])
    removed = set(features.loc[features.location_cluster.isin(suspect_locations), "camera_id"])
    variant("exclude_audited_clock_discrepancy_locations", h[~h.camera_id.isin(removed)])
    variant("exclude_known_label_discrepancies", h[~h.camera_id.isin(["1188", "529"])])
    without_bicycles = d.copy()
    without_bicycles["vehicle_count"] = without_bicycles.vehicle_count - without_bicycles.bicycle_count
    variant("four_motor_vehicle_classes_only", hourly_cube(without_bicycles))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(output / "sensitivity_checks.csv", index=False)

    # Descriptive location resampling is conditional on this one week. It is not
    # a confidence interval for school causation, future weeks or all Bangkok.
    rng = np.random.default_rng(20261114)
    locations, resampling = [], []
    for period, part in periods.groupby("period"):
        a = part.merge(features[["camera_id", "location_cluster"]], on="camera_id", validate="many_to_one")
        grouped = a.groupby("location_cluster").vehicle_relative_pct.median().dropna()
        boot = np.median(rng.choice(grouped.to_numpy(), size=(2000, len(grouped)), replace=True), axis=1)
        resampling.append(
            dict(
                period=period,
                location_groups=len(grouped),
                median_location_effect_pct=grouped.median(),
                resampling_p025_pct=np.quantile(boot, 0.025),
                resampling_p975_pct=np.quantile(boot, 0.975),
            )
        )
        locations.extend(dict(period=period, location_cluster=k, relative_pct=v) for k, v in grouped.items())
    pd.DataFrame(resampling).to_csv(output / "location_resampling.csv", index=False)
    pd.DataFrame(locations).to_csv(output / "location_effects.csv", index=False)

    types = []
    for period, part in periods.groupby("period"):
        for metric in CLASSES:
            wkshare = part["weekday_" + metric + "_mean"] / part.weekday_vehicle_mean.where(
                part.weekday_vehicle_mean.gt(0)
            )
            weshare = part["weekend_" + metric + "_mean"] / part.weekend_vehicle_mean.where(
                part.weekend_vehicle_mean.gt(0)
            )
            types.append(
                dict(
                    period=period,
                    vehicle_class=metric,
                    n_cameras=len(part),
                    median_weekday_mean=part["weekday_" + metric + "_mean"].median(),
                    median_weekend_mean=part["weekend_" + metric + "_mean"].median(),
                    median_difference=part[metric + "_difference"].median(),
                    n_stable_class_ratios=int(part[metric + "_relative_pct"].notna().sum()),
                    median_relative_pct=(
                        part[metric + "_relative_pct"].median()
                        if part[metric + "_relative_pct"].notna().sum() >= 10
                        else None
                    ),
                    median_share_change_pp=(100 * (wkshare - weshare)).median(),
                )
            )
    types = pd.DataFrame(types)
    types.to_csv(output / "vehicle_class_comparisons.csv", index=False)

    spatial = effects[~effects.camera_id.isin(["1188", "529"])].copy()
    spatial_summary = (
        spatial.groupby("nearest_school_band", observed=True)
        .agg(
            cameras=("camera_id", "nunique"),
            median_morning_excess_pct=("morning_07_09_vehicle_relative_pct", "median"),
            median_specific_excess_pct=("morning_specific_excess_pct", "median"),
            median_afternoon_excess_pct=("afternoon_15_17_vehicle_relative_pct", "median"),
        )
        .reset_index()
    )
    spatial_summary.to_csv(output / "proximity_band_comparisons.csv", index=False)
    strata = (
        spatial.groupby(["road_class", "nearest_school_band"], observed=True)
        .agg(
            cameras=("camera_id", "nunique"),
            morning_excess_pct=("morning_07_09_vehicle_relative_pct", "median"),
        )
        .reset_index()
    )
    strata.to_csv(output / "road_class_proximity_context.csv", index=False)
    rank_corr = spatial.screened_nearest_school_distance_m.rank().corr(
        spatial.morning_specific_excess_pct.rank()
    )
    peaks, peak_daily = holdout_peaks(h)
    peaks = peaks.merge(features, on="camera_id", validate="one_to_one")
    peaks.to_csv(output / "weekday_peak_holdout.csv", index=False)
    peak_daily.to_csv(output / "peak_candidate_daily_evidence.csv", index=False)
    schedules = pd.read_csv(SCHOOL / "school_schedules.csv", dtype={"camera_id": str}, keep_default_na=False)
    anchor = school_anchor_comparisons(d, schedules)
    anchor.to_csv(output / "school_anchor_daily_sensitivity.csv", index=False)
    accepted = anchor[anchor.coverage_pass]
    group = [
        "camera_id",
        "schedule_id",
        "scope",
        "date_applicability",
        "anchor_shift_minutes",
        "width_minutes",
        "baseline",
    ]
    case = (
        accepted.groupby(group)
        .agg(
            days=("date_local", "nunique"),
            event_vehicle_mean=("vehicle_event_mean", "mean"),
            baseline_vehicle_mean=("vehicle_baseline_mean", "mean"),
            vehicle_difference=("vehicle_difference", "mean"),
            motorcycle_difference=("motorcycle_difference", "mean"),
            truck_difference=("truck_difference", "mean"),
            all_pairs_before_18=("pair_ends_before_18", "all"),
            min_daily_vehicle_difference=("vehicle_difference", "min"),
            max_daily_vehicle_difference=("vehicle_difference", "max"),
        )
        .reset_index()
    )
    case["relative_pct"] = 100 * (
        case.event_vehicle_mean / case.baseline_vehicle_mean.where(case.baseline_vehicle_mean.ge(1)) - 1
    )
    case.to_csv(output / "school_anchor_summary.csv", index=False)
    base_case = case[case.width_minutes.eq(60) & case.anchor_shift_minutes.eq(0)]
    base_case.to_csv(output / "school_case_headline_evidence.csv", index=False)

    primary = sensitivity[sensitivity.variant.eq("primary_balanced_days")].copy()
    primary.to_csv(output / "headline_period_evidence.csv", index=False)
    candidate = peaks[peaks.discovery_screen_pass]
    observed_holdouts = candidate[candidate.holdout_available]
    summary = dict(
        images=len(d),
        cameras=d.camera_id.nunique(),
        usable_images=int(d.eligible_temporal.sum()),
        balanced_comparison_cameras=len(effects),
        sampled_location_clusters=len(set(clusters.values())),
        primary_hours="07:00-17:00 collection time; primary periods use eight fixed clock hours",
        accepted_hour_cells=int(h.coverage_pass.sum()),
        total_hour_cells=len(h),
        all_cameras_near_school=bool(features.near_school.all()),
        cameras_also_near_university_college=int(features.near_university_college.sum()),
        excluded_non_general_school_records=int(schools.excluded_from_general_school_screen.sum()),
        cameras_with_screened_school_within_1500m=int(
            features.screened_nearest_school_distance_m.notna().sum()
        ),
        public_holiday_dates=int(d.groupby("date_local").is_public_holiday.first().sum()),
        matched_hours_cameras=mh.camera_id.nunique(),
        primary_periods=primary.to_dict("records"),
        median_morning_specific_excess_pct=float(effects.morning_specific_excess_pct.median()),
        proximity_rank_correlation=float(rank_corr),
        discovery_peak_candidates=len(candidate),
        candidates_with_monday_check=len(observed_holdouts),
        monday_direction_repeated=int(observed_holdouts.holdout_direction_repeated.sum()),
        excluded_clock_cameras=sorted(removed),
        independent_accuracy_established=False,
        image_inference_executed=False,
        random_seed=20261114,
    )
    dump(output / "summary.json", summary)
    plot(output, hp, periods, coverage, effects, types, base_case, daily)
    # Independent CSV reload/means reconciliation from the saved analysis cube.
    saved = pd.read_parquet(output / "camera_day_hour.parquet")
    assert len(saved) == 10500 and not saved.duplicated(["camera_id", "date_local", "hour_local"]).any()
    assert saved.loc[~saved.coverage_pass, "vehicle_mean"].isna().all()
    assert np.allclose(
        saved.loc[saved.coverage_pass, [c + "_mean" for c in CLASSES]].sum(axis=1),
        saved.loc[saved.coverage_pass, "vehicle_mean"],
    )
    for _, row in primary.iterrows():
        recomputed = periods[periods.period.eq(row.period)].vehicle_relative_pct.median()
        assert np.isclose(recomputed, row.median_relative_pct)
    assert all(sha(ROOT / p) == value for p, value in fingerprints.items())
    dump(
        output / "verification.json",
        dict(
            result="pass",
            checks=[
                "Original input and calendar files unchanged by analysis",
                "Class means reconcile with vehicle means",
                "Complete camera/date/hour grid with null excluded counts",
                "Saved headline medians independently reconcile",
                "Minimum three weekdays and both weekend days in primary comparisons",
                "Every primary period uses identical accepted dates and required clock-hour coverage",
                "Peak selection uses Tue-Fri only; Monday never selects the peak",
                "No image inference or changes to reviewer files",
                "School evidence retains scope and unknown applicability",
            ],
        ),
    )
    dump(
        output / "manifest.json",
        dict(
            inputs=fingerprints,
            code={
                str(p.relative_to(ROOT)): sha(p)
                for p in [Path(__file__).resolve(), ROOT / "workflow/traffic.py"]
            },
            files={
                str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=sha(p))
                for p in output.rglob("*")
                if p.is_file() and "matplotlib" not in p.parts
            },
        ),
    )
    print(json.dumps(summary, indent=2))


def plot(out, profile, periods, coverage, effects, types, cases, daily):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    colors = ["#286b8c", "#db9340", "#577d50"]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    x = profile.hour_local.to_numpy()
    ax.fill_between(
        x,
        profile.q25.to_numpy() * 100,
        profile.q75.to_numpy() * 100,
        color=colors[0],
        alpha=0.16,
        label="Middle 50% of camera ratios",
    )
    ax.plot(x, profile["median"] * 100, "o-", color=colors[0], label="Median camera ratio")
    ax.axhline(100, color="#777", linestyle="--")
    ax.set(
        xticks=range(7, 17),
        xlabel="Bangkok collection hour (07 = 07:00-07:59)",
        ylabel="Weekday / weekend sightings, %",
        title="The weekday excess concentrates in commute-like hours",
    )
    ax.legend(frameon=False)
    fig.text(
        0.1,
        0.01,
        "Same camera and clock hour; >=3 weekdays and both weekend days. Sightings per snapshot, not flow.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "01_same_clock_hour.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, (period, hours), color in zip(axes, PERIODS.items(), colors):
        p = periods[periods.period.eq(period)].sort_values("vehicle_relative_pct")
        ax.scatter(range(len(p)), p.vehicle_relative_pct, s=12, color=color, alpha=0.75)
        ax.axhline(0, color="#777", linestyle="--")
        ax.axhline(p.vehicle_relative_pct.median(), color=color, linewidth=2)
        ax.set(
            title=f"{period.split('_')[0].title()} {hours[0]:02d}:00-{hours[-1] + 1:02d}:00\n"
            f"Median {p.vehicle_relative_pct.median():+.1f}% | n={len(p)}",
            xlabel="Cameras sorted within each panel",
        )
    axes[0].set_ylabel("Weekday versus weekend sightings, %")
    fig.suptitle("A morning signature is widespread; midday differences are mixed", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "02_camera_period_distribution.png", dpi=160)
    plt.close(fig)
    matrix = coverage.pivot(index="date_local", columns="hour_local", values="cameras_passing")
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="Blues")
    ax.set(
        xticks=range(15),
        xticklabels=range(6, 21),
        yticks=range(7),
        yticklabels=matrix.index,
        xlabel="Bangkok collection hour",
        title="Coverage must be checked before interpreting low counts",
    )
    fig.colorbar(im, ax=ax, label="Cameras with accepted usable coverage")
    fig.tight_layout()
    fig.savefig(out / "03_coverage.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in zip(["0-250m", "250-500m", "500-1500m"], colors):
        q = effects[effects.nearest_school_band.eq(label) & ~effects.camera_id.isin(["1188", "529"])]
        ax.scatter(
            q.screened_nearest_school_distance_m,
            q.morning_specific_excess_pct,
            s=25,
            alpha=0.65,
            color=color,
            label=label,
        )
    ax.axhline(0, color="#777", linestyle="--")
    ax.set(
        xlabel="Nearest catalogue school reference point, metres (not gate distance)",
        ylabel="Morning/midday weekday contrast versus weekend, %",
        title="Distance alone does not identify a school effect",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "04_proximity_context.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    q = types[types.period.eq("morning_07_09")]
    ax.bar(q.vehicle_class, q.median_share_change_pp, color=colors[0])
    ax.axhline(0, color="#777")
    ax.set(
        ylabel="Weekday minus weekend share\n(median change, percentage points)",
        title="Morning vehicle mix: weekday versus the same weekend hours",
    )
    fig.tight_layout()
    fig.savefig(out / "05_morning_vehicle_mix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    names = {
        "hw_assembly": "Horwang arrival reference",
        "sam_m1_1_start": "Samsen M1/1 first period",
        "sam_m1_1_end_1640": "Samsen M1/1 16:40 end",
        "sam_m1_1_end_1500": "Samsen M1/1 Thursday 15:00 end",
    }
    ids = list(names)
    for offset, baseline, color in [(-0.17, "earlier", colors[0]), (0.17, "later", colors[1])]:
        p = cases[cases.baseline.eq(baseline)].set_index("schedule_id")
        idx = [i for i, k in enumerate(ids) if k in p.index]
        ax.barh(
            np.array(idx) + offset,
            [p.loc[ids[i], "vehicle_difference"] for i in idx],
            height=0.3,
            color=color,
            label=f"Versus {baseline} hour",
        )
        for i in idx:
            r = p.loc[ids[i]]
            ax.text(r.vehicle_difference, i + offset, f" n={int(r.days)}", va="center", fontsize=8)
    ax.set(
        yticks=range(len(ids)),
        yticklabels=[names[i] for i in ids],
        xlabel="Window minus baseline: mean visible vehicles per snapshot",
        title="Documented clock references do not establish school-caused traffic",
    )
    ax.axvline(0, color="#777")
    for i, ident in enumerate(ids):
        if not cases.schedule_id.eq(ident).any():
            ax.text(0.1, i, "Insufficient paired coverage", va="center", fontsize=9, color="#777")
    ax.set_ylim(-0.6, len(ids) - 0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "06_school_baseline_sensitivity.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, cam, title in zip(
        axes.flat,
        ["139", "150", "239", "1060"],
        ["Horwang context", "Samsen context", "Triam Udom context", "Saipanya context"],
    ):
        q = daily[daily.camera_id.eq(cam)]
        for weekend, color, label in [(False, colors[0], "Weekdays"), (True, colors[1], "Weekend")]:
            p = q[q.is_weekend.eq(weekend)].groupby("period").vehicle_mean.mean().reindex(PERIODS)
            ax.plot(range(3), p.to_numpy(), "o-", color=color, label=label)
        ax.set(
            title=f"Camera {cam}: {title}",
            xticks=range(3),
            xticklabels=["07-09", "10-14", "15-17"],
            ylabel="Sightings per snapshot",
        )
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("School-adjacent case views: compare times within each view", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "07_school_case_profiles.png", dpi=160)
    plt.close(fig)
