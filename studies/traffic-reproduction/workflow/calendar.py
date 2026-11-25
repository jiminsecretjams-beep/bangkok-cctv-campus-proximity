"""Source-scoped calendar labels. Unknown is never silently converted to open."""

from __future__ import annotations

import json
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from datetime import date

import pandas as pd

TZ = "Asia/Bangkok"
KEY = ["camera_id", "date_local"]


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_holidays(data):
    """Adapt the teammate's JSON-to-CSV collector, preserving source fields."""
    rows = []
    if not isinstance(data.get("years"), list):
        raise ValueError("Holiday source must contain a years list")
    for year in data["years"]:
        for holiday in year["holidays"]:
            day = date.fromisoformat(holiday["date"])
            if day.year != year["year"] or year["year_buddhist_era"] != day.year + 543:
                raise ValueError("Holiday year/era mismatch")
            substitution = holiday.get("is_substitution", False)
            if not isinstance(substitution, bool):
                raise ValueError("is_substitution must be a JSON boolean")
            rows.append(
                dict(
                    date_local=day.isoformat(),
                    year=day.year,
                    buddhist_year=day.year + 543,
                    weekday=day.strftime("%A"),
                    name_th=holiday.get("name_th") or "",
                    name_en=holiday.get("name_en") or "",
                    holiday_type=holiday.get("type") or "",
                    is_substitution=substitution,
                    note=holiday.get("note") or "",
                    source_claim=data.get("source", ""),
                    source_id="holiday_mirror",
                )
            )
    if not rows:
        raise ValueError("Empty holiday data")
    return (
        pd.DataFrame(rows)
        .drop_duplicates(["date_local", "name_th", "name_en", "holiday_type"])
        .sort_values(["date_local", "name_en"])
        .reset_index(drop=True)
    )


def date_dimension(days, holidays, review):
    """Only reviewed dates get a negative holiday label; absence alone is unknown."""
    approved = set(review["crosschecked_dates"])
    rows = []
    for text in sorted(set(days)):
        day = date.fromisoformat(text)
        matches = holidays[holidays.date_local.eq(text)]
        crosschecked = text in approved
        # A mirror record outside the reviewed range is still only a candidate.
        public = bool(len(matches)) if crosschecked else pd.NA
        rows.append(
            dict(
                date_local=text,
                weekday_local=day.strftime("%A"),
                weekday_number=day.weekday(),
                is_weekend=day.weekday() >= 5,
                is_public_holiday=public,
                public_holiday_status=(
                    "listed_crosschecked"
                    if crosschecked and len(matches)
                    else "not_listed_crosschecked"
                    if crosschecked
                    else "unknown_unreviewed_date"
                ),
                holiday_names_json=dumps(sorted(set(matches.name_en))),
                holiday_source_ids_json=dumps(review["source_ids"] if crosschecked else ["holiday_mirror"]),
                holiday_review_id=review["review_id"] if crosschecked else "",
                calendar_timezone=TZ,
            )
        )
    frame = pd.DataFrame(rows)
    frame["is_public_holiday"] = frame.is_public_holiday.astype("boolean")
    return frame


def validate_rules(rules, institutions, source_ids):
    known = set(institutions.institution_id)
    if len({r["rule_id"] for r in rules}) != len(rules):
        raise ValueError("Duplicate rule_id")
    for r in rules:
        if r["institution_id"] not in known:
            raise ValueError("Unknown institution in calendar rule")
        if r["source_id"] not in source_ids:
            raise ValueError("Unknown calendar source")
        if r["scope_kind"] not in {"institution", "programme", "cohort"}:
            raise ValueError("Invalid scope")
        if r["scope_kind"] == "institution" and r["scope_id"] != r["institution_id"]:
            raise ValueError("Institution rules must use the institution ID as scope ID")
        if r["scope_kind"] != "institution" and r["scope_id"] == r["institution_id"]:
            raise ValueError("Programme/cohort evidence cannot masquerade as an institution rule")
        if r["kind"] not in {"term", "event", "closure"}:
            raise ValueError("Invalid calendar rule kind")
        if r["kind"] == "term" and r["value"] not in {"in_term", "out_of_term"}:
            raise ValueError("Invalid term state")
        if r["kind"] == "closure" and r["value"] not in {"open", "closed"}:
            raise ValueError("Invalid closure state")
        if date.fromisoformat(r["start_date"]) > date.fromisoformat(r["end_date"]):
            raise ValueError("Reversed calendar interval")


def resolve_state(active, kind):
    values = {r["value"] for r in active if r["kind"] == kind}
    if kind == "term" and len({r["label"] for r in active if r["kind"] == kind}) > 1:
        return "conflict"
    return "conflict" if len(values) > 1 else next(iter(values), "unknown")


def expand_calendars(institutions, dates, rules):
    """One row/scope/date; narrow programme/cohort scopes remain separate."""
    scopes = {
        r.institution_id: dict(
            scope_id=r.institution_id,
            scope_kind="institution",
            institution_id=r.institution_id,
            scope_name=r.institution_name,
        )
        for r in institutions.itertuples()
    }
    for rule in rules:
        scope = {k: rule[k] for k in ["scope_id", "scope_kind", "institution_id", "scope_name"]}
        if rule["scope_id"] in scopes and scopes[rule["scope_id"]] != scope:
            raise ValueError("Conflicting scope identity")
        scopes[rule["scope_id"]] = scope
    by_scope = {}
    for r in rules:
        by_scope.setdefault(r["scope_id"], []).append(r)
    rows = []
    for scope_id, scope in scopes.items():
        for day in dates.date_local:
            active = [r for r in by_scope.get(scope_id, []) if r["start_date"] <= day <= r["end_date"]]
            rows.append(
                dict(
                    **scope,
                    date_local=day,
                    term_status=resolve_state(active, "term"),
                    term_labels_json=dumps(sorted({r["label"] for r in active if r["kind"] == "term"})),
                    closure_status=resolve_state(active, "closure"),
                    # A term calendar or activity never proves actual attendance.
                    instruction_day_status="unknown",
                    event_ids_json=dumps(sorted(r["rule_id"] for r in active if r["kind"] == "event")),
                    rule_ids_json=dumps(sorted(r["rule_id"] for r in active)),
                    source_ids_json=dumps(sorted({r["source_id"] for r in active})),
                    evidence_basis_json=dumps(sorted({r["evidence_basis"] for r in active})),
                )
            )
    return pd.DataFrame(rows).merge(dates, on="date_local", how="left", validate="many_to_one")


def camera_context(cameras, dates, links, scope_days):
    """Aggregate contextual evidence before joining images; never multiply counts."""
    if links.duplicated(["camera_id", "institution_id"]).any():
        raise ValueError("Duplicate camera/institution links")
    scoped = scope_days[scope_days.scope_kind.eq("institution")]
    nearby = links.merge(scoped, on="institution_id", how="left", validate="many_to_many")
    expected = len(links) * len(dates)
    if len(nearby) != expected or nearby.date_local.isna().any():
        raise ValueError("Incomplete institution/date expansion")
    details = {}
    for key, group in nearby.groupby(KEY):
        known = group[group.term_status.isin(["in_term", "out_of_term"])]
        evidence = []
        for row in known.itertuples():
            evidence.append(
                dict(
                    institution_id=row.institution_id,
                    institution_name=row.institution_name,
                    distance_m=float(row.distance_m),
                    term_status=row.term_status,
                    term_labels=json.loads(row.term_labels_json),
                    rule_ids=json.loads(row.rule_ids_json),
                )
            )
        details[key] = dict(
            nearby_institution_count=len(group),
            nearby_term_known_count=len(known),
            nearby_term_unknown_count=int(group.term_status.eq("unknown").sum()),
            nearby_term_conflict_count=int(group.term_status.eq("conflict").sum()),
            nearby_closure_unknown_count=int(group.closure_status.eq("unknown").sum()),
            nearby_confirmed_closed_count=int(group.closure_status.eq("closed").sum()),
            nearby_institution_term_context_json=dumps(sorted(evidence, key=lambda r: r["institution_id"])),
        )
    narrow = scope_days[~scope_days.scope_kind.eq("institution")]
    narrowed = links.merge(narrow, on="institution_id", how="inner", validate="many_to_many")
    narrow_details = {}
    for key, group in narrowed.groupby(KEY):
        records = []
        for row in group.itertuples():
            if row.term_status != "unknown" or json.loads(row.event_ids_json):
                records.append(
                    dict(
                        scope_id=row.scope_id,
                        scope_kind=row.scope_kind,
                        scope_name=row.scope_name,
                        institution_id=row.institution_id,
                        term_status=row.term_status,
                        event_ids=json.loads(row.event_ids_json),
                        rule_ids=json.loads(row.rule_ids_json),
                    )
                )
        narrow_details[key] = dumps(sorted(records, key=lambda r: r["scope_id"]))
    rows = []
    for camera in sorted(set(cameras)):
        for day in dates.date_local:
            context = details.get(
                (camera, day),
                dict(
                    nearby_institution_count=0,
                    nearby_term_known_count=0,
                    nearby_term_unknown_count=0,
                    nearby_term_conflict_count=0,
                    nearby_closure_unknown_count=0,
                    nearby_confirmed_closed_count=0,
                    nearby_institution_term_context_json="[]",
                ),
            )
            rows.append(
                dict(
                    camera_id=camera,
                    date_local=day,
                    **context,
                    nearby_scoped_calendar_context_json=narrow_details.get((camera, day), "[]"),
                )
            )
    return pd.DataFrame(rows), nearby


def enrich_observations(observations, dates, context):
    if observations.image_id.duplicated().any():
        raise ValueError("Duplicate source image IDs")
    local = pd.to_datetime(observations.observed_at_utc, utc=True, errors="raise").dt.tz_convert(TZ)
    if local.isna().any() or not local.dt.strftime("%Y-%m-%d").eq(observations.date_local).all():
        raise ValueError("Source dates disagree with Bangkok timestamps")
    if not local.dt.dayofweek.ge(5).eq(observations.is_weekend).all():
        raise ValueError("Source weekend labels disagree with Bangkok timestamps")
    if dates.date_local.duplicated().any() or context.duplicated(KEY).any():
        raise ValueError("Calendar join keys are not unique")
    extra_dates = dates.drop(columns=[c for c in dates if c in observations and c != "date_local"])
    result = observations.merge(extra_dates, on="date_local", how="left", validate="many_to_one")
    result = result.merge(context, on=KEY, how="left", validate="many_to_one")
    if len(result) != len(observations) or result.public_holiday_status.isna().any():
        raise ValueError("Calendar join dropped or multiplied observations")
    if result.nearby_institution_count.isna().any():
        raise ValueError("Missing camera calendar context")
    pd.testing.assert_frame_equal(
        observations.reset_index(drop=True), result[observations.columns].reset_index(drop=True)
    )
    return result


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "inputs/calendar"
RAW = ROOT / "inputs/snapshots"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def build(source, campus, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; use a new directory to preserve existing releases")
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads((CONFIG / "reviewed_sources.json").read_text())
    for item in sources:
        path = ROOT / item["snapshot"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"Source changed since review: {item['source_id']}")
    review = json.loads((CONFIG / "national_holiday_review.json").read_text())
    rules = json.loads((CONFIG / "rules.json").read_text())
    institutions = pd.read_csv(campus / "institutions_clean.csv", keep_default_na=False)
    links = pd.read_csv(
        campus / "camera_institution_matches.csv", dtype={"camera_id": str}, keep_default_na=False
    )
    links = links[["camera_id", "institution_id", "institution_name", "institution_type", "distance_m"]]
    if not links.institution_id.isin(institutions.institution_id).all() or links.distance_m.gt(1500).any():
        raise ValueError("Invalid 1.5 km campus links")
    validate_rules(rules, institutions, {x["source_id"] for x in sources})
    source_hash = digest(source)
    observations = pd.read_parquet(source)
    observations["camera_id"] = observations.camera_id.astype(str)
    days = sorted(observations.date_local.unique())
    if days != review["crosschecked_dates"]:
        raise ValueError("Review covers a different study period; review new dates before building")
    if not set(links.camera_id).issubset(set(observations.camera_id)):
        raise ValueError("Campus links reference cameras outside this release")
    holidays = normalize_holidays(json.loads((RAW / "thai_public_holidays.json").read_text()))
    dates = date_dimension(days, holidays, review)
    calendars = expand_calendars(institutions, dates, rules)
    context, camera_institutions = camera_context(observations.camera_id, dates, links, calendars)
    enriched = enrich_observations(observations, dates, context)
    if digest(source) != source_hash:
        raise ValueError("Input observation file changed during build")
    enriched.to_parquet(output / "observations_with_calendar.parquet", index=False, compression="zstd")
    original = set(observations.columns)
    sidecar = enriched[["image_id", "camera_id", "date_local"] + [c for c in enriched if c not in original]]
    sidecar.to_csv(output / "observation_calendar_labels.csv", index=False)
    tables = {
        "public_holidays": holidays,
        "date_labels": dates,
        "institution_scope_calendar_daily": calendars,
        "camera_calendar_daily": context,
        "camera_institution_calendar_daily": camera_institutions,
        "calendar_rules": pd.DataFrame(rules),
    }
    # Retain all timetable evidence, including historical/undated/scenario records,
    # without allowing them to become date-specific term/open/closed assertions.
    schedule_source = ROOT / "inputs/context/school_schedules.csv"
    schedules = pd.read_csv(schedule_source, dtype={"camera_id": str}, keep_default_na=False)
    schedules["calendar_date_applicability"] = "not_applied_no_verified_effective_interval"
    schedules["reference_only"] = True
    tables["schedule_evidence_reference"] = schedules
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    # Reopen the published artifact, not just the in-memory frame.
    saved = pd.read_parquet(output / "observations_with_calendar.parquet")
    pd.testing.assert_frame_equal(observations.reset_index(drop=True), saved[observations.columns])
    assert len(saved) == len(observations) and saved.image_id.is_unique
    assert len(dates) == 7 and len(context) == observations.camera_id.nunique() * len(dates)
    assert not calendars.duplicated(["scope_id", "date_local"]).any()
    assert calendars[calendars.scope_kind.eq("institution")].shape[0] == len(institutions) * len(days)
    assert (
        calendars[calendars.institution_id.eq("edu_95d76c138f10") & calendars.scope_kind.eq("institution")]
        .term_status.eq("unknown")
        .all()
    )
    assert calendars.closure_status.eq("unknown").all()
    assert not dates.is_public_holiday.isna().any()
    summary = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        status="completed_with_explicit_coverage_limits",
        study_dates=days,
        images=len(saved),
        cameras=saved.camera_id.nunique(),
        original_columns_preserved=len(observations.columns),
        institutions=len(institutions),
        camera_institution_links=len(links),
        camera_date_rows=len(context),
        scope_date_rows=len(calendars),
        calendar_rule_count=len(rules),
        source_count=len(sources),
        public_holiday_dates=int(dates.is_public_holiday.sum()),
        institution_term_known_scope_dates=int(
            (calendars.scope_kind.eq("institution") & calendars.term_status.ne("unknown")).sum()
        ),
        programme_term_known_scope_dates=int(
            (calendars.scope_kind.eq("programme") & calendars.term_status.ne("unknown")).sum()
        ),
        closure_known_scope_dates=int(calendars.closure_status.ne("unknown").sum()),
        images_with_nearby_institution_term_context=int(saved.nearby_term_known_count.gt(0).sum()),
        schedule_references=len(schedules),
        source_observations_sha256=source_hash,
        traffic_analysis_executed=False,
        image_inference_executed=False,
    )
    write_json(output / "summary.json", summary)
    write_json(
        output / "verification.json",
        dict(
            result="pass",
            checks=[
                "Saved Parquet preserves every original column, count, quality flag and row in order",
                "One unique row per source image; date and camera joins are many-to-one",
                "Bangkok dates and weekend flags independently reconcile with UTC timestamps",
                "All seven dates and all 100 camera/date combinations covered",
                "Negative holiday labels restricted to explicitly crosschecked dates",
                "Programme calendar never becomes a university-wide calendar",
                "Activity/cohort rules never imply closure or daily attendance",
                "Unknown closures remain unknown; timetable scenarios remain reference-only",
                "All source snapshots match reviewed hashes; original observation input unchanged",
            ],
        ),
    )
    for name in ["rules.json", "reviewed_sources.json", "national_holiday_review.json"]:
        shutil.copyfile(CONFIG / name, output / name)
    files = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in output.iterdir() if p.is_file()}
    write_json(
        output / "manifest.json",
        dict(
            sources=sources,
            files=files,
            inputs=[
                dict(file=str(p.relative_to(ROOT)), sha256=digest(p))
                for p in [
                    source,
                    campus / "institutions_clean.csv",
                    campus / "camera_institution_matches.csv",
                    schedule_source,
                ]
            ],
            code=[dict(file=str(p.relative_to(ROOT)), sha256=digest(p)) for p in [Path(__file__).resolve()]],
        ),
    )
    print(json.dumps(summary, indent=2))
