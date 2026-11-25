"""Draw the six charts and the five-page report. All numbers come from run.py."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_charts(metrics, assets):
    day = pd.DataFrame(metrics["day"]).set_index("date")
    camera_top = pd.DataFrame(metrics["cameras"]).set_index("camera_id")
    district_top = pd.DataFrame(metrics["districts"]).set_index("district")
    durations = pd.DataFrame(metrics["durations"]).set_index("failure_type")
    pair = metrics["pair"]
    rows = (pair["freeze_union_bins"] + 9) // 10
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.titleweight": "bold",
            "text.color": "#1c2933",
            "axes.labelcolor": "#536675",
        }
    )
    ink, red, gold, pale = "#253746", "#cc5145", "#d79b35", "#dce4ea"

    def finish(name):
        plt.savefig(assets / name, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()

    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    ax.bar(
        range(len(day)),
        day.percent,
        color=[red if value == day.percent.max() else pale for value in day.percent],
        width=0.62,
    )
    for i, value in enumerate(day.percent):
        ax.text(
            i,
            value + 0.55,
            f"{value:.1f}%",
            ha="center",
            weight="bold",
            fontsize=17,
            color=red if value == day.percent.max() else ink,
        )
    ax.set(
        xticks=range(len(day)),
        xticklabels=[pd.Timestamp(date).strftime("%a\n%d %b") for date in day.index],
        ylim=(0, max(23, day.percent.max() * 1.2)),
        ylabel="Failed camera checks (%)",
    )
    ax.tick_params(length=0)
    ax.grid(axis="y", alpha=0.12)
    ax.set_axisbelow(True)
    fig.tight_layout()
    finish("q1_when.png")

    for table, name, is_district in [
        (camera_top, "q2_cameras.png", False),
        (district_top, "q2_districts.png", True),
    ]:
        fig, ax = plt.subplots(figsize=(9.5, 3.3))
        labels = (
            [
                f"{district} ({int(row.cameras)} {'camera' if row.cameras == 1 else 'cameras'})"
                for district, row in table.iterrows()
            ]
            if is_district
            else [f"Camera {cam}" for cam in table.index]
        )
        values = table.percent.to_numpy()
        ax.barh(labels[::-1], values[::-1], color=ink if is_district else gold, height=0.6)
        for i, value in enumerate(values[::-1]):
            ax.text(
                value + 0.45,
                i,
                f"{value:.1f}%",
                va="center",
                weight="bold",
                fontsize=17,
            )
        ax.set(xlim=(0, 25), xlabel="Failed camera checks (%)", xticks=[0, 5, 10, 15, 20])
        ax.tick_params(length=0)
        ax.grid(axis="x", alpha=0.12)
        ax.set_axisbelow(True)
        fig.tight_layout()
        finish(name)

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    for i in range(pair["freeze_union_bins"]):
        x, y = i % 10, rows - 1 - i // 10
        ax.add_patch(plt.Rectangle((x, y), 0.8, 0.8, color=gold if i < pair["cofreeze_bins"] else pale))
    ax.set(xlim=(-0.2, 10), ylim=(-0.2, rows))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        10.8,
        rows * 0.78,
        str(pair["cofreeze_bins"]),
        fontsize=57,
        weight="bold",
        color=gold,
    )
    ax.text(10.8, rows * 0.66, "both cameras frozen", fontsize=17)
    ax.text(
        10.8,
        rows * 0.39,
        str(pair["freeze_union_bins"] - pair["cofreeze_bins"]),
        fontsize=42,
        weight="bold",
        color=ink,
    )
    ax.text(10.8, rows * 0.27, "only one frozen", fontsize=17)
    ax.set_xlim(-0.2, 21)
    fig.tight_layout()
    finish("q3_together.png")

    ratio = metrics["traffic"]["median_camera_ratio"] * 100
    lo, hi = np.array(metrics["traffic"]["camera_bootstrap_95pct"]) * 100
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.bar([0, 1], [100, ratio], color=[pale, gold], width=0.46)
    ax.errorbar(
        1,
        ratio,
        yerr=[[ratio - lo], [hi - ratio]],
        fmt="none",
        ecolor=ink,
        capsize=12,
        lw=2.2,
    )
    ax.axhline(100, color="#8d9da8", ls="--", lw=1.2)
    ax.text(0, 103, "100", ha="center", weight="bold", fontsize=27)
    ax.text(
        1.28,
        ratio,
        f"{ratio:.0f}",
        ha="left",
        va="center",
        weight="bold",
        fontsize=27,
        color=gold,
    )
    ax.text(1.16, hi + 4, f"{lo:.0f}-{hi:.0f}: uncertainty range", fontsize=14, ha="center")
    ax.set(
        xticks=[0, 1],
        xticklabels=[
            "Usual traffic\nat the same camera/time",
            "Traffic before\nfreezing was detected",
        ],
        ylim=(0, 157),
        xlim=(-0.55, 1.8),
        yticks=[0, 50, 100, 150],
        ylabel="Traffic index: usual = 100",
    )
    ax.tick_params(length=0)
    fig.tight_layout()
    finish("q4_traffic.png")

    names = {
        "missing": "No image",
        "offline placeholder": "White offline image",
        "duplicate frame": "Frozen image",
        "decode artifact": "Grey image",
        "test pattern": "Colour bars",
        "partial corruption": "Partly damaged image",
        "obstructed view": "Blocked lens",
    }
    order = list(names)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    values = [float(durations.loc[k, "median_min"]) for k in order]
    ax.barh(
        range(7),
        values,
        color=[gold if k == "duplicate frame" else ink for k in order],
        height=0.58,
    )
    for i, key in enumerate(order):
        value = values[i]
        n = int(durations.loc[key, "episodes"])
        ax.text(
            value + 1.5,
            i,
            f"{value:.1f} min  ({n} {'episode' if n == 1 else 'episodes'})",
            va="center",
            fontsize=13,
        )
    ax.set(
        yticks=range(7),
        yticklabels=[names[k] for k in order],
        xlim=(0, 132),
        xlabel="Typical episode duration in minutes (median)",
        xticks=[0, 20, 40, 60, 80, 100, 120],
    )
    ax.invert_yaxis()
    ax.tick_params(length=0)
    fig.tight_layout()
    finish("q5_duration.png")


def write_report(metrics, output):
    from PIL import Image
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph

    assets = output / "charts"
    assets.mkdir(parents=True, exist_ok=True)
    make_charts(metrics, assets)
    pdf_path = output / "CCTV_Failures_Five_Questions.pdf"
    fontdir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    worst = max(metrics["day"], key=lambda row: row["percent"])
    worst_date = pd.Timestamp(worst["date"])
    types = metrics["types"]
    pair = metrics["pair"]
    traffic = metrics["traffic"]
    index = round(traffic["median_camera_ratio"] * 100)
    low, high = [round(value * 100) for value in traffic["camera_bootstrap_95pct"]]
    longest = metrics["longest_freeze"]
    hours, minutes = divmod(round(longest["minutes"]), 60)
    duration = {row["failure_type"]: row for row in metrics["durations"]}
    district = metrics["districts"][0]
    run_date = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%d %b %Y").upper()
    pdfmetrics.registerFont(TTFont("Easy", str(fontdir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Easy-Bold", str(fontdir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily(
        "Easy", normal="Easy", bold="Easy-Bold", italic="Easy", boldItalic="Easy-Bold"
    )
    W, H = 1040, 1320
    ink, muted, red, gold, pale, linecolor = [
        colors.HexColor(v) for v in ["#1d2a34", "#526675", "#ca5145", "#bd8422", "#f0f4f6", "#dce4e9"]
    ]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=(W, H), pageCompression=1)
    c.setTitle("Why Bangkok CCTV fails - five questions answered")
    c.setAuthor("Bangkok CCTV project team")

    def box(x, top, w, h, fill=pale):
        c.setFillColor(fill)
        c.roundRect(x, H - top - h, w, h, 12, fill=1, stroke=0)

    def text(x, top, value, size=21, bold=False, color=ink):
        c.setFont("Easy-Bold" if bold else "Easy", size)
        c.setFillColor(color)
        c.drawString(x, H - top - size, str(value))

    def para(x, top, w, value, size=21, color=ink, limit=None):
        p = Paragraph(
            value,
            ParagraphStyle(
                "body",
                fontName="Easy",
                fontSize=size,
                leading=size * 1.35,
                textColor=color,
            ),
        )
        _, h = p.wrap(w, 2000)
        if limit is not None and h > limit:
            raise ValueError(f"Text too tall ({h}>{limit}): {value}")
        p.drawOn(c, x, H - top - h)
        return h

    def image(file, top, height):
        p = assets / file
        with Image.open(p) as im:
            iw, ih = im.size
        scale = min(936 / iw, height / ih)
        w, h = iw * scale, ih * scale
        c.drawImage(str(p), (W - w) / 2, H - top - h, w, h, mask="auto")

    def head(number, question, answer):
        box(44, 29, 952, 37, ink)
        text(
            60,
            38,
            f"BANGKOK CCTV  /  25-31 AUG 2026  /  {metrics['cameras_total']} CAMERAS",
            13,
            color=colors.white,
        )
        text(44, 92, f"QUESTION {number} OF 5", 15, True, color=red)
        para(44, 124, 952, question, 33, limit=95)
        box(44, 222, 952, 109)
        para(65, 241, 910, f"<b>{answer}</b>", 26, limit=80)
        text(
            44,
            1284,
            f"{run_date}  /  {number} OF 5  /  SOURCE: TEAM FAILURE KIT + YOLO26x COUNTS",
            12,
            color=muted,
        )

    def small_note(top, value):
        para(44, top, 952, value, 17, muted, limit=90)

    head(
        1,
        "When do failures happen? Does it differ by error type?",
        f"The highest failure rate was on {worst_date:%A, %d %B}.",
    )
    text(44, 357, f"{worst['percent']:.1f}%", 51, True, color=red)
    para(
        270,
        370,
        710,
        f"of camera checks failed that day<br/><b>{worst['failed']:,} failed out of {worst['checks']:,} checks</b>",
        22,
        limit=70,
    )
    image("q1_when.png", 455, 315)
    text(44, 797, "Different problems have different timing", 23, True)
    rows = [
        (
            "White offline pictures",
            f"All {types['offline placeholder']['images']} were between {types['offline placeholder']['min_hour']:02}:00 and {types['offline placeholder']['max_hour']:02}:59.",
        ),
        (
            "Colour bars",
            f"{types['test pattern']['images']} pictures, on {types['test pattern']['days']} day, between {types['test pattern']['min_hour']:02}:00 and {types['test pattern']['max_hour']:02}:59.",
        ),
        (
            "Frozen pictures",
            f"{types['duplicate frame']['images']} across {types['duplicate frame']['days']} days and many hours.",
        ),
        (
            "Other picture faults",
            f"{types['decode artifact']['images']} grey, {types['partial corruption']['images']} partly damaged, {types['obstructed view']['images']} blocked lens.",
        ),
    ]
    y = 840
    for label, value in rows:
        text(44, y, label, 20, True)
        para(355, y, 635, value, 20, limit=53)
        c.setStrokeColor(linecolor)
        c.line(44, H - y - 48, 996, H - y - 48)
        y += 60
    box(44, 1090, 952, 118)
    para(
        64,
        1106,
        912,
        f"<b>Lighting:</b> {metrics['lighting']['Day']:.1f}% of daytime checks failed, compared with {metrics['lighting']['Dark']:.1f}% in darkness. "
        "The big outage happened in the morning, so this does <b>not</b> show that sunlight caused the failures.",
        20,
        limit=90,
    )
    small_note(
        1223,
        "A camera check means one expected snapshot, about every 10 minutes. Missing and unusable pictures count as failed checks. No outage periods are removed from this page.",
    )
    c.showPage()

    head(
        2,
        "Which cameras or districts fail most?",
        "Five cameras stand out. A district score can be driven by just one or two cameras.",
    )
    text(44, 357, "Cameras with the highest failure rates", 23, True)
    image("q2_cameras.png", 401, 285)
    text(44, 718, "Districts with the highest failure rates", 23, True)
    image("q2_districts.png", 764, 285)
    box(44, 1080, 952, 124)
    para(
        65,
        1095,
        910,
        f"<b>Important:</b> {district['district']} has only <b>{district['cameras']} sampled cameras</b>. "
        "The district score is a warning about the sampled cameras, not proof that the whole district has worse infrastructure.",
        20,
        limit=96,
    )
    small_note(
        1220,
        "All checks from the full week are included. Failure rate = failed checks divided by all checks for that camera or district. "
        f"The five cameras hold {metrics['freeze_top5_images']} of {metrics['freeze_images']} frozen pictures ({metrics['freeze_top5_percent']:.0f}%).",
    )
    c.showPage()

    head(
        3,
        "Do cameras fail together? Are they nearby or on the same subnet?",
        f"Cameras {pair['camera_a']} and {pair['camera_b']} share the strongest freeze overlap in this sample.",
    )
    text(
        44,
        359,
        f"{pair['cofreeze_bins']} out of {pair['freeze_union_bins']}",
        45,
        True,
        color=gold,
    )
    para(
        44,
        424,
        952,
        "ten-minute periods when at least one of the pair froze had <b>both cameras frozen</b>. "
        f"This is not {100 * pair['jaccard']:.0f}% of the whole week.",
        22,
        limit=70,
    )
    image("q3_together.png", 535, 350)
    small_note(
        894,
        "Each square represents one such ten-minute period. We only compare periods when both cameras delivered an image.",
    )
    box(44, 969, 456, 108)
    box(520, 969, 476, 108)
    text(63, 983, f"{pair['distance_km']:.2f} km apart", 26, True)
    para(63, 1027, 420, "Distance from the supplied GPS locations.", 18, limit=50)
    text(
        540,
        983,
        "Same address group" if pair["same_subnet"] else "Different address groups",
        26,
        True,
    )
    para(540, 1027, 438, "Based on supplied address group labels.", 18, limit=50)
    para(
        44,
        1107,
        952,
        "<b>What that means:</b> these feeds share a recorded address group. "
        "Their timing makes a shared delivery problem plausible. The faulty equipment remains unknown.",
        21,
        limit=90,
    )
    small_note(
        1220,
        "The wider outage also affected all 100 cameras: none had an OK picture before 08:50 on 27 August. Such shared timing points toward a shared delivery or collection problem.",
    )
    c.showPage()

    head(
        4,
        "Does traffic matter?",
        "We found no clear evidence that unusually heavy traffic comes before freezing.",
    )
    para(
        44,
        360,
        952,
        f"Using <b>usual traffic = 100</b>, traffic before freezing was detected was <b>{index}</b>. "
        "The difference is small compared with the uncertainty.",
        24,
        limit=85,
    )
    image("q4_traffic.png", 472, 365)
    box(44, 868, 952, 110)
    para(
        64,
        883,
        912,
        f"<b>How certain is that?</b> The uncertainty range is <b>{low} to {high}</b>. "
        "It includes both less and more traffic than usual, so we cannot call the rise a reliable pattern.",
        21,
        limit=82,
    )
    text(44, 1010, "How we checked", 23, True)
    methods = [
        (
            f"{traffic['freeze_episodes']} freeze episodes",
            f"across {traffic['freeze_cameras']} cameras, with enough usable pictures.",
        ),
        ("Before the freeze flag", "vehicle counts in the preceding 30 minutes."),
        ("A fairer comparison", "the same camera and time on other weekdays."),
    ]
    y = 1055
    for label, value in methods:
        text(44, y, label, 20, True)
        para(350, y, 646, value, 20, limit=55)
        y += 52
    small_note(
        1220,
        "Missing or unusable pictures are never counted as zero traffic. This is an observational comparison; it does not prove that traffic can never contribute to a fault.",
    )
    c.showPage()

    head(
        5,
        "How long do failures last for each error type?",
        "Many faults last around 10 minutes. Some cameras show the same stuck picture for hours.",
    )
    box(44, 359, 952, 99)
    text(64, 369, f"{hours} hours {minutes} minutes", 37, True, color=red)
    text(
        64,
        420,
        f"Longest recorded freeze: camera {longest['camera_id']} on {pd.Timestamp(longest['date']):%d %B}.",
        20,
    )
    image("q5_duration.png", 501, 450)
    text(44, 983, "What stands out", 24, True)
    para(
        44,
        1029,
        952,
        f"<b>Frozen picture:</b> typically about {duration['duplicate frame']['median_min']:.0f} minutes.<br/>"
        f"<b>White offline picture:</b> typically {duration['offline placeholder']['median_min']:.0f} minutes.<br/>"
        f"<b>Colour bars:</b> typically {duration['test pattern']['median_min']:.0f} minutes, from only {duration['test pattern']['episodes']} episodes.",
        22,
        limit=115,
    )
    small_note(
        1181,
        "An episode is one continuous stretch of the same error; typical means the median length. Durations are estimates between snapshots. The longest freeze was still ongoing when recording ended.",
    )
    c.showPage()
    c.save()
    return pdf_path
