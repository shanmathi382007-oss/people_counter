"""
People Counter Analytics Dashboard
Streamlit web application for real-time occupancy monitoring, flow trends,
and crossing event analytics.
"""

from __future__ import annotations
import os
import sys
import time
import csv
import io
from html import escape
from collections import Counter
from datetime import datetime

import streamlit as st

# Add parent path to allow loading counter package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from counter.logger import CrossingLogger
from counter.utils import load_config

# Page configuration
st.set_page_config(
    page_title="Live People Counter Dashboard",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern CSS styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e2638 0%, #151a27 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.35);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 4px;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .metric-sub {
        font-size: 0.80rem;
        color: #8b9bb4;
    }
    .in-color { color: #2ecc71; }
    .out-color { color: #e74c3c; }
    .occ-color { color: #3498db; }
    .header-badge {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(52, 152, 219, 0.15);
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-radius: 20px;
        font-size: 0.75rem;
        color: #3498db;
        font-weight: 600;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)


def render_flow_chart(events):
    """Render entry, exit, and occupancy trends as a self-contained SVG."""
    width, height = 900, 320
    left, right, top, bottom = 58, 20, 24, 52
    plot_width = width - left - right
    plot_height = height - top - bottom
    chart_events = events[-300:]
    metrics = [
        ("in_total", "Total IN", "#2ecc71"),
        ("out_total", "Total OUT", "#e74c3c"),
        ("occupancy", "Occupancy", "#3498db"),
    ]
    max_count = max(
        1,
        max(event[key] for event in chart_events for key, _, _ in metrics),
    )
    grid = []
    for step in range(5):
        y = top + plot_height * step / 4
        value = round(max_count * (4 - step) / 4)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="#394252" stroke-width="1"/>'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#aab4c3" font-size="12">{value}</text>'
        )

    if len(chart_events) == 1:
        x_values = [left + plot_width / 2]
    else:
        x_values = [
            left + index * plot_width / (len(chart_events) - 1)
            for index in range(len(chart_events))
        ]

    series = []
    for key, label, color in metrics:
        points = [
            (
                x,
                top + plot_height * (1 - event[key] / max_count),
            )
            for x, event in zip(x_values, chart_events)
        ]
        point_list = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        series.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
            f'points="{point_list}"/>'
        )
        series.extend(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
            for x, y in points
        )

    tick_count = min(5, len(chart_events))
    tick_indices = sorted({
        round(index * (len(chart_events) - 1) / max(1, tick_count - 1))
        for index in range(tick_count)
    })
    labels = [
        f'<text x="{x_values[index]:.1f}" y="{height - 20}" text-anchor="middle" '
        f'fill="#aab4c3" font-size="11">'
        f'{escape(chart_events[index]["timestamp"].strftime("%H:%M:%S"))}</text>'
        for index in tick_indices
    ]
    legend = [
        f'<circle cx="{left + index * 150}" cy="12" r="5" fill="{color}"/>'
        f'<text x="{left + 10 + index * 150}" y="16" fill="#e5eaf1" font-size="12">'
        f'{label}</text>'
        for index, (_, label, color) in enumerate(metrics)
    ]
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Occupancy and crossing trends" '
        'style="width:100%;height:auto;background:#151a27;border-radius:8px">'
        f'{"".join(grid)}{"".join(series)}{"".join(labels)}{"".join(legend)}</svg>'
    )


def render_direction_chart(counts):
    """Render total IN and OUT crossings as a self-contained SVG."""
    width, height = 520, 260
    left, right, top, bottom = 54, 20, 28, 42
    plot_height = height - top - bottom
    max_count = max(1, *counts.values())
    grid = []
    for step in range(5):
        y = top + plot_height * step / 4
        value = round(max_count * (4 - step) / 4)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="#394252" stroke-width="1"/>'
            f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#aab4c3" font-size="12">{value}</text>'
        )
    colors = {"IN": "#2ecc71", "OUT": "#e74c3c"}
    bars = []
    slot_width = (width - left - right) / 2
    bar_width = slot_width * 0.45
    for index, direction in enumerate(("IN", "OUT")):
        count = counts.get(direction, 0)
        bar_height = plot_height * count / max_count
        x = left + slot_width * index + (slot_width - bar_width) / 2
        y = top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" rx="5" fill="{colors[direction]}"/>'
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 14}" text-anchor="middle" '
            f'fill="#e5eaf1" font-size="13">{direction}</text>'
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" '
            f'fill="#e5eaf1" font-size="12">{count}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Crossings by direction" '
        'style="width:100%;height:auto;background:#151a27;border-radius:8px">'
        f'{"".join(grid)}{"".join(bars)}</svg>'
    )


def render_hourly_chart(counts):
    """Render hourly crossing totals grouped by direction as a self-contained SVG."""
    width, height = 660, 260
    left, right, top, bottom = 44, 12, 22, 40
    plot_width = width - left - right
    plot_height = height - top - bottom
    hours = sorted({hour for hour, _ in counts})
    if not hours:
        hours = [0]
    max_count = max(1, *counts.values())
    grid = []
    for step in range(5):
        y = top + plot_height * step / 4
        value = round(max_count * (4 - step) / 4)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="#394252" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#aab4c3" font-size="11">{value}</text>'
        )
    slot_width = plot_width / len(hours)
    bar_width = min(14, slot_width * 0.32)
    bars = []
    for index, hour in enumerate(hours):
        center = left + (index + 0.5) * slot_width
        for direction, offset, color in (
            ("IN", -bar_width / 2, "#2ecc71"),
            ("OUT", bar_width / 2, "#e74c3c"),
        ):
            count = counts.get((hour, direction), 0)
            bar_height = plot_height * count / max_count
            x = center + offset
            y = top + plot_height - bar_height
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}"/>'
            )
        bars.append(
            f'<text x="{center:.1f}" y="{height - 14}" text-anchor="middle" '
            f'fill="#aab4c3" font-size="10">{hour}</text>'
        )
    legend = (
        '<circle cx="475" cy="12" r="5" fill="#2ecc71"/>'
        '<text x="485" y="16" fill="#e5eaf1" font-size="11">IN</text>'
        '<circle cx="530" cy="12" r="5" fill="#e74c3c"/>'
        '<text x="540" y="16" fill="#e5eaf1" font-size="11">OUT</text>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Crossings by hour and direction" '
        'style="width:100%;height:auto;background:#151a27;border-radius:8px">'
        f'{"".join(grid)}{"".join(bars)}{legend}</svg>'
    )


def render_events_table(events):
    """Render filtered event records as an escaped HTML table."""
    columns = [
        ("id", "Event #"),
        ("timestamp", "Timestamp"),
        ("track_id", "Track ID"),
        ("direction", "Direction"),
        ("in_total", "Total IN"),
        ("out_total", "Total OUT"),
        ("occupancy", "Occupancy"),
    ]
    header = "".join(
        f'<th scope="col">{label}</th>' for _, label in columns
    )
    rows = []
    for event in events:
        cells = []
        for key, _ in columns:
            value = event[key]
            if key == "timestamp":
                value = value.strftime("%Y-%m-%d %H:%M:%S")
            cells.append(f"<td>{escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse">'
        f'<thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def init_dashboard():
    """Load configuration and initialize logger."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    config = load_config(config_path)
    db_path = config.get("logging", {}).get("db_path", "people_counter.db")
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_dir, db_path)

    csv_path = config.get("logging", {}).get("csv_path", "outputs/crossings_log.csv")
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(project_dir, csv_path)

    logger = CrossingLogger(db_path=db_path, default_csv_path=csv_path)
    return config, logger


def main():
    config, logger = init_dashboard()

    # Sidebar Controls
    with st.sidebar:
        st.title("⚙️ Controls & Settings")
        st.markdown("Monitor real-time pedestrian flow and line crossing statistics.")

        # Auto-refresh interval
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_rate = st.slider("Refresh Interval (seconds)", min_value=1, max_value=30, value=3)

        st.divider()

        # Database info
        st.subheader("Database Info")
        st.caption(f"**DB Path:** `{logger.db_path}`")
        db_exists = os.path.exists(logger.db_path)
        if db_exists:
            st.success("🟢 Connected to SQLite")
        else:
            st.warning("🟡 Database file not created yet (starts on first crossing)")

        st.divider()
        if st.button("🔄 Manual Refresh Now", use_container_width=True):
            st.rerun()

    # Main Header
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.title("🚶 Real-Time People Counter")
        st.caption(f"Active Monitoring • YOLOv8 + ByteTrack Virtual Line Crossings • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col_head2:
        st.markdown("<div style='text-align: right; padding-top: 15px;'><span class='header-badge'>LIVE FEED ACTIVE</span></div>", unsafe_allow_html=True)

    # Fetch data
    df = logger.get_all_events()
    for event in df:
        event["timestamp"] = datetime.fromisoformat(event["timestamp"])
    in_total, out_total, occupancy = logger.get_latest_counts()

    # 1. Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title in-color">Total Entries (IN)</div>
            <div class="metric-value in-color">{in_total}</div>
            <div class="metric-sub">Cumulative pedestrian entries</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title out-color">Total Exits (OUT)</div>
            <div class="metric-value out-color">{out_total}</div>
            <div class="metric-sub">Cumulative pedestrian exits</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        occ_style = "occ-color"
        status_label = "Optimal"
        if occupancy > 50:
            status_label = "High Capacity"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title {occ_style}">Current Occupancy</div>
            <div class="metric-value {occ_style}">{occupancy}</div>
            <div class="metric-sub">Active inside ({status_label})</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        total_crossings = in_total + out_total
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title" style="color: #9b59b6;">Total Traffic</div>
            <div class="metric-value" style="color: #9b59b6;">{total_crossings}</div>
            <div class="metric-sub">All bidirectional crossings</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Section
    if df:
        st.subheader("📈 Flow & Occupancy Trends")

        tab_chart1, tab_chart2 = st.tabs(["Occupancy Over Time", "Direction Breakdown"])

        with tab_chart1:
            metric_labels = {
                "occupancy": "Occupancy",
                "in_total": "Total IN",
                "out_total": "Total OUT",
            }
            st.markdown(render_flow_chart(df), unsafe_allow_html=True)

        with tab_chart2:
            col_bar1, col_bar2 = st.columns(2)
            with col_bar1:
                direction_counts = Counter(event["direction"] for event in df)
                st.markdown(
                    render_direction_chart(direction_counts),
                    unsafe_allow_html=True,
                )

            with col_bar2:
                hourly_counts = Counter(
                    (event["timestamp"].hour, event["direction"]) for event in df
                )
                st.markdown(
                    render_hourly_chart(hourly_counts),
                    unsafe_allow_html=True,
                )

    else:
        st.info("ℹ️ No crossing events recorded yet. Start `main.py` to stream video and record crossings.")

    # 3. Recent Events Table & Export
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Recent Crossing Events")

    if df:
        col_filter1, col_filter2, col_export = st.columns([2, 2, 2])
        with col_filter1:
            dir_filter = st.selectbox("Filter Direction", ["All", "IN", "OUT"])
        with col_filter2:
            search_id = st.text_input("Filter Track ID", "")

        filtered_df = df
        if dir_filter != "All":
            filtered_df = [
                event for event in filtered_df if event["direction"] == dir_filter
            ]
        if search_id.strip().isdigit():
            track_id = int(search_id.strip())
            filtered_df = [
                event for event in filtered_df if event["track_id"] == track_id
            ]

        # Display descending table
        display_df = sorted(filtered_df, key=lambda event: event["id"], reverse=True)
        st.markdown(render_events_table(display_df), unsafe_allow_html=True)

        with col_export:
            csv_buffer = io.StringIO()
            fieldnames = [
                "id", "timestamp", "track_id", "direction",
                "in_total", "out_total", "occupancy",
            ]
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_df)
            csv_data = csv_buffer.getvalue().encode("utf-8")
            st.download_button(
                label="📥 Download CSV Log",
                data=csv_data,
                file_name=f"people_counter_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.caption("Waiting for line crossing events...")

    # Auto refresh trigger
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
