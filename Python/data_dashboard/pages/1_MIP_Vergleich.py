"""
MIP Comparison: Media Agenda vs. Most Important Problem survey data.
Interactive line chart comparing daily news topic coverage with MIP survey results.
"""

import datetime
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ── Paths ──
DASHBOARD_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = DASHBOARD_DIR / "data" / "articles_enriched.csv"
PROJECT_ROOT = DASHBOARD_DIR.parent.parent  # news_articles_classification_thesis/
MIP_PATH = PROJECT_ROOT / "data" / "most_important_problem" / "mip_wide_dataframe.csv"

st.set_page_config(page_title="MIP Comparison", layout="wide")

# ── Consistent color map for topics ──
TOPIC_COLORS = {
    "AfD/Rechte": "#1f77b4",
    "Arbeitslosigkeit": "#ff7f0e",
    "Bundeswehr/Verteidigung": "#2ca02c",
    "Gesundheitswesen, Pflege": "#d62728",
    "Klima / Energie": "#9467bd",
    "Kosten/Löhne/Preise": "#8c564b",
    "Politikverdruss": "#e377c2",
    "Renten": "#7f7f7f",
    "Soziales Gefälle": "#bcbd22",
    "Ukraine/Krieg/Russland": "#17becf",
    "Wirtschaftslage": "#ff9896",
    "Zuwanderung": "#aec7e8",
}

# Extra line colors (for up to 6 extra lines)
EXTRA_COLORS = ["#e6550d", "#31a354", "#756bb1", "#636363", "#de2d26", "#3182bd"]


@st.cache_data
def load_articles():
    cols = ["date_time", "domain", "prediction", "prediction_score",
            "political_spectrum", "quality", "state"]
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["date_time"])
    df["date"] = df["date_time"].dt.date
    return df


@st.cache_data
def load_mip():
    mip = pd.read_csv(MIP_PATH)
    mip["date"] = pd.to_datetime(mip["date"]).dt.date
    return mip


df = load_articles()
mip_df = load_mip()

# Categories excluding "Andere"
ALL_CATEGORIES = sorted([c for c in df["prediction"].dropna().unique() if c != "Andere"])
MIP_TOPICS = [c for c in mip_df.columns if c != "date"]

st.title("Media Agenda vs. Most Important Problem")

# ── Layout: chart (left) + filters (right) ──
col_chart, col_filters = st.columns([7, 3])

# Store all computed curves for area-between-curves calculation
all_curves = {}

# ── Right column: Filters ──
with col_filters:
    # Axis settings
    with st.expander("Axis Settings", expanded=False):
        y_max = st.slider("Y-axis maximum (%)", 5, 100, 50, step=5)
        min_date = df["date"].min()
        max_date = df["date"].max()
        sel_date_range = st.date_input(
            "Date range", value=(min_date, max_date),
            min_value=min_date, max_value=max_date,
        )

    # Article filters
    with st.expander("Article Filters", expanded=True):
        domains = sorted(df["domain"].dropna().unique())
        sel_domains = st.multiselect("Domain", domains, default=[])

        states = sorted(df["state"].dropna().unique())
        sel_states = st.multiselect("State", states, default=[])

        spectrums = sorted(df["political_spectrum"].dropna().unique())
        sel_spectrum = st.multiselect("Political Spectrum", spectrums, default=[])

        qualities = sorted(df["quality"].dropna().unique())
        sel_quality = st.multiselect("Quality", qualities, default=[])

        score_min = st.slider(
            "Min. Prediction Score (%)", 0, 100, 0, step=1,
            help="Only include articles with score >= value/100",
        )

    # Prediction category lines
    with st.expander("Categories (Lines)", expanded=True):
        sel_preds = st.multiselect(
            "Article Categories", ALL_CATEGORIES, default=[],
        )

    # MIP overlay
    with st.expander("MIP Overlay", expanded=True):
        sel_mip = st.multiselect("MIP Topics", MIP_TOPICS, default=[])
        mip_shift = st.slider("MIP Time Shift (days)", -60, 60, 0)

    # Trendline settings
    with st.expander("Trendline", expanded=False):
        trend_type = st.selectbox(
            "Type", ["None", "Moving Average", "Exp. Moving Average (EMA)"],
        )
        if trend_type != "None":
            trend_window = st.slider("Window (days)", 2, 30, 7)
            show_raw = st.checkbox("Show raw data", value=True)

    # Extra lines (custom filtered lines)
    with st.expander("Extra Lines", expanded=False):
        n_extra = st.number_input("Number of extra lines", 0, 6, 0, step=1)
        extra_lines_config = []
        for i in range(int(n_extra)):
            st.markdown(f"**Line {i+1}**")
            ex_pred = st.selectbox(f"Category #{i+1}", ALL_CATEGORIES, key=f"ex_pred_{i}")
            ex_quality = st.multiselect(f"Quality #{i+1}", qualities, default=[], key=f"ex_qual_{i}")
            ex_spectrum = st.multiselect(f"Political Spectrum #{i+1}", spectrums, default=[], key=f"ex_spec_{i}")
            ex_domain = st.multiselect(f"Domain #{i+1}", domains, default=[], key=f"ex_dom_{i}")
            # Build label from selections
            parts = [ex_pred]
            if ex_quality:
                parts.append("+".join(ex_quality))
            if ex_spectrum:
                parts.append("+".join(ex_spectrum))
            if ex_domain:
                parts.append("+".join(ex_domain[:2]) + ("..." if len(ex_domain) > 2 else ""))
            label = " | ".join(parts)
            extra_lines_config.append({
                "label": label, "prediction": ex_pred,
                "quality": ex_quality, "spectrum": ex_spectrum, "domain": ex_domain,
            })
            st.divider()

    # Area between curves
    with st.expander("Area Between Curves", expanded=False):
        st.caption("Select two curves to compare. The area between them will be calculated.")
        abc_curve1 = st.selectbox("Curve 1", ["(none)"], key="abc1")
        abc_curve2 = st.selectbox("Curve 2", ["(none)"], key="abc2")

# ── Compute ──
# Base mask: exclude "Andere", apply global filters
mask = df["prediction"] != "Andere"
if sel_domains:
    mask = mask & df["domain"].isin(sel_domains)
if sel_states:
    mask = mask & df["state"].isin(sel_states)
if sel_spectrum:
    mask = mask & df["political_spectrum"].isin(sel_spectrum)
if sel_quality:
    mask = mask & df["quality"].isin(sel_quality)
if score_min > 0:
    mask = mask & (df["prediction_score"] >= score_min / 100.0)
if len(sel_date_range) == 2:
    mask = mask & (df["date"] >= sel_date_range[0]) & (df["date"] <= sel_date_range[1])

filtered = df[mask]

# Date range for reindexing
if len(sel_date_range) == 2:
    all_dates = pd.date_range(sel_date_range[0], sel_date_range[1], freq="D").date
else:
    all_dates = pd.date_range(filtered["date"].min(), filtered["date"].max(), freq="D").date


def compute_daily_pct(data, prediction, all_dates):
    """Compute daily % for a prediction category within given data."""
    total = data.groupby("date").size()
    counts = data[data["prediction"] == prediction].groupby("date").size()
    pct = (counts / total * 100).reindex(all_dates, fill_value=0)
    return pct


def apply_trend(series, trend_type, trend_window):
    """Apply trendline smoothing to a series."""
    if trend_type == "Moving Average":
        return series.rolling(window=trend_window, min_periods=1, center=True).mean()
    elif trend_type == "Exp. Moving Average (EMA)":
        return series.ewm(span=trend_window, adjust=False).mean()
    return series


# ── Build chart ──
fig = go.Figure()

# Main article lines
if sel_preds and not filtered.empty:
    daily_total = filtered.groupby("date").size()
    daily_pred = (
        filtered[filtered["prediction"].isin(sel_preds)]
        .groupby(["date", "prediction"])
        .size()
        .reset_index(name="count")
    )
    daily_pred["total"] = daily_pred["date"].map(daily_total)
    daily_pred["pct"] = (daily_pred["count"] / daily_pred["total"]) * 100

    for pred in sel_preds:
        color = TOPIC_COLORS.get(pred, "#333333")
        subset = daily_pred[daily_pred["prediction"] == pred].set_index("date")["pct"]
        subset = subset.reindex(all_dates, fill_value=0)

        # Determine the curve values to store (use trend if active, else raw)
        if trend_type != "None":
            curve_values = apply_trend(subset, trend_type, trend_window)
        else:
            curve_values = subset
        curve_name = f"Articles: {pred}"
        all_curves[curve_name] = {"dates": np.array(all_dates), "values": curve_values.values}

        subset_df = subset.reset_index()
        subset_df.columns = ["date", "pct"]

        # Raw data line
        if trend_type == "None" or show_raw:
            fig.add_trace(go.Scatter(
                x=subset_df["date"], y=subset_df["pct"],
                mode="lines", name=curve_name,
                line=dict(width=1.5 if trend_type != "None" else 2, color=color),
                opacity=0.35 if trend_type != "None" else 1.0,
                legendgroup=pred,
                showlegend=trend_type == "None",
            ))

        # Trendline
        if trend_type != "None":
            smoothed = apply_trend(subset_df["pct"], trend_type, trend_window)
            fig.add_trace(go.Scatter(
                x=subset_df["date"], y=smoothed,
                mode="lines", name=curve_name,
                line=dict(width=3, color=color),
                legendgroup=pred,
            ))

# MIP overlay
if sel_mip:
    mip_valid = mip_df.dropna(subset=sel_mip, how="all")
    for topic in sel_mip:
        color = TOPIC_COLORS.get(topic, "#333333")
        topic_data = mip_valid[["date", topic]].dropna()
        if topic_data.empty:
            continue
        shifted_dates = np.array([
            d + datetime.timedelta(days=mip_shift) for d in topic_data["date"]
        ])
        curve_name = f"MIP: {topic}"
        all_curves[curve_name] = {"dates": shifted_dates, "values": topic_data[topic].values}

        fig.add_trace(go.Scatter(
            x=shifted_dates, y=topic_data[topic],
            mode="lines+markers", name=curve_name,
            line=dict(dash="dash", width=2.5, color=color),
            marker=dict(size=8, color=color),
            legendgroup=topic,
        ))

# Extra lines
if extra_lines_config and not filtered.empty:
    base_mask = df["prediction"] != "Andere"
    if score_min > 0:
        base_mask = base_mask & (df["prediction_score"] >= score_min / 100.0)
    if len(sel_date_range) == 2:
        base_mask = base_mask & (df["date"] >= sel_date_range[0]) & (df["date"] <= sel_date_range[1])
    base_data = df[base_mask]

    for i, cfg in enumerate(extra_lines_config):
        ex_mask = base_data["prediction"] != "___"  # always True
        if cfg["quality"]:
            ex_mask = ex_mask & base_data["quality"].isin(cfg["quality"])
        if cfg["spectrum"]:
            ex_mask = ex_mask & base_data["political_spectrum"].isin(cfg["spectrum"])
        if cfg["domain"]:
            ex_mask = ex_mask & base_data["domain"].isin(cfg["domain"])
        ex_data = base_data[ex_mask]

        if ex_data.empty:
            continue

        pct = compute_daily_pct(ex_data, cfg["prediction"], all_dates)
        if trend_type != "None":
            curve_values = apply_trend(pct, trend_type, trend_window)
        else:
            curve_values = pct

        curve_name = f"Extra: {cfg['label']}"
        color = EXTRA_COLORS[i % len(EXTRA_COLORS)]
        all_curves[curve_name] = {"dates": np.array(all_dates), "values": curve_values.values}

        # Raw
        if trend_type == "None" or show_raw:
            fig.add_trace(go.Scatter(
                x=all_dates, y=pct.values,
                mode="lines", name=curve_name,
                line=dict(width=1.5 if trend_type != "None" else 2, color=color, dash="dot"),
                opacity=0.35 if trend_type != "None" else 1.0,
                legendgroup=f"extra_{i}",
                showlegend=trend_type == "None",
            ))

        if trend_type != "None":
            smoothed = apply_trend(pct, trend_type, trend_window)
            fig.add_trace(go.Scatter(
                x=all_dates, y=smoothed.values,
                mode="lines", name=curve_name,
                line=dict(width=3, color=color, dash="dot"),
                legendgroup=f"extra_{i}",
            ))

fig.update_layout(
    title="Media Agenda vs. Most Important Problem",
    xaxis_title="Date",
    yaxis_title="% of articles (excl. 'Andere')",
    yaxis=dict(range=[0, y_max]),
    legend=dict(orientation="v", x=1.02, y=1),
    height=700,
    hovermode="x unified",
)

# ── Area Between Curves computation ──
def compute_area_between(c1_data, c2_data):
    """Compute area between two curves using trapezoidal rule on overlapping dates."""
    common_dates = sorted(set(c1_data["dates"]) & set(c2_data["dates"]))
    if len(common_dates) < 2:
        s1 = pd.Series(c1_data["values"], index=pd.to_datetime(c1_data["dates"]))
        s2 = pd.Series(c2_data["values"], index=pd.to_datetime(c2_data["dates"]))
        combined_idx = s1.index.union(s2.index)
        s1 = s1.reindex(combined_idx).interpolate(method="index")
        s2 = s2.reindex(combined_idx).interpolate(method="index")
        overlap = s1.index.intersection(s2.index)
        if len(overlap) < 2:
            return None
        diff = (s1.loc[overlap] - s2.loc[overlap]).abs()
        days = np.array([(d - overlap[0]).days for d in overlap], dtype=float)
        return float(np.trapezoid(diff.values, days))
    else:
        s1 = pd.Series(c1_data["values"], index=c1_data["dates"])
        s2 = pd.Series(c2_data["values"], index=c2_data["dates"])
        vals1 = s1.loc[common_dates].values
        vals2 = s2.loc[common_dates].values
        diff = np.abs(vals1 - vals2)
        days = np.array([(d - common_dates[0]).days for d in common_dates], dtype=float)
        return float(np.trapezoid(diff, days))


# ── Render ──
with col_chart:
    if filtered.empty:
        st.warning("No articles match the selected filters.")
    elif not sel_preds and not sel_mip and not extra_lines_config:
        st.info("Select at least one article category or MIP topic.")
    st.plotly_chart(fig, use_container_width=True)

    # Bottom row: stats left, area-between-curves right
    col_stats, col_abc = st.columns(2)
    with col_stats:
        st.caption(f"Filtered articles (excl. 'Andere'): {len(filtered):,}")

    with col_abc:
        if len(all_curves) >= 2:
            curve_names = list(all_curves.keys())
            abc1 = st.selectbox("Curve 1", ["(none)"] + curve_names, key="abc_sel1")
            abc2 = st.selectbox("Curve 2", ["(none)"] + curve_names, key="abc_sel2")
            if abc1 != "(none)" and abc2 != "(none)" and abc1 != abc2:
                area = compute_area_between(all_curves[abc1], all_curves[abc2])
                if area is not None:
                    st.metric("Area Between Curves", f"{area:.1f} %-days")
                else:
                    st.warning("Not enough overlapping data points.")
