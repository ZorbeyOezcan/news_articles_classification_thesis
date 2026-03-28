"""
Interactive Dashboard for classified news articles.
Run: streamlit run Python/data_dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "articles_enriched.csv"

st.set_page_config(page_title="Artikel-Klassifikation Dashboard", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["date"] = df["date_time"].dt.date
    return df


df = load_data()

st.title("Artikel-Klassifikation Dashboard")

# ── Sidebar filters ──
st.sidebar.header("Filter")

# Domain filter
domains = sorted(df["domain"].dropna().unique())
sel_domains = st.sidebar.multiselect("Domain", domains, default=[])

# Prediction filter
predictions = sorted(df["prediction"].dropna().unique())
sel_predictions = st.sidebar.multiselect("Prediction (Kategorie)", predictions, default=[])

# Medientyp filter
medientypen = sorted(df["Medientyp"].dropna().unique())
sel_medientyp = st.sidebar.multiselect("Medientyp", medientypen, default=[])

# Bundesland filter
bundeslaender = sorted(df["Bundesland"].dropna().unique())
sel_bundesland = st.sidebar.multiselect("Bundesland", bundeslaender, default=[])

# Verlag filter
verlage = sorted(df["Verlag"].dropna().unique())
sel_verlag = st.sidebar.multiselect("Verlag", verlage, default=[])

# Prediction score slider
min_score, max_score = float(df["prediction_score"].min()), float(df["prediction_score"].max())
sel_score = st.sidebar.slider("Min. Prediction Score", min_score, max_score, min_score, step=0.01)

# Date range
min_date = df["date"].min()
max_date = df["date"].max()
sel_date_range = st.sidebar.date_input("Zeitraum", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Apply filters
filtered = df.copy()
if sel_domains:
    filtered = filtered[filtered["domain"].isin(sel_domains)]
if sel_predictions:
    filtered = filtered[filtered["prediction"].isin(sel_predictions)]
if sel_medientyp:
    filtered = filtered[filtered["Medientyp"].isin(sel_medientyp)]
if sel_bundesland:
    filtered = filtered[filtered["Bundesland"].isin(sel_bundesland)]
if sel_verlag:
    filtered = filtered[filtered["Verlag"].isin(sel_verlag)]
filtered = filtered[filtered["prediction_score"] >= sel_score]
if len(sel_date_range) == 2:
    filtered = filtered[(filtered["date"] >= sel_date_range[0]) & (filtered["date"] <= sel_date_range[1])]

# ── KPIs ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Artikel gesamt", f"{len(filtered):,}")
col2.metric("Domains", filtered["domain"].nunique())
col3.metric("Kategorien", filtered["prediction"].nunique())
col4.metric("Ø Prediction Score", f"{filtered['prediction_score'].mean():.3f}")

st.divider()

# ── Charts ──
tab_overview, tab_time, tab_media, tab_detail, tab_edit = st.tabs(
    ["Kategorien", "Zeitverlauf", "Medien-Info", "Artikel-Detail", "Daten bearbeiten"]
)

# -- Tab 1: Kategorien --
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        pred_counts = filtered["prediction"].value_counts().reset_index()
        pred_counts.columns = ["Kategorie", "Anzahl"]
        fig = px.bar(pred_counts, x="Kategorie", y="Anzahl", title="Artikel pro Kategorie", color="Kategorie")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        domain_counts = filtered.groupby("domain")["prediction"].value_counts().reset_index()
        domain_counts.columns = ["Domain", "Kategorie", "Anzahl"]
        fig2 = px.bar(domain_counts, x="Domain", y="Anzahl", color="Kategorie",
                      title="Kategorien pro Domain", barmode="stack")
        st.plotly_chart(fig2, use_container_width=True)

    # Score distribution
    score_cols = [c for c in filtered.columns if c.startswith("score_")]
    if score_cols:
        st.subheader("Score-Verteilung (alle Kategorien)")
        score_means = filtered[score_cols].mean().reset_index()
        score_means.columns = ["Kategorie", "Ø Score"]
        score_means["Kategorie"] = score_means["Kategorie"].str.replace("score_", "")
        fig3 = px.bar(score_means, x="Kategorie", y="Ø Score", title="Durchschnittliche Scores")
        st.plotly_chart(fig3, use_container_width=True)

# -- Tab 2: Zeitverlauf --
with tab_time:
    daily = filtered.groupby(["date", "prediction"]).size().reset_index(name="Anzahl")
    fig_time = px.line(daily, x="date", y="Anzahl", color="prediction",
                       title="Artikel pro Tag nach Kategorie")
    st.plotly_chart(fig_time, use_container_width=True)

    weekly = filtered.copy()
    weekly["week"] = weekly["date_time"].dt.isocalendar().week.astype(int)
    weekly_agg = weekly.groupby(["week", "prediction"]).size().reset_index(name="Anzahl")
    fig_week = px.bar(weekly_agg, x="week", y="Anzahl", color="prediction",
                      title="Artikel pro Woche nach Kategorie", barmode="stack")
    st.plotly_chart(fig_week, use_container_width=True)

# -- Tab 3: Medien-Info --
with tab_media:
    media_cols = ["Publikation", "Medientyp", "Verlag", "Ausgabe", "Markengruppe",
                  "Hauptausgabe", "redaktionelleEinheit", "Ort", "Bundesland", "Region",
                  "Rhythmus", "Erscheinungstage", "paidcontent", "Themengebiete"]
    available_media_cols = [c for c in media_cols if c in filtered.columns]

    c1, c2 = st.columns(2)
    with c1:
        if "Medientyp" in filtered.columns:
            mt_counts = filtered["Medientyp"].value_counts().reset_index()
            mt_counts.columns = ["Medientyp", "Anzahl"]
            fig_mt = px.pie(mt_counts, names="Medientyp", values="Anzahl", title="Verteilung Medientyp")
            st.plotly_chart(fig_mt, use_container_width=True)

    with c2:
        if "Bundesland" in filtered.columns:
            bl_counts = filtered["Bundesland"].value_counts().reset_index()
            bl_counts.columns = ["Bundesland", "Anzahl"]
            fig_bl = px.bar(bl_counts, x="Bundesland", y="Anzahl", title="Artikel pro Bundesland")
            st.plotly_chart(fig_bl, use_container_width=True)

    if "Verlag" in filtered.columns:
        verlag_counts = filtered["Verlag"].value_counts().head(20).reset_index()
        verlag_counts.columns = ["Verlag", "Anzahl"]
        fig_v = px.bar(verlag_counts, x="Anzahl", y="Verlag", orientation="h",
                       title="Top 20 Verlage")
        fig_v.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_v, use_container_width=True)

    st.subheader("Medien-Metadaten pro Domain")
    domain_media = filtered.drop_duplicates(subset="domain")[["domain"] + available_media_cols]
    st.dataframe(domain_media, use_container_width=True, hide_index=True)

# -- Tab 4: Artikel-Detail --
with tab_detail:
    st.subheader("Artikelliste")
    display_cols = ["id", "domain", "headline", "prediction", "prediction_score", "date", "author",
                    "Medientyp", "Verlag", "Bundesland"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].sort_values("prediction_score", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    st.subheader("Artikel-Suche")
    search_term = st.text_input("Suche in Headline oder Text")
    if search_term:
        mask = (
            filtered["headline"].str.contains(search_term, case=False, na=False)
            | filtered["text"].str.contains(search_term, case=False, na=False)
        )
        results = filtered[mask]
        st.write(f"{len(results)} Treffer")
        st.dataframe(
            results[display_cols].sort_values("prediction_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# -- Tab 5: Daten bearbeiten --
with tab_edit:
    st.subheader("Prediction manuell korrigieren")
    st.caption("Wähle einen Artikel per ID und weise ihm eine neue Kategorie zu.")

    edit_id = st.number_input("Artikel-ID", min_value=int(df["id"].min()), max_value=int(df["id"].max()), step=1)
    article = df[df["id"] == edit_id]
    if not article.empty:
        row = article.iloc[0]
        st.write(f"**Headline:** {row['headline']}")
        st.write(f"**Domain:** {row['domain']}")
        st.write(f"**Aktuelle Prediction:** {row['prediction']} (Score: {row['prediction_score']:.4f})")

        new_pred = st.selectbox("Neue Kategorie", predictions, index=predictions.index(row["prediction"]))

        if st.button("Speichern"):
            full_df = pd.read_csv(DATA_PATH, low_memory=False)
            full_df.loc[full_df["id"] == edit_id, "prediction"] = new_pred
            full_df.to_csv(DATA_PATH, index=False)
            st.success(f"Artikel {edit_id} -> '{new_pred}' gespeichert. Seite neu laden (F5) um Änderungen zu sehen.")
    else:
        st.warning("Artikel-ID nicht gefunden.")
