#!/usr/bin/env python3
"""
Visualisierung aller Performance-Reports.

Lädt ALLE Reports automatisch von Google Drive herunter und erstellt
eine Übersicht aller Modelle, sortiert nach F1 Macro (höchstes oben).

Usage:
    python visualize_all_reports.py                          # Download + PNGs speichern + öffnen
    python visualize_all_reports.py --no-open                # PNGs speichern, nicht öffnen
    python visualize_all_reports.py --no-download            # Nur lokal, kein Download
"""

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive Backend, verhindert hängen
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_REPORTS_DIR = _SCRIPT_DIR / "performance_reports"

GDRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1ecwGugqLR0P8qbFElK9x-iR9gB4JcWSP"
)

# Kanonische Label-Reihenfolge (kurze Namen)
CANONICAL_LABELS = [
    "Klima / Energie",
    "Zuwanderung",
    "Renten",
    "Soziales Gefälle",
    "AfD/Rechte",
    "Arbeitslosigkeit",
    "Wirtschaftslage",
    "Politikverdruss",
    "Gesundheitswesen, Pflege",
    "Kosten/Löhne/Preise",
    "Ukraine/Krieg/Russland",
    "Bundeswehr/Verteidigung",
    "Andere",
]


# ---------------------------------------------------------------------------
# Google Drive Sync
# ---------------------------------------------------------------------------

def sync_reports_from_drive(reports_dir: Path) -> None:
    """Lädt ALLE Reports von Google Drive herunter (überschreibt vorhandene)."""
    import requests as _req
    from gdown.download_folder import _download_and_parse_google_drive_link

    print("Lade Reports von Google Drive herunter …")
    print(f"  Quelle:  {GDRIVE_FOLDER_URL}")
    print(f"  Ziel:    {reports_dir}")

    reports_dir.mkdir(parents=True, exist_ok=True)

    # Schritt 1: Folder-Listing via gdown (funktioniert zuverlässig)
    session = _req.Session()
    try:
        ok, gdrive_file = _download_and_parse_google_drive_link(
            sess=session, url=GDRIVE_FOLDER_URL, quiet=False,
        )
        if not ok or gdrive_file is None:
            raise RuntimeError("Folder-Listing fehlgeschlagen.")
    except Exception as e:
        print(f"\n  ⚠ Folder-Listing fehlgeschlagen: {e}")
        print("    Verwende bereits vorhandene lokale Reports.\n")
        n_json = len(list(reports_dir.glob("*.json")))
        if n_json == 0:
            raise FileNotFoundError(
                "Keine lokalen Reports vorhanden.\n"
                "Nutze --no-download wenn Reports bereits lokal vorliegen."
            )
        print(f"  {n_json} JSON-Report(s) lokal verfügbar.\n")
        return

    # Schritt 2: Nur JSON/MD einzeln herunterladen via requests
    report_files = [
        c for c in gdrive_file.children
        if c.name.endswith(".json") or c.name.endswith(".md")
    ]
    print(f"  {len(report_files)} Report-Dateien gefunden. Lade herunter …")

    downloaded, failed = 0, 0
    for f in report_files:
        dest = reports_dir / f.name
        try:
            resp = session.get(
                "https://drive.google.com/uc",
                params={"id": f.id, "export": "download", "confirm": "t"},
                timeout=30,
            )
            if resp.status_code == 200 and "text/html" not in resp.headers.get("Content-Type", ""):
                dest.write_bytes(resp.content)
                downloaded += 1
            else:
                failed += 1
                print(f"    ⚠ Fehlgeschlagen: {f.name}")
        except Exception:
            failed += 1
            print(f"    ⚠ Fehlgeschlagen: {f.name}")

    print(f"  {downloaded} heruntergeladen, {failed} fehlgeschlagen.")

    n_json = len(list(reports_dir.glob("*.json")))
    if n_json == 0:
        raise FileNotFoundError(
            "Download fehlgeschlagen und keine lokalen Reports vorhanden.\n"
            "Nutze --no-download wenn Reports bereits lokal vorliegen."
        )
    print(f"  {n_json} JSON-Report(s) verfügbar.\n")


# ---------------------------------------------------------------------------
# Label-Normalisierung
# ---------------------------------------------------------------------------

def _build_nli_to_canonical(report: Dict[str, Any]) -> Dict[str, str]:
    """Erstellt Mapping von NLI-Phrasen zurück auf kanonische Label-Namen."""
    mapping = report.get("classification_config", {}).get("label_mapping", {})
    # label_mapping: { "Klima / Energie": "vom Klima, …" }  (canonical -> nli)
    reverse = {}
    for canonical, nli_phrase in mapping.items():
        if canonical != nli_phrase:
            reverse[nli_phrase] = canonical
    return reverse


def _normalize_label(label: str, nli_map: Dict[str, str]) -> str:
    """Mappt ein Label auf den kanonischen Kurznamen."""
    return nli_map.get(label, label)


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------

def load_all_reports(reports_dir: Path) -> List[Dict[str, Any]]:
    """Lädt alle JSON-Report-Dateien aus dem Verzeichnis."""
    json_files = sorted(reports_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"Keine JSON-Reports gefunden in: {reports_dir}\n"
            f"Bitte mit --no-download prüfen oder Ordner auf Google Drive freigeben."
        )

    reports = []
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        data["_source_file"] = jf.name
        reports.append(data)

    print(f"{len(reports)} Report(s) geladen aus: {reports_dir}")
    return reports


def reports_to_dataframe(reports: List[Dict[str, Any]]) -> pd.DataFrame:
    """Konvertiert Reports in einen sortierten DataFrame (F1 Macro absteigend)."""
    rows = []
    for r in reports:
        model = r.get("model", {})
        metrics = r.get("metrics", {})
        runtime = r.get("runtime", {})
        dataset = r.get("dataset", {})

        model_name = model.get("name", r.get("report_id", "unknown"))
        model_type = model.get("type", "")
        report_id = r.get("report_id", "")

        # Report-Nummer aus der ID extrahieren (z.B. "190226_model_report_002" -> "002")
        parts = report_id.rsplit("_", 1)
        suffix = parts[-1] if len(parts) > 1 and parts[-1].isdigit() else ""

        # Eindeutiger Anzeigename: bei Duplikaten Report-Nummer anhängen
        display_name = f"{model_name} ({model_type})"
        if suffix and suffix != "001":
            display_name = f"{model_name} v{int(suffix)} ({model_type})"

        rows.append({
            "report_id": report_id,
            "model_name": model_name,
            "model_type": model_type,
            "display_name": display_name,
            "parameters": model.get("parameters", "N/A"),
            "f1_macro": metrics.get("f1_macro", 0),
            "f1_weighted": metrics.get("f1_weighted", 0),
            "precision_macro": metrics.get("precision_macro", 0),
            "precision_weighted": metrics.get("precision_weighted", 0),
            "recall_macro": metrics.get("recall_macro", 0),
            "recall_weighted": metrics.get("recall_weighted", 0),
            "accuracy": metrics.get("accuracy", 0),
            "n_articles": dataset.get("n_articles", 0),
            "duration_s": runtime.get("duration_seconds", 0),
            "gpu": runtime.get("gpu", {}).get("gpu_name", "N/A"),
        })

    df = pd.DataFrame(rows)
    df.sort_values("f1_macro", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def get_per_class_data(reports: List[Dict[str, Any]], df_sorted: pd.DataFrame) -> pd.DataFrame:
    """Erstellt DataFrame mit normalisierten Per-Class F1 für alle Modelle."""
    report_lookup = {r.get("report_id", ""): r for r in reports}
    rows = []
    for _, model_row in df_sorted.iterrows():
        rid = model_row["report_id"]
        r = report_lookup.get(rid)
        if not r:
            continue
        nli_map = _build_nli_to_canonical(r)
        for entry in r.get("per_class", []):
            rows.append({
                "display_name": model_row["display_name"],
                "rank": model_row["rank"],
                "label": _normalize_label(entry["label"], nli_map),
                "f1": entry["f1"],
                "precision": entry["precision"],
                "recall": entry["recall"],
                "support": entry["support"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisierungen
# ---------------------------------------------------------------------------

def _style_setup():
    """Globales Styling."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "grid.color": "#e0e0e0",
        "grid.alpha": 0.7,
    })


def plot_f1_macro_ranking(df: pd.DataFrame, ax: plt.Axes):
    """Horizontales Balkendiagramm: F1 Macro aller Modelle (höchstes oben)."""
    df_plot = df.iloc[::-1]

    colors = sns.color_palette("viridis", len(df_plot))[::-1]
    bars = ax.barh(
        df_plot["display_name"], df_plot["f1_macro"],
        color=colors, edgecolor="white", height=0.6,
    )

    for bar, val in zip(bars, df_plot["f1_macro"]):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("F1 Macro", fontsize=12)
    ax.set_title("Modell-Ranking nach F1 Macro", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, min(1.0, df["f1_macro"].max() + 0.08))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def plot_aggregate_heatmap(df: pd.DataFrame, ax: plt.Axes):
    """Heatmap: Aggregate-Metriken pro Modell (Modelle sortiert nach F1 Macro)."""
    metric_cols = ["f1_macro", "f1_weighted", "precision_macro", "recall_macro", "accuracy"]
    metric_labels = ["F1 Macro", "F1 Weighted", "Precision\nMacro", "Recall\nMacro", "Accuracy"]

    heat_data = df.set_index("display_name")[metric_cols].copy()
    heat_data.columns = metric_labels

    sns.heatmap(
        heat_data, annot=True, fmt=".4f", cmap="YlOrRd", linewidths=0.8,
        vmin=0, vmax=1, ax=ax,
        cbar_kws={"label": "Score", "shrink": 0.8},
        annot_kws={"fontsize": 10},
    )
    ax.set_title("Aggregate Metriken (Modelle sortiert nach F1 Macro)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def plot_per_class_heatmap(pc_df: pd.DataFrame, df_sorted: pd.DataFrame, ax: plt.Axes):
    """Heatmap: Per-Class F1 für alle Modelle (Modelle sortiert nach F1 Macro)."""
    if pc_df.empty:
        ax.text(0.5, 0.5, "Keine Per-Class Daten vorhanden", ha="center", va="center", fontsize=14)
        return

    pivot = pc_df.pivot_table(index="display_name", columns="label", values="f1", aggfunc="first")

    # Kanonische Spaltenreihenfolge
    ordered_cols = [l for l in CANONICAL_LABELS if l in pivot.columns]
    extra_cols = [c for c in pivot.columns if c not in CANONICAL_LABELS]
    pivot = pivot[ordered_cols + extra_cols]

    # Zeilenreihenfolge nach Ranking
    order = df_sorted["display_name"].tolist()
    pivot = pivot.reindex(order)

    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlGn", linewidths=0.5,
        vmin=0, vmax=1, ax=ax,
        cbar_kws={"label": "F1 Score", "shrink": 0.8},
        annot_kws={"fontsize": 9},
    )
    ax.set_title("Per-Class F1 Score (Modelle sortiert nach F1 Macro)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="y", rotation=0)


def plot_leaderboard_table(df: pd.DataFrame, ax: plt.Axes):
    """Tabellarische Übersicht aller Modelle."""
    ax.axis("off")

    headers = ["#", "Modell", "Typ", "F1 Macro", "F1 Wtd", "Prec", "Recall", "Acc", "Artikel", "GPU"]
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row["rank"],
            row["model_name"],
            row["model_type"],
            f"{row['f1_macro']:.4f}",
            f"{row['f1_weighted']:.4f}",
            f"{row['precision_macro']:.4f}",
            f"{row['recall_macro']:.4f}",
            f"{row['accuracy']:.4f}",
            row["n_articles"],
            row["gpu"],
        ])

    table = ax.table(
        cellText=table_data, colLabels=headers,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(len(table_data)):
        for j in range(len(headers)):
            cell = table[i + 1, j]
            if i == 0:
                cell.set_facecolor("#d5f5e3")
            elif i % 2 == 0:
                cell.set_facecolor("#f8f9fa")
            else:
                cell.set_facecolor("white")

    ax.set_title("Leaderboard – Alle Modelle", fontsize=14, fontweight="bold", pad=20)


def plot_per_class_f1_grouped(pc_df: pd.DataFrame, df_sorted: pd.DataFrame, ax: plt.Axes):
    """Gruppiertes Balkendiagramm: Per-Class F1 für jedes Modell."""
    if pc_df.empty:
        return

    # Kanonische Label-Reihenfolge verwenden
    all_labels = pc_df["label"].unique()
    labels = [l for l in CANONICAL_LABELS if l in all_labels]
    extra = [l for l in all_labels if l not in CANONICAL_LABELS]
    labels = labels + extra

    models = df_sorted["display_name"].tolist()
    n_models = len(models)

    x = np.arange(len(labels))
    width = 0.8 / n_models

    colors = sns.color_palette("tab10", n_models)
    for i, model in enumerate(models):
        model_data = pc_df[pc_df["display_name"] == model].drop_duplicates("label").set_index("label")
        vals = [float(model_data.loc[lbl, "f1"]) if lbl in model_data.index else 0.0 for lbl in labels]
        ax.bar(
            x + i * width - (n_models - 1) * width / 2, vals, width,
            label=model, color=colors[i], edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1: Modellvergleich", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=min(n_models, 3))


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualisierung aller Performance-Reports")
    parser.add_argument(
        "--reports-dir", type=str, default=str(_DEFAULT_REPORTS_DIR),
        help=f"Pfad zum Verzeichnis mit JSON-Reports (Standard: {_DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="PNGs nur speichern, nicht automatisch öffnen",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Kein Download von Google Drive, nur lokale Reports verwenden",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)

    # --- Schritt 1: Immer frisch von Google Drive laden ---
    if not args.no_download:
        sync_reports_from_drive(reports_dir)

    # --- Schritt 2: Reports einlesen ---
    reports = load_all_reports(reports_dir)
    df = reports_to_dataframe(reports)
    pc_df = get_per_class_data(reports, df)

    # --- Konsolen-Ranking ---
    print(f"\n{'='*60}")
    print("  MODELL-RANKING (sortiert nach F1 Macro)")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        marker = " ★" if row["rank"] == 1 else ""
        print(f"  #{row['rank']:>2}  {row['f1_macro']:.4f}  {row['display_name']}{marker}")
    print(f"{'='*60}\n")

    # --- Schritt 3: Visualisierung ---
    _style_setup()

    n_models = len(df)
    fig_height_ranking = max(4, n_models * 0.7 + 2)
    fig_height_table = max(3, n_models * 0.6 + 2)

    # Figure 1: Ranking + Leaderboard
    fig1, (ax_rank, ax_table) = plt.subplots(
        2, 1, figsize=(14, fig_height_ranking + fig_height_table),
        gridspec_kw={"height_ratios": [fig_height_ranking, fig_height_table]},
    )
    plot_f1_macro_ranking(df, ax_rank)
    plot_leaderboard_table(df, ax_table)
    fig1.suptitle("Performance-Übersicht aller Modelle", fontsize=16, fontweight="bold", y=1.01)
    fig1.tight_layout()

    # Figure 2: Aggregate Metriken (Heatmap)
    fig2, ax_agg = plt.subplots(figsize=(8, max(4, n_models * 0.55 + 1)))
    plot_aggregate_heatmap(df, ax_agg)
    fig2.tight_layout()

    # Figure 3: Per-Class F1 Heatmap
    if not pc_df.empty:
        n_labels = pc_df["label"].nunique()
        fig3, ax_heat = plt.subplots(figsize=(max(12, n_labels * 0.95), max(5, n_models * 0.65 + 1)))
        plot_per_class_heatmap(pc_df, df, ax_heat)
        fig3.tight_layout()
    else:
        fig3 = None

    # Figure 4: Per-Class F1 Gruppiert
    if not pc_df.empty:
        fig4, ax_pcf1 = plt.subplots(figsize=(max(14, n_labels * 1.1), 7))
        plot_per_class_f1_grouped(pc_df, df, ax_pcf1)
        fig4.tight_layout()
    else:
        fig4 = None

    # --- Immer als PNG speichern ---
    out_dir = reports_dir / "visualizations"
    out_dir.mkdir(exist_ok=True)

    saved_files = []
    for fig, name in [
        (fig1, "01_ranking_leaderboard.png"),
        (fig2, "02_aggregate_metrics.png"),
        (fig3, "03_per_class_heatmap.png"),
        (fig4, "04_per_class_f1_grouped.png"),
    ]:
        if fig is not None:
            path = out_dir / name
            fig.savefig(path, dpi=150, bbox_inches="tight")
            saved_files.append(path)

    print(f"\n{len(saved_files)} Plots gespeichert in: {out_dir}")
    for p in saved_files:
        print(f"  → {p.name}")

    # --- Automatisch im Standard-Viewer öffnen ---
    if not args.no_open:
        for p in saved_files:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", str(p)])
            elif platform.system() == "Linux":
                subprocess.Popen(["xdg-open", str(p)])
            elif platform.system() == "Windows":
                subprocess.Popen(["start", str(p)], shell=True)


if __name__ == "__main__":
    main()
