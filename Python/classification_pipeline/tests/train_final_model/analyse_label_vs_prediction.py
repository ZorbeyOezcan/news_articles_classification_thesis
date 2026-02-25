#!/usr/bin/env python3
"""
Analyse: Label vs. Prediction fuer train/test-Artikel.

Laedt die CSV aus classify_all_articles.ipynb und vergleicht die
Original-Labels mit den Modell-Predictions fuer Artikel aus train/test-Splits.

Usage:
    python analyse_label_vs_prediction.py                              # Default-Pfad
    python analyse_label_vs_prediction.py path/to/all_articles_classified.csv
    python analyse_label_vs_prediction.py --no-open                    # PNGs speichern, nicht oeffnen
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR.parent / "performance_reports" / "all_articles_classified.csv"
OUTPUT_DIR = SCRIPT_DIR / "label_vs_prediction_analysis"

ALL_LABELS = [
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

# Kurze Label-Namen fuer Plots
SHORT_LABELS = {
    "Klima / Energie": "Klima/Energie",
    "Zuwanderung": "Zuwanderung",
    "Renten": "Renten",
    "Soziales Gefälle": "Soz. Gefälle",
    "AfD/Rechte": "AfD/Rechte",
    "Arbeitslosigkeit": "Arbeitslos.",
    "Wirtschaftslage": "Wirtschaft",
    "Politikverdruss": "Politikverd.",
    "Gesundheitswesen, Pflege": "Gesundheit",
    "Kosten/Löhne/Preise": "Kosten/Löhne",
    "Ukraine/Krieg/Russland": "Ukraine/Krieg",
    "Bundeswehr/Verteidigung": "Bundeswehr",
    "Andere": "Andere",
}

COLORS = {
    "train": "#2196F3",
    "test": "#FF9800",
    "combined": "#4CAF50",
    "correct": "#4CAF50",
    "incorrect": "#F44336",
}


# ---------------------------------------------------------------------------
# Daten laden & vorbereiten
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    """CSV laden und auf gelabelte Artikel (train+test) filtern."""
    print(f"Lade CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Gesamt: {len(df):,} Zeilen, {df.shape[1]} Spalten")

    # Nur train+test mit vorhandenem Label
    labeled = df[df["split"].isin(["train", "test"])].copy()
    labeled = labeled[labeled["label"].notna() & (labeled["label"] != "")]
    labeled = labeled.reset_index(drop=True)

    print(f"  Gelabelt (train+test): {len(labeled):,}")
    print(f"    train: {(labeled['split'] == 'train').sum():,}")
    print(f"    test:  {(labeled['split'] == 'test').sum():,}")

    # Korrekt-Flag
    labeled["correct"] = labeled["label"] == labeled["prediction"]

    return labeled


# ---------------------------------------------------------------------------
# Metriken berechnen
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, split_name: str = "all") -> dict:
    """Berechne Accuracy, F1, Precision, Recall fuer einen DataFrame."""
    y_true = df["label"]
    y_pred = df["prediction"]

    present_labels = sorted(set(y_true) | set(y_pred))
    labels = [l for l in ALL_LABELS if l in present_labels]

    return {
        "split": split_name,
        "n": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "labels": labels,
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=labels, zero_division=0,
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def per_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Berechne pro Klasse: Accuracy, F1, Support — getrennt nach train/test."""
    rows = []
    for label in ALL_LABELS:
        for split_name in ["train", "test", "combined"]:
            if split_name == "combined":
                sub = df[df["label"] == label]
            else:
                sub = df[(df["label"] == label) & (df["split"] == split_name)]

            if len(sub) == 0:
                continue

            correct = (sub["label"] == sub["prediction"]).sum()
            rows.append({
                "label": label,
                "short_label": SHORT_LABELS.get(label, label),
                "split": split_name,
                "support": len(sub),
                "correct": correct,
                "accuracy": correct / len(sub),
                "mean_confidence": sub["prediction_score"].mean(),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Textuelle Zusammenfassung
# ---------------------------------------------------------------------------

def print_summary(labeled: pd.DataFrame, metrics_all: dict,
                  metrics_train: dict, metrics_test: dict) -> str:
    """Gibt eine textuelle Zusammenfassung aus und gibt sie als String zurueck."""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 70)
    p("  ANALYSE: LABEL vs. PREDICTION (train + test)")
    p("=" * 70)

    # 1. Uebersicht
    p(f"\n1. Uebersicht")
    p(f"   Gelabelte Artikel gesamt:  {len(labeled):,}")
    p(f"   Davon korrekt klassifiz.:  {labeled['correct'].sum():,} "
      f"({labeled['correct'].mean()*100:.1f}%)")
    p(f"   Davon falsch klassifiz.:   {(~labeled['correct']).sum():,} "
      f"({(~labeled['correct']).mean()*100:.1f}%)")

    # 2. Metriken pro Split
    p(f"\n2. Metriken pro Split")
    p(f"   {'Split':<10} {'N':>6} {'Acc':>8} {'F1-Mac':>8} {'F1-Wt':>8} {'Prec':>8} {'Rec':>8}")
    p(f"   {'-'*58}")
    for m in [metrics_train, metrics_test, metrics_all]:
        p(f"   {m['split']:<10} {m['n']:>6,} {m['accuracy']:>8.4f} "
          f"{m['f1_macro']:>8.4f} {m['f1_weighted']:>8.4f} "
          f"{m['precision_macro']:>8.4f} {m['recall_macro']:>8.4f}")

    # 3. Per-Class Report (test)
    p(f"\n3. Classification Report — TEST Split (n={metrics_test['n']:,})")
    p(metrics_test["report"])

    # 4. Per-Class Report (train)
    p(f"4. Classification Report — TRAIN Split (n={metrics_train['n']:,})")
    p(metrics_train["report"])

    # 5. Haeufigste Fehler
    errors = labeled[~labeled["correct"]].copy()
    if len(errors) > 0:
        p(f"5. Haeufigste Fehlklassifikationen (Top 15)")
        error_pairs = (
            errors.groupby(["label", "prediction"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(15)
        )
        p(f"   {'Label (true)':<30} {'Prediction':<30} {'N':>5}")
        p(f"   {'-'*67}")
        for _, row in error_pairs.iterrows():
            p(f"   {row['label']:<30} {row['prediction']:<30} {row['count']:>5}")

    # 6. Confidence korrekt vs. falsch
    p(f"\n6. Confidence-Statistik")
    for label_val, name in [(True, "Korrekt"), (False, "Falsch")]:
        sub = labeled[labeled["correct"] == label_val]["prediction_score"]
        if len(sub) > 0:
            p(f"   {name:<10} n={len(sub):>5,}  mean={sub.mean():.4f}  "
              f"median={sub.median():.4f}  std={sub.std():.4f}  "
              f"min={sub.min():.4f}")

    p(f"\n{'='*70}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualisierungen
# ---------------------------------------------------------------------------

def plot_confusion_matrix(df: pd.DataFrame, split_name: str,
                          output_dir: Path, labels: list) -> Path:
    """Confusion Matrix als Heatmap."""
    short = [SHORT_LABELS.get(l, l) for l in labels]

    y_true = df["label"]
    y_pred = df["prediction"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalisiert (Zeilen-%)
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums * 100

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(f"Confusion Matrix — {split_name} (n={len(df):,})",
                 fontsize=16, fontweight="bold", y=1.02)

    # Absolute Werte
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=short,
                yticklabels=short, ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("Absolute Counts", fontsize=13)
    axes[0].set_ylabel("True Label", fontsize=12)
    axes[0].set_xlabel("Prediction", fontsize=12)
    axes[0].tick_params(axis="both", labelsize=9)

    # Normalisiert (%)
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Oranges", xticklabels=short,
                yticklabels=short, ax=axes[1], vmin=0, vmax=100,
                cbar_kws={"shrink": 0.8, "format": mticker.PercentFormatter()})
    axes[1].set_title("Row-Normalized (%)", fontsize=13)
    axes[1].set_ylabel("True Label", fontsize=12)
    axes[1].set_xlabel("Prediction", fontsize=12)
    axes[1].tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    path = output_dir / f"01_confusion_matrix_{split_name.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gespeichert: {path.name}")
    return path


def plot_per_class_accuracy(pc_df: pd.DataFrame, output_dir: Path) -> Path:
    """Per-Class Accuracy: train vs. test als gruppierte Balken."""
    plot_df = pc_df[pc_df["split"].isin(["train", "test"])].copy()
    plot_df = plot_df.sort_values("accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    labels_order = [l for l in ALL_LABELS if l in plot_df["label"].values]
    short_order = [SHORT_LABELS.get(l, l) for l in labels_order]

    x = np.arange(len(labels_order))
    width = 0.35

    train_vals = []
    test_vals = []
    for label in labels_order:
        t = plot_df[(plot_df["label"] == label) & (plot_df["split"] == "train")]
        train_vals.append(t["accuracy"].values[0] if len(t) > 0 else 0)
        t = plot_df[(plot_df["label"] == label) & (plot_df["split"] == "test")]
        test_vals.append(t["accuracy"].values[0] if len(t) > 0 else 0)

    bars_train = ax.barh(x + width/2, train_vals, width, label="Train",
                         color=COLORS["train"], alpha=0.85)
    bars_test = ax.barh(x - width/2, test_vals, width, label="Test",
                        color=COLORS["test"], alpha=0.85)

    # Werte anzeigen
    for bars in [bars_train, bars_test]:
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{w:.1%}", va="center", fontsize=8)

    ax.set_yticks(x)
    ax.set_yticklabels(short_order, fontsize=10)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy: Train vs. Test", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 1.12)
    ax.axvline(x=1.0, color="grey", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = output_dir / "02_per_class_accuracy_train_vs_test.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gespeichert: {path.name}")
    return path


def plot_confidence_distribution(labeled: pd.DataFrame, output_dir: Path) -> Path:
    """Confidence-Verteilung: korrekt vs. falsch, getrennt nach Split."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confidence-Verteilung: Korrekte vs. Falsche Predictions",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, (split_name, title) in enumerate([
        ("train", "Train"), ("test", "Test"), (None, "Combined")
    ]):
        ax = axes[i]
        if split_name:
            sub = labeled[labeled["split"] == split_name]
        else:
            sub = labeled

        correct = sub[sub["correct"]]["prediction_score"]
        incorrect = sub[~sub["correct"]]["prediction_score"]

        bins = np.linspace(0, 1, 30)
        ax.hist(correct, bins=bins, alpha=0.7, label=f"Korrekt (n={len(correct):,})",
                color=COLORS["correct"], density=True)
        ax.hist(incorrect, bins=bins, alpha=0.7, label=f"Falsch (n={len(incorrect):,})",
                color=COLORS["incorrect"], density=True)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Prediction Score", fontsize=11)
        ax.set_ylabel("Dichte", fontsize=11)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = output_dir / "03_confidence_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gespeichert: {path.name}")
    return path


def plot_error_heatmap(labeled: pd.DataFrame, output_dir: Path) -> Path:
    """Heatmap: Wohin werden falsche Predictions gesendet?"""
    errors = labeled[~labeled["correct"]].copy()
    if len(errors) == 0:
        print("  Keine Fehler — Error-Heatmap uebersprungen.")
        return None

    present_labels = sorted(set(errors["label"]) | set(errors["prediction"]))
    labels = [l for l in ALL_LABELS if l in present_labels]
    short = [SHORT_LABELS.get(l, l) for l in labels]

    cm = confusion_matrix(errors["label"], errors["prediction"], labels=labels)
    # Diagonale auf 0 setzen (nur Fehler zeigen)
    np.fill_diagonal(cm, 0)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=short,
                yticklabels=short, ax=ax, cbar_kws={"shrink": 0.8},
                linewidths=0.5, linecolor="white")
    ax.set_title(f"Fehlklassifikationen: True Label → Falsche Prediction (n={len(errors):,})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Falsche Prediction", fontsize=12)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    path = output_dir / "04_error_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gespeichert: {path.name}")
    return path


def plot_train_test_gap(pc_df: pd.DataFrame, output_dir: Path) -> Path:
    """Differenz Train-Accuracy minus Test-Accuracy pro Klasse (Overfitting-Indikator)."""
    train_acc = pc_df[pc_df["split"] == "train"].set_index("label")["accuracy"]
    test_acc = pc_df[pc_df["split"] == "test"].set_index("label")["accuracy"]

    common_labels = [l for l in ALL_LABELS if l in train_acc.index and l in test_acc.index]
    if not common_labels:
        print("  Keine gemeinsamen Labels — Train-Test-Gap uebersprungen.")
        return None

    gaps = pd.DataFrame({
        "label": common_labels,
        "short_label": [SHORT_LABELS.get(l, l) for l in common_labels],
        "train_acc": [train_acc[l] for l in common_labels],
        "test_acc": [test_acc[l] for l in common_labels],
        "gap": [train_acc[l] - test_acc[l] for l in common_labels],
    }).sort_values("gap", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ["#F44336" if g > 0 else "#4CAF50" for g in gaps["gap"]]
    bars = ax.barh(gaps["short_label"], gaps["gap"] * 100, color=colors, alpha=0.85)

    for bar, gap_val in zip(bars, gaps["gap"]):
        w = bar.get_width()
        ax.text(w + (0.3 if w >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
                f"{gap_val:+.1%}", va="center", fontsize=9,
                ha="left" if w >= 0 else "right")

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Train Accuracy − Test Accuracy (Prozentpunkte)", fontsize=11)
    ax.set_title("Overfitting-Indikator: Train−Test Accuracy Gap pro Klasse",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)

    # Legende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F44336", alpha=0.85, label="Overfitting (Train > Test)"),
        Patch(facecolor="#4CAF50", alpha=0.85, label="Test besser (Test > Train)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    path = output_dir / "05_train_test_accuracy_gap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gespeichert: {path.name}")
    return path


def plot_confidence_per_class(labeled: pd.DataFrame, output_dir: Path) -> Path:
    """Boxplot der Confidence pro Klasse, korrekt vs. falsch."""
    present_labels = [l for l in ALL_LABELS if l in labeled["label"].values]
    short_map = {l: SHORT_LABELS.get(l, l) for l in present_labels}
    labeled_copy = labeled.copy()
    labeled_copy["short_label"] = labeled_copy["label"].map(short_map)
    labeled_copy["Ergebnis"] = labeled_copy["correct"].map(
        {True: "Korrekt", False: "Falsch"}
    )

    order = [SHORT_LABELS.get(l, l) for l in present_labels]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=labeled_copy, x="short_label", y="prediction_score",
        hue="Ergebnis", order=order, ax=ax,
        palette={"Korrekt": COLORS["correct"], "Falsch": COLORS["incorrect"]},
        fliersize=2, linewidth=0.8,
    )
    ax.set_xlabel("True Label", fontsize=12)
    ax.set_ylabel("Prediction Score (Confidence)", fontsize=12)
    ax.set_title("Confidence pro Klasse: Korrekte vs. Falsche Predictions",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = output_dir / "06_confidence_per_class_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gespeichert: {path.name}")
    return path


def plot_prediction_distribution(labeled: pd.DataFrame, output_dir: Path) -> Path:
    """Vergleich: Label-Verteilung vs. Prediction-Verteilung."""
    present_labels = [l for l in ALL_LABELS if l in labeled["label"].values
                      or l in labeled["prediction"].values]
    short = [SHORT_LABELS.get(l, l) for l in present_labels]

    label_counts = labeled["label"].value_counts()
    pred_counts = labeled["prediction"].value_counts()

    label_vals = [label_counts.get(l, 0) for l in present_labels]
    pred_vals = [pred_counts.get(l, 0) for l in present_labels]

    x = np.arange(len(present_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, label_vals, width, label="True Label", color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, pred_vals, width, label="Prediction", color="#FF9800", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Anzahl Artikel", fontsize=12)
    ax.set_title("Verteilung: True Labels vs. Predictions (train+test)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = output_dir / "07_label_vs_prediction_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gespeichert: {path.name}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse: Label vs. Prediction fuer train/test-Artikel."
    )
    parser.add_argument(
        "csv_path", nargs="?", type=Path, default=DEFAULT_CSV,
        help=f"Pfad zur CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="PNGs nicht automatisch oeffnen.",
    )
    args = parser.parse_args()

    csv_path = args.csv_path.resolve()
    if not csv_path.exists():
        print(f"FEHLER: CSV nicht gefunden: {csv_path}")
        print(f"\nBitte zuerst classify_all_articles.ipynb ausfuehren,")
        print(f"dann die CSV hierhin kopieren:")
        print(f"  {DEFAULT_CSV}")
        sys.exit(1)

    # Daten laden
    labeled = load_data(csv_path)
    if len(labeled) == 0:
        print("FEHLER: Keine gelabelten Artikel gefunden.")
        sys.exit(1)

    # Output-Verzeichnis
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {OUTPUT_DIR}")

    # Metriken
    train_df = labeled[labeled["split"] == "train"]
    test_df = labeled[labeled["split"] == "test"]

    metrics_all = compute_metrics(labeled, "combined")
    metrics_train = compute_metrics(train_df, "train") if len(train_df) > 0 else None
    metrics_test = compute_metrics(test_df, "test") if len(test_df) > 0 else None

    # Per-Class
    pc_df = per_class_metrics(labeled)

    # Textuelle Zusammenfassung
    summary_text = print_summary(labeled, metrics_all, metrics_train, metrics_test)

    # Summary speichern
    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n  Gespeichert: summary.txt")

    # Visualisierungen
    print(f"\nErstelle Visualisierungen...")
    png_paths = []

    # 1. Confusion Matrices
    labels = metrics_all["labels"]
    png_paths.append(plot_confusion_matrix(labeled, "Combined", OUTPUT_DIR, labels))
    if metrics_train:
        png_paths.append(plot_confusion_matrix(train_df, "Train", OUTPUT_DIR, labels))
    if metrics_test:
        png_paths.append(plot_confusion_matrix(test_df, "Test", OUTPUT_DIR, labels))

    # 2. Per-Class Accuracy train vs test
    png_paths.append(plot_per_class_accuracy(pc_df, OUTPUT_DIR))

    # 3. Confidence-Verteilung
    png_paths.append(plot_confidence_distribution(labeled, OUTPUT_DIR))

    # 4. Error Heatmap
    p = plot_error_heatmap(labeled, OUTPUT_DIR)
    if p:
        png_paths.append(p)

    # 5. Train-Test Accuracy Gap
    p = plot_train_test_gap(pc_df, OUTPUT_DIR)
    if p:
        png_paths.append(p)

    # 6. Confidence per Class
    png_paths.append(plot_confidence_per_class(labeled, OUTPUT_DIR))

    # 7. Label vs Prediction distribution
    png_paths.append(plot_prediction_distribution(labeled, OUTPUT_DIR))

    # Per-Class CSV speichern
    pc_csv = OUTPUT_DIR / "per_class_metrics.csv"
    pc_df.to_csv(pc_csv, index=False, encoding="utf-8")
    print(f"  Gespeichert: per_class_metrics.csv")

    print(f"\n{'='*70}")
    print(f"  FERTIG — {len(png_paths)} Visualisierungen + Summary + CSV")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")

    # PNGs oeffnen (macOS)
    if not args.no_open and platform.system() == "Darwin":
        valid_paths = [p for p in png_paths if p is not None]
        if valid_paths:
            subprocess.run(["open"] + [str(p) for p in valid_paths])


if __name__ == "__main__":
    main()
