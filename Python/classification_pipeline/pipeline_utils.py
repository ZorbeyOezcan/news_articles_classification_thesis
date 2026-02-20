"""
pipeline_utils.py — Shared utilities for the German news classification pipeline.

All classify_*.ipynb notebooks import from this module.
Provides: configuration, data access, evaluation, reporting, timing, and visualization.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
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
# CONSTANTS (defaults — can be overridden per notebook)
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATE_LABELS: List[str] = [
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

DEFAULT_HYPOTHESIS_TEMPLATE: str = "Dieser Text handelt von {}."

DATASET_ID: str = "Zorryy/news_articles_2025_elections_germany"

DEFAULT_LABEL_MAPPING: Dict[str, str] = {
    "Klima / Energie": "Klima / Energie",
    "Zuwanderung": "Zuwanderung",
    "Renten": "Renten",
    "Soziales Gefälle": "Soziales Gefälle",
    "AfD/Rechte": "AfD/Rechte",
    "Arbeitslosigkeit": "Arbeitslosigkeit",
    "Wirtschaftslage": "Wirtschaftslage",
    "Politikverdruss": "Politikverdruss",
    "Gesundheitswesen, Pflege": "Gesundheitswesen, Pflege",
    "Kosten/Löhne/Preise": "Kosten/Löhne/Preise",
    "Ukraine/Krieg/Russland": "Ukraine/Krieg/Russland",
    "Bundeswehr/Verteidigung": "Bundeswehr/Verteidigung",
    "Andere": "Andere",
}

_DRIVE_REPORTS = Path("/content/drive/MyDrive/thesis_reports/performance_reports")
_LOCAL_REPORTS = Path(__file__).parent / "performance_reports"

# Google Drive if mounted (persistent), otherwise local (lost after session)
REPORTS_DIR: Path = _DRIVE_REPORTS if _DRIVE_REPORTS.parent.exists() else _LOCAL_REPORTS


# ---------------------------------------------------------------------------
# DATA SHARING (module-level cache)
# ---------------------------------------------------------------------------

_runtime_data: Dict[str, Any] = {}


def set_runtime_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_df: Optional[pd.DataFrame],
    split_config: Dict[str, Any],
    label_mapping: Dict[str, str],
) -> None:
    """Store dataset splits in module-level cache for cross-notebook access.

    Called by init_data.ipynb. All classify notebooks retrieve via get_runtime_data().
    """
    _runtime_data["train"] = train_df
    _runtime_data["eval"] = eval_df
    _runtime_data["test"] = test_df
    _runtime_data["raw"] = raw_df
    _runtime_data["split_config"] = split_config
    _runtime_data["label_mapping"] = label_mapping
    print(
        f"Runtime data set: train={len(train_df)}, eval={len(eval_df)}, "
        f"test={len(test_df)}, raw={len(raw_df) if raw_df is not None else 0}"
    )


def get_runtime_data() -> Dict[str, Any]:
    """Retrieve dataset splits from module-level cache.

    Returns dict with keys: 'train', 'eval', 'test', 'raw',
    'split_config', 'label_mapping'.
    """
    if not _runtime_data:
        raise RuntimeError(
            "No runtime data loaded. Run init_data.ipynb first in this session."
        )
    return _runtime_data


def get_train_df() -> pd.DataFrame:
    return get_runtime_data()["train"]


def get_eval_df() -> pd.DataFrame:
    return get_runtime_data()["eval"]


def get_test_df() -> pd.DataFrame:
    return get_runtime_data()["test"]


def get_raw_df() -> Optional[pd.DataFrame]:
    return get_runtime_data()["raw"]


# ---------------------------------------------------------------------------
# DATA LOADING (self-contained — no init_data.ipynb needed)
# ---------------------------------------------------------------------------


def load_data(
    split_mode: str = "percentage",
    eval_fraction: float = 0.2,
    eval_per_class: int = 10,
    random_seed: int = 42,
    load_raw: bool = False,
    label_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Load dataset from HuggingFace, apply label mapping, create train/eval/test splits.

    Each notebook calls this once — no shared runtime state needed.

    Args:
        split_mode: "percentage" (eval_fraction of train) or "absolute" (eval_per_class per label).
        eval_fraction: Fraction of train data for eval (only if split_mode="percentage").
        eval_per_class: Exact number of eval samples per class (only if split_mode="absolute").
        random_seed: Seed for reproducible splits.
        load_raw: If True, also load the ~259k unlabeled raw split.
        label_mapping: Dict mapping original labels to new names. None = DEFAULT_LABEL_MAPPING.

    Returns:
        Dict with keys: 'train', 'eval', 'test', 'raw', 'split_config', 'label_mapping'
    """
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    if label_mapping is None:
        label_mapping = dict(DEFAULT_LABEL_MAPPING)

    # --- Load from HuggingFace ---
    ds = load_dataset(DATASET_ID)

    test_df = ds["test"].to_pandas()
    train_full_df = ds["train"].to_pandas()

    raw_df = None
    if load_raw:
        raw_df = ds["raw"].to_pandas()
        print(f"Raw:   {len(raw_df):>7} Artikel (unlabeled)")
    else:
        print("Raw-Daten nicht geladen (load_raw=False)")

    print(f"Test:  {len(test_df):>7} Artikel (FROZEN)")
    print(f"Train: {len(train_full_df):>7} Artikel (wird in Train + Eval aufgeteilt)")

    # --- Apply label mapping ---
    def _apply_mapping(df: pd.DataFrame) -> pd.DataFrame:
        if "label" not in df.columns:
            return df
        original_labels = set(df["label"].unique())
        missing = original_labels - set(label_mapping.keys())
        if missing:
            print(f"  WARNUNG: Labels ohne Mapping: {missing}")
        df = df.copy()
        df["label"] = df["label"].map(label_mapping).fillna(df["label"])
        return df

    test_df = _apply_mapping(test_df)
    train_full_df = _apply_mapping(train_full_df)
    if raw_df is not None and "label" in raw_df.columns:
        raw_df = _apply_mapping(raw_df)

    remapped = {k: v for k, v in label_mapping.items() if k != v}
    if remapped:
        print("Label-Mapping Änderungen:")
        for orig, new in remapped.items():
            print(f"  {orig} → {new}")

    # --- Train / Eval split ---
    if split_mode == "percentage":
        class_counts = train_full_df["label"].value_counts()
        small_classes = class_counts[class_counts < 2].index.tolist()

        if small_classes:
            print(f"Klassen mit <2 Artikeln (komplett in Train): {small_classes}")
            mask_small = train_full_df["label"].isin(small_classes)
            splittable_df = train_full_df[~mask_small]
            small_df = train_full_df[mask_small]

            train_df, eval_df = train_test_split(
                splittable_df,
                test_size=eval_fraction,
                stratify=splittable_df["label"],
                random_state=random_seed,
            )
            train_df = pd.concat([train_df, small_df])
        else:
            train_df, eval_df = train_test_split(
                train_full_df,
                test_size=eval_fraction,
                stratify=train_full_df["label"],
                random_state=random_seed,
            )

    elif split_mode == "absolute":
        eval_parts, train_parts = [], []
        for label in train_full_df["label"].unique():
            class_df = train_full_df[train_full_df["label"] == label]
            n = min(len(class_df), eval_per_class)
            if n < 2:
                print(f"  {label}: nur {len(class_df)} Artikel -> komplett in Train")
                train_parts.append(class_df)
                continue
            eval_sample = class_df.sample(n=n, random_state=random_seed)
            eval_parts.append(eval_sample)
            train_parts.append(class_df.drop(eval_sample.index))
        eval_df = pd.concat(eval_parts).reset_index(drop=True)
        train_df = pd.concat(train_parts).reset_index(drop=True)

    else:
        raise ValueError(f"Ungültiger split_mode: {split_mode}. Nutze 'percentage' oder 'absolute'.")

    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    print(f"\nTrain: {len(train_df):>6} Artikel")
    print(f"Eval:  {len(eval_df):>6} Artikel")
    print(f"Test:  {len(test_df):>6} Artikel (FROZEN)")

    split_config = {
        "dataset_id": DATASET_ID,
        "split_mode": split_mode,
        "eval_fraction": eval_fraction if split_mode == "percentage" else None,
        "eval_per_class": eval_per_class if split_mode == "absolute" else None,
        "random_seed": random_seed,
        "load_raw": load_raw,
        "train_size": len(train_df),
        "eval_size": len(eval_df),
        "test_size": len(test_df),
        "raw_size": len(raw_df) if raw_df is not None else 0,
    }

    result = {
        "train": train_df,
        "eval": eval_df,
        "test": test_df,
        "raw": raw_df,
        "split_config": split_config,
        "label_mapping": label_mapping,
    }

    # Also populate runtime cache for backwards compatibility
    set_runtime_data(train_df, eval_df, test_df, raw_df, split_config, label_mapping)

    return result


# ---------------------------------------------------------------------------
# MODEL CONFIG EXTRACTION
# ---------------------------------------------------------------------------

# Fields to extract from HuggingFace model config.json
_MODEL_CONFIG_FIELDS = [
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "vocab_size",
    "max_position_embeddings",
]


def extract_model_config(classifier) -> Dict[str, Any]:
    """Extract key architecture fields from a HuggingFace pipeline's model config.

    Args:
        classifier: A transformers.Pipeline object (e.g. from ``pipeline("zero-shot-classification", ...)``).

    Returns:
        Dict with architecture fields (only those present in the model config).
    """
    config = classifier.model.config.to_dict()
    extracted = {}
    for field in _MODEL_CONFIG_FIELDS:
        if field in config:
            val = config[field]
            # architectures is a list, join for display
            if field == "architectures" and isinstance(val, list):
                val = val[0] if len(val) == 1 else ", ".join(val)
            extracted[field] = val
    return extracted


# ---------------------------------------------------------------------------
# EXPERIMENT TIMER
# ---------------------------------------------------------------------------


class ExperimentTimer:
    """Context manager for timing classification experiments.

    Usage:
        timer = ExperimentTimer()
        with timer:
            predictions = classify(...)
        print(timer.duration_formatted)
        print(timer.articles_per_second(617))
    """

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "ExperimentTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.time()

    @property
    def duration_seconds(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def duration_formatted(self) -> str:
        secs = self.duration_seconds
        if secs < 60:
            return f"{secs:.1f}s"
        mins = int(secs // 60)
        remaining = secs % 60
        return f"{mins}m {remaining:.0f}s"

    def articles_per_second(self, n_articles: int) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return n_articles / self.duration_seconds


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------


def evaluate(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    experiment_name: str = "Experiment",
) -> Dict[str, Any]:
    """Compute all classification metrics in a standardized way.

    Returns dict with f1_macro, f1_weighted, precision_macro, precision_weighted,
    recall_macro, recall_weighted, accuracy, per_class_df, confusion_matrix,
    confusion_matrix_normalized, classification_report_dict, labels.
    """
    if labels is None:
        labels = DEFAULT_CANDIDATE_LABELS

    f1_mac = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    prec_mac = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    prec_w = precision_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    rec_mac = recall_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    rec_w = recall_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    per_class = pd.DataFrame(
        {
            "Label": labels,
            "Precision": [report_dict[l]["precision"] for l in labels],
            "Recall": [report_dict[l]["recall"] for l in labels],
            "F1": [report_dict[l]["f1-score"] for l in labels],
            "Support": [int(report_dict[l]["support"]) for l in labels],
        }
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    )

    return {
        "experiment_name": experiment_name,
        "f1_macro": f1_mac,
        "f1_weighted": f1_w,
        "precision_macro": prec_mac,
        "precision_weighted": prec_w,
        "recall_macro": rec_mac,
        "recall_weighted": rec_w,
        "accuracy": acc,
        "per_class_df": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "classification_report_dict": report_dict,
        "labels": labels,
    }


def print_metrics(metrics: Dict[str, Any], experiment_name: str = "") -> None:
    """Pretty-print metrics to stdout."""
    name = experiment_name or metrics.get("experiment_name", "Experiment")
    print(f"{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"\n  F1 Macro:           {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted:        {metrics['f1_weighted']:.4f}")
    print(f"  Precision Macro:    {metrics['precision_macro']:.4f}")
    print(f"  Precision Weighted: {metrics['precision_weighted']:.4f}")
    print(f"  Recall Macro:       {metrics['recall_macro']:.4f}")
    print(f"  Recall Weighted:    {metrics['recall_weighted']:.4f}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"\n  Per-Class Metrics:")
    print(metrics["per_class_df"].to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (28, 11),
) -> plt.Figure:
    """Plot side-by-side raw and normalized confusion matrices."""
    cm = metrics["confusion_matrix"]
    cm_norm = metrics["confusion_matrix_normalized"]
    labels = metrics["labels"]
    short_labels = [l[:20] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"{title} (Counts)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_per_class_f1(
    metrics_list: List[Dict[str, Any]],
    experiment_names: List[str],
    title: str = "Per-Class F1 Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing per-class F1 across multiple experiments."""
    labels = metrics_list[0]["labels"]
    n_labels = len(labels)
    n_experiments = len(metrics_list)
    x = np.arange(n_labels)
    width = 0.8 / n_experiments
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (m, name) in enumerate(zip(metrics_list, experiment_names)):
        f1_values = m["per_class_df"]["F1"].values
        f1_mac = m["f1_macro"]
        offset = (i - n_experiments / 2 + 0.5) * width
        ax.bar(
            x + offset,
            f1_values,
            width,
            label=f"{name} (F1 macro={f1_mac:.3f})",
            color=colors[i % len(colors)],
            alpha=0.85,
        )

    ax.set_xlabel("Category")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:18] for l in labels], rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# GPU UTILITIES
# ---------------------------------------------------------------------------


def get_gpu_info() -> Dict[str, str]:
    """Return GPU name, VRAM, CUDA version as a dict."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "gpu_name": props.name,
                "gpu_vram_gb": f"{props.total_memory / 1e9:.1f}",
                "cuda_version": torch.version.cuda or "N/A",
            }
    except ImportError:
        pass
    return {"gpu_name": "N/A (CPU)", "gpu_vram_gb": "0", "cuda_version": "N/A"}


def check_gpu_available() -> bool:
    """Return True if CUDA GPU is available. Print info."""
    try:
        import torch

        available = torch.cuda.is_available()
        if available:
            info = get_gpu_info()
            print(f"GPU: {info['gpu_name']} ({info['gpu_vram_gb']} GB)")
        else:
            print("No GPU available — running on CPU.")
        return available
    except ImportError:
        print("PyTorch not installed — cannot check GPU.")
        return False


# ---------------------------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------------------------


def _next_report_number(date_str: str, model_name: str) -> int:
    """Find next sequential report number for a given model on a given date."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(
        rf"^{re.escape(date_str)}_{re.escape(model_name)}_report_(\d+)\."
    )
    max_num = 0
    for f in REPORTS_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def _estimate_cost_per_1000(
    timer: ExperimentTimer,
    n_articles: int,
    gpu_type: str = "T4",
) -> str:
    """Estimate Colab compute cost per 1000 articles.

    Approximate Colab Pro credit rates:
      T4:   ~1.41 compute units/hour
      L4:   ~1.71 compute units/hour
      A100: ~7.52 compute units/hour
      V100: ~3.3 compute units/hour
    1 compute unit ~ $0.10 USD (approximate).
    """
    rates = {"T4": 1.41, "A100": 7.52, "V100": 3.3, "L4": 1.71}
    cu_per_hour = rates.get(gpu_type, 1.5)

    if n_articles == 0 or timer.duration_seconds == 0:
        return "N/A"

    hours_per_1000 = (timer.duration_seconds / n_articles) * 1000 / 3600
    cost_usd = hours_per_1000 * cu_per_hour * 0.10
    return f"${cost_usd:.2f} ({gpu_type}, estimated)"


def _format_confusion_matrix_markdown(
    cm: np.ndarray,
    labels: List[str],
) -> str:
    """Render confusion matrix as a Markdown table."""
    short = [l[:15] for l in labels]
    header = "| |" + "|".join(f" {s} " for s in short) + "|"
    separator = "|---|" + "|".join("---" for _ in short) + "|"
    rows = []
    for i, label in enumerate(short):
        vals = "|".join(f" {cm[i, j]} " for j in range(len(short)))
        rows.append(f"| **{label}** |{vals}|")
    return "\n".join([header, separator] + rows)


def generate_report(
    model_name: str,
    model_type: Literal["zero-shot", "one-shot", "few-shot", "fine-tuned"],
    metrics: Dict[str, Any],
    timer: ExperimentTimer,
    model_info: Dict[str, Any],
    candidate_labels: Optional[List[str]] = None,
    hypothesis_template: Optional[str] = None,
    experiment_notes: str = "",
    training_params: Optional[Dict[str, Any]] = None,
    uploaded_model_url: Optional[str] = None,
    split_config: Optional[Dict[str, Any]] = None,
    label_mapping: Optional[Dict[str, str]] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a standardized Markdown report + JSON sidecar.

    Args:
        split_config: Dict from load_data()['split_config']. If None, falls back to runtime cache.
        label_mapping: Dict from load_data()['label_mapping']. If None, falls back to runtime cache.

    Returns absolute path to the saved .md report file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    date_str = now.strftime("%d%m%y")
    report_num = _next_report_number(date_str, model_name)
    base_name = f"{date_str}_{model_name}_report_{report_num:03d}"
    md_path = REPORTS_DIR / f"{base_name}.md"
    json_path = REPORTS_DIR / f"{base_name}.json"

    # Gather split info: prefer direct parameter, fall back to runtime cache
    sc = split_config
    if sc is None:
        try:
            sc = get_runtime_data().get("split_config", {})
        except RuntimeError:
            sc = {}

    split_info = {}
    if sc:
        split_info = {
            "n_train": sc.get("train_size", "N/A"),
            "n_eval": sc.get("eval_size", "N/A"),
            "n_test": sc.get("test_size", "N/A"),
            "n_raw": sc.get("raw_size", "N/A"),
            "n_total": (
                sc.get("train_size", 0)
                + sc.get("eval_size", 0)
                + sc.get("test_size", 0)
                + sc.get("raw_size", 0)
            ),
            "split_mode": sc.get("split_mode", "N/A"),
            "eval_fraction": sc.get("eval_fraction"),
            "eval_per_class": sc.get("eval_per_class"),
            "random_seed": sc.get("random_seed", "N/A"),
        }
    else:
        split_info = {"note": "Split info not available."}

    if label_mapping is None:
        try:
            label_mapping = get_runtime_data().get("label_mapping", {})
        except RuntimeError:
            label_mapping = {}

    used_labels = candidate_labels or metrics.get("labels", DEFAULT_CANDIDATE_LABELS)
    used_template = hypothesis_template or DEFAULT_HYPOTHESIS_TEMPLATE

    gpu_info = get_gpu_info()
    gpu_type = gpu_info["gpu_name"].split()[-1] if gpu_info["gpu_name"] != "N/A (CPU)" else "CPU"
    n_articles = int(metrics["per_class_df"]["Support"].sum())
    cost_est = _estimate_cost_per_1000(timer, n_articles, gpu_type)

    # --- Build Markdown ---
    lines = [
        f"# Performance Report: {model_name}",
        f"**Report ID:** {base_name}",
        f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Model Information",
        "| Property | Value |",
        "|---|---|",
        f"| Model Name | {model_name} |",
        f"| Model Type | {model_type} |",
        f"| HuggingFace ID | {model_info.get('huggingface_id', 'N/A')} |",
        f"| Language | {model_info.get('language', 'N/A')} |",
        f"| Max Tokens | {model_info.get('max_tokens', 'N/A')} |",
        f"| Parameters | {model_info.get('parameters', 'N/A')} |",
    ]

    if model_info.get("notes"):
        lines.append(f"| Notes | {model_info['notes']} |")
    if uploaded_model_url:
        lines.append(f"| Uploaded Model | {uploaded_model_url} |")

    # Model Architecture (from config.json)
    if model_config:
        _CONFIG_DISPLAY = {
            "architectures": "Architecture",
            "model_type": "Model Type",
            "hidden_size": "Hidden Size",
            "num_hidden_layers": "Num Layers",
            "num_attention_heads": "Attention Heads",
            "vocab_size": "Vocab Size",
            "max_position_embeddings": "Max Position Embeddings",
        }
        lines += [
            "",
            "## Model Architecture",
            "| Property | Value |",
            "|---|---|",
        ]
        for key, display_name in _CONFIG_DISPLAY.items():
            if key in model_config:
                lines.append(f"| {display_name} | {model_config[key]} |")

    # NLI / Classification Config
    lines += [
        "",
        "## Classification Config",
        "| Property | Value |",
        "|---|---|",
        f"| Hypothesis Template | `{used_template}` |",
        f"| Num Labels | {len(used_labels)} |",
    ]

    # Detaillierte Label-Tabelle mit NLI-Phrasen
    remapped = {k: v for k, v in label_mapping.items() if k != v}
    if remapped:
        lines += [
            "",
            "## Candidate Labels & NLI Phrases",
            "| # | Original | Candidate Label | NLI Phrase |",
            "|---|---|---|---|",
        ]
        reverse_map = {v: k for k, v in label_mapping.items()}
        for i, label in enumerate(used_labels, 1):
            orig = reverse_map.get(label, label)
            nli_phrase = used_template.format(label)
            if orig != label:
                lines.append(f"| {i} | {orig} | {label} | {nli_phrase} |")
            else:
                lines.append(f"| {i} | — | {label} | {nli_phrase} |")
    else:
        lines += [
            "",
            "## Candidate Labels & NLI Phrases",
            "| # | Candidate Label | NLI Phrase |",
            "|---|---|---|",
        ]
        for i, label in enumerate(used_labels, 1):
            nli_phrase = used_template.format(label)
            lines.append(f"| {i} | {label} | {nli_phrase} |")

    # Dataset info
    lines += [
        "",
        "## Dataset Information",
        "| Property | Value |",
        "|---|---|",
        f"| Dataset | {DATASET_ID} |",
        f"| Evaluated On | {metrics.get('experiment_name', 'N/A')} ({n_articles} articles) |",
        f"| N Train | {split_info.get('n_train', 'N/A')} |",
        f"| N Eval | {split_info.get('n_eval', 'N/A')} |",
        f"| N Test | {split_info.get('n_test', 'N/A')} |",
        f"| N Raw | {split_info.get('n_raw', 'N/A')} |",
        f"| N Total | {split_info.get('n_total', 'N/A')} |",
        f"| Split Mode | {split_info.get('split_mode', 'N/A')} |",
        f"| Random Seed | {split_info.get('random_seed', 'N/A')} |",
    ]

    # Runtime
    lines += [
        "",
        "## Runtime",
        "| Property | Value |",
        "|---|---|",
        f"| Duration | {timer.duration_formatted} |",
        f"| Articles/Second | {timer.articles_per_second(n_articles):.2f} |",
        f"| GPU | {gpu_info['gpu_name']} ({gpu_info['gpu_vram_gb']} GB) |",
        f"| CUDA | {gpu_info['cuda_version']} |",
        f"| Est. Cost / 1000 Articles | {cost_est} |",
    ]

    # Training params
    if training_params:
        lines += [
            "",
            "## Training Parameters",
            "| Parameter | Value |",
            "|---|---|",
        ]
        for k, v in training_params.items():
            lines.append(f"| {k} | {v} |")

    # Aggregate metrics
    lines += [
        "",
        "## Aggregate Metrics",
        "| Metric | Value |",
        "|---|---|",
        f"| **F1 Macro** | **{metrics['f1_macro']:.4f}** |",
        f"| F1 Weighted | {metrics['f1_weighted']:.4f} |",
        f"| Precision Macro | {metrics['precision_macro']:.4f} |",
        f"| Precision Weighted | {metrics['precision_weighted']:.4f} |",
        f"| Recall Macro | {metrics['recall_macro']:.4f} |",
        f"| Recall Weighted | {metrics['recall_weighted']:.4f} |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
    ]

    # Per-class metrics
    pc = metrics["per_class_df"]
    lines += [
        "",
        "## Per-Class Metrics",
        "| Label | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for _, row in pc.iterrows():
        lines.append(
            f"| {row['Label']} | {row['Precision']:.4f} | {row['Recall']:.4f} "
            f"| {row['F1']:.4f} | {int(row['Support'])} |"
        )

    # Confusion matrix
    lines += [
        "",
        "## Confusion Matrix (Counts)",
        _format_confusion_matrix_markdown(metrics["confusion_matrix"], used_labels),
    ]

    # Notes
    if experiment_notes:
        lines += ["", "## Notes", experiment_notes]

    lines += [
        "",
        "---",
        "*Generated by pipeline_utils.generate_report()*",
    ]

    md_content = "\n".join(lines)
    md_path.write_text(md_content, encoding="utf-8")

    # --- Build JSON sidecar ---
    json_data = {
        "report_id": base_name,
        "generated_at": now.isoformat(),
        "model": {
            "name": model_name,
            "type": model_type,
            **{k: v for k, v in model_info.items()},
            "config": model_config,
        },
        "classification_config": {
            "hypothesis_template": used_template,
            "candidate_labels": used_labels,
            "nli_phrases": [used_template.format(l) for l in used_labels],
            "label_mapping": label_mapping,
            "label_mapping_remapped": remapped if remapped else None,
        },
        "dataset": {
            "id": DATASET_ID,
            "evaluated_on": metrics.get("experiment_name", "N/A"),
            "n_articles": n_articles,
            "n_classes": len(used_labels),
            "split_info": split_info,
        },
        "runtime": {
            "duration_seconds": round(timer.duration_seconds, 2),
            "duration_formatted": timer.duration_formatted,
            "articles_per_second": round(timer.articles_per_second(n_articles), 2),
            "gpu": gpu_info,
            "estimated_cost_per_1000": cost_est,
        },
        "training_params": training_params,
        "uploaded_model_url": uploaded_model_url,
        "metrics": {
            "f1_macro": round(metrics["f1_macro"], 4),
            "f1_weighted": round(metrics["f1_weighted"], 4),
            "precision_macro": round(metrics["precision_macro"], 4),
            "precision_weighted": round(metrics["precision_weighted"], 4),
            "recall_macro": round(metrics["recall_macro"], 4),
            "recall_weighted": round(metrics["recall_weighted"], 4),
            "accuracy": round(metrics["accuracy"], 4),
        },
        "per_class": [
            {
                "label": row["Label"],
                "precision": round(row["Precision"], 4),
                "recall": round(row["Recall"], 4),
                "f1": round(row["F1"], 4),
                "support": int(row["Support"]),
            }
            for _, row in pc.iterrows()
        ],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "notes": experiment_notes,
    }

    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Report saved: {md_path}")
    print(f"JSON saved:   {json_path}")
    return str(md_path)


# ---------------------------------------------------------------------------
# REPORT COMPARISON
# ---------------------------------------------------------------------------


def load_report_json(report_path: str) -> Dict[str, Any]:
    """Load a report JSON. Accepts .md or .json path."""
    p = Path(report_path)
    if p.suffix == ".md":
        p = p.with_suffix(".json")
    return json.loads(p.read_text(encoding="utf-8"))


def compare_reports(report_paths: List[str]) -> pd.DataFrame:
    """Load multiple report JSONs and return a comparison DataFrame."""
    rows = []
    for path in report_paths:
        data = load_report_json(path)
        row = {
            "report_id": data["report_id"],
            "model_name": data["model"]["name"],
            "model_type": data["model"]["type"],
            "n_articles": data["dataset"]["n_articles"],
            "duration_s": data["runtime"]["duration_seconds"],
            "articles_per_sec": data["runtime"]["articles_per_second"],
            **data["metrics"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MODEL UPLOAD (optional, fine-tuned models only)
# ---------------------------------------------------------------------------


def upload_model_to_hub(
    model: Any,
    tokenizer: Any,
    repo_name: str,
    private: bool = True,
    training_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Upload fine-tuned model + tokenizer to HuggingFace Hub.

    Args:
        model: The trained model (transformers PreTrainedModel).
        tokenizer: The tokenizer.
        repo_name: Full repo name, e.g. "Zorryy/gbert-news-classifier-v1".
        private: If True, create a private repository.
        training_params: If provided, saved as training_config.json in the repo.

    Returns:
        HuggingFace Hub URL of the uploaded model.
    """
    from huggingface_hub import HfApi

    print(f"Uploading model to {repo_name} (private={private})...")

    model.push_to_hub(repo_name, private=private)
    tokenizer.push_to_hub(repo_name, private=private)

    if training_params:
        api = HfApi()
        config_content = json.dumps(training_params, ensure_ascii=False, indent=2)
        api.upload_file(
            path_or_fileobj=config_content.encode("utf-8"),
            path_in_repo="training_config.json",
            repo_id=repo_name,
        )

    url = f"https://huggingface.co/{repo_name}"
    print(f"Model uploaded: {url}")
    return url
