"""
hpt_utils.py — Shared utilities for Optuna-based hyperparameter tuning.

Provides:
- make_compute_metrics: Factory for HF Trainer compute_metrics with per-class scores
- HPTMetricsCallback: Per-epoch metric capture, NaN/LR=0 detection
- store_fold_metrics / store_trial_summary: Rich metric storage in Optuna user_attrs
- resolve_db_path / setup_study: DB management (new/continue mode)
- run_hpt_with_nan_exclusion: NaN-aware trial loop
"""

import gc
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Module-level stores (bridging compute_metrics ↔ callback)
# ---------------------------------------------------------------------------

_prediction_store: Dict[str, Any] = {}
_epoch_metrics_store: Dict[str, List[Dict]] = {}


def _safe_label_key(label_name: str) -> str:
    """Convert label name to a safe key for flat metric dicts."""
    return (
        label_name.replace("/", "_")
        .replace(",", "")
        .replace(" ", "_")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
    )


# ---------------------------------------------------------------------------
# Compute Metrics Factory
# ---------------------------------------------------------------------------


def make_compute_metrics(
    all_labels: List[str],
    id2label: Dict[int, str],
) -> Callable:
    """Return a compute_metrics function that captures per-class scores.

    The returned function:
    - Computes aggregate metrics (f1_macro, f1_weighted, precision_macro, recall_macro, accuracy)
    - Computes per-class precision/recall/F1 as flat keys (class_{safe_name}_{metric})
    - Stores raw (preds, labels) in _prediction_store for confusion matrix in callback
    """
    n_classes = len(all_labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Store for confusion matrix computation in callback
        _prediction_store["current"] = {"preds": preds, "labels": labels}

        result = {
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
            "accuracy": accuracy_score(labels, preds),
        }

        # Per-class metrics
        report = classification_report(
            labels,
            preds,
            labels=list(range(n_classes)),
            target_names=all_labels,
            output_dict=True,
            zero_division=0,
        )
        for label_name in all_labels:
            safe = _safe_label_key(label_name)
            if label_name in report:
                result[f"class_{safe}_f1"] = report[label_name]["f1-score"]
                result[f"class_{safe}_precision"] = report[label_name]["precision"]
                result[f"class_{safe}_recall"] = report[label_name]["recall"]

        return result

    return compute_metrics


# ---------------------------------------------------------------------------
# HPT Metrics Callback
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object  # allow import without transformers installed


class HPTMetricsCallback(TrainerCallback):
    """Captures per-epoch metrics, detects NaN/LR=0, stores to module-level dict.

    Usage:
        callback = HPTMetricsCallback(trial_number=0, fold_idx=0,
                                       all_labels=LABELS, id2label=ID2LABEL)
        trainer = Trainer(..., callbacks=[callback])
        trainer.train()
        if callback.nan_detected:
            raise optuna.TrialPruned()
    """

    def __init__(
        self,
        trial_number: int,
        fold_idx: int,
        all_labels: List[str],
        id2label: Dict[int, str],
    ):
        self.trial_number = trial_number
        self.fold_idx = fold_idx
        self.all_labels = all_labels
        self.id2label = id2label
        self.store_key = f"trial_{trial_number}_fold_{fold_idx}"
        _epoch_metrics_store[self.store_key] = []
        self._nan_detected = False
        self._lr_zero_detected = False

    # --- Step-level: check training loss for NaN ---
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        loss = logs.get("loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            self._nan_detected = True
            print(
                f"  [NaN] Loss NaN/Inf at step {state.global_step} "
                f"(Trial {self.trial_number}, Fold {self.fold_idx})"
            )
            control.should_training_stop = True

        lr = logs.get("learning_rate")
        if lr is not None and lr == 0.0 and state.global_step > 0:
            self._lr_zero_detected = True
            print(
                f"  [LR=0] Learning rate reached 0 at step {state.global_step} "
                f"(Trial {self.trial_number}, Fold {self.fold_idx})"
            )
            control.should_training_stop = True

    # --- Epoch-level: capture full eval metrics ---
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        # Determine current learning rate from log history
        current_lr = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "learning_rate" in entry:
                    current_lr = entry["learning_rate"]
                    break

        epoch_data = {
            "epoch": int(state.epoch) if state.epoch else len(
                _epoch_metrics_store[self.store_key]
            ) + 1,
            "eval_loss": metrics.get("eval_loss"),
            "eval_f1_macro": metrics.get("eval_f1_macro"),
            "eval_f1_weighted": metrics.get("eval_f1_weighted"),
            "eval_precision_macro": metrics.get("eval_precision_macro"),
            "eval_recall_macro": metrics.get("eval_recall_macro"),
            "eval_accuracy": metrics.get("eval_accuracy"),
            "learning_rate": current_lr,
            "nan_detected": self._nan_detected,
            "lr_zero_detected": self._lr_zero_detected,
        }

        # Per-class metrics from compute_metrics output
        per_class = {}
        for label_name in self.all_labels:
            safe = _safe_label_key(label_name)
            per_class[label_name] = {
                "f1": metrics.get(f"eval_class_{safe}_f1"),
                "precision": metrics.get(f"eval_class_{safe}_precision"),
                "recall": metrics.get(f"eval_class_{safe}_recall"),
            }
        epoch_data["per_class"] = per_class

        # Confusion matrix from prediction store
        stored = _prediction_store.get("current")
        if stored is not None:
            cm = confusion_matrix(
                stored["labels"],
                stored["preds"],
                labels=list(range(len(self.all_labels))),
            )
            epoch_data["confusion_matrix"] = cm.tolist()

        _epoch_metrics_store[self.store_key].append(epoch_data)

        # Check eval metrics for NaN / collapse
        eval_loss = metrics.get("eval_loss", 0)
        eval_f1 = metrics.get("eval_f1_macro", 1.0)
        if eval_loss is not None and (math.isnan(eval_loss) or math.isinf(eval_loss)):
            self._nan_detected = True
            control.should_training_stop = True
        if eval_f1 is not None and eval_f1 < 0.05:
            self._nan_detected = True
            control.should_training_stop = True

    @property
    def nan_detected(self) -> bool:
        return self._nan_detected

    @property
    def lr_zero_detected(self) -> bool:
        return self._lr_zero_detected

    def get_epoch_history(self) -> List[Dict]:
        return _epoch_metrics_store.get(self.store_key, [])


# ---------------------------------------------------------------------------
# Metric Storage Helpers
# ---------------------------------------------------------------------------


def store_fold_metrics(
    trial,
    fold_idx: int,
    callback: HPTMetricsCallback,
    eval_result: Dict[str, Any],
    all_labels: List[str],
    id2label: Dict[int, str],
) -> None:
    """Store comprehensive fold metrics in trial user attributes."""
    stored = _prediction_store.get("current")

    fold_data: Dict[str, Any] = {
        "fold_idx": fold_idx,
        "f1_macro": eval_result.get("eval_f1_macro"),
        "f1_weighted": eval_result.get("eval_f1_weighted"),
        "precision_macro": eval_result.get("eval_precision_macro"),
        "recall_macro": eval_result.get("eval_recall_macro"),
        "accuracy": eval_result.get("eval_accuracy"),
        "per_class": {},
        "confusion_matrix": None,
        "epoch_history": callback.get_epoch_history(),
        "best_epoch": None,
        "total_epochs_trained": len(callback.get_epoch_history()),
        "nan_detected": callback.nan_detected,
        "lr_zero_detected": callback.lr_zero_detected,
    }

    # Per-class from final eval
    for label_name in all_labels:
        safe = _safe_label_key(label_name)
        fold_data["per_class"][label_name] = {
            "f1": eval_result.get(f"eval_class_{safe}_f1"),
            "precision": eval_result.get(f"eval_class_{safe}_precision"),
            "recall": eval_result.get(f"eval_class_{safe}_recall"),
        }

    # Confusion matrix from final eval
    if stored is not None:
        cm = confusion_matrix(
            stored["labels"],
            stored["preds"],
            labels=list(range(len(all_labels))),
        )
        fold_data["confusion_matrix"] = cm.tolist()

    # Best epoch (highest F1 macro in epoch history)
    history = callback.get_epoch_history()
    if history:
        f1_values = [e.get("eval_f1_macro") or 0 for e in history]
        fold_data["best_epoch"] = int(np.argmax(f1_values)) + 1

    trial.set_user_attr(f"fold_{fold_idx}_metrics", json.dumps(fold_data))


def store_fold_metrics_partial(
    trial,
    fold_idx: int,
    callback: HPTMetricsCallback,
    all_labels: List[str],
) -> None:
    """Store partial fold data for NaN/failed trials (no eval_result available)."""
    fold_data: Dict[str, Any] = {
        "fold_idx": fold_idx,
        "f1_macro": None,
        "f1_weighted": None,
        "precision_macro": None,
        "recall_macro": None,
        "accuracy": None,
        "per_class": {},
        "confusion_matrix": None,
        "epoch_history": callback.get_epoch_history(),
        "best_epoch": None,
        "total_epochs_trained": len(callback.get_epoch_history()),
        "nan_detected": callback.nan_detected,
        "lr_zero_detected": callback.lr_zero_detected,
    }

    trial.set_user_attr(f"fold_{fold_idx}_metrics", json.dumps(fold_data))


def store_trial_summary(
    trial,
    n_folds: int,
    all_labels: List[str],
) -> None:
    """Aggregate fold metrics into trial-level summary in user attributes."""
    fold_data_list = []
    for fold_idx in range(n_folds):
        key = f"fold_{fold_idx}_metrics"
        raw = trial.user_attrs.get(key)
        if raw:
            fold_data_list.append(json.loads(raw))

    if not fold_data_list:
        return

    fold_f1s = [
        fd["f1_macro"]
        for fd in fold_data_list
        if fd["f1_macro"] is not None
    ]

    summary: Dict[str, Any] = {
        "mean_f1_macro": float(np.mean(fold_f1s)) if fold_f1s else None,
        "std_f1_macro": float(np.std(fold_f1s)) if fold_f1s else None,
        "per_fold_f1": fold_f1s,
        "mean_per_class": {},
        "mean_confusion_matrix": None,
        "any_nan": any(fd.get("nan_detected", False) for fd in fold_data_list),
        "folds_completed": len(fold_data_list),
    }

    # Average per-class metrics across folds
    for label_name in all_labels:
        class_f1s = [
            fd["per_class"].get(label_name, {}).get("f1")
            for fd in fold_data_list
            if fd["per_class"].get(label_name, {}).get("f1") is not None
        ]
        class_precs = [
            fd["per_class"].get(label_name, {}).get("precision")
            for fd in fold_data_list
            if fd["per_class"].get(label_name, {}).get("precision") is not None
        ]
        class_recs = [
            fd["per_class"].get(label_name, {}).get("recall")
            for fd in fold_data_list
            if fd["per_class"].get(label_name, {}).get("recall") is not None
        ]
        summary["mean_per_class"][label_name] = {
            "f1": float(np.mean(class_f1s)) if class_f1s else None,
            "precision": float(np.mean(class_precs)) if class_precs else None,
            "recall": float(np.mean(class_recs)) if class_recs else None,
        }

    # Average confusion matrix across folds
    cms = [
        np.array(fd["confusion_matrix"])
        for fd in fold_data_list
        if fd["confusion_matrix"] is not None
    ]
    if cms:
        summary["mean_confusion_matrix"] = np.mean(cms, axis=0).tolist()

    trial.set_user_attr("trial_summary", json.dumps(summary))


# ---------------------------------------------------------------------------
# DB Management
# ---------------------------------------------------------------------------


def resolve_db_path(
    db_mode: str,
    study_name: str,
    base_dir: Path,
) -> Tuple[str, str]:
    """Resolve database path based on mode.

    Args:
        db_mode: "new" or "continue"
        study_name: Optuna study name (used to derive filename)
        base_dir: Directory for DB files (e.g., REPORTS_DIR on Google Drive)

    Returns:
        (storage_url, storage_path_str) tuple
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"hpt_{study_name}"

    if db_mode == "continue":
        candidates = sorted(base_dir.glob(f"{base_name}*.db"))
        if not candidates:
            raise FileNotFoundError(
                f"No existing DB found for study '{study_name}' in {base_dir}.\n"
                f"Looked for: {base_name}*.db\n"
                f"Use DB_MODE='new' to create one."
            )
        db_path = candidates[-1]
        print(f"[DB CONTINUE] Using existing DB: {db_path}")

    elif db_mode == "new":
        existing = sorted(base_dir.glob(f"{base_name}*.db"))

        if not existing:
            db_path = base_dir / f"{base_name}.db"
        else:
            max_suffix = 0
            for p in existing:
                stem = p.stem
                if stem == base_name:
                    max_suffix = max(max_suffix, 1)
                elif stem.startswith(base_name + "_"):
                    try:
                        suffix_num = int(stem[len(base_name) + 1 :])
                        max_suffix = max(max_suffix, suffix_num)
                    except ValueError:
                        pass
            next_suffix = max_suffix + 1
            db_path = base_dir / f"{base_name}_{next_suffix}.db"

        print(f"[DB NEW] Creating new DB: {db_path}")

    else:
        raise ValueError(
            f"Invalid db_mode: '{db_mode}'. Use 'new' or 'continue'."
        )

    storage_url = f"sqlite:///{db_path}"
    return storage_url, str(db_path)


def setup_study(
    storage_url: str,
    study_name: str,
    db_mode: str,
    n_trials_target: int,
    sampler,
    pruner,
) -> Tuple:
    """Create or load study, handle RUNNING trials, calculate remaining trials.

    Returns:
        (study, n_remaining, n_complete_existing)
    """
    import optuna

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=(db_mode == "continue"),
    )

    if db_mode == "continue":
        complete_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        running_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.RUNNING
        ]

        n_complete = len(complete_trials)
        n_running = len(running_trials)

        if n_running > 0:
            print(
                f"[DB CONTINUE] Found {n_running} RUNNING trials "
                f"(from previous crash/timeout)."
            )
            print(
                f"  These will be ignored by Optuna and re-sampled as needed."
            )

        n_remaining = max(0, n_trials_target - n_complete)
        print(
            f"[DB CONTINUE] {n_complete} COMPLETE trials found. "
            f"{n_remaining} remaining to reach target of {n_trials_target}."
        )

        return study, n_remaining, n_complete
    else:
        return study, n_trials_target, 0


def count_valid_complete_trials(study) -> int:
    """Count COMPLETE trials excluding NaN-flagged ones."""
    import optuna

    count = 0
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        # Check if this was a NaN trial
        if trial.user_attrs.get("nan_trial"):
            continue
        count += 1
    return count


def run_hpt_with_nan_exclusion(
    study,
    objective: Callable,
    n_valid_target: int,
    timeout: Optional[int] = None,
    max_total_trials: Optional[int] = None,
    gc_after_trial: bool = True,
    show_progress_bar: bool = True,
) -> int:
    """Run HPT trials until n_valid_target non-NaN COMPLETE trials exist.

    NaN trials (detected via trial user_attrs) do not count toward the target.
    max_total_trials is a safety cap to prevent infinite loops.

    Returns:
        Number of valid complete trials achieved.
    """
    if max_total_trials is None:
        max_total_trials = n_valid_target * 3

    n_valid = count_valid_complete_trials(study)
    total_attempted = len(study.trials)

    print(
        f"[HPT] Target: {n_valid_target} valid trials. "
        f"Currently: {n_valid} valid, {total_attempted} total."
    )

    while n_valid < n_valid_target and total_attempted < max_total_trials:
        remaining = n_valid_target - n_valid
        batch = min(remaining + 2, 5)

        study.optimize(
            objective,
            n_trials=batch,
            timeout=timeout,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )

        n_valid = count_valid_complete_trials(study)
        total_attempted = len(study.trials)
        print(
            f"[HPT] Progress: {n_valid}/{n_valid_target} valid trials "
            f"({total_attempted} total attempted)"
        )

    if n_valid < n_valid_target:
        print(
            f"[HPT] WARNING: Only achieved {n_valid}/{n_valid_target} valid "
            f"trials after {total_attempted} total attempts."
        )

    return n_valid
