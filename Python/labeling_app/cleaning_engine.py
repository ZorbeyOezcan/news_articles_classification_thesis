import os

import numpy as np
import pandas as pd

from data_loader import LABELED_CSV_PATH
from labeling_engine import CATEGORIES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default output directories (can be overridden via config)
UMAP_OUTPUT_DIR = os.path.join(BASE_DIR, "label_quality_output")
CLEANLAB_OUTPUT_DIR = os.path.join(BASE_DIR, "cleanlab_output")


def get_available_sources():
    """Check which quality-check output files exist and return availability info."""
    umap_csv = os.path.join(UMAP_OUTPUT_DIR, "labeled_with_quality_scores.csv")
    cleanlab_csv = os.path.join(CLEANLAB_OUTPUT_DIR, "labeled_with_cleanlab_scores.csv")
    cleanlab_probs = os.path.join(CLEANLAB_OUTPUT_DIR, "oof_pred_probs.npy")

    return {
        "umap": {
            "available": os.path.exists(umap_csv),
            "path": UMAP_OUTPUT_DIR,
        },
        "cleanlab": {
            "available": os.path.exists(cleanlab_csv) and os.path.exists(cleanlab_probs),
            "path": CLEANLAB_OUTPUT_DIR,
        },
    }


class CleaningSession:
    def __init__(self, conn, config):
        self.conn = conn
        self.source_type = config["source"]  # "umap" or "cleanlab"

        # Load flagged articles based on source
        if self.source_type == "umap":
            self._load_umap_data(config)
        else:
            self._load_cleanlab_data(config)

        # Position in the flagged items list
        self.current_index = -1
        self.current_article_id = None

        # Change tracking
        self.changes = []  # list of {id, action, old_label, new_label}
        self.undo_stack = []  # for undo support
        self.reviewed_ids = set()
        self.skipped_ids = set()

        # Counters
        self.n_relabeled = 0
        self.n_kept = 0
        self.n_removed = 0
        self.n_skipped = 0

    def _load_umap_data(self, config):
        """Load UMAP outlier detection results."""
        csv_path = os.path.join(UMAP_OUTPUT_DIR, "labeled_with_quality_scores.csv")
        df = pd.read_csv(csv_path, encoding="utf-8")

        # Filter to outliers
        filter_both = config.get("filter_both_methods", False)
        if filter_both:
            mask = df["centroid_outlier"].astype(bool) & df["lof_outlier"].astype(bool)
        else:
            mask = df["centroid_outlier"].astype(bool) | df["lof_outlier"].astype(bool)

        flagged = df[mask].copy()

        # Sort by centroid_dist descending (most suspicious first)
        flagged = flagged.sort_values("centroid_dist", ascending=False).reset_index(drop=True)

        self.flagged_items = []
        for _, row in flagged.iterrows():
            methods = []
            if row["centroid_outlier"]:
                methods.append("Centroid")
            if row["lof_outlier"]:
                methods.append("LOF")

            self.flagged_items.append({
                "id": int(row["id"]),
                "current_label": row["label"],
                "quality_signals": {
                    "source": "umap",
                    "centroid_dist": round(float(row["centroid_dist"]), 4),
                    "methods": methods,
                },
                "suggestions": {},  # No per-class confidence from UMAP
            })

        self.total = len(self.flagged_items)

    def _load_cleanlab_data(self, config):
        """Load Cleanlab label issue results."""
        csv_path = os.path.join(CLEANLAB_OUTPUT_DIR, "labeled_with_cleanlab_scores.csv")
        probs_path = os.path.join(CLEANLAB_OUTPUT_DIR, "oof_pred_probs.npy")

        df = pd.read_csv(csv_path, encoding="utf-8")
        pred_probs = np.load(probs_path)

        # Filter to label issues
        score_threshold = config.get("score_threshold", 1.0)
        mask = df["is_label_issue"].astype(bool)
        if score_threshold < 1.0:
            mask = mask & (df["label_quality_score"] <= score_threshold)

        flagged = df[mask].copy()
        flagged_indices = flagged.index.tolist()

        # Sort by label_quality_score ascending (most suspicious first)
        sort_order = flagged["label_quality_score"].argsort()
        flagged = flagged.iloc[sort_order].reset_index(drop=True)
        flagged_indices = [flagged_indices[i] for i in sort_order]

        self.flagged_items = []
        for row_pos, (_, row) in enumerate(flagged.iterrows()):
            original_idx = flagged_indices[row_pos]
            probs = pred_probs[original_idx]

            # Build suggestions dict: {category: probability}
            suggestions = {}
            for cat_idx, cat in enumerate(CATEGORIES):
                if cat_idx < len(probs):
                    suggestions[cat] = round(float(probs[cat_idx]), 4)

            self.flagged_items.append({
                "id": int(row["id"]),
                "current_label": row["label"],
                "quality_signals": {
                    "source": "cleanlab",
                    "label_quality_score": round(float(row["label_quality_score"]), 4),
                    "predicted_label": row.get("predicted_label", ""),
                },
                "suggestions": suggestions,
            })

        self.total = len(self.flagged_items)

    def next_article(self):
        """Get the next flagged article to review."""
        self.current_index += 1

        # Skip already-reviewed items
        while self.current_index < self.total:
            item = self.flagged_items[self.current_index]
            if item["id"] not in self.reviewed_ids:
                break
            self.current_index += 1

        if self.current_index >= self.total:
            return None

        item = self.flagged_items[self.current_index]
        self.current_article_id = item["id"]

        # Fetch full article text from DB
        row = self.conn.execute(
            "SELECT headline, text, domain FROM articles WHERE id = ?",
            (item["id"],),
        ).fetchone()

        if row is None:
            # Article not in DB, use what we have
            return {
                "id": item["id"],
                "headline": "(not found)",
                "text": "",
                "domain": "",
                "current_label": item["current_label"],
                "quality_signals": item["quality_signals"],
                "suggestions": item["suggestions"],
            }

        return {
            "id": item["id"],
            "headline": row[0] or "",
            "text": row[1] or "",
            "domain": row[2] or "",
            "current_label": item["current_label"],
            "quality_signals": item["quality_signals"],
            "suggestions": item["suggestions"],
        }

    def relabel(self, article_id, new_label):
        """Change the label of an article."""
        item = self._find_item(article_id)
        if item is None:
            return {"error": "Article not found"}

        old_label = item["current_label"]
        self.changes.append({
            "id": article_id,
            "action": "relabel",
            "old_label": old_label,
            "new_label": new_label,
        })
        self.undo_stack.append({
            "id": article_id,
            "action": "relabel",
            "old_label": old_label,
            "new_label": new_label,
        })
        self.reviewed_ids.add(article_id)
        self.n_relabeled += 1

        return self.get_stats()

    def keep(self, article_id):
        """Keep the current label as is."""
        item = self._find_item(article_id)
        if item is None:
            return {"error": "Article not found"}

        self.changes.append({
            "id": article_id,
            "action": "keep",
            "old_label": item["current_label"],
            "new_label": item["current_label"],
        })
        self.undo_stack.append({
            "id": article_id,
            "action": "keep",
            "old_label": item["current_label"],
        })
        self.reviewed_ids.add(article_id)
        self.n_kept += 1

        return self.get_stats()

    def remove_label(self, article_id):
        """Remove the label (article will be deleted from labeled CSV)."""
        item = self._find_item(article_id)
        if item is None:
            return {"error": "Article not found"}

        self.changes.append({
            "id": article_id,
            "action": "remove",
            "old_label": item["current_label"],
            "new_label": None,
        })
        self.undo_stack.append({
            "id": article_id,
            "action": "remove",
            "old_label": item["current_label"],
        })
        self.reviewed_ids.add(article_id)
        self.n_removed += 1

        return self.get_stats()

    def skip(self, article_id):
        """Skip without any change (not counted as reviewed)."""
        self.undo_stack.append({
            "id": article_id,
            "action": "skip",
        })
        self.skipped_ids.add(article_id)
        self.n_skipped += 1

        return self.get_stats()

    def undo_last(self):
        """Undo the most recent action."""
        if not self.undo_stack:
            return {"error": "Nothing to undo"}

        last = self.undo_stack.pop()
        article_id = last["id"]
        action = last["action"]

        if action == "skip":
            self.skipped_ids.discard(article_id)
            self.n_skipped -= 1
        else:
            self.reviewed_ids.discard(article_id)
            # Remove the corresponding change
            for i in range(len(self.changes) - 1, -1, -1):
                if self.changes[i]["id"] == article_id:
                    self.changes.pop(i)
                    break
            if action == "relabel":
                self.n_relabeled -= 1
            elif action == "keep":
                self.n_kept -= 1
            elif action == "remove":
                self.n_removed -= 1

        # Rewind index to show this article again
        for idx, item in enumerate(self.flagged_items):
            if item["id"] == article_id:
                self.current_index = idx - 1  # will be incremented by next_article
                break

        # Fetch and return the article
        article = self.next_article()
        return {
            "article": article,
            "stats": self.get_stats(),
        }

    def save_to_csv(self):
        """Apply all changes to the labeled CSV file."""
        if not self.changes:
            return 0

        # Collect actionable changes (skip "keep" actions)
        relabels = {}  # id -> new_label
        removals = set()  # ids to remove

        for change in self.changes:
            aid = change["id"]
            if change["action"] == "relabel":
                relabels[aid] = change["new_label"]
                removals.discard(aid)  # in case it was previously marked for removal
            elif change["action"] == "remove":
                removals.add(aid)
                relabels.pop(aid, None)
            # "keep" doesn't need any CSV change

        if not relabels and not removals:
            return 0

        # Load the full labeled CSV
        df = pd.read_csv(LABELED_CSV_PATH, encoding="utf-8")

        # Apply relabels
        for aid, new_label in relabels.items():
            mask = df["id"] == aid
            df.loc[mask, "label"] = new_label

        # Remove labels
        if removals:
            df = df[~df["id"].isin(removals)]

        # Write back (overwrite, not append)
        df.to_csv(LABELED_CSV_PATH, index=False, encoding="utf-8")

        # Update SQLite labeled table
        for aid, new_label in relabels.items():
            self.conn.execute(
                "UPDATE labeled SET label = ? WHERE id = ?",
                (new_label, aid),
            )
        for aid in removals:
            self.conn.execute("DELETE FROM labeled WHERE id = ?", (aid,))
        self.conn.commit()

        n_changes = len(relabels) + len(removals)
        return n_changes

    def get_stats(self):
        """Return current cleaning session statistics."""
        n_reviewed = len(self.reviewed_ids)
        progress = n_reviewed / self.total if self.total > 0 else 0

        return {
            "total_flagged": self.total,
            "reviewed": n_reviewed,
            "remaining": self.total - n_reviewed,
            "progress": progress,
            "relabeled": self.n_relabeled,
            "kept": self.n_kept,
            "removed": self.n_removed,
            "skipped": self.n_skipped,
            "source": self.source_type,
            "all_done": n_reviewed >= self.total,
        }

    def _find_item(self, article_id):
        """Find a flagged item by article ID."""
        for item in self.flagged_items:
            if item["id"] == article_id:
                return item
        return None
