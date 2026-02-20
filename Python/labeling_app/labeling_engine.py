import os
import re
import sqlite3

import pandas as pd

from data_loader import LABELED_CSV_PATH, get_total_domain_distribution

CATEGORIES = [
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

SPECIAL_LABELS = ["skipped", "not_clean"]


class LabelingSession:
    def __init__(self, conn, config, existing_category_counts):
        self.conn = conn
        self.existing_counts = {cat: existing_category_counts.get(cat, 0) for cat in CATEGORIES}

        # Targets per category for THIS session
        if config.get("normalise"):
            norm_target = config["normalise_count"]
            self.targets = {
                cat: max(0, norm_target - self.existing_counts.get(cat, 0))
                for cat in CATEGORIES
            }
        elif config["mode"] == "uniform":
            self.targets = {cat: config["count"] for cat in CATEGORIES}
        else:
            self.targets = {cat: config["targets"].get(cat, 0) for cat in CATEGORIES}

        # Session counts (labels applied this session only)
        self.session_counts = {cat: 0 for cat in CATEGORIES}

        # Special label counts (this session)
        self.session_skipped = 0
        self.session_not_clean = 0

        # Include/exclude settings from start page
        self.include_skipped = config.get("include_skipped", False)
        self.include_not_clean = config.get("include_not_clean", False)

        # Domain balanced mode
        self.domain_balanced = config.get("domain_balanced", True)
        if self.domain_balanced:
            self.domain_quotas = self._compute_domain_quotas()
            self.domain_session_counts = {d: 0 for d in self.domain_quotas}
        else:
            self.domain_quotas = {}
            self.domain_session_counts = {}

        # Undo stack: list of (article_id, label, domain, is_special) tuples
        self.undo_stack = []

        # Session labels: ordered list of {id, label}
        self.session_labels = []

        # Create temp table for seen IDs (labeled or skipped this session)
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS seen_ids (id INTEGER PRIMARY KEY)")
        self.conn.execute("DELETE FROM seen_ids")

        # Search mode
        self.search_query = None  # None = random mode, string = search mode

        self.current_article_id = None

    def _compute_domain_quotas(self):
        total_target = sum(self.targets.values())
        full_dist = get_total_domain_distribution(self.conn)
        total_articles = sum(full_dist.values())

        if total_articles == 0:
            return {}

        quotas = {}
        for domain, count in full_dist.items():
            proportion = count / total_articles
            quota = round(proportion * total_target)
            quotas[domain] = min(quota, count)

        diff = total_target - sum(quotas.values())
        sorted_domains = sorted(quotas.keys(), key=lambda d: full_dist.get(d, 0), reverse=True)
        for d in sorted_domains:
            if diff == 0:
                break
            if diff > 0:
                quotas[d] += 1
                diff -= 1
            elif diff < 0 and quotas[d] > 0:
                quotas[d] -= 1
                diff += 1

        return quotas

    def _build_exclusion_clause(self):
        """Build SQL clause + params to exclude already-labeled articles,
        optionally re-including 'skipped' and/or 'not_clean' articles."""
        re_include = []
        if self.include_skipped:
            re_include.append("skipped")
        if self.include_not_clean:
            re_include.append("not_clean")

        if re_include:
            placeholders = ",".join("?" * len(re_include))
            clause = f"id NOT IN (SELECT id FROM labeled WHERE label NOT IN ({placeholders}))"
            return clause, re_include
        else:
            return "id NOT IN (SELECT id FROM labeled)", []

    def _build_search_clause(self):
        """Build SQL WHERE clause for search query. Returns (clause, params) or (None, [])."""
        if not self.search_query:
            return None, []
        return _parse_search_query(self.search_query)

    def next_article(self):
        """Select the next random article, respecting domain quotas and search filter."""
        if self._all_targets_met():
            return None

        excl_clause, excl_params = self._build_exclusion_clause()
        search_clause, search_params = self._build_search_clause()

        where_parts = [
            excl_clause,
            "id NOT IN (SELECT id FROM seen_ids)",
        ]
        params = list(excl_params)

        if search_clause:
            where_parts.append(search_clause)
            params.extend(search_params)

        if self.domain_balanced:
            eligible_domains = [
                d for d, quota in self.domain_quotas.items()
                if self.domain_session_counts.get(d, 0) < quota
            ]
            if not eligible_domains:
                return None

            placeholders = ",".join("?" * len(eligible_domains))
            where_parts.append(f"domain IN ({placeholders})")
            params.extend(eligible_domains)

        where_sql = " AND ".join(where_parts)
        row = self.conn.execute(
            f"SELECT id, domain, headline, text FROM articles WHERE {where_sql} ORDER BY RANDOM() LIMIT 1",
            params,
        ).fetchone()

        if row is None:
            return None

        self.current_article_id = row[0]
        self.conn.execute("INSERT OR IGNORE INTO seen_ids VALUES (?)", (row[0],))
        return {"id": row[0], "domain": row[1], "headline": row[2], "text": row[3]}

    def apply_label(self, article_id, label):
        """Record a real category label. Returns updated stats."""
        self.session_counts[label] = self.session_counts.get(label, 0) + 1
        self.session_labels.append({"id": article_id, "label": label})

        domain = self.conn.execute(
            "SELECT domain FROM articles WHERE id = ?", (article_id,)
        ).fetchone()[0]

        if self.domain_balanced:
            self.domain_session_counts[domain] = self.domain_session_counts.get(domain, 0) + 1

        self.undo_stack.append((article_id, label, domain, False))
        return self.get_stats()

    def skip(self, article_id):
        """Mark article as 'skipped'. Saved to CSV but no category/domain counts."""
        self.session_skipped += 1
        self.session_labels.append({"id": article_id, "label": "skipped"})

        domain = self.conn.execute(
            "SELECT domain FROM articles WHERE id = ?", (article_id,)
        ).fetchone()[0]

        self.undo_stack.append((article_id, "skipped", domain, True))
        return self.get_stats()

    def mark_not_clean(self, article_id):
        """Mark article as 'not_clean'. Saved to CSV but no category/domain counts."""
        self.session_not_clean += 1
        self.session_labels.append({"id": article_id, "label": "not_clean"})

        domain = self.conn.execute(
            "SELECT domain FROM articles WHERE id = ?", (article_id,)
        ).fetchone()[0]

        self.undo_stack.append((article_id, "not_clean", domain, True))
        return self.get_stats()

    def undo_last(self):
        """Undo the most recent action (label, skip, or not_clean)."""
        if not self.undo_stack:
            return {"error": "Nothing to undo"}

        article_id, label, domain, is_special = self.undo_stack.pop()

        if is_special:
            if label == "skipped":
                self.session_skipped -= 1
            elif label == "not_clean":
                self.session_not_clean -= 1
        else:
            self.session_counts[label] -= 1
            if self.domain_balanced:
                self.domain_session_counts[domain] -= 1

        # Remove from session labels (last occurrence)
        for i in range(len(self.session_labels) - 1, -1, -1):
            if self.session_labels[i]["id"] == article_id:
                self.session_labels.pop(i)
                break

        self.conn.execute("DELETE FROM seen_ids WHERE id = ?", (article_id,))

        row = self.conn.execute(
            "SELECT id, domain, headline, text FROM articles WHERE id = ?", (article_id,)
        ).fetchone()

        return {
            "article": {"id": row[0], "domain": row[1], "headline": row[2], "text": row[3]},
            "stats": self.get_stats(),
        }

    def set_search(self, query):
        """Set search mode with given query. Pass None or empty to clear."""
        self.search_query = query if query else None

    def save_to_csv(self):
        """Write all session labels (including skipped/not_clean) to CSV."""
        if not self.session_labels:
            return 0

        labeled_ids = [s["id"] for s in self.session_labels]
        label_map = {s["id"]: s["label"] for s in self.session_labels}

        placeholders = ",".join("?" * len(labeled_ids))
        rows = self.conn.execute(
            f"""
            SELECT id, domain, url, date_time, headline, author, text, text_length
            FROM articles WHERE id IN ({placeholders})
            """,
            labeled_ids,
        ).fetchall()

        df = pd.DataFrame(
            rows,
            columns=["id", "domain", "url", "date_time", "headline", "author", "text", "text_length"],
        )
        df["label"] = df["id"].map(label_map)

        file_exists = os.path.exists(LABELED_CSV_PATH)
        df.to_csv(LABELED_CSV_PATH, mode="a", header=not file_exists, index=False, encoding="utf-8")

        for s in self.session_labels:
            self.conn.execute(
                "INSERT OR REPLACE INTO labeled (id, label) VALUES (?, ?)",
                (s["id"], s["label"]),
            )
        self.conn.commit()

        count = len(self.session_labels)
        self.session_labels.clear()
        return count

    def get_stats(self):
        category_stats = {}
        for cat in CATEGORIES:
            session = self.session_counts.get(cat, 0)
            total = self.existing_counts.get(cat, 0) + session
            target = self.targets.get(cat, 0)
            full = session >= target
            category_stats[cat] = {
                "session": session,
                "target": target,
                "total": total,
                "full": full,
            }

        domain_stats = {}
        if self.domain_balanced:
            for domain, quota in self.domain_quotas.items():
                current = self.domain_session_counts.get(domain, 0)
                complete = current >= quota
                domain_stats[domain] = {
                    "current": current,
                    "quota": quota,
                    "complete": complete,
                }

        total_session = sum(self.session_counts.values())
        total_target = sum(self.targets.values())
        progress = total_session / total_target if total_target > 0 else 0

        return {
            "categories": category_stats,
            "domains": domain_stats,
            "total_session": total_session,
            "total_target": total_target,
            "progress": progress,
            "all_done": self._all_targets_met(),
            "domain_balanced": self.domain_balanced,
            "session_skipped": self.session_skipped,
            "session_not_clean": self.session_not_clean,
            "search_active": self.search_query is not None,
            "search_query": self.search_query or "",
        }

    def _all_targets_met(self):
        return all(
            self.session_counts.get(cat, 0) >= self.targets[cat]
            for cat in CATEGORIES
        )


def _parse_search_query(query):
    """Parse a search query into a SQL WHERE clause and params.

    Supports:
      - Simple terms: word1 word2 (implicit AND)
      - Explicit AND: term1 AND term2
      - Explicit OR: term1 OR term2
      - Quoted phrases: "exact phrase"
      - NOT: NOT term, -term
      - Mixed: "climate change" AND energy OR solar
    """
    query = query.strip()
    if not query:
        return None, []

    # Tokenize: extract quoted phrases first, then split rest
    tokens = []
    remainder = query
    # Extract quoted phrases
    for match in re.finditer(r'"([^"]+)"', query):
        tokens.append(("PHRASE", match.group(1)))
    remainder = re.sub(r'"[^"]*"', " ", remainder)

    # Split remainder into words and operators
    parts = remainder.split()
    i = 0
    while i < len(parts):
        word = parts[i]
        upper = word.upper()
        if upper == "AND":
            tokens.append(("OP", "AND"))
        elif upper == "OR":
            tokens.append(("OP", "OR"))
        elif upper == "NOT" or word.startswith("-"):
            term = word[1:] if word.startswith("-") else (parts[i + 1] if i + 1 < len(parts) else "")
            if upper == "NOT":
                i += 1
                term = parts[i] if i < len(parts) else ""
            if term:
                tokens.append(("NOT_TERM", term))
        else:
            tokens.append(("TERM", word))
        i += 1

    if not tokens:
        return None, []

    # Build SQL conditions
    conditions = []
    params = []
    pending_op = "AND"  # default operator between terms

    for ttype, tval in tokens:
        if ttype == "OP":
            pending_op = tval
            continue

        if ttype == "NOT_TERM":
            cond = "LOWER(text) NOT LIKE ?"
            # Support wildcard * (e.g. -Migrat* excludes migration, migranten, etc.)
            like_val = tval.lower().replace("*", "%")
            if "%" not in like_val:
                like_val = f"%{like_val}%"
            params.append(like_val)
        else:
            # TERM or PHRASE
            cond = "LOWER(text) LIKE ?"
            # Support wildcard * (e.g. Migrat* matches migration, migranten, etc.)
            like_val = tval.lower().replace("*", "%")
            if "%" not in like_val:
                like_val = f"%{like_val}%"
            params.append(like_val)

        if conditions:
            conditions.append(pending_op)
        conditions.append(cond)
        pending_op = "AND"  # reset to default

    # Build final clause with proper grouping for OR
    # Simple approach: join conditions with their operators
    if len(conditions) == 1:
        return f"({conditions[0]})", params

    clause_parts = [conditions[0]]
    for i in range(1, len(conditions), 2):
        op = conditions[i]
        cond = conditions[i + 1]
        clause_parts.append(f" {op} {cond}")

    return "(" + "".join(clause_parts) + ")", params
