import csv
import os
import sqlite3
import sys

import pandas as pd

# Paths relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "articles")
CSV_PATH = os.path.join(DATA_DIR, "cleaned_articles.csv")
LABELED_CSV_PATH = os.path.join(DATA_DIR, "cleaned_articles_labeled.csv")
DB_PATH = os.path.join(DATA_DIR, "articles.db")

COLUMNS = ["id", "domain", "url", "date_time", "headline", "author", "text", "text_length"]


def ensure_database():
    """Create SQLite DB from CSV if it doesn't exist or is older than the CSV."""
    if os.path.exists(DB_PATH):
        if os.path.getmtime(DB_PATH) >= os.path.getmtime(CSV_PATH):
            print("SQLite database is up to date.")
            return
        else:
            print("CSV is newer than database, re-importing...")

    print("Importing CSV into SQLite (one-time operation)...")
    csv.field_size_limit(sys.maxsize)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS articles")
    conn.execute("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            domain TEXT NOT NULL,
            url TEXT,
            date_time TEXT,
            headline TEXT,
            author TEXT,
            text TEXT,
            text_length INTEGER
        )
    """)

    chunks = pd.read_csv(
        CSV_PATH,
        chunksize=50000,
        encoding="utf-8",
        dtype={"id": int, "text_length": "Int64"},
    )
    total = 0
    for chunk in chunks:
        chunk.to_sql("articles", conn, if_exists="append", index=False)
        total += len(chunk)
        print(f"  Imported {total} articles...")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON articles(domain)")
    conn.commit()
    conn.close()
    print(f"Database ready. Total articles: {total}")


def get_connection():
    """Return a new SQLite connection to the articles database.

    Uses WAL mode for better concurrent read/write behaviour and
    serialized thread mode so a single connection can be safely
    shared across Flask's threaded request handling.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def load_labeled_ids(conn):
    """Load already-labeled IDs from the labeled CSV into the database.

    Returns the count of previously labeled articles and a dict of {category: count}.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS labeled (id INTEGER PRIMARY KEY, label TEXT)")

    if os.path.exists(LABELED_CSV_PATH):
        existing = pd.read_csv(LABELED_CSV_PATH, usecols=["id", "label"], encoding="utf-8")
        # Clear and re-populate
        conn.execute("DELETE FROM labeled")
        for _, row in existing.iterrows():
            conn.execute(
                "INSERT OR IGNORE INTO labeled (id, label) VALUES (?, ?)",
                (int(row["id"]), row["label"]),
            )
        conn.commit()

        # Count per category
        rows = conn.execute("SELECT label, COUNT(*) FROM labeled GROUP BY label").fetchall()
        category_counts = {r[0]: r[1] for r in rows}
        # Exclude 'skipped' and 'not_clean' from the total count
        total = conn.execute(
            "SELECT COUNT(*) FROM labeled WHERE label NOT IN ('skipped', 'not_clean')"
        ).fetchone()[0]
        return total, category_counts

    return 0, {}


def get_domain_distribution(conn):
    """Return {domain: count} for unlabeled articles."""
    rows = conn.execute(
        "SELECT domain, COUNT(*) FROM articles "
        "WHERE id NOT IN (SELECT id FROM labeled) "
        "GROUP BY domain"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def get_total_domain_distribution(conn):
    """Return {domain: count} for ALL articles (used for proportion calculation)."""
    rows = conn.execute(
        "SELECT domain, COUNT(*) FROM articles GROUP BY domain"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


if __name__ == "__main__":
    ensure_database()
    conn = get_connection()
    total, cats = load_labeled_ids(conn)
    print(f"Already labeled: {total}")
    if cats:
        for cat, count in sorted(cats.items()):
            print(f"  {cat}: {count}")
    dist = get_total_domain_distribution(conn)
    print(f"\nDomain distribution ({len(dist)} domains):")
    for domain, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")
    conn.close()
