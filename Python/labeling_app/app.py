import csv
import os
import sys
import threading

from flask import Flask, jsonify, render_template, request

csv.field_size_limit(sys.maxsize)

from data_loader import ensure_database, get_connection, load_labeled_ids
from labeling_engine import CATEGORIES, LabelingSession
from cleaning_engine import CleaningSession, get_available_sources

app = Flask(__name__)

# Global session state (single-user local app)
session_obj = None
cleaning_session = None
db_conn = None
db_lock = threading.Lock()
existing_category_counts = {}
total_already_labeled = 0


def init_app():
    global db_conn, total_already_labeled, existing_category_counts
    ensure_database()
    db_conn = get_connection()
    total_already_labeled, existing_category_counts = load_labeled_ids(db_conn)
    print(f"Already labeled: {total_already_labeled} articles")


@app.route("/")
def index():
    sources = get_available_sources()
    return render_template(
        "index.html",
        already_labeled=total_already_labeled,
        categories=CATEGORIES,
        category_counts=existing_category_counts,
        clean_sources=sources,
    )


@app.route("/start-session", methods=["POST"])
def start_session():
    global session_obj
    config = request.json
    with db_lock:
        session_obj = LabelingSession(db_conn, config, existing_category_counts)
    return jsonify({"status": "ok"})


@app.route("/label")
def label_page():
    if session_obj is None:
        return render_template(
            "index.html",
            already_labeled=total_already_labeled,
            categories=CATEGORIES,
            category_counts=existing_category_counts,
        )
    return render_template(
        "labeling.html",
        categories=CATEGORIES,
        domain_balanced=session_obj.domain_balanced,
    )


@app.route("/api/article")
def get_article():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    with db_lock:
        stats = session_obj.get_stats()
        if stats["all_done"]:
            return jsonify({"done": True, "all_done": True})
        article = session_obj.next_article()
        if article is None:
            return jsonify({"done": True, "all_done": False})
        return jsonify(article)


@app.route("/api/label", methods=["POST"])
def label_article():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    data = request.json
    with db_lock:
        stats = session_obj.apply_label(data["article_id"], data["label"])
        next_art = session_obj.next_article()
    all_done = stats["all_done"]
    no_more = next_art is None and not all_done
    return jsonify({"stats": stats, "next_article": next_art, "done": all_done, "no_more": no_more})


@app.route("/api/skip", methods=["POST"])
def skip_article():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    data = request.json
    with db_lock:
        stats = session_obj.skip(data["article_id"])
        next_art = session_obj.next_article()
    if next_art is None:
        return jsonify({"done": True, "stats": stats})
    return jsonify({"next_article": next_art, "stats": stats})


@app.route("/api/not-clean", methods=["POST"])
def not_clean():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    data = request.json
    with db_lock:
        stats = session_obj.mark_not_clean(data["article_id"])
        next_art = session_obj.next_article()
    if next_art is None:
        return jsonify({"done": True, "stats": stats})
    return jsonify({"next_article": next_art, "stats": stats})


@app.route("/api/redo", methods=["POST"])
def redo():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    with db_lock:
        result = session_obj.undo_last()
    return jsonify(result)


@app.route("/api/search", methods=["POST"])
def search():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    data = request.json
    query = data.get("query", "").strip()
    with db_lock:
        session_obj.set_search(query if query else None)
        article = session_obj.next_article()
        stats = session_obj.get_stats()
    if article is None:
        return jsonify({"done": True, "stats": stats, "no_results": True})
    return jsonify({"next_article": article, "stats": stats})


@app.route("/api/save", methods=["POST"])
def save():
    global total_already_labeled, existing_category_counts
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    with db_lock:
        count = session_obj.save_to_csv()
        total_already_labeled, existing_category_counts = load_labeled_ids(db_conn)
    return jsonify({"status": "saved", "count": count, "total": total_already_labeled})


@app.route("/api/emergency-save", methods=["POST"])
def emergency_save():
    """Save session labels to CSV without querying the DB (for DB corruption recovery)."""
    import json
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    labels = session_obj.session_labels
    if not labels:
        return jsonify({"status": "nothing to save", "count": 0})
    recovery_path = os.path.join(os.path.dirname(__file__), "recovered_labels.json")
    with open(recovery_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "saved", "count": len(labels), "path": recovery_path})


@app.route("/api/stats")
def stats():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    with db_lock:
        return jsonify(session_obj.get_stats())


# ===== Cleaning Mode Routes =====


@app.route("/start-cleaning", methods=["POST"])
def start_cleaning():
    global cleaning_session
    config = request.json
    with db_lock:
        cleaning_session = CleaningSession(db_conn, config)
    return jsonify({"status": "ok", "total_flagged": cleaning_session.total})


@app.route("/clean")
def clean_page():
    if cleaning_session is None:
        return render_template(
            "index.html",
            already_labeled=total_already_labeled,
            categories=CATEGORIES,
            category_counts=existing_category_counts,
            clean_sources=get_available_sources(),
        )
    return render_template(
        "cleaning.html",
        categories=CATEGORIES,
        source=cleaning_session.source_type,
    )


@app.route("/api/clean/article")
def get_clean_article():
    if cleaning_session is None:
        return jsonify({"error": "No cleaning session active"}), 400
    with db_lock:
        stats = cleaning_session.get_stats()
        if stats["all_done"]:
            return jsonify({"done": True, "stats": stats})
        article = cleaning_session.next_article()
        if article is None:
            return jsonify({"done": True, "stats": stats})
        return jsonify({"article": article, "stats": stats})


@app.route("/api/clean/relabel", methods=["POST"])
def clean_relabel():
    if cleaning_session is None:
        return jsonify({"error": "No cleaning session active"}), 400
    data = request.json
    with db_lock:
        stats = cleaning_session.relabel(data["article_id"], data["new_label"])
        next_art = cleaning_session.next_article()
    return jsonify({"stats": stats, "next_article": next_art})


@app.route("/api/clean/keep", methods=["POST"])
def clean_keep():
    if cleaning_session is None:
        return jsonify({"error": "No cleaning session active"}), 400
    data = request.json
    with db_lock:
        stats = cleaning_session.keep(data["article_id"])
        next_art = cleaning_session.next_article()
    return jsonify({"stats": stats, "next_article": next_art})


@app.route("/api/clean/skip", methods=["POST"])
def clean_skip():
    if cleaning_session is None:
        return jsonify({"error": "No cleaning session active"}), 400
    data = request.json
    with db_lock:
        stats = cleaning_session.skip(data["article_id"])
        next_art = cleaning_session.next_article()
    return jsonify({"stats": stats, "next_article": next_art})


@app.route("/api/clean/remove", methods=["POST"])
def clean_remove():
    if cleaning_session is None:
        return jsonify({"error": "No cleaning session active"}), 400
    data = request.json
    with db_lock:
        stats = cleaning_session.remove_label(data["article_id"])
        next_art = cleaning_session.next_article()
    return jsonify({"stats": stats, "next_article": next_art})


@app.route("/api/clean/undo", methods=["POST"])
def clean_undo():
    if cleaning_session is None:
        return jsonify({"error": "No cleaning session active"}), 400
    with db_lock:
        result = cleaning_session.undo_last()
    return jsonify(result)


@app.route("/api/clean/save", methods=["POST"])
def clean_save():
    global total_already_labeled, existing_category_counts
    if cleaning_session is None:
        return jsonify({"error": "No cleaning session active"}), 400
    with db_lock:
        n_changes = cleaning_session.save_to_csv()
        total_already_labeled, existing_category_counts = load_labeled_ids(db_conn)
    return jsonify({"status": "saved", "changes": n_changes, "total": total_already_labeled})


if __name__ == "__main__":
    init_app()
    app.run(debug=True, port=5000)
