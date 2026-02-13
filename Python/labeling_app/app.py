import csv
import sys

from flask import Flask, jsonify, render_template, request

csv.field_size_limit(sys.maxsize)

from data_loader import ensure_database, get_connection, load_labeled_ids
from labeling_engine import CATEGORIES, LabelingSession

app = Flask(__name__)

# Global session state (single-user local app)
session_obj = None
db_conn = None
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
    return render_template(
        "index.html",
        already_labeled=total_already_labeled,
        categories=CATEGORIES,
    )


@app.route("/start-session", methods=["POST"])
def start_session():
    global session_obj
    config = request.json
    session_obj = LabelingSession(db_conn, config, existing_category_counts)
    return jsonify({"status": "ok"})


@app.route("/label")
def label_page():
    if session_obj is None:
        return render_template(
            "index.html",
            already_labeled=total_already_labeled,
            categories=CATEGORIES,
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
    article = session_obj.next_article()
    if article is None:
        return jsonify({"done": True})
    return jsonify(article)


@app.route("/api/label", methods=["POST"])
def label_article():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    data = request.json
    stats = session_obj.apply_label(data["article_id"], data["label"])
    next_art = session_obj.next_article()
    done = next_art is None or stats["all_done"]
    return jsonify({"stats": stats, "next_article": next_art, "done": done})


@app.route("/api/skip", methods=["POST"])
def skip_article():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    data = request.json
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
    stats = session_obj.mark_not_clean(data["article_id"])
    next_art = session_obj.next_article()
    if next_art is None:
        return jsonify({"done": True, "stats": stats})
    return jsonify({"next_article": next_art, "stats": stats})


@app.route("/api/redo", methods=["POST"])
def redo():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    result = session_obj.undo_last()
    return jsonify(result)


@app.route("/api/search", methods=["POST"])
def search():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    data = request.json
    query = data.get("query", "").strip()
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
    count = session_obj.save_to_csv()
    total_already_labeled, existing_category_counts = load_labeled_ids(db_conn)
    return jsonify({"status": "saved", "count": count, "total": total_already_labeled})


@app.route("/api/stats")
def stats():
    if session_obj is None:
        return jsonify({"error": "No session active"}), 400
    return jsonify(session_obj.get_stats())


if __name__ == "__main__":
    init_app()
    app.run(debug=True, port=5000)
