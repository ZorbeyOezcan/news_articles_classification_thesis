let currentArticleId = null;
let currentItem = null;

async function init() {
    // Set source badge
    const badge = document.getElementById('source-badge');
    if (CLEANING_SOURCE === 'umap') {
        badge.textContent = 'UMAP Outlier';
        badge.classList.add('badge-umap');
    } else {
        badge.textContent = 'Cleanlab';
        badge.classList.add('badge-cleanlab');
    }

    // Fetch first article
    const resp = await fetch('/api/clean/article');
    const data = await resp.json();

    if (data.done) {
        showDone(data.stats);
    } else if (data.article) {
        renderItem(data.article);
        updateStats(data.stats);
    }
}

function renderItem(item) {
    if (!item) return;
    currentItem = item;
    currentArticleId = item.id;

    // Article content
    document.getElementById('article-headline').textContent = item.headline || '(No headline)';
    document.getElementById('article-domain').textContent = item.domain || '';
    document.getElementById('article-text').textContent = item.text || '';
    document.getElementById('article-content').scrollTop = 0;

    // Quality signals
    renderSignals(item);

    // Category grid coloring
    colorGrid(item);
}

function renderSignals(item) {
    const signals = item.quality_signals;

    // Current label
    document.getElementById('signal-current-value').textContent = item.current_label;

    // Source
    document.getElementById('signal-source-value').textContent =
        signals.source === 'umap' ? 'UMAP Outlier' : 'Cleanlab';

    // Details (source-specific)
    const details = document.getElementById('signal-details');
    details.innerHTML = '';

    if (signals.source === 'umap') {
        details.innerHTML =
            '<div class="signal-card">' +
                '<span class="signal-label">Centroid Distance</span>' +
                '<span class="signal-value">' + signals.centroid_dist + '</span>' +
            '</div>' +
            '<div class="signal-card">' +
                '<span class="signal-label">Detection Method</span>' +
                '<span class="signal-value">' + signals.methods.join(' + ') + '</span>' +
            '</div>';
    } else if (signals.source === 'cleanlab') {
        const score = signals.label_quality_score;
        let scoreClass = 'score-high';
        if (score < 0.3) scoreClass = 'score-low';
        else if (score < 0.6) scoreClass = 'score-mid';

        details.innerHTML =
            '<div class="signal-card">' +
                '<span class="signal-label">Quality Score</span>' +
                '<span class="signal-value ' + scoreClass + '">' + score.toFixed(3) + '</span>' +
                '<div class="score-bar"><div class="score-bar-fill" style="width:' + (score * 100) + '%"></div></div>' +
            '</div>' +
            '<div class="signal-card">' +
                '<span class="signal-label">Predicted Label</span>' +
                '<span class="signal-value signal-predicted">' + escapeHtml(signals.predicted_label) + '</span>' +
            '</div>';
    }
}

function colorGrid(item) {
    const suggestions = item.suggestions || {};
    const currentLabel = item.current_label;
    const hasSuggestions = Object.keys(suggestions).length > 0;

    CATEGORIES.forEach((cat, idx) => {
        const box = document.querySelector('.category-box[data-category="' + cat + '"]');
        const confEl = document.getElementById('conf-' + idx);
        if (!box) return;

        // Reset classes
        box.classList.remove('current-label', 'suggested', 'conf-high', 'conf-mid', 'conf-low');
        box.style.opacity = '';

        if (confEl) confEl.textContent = '';

        if (cat === currentLabel) {
            // Current label: grayed out
            box.classList.add('current-label');
        } else if (hasSuggestions) {
            const prob = suggestions[cat] || 0;

            // Show confidence percentage
            if (confEl) {
                confEl.textContent = (prob * 100).toFixed(0) + '%';
            }

            // Predicted label (highest confidence that isn't current)
            const signals = item.quality_signals;
            if (signals.source === 'cleanlab' && signals.predicted_label === cat) {
                box.classList.add('suggested');
            }

            // Opacity based on confidence
            if (prob >= 0.3) {
                box.classList.add('conf-high');
            } else if (prob >= 0.1) {
                box.classList.add('conf-mid');
            } else {
                box.classList.add('conf-low');
            }
        }
    });
}

function updateStats(stats) {
    if (!stats) return;

    const pct = Math.round(stats.progress * 100);
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-text').textContent =
        stats.reviewed + ' / ' + stats.total_flagged + ' (' + pct + '%)';

    document.getElementById('relabeled-count').textContent = 'Relabeled: ' + stats.relabeled;
    document.getElementById('kept-count').textContent = 'Kept: ' + stats.kept;
    document.getElementById('removed-count').textContent = 'Removed: ' + stats.removed;
    document.getElementById('skipped-count').textContent = 'Skipped: ' + stats.skipped;
}

async function onRelabel(label) {
    if (currentArticleId === null) return;

    // Don't relabel to the same label (use keep instead)
    if (currentItem && label === currentItem.current_label) {
        onKeep();
        return;
    }

    const resp = await fetch('/api/clean/relabel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ article_id: currentArticleId, new_label: label })
    });

    const data = await resp.json();
    updateStats(data.stats);

    if (data.next_article) {
        renderItem(data.next_article);
    } else {
        showDone(data.stats);
    }
}

async function onKeep() {
    if (currentArticleId === null) return;

    const resp = await fetch('/api/clean/keep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ article_id: currentArticleId })
    });

    const data = await resp.json();
    updateStats(data.stats);

    if (data.next_article) {
        renderItem(data.next_article);
    } else {
        showDone(data.stats);
    }
}

async function onSkip() {
    if (currentArticleId === null) return;

    const resp = await fetch('/api/clean/skip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ article_id: currentArticleId })
    });

    const data = await resp.json();
    updateStats(data.stats);

    if (data.next_article) {
        renderItem(data.next_article);
    } else {
        showDone(data.stats);
    }
}

async function onRemove() {
    if (currentArticleId === null) return;

    const resp = await fetch('/api/clean/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ article_id: currentArticleId })
    });

    const data = await resp.json();
    updateStats(data.stats);

    if (data.next_article) {
        renderItem(data.next_article);
    } else {
        showDone(data.stats);
    }
}

async function onUndo() {
    const resp = await fetch('/api/clean/undo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });

    const data = await resp.json();
    if (data.error) {
        alert(data.error);
        return;
    }

    if (data.article) {
        renderItem(data.article);
    }
    if (data.stats) {
        updateStats(data.stats);
    }
}

async function onSave() {
    if (!confirm('Save all changes and exit?')) return;
    await saveAndExit();
}

async function saveAndExit() {
    const resp = await fetch('/api/clean/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });

    const data = await resp.json();
    alert('Saved ' + data.changes + ' changes. Total labeled: ' + data.total);
    window.location.href = '/';
}

function showDone(stats) {
    const summary = document.getElementById('done-summary');
    if (stats) {
        summary.textContent =
            'Relabeled: ' + stats.relabeled +
            ' | Kept: ' + stats.kept +
            ' | Removed: ' + stats.removed +
            ' | Skipped: ' + stats.skipped;
    }
    document.getElementById('done-overlay').style.display = 'flex';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== Keyboard Shortcuts =====

document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const keyMap = {
        '1': 0, '2': 1, '3': 2, '4': 3,
        '5': 4, '6': 5, '7': 6, '8': 7,
        '9': 8, '0': 9, '-': 10, '=': 11
    };

    if (e.key in keyMap) {
        const idx = keyMap[e.key];
        if (idx < CATEGORIES.length) {
            onRelabel(CATEGORIES[idx]);
        }
        e.preventDefault();
    } else if (e.key === 'a' || e.key === 'A') {
        onRelabel('Andere');
        e.preventDefault();
    } else if (e.key === 'k' || e.key === 'K') {
        onKeep();
        e.preventDefault();
    } else if (e.key === 's' || e.key === 'S') {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            onSave();
        } else {
            onSkip();
            e.preventDefault();
        }
    } else if (e.key === 'r' || e.key === 'R') {
        onRemove();
        e.preventDefault();
    } else if (e.key === 'z' || e.key === 'Z') {
        onUndo();
        e.preventDefault();
    }
});

document.addEventListener('DOMContentLoaded', function () {
    init();
});
