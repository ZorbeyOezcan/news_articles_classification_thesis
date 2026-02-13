let currentArticleId = null;
let searchMode = false;

async function init() {
    const statsResp = await fetch('/api/stats');
    const stats = await statsResp.json();
    updateStats(stats);

    const artResp = await fetch('/api/article');
    const article = await artResp.json();
    if (article.done) {
        showCongratulations(stats.total_session);
    } else {
        renderArticle(article);
    }
}

function renderArticle(article) {
    if (!article) return;
    currentArticleId = article.id;
    document.getElementById('article-headline').textContent = article.headline || '(No headline)';
    document.getElementById('article-domain').textContent = article.domain || '';
    document.getElementById('article-text').textContent = article.text || '';
    document.getElementById('article-content').scrollTop = 0;
}

function updateStats(stats) {
    // Progress bar
    const pct = Math.round(stats.progress * 100);
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-text').textContent =
        stats.total_session + ' / ' + stats.total_target + ' (' + pct + '%)';

    // Special label counts
    document.getElementById('skipped-count').textContent = 'Skipped: ' + (stats.session_skipped || 0);
    document.getElementById('not-clean-count').textContent = 'Not clean: ' + (stats.session_not_clean || 0);

    // Category boxes
    CATEGORIES.forEach((cat, idx) => {
        const catStat = stats.categories[cat];
        if (!catStat) return;

        const sessionEl = document.getElementById('stat-session-' + idx);
        const totalEl = document.getElementById('stat-total-' + idx);

        if (sessionEl) {
            sessionEl.textContent = catStat.session + ' of ' + catStat.target + ' labelled';
        }
        if (totalEl) {
            totalEl.textContent = 'total: ' + catStat.total;
        }

        const box = document.querySelector('.category-box[data-category="' + cat + '"]');
        if (box) {
            if (catStat.full) {
                box.classList.add('full');
            } else {
                box.classList.remove('full');
            }
        }
    });

    // Domain sidebar
    if (DOMAIN_BALANCED && stats.domains) {
        const domainList = document.getElementById('domain-list');
        if (domainList) {
            const sortedDomains = Object.entries(stats.domains).sort((a, b) => {
                if (a[1].complete !== b[1].complete) return a[1].complete ? 1 : -1;
                return a[0].localeCompare(b[0]);
            });

            domainList.innerHTML = '';
            for (const [domain, dstat] of sortedDomains) {
                if (dstat.quota === 0) continue;
                const pctDomain = Math.min(100, Math.round((dstat.current / dstat.quota) * 100));
                const item = document.createElement('div');
                item.className = 'domain-item' + (dstat.complete ? ' complete' : '');
                item.innerHTML =
                    '<span class="domain-name">' + escapeHtml(domain) + '</span>' +
                    '<span class="domain-progress">' +
                    (dstat.complete
                        ? 'complete'
                        : dstat.current + ' of ' + dstat.quota) +
                    '</span>' +
                    '<div class="domain-bar"><div class="domain-bar-fill" style="width:' + pctDomain + '%"></div></div>';
                domainList.appendChild(item);
            }
        }
    }
}

async function onCategoryClick(label) {
    if (currentArticleId === null) return;

    const box = document.querySelector('.category-box[data-category="' + label + '"]');
    if (box && box.classList.contains('full')) return;

    const resp = await fetch('/api/label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ article_id: currentArticleId, label: label })
    });

    const data = await resp.json();
    updateStats(data.stats);

    if (data.done) {
        showCongratulations(data.stats.total_session);
    } else if (data.next_article) {
        renderArticle(data.next_article);
    }
}

async function onSkip() {
    if (currentArticleId === null) return;

    const resp = await fetch('/api/skip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ article_id: currentArticleId })
    });

    const data = await resp.json();
    if (data.stats) updateStats(data.stats);
    if (data.done) {
        alert('No more articles available to label.');
    } else if (data.next_article) {
        renderArticle(data.next_article);
    }
}

async function onNotClean() {
    if (currentArticleId === null) return;

    const resp = await fetch('/api/not-clean', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ article_id: currentArticleId })
    });

    const data = await resp.json();
    if (data.stats) updateStats(data.stats);
    if (data.done) {
        alert('No more articles available to label.');
    } else if (data.next_article) {
        renderArticle(data.next_article);
    }
}

async function onRedo() {
    const resp = await fetch('/api/redo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });

    const data = await resp.json();
    if (data.error) {
        alert(data.error);
        return;
    }

    renderArticle(data.article);
    updateStats(data.stats);
}

async function onSave() {
    if (!confirm('Save all progress and exit?')) return;
    await saveAndExit();
}

async function saveAndExit() {
    const resp = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });

    const data = await resp.json();
    alert('Saved ' + data.count + ' labels. Total labeled: ' + data.total);
    window.location.href = '/';
}

function showCongratulations(count) {
    document.getElementById('congrats-count').textContent =
        count + ' articles labeled in this session.';
    document.getElementById('congrats-overlay').style.display = 'flex';
}

// ===== Search Mode =====

function toggleSearchMode() {
    searchMode = true;
    document.getElementById('btn-search').classList.add('active');
    document.getElementById('btn-random').classList.remove('active');
    document.getElementById('search-bar').style.display = 'flex';
    document.getElementById('search-input').focus();
}

async function toggleRandomMode() {
    searchMode = false;
    document.getElementById('btn-random').classList.add('active');
    document.getElementById('btn-search').classList.remove('active');
    document.getElementById('search-bar').style.display = 'none';
    document.getElementById('search-input').value = '';

    // Clear search filter on backend and get next random article
    const resp = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: '' })
    });

    const data = await resp.json();
    if (data.stats) updateStats(data.stats);
    if (data.next_article) {
        renderArticle(data.next_article);
    }
}

async function onSearch() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) return;

    const resp = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
    });

    const data = await resp.json();
    if (data.stats) updateStats(data.stats);
    if (data.no_results) {
        document.getElementById('article-headline').textContent = 'No results found';
        document.getElementById('article-domain').textContent = '';
        document.getElementById('article-text').textContent = 'No articles match your search query: "' + query + '". Try different terms.';
        currentArticleId = null;
    } else if (data.next_article) {
        renderArticle(data.next_article);
    }
}

// ===== Utilities =====

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== Keyboard Shortcuts =====

document.addEventListener('keydown', function (e) {
    // Don't trigger in input fields
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const keyMap = {
        '1': 0, '2': 1, '3': 2, '4': 3,
        '5': 4, '6': 5, '7': 6, '8': 7,
        '9': 8, '0': 9, '-': 10, '=': 11
    };

    if (e.key in keyMap) {
        const idx = keyMap[e.key];
        if (idx < CATEGORIES.length) {
            onCategoryClick(CATEGORIES[idx]);
        }
        e.preventDefault();
    } else if (e.key === 'a' || e.key === 'A') {
        onCategoryClick('Andere');
        e.preventDefault();
    } else if (e.key === 's' || e.key === 'S') {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            onSave();
        } else {
            onSkip();
            e.preventDefault();
        }
    } else if (e.key === 'z' || e.key === 'Z') {
        onRedo();
        e.preventDefault();
    } else if (e.key === 'c' || e.key === 'C') {
        // 'c' for cleaning needed
        onNotClean();
        e.preventDefault();
    } else if (e.key === 'f' || e.key === 'F') {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            toggleSearchMode();
        }
    } else if (e.key === 'Escape') {
        if (searchMode) {
            toggleRandomMode();
            e.preventDefault();
        }
    }
});

// Enter key triggers search when in search input
document.addEventListener('DOMContentLoaded', function () {
    init();

    document.getElementById('search-input').addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            onSearch();
        }
    });
});
