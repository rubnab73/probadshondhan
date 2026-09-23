/**
 * Bangla Situation-to-Proverb Semantic Retrieval System
 * Client-side Controller & UI Interaction Handler
 */

// Global State
const state = {
    activeModel: 'unified',
    proverbs: [],
    samples: [],
    benchmarkData: null
};

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initModelSelector();
    initSearch();
    loadStats();
    loadSamples();
    loadProverbsDatabase();
    loadBenchmarkData();
});

// Tab Navigation
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetEl = document.getElementById(targetTab);
            if (targetEl) {
                targetEl.classList.add('active');
            }
        });
    });
}

// Model Selector
function initModelSelector() {
    const modelBtns = document.querySelectorAll('.pill-btn');
    modelBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modelBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeModel = btn.getAttribute('data-model');
        });
    });
}

// Samples Loading
async function loadSamples() {
    try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        state.samples = data;
        renderSamples();
    } catch (err) {
        console.error('নমুনা পরিস্থিতি লোড করতে ব্যর্থ হয়েছে:', err);
    }
}

function renderSamples() {
    const container = document.getElementById('sample-chips-container');
    if (!container) return;

    container.innerHTML = '';
    state.samples.forEach(sample => {
        const chip = document.createElement('button');
        chip.className = 'sample-chip';
        chip.type = 'button';
        chip.innerHTML = `<span class="chip-dot"></span> <span class="chip-label">${sample.label}</span>`;
        chip.title = sample.situation;
        chip.addEventListener('click', () => {
            const input = document.getElementById('situation-input');
            input.value = sample.situation;
            input.focus();
            
            // Highlight chip temporarily
            document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('chip-selected'));
            chip.classList.add('chip-selected');
        });
        container.appendChild(chip);
    });
}

// Search Proverb Engine
function initSearch() {
    const btn = document.getElementById('btn-find-proverb');
    const input = document.getElementById('situation-input');

    const handleSearch = async () => {
        const text = input.value.trim();
        if (!text) {
            alert('অনুগ্রহ করে বাংলায় কোনো ঘটনা বা পরিস্থিতির বিবরণ লিখুন।');
            input.focus();
            return;
        }

        const spinner = document.getElementById('search-spinner');
        const btnText = btn.querySelector('.btn-text');
        const btnIcon = btn.querySelector('.btn-icon');

        spinner.style.display = 'inline-block';
        btnIcon.style.display = 'none';
        btn.disabled = true;

        try {
            const res = await fetch('/api/retrieve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    situation: text,
                    model: state.activeModel,
                    top_k: 5
                })
            });

            const data = await res.json();
            if (data.error) {
                alert('ত্রুটি: ' + data.error);
            } else {
                renderResults(data);
            }
        } catch (err) {
            console.error('অনুসন্ধানে ত্রুটি:', err);
            alert('সার্ভারে যোগাযোগ করতে সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।');
        } finally {
            spinner.style.display = 'none';
            btnIcon.style.display = 'inline-block';
            btn.disabled = false;
        }
    };

    btn.addEventListener('click', handleSearch);

    // Ctrl + Enter to search
    input.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            handleSearch();
        }
    });
}

function renderResults(data) {
    const container = document.getElementById('results-container');
    const cardsContainer = document.getElementById('results-cards');
    const activeModelName = document.getElementById('active-model-name');

    activeModelName.textContent = data.model;
    cardsContainer.innerHTML = '';

    if (!data.results || data.results.length === 0) {
        cardsContainer.innerHTML = `
            <div class="empty-state">
                <p>কোনো মানানসই প্রবাদ খুঁজে পাওয়া যায়নি। অনুগ্রহ করে আরও বিস্তারিত বিবরণ লিখে পুনরায় চেষ্টা করুন।</p>
            </div>
        `;
        container.style.display = 'block';
        return;
    }

    data.results.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'glass-card proverb-card';

        let rankBadgeClass = 'rank-other';
        let rankLabel = `#${item.rank}`;
        if (item.rank === 1) {
            rankBadgeClass = 'rank-1';
            rankLabel = '🥇 ১ম স্থান (শীর্ষ প্রাসঙ্গিক)';
        } else if (item.rank === 2) {
            rankBadgeClass = 'rank-2';
            rankLabel = '🥈 ২য় স্থান';
        } else if (item.rank === 3) {
            rankBadgeClass = 'rank-3';
            rankLabel = '🥉 ৩য় স্থান';
        } else if (item.rank === 4) {
            rankBadgeClass = 'rank-4';
            rankLabel = '৪র্থ স্থান';
        } else if (item.rank === 5) {
            rankBadgeClass = 'rank-5';
            rankLabel = '৫ম স্থান';
        }

        const rawScore = Number(item.score) || 0;
        const scorePercent = Math.min(Math.max(Math.round(rawScore * 100), 5), 100);

        const exampleBox = item.example
            ? `<div class="card-example-box">
                    <span class="example-label">বাস্তব ঘটনার প্রেক্ষাপট / দৃষ্টান্ত:</span>
                    <p class="example-text">${escapeHtml(item.example)}</p>
               </div>`
            : '';

        const matchInsightText = item.match_type
            ? escapeHtml(item.match_type)
            : 'পরিস্থিতির সাথে গভীর ভাবার্থের মিল নিরূপিত';

        card.innerHTML = `
            <div class="card-top">
                <div class="card-rank-badge ${rankBadgeClass}">
                    <span>${rankLabel}</span>
                </div>
                <div class="score-badge-wrapper">
                    <span class="score-label">ম্যাচিং স্কোর:</span>
                    <span class="score-text">${rawScore.toFixed(3)}</span>
                    <div class="similarity-bar-container" title="সাদৃশ্য মান: ${rawScore.toFixed(4)}">
                        <div class="similarity-bar-fill" style="width: ${scorePercent}%"></div>
                    </div>
                </div>
            </div>

            <div class="card-body">
                <div class="proverb-heading-row">
                    <h3 class="card-proverb-text">${escapeHtml(item.proverb)}</h3>
                    <span class="proverb-id-pill">${escapeHtml(item.proverb_id || '')}</span>
                </div>
                
                <div class="card-meaning-box">
                    <span class="meaning-label">অন্তর্নিহিত অর্থ ও তাৎপর্য:</span>
                    <p class="meaning-text">${escapeHtml(item.meaning)}</p>
                </div>

                ${exampleBox}
            </div>

            <div class="card-footer-row">
                <div class="match-insight">
                    <span class="insight-icon">✨</span>
                    <span class="insight-text">${matchInsightText}</span>
                </div>
                <button class="copy-btn" onclick="copyProverbText('${escapeHtml(item.proverb)}', this)">
                    <span class="copy-icon">📋</span>
                    <span class="copy-label">কপি করুন</span>
                </button>
            </div>
        `;

        cardsContainer.appendChild(card);
    });

    container.style.display = 'block';
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Copy to Clipboard Helper
window.copyProverbText = function(text, btnElement) {
    if (!navigator.clipboard) {
        // Fallback
        const el = document.createElement('textarea');
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
    } else {
        navigator.clipboard.writeText(text);
    }

    if (btnElement) {
        const originalHtml = btnElement.innerHTML;
        btnElement.innerHTML = `<span class="copy-icon">✓</span> <span class="copy-label">কপি হয়েছে!</span>`;
        btnElement.classList.add('copied');
        setTimeout(() => {
            btnElement.innerHTML = originalHtml;
            btnElement.classList.remove('copied');
        }, 1800);
    }
};

// Proverbs Knowledge Base Tab
async function loadProverbsDatabase() {
    try {
        const res = await fetch('/api/proverbs');
        const data = await res.json();
        state.proverbs = data.proverbs || [];

        const countBadge = document.getElementById('db-count-badge');
        const navCount = document.getElementById('nav-proverbs-count');
        if (countBadge) {
            countBadge.textContent = `মোট ${toBengaliNumber(state.proverbs.length)}টি প্রবাদ সংরক্ষিত`;
        }
        if (navCount) {
            navCount.textContent = toBengaliNumber(state.proverbs.length);
        }

        renderProverbsTable(state.proverbs);
        initDatabaseSearch();
    } catch (err) {
        console.error('প্রবাদ ডেটাবেস লোড করতে ত্রুটি:', err);
    }
}

function renderProverbsTable(list) {
    const tbody = document.getElementById('db-table-body');
    const countBadge = document.getElementById('db-count-badge');
    if (!tbody) return;

    if (countBadge) {
        if (list.length === state.proverbs.length) {
            countBadge.textContent = `মোট ${toBengaliNumber(list.length)}টি প্রবাদ সংরক্ষিত`;
        } else {
            countBadge.textContent = `প্রদর্শিত: ${toBengaliNumber(list.length)}টি / মোট ${toBengaliNumber(state.proverbs.length)}টি`;
        }
    }

    tbody.innerHTML = '';
    if (list.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="4" style="text-align:center; padding: 2.5rem; color: var(--text-dim);">
                    কোনো প্রবাদ খুঁজে পাওয়া যায়নি। অন্য শব্দ দিয়ে চেষ্টা করুন।
                </td>
            </tr>
        `;
        return;
    }

    const fragment = document.createDocumentFragment();
    list.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="col-id">${escapeHtml(item.proverb_id)}</td>
            <td class="col-proverb">
                <strong>${escapeHtml(item.proverb)}</strong>
            </td>
            <td class="col-meaning">${escapeHtml(item.meaning)}</td>
            <td class="col-action">
                <button class="test-proverb-btn" title="এই প্রবাদের অর্থ দিয়ে সার্চ করুন" onclick="useProverbInSearch('${escapeHtml(item.meaning)}')">
                    <span>পরীক্ষা ↗</span>
                </button>
            </td>
        `;
        fragment.appendChild(tr);
    });
    tbody.appendChild(fragment);
}

function initDatabaseSearch() {
    const searchInput = document.getElementById('db-search-input');
    if (!searchInput) return;

    let debounceTimer;
    searchInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            const query = searchInput.value.toLowerCase().trim();
            if (!query) {
                renderProverbsTable(state.proverbs);
                return;
            }

            const filtered = state.proverbs.filter(p => {
                const proverbText = (p.proverb || '').toLowerCase();
                const meaningText = (p.meaning || '').toLowerCase();
                const idText = (p.proverb_id || '').toLowerCase();
                return proverbText.includes(query) || meaningText.includes(query) || idText.includes(query);
            });

            renderProverbsTable(filtered);
        }, 150);
    });
}

// Transfer proverb meaning into main search
window.useProverbInSearch = function(meaning) {
    const searchTabBtn = document.getElementById('btn-tab-search');
    const input = document.getElementById('situation-input');

    if (searchTabBtn) {
        searchTabBtn.click();
    }
    if (input) {
        input.value = meaning;
        input.focus();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
};

// Benchmark Leaderboard Tab
async function loadBenchmarkData() {
    try {
        const res = await fetch('/api/benchmark');
        const data = await res.json();
        state.benchmarkData = data;
        renderBenchmark(data);
    } catch (err) {
        console.error('বেঞ্চমার্ক ডেটা লোড করতে ত্রুটি:', err);
    }
}

// Load Dataset Statistics
async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        const statEl = document.getElementById('bm-corpus-count');
        if (statEl && data.total_situations) {
            statEl.textContent = `${toBengaliNumber(data.total_situations)}+`;
        }
        const badgeEl = document.getElementById('hero-training-badge');
        if (badgeEl && data.total_situations) {
            badgeEl.textContent = `BanglaBERT Neural & TF-IDF Ensemble Engine • ${toBengaliNumber(data.total_situations)}+ পরিস্থিতি প্রশিক্ষিত`;
        }
    } catch (err) {
        console.error('পরিসংখ্যান লোড করতে সমস্যা:', err);
    }
}

function renderBenchmark(data) {
    // 1. Update top highlight cards if present in benchmark
    if (data.dev_split && data.dev_split['Unified Proverb Engine']) {
        const uDev = data.dev_split['Unified Proverb Engine'];
        const devTop5El = document.getElementById('bm-top5-dev');
        const devTop1El = document.getElementById('bm-top1-dev');
        if (devTop5El) devTop5El.textContent = `${toBengaliNumber((uDev.top_5 * 100).toFixed(1))}%`;
        if (devTop1El) devTop1El.textContent = `${toBengaliNumber((uDev.top_1 * 100).toFixed(1))}%`;
    }

    if (data.human_test && data.human_test['Unified Proverb Engine']) {
        const uHuman = data.human_test['Unified Proverb Engine'];
        const humanTop5El = document.getElementById('bm-top5-human');
        if (humanTop5El) humanTop5El.textContent = `${toBengaliNumber((uHuman.top_5 * 100).toFixed(1))}%`;
    }

    // 2. Render Human Test Table
    const humanBody = document.getElementById('benchmark-human-body');
    if (humanBody && data.human_test) {
        humanBody.innerHTML = '';
        Object.entries(data.human_test).forEach(([method, m]) => {
            const isWinner = method.includes('Unified');
            const tr = document.createElement('tr');
            if (isWinner) tr.className = 'winner-row';
            tr.innerHTML = `
                <td class="method-cell">
                    <strong>${method}</strong>
                    ${isWinner ? '<span class="sota-tag">SOTA</span>' : ''}
                </td>
                <td class="${m.top_1 >= 0.55 ? 'score-highlight' : ''}">${toBengaliNumber((m.top_1 * 100).toFixed(1))}%</td>
                <td>${toBengaliNumber((m.top_3 * 100).toFixed(1))}%</td>
                <td class="${m.top_5 >= 0.85 ? 'score-winner' : ''}">${toBengaliNumber((m.top_5 * 100).toFixed(1))}%</td>
                <td>${toBengaliNumber(m.mrr.toFixed(4))}</td>
                <td>${toBengaliNumber(m.ndcg_5.toFixed(4))}</td>
            `;
            humanBody.appendChild(tr);
        });
    }

    // 3. Render Dev Split Table
    const devBody = document.getElementById('benchmark-dev-body');
    if (devBody && data.dev_split) {
        devBody.innerHTML = '';
        Object.entries(data.dev_split).forEach(([method, m]) => {
            const isWinner = method.includes('Unified');
            const tr = document.createElement('tr');
            if (isWinner) tr.className = 'winner-row';
            tr.innerHTML = `
                <td class="method-cell">
                    <strong>${method}</strong>
                    ${isWinner ? '<span class="sota-tag">SOTA</span>' : ''}
                </td>
                <td class="${m.top_1 >= 0.65 ? 'score-highlight' : ''}">${toBengaliNumber((m.top_1 * 100).toFixed(1))}%</td>
                <td>${toBengaliNumber((m.top_3 * 100).toFixed(1))}%</td>
                <td class="${m.top_5 >= 0.88 ? 'score-winner' : ''}">${toBengaliNumber((m.top_5 * 100).toFixed(1))}%</td>
                <td>${toBengaliNumber(m.mrr.toFixed(4))}</td>
                <td>${toBengaliNumber(m.ndcg_5.toFixed(4))}</td>
            `;
            devBody.appendChild(tr);
        });
    }
}

// Helpers
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function toBengaliNumber(n) {
    const banglaDigits = ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯'];
    return String(n).replace(/\d/g, d => banglaDigits[d]);
}
