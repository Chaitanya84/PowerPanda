/*
 * PowerPanda™
 * © 2026 Chaitanya Priya. All rights reserved.
 */
/* ── State ───────────────────────────────────────────────────── */
let selectedFiles = [];



/* ── Boot ────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
  loadEmbeddedFiles();
  setupDrop();
});

/* ── Status Dot ──────────────────────────────────────────────── */
function setStatus(state, text) {
  const dot  = document.getElementById('statusDot');
  const span = document.getElementById('statusText');
  dot.className = `status-dot ${state}`;
  span.textContent = text;
}

/* ── Drag & Drop ─────────────────────────────────────────────── */
function setupDrop() {
  const zone = document.getElementById('dropZone');
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    handleFileSelect(e.dataTransfer.files);
  });
  zone.addEventListener('click', () => document.getElementById('fileInput').click());
}

/* ── File Selection ──────────────────────────────────────────── */
function handleFileSelect(files) {
  for (const f of files) selectedFiles.push(f);
  renderFileList();
}

function renderFileList() {
  const list = document.getElementById('fileList');
  const btn  = document.getElementById('processBtn');
  if (!selectedFiles.length) { list.classList.add('hidden'); btn.disabled = true; return; }
  list.classList.remove('hidden');
  btn.disabled = false;
  list.innerHTML = selectedFiles.map((f, i) => `
    <div class="file-item">
      <span class="file-name">${f.name}</span>
      <span class="file-size">${(f.size/1024).toFixed(1)} KB</span>
      <button class="remove-btn" onclick="removeFile(${i})">✕</button>
    </div>`).join('');
}

function removeFile(i) {
  selectedFiles.splice(i, 1);
  renderFileList();
}

/* ── Process Files ───────────────────────────────────────────── */
async function processFiles() {
  const graphName = document.getElementById('graphName').value.trim();
  const statusEl  = document.getElementById('uploadStatus');

  if (!selectedFiles.length) return;

  setStatus('busy', 'Processing…');
  document.getElementById('processBtn').disabled = true;

  const form = new FormData();
  form.append('graph_name', graphName);
  selectedFiles.forEach(f => form.append('files', f));

  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Upload failed');

    const lines = data.results.map(r =>
      `${r.status === 'processed' ? '✅' : '⏭'} ${r.file} — ${r.status}`
    ).join('\n');
    showMsg(statusEl, 'ok', lines);
    setStatus('success', 'Done');
    selectedFiles = [];
    renderFileList();
    await loadEmbeddedFiles();
    } catch (err) {
    showMsg(statusEl, 'err', err.message);
    setStatus('error', 'Error');
  }
}

/* ── Embedded Files ──────────────────────────────────────────── */
async function loadEmbeddedFiles() {
  const el = document.getElementById('embeddedList');
  try {
    const res  = await fetch('/api/files');
    const data = await res.json();
    const keys = Object.keys(data);
    if (!keys.length) {
      el.innerHTML = '<p class="muted">No files embedded yet.</p>';
    } else {
      el.innerHTML = keys.map(k =>
        `<div class="embedded-item">${k}</div>`
      ).join('');
    }
  } catch { el.innerHTML = '<p class="muted">Could not load files.</p>'; }
}



/* ── Query ───────────────────────────────────────────────────── */
async function runQuery() {
  const query     = document.getElementById('queryInput').value.trim();
  const graphName = document.getElementById('graphName').value.trim();
  const topKDocs  = +document.getElementById('topKDocs').value;
  const topKNodes = +document.getElementById('topKNodes').value;
  const hops      = +document.getElementById('hops').value;

  if (!query)  return;

  document.getElementById('queryResult').classList.add('hidden');
  document.getElementById('queryLoader').classList.remove('hidden');
  document.getElementById('queryBtn').disabled = true;
  setStatus('busy', 'Thinking…');

  try {
    const res  = await fetch('/api/query', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, graph_name: graphName,
                             top_k_docs: topKDocs, top_k_nodes: topKNodes, hops }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Query failed');

    // Answer
    document.getElementById('answerText').textContent = data.answer;

    // Sources
    const srcBox  = document.getElementById('sourcesBox');
    const srcList = document.getElementById('sourcesList');
    if (data.sources?.length) {
      srcList.innerHTML = data.sources.map(s => `<span class="source-tag">${s}</span>`).join('');
      srcBox.classList.remove('hidden');
    } else { srcBox.classList.add('hidden'); }

    // Graph context
    const ctxBox  = document.getElementById('graphCtxBox');
    const ctxText = document.getElementById('graphCtxText');
    if (data.graph_context?.trim()) {
      ctxText.textContent = data.graph_context;
      ctxBox.classList.remove('hidden');
    } else { ctxBox.classList.add('hidden'); }

    document.getElementById('queryResult').classList.remove('hidden');
    setStatus('success', 'Done');
  } catch (err) {
    alert(err.message);
    setStatus('error', 'Error');
  } finally {
    document.getElementById('queryLoader').classList.add('hidden');
    document.getElementById('queryBtn').disabled = false;
  }
}

// Allow Ctrl+Enter to submit query
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('queryInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runQuery();
  });
});

/* ── Clear ───────────────────────────────────────────────────── */
async function clearData() {
  if (!confirm('Delete all embeddings and graph data?')) return;
  const graphName = document.getElementById('graphName').value.trim();
  setStatus('busy', 'Clearing…');
  try {
    await fetch(`/api/clear?graph_name=${graphName}`, { method: 'DELETE' });
    setStatus('idle', 'Cleared');
    await loadEmbeddedFiles();
      document.getElementById('queryResult').classList.add('hidden');
    document.getElementById('graphStats').innerHTML = '';
  } catch { setStatus('error', 'Error'); }
}



/* ── Util ────────────────────────────────────────────────────── */
function showMsg(el, type, text) {
  el.className = `status-msg ${type}`;
  el.textContent = text;
  el.classList.remove('hidden');
}
