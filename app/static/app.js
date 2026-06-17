'use strict';

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => {
      t.classList.remove('active');
      t.classList.add('hidden');
    });
    btn.classList.add('active');
    const target = document.getElementById(`tab-${btn.dataset.tab}`);
    target.classList.remove('hidden');
    target.classList.add('active');
  });
});

// ── Status polling ────────────────────────────────────────────────────────────
const statusBadge    = document.getElementById('status-badge');
const qitStatusNote  = document.getElementById('qit-status-note');
const modelInfo      = document.getElementById('model-info');

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    if (s.has_model) {
      statusBadge.className = 'status-badge status-ready';
      statusBadge.textContent = `Model ready · ${s.n_qubits} qubits · ${s.n_parameters} params`;
      document.getElementById('gen-btn').disabled = false;
      modelInfo.textContent = `vocab=${s.vocab_size}  qubits=${s.n_qubits}  params=${s.n_parameters}`;
      qitStatusNote.textContent = '⚛ QIT re-ranking active';
      qitStatusNote.className = 'pill pill-purple';
    } else if (s.active_jobs > 0) {
      statusBadge.className = 'status-badge status-training';
      statusBadge.textContent = 'Training…';
    } else {
      statusBadge.className = 'status-badge status-idle';
      statusBadge.textContent = s.has_corpus
        ? `Corpus loaded (${fmtBytes(s.corpus_chars)}) — no model yet`
        : 'No model loaded';
      qitStatusNote.textContent = 'No QIT model — using TF-IDF only';
      qitStatusNote.className = 'pill pill-yellow';
    }
  } catch (_) {}
}
setInterval(refreshStatus, 4000);
refreshStatus();

// ── Corpus ────────────────────────────────────────────────────────────────────
const corpusInput    = document.getElementById('corpus-input');
const corpusCount    = document.getElementById('corpus-char-count');
const setCorpusBtn   = document.getElementById('set-corpus-btn');
const fileUpload     = document.getElementById('file-upload');
const validUpload    = document.getElementById('valid-upload');
const corpusFileInfo = document.getElementById('corpus-file-info');
const validFileInfo  = document.getElementById('valid-file-info');

function fmtBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(2)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// Train file: show name+size immediately from File API, then upload in background
fileUpload.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  // Immediate feedback — no network needed to show filename and size
  corpusFileInfo.textContent = `${file.name} — ${fmtBytes(file.size)} · Uploading…`;
  corpusFileInfo.style.setProperty('color', 'var(--muted)');
  // Yield to the browser so the DOM repaint happens before the fetch blocks
  await new Promise(r => requestAnimationFrame(r));
  const form = new FormData();
  form.append('file', file);
  try {
    const r = await fetch('/api/corpus/upload', { method: 'POST', body: form });
    if (!r.ok) {
      let detail = 'Upload failed';
      try { detail = (await r.json()).detail; } catch (_) {}
      throw new Error(detail);
    }
    const d = await r.json();
    corpusFileInfo.textContent = `${file.name} — ${fmtBytes(d.bytes)} · Ready`;
    corpusFileInfo.style.setProperty('color', 'var(--green)');
    refreshStatus();
  } catch (err) {
    corpusFileInfo.textContent = `${file.name} · Error: ${err.message}`;
    corpusFileInfo.style.setProperty('color', 'var(--red)');
    console.error('Corpus upload failed:', err);
  }
  fileUpload.value = '';
});

// Validation file: same pattern — immediate feedback, then upload
validUpload.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  validFileInfo.textContent = `${file.name} — ${fmtBytes(file.size)} · Uploading…`;
  validFileInfo.style.setProperty('color', 'var(--muted)');
  await new Promise(r => requestAnimationFrame(r));
  const form = new FormData();
  form.append('file', file);
  try {
    const r = await fetch('/api/corpus/upload-valid', { method: 'POST', body: form });
    if (!r.ok) {
      let detail = 'Upload failed';
      try { detail = (await r.json()).detail; } catch (_) {}
      throw new Error(detail);
    }
    const d = await r.json();
    validFileInfo.textContent = `${file.name} — ${fmtBytes(d.bytes)} · Ready`;
    validFileInfo.style.setProperty('color', 'var(--green)');
    refreshStatus();
  } catch (err) {
    validFileInfo.textContent = `${file.name} · Error: ${err.message}`;
    validFileInfo.style.setProperty('color', 'var(--red)');
    console.error('Valid corpus upload failed:', err);
  }
  validUpload.value = '';
});

// Paste path (textarea → /api/corpus/text)
corpusInput.addEventListener('input', () => {
  corpusCount.textContent = `${corpusInput.value.length.toLocaleString()} characters`;
});

setCorpusBtn.addEventListener('click', async () => {
  const text = corpusInput.value.trim();
  if (!text) { alert('Paste some text first.'); return; }
  setCorpusBtn.disabled = true;
  setCorpusBtn.textContent = 'Setting…';
  try {
    const r = await fetch('/api/corpus/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    corpusFileInfo.textContent = `Pasted text — ${text.length.toLocaleString()} chars`;
    corpusFileInfo.style.color = 'var(--green)';
    setCorpusBtn.textContent = '✓ Set';
    setTimeout(() => { setCorpusBtn.textContent = 'Set Corpus'; setCorpusBtn.disabled = false; }, 1500);
    refreshStatus();
  } catch (err) {
    alert(`Error: ${err.message}`);
    setCorpusBtn.textContent = 'Set Corpus';
    setCorpusBtn.disabled = false;
  }
});

// ── Qubit label updater ───────────────────────────────────────────────────────
const cfgCtx     = document.getElementById('cfg-ctx');
const cfgLayers  = document.getElementById('cfg-layers');
const qubitsLabel = document.getElementById('qubits-label');

function updateQubitsLabel() {
  const ctx = parseInt(cfgCtx.value) || 6;
  const q = ctx * 2;
  qubitsLabel.textContent = `Total qubits: ${q}  (2^${q} = ${(2**q).toLocaleString()} states)`;
}
cfgCtx.addEventListener('input', updateQubitsLabel);
updateQubitsLabel();

// ── Training ──────────────────────────────────────────────────────────────────
const trainBtn      = document.getElementById('train-btn');
const epochLabel    = document.getElementById('epoch-label');
const trainingIdle  = document.getElementById('training-idle');
const trainingActive = document.getElementById('training-active');
const sampleCard    = document.getElementById('sample-card');
const sampleOutput  = document.getElementById('sample-output');

// Chart.js setup
let chart = null;
const chartLabels = [], trainData = [], valData = [];

function initChart() {
  if (chart) chart.destroy();
  chartLabels.length = 0; trainData.length = 0; valData.length = 0;
  const ctx = document.getElementById('loss-chart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: chartLabels,
      datasets: [
        { label: 'Train loss', data: trainData, borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,.08)', tension: 0.3, pointRadius: 2 },
        { label: 'Val loss',   data: valData,   borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,.08)',  tension: 0.3, pointRadius: 2 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      plugins: { legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
      },
    },
  });
}

const trainingPhase = document.getElementById('training-phase');
const trainingLog   = document.getElementById('training-log');
const trainingProgressBar = document.getElementById('training-progress-bar');
const trainingPct   = document.getElementById('training-pct');
const trainingEta   = document.getElementById('training-eta');

const MAX_CHART_POINTS = 400;

function formatDuration(seconds) {
  const sec = Math.round(seconds);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function formatEta(seconds) {
  if (seconds == null || seconds < 0 || !Number.isFinite(seconds)) return '';
  if (seconds < 60) return `~${seconds}s remaining`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `~${m}m ${s}s remaining`;
  const h = Math.floor(m / 60);
  return `~${h}h ${m % 60}m remaining`;
}

function setProgress(pct) {
  if (pct == null || Number.isNaN(pct)) return;
  const clamped = Math.max(0, Math.min(100, pct));
  trainingProgressBar.style.width = `${clamped}%`;
  trainingPct.textContent = `${clamped.toFixed(0)}%`;
}

function appendTrainingLog(message) {
  const ts = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.className = 'training-log-line';
  line.textContent = `${ts}  ${message}`;
  trainingLog.appendChild(line);
  trainingLog.scrollTop = trainingLog.scrollHeight;
  trainingPhase.textContent = message;
}

function setTrainingStatus(message, pct = null) {
  appendTrainingLog(message);
  if (pct != null) setProgress(pct);
}

function updateBatchChart(epoch, batch, totalBatches, loss) {
  const label = `E${epoch}.${batch}`;
  chartLabels.push(label);
  trainData.push(loss);
  while (chartLabels.length > MAX_CHART_POINTS) {
    chartLabels.shift();
    trainData.shift();
  }
  if (chart) {
    chart.data.datasets[1].data = valData;
    chart.update('none');
  }
}

function updateMetrics(d) {
  document.getElementById('m-train-loss').textContent = d.train_loss.toFixed(4);
  document.getElementById('m-val-loss').textContent   = d.val_loss.toFixed(4);
  document.getElementById('m-ppl').textContent        = d.val_ppl.toFixed(2);
  document.getElementById('m-bpc').textContent        = d.bpc.toFixed(3);
  epochLabel.textContent = `Epoch ${d.epoch} / ${d.epochs || d.epoch}`;
  if (d.pct_overall != null) setProgress(d.pct_overall);
  if (d.eta_total_sec != null) {
    trainingEta.textContent =
      `Last epoch: ${formatDuration(d.elapsed)} · ${formatEta(d.eta_total_sec)} for remaining epochs`;
  }

  chartLabels.push(`E${d.epoch}`);
  trainData.push(d.train_loss);
  valData.push(d.val_loss);
  while (chartLabels.length > MAX_CHART_POINTS) {
    chartLabels.shift();
    trainData.shift();
    if (valData.length) valData.shift();
  }
  if (chart) chart.update('none');
}

trainBtn.addEventListener('click', async () => {
  const corpus = corpusInput.value.trim() || null;
  if (!corpus && !(await hasActiveCorpus())) {
    alert('Load a corpus first (upload a .txt file or paste text).'); return;
  }

  const maxStepsEl = document.getElementById('cfg-max-steps');
  const maxStepsVal = maxStepsEl ? parseInt(maxStepsEl.value) : 200;

  const cfg = {
    ctx_len:               parseInt(cfgCtx.value),
    n_layers:              parseInt(cfgLayers.value),
    epochs:                parseInt(document.getElementById('cfg-epochs').value),
    lr:                    parseFloat(document.getElementById('cfg-lr').value),
    batch_size:            parseInt(document.getElementById('cfg-batch').value),
    gen_every:             parseInt(document.getElementById('cfg-gen-every').value),
    max_steps_per_epoch:   isNaN(maxStepsVal) ? null : maxStepsVal,
    corpus_text:           corpus || undefined,
  };

  // Start job
  let jobId;
  try {
    const r = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    const resp = await r.json();
    jobId = resp.job_id;
    if (resp.warning) {
      // Show cap warning inline (non-blocking)
      const warn = document.createElement('div');
      warn.style.cssText =
        'background:rgba(210,153,34,.15);border:1px solid rgba(210,153,34,.4);' +
        'color:#d29922;border-radius:6px;padding:8px 12px;font-size:12px;margin-bottom:10px;';
      warn.textContent = '⚠ ' + resp.warning;
      trainingActive.insertAdjacentElement('beforebegin', warn);
      setTimeout(() => warn.remove(), 12000);
    }
  } catch (err) {
    alert(`Failed to start training: ${err.message}`); return;
  }

  // Show training UI
  trainingIdle.classList.add('hidden');
  trainingActive.classList.remove('hidden');
  trainingLog.innerHTML = '';
  trainingPhase.textContent = 'Starting…';
  setProgress(0);
  trainingEta.textContent = '';
  sampleCard.style.display = 'none';
  trainBtn.disabled = true;
  trainBtn.textContent = '⏳ Training…';
  statusBadge.className = 'status-badge status-training';
  statusBadge.textContent = 'Training…';
  initChart();

  setTrainingStatus('Connecting to training stream…');

  // SSE stream
  let trainingDone = false;
  let sawTrainingEvent = false;
  const evtSource = new EventSource(`/api/train/${jobId}/stream`);
  evtSource.onmessage = e => {
    const data = JSON.parse(e.data);
    if (data.type === 'status') {
      sawTrainingEvent = true;
      setTrainingStatus(data.message, data.pct);
    } else if (data.type === 'heartbeat') {
      if (!trainingDone && !sawTrainingEvent) {
        setTrainingStatus('Connected — waiting for first training update…');
      }
    } else if (data.type === 'batch') {
      sawTrainingEvent = true;
      epochLabel.textContent =
        `Epoch ${data.epoch}/${data.epochs || '?'}  —  batch ${data.batch} / ${data.total_batches}`;
      document.getElementById('m-train-loss').textContent =
        data.running_loss.toFixed(4);
      if (data.pct != null) setProgress(data.pct_overall ?? data.pct);
      if (data.eta_epoch_sec != null) {
        trainingEta.textContent =
          `Epoch ${data.epoch}: ${formatEta(data.eta_epoch_sec)} in this epoch`;
      }
      trainingPhase.textContent =
        `Epoch ${data.epoch}: batch ${data.batch}/${data.total_batches} ` +
        `(${data.pct ?? 0}% of epoch, loss ${data.running_loss.toFixed(4)})`;
      updateBatchChart(data.epoch, data.batch, data.total_batches, data.running_loss);
    } else if (data.type === 'progress') {
      sawTrainingEvent = true;
      updateMetrics(data);
      trainingPhase.textContent =
        `Epoch ${data.epoch} complete — val loss ${data.val_loss.toFixed(4)}, ppl ${data.val_ppl.toFixed(2)}`;
      if (data.sample) {
        sampleCard.style.display = '';
        sampleOutput.textContent = data.sample;
      }
    } else if (data.type === 'done') {
      trainingDone = true;
      setTrainingStatus('Training finished successfully.');
      evtSource.close();
      trainBtn.disabled = false;
      trainBtn.textContent = '▶ Train';
      refreshStatus();
    } else if (data.type === 'error') {
      trainingDone = true;
      evtSource.close();
      alert(`Training error: ${data.message}`);
      trainBtn.disabled = false;
      trainBtn.textContent = '▶ Train';
      statusBadge.className = 'status-badge status-error';
      statusBadge.textContent = 'Training failed';
    }
  };
  evtSource.onerror = () => {
    if (trainingDone) return;
    // EventSource retries automatically while CONNECTING; only bail when closed.
    if (evtSource.readyState === EventSource.CLOSED) {
      evtSource.close();
      trainBtn.disabled = false;
      trainBtn.textContent = '▶ Train';
      statusBadge.className = 'status-badge status-error';
      statusBadge.textContent = 'Training stream lost';
    }
  };
});

async function hasActiveCorpus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    return s.has_corpus;
  } catch { return false; }
}

// ── Generate ──────────────────────────────────────────────────────────────────
const genBtn    = document.getElementById('gen-btn');
const genOutput = document.getElementById('gen-output');

genBtn.addEventListener('click', async () => {
  const prompt = document.getElementById('gen-prompt').value;
  const maxTokens = parseInt(document.getElementById('gen-tokens').value);
  const temperature = parseFloat(document.getElementById('gen-temp').value);
  const topK = parseInt(document.getElementById('gen-topk').value);

  genBtn.disabled = true;
  genBtn.textContent = '…';
  genOutput.textContent = 'Generating with quantum circuits…';
  genOutput.classList.remove('muted');

  try {
    const r = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, max_tokens: maxTokens, temperature, top_k: topK }),
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    const { text } = await r.json();
    genOutput.textContent = text;
  } catch (err) {
    genOutput.textContent = `Error: ${err.message}`;
    genOutput.classList.add('muted');
  } finally {
    genBtn.disabled = false;
    genBtn.textContent = 'Generate';
  }
});

// ── Chat ──────────────────────────────────────────────────────────────────────
const chatHistory  = document.getElementById('chat-history');
const chatInput    = document.getElementById('chat-input');
const chatSendBtn  = document.getElementById('chat-send-btn');
const clearChatBtn = document.getElementById('clear-chat-btn');
const apiKeyInput  = document.getElementById('api-key-input');

clearChatBtn.addEventListener('click', () => {
  chatHistory.innerHTML = `
    <div class="chat-welcome">
      <div class="welcome-icon">⚛</div>
      <h3>Ask about your corpus</h3>
      <p class="muted">Train a model in QIT Studio, then ask questions here. Claude answers using passages retrieved from your text, re-ranked by the quantum model when available.</p>
    </div>`;
});

// Auto-resize textarea
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

// Send on Enter (not Shift+Enter)
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
chatSendBtn.addEventListener('click', sendMessage);

function appendMessage(role, text, sources, qitPpls, method) {
  // Remove welcome screen if present
  chatHistory.querySelector('.chat-welcome')?.remove();

  const msg = document.createElement('div');
  msg.className = `chat-msg ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  msg.appendChild(bubble);

  if (role === 'assistant' && sources && sources.length) {
    const sourcesRow = document.createElement('div');
    sourcesRow.className = 'sources-row';

    const methodSpan = document.createElement('span');
    methodSpan.className = 'method-label';
    methodSpan.textContent = method === 'tfidf+qit' ? '⚛ QIT+TF-IDF' : '🔍 TF-IDF';
    sourcesRow.appendChild(methodSpan);

    sources.forEach((s, i) => {
      const chip = document.createElement('div');
      chip.className = 'source-chip';
      const pplLabel = qitPpls && qitPpls[i] ? ` · ppl ${qitPpls[i]}` : '';
      chip.textContent = s.slice(0, 60) + '…' + pplLabel;
      chip.title = s;
      sourcesRow.appendChild(chip);
    });
    msg.appendChild(sourcesRow);
  }

  chatHistory.appendChild(msg);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return msg;
}

function appendThinking() {
  chatHistory.querySelector('.chat-welcome')?.remove();
  const msg = document.createElement('div');
  msg.className = 'chat-msg assistant';
  msg.id = 'thinking-msg';
  const bubble = document.createElement('div');
  bubble.className = 'bubble thinking-dots';
  bubble.innerHTML = '<span>•</span><span>•</span><span>•</span>';
  msg.appendChild(bubble);
  chatHistory.appendChild(msg);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return msg;
}

async function sendMessage() {
  const question = chatInput.value.trim();
  if (!question) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';
  chatSendBtn.disabled = true;

  appendMessage('user', question);
  const thinkingMsg = appendThinking();

  try {
    const r = await fetch('/api/qa', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        api_key: apiKeyInput.value.trim() || undefined,
        claude_model: document.getElementById('claude-model').value,
      }),
    });
    const data = await r.json();
    thinkingMsg.remove();
    if (!r.ok) {
      appendMessage('assistant', `Error: ${data.detail || 'Unknown error'}`);
    } else {
      appendMessage('assistant', data.answer, data.passages, data.qit_ppls, data.method);
    }
  } catch (err) {
    thinkingMsg.remove();
    appendMessage('assistant', `Network error: ${err.message}`);
  } finally {
    chatSendBtn.disabled = false;
  }
}
