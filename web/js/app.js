// Piano Studio web app: configuration panel, live 3D preview synced to the
// rendered audio, fingering editor, validation report, render + batch jobs.
import { Studio } from './studio.js';
import { SKIN_TONES } from './hands.js';

const $ = (id) => document.getElementById(id);
const api = async (url, opts = {}) => {
  const r = await fetch(url, { headers: opts.body && !(opts.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}, ...opts });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
};
const toast = (msg, ms = 2600) => { const t = $('toast'); t.textContent = msg; t.classList.remove('hidden'); clearTimeout(t._h); t._h = setTimeout(() => t.classList.add('hidden'), ms); };
const fmt = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60); return `${m}:${(s % 60).toFixed(1).padStart(4, '0')}`; };

const S = { t: 0, meta: null, pid: null, cfg: null, analysis: null, studio: null, audio: new Audio(), playing: false,
  selected: null, rollView: { t0: 0, pxPerSec: 90 }, dirty: false, saveTimer: null };

// ------------------------------------------------------------ boot
async function boot() {
  S.meta = await api('/api/meta');
  const ex = $('example');
  S.meta.examples.forEach((e) => ex.add(new Option(e.replace(/_/g, ' ').replace('.mid', ''), e)));
  for (const sel of [$('newPreset'), $('batchPreset')]) S.meta.presets.forEach((p) => sel.add(new Option(p, p)));
  Object.entries(S.meta.videoPresets).forEach(([k, v]) => $('videoPreset').add(new Option(`${k} · ${v.width}×${v.height} · ${v.fps} fps`, k)));
  buildThemes(); buildSkins(); bindControls(); bindTransport(); bindRoll(); bindTabs(); bindDialogs();
  const list = await api('/api/projects');
  if (list.length && location.hash.length > 1) openProject(location.hash.slice(1));
  setInterval(pollJobs, 1500);
}

function buildThemes() {
  const box = $('themes');
  S.meta.themes.forEach((t) => {
    const d = document.createElement('div');
    d.className = 'theme'; d.dataset.id = t.id;
    d.innerHTML = `<div class="sw" style="background:linear-gradient(${t.background.top},${t.background.bottom})">
      <i style="left:30%;height:18px;background:${t.notes.L};box-shadow:0 0 8px ${t.notes.L}"></i>
      <i style="left:55%;height:12px;background:${t.notes.R};box-shadow:0 0 8px ${t.notes.R}"></i></div><span>${t.name}</span>`;
    d.onclick = () => { S.cfg.theme = t.id; S.cfg.colors = null; syncForm(); changed(true); };
    box.appendChild(d);
  });
}
function buildSkins() {
  const box = $('skin');
  Object.entries(SKIN_TONES).forEach(([k, c]) => {
    const b = document.createElement('button');
    b.style.background = c; b.title = k; b.dataset.tone = k;
    b.onclick = () => { S.cfg.hands.skinTone = k; syncForm(); changed(true); };
    box.appendChild(b);
  });
}

// ------------------------------------------------------------ form <-> config
const F = {
  title: ['title'], composer: ['composer'], difficulty: ['difficulty'], introText: ['structure', 'introText'],
  range: ['range'], sleeve: ['hands', 'sleeve'], sleeveColor: ['hands', 'sleeveColor'],
  diffPreset: ['notes', 'difficultyPreset'], travel: ['notes', 'travelTime'], glow: ['notes', 'glow'],
  showFingers: ['notes', 'showFingers'], showNames: ['notes', 'showNoteNames'],
  intro: ['structure', 'intro'], practice: ['structure', 'practice'], practiceRate: ['structure', 'practiceRate'],
  practiceBars: ['structure', 'practiceBars'], practiceRepeats: ['structure', 'practiceRepeats'],
  progressBar: ['structure', 'progressBar'], outro: ['structure', 'outro'], metronome: ['audio', 'metronome'],
  reverb: ['audio', 'reverb'], channel: ['branding', 'channel'], tagline: ['branding', 'tagline'],
  watermark: ['branding', 'watermark'], videoPreset: ['video', 'preset'], strict: ['strict'], debug: ['debug'],
};
const getP = (o, p) => p.reduce((a, k) => (a == null ? a : a[k]), o);
const setP = (o, p, v) => { let a = o; p.slice(0, -1).forEach((k) => { a[k] = a[k] || {}; a = a[k]; }); a[p[p.length - 1]] = v; };
// changes that only affect visuals can be previewed without re-baking; all go through analyze for simplicity
const NEEDS_AUDIO = new Set(['intro', 'practice', 'practiceRate', 'practiceBars', 'practiceRepeats', 'outro', 'metronome', 'reverb', 'travel', 'diffPreset']);

function syncForm() {
  const c = S.cfg;
  for (const [id, p] of Object.entries(F)) {
    const el = $(id), v = getP(c, p);
    if (el.type === 'checkbox') el.checked = !!v; else if (v !== undefined && v !== null) el.value = v;
  }
  const theme = S.meta.themes.find((t) => t.id === c.theme) || S.meta.themes[0];
  document.querySelectorAll('.theme').forEach((d) => d.classList.toggle('active', d.dataset.id === c.theme));
  document.querySelectorAll('#skin button').forEach((b) => b.classList.toggle('active', b.dataset.tone === c.hands.skinTone));
  $('colR').value = (c.colors && c.colors.R) || theme.notes.R;
  $('colL').value = (c.colors && c.colors.L) || theme.notes.L;
  $('travelVal').textContent = `${(+c.notes.travelTime).toFixed(1)} s`;
  $('glowVal').textContent = (+c.notes.glow).toFixed(2);
  $('reverbVal').textContent = (+c.audio.reverb).toFixed(2);
  $('projTitle').textContent = `${c.title}${c.composer ? ' — ' + c.composer : ''}`;
}

function bindControls() {
  for (const [id, p] of Object.entries(F)) {
    const el = $(id);
    el.addEventListener(el.type === 'range' || el.type === 'text' ? 'input' : 'change', () => {
      let v = el.type === 'checkbox' ? el.checked : el.type === 'range' || el.type === 'number' ? +el.value : el.value;
      if (id === 'practiceRate') v = +v;
      setP(S.cfg, p, v);
      if (id === 'diffPreset') {
        const pr = { beginner: [4.0, true, true], intermediate: [3.0, false, false], advanced: [2.2, false, false] }[v];
        S.cfg.notes.travelTime = pr[0]; S.cfg.notes.showNoteNames = pr[1]; S.cfg.notes.showFingers = pr[2];
      }
      syncForm();
      changed(el.type !== 'text' || ['title', 'composer', 'introText', 'channel', 'tagline', 'watermark'].includes(id), NEEDS_AUDIO.has(id));
    });
  }
  for (const h of ['R', 'L']) $('col' + h).addEventListener('change', () => {
    S.cfg.colors = { R: $('colR').value, L: $('colL').value }; changed(true);
  });
  $('btnThemeColors').onclick = () => { S.cfg.colors = null; syncForm(); changed(true); };
  $('btnNew').onclick = () => { stop(); $('placeholder').classList.remove('hidden'); };
  $('btnExample').onclick = () => createProject({ example: $('example').value });
  $('file').onchange = (e) => e.target.files[0] && createProject({ file: e.target.files[0] });
  $('btnAnalyze').onclick = () => analyze(true);
  $('btnRender').onclick = renderVideo;
  $('aspect').onchange = () => S.analysis && loadPreview();
}

function changed(reanalyze = true, audio = false) {
  if (!S.pid) return;
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(async () => {
    await api(`/api/projects/${S.pid}`, { method: 'PUT', body: JSON.stringify(S.cfg) });
    if (reanalyze) analyze(audio || !S.audio.src);
  }, 450);
}

// ------------------------------------------------------------ projects
async function createProject({ example, file }) {
  const fd = new FormData();
  if (file) fd.append('file', file); else fd.append('example', example);
  if ($('newPreset').value) fd.append('preset', $('newPreset').value);
  const r = await api('/api/projects', { method: 'POST', body: fd });
  await openProject(r.id);
}
async function openProject(pid) {
  const r = await api(`/api/projects/${pid}`);
  S.pid = pid; S.cfg = r.config; location.hash = pid;
  $('placeholder').classList.add('hidden');
  syncForm();
  await analyze(true);
}

async function analyze(withAudio = true) {
  if (!S.pid) return;
  busy(withAudio ? 'Analysing, fingering, animating and rendering audio…' : 'Updating preview…');
  try {
    const a = await api(`/api/projects/${S.pid}/analyze?audio=${withAudio ? 1 : 0}`, { method: 'POST' });
    S.analysis = a;
    if (withAudio) { S.audio.src = `/api/projects/${S.pid}/preview/audio.wav?ts=${a.stamp}`; S.audio.load(); }
    await loadPreview();
    renderReport(); drawRoll(); updateInspector();
  } catch (e) { toast('Analysis failed: ' + e.message, 6000); }
  busy(null);
}

async function loadPreview() {
  const T = S.t || 0;
  const [perf, bake] = await Promise.all([
    fetch(`/api/projects/${S.pid}/preview/perf.json?ts=${S.analysis.stamp}`).then((r) => r.json()),
    fetch(`/api/projects/${S.pid}/preview/bake.bin?ts=${S.analysis.stamp}`).then((r) => r.arrayBuffer()),
  ]);
  const vp = $('viewport');
  const portrait = $('aspect').value === '9:16';
  let h = vp.clientHeight, w = portrait ? h * 9 / 16 : h * 16 / 9;
  if (w > vp.clientWidth) { w = vp.clientWidth; h = portrait ? w * 16 / 9 : w * 9 / 16; }
  w = Math.floor(w); h = Math.floor(h);
  perf.program.width = w; perf.program.height = h;
  perf.options.hands = { ...perf.options.hands, quality: 'low' };
  vp.querySelectorAll('canvas').forEach((c) => c.remove());
  if (S.studio) S.studio.dispose();
  const canvas = document.createElement('canvas');
  canvas.width = w * devicePixelRatio; canvas.height = h * devicePixelRatio;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  vp.prepend(canvas);
  const studio = new Studio(canvas, { width: w, height: h, pixelRatio: Math.min(2, devicePixelRatio), preview: true });
  perf.options.fontBase = '/web/fonts/';
  await studio.load(perf, bake);
  S.studio = studio; S.perf = perf;
  $('scrub').max = perf.program.duration;
  $('dur').textContent = fmt(perf.program.duration);
  S.t = Math.min(T || perf.program.segments[0].end + 6, perf.program.duration);
  if (!S.playing) frame();
}

// ------------------------------------------------------------ transport
function bindTransport() {
  $('btnPlay').onclick = () => (S.playing ? stop() : play());
  $('scrub').oninput = () => seek(+$('scrub').value);
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); S.playing ? stop() : play(); }
    if (S.selected != null) {
      if (/^Digit[1-5]$/.test(e.code)) setOverride(S.selected, { finger: +e.code.slice(5) });
      if (e.key === 'l' || e.key === 'L') setOverride(S.selected, { hand: 'L' });
      if (e.key === 'r' || e.key === 'R') setOverride(S.selected, { hand: 'R' });
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const notes = S.analysis.notes, i = notes.findIndex((n) => n.id === S.selected);
        const j = Math.max(0, Math.min(notes.length - 1, i + (e.key === 'ArrowRight' ? 1 : -1)));
        selectNote(notes[j].id, true);
      }
    }
  });
}
function seek(T) { S.t = T; try { S.audio.currentTime = T; } catch (e) { /* audio not ready */ } if (!S.playing) frame(); }
function play() {
  if (!S.studio) return;
  S.playing = true; $('btnPlay').textContent = '❚❚';
  try { S.audio.currentTime = S.t; } catch (e) {}
  S.wall = performance.now(); S.wallT = S.t;
  S.audio.play().catch(() => {});   // if audio cannot play, the clock below still runs
  requestAnimationFrame(loop);
}
function stop() { S.playing = false; $('btnPlay').textContent = '▶'; S.audio.pause(); }
function loop() {
  if (!S.playing) return;
  const audioOk = !S.audio.paused && S.audio.readyState >= 2;
  S.t = audioOk ? S.audio.currentTime : S.wallT + (performance.now() - S.wall) / 1000;
  frame();
  if (S.perf && S.t >= S.perf.program.duration) stop();
  requestAnimationFrame(loop);
}
function frame() {
  const T = S.t;
  S.studio && S.studio.render(T);
  $('tc').textContent = fmt(T);
  $('scrub').value = T;
  drawRoll();
}
function songTimeAt(T) { return S.studio ? S.studio.programAt(T).song : 0; }
function videoTimeForSong(t) {
  if (!S.perf) return 0;
  const seg = S.perf.program.segments.find((s) => s.type === 'performance');
  return seg.start + (t - seg.songStart);
}

// ------------------------------------------------------------ fingering editor (piano roll)
function bindRoll() {
  const c = $('roll');
  c.addEventListener('mousedown', (e) => {
    if (!S.analysis) return;
    const hit = rollHit(e.offsetX, e.offsetY);
    if (hit) selectNote(hit.id, false);
    else seek(videoTimeForSong(S.rollView.t0 + e.offsetX / S.rollView.pxPerSec));
  });
  c.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) S.rollView.pxPerSec = Math.max(20, Math.min(400, S.rollView.pxPerSec * (e.deltaY < 0 ? 1.15 : 0.87)));
    else S.rollView.t0 = Math.max(-2, S.rollView.t0 + (e.deltaY + e.deltaX) / S.rollView.pxPerSec);
    drawRoll();
  }, { passive: false });
  new ResizeObserver(() => drawRoll()).observe(c);
}
function rollGeom() {
  const c = $('roll'), r = S.analysis.range;
  const lo = r.lo, hi = r.hi;
  const rowH = Math.max(3, (c.clientHeight - 10) / (hi - lo + 1));
  return { lo, hi, rowH, H: c.clientHeight };
}
function rollHit(x, y) {
  const { lo, hi, rowH } = rollGeom();
  const t = S.rollView.t0 + x / S.rollView.pxPerSec;
  const pitch = hi - Math.floor((y - 5) / rowH);
  return S.analysis.notes.find((n) => n.p === pitch && t >= n.t0 && t <= Math.max(n.t1, n.t0 + 0.12));
}
function drawRoll() {
  const c = $('roll');
  if (!S.analysis || !c.clientWidth) return;
  const dpr = devicePixelRatio;
  if (c.width !== c.clientWidth * dpr) { c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr; }
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = c.clientWidth, { lo, hi, rowH, H } = rollGeom();
  const pps = S.rollView.pxPerSec, t0 = S.rollView.t0;
  g.fillStyle = '#0e1015'; g.fillRect(0, 0, W, H);
  for (let p = lo; p <= hi; p++) {
    const y = 5 + (hi - p) * rowH;
    if ([1, 3, 6, 8, 10].includes(p % 12)) { g.fillStyle = '#12151c'; g.fillRect(0, y, W, rowH); }
    if (p % 12 === 0) { g.fillStyle = '#262b38'; g.fillRect(0, y + rowH - 1, W, 1); g.fillStyle = '#596075'; g.font = '10px Inter'; g.fillText('C' + (p / 12 - 1), 4, y + rowH - 3); }
  }
  const colors = S.perf ? S.perf.options.colors : { R: '#ffb25c', L: '#5fa8ff' };
  const flagged = S.analysis.issuesByNote || {};
  const ov = S.cfg.overrides || {};
  for (const n of S.analysis.notes) {
    const x = (n.t0 - t0) * pps, w = Math.max(3, (n.t1 - n.t0) * pps - 1);
    if (x > W || x + w < 0) continue;
    const y = 5 + (hi - n.p) * rowH;
    g.fillStyle = colors[n.h] || '#999';
    g.globalAlpha = n.fs === 'auto' ? 0.78 : 1;
    g.beginPath(); g.roundRect(x, y + 0.5, w, rowH - 1, 2); g.fill();
    g.globalAlpha = 1;
    const fl = flagged[String(n.id)];
    if (fl) { g.strokeStyle = fl.some((i) => i.severity === 'error') ? '#ff6b6b' : '#f3c25b'; g.lineWidth = 1.5; g.stroke(); }
    if (ov[n.id]) { g.fillStyle = '#fff'; g.fillRect(x, y + 0.5, 2, rowH - 1); }
    if (n.id === S.selected) { g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke(); }
    if (rowH >= 8 && w > 9) { g.fillStyle = '#111'; g.font = `600 ${Math.min(11, rowH - 1)}px Inter`; g.fillText(String(n.f || '?'), x + 3, y + rowH - 2); }
  }
  if (S.studio) {
    const ts = songTimeAt(S.t || 0);
    const x = (ts - t0) * pps;
    g.fillStyle = '#fff'; g.fillRect(x, 0, 1, H);
    if (S.playing && (x > W * 0.8 || x < 0)) S.rollView.t0 = ts - (W * 0.2) / pps;
  }
}
function selectNote(id, doSeek) {
  S.selected = id;
  const n = S.analysis.notes.find((m) => m.id === id);
  if (doSeek && n) seek(videoTimeForSong(n.t0 - 0.05));
  updateInspector(); drawRoll();
}
const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
function updateInspector() {
  const box = $('inspector');
  const n = S.analysis && S.analysis.notes.find((m) => m.id === S.selected);
  if (!n) return;
  const issues = (S.analysis.issuesByNote || {})[String(n.id)] || [];
  box.innerHTML = `<div class="big">${NAMES[n.p % 12]}${Math.floor(n.p / 12) - 1}</div>
    <div class="muted">note #${n.id} · ${n.t0.toFixed(3)}s · ${(n.t1 - n.t0).toFixed(2)}s · vel ${n.v}</div>
    <div style="margin-top:10px">Hand <span class="muted">(${n.hs})</span></div>
    <div class="btns">${['L', 'R'].map((h) => `<button data-h="${h}" class="${n.h === h ? 'on' : ''}">${h === 'L' ? 'Left' : 'Right'}</button>`).join('')}</div>
    <div>Finger <span class="muted">(${n.fs})</span></div>
    <div class="btns">${[1, 2, 3, 4, 5].map((f) => `<button data-f="${f}" class="${n.f === f ? 'on' : ''}">${f}</button>`).join('')}</div>
    ${(S.cfg.overrides || {})[n.id] ? '<button data-clear="1" class="ghost">Reset to automatic</button>' : ''}
    ${issues.map((i) => `<div class="sev-${i.severity}" style="margin-top:8px">⚑ ${i.message}</div>`).join('')}`;
  box.querySelectorAll('[data-h]').forEach((b) => (b.onclick = () => setOverride(n.id, { hand: b.dataset.h })));
  box.querySelectorAll('[data-f]').forEach((b) => (b.onclick = () => setOverride(n.id, { finger: +b.dataset.f })));
  const cl = box.querySelector('[data-clear]');
  if (cl) cl.onclick = () => { delete S.cfg.overrides[n.id]; changed(true); };
}
function setOverride(id, o) {
  S.cfg.overrides = S.cfg.overrides || {};
  S.cfg.overrides[id] = { ...(S.cfg.overrides[id] || {}), ...o };
  S.cfg.approved = false;
  const n = S.analysis.notes.find((m) => m.id === id);
  if (o.finger) { n.f = o.finger; n.fs = 'user'; }
  if (o.hand) { n.h = o.hand; n.hs = 'user'; }
  updateInspector(); drawRoll();
  changed(true, false);
}

// ------------------------------------------------------------ validation report
function renderReport() {
  const r = S.analysis.report;
  const pill = $('statusPill');
  pill.textContent = r.status.replace('_', ' '); pill.className = 'pill ' + r.status;
  const checks = r.machine_checked.map((k) => {
    const c = r.checks[k] || {};
    return `<div class="card"><b>${c.passed ? '✅' : '❌'} ${k.replace(/_/g, ' ')}</b><div class="muted">${Object.entries(c).filter(([a]) => a !== 'passed').map(([a, b]) => `${a}: ${typeof b === 'object' ? JSON.stringify(b) : b}`).join(' · ')}</div></div>`;
  }).join('');
  $('report').innerHTML = `
    <p><b>Status: ${r.status.toUpperCase()}</b> · ${r.summary.notes} notes · ${r.summary.errors} errors · ${r.summary.review_warnings} review warnings · difficulty ${S.analysis.difficulty}
      ${r.status === 'needs_review' ? '<button id="btnApprove" class="primary" style="margin-left:10px">I reviewed the fingering — approve</button>' : ''}</p>
    <h4>Machine-checked</h4><div class="grid">${checks}</div>
    <h4>Needs human judgement</h4><ul>${r.needs_human_review.map((x) => `<li class="muted">${x}</li>`).join('')}</ul>
    <h4>Issues</h4>${r.issues.map((i) => `<div class="issue sev-${i.severity}" data-ids="${(i.note_ids || []).join(',')}" data-t="${i.time ?? ''}">${i.severity.toUpperCase()} · ${i.code}${i.time != null ? ' @ ' + i.time.toFixed(2) + 's' : ''} — ${i.message}</div>`).join('') || '<p class="muted">None</p>'}`;
  const ap = $('btnApprove');
  if (ap) ap.onclick = () => { S.cfg.approved = true; changed(true); toast('Fingering approved for export'); };
  $('report').querySelectorAll('.issue').forEach((d) => (d.onclick = () => {
    const ids = d.dataset.ids.split(',').filter(Boolean).map(Number);
    if (ids.length) { selectNote(ids[0], true); document.querySelector('[data-tab=fingering]').click(); }
    else if (d.dataset.t) seek(videoTimeForSong(+d.dataset.t));
  }));
}

// ------------------------------------------------------------ tabs, dialogs, jobs
function bindTabs() {
  document.querySelectorAll('.tabs button').forEach((b) => (b.onclick = () => {
    document.querySelectorAll('.tabs button').forEach((x) => x.classList.toggle('active', x === b));
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('hidden', t.id !== 'tab-' + b.dataset.tab));
    drawRoll();
  }));
}
function bindDialogs() {
  $('btnBatch').onclick = () => $('batchDlg').showModal();
  $('batchGo').onclick = async () => {
    try {
      const r = await api('/api/batch', { method: 'POST', body: JSON.stringify({ folder: $('batchFolder').value, preset: $('batchPreset').value, dryRun: $('batchDry').checked }) });
      toast('Batch started'); document.querySelector('[data-tab=jobs]').click(); S.watch = r.job;
    } catch (e) { toast(e.message, 5000); }
  };
  $('btnPreset').onclick = () => S.cfg ? $('presetDlg').showModal() : toast('Open a project first');
  $('presetGo').onclick = async () => {
    const r = await api('/api/presets', { method: 'POST', body: JSON.stringify({ name: $('presetName').value, config: S.cfg }) });
    toast(`Preset "${r.name}" saved`);
    for (const sel of [$('newPreset'), $('batchPreset')]) if (![...sel.options].some((o) => o.value === r.name)) sel.add(new Option(r.name, r.name));
  };
}
async function renderVideo() {
  if (!S.pid) return toast('Open a project first');
  const st = S.analysis && S.analysis.report.status;
  let force = false;
  if (st === 'blocked') return toast('Validation found errors — fix them in the fingering editor first.', 5000);
  if (st === 'needs_review' && S.cfg.strict && !S.cfg.approved) {
    if (!confirm('The fingering has items that need review. Render anyway?')) return;
    force = true;
  }
  await api(`/api/projects/${S.pid}/render`, { method: 'POST', body: JSON.stringify({ force }) });
  toast('Rendering started — see the Renders tab');
  document.querySelector('[data-tab=jobs]').click();
}
const rel = (p) => p ? '/output/' + p.split('/output/')[1] : '';
async function pollJobs() {
  const jobs = await api('/api/jobs').catch(() => []);
  if (!jobs.length) { $('jobs').innerHTML = '<p class="muted">No renders yet.</p>'; return; }
  const full = await Promise.all(jobs.slice(-8).reverse().map((j) => api('/api/jobs/' + j.id)));
  $('jobs').innerHTML = full.map((j) => {
    const p = j.progress || {};
    const pct = p.total ? (100 * p.frames / p.total) : (j.state === 'done' ? 100 : 0);
    const res = j.result || {};
    let links = '';
    if (j.kind === 'render' && res.rendered) {
      links = `<a href="${rel(res.video)}" target="_blank">▶ Video</a><a href="${rel(res.thumbnail)}" target="_blank">Thumbnail</a>
        <a href="${rel(res.out_dir + '/metadata.json')}" target="_blank">Metadata</a><a href="${rel(res.out_dir + '/validation.md')}" target="_blank">Report</a>
        <span class="muted">${res.integrity && res.integrity.passed ? '✓ export integrity checks passed' : '⚠ integrity check failed'} · ${res.seconds}s</span>`;
    } else if (j.kind === 'render' && res.reason) links = `<span class="sev-warning">${res.reason}</span>`;
    if (j.kind === 'batch' && res.counts) links = `<span>${JSON.stringify(res.counts)}</span> <a href="/output/batch/batch_summary.md" target="_blank">Summary</a>`;
    return `<div class="job"><b>${j.kind === 'batch' ? 'Batch · ' + j.folder : j.title}</b> <span class="muted">· ${j.state} · ${j.stage || ''}</span>
      <div class="bar"><i style="width:${pct.toFixed(1)}%"></i></div>
      <div class="muted">${p.total ? `${p.frames}/${p.total} frames · ${p.fps} fps · eta ${Math.round(p.eta)}s` : ''} ${j.error ? '<span class="sev-error">' + j.error + '</span>' : ''}</div>
      ${(j.songs || []).filter((s) => s.event === 'song_done').map((s) => `<div class="muted">${s.file}: ${s.status}${s.rendered ? ' · rendered' : ''}</div>`).join('')}
      <div style="margin-top:6px">${links}</div></div>`;
  }).join('');
}
function busy(msg) { $('busy').classList.toggle('hidden', !msg); if (msg) $('busyText').textContent = msg; }

boot();
