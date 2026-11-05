// 2D motion-graphics layer (titles, progress, practice countdown, outro).
// Drawn on a canvas, uploaded as a premultiplied texture and composited in
// the final post-processing pass, so it is part of the captured frame.
import * as THREE from 'three';

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const ease = (x) => { x = clamp(x, 0, 1); return 1 - Math.pow(1 - x, 3); };
const inout = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };

const FONT_FILES = [
  ['Inter', 'inter-latin-300-normal.woff2', '300', 'normal'],
  ['Inter', 'inter-latin-400-normal.woff2', '400', 'normal'],
  ['Inter', 'inter-latin-500-normal.woff2', '500', 'normal'],
  ['Inter', 'inter-latin-600-normal.woff2', '600', 'normal'],
  ['Cormorant Garamond', 'cormorant-garamond-latin-300-normal.woff2', '300', 'normal'],
  ['Cormorant Garamond', 'cormorant-garamond-latin-400-normal.woff2', '400', 'normal'],
  ['Cormorant Garamond', 'cormorant-garamond-latin-500-normal.woff2', '500', 'normal'],
  ['Cormorant Garamond', 'cormorant-garamond-latin-400-italic.woff2', '400', 'italic'],
];

let fontsPromise = null;
export function loadFonts(base = '/fonts/') {
  if (!fontsPromise) {
    fontsPromise = Promise.all(FONT_FILES.map(([fam, file, weight, style]) => {
      const ff = new FontFace(fam, `url(${base}${file})`, { weight, style });
      return ff.load().then((f) => document.fonts.add(f)).catch(() => null);
    }));
  }
  return fontsPromise;
}

export class Overlay {
  constructor(w, h, perf, theme) {
    this.perf = perf; this.theme = theme; this.opts = perf.options;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.premultiplyAlpha = true;
    this.texture.colorSpace = THREE.NoColorSpace;
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.generateMipmaps = false;
    this.resize(w, h);
    this.lastKey = '';
  }
  ready() { return loadFonts(this.opts.fontBase || '/fonts/'); }
  resize(w, h) {
    this.canvas.width = w; this.canvas.height = h;
    this.w = w; this.h = h;
    this.u = Math.min(w, h) / 1080;      // layout unit (1 at 1080p)
    this.lastKey = '';
  }

  update(T, seg, local, song, progress) {
    const dur = seg.end - seg.start;
    // quantised state key: only redraw when something visible changes
    const q = (x) => Math.round(x * 240) / 240;
    let key = seg.type + '|' + seg.start;
    const anim = ['intro', 'outro', 'countdown'].includes(seg.type) || local < 2.5 || dur - local < 1.2;
    if (anim) key += '|' + q(local);
    if (this.opts.progressBar) key += '|' + Math.round(progress * this.w);
    if (seg.type === 'practice') key += '|' + Math.floor(local);
    if (key === this.lastKey) return;
    this.lastKey = key;
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    if (seg.type === 'intro') this.drawIntro(local, dur);
    else if (seg.type === 'performance') this.drawPerformance(local, dur, progress);
    else if (seg.type === 'countdown') this.drawCountdown(seg, local, dur);
    else if (seg.type === 'practice') this.drawPractice(seg, local, dur);
    else if (seg.type === 'outro') this.drawOutro(local, dur);
    this.texture.needsUpdate = true;
  }

  font(weight, sizePx, family, style = 'normal') {
    return `${style} ${weight} ${Math.round(sizePx)}px "${family}", sans-serif`;
  }

  text(str, x, y, { size = 40, weight = 400, family, color, alpha = 1, align = 'center', spacing = 0, style = 'normal', shadow = 0 } = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color || this.theme.ui.text;
    ctx.font = this.font(weight, size * this.u, family || this.theme.ui.fontBody, style);
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    if ('letterSpacing' in ctx) ctx.letterSpacing = `${spacing * this.u}px`;
    if (shadow) { ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = shadow * this.u; }
    ctx.fillText(str, x, y);
    ctx.restore();
  }

  scrim(alpha) {
    const ctx = this.ctx;
    const g = ctx.createLinearGradient(0, 0, 0, this.h);
    const dark = this.isLight() ? '246,242,234' : '0,0,0';
    g.addColorStop(0, `rgba(${dark},${0.75 * alpha})`);
    g.addColorStop(0.62, `rgba(${dark},${0.55 * alpha})`);
    g.addColorStop(1, `rgba(${dark},${0.2 * alpha})`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, this.w, this.h);
  }

  isLight() { return this.theme.id === 'ivory_minimal'; }

  drawIntro(t, dur) {
    const S = this.perf.song, ui = this.theme.ui;
    const out = 1 - inout((t - (dur - 1.0)) / 0.9);
    this.scrim(0.9 * out);
    const cx = this.w / 2, portrait = this.h > this.w;
    const cy = this.h * (portrait ? 0.36 : 0.40);
    const a1 = ease((t - 0.3) / 1.0) * out;
    const a2 = ease((t - 0.8) / 1.0) * out;
    const a3 = ease((t - 1.3) / 1.0) * out;
    const rise = (a) => (1 - a) * 18 * this.u;
    this.text((S.kicker || 'Piano Tutorial').toUpperCase(), cx, cy - 120 * this.u + rise(a1),
      { size: 20, weight: 500, color: ui.accent, alpha: a1, spacing: 6 });
    const titleSize = clamp(1500 / Math.max(8, S.title.length), 56, portrait ? 92 : 118);
    this.text(S.title, cx, cy - 10 * this.u + rise(a1),
      { size: titleSize, weight: 400, family: ui.fontTitle, alpha: a1, spacing: ui.fontTitle === 'Inter' ? -1 : 0.5 });
    // thin rule
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = a2 * 0.6;
    ctx.fillStyle = ui.text;
    const rw = 120 * this.u * ease((t - 0.8) / 1.2);
    ctx.fillRect(cx - rw / 2, cy + 70 * this.u, rw, Math.max(1, 1.2 * this.u));
    ctx.restore();
    if (S.composer) {
      this.text(S.composer, cx, cy + 112 * this.u + rise(a2),
        { size: 34, weight: 400, family: ui.fontTitle, style: ui.fontTitle === 'Inter' ? 'normal' : 'italic', color: ui.text, alpha: a2 * 0.92 });
    }
    // difficulty chip + legend
    const chipY = cy + 190 * this.u + rise(a3);
    if (S.difficulty) this.chip(S.difficulty.toUpperCase(), cx, chipY, a3);
    if (this.opts.showLegend !== false) this.legend(cx, chipY + 70 * this.u, a3);
    if (S.introText) {
      this.text(S.introText, cx, chipY + 130 * this.u, { size: 22, weight: 300, color: ui.muted, alpha: a3 });
    }
  }

  chip(label, cx, cy, alpha) {
    const ctx = this.ctx, ui = this.theme.ui;
    ctx.save();
    ctx.font = this.font(600, 15 * this.u, ui.fontBody);
    if ('letterSpacing' in ctx) ctx.letterSpacing = `${4 * this.u}px`;
    const w = ctx.measureText(label).width + 40 * this.u, h = 36 * this.u;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = ui.accent;
    ctx.lineWidth = Math.max(1, 1.2 * this.u);
    roundRect(ctx, cx - w / 2, cy - h / 2, w, h, h / 2);
    ctx.stroke();
    ctx.fillStyle = ui.accent;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(label, cx + 2 * this.u, cy + 1);
    ctx.restore();
  }

  legend(cx, cy, alpha) {
    const ui = this.theme.ui, c = this.opts.colors;
    const items = [['Left hand', c.L], ['Right hand', c.R]];
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = this.font(400, 18 * this.u, ui.fontBody);
    const gap = 46 * this.u;
    const widths = items.map(([l]) => ctx.measureText(l).width + 26 * this.u);
    let x = cx - (widths.reduce((a, b) => a + b, 0) + gap) / 2;
    items.forEach(([label, col], i) => {
      ctx.fillStyle = col;
      ctx.shadowColor = col; ctx.shadowBlur = 12 * this.u;
      roundRect(ctx, x, cy - 6 * this.u, 14 * this.u, 12 * this.u, 4 * this.u); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = ui.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(label, x + 24 * this.u, cy + 1);
      x += widths[i] + gap;
    });
    ctx.restore();
  }

  drawPerformance(t, dur, progress) {
    const ui = this.theme.ui, S = this.perf.song;
    const portrait = this.h > this.w;
    const a = ease(t / 1.2) * (1 - inout((t - 5.5) / 1.5)) ;
    const pad = 54 * this.u;
    if (a > 0.003) {
      this.text(S.title, portrait ? this.w / 2 : pad, pad * 0.95, { size: 26, weight: 400, family: ui.fontTitle, align: portrait ? 'center' : 'left', alpha: a * 0.95, shadow: 8 });
      if (S.composer) this.text(S.composer, portrait ? this.w / 2 : pad, pad * 0.95 + 30 * this.u, { size: 15, weight: 400, color: ui.muted, align: portrait ? 'center' : 'left', alpha: a * 0.9, spacing: 1.5 });
    }
    if (this.opts.branding && this.opts.branding.watermark) {
      this.text(this.opts.branding.watermark, this.w - pad, pad * 0.95, { size: 15, weight: 500, color: ui.muted, align: 'right', alpha: 0.55, spacing: 3 });
    }
    if (this.opts.progressBar) {
      const ctx = this.ctx;
      const y = Math.round(this.h - 3 * this.u);
      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,0.10)';
      ctx.fillRect(0, y, this.w, Math.max(2, 3 * this.u));
      ctx.fillStyle = ui.progress;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(0, y, this.w * progress, Math.max(2, 3 * this.u));
      ctx.restore();
    }
  }

  drawCountdown(seg, t, dur) {
    const ui = this.theme.ui;
    const cx = this.w / 2, cy = this.h * 0.36;
    this.scrim(0.55 * (1 - inout((t - (dur - 0.4)) / 0.4)) * inout(t / 0.4));
    const pct = Math.round((seg.rate || 1) * 100);
    this.text('PRACTICE MODE', cx, cy - 110 * this.u, { size: 20, weight: 500, color: ui.accent, spacing: 6, alpha: ease(t / 0.5) });
    this.text(`${seg.label || 'Slow practice'} · ${pct}% speed`, cx, cy - 64 * this.u, { size: 30, weight: 400, family: ui.fontTitle, alpha: ease(t / 0.6) });
    const n = Math.max(1, Math.round(seg.count || 3));
    const step = (dur - 0.3) / n;
    const k = Math.floor(t / step);
    if (k < n) {
      const lt = (t - k * step) / step;
      const num = String(n - k);
      const sc = 1.25 - 0.25 * ease(lt * 2);
      this.text(num, cx, cy + 60 * this.u, { size: 150 * sc, weight: 300, family: ui.fontBody, alpha: (1 - inout((lt - 0.6) / 0.4)) * ease(lt * 4) });
    }
  }

  drawPractice(seg, t, dur) {
    const ui = this.theme.ui;
    const pad = 54 * this.u, portrait = this.h > this.w;
    const pct = Math.round((seg.rate || 1) * 100);
    const label = `Practice · ${pct}%${seg.repeatTotal > 1 ? ` · ${seg.repeatIndex}/${seg.repeatTotal}` : ''}`;
    this.text(label, portrait ? this.w / 2 : pad, pad * 0.95, { size: 18, weight: 500, color: ui.accent, align: portrait ? 'center' : 'left', spacing: 2, shadow: 8 });
    this.text('Finger numbers: 1 thumb · 2 index · 3 middle · 4 ring · 5 little', portrait ? this.w / 2 : pad, pad * 0.95 + 30 * this.u,
      { size: 14, weight: 400, color: ui.muted, align: portrait ? 'center' : 'left', alpha: 0.9 });
  }

  drawOutro(t, dur) {
    const ui = this.theme.ui, S = this.perf.song, br = this.opts.branding || {};
    const a = ease((t - 0.2) / 1.0);
    this.scrim(0.92 * ease(t / 0.9));
    const cx = this.w / 2, cy = this.h * 0.42;
    this.text(br.outroLine || 'Thank you for playing along', cx, cy - 30 * this.u, { size: 54, weight: 400, family: ui.fontTitle, alpha: a });
    this.text(S.title + (S.composer ? ' — ' + S.composer : ''), cx, cy + 34 * this.u, { size: 20, weight: 300, color: ui.muted, alpha: a * 0.9 });
    if (br.channel) {
      this.text(br.channel.toUpperCase(), cx, cy + 120 * this.u, { size: 18, weight: 600, color: ui.accent, spacing: 7, alpha: ease((t - 0.7) / 1.0) });
      if (br.tagline) this.text(br.tagline, cx, cy + 156 * this.u, { size: 16, weight: 300, color: ui.muted, alpha: ease((t - 0.9) / 1.0) });
    }
    if (this.perf.credits) {
      this.text(this.perf.credits, cx, this.h - 48 * this.u, { size: 12, weight: 300, color: ui.muted, alpha: 0.7 * a });
    }
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
