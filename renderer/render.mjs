#!/usr/bin/env node
// Frame-accurate offline renderer: drives the Three.js scene in headless
// Chromium, pulls raw RGBA frames and pipes them into FFmpeg (H.264/AAC MP4).
//
// usage: node renderer/render.mjs --job <dir> --out video.mp4 [--audio a.wav]
//        [--start 0] [--end N] [--workers 2] [--crf 18] [--preset medium]
//        [--still T --png out.png] [--tips T]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
let playwright;
try { playwright = require('playwright'); } catch {
  playwright = require(path.join(process.env.NODE_GLOBAL || '/opt/node22/lib/node_modules', 'playwright'));
}
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]);
  return acc;
}, []));
const log = (o) => process.stdout.write(JSON.stringify(o) + '\n');

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
  '.woff2': 'font/woff2', '.bin': 'application/octet-stream', '.png': 'image/png', '.css': 'text/css' };

function startServer(jobDir, onFrame) {
  const roots = [['/web/', path.join(ROOT, 'web')], ['/vendor/three/', path.join(ROOT, 'node_modules', 'three')],
    ['/job/', jobDir]];
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    if (req.method === 'POST' && url.pathname === '/frame') {
      const chunks = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', async () => {
        try { await onFrame(+url.searchParams.get('i'), Buffer.concat(chunks)); res.end('ok'); }
        catch (e) { res.statusCode = 500; res.end(String(e)); }
      });
      return;
    }
    for (const [prefix, dir] of roots) {
      if (url.pathname.startsWith(prefix)) {
        const fp = path.join(dir, decodeURIComponent(url.pathname.slice(prefix.length)));
        if (!fp.startsWith(dir)) break;
        fs.readFile(fp, (err, data) => {
          if (err) { res.statusCode = 404; res.end('not found'); return; }
          res.setHeader('Content-Type', MIME[path.extname(fp)] || 'application/octet-stream');
          res.end(data);
        });
        return;
      }
    }
    res.statusCode = 404; res.end();
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function openPage(browser, port, w, h) {
  const page = await browser.newPage({ viewport: { width: Math.min(w, 1920), height: Math.min(h, 1920) } });
  page.on('pageerror', (e) => log({ event: 'pageerror', message: String(e) }));
  page.on('console', (m) => { if (m.type() === 'error') log({ event: 'console', message: m.text() }); });
  await page.goto(`http://127.0.0.1:${port}/web/render.html?job=/job/&w=${w}&h=${h}`);
  await page.waitForFunction(() => window.__studioReady || window.__studioError, null, { timeout: 120000 });
  const err = await page.evaluate(() => window.__studioError);
  if (err) throw new Error(err);
  return page;
}

function launchBrowser() {
  const gpu = args.gpu === true || args.gpu === 'true';
  const flags = gpu ? ['--ignore-gpu-blocklist', '--enable-gpu-rasterization', '--use-angle=default']
    : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
  return playwright.chromium.launch({ args: flags.concat(['--disable-dev-shm-usage', '--no-sandbox']) });
}

function ffmpegEncoder(out, w, h, fps, { audio, crf = 18, preset = 'medium' } = {}) {
  const a = ['-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${w}x${h}`, '-r', String(fps), '-i', 'pipe:0'];
  if (audio) a.push('-i', audio);
  a.push('-vf', 'vflip,format=yuv420p', '-c:v', 'libx264', '-preset', String(preset), '-crf', String(crf),
    '-profile:v', 'high', '-r', String(fps), '-g', String(fps * 2), '-bf', '2', '-colorspace', 'bt709',
    '-color_primaries', 'bt709', '-color_trc', 'bt709');
  if (audio) a.push('-c:a', 'aac', '-b:a', '320k', '-ar', '48000');
  a.push('-movflags', '+faststart', out);
  const p = spawn('ffmpeg', a, { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => p.on('close', (c) => (c === 0 ? res() : rej(new Error('ffmpeg exited ' + c)))));
  return { proc: p, done };
}

async function renderRange({ browser, port, w, h, fps, start, end, out, audio, crf, preset, workerId, progress }) {
  const enc = ffmpegEncoder(out, w, h, fps, { audio, crf, preset });
  const pending = new Map();
  let next = start;
  const write = (buf) => new Promise((res) => { if (enc.proc.stdin.write(buf)) res(); else enc.proc.stdin.once('drain', res); });
  const server = await startServer(args.job, async (i, buf) => {
    pending.set(i, buf);
    while (pending.has(next)) {
      const b = pending.get(next); pending.delete(next);
      await write(b); next++;
    }
  });
  const page = await openPage(browser, server.address().port, w, h);
  for (let i = start; i < end; i++) {
    await page.evaluate(([T, i]) => window.__studio.frame(T, i), [i / fps, i]);
    progress(workerId, i);
  }
  enc.proc.stdin.end();
  await enc.done;
  await page.close();
  server.close();
}

async function main() {
  if (!args.job) { console.error('missing --job'); process.exit(2); }
  args.job = path.resolve(args.job);
  const perf = JSON.parse(fs.readFileSync(path.join(args.job, 'perf.json'), 'utf8'));
  const w = +(args.w || perf.program.width), h = +(args.h || perf.program.height), fps = perf.program.fps;
  const browser = await launchBrowser();
  try {
    if (args.still !== undefined || args.tips !== undefined || args.profile !== undefined) {
      const server = await startServer(args.job, async () => {});
      const page = await openPage(browser, server.address().port, w, h);
      const info = await page.evaluate(() => window.__studio.info());
      if (args.profile !== undefined) log({ event: 'profile', ...(await page.evaluate((T) => window.__studio.profile(T), +args.profile)) });
      if (args.tips !== undefined) {
        const tips = await page.evaluate((T) => window.__studio.debugTips(T), +args.tips);
        log({ event: 'tips', tips, info });
      }
      if (args.still !== undefined) {
        const times = String(args.still).split(',').map(Number);
        const outs = String(args.png).split(',');
        for (let k = 0; k < times.length; k++) {
          const data = await page.evaluate((T) => window.__studio.still(T), times[k]);
          fs.writeFileSync(outs[k], Buffer.from(data.split(',')[1], 'base64'));
          log({ event: 'still', t: times[k], file: outs[k] });
        }
      }
      log({ event: 'info', ...info });
      server.close();
      return;
    }
    const total = perf.program.frames;
    const start = +(args.start || 0), end = Math.min(total, +(args.end || total));
    const workers = Math.max(1, Math.min(+(args.workers || 1), 8));
    const out = path.resolve(args.out || path.join(args.job, 'video.mp4'));
    const t0 = Date.now();
    const done = new Array(workers).fill(0);
    let lastLog = 0;
    const progress = (wid, i) => {
      done[wid]++;
      const n = done.reduce((a, b) => a + b, 0);
      const now = Date.now();
      if (now - lastLog > 1000 || n === end - start) {
        lastLog = now;
        const el = (now - t0) / 1000;
        log({ event: 'progress', frames: n, total: end - start, fps: +(n / el).toFixed(2), eta: +(((end - start) - n) / (n / el)).toFixed(1) });
      }
    };
    log({ event: 'start', frames: end - start, w, h, fps, workers });
    if (workers === 1) {
      await renderRange({ browser, w, h, fps, start, end, out, audio: args.audio, crf: args.crf, preset: args.preset, workerId: 0, progress });
    } else {
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ps-chunks-'));
      const per = Math.ceil((end - start) / workers);
      const parts = [];
      await Promise.all(Array.from({ length: workers }, (_, k) => {
        const s = start + k * per, e = Math.min(end, s + per);
        if (s >= e) return null;
        const f = path.join(tmp, `part${k}.mp4`);
        parts.push(f);
        return renderRange({ browser, w, h, fps, start: s, end: e, out: f, crf: args.crf, preset: args.preset, workerId: k, progress });
      }));
      parts.sort();
      const list = path.join(tmp, 'list.txt');
      fs.writeFileSync(list, parts.map((p) => `file '${p}'`).join('\n'));
      const a = ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list];
      if (args.audio) a.push('-i', args.audio, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000');
      a.push('-c:v', 'copy', '-movflags', '+faststart', out);
      await new Promise((res, rej) => spawn('ffmpeg', a, { stdio: 'inherit' }).on('close', (c) => (c ? rej(new Error('concat failed')) : res())));
      fs.rmSync(tmp, { recursive: true, force: true });
    }
    log({ event: 'done', out, seconds: (Date.now() - t0) / 1000 });
  } finally {
    await browser.close();
  }
}

main().catch((e) => { log({ event: 'error', message: String(e && e.stack || e) }); process.exit(1); });
