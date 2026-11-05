// Procedural, anatomically-proportioned, skinned human hands.
//
// Geometry: a signed-distance model of the hand (palm, thenar eminence,
// tapered phalanges, knuckles, wrist) is meshed once with "surface nets" in a
// natural bind pose, then skinned (smooth distance-based weights) to a bone
// hierarchy that follows exactly the conventions of pianostudio/animation.py:
//   root (wrist) : position P, Euler YXZ (yaw, pitch, roll)
//   finger base  : base_f (x scaled by splay), fixed base rotation (Euler YXZ)
//   yaw bone     : rotation.y = a
//   joint bones  : rotation.x = -theta_i, bone length L_i along local -z
// so the renderer reproduces the solver's forward kinematics exactly.
import * as THREE from 'three';

export const SKIN_TONES = {
  porcelain: '#f0d2c0', light: '#e3b79a', medium: '#c48e6a', tan: '#a06a48', deep: '#6b432f', ebony: '#47291c',
};
const SPLAY_K = { 1: 0, 2: 1, 3: 0, 4: 1, 5: 1 };
// bind pose (moderately curved, like a relaxed hand above the keys)
const BIND = { 1: [0.15, 0.25, 0.2], 2: [0.30, 0.45, 0.32], 3: [0.30, 0.45, 0.32], 4: [0.30, 0.45, 0.32], 5: [0.30, 0.45, 0.32] };
const BIND_YAW = { 1: 0.0, 2: 0.06, 3: 0.0, 4: -0.06, 5: -0.14 };

const geometryCache = new Map();

// ------------------------------------------------------------------ SDF helpers
function sdRoundCone(px, py, pz, a, b, r1, r2) {
  // Inigo Quilez round cone between points a and b with radii r1, r2
  const bax = b[0] - a[0], bay = b[1] - a[1], baz = b[2] - a[2];
  const l2 = bax * bax + bay * bay + baz * baz;
  const rr = r1 - r2, a2 = l2 - rr * rr, il2 = 1 / l2;
  const pax = px - a[0], pay = py - a[1], paz = pz - a[2];
  const y = pax * bax + pay * bay + paz * baz;
  const z = y - l2;
  const xx = (pax * l2 - bax * y), xy = (pay * l2 - bay * y), xz = (paz * l2 - baz * y);
  const x2 = xx * xx + xy * xy + xz * xz;
  const y2 = y * y * l2, z2 = z * z * l2;
  const k = Math.sign(rr) * rr * rr * x2;
  if (Math.sign(z) * a2 * z2 > k) return Math.sqrt(x2 + z2) * il2 - r2;
  if (Math.sign(y) * a2 * y2 < k) return Math.sqrt(x2 + y2) * il2 - r1;
  return (Math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
function sdEllipsoid(px, py, pz, c, r) {
  const x = (px - c[0]) / r[0], y = (py - c[1]) / r[1], z = (pz - c[2]) / r[2];
  const k0 = Math.sqrt(x * x + y * y + z * z);
  const x1 = x / r[0], y1 = y / r[1], z1 = z / r[2];
  const k1 = Math.sqrt(x1 * x1 + y1 * y1 + z1 * z1);
  return k0 < 1e-6 ? -Math.min(...r) : k0 * (k0 - 1) / k1;
}
function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

// ------------------------------------------------------------------ skeleton
function makeSkeleton(s, handGeo) {
  const root = new THREE.Bone();
  root.name = 'wrist';
  root.rotation.order = 'YXZ';
  const forearm = new THREE.Bone();
  forearm.name = 'forearm';
  root.add(forearm);
  const fingers = {};
  const bones = [root, forearm];
  for (const f of [1, 2, 3, 4, 5]) {
    const spec = handGeo.fingers[String(f)];
    const base = new THREE.Bone();
    base.position.set(s * spec.base[0], spec.base[1], spec.base[2]);
    base.rotation.order = 'YXZ';
    base.rotation.set(spec.baseEuler[1], s * spec.baseEuler[0], s * spec.baseEuler[2]);
    root.add(base);
    const yaw = new THREE.Bone();
    base.add(yaw);
    const joints = [];
    let parent = yaw;
    for (let i = 0; i < 3; i++) {
      const j = new THREE.Bone();
      if (i > 0) j.position.set(0, 0, -spec.lengths[i - 1]);
      parent.add(j);
      joints.push(j);
      parent = j;
    }
    const tip = new THREE.Bone();
    tip.position.set(0, 0, -spec.lengths[2]);
    joints[2].add(tip);
    bones.push(base, yaw, ...joints, tip);
    fingers[f] = { base, yaw, joints, tip, spec };
  }
  return { root, forearm, fingers, bones };
}

function setBindPose(sk, s) {
  sk.root.position.set(0, 0, 0);
  sk.root.rotation.set(0, 0, 0);
  sk.forearm.rotation.set(0, 0, 0);
  for (const f of [1, 2, 3, 4, 5]) {
    const fg = sk.fingers[f];
    fg.base.position.x = s * fg.spec.base[0];
    fg.yaw.rotation.y = s * BIND_YAW[f];
    BIND[f].forEach((a, i) => { fg.joints[i].rotation.x = -a; });
  }
  sk.root.updateMatrixWorld(true);
}

// ------------------------------------------------------------------ mesh generation
function buildHandGeometry(handGeo, quality) {
  const s = 1; // generated for the right hand; the left hand mirrors it
  const sk = makeSkeleton(s, handGeo);
  setBindPose(sk, s);
  const wp = (o) => { const v = new THREE.Vector3(); o.getWorldPosition(v); return [v.x, v.y, v.z]; };
  const fingerPrims = [];   // round cones (unioned with small blending)
  const segs = [];          // skinning segments: [boneIndexKey, a, b]
  for (const f of [1, 2, 3, 4, 5]) {
    const fg = sk.fingers[f], spec = fg.spec;
    const pts = [wp(fg.joints[0]), wp(fg.joints[1]), wp(fg.joints[2]), wp(fg.tip)];
    const r = spec.radii;
    const rad = f === 1 ? [r[0] * 1.0, r[1] * 0.9, r[2] * 0.86, r[2] * 0.76] : [r[0] * 0.9, r[1] * 0.86, r[2] * 0.84, r[2] * 0.74];
    for (let i = 0; i < 3; i++) {
      fingerPrims.push({ f, a: pts[i], b: pts[i + 1], r1: rad[i], r2: rad[i + 1] });
      segs.push({ bone: `${f}.${i}`, a: pts[i], b: pts[i + 1] });
    }
  }
  const mcp = [2, 3, 4, 5].map((f) => wp(sk.fingers[f].joints[0]));
  const palmPrims = [];
  // palm: fleshy cones from the wrist fan to each knuckle
  const wristPts = [[-1.9, -0.25, -0.6], [-0.4, 0.0, -0.4], [1.0, 0.0, -0.4], [2.3, -0.2, -0.6]];
  mcp.forEach((m, i) => palmPrims.push({ a: wristPts[i], b: [m[0], m[1] - 0.25, m[2] + 0.2], r1: 1.2, r2: i === 3 ? 0.82 : 0.9 }));
  palmPrims.push({ a: [-1.6, -0.55, -3.0], b: [2.6, -0.55, -4.6], r1: 1.05, r2: 0.9 });   // palm body
  palmPrims.push({ a: [-2.0, -0.45, -6.4], b: [3.3, -0.45, -6.2], r1: 0.9, r2: 0.82 });   // distal palm
  const wrist = { a: [0.2, -0.35, 0.6], b: [0.45, -0.6, 13.0], r1: 1.9, r2: 2.35 };      // wrist + forearm
  const thenar = { c: [-2.0, -0.95, -3.6], r: [1.3, 1.05, 2.4] };
  const hypothenar = { c: [2.3, -0.8, -3.4], r: [1.0, 0.9, 2.5] };
  const knuckles = mcp.map((m, i) => ({ c: m, r: [2, 3, 4, 5].map((f) => handGeo.fingers[String(f)].radii[0])[i] * 0.98 }));

  // bounding box
  const all = [...fingerPrims.flatMap((p) => [p.a, p.b]), ...palmPrims.flatMap((p) => [p.a, p.b]), wrist.a, wrist.b];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of all) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  for (let k = 0; k < 3; k++) { lo[k] -= 3.2; hi[k] += 3.2; }
  hi[2] = Math.min(hi[2], 12.0);
  const h = quality === 'high' ? 0.085 : quality === 'low' ? 0.16 : 0.11;
  const nx = Math.ceil((hi[0] - lo[0]) / h) + 1, ny = Math.ceil((hi[1] - lo[1]) / h) + 1, nz = Math.ceil((hi[2] - lo[2]) / h) + 1;
  const N = nx * ny * nz;
  const F = new Float32Array(N).fill(10);   // fingers (plain min)
  const P = new Float32Array(N).fill(10);   // palm group (smooth union)
  const id = (i, j, k) => i + nx * (j + ny * k);
  const raster = (bbLo, bbHi, fn) => {
    const i0 = Math.max(0, Math.floor((bbLo[0] - lo[0]) / h)), i1 = Math.min(nx - 1, Math.ceil((bbHi[0] - lo[0]) / h));
    const j0 = Math.max(0, Math.floor((bbLo[1] - lo[1]) / h)), j1 = Math.min(ny - 1, Math.ceil((bbHi[1] - lo[1]) / h));
    const k0 = Math.max(0, Math.floor((bbLo[2] - lo[2]) / h)), k1 = Math.min(nz - 1, Math.ceil((bbHi[2] - lo[2]) / h));
    for (let k = k0; k <= k1; k++) {
      const z = lo[2] + k * h;
      for (let j = j0; j <= j1; j++) {
        const y = lo[1] + j * h;
        let o = id(i0, j, k);
        for (let i = i0; i <= i1; i++, o++) fn(o, lo[0] + i * h, y, z);
      }
    }
  };
  const bbOf = (pts, m) => {
    const a = [Infinity, Infinity, Infinity], b = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) for (let k = 0; k < 3; k++) { a[k] = Math.min(a[k], p[k] - m); b[k] = Math.max(b[k], p[k] + m); }
    return [a, b];
  };
  for (const p of fingerPrims) {
    const [a, b] = bbOf([p.a, p.b], Math.max(p.r1, p.r2) + 1.0);
    raster(a, b, (o, x, y, z) => {
      // flatten slightly on the palmar/dorsal axis for a less tubular look
      const d = sdRoundCone(x, p.a[1] + (y - p.a[1]) * 1.06, z, p.a, p.b, p.r1, p.r2);
      if (d < F[o]) F[o] = d;
    });
  }
  const palmK = 1.35;
  for (const p of palmPrims) {
    const [a, b] = bbOf([p.a, p.b], Math.max(p.r1, p.r2) + 2.0);
    raster(a, b, (o, x, y, z) => { P[o] = smin(P[o], sdRoundCone(x, y, z, p.a, p.b, p.r1, p.r2) / 1.0, palmK); });
  }
  {
    const [a, b] = bbOf([wrist.a, wrist.b], 4.5);
    raster(a, b, (o, x, y, z) => {
      const d = sdRoundCone(x, wrist.a[1] + (y - wrist.a[1]) / 0.66, z, wrist.a, wrist.b, wrist.r1, wrist.r2) * 0.75;
      P[o] = smin(P[o], d, 2.4);
    });
  }
  for (const e of [thenar, hypothenar]) {
    const [a, b] = bbOf([e.c], Math.max(...e.r) + 1.5);
    raster(a, b, (o, x, y, z) => { P[o] = smin(P[o], sdEllipsoid(x, y, z, e.c, e.r), 1.1); });
  }
  for (const kn of knuckles) {
    const [a, b] = bbOf([kn.c], kn.r + 1.0);
    raster(a, b, (o, x, y, z) => {
      const dx = x - kn.c[0], dy = y - kn.c[1], dz = z - kn.c[2];
      P[o] = smin(P[o], Math.sqrt(dx * dx + dy * dy + dz * dz) - kn.r, 0.9);
    });
  }
  // combine: fingers blend into the palm, but fingers never web into each other
  const D = new Float32Array(N);
  for (let o = 0; o < N; o++) D[o] = smin(P[o], F[o], 0.55);

  // ---- surface nets
  const cellIndex = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cid = (i, j, k) => i + (nx - 1) * (j + (ny - 1) * k);
  const pos = [];
  const corner = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const v = new Float32Array(8);
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    let mask = 0;
    for (let c = 0; c < 8; c++) {
      v[c] = D[id(i + corner[c][0], j + corner[c][1], k + corner[c][2])];
      if (v[c] < 0) mask |= 1 << c;
    }
    if (mask === 0 || mask === 255) continue;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const [a, b] of edges) {
      if ((v[a] < 0) === (v[b] < 0)) continue;
      const t = v[a] / (v[a] - v[b]);
      sx += corner[a][0] + t * (corner[b][0] - corner[a][0]);
      sy += corner[a][1] + t * (corner[b][1] - corner[a][1]);
      sz += corner[a][2] + t * (corner[b][2] - corner[a][2]);
      n++;
    }
    cellIndex[cid(i, j, k)] = pos.length / 3;
    pos.push(lo[0] + (i + sx / n) * h, lo[1] + (j + sy / n) * h, lo[2] + (k + sz / n) * h);
  }
  const idx = [];
  const quad = (a, b, c, d, flip) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) idx.push(a, b, c, a, c, d); else idx.push(a, c, b, a, d, c);
  };
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const d0 = D[id(i, j, k)] < 0;
    if (d0 !== (D[id(i + 1, j, k)] < 0)) // x edge
      quad(cellIndex[cid(i, j - 1, k - 1)], cellIndex[cid(i, j, k - 1)], cellIndex[cid(i, j, k)], cellIndex[cid(i, j - 1, k)], d0);
    if (d0 !== (D[id(i, j + 1, k)] < 0)) // y edge
      quad(cellIndex[cid(i - 1, j, k - 1)], cellIndex[cid(i - 1, j, k)], cellIndex[cid(i, j, k)], cellIndex[cid(i, j, k - 1)], d0);
    if (d0 !== (D[id(i, j, k + 1)] < 0)) // z edge
      quad(cellIndex[cid(i - 1, j - 1, k)], cellIndex[cid(i, j - 1, k)], cellIndex[cid(i, j, k)], cellIndex[cid(i - 1, j, k)], d0);
  }
  let geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setIndex(idx);
  laplacianSmooth(geo, 2);
  geo.computeVertexNormals();

  // ---- skin weights (smooth, distance based) + subtle skin colouring
  const boneKeys = ['wrist', 'forearm'];
  for (const f of [1, 2, 3, 4, 5]) for (let i = 0; i < 3; i++) boneKeys.push(`${f}.${i}`);
  const bi = Object.fromEntries(boneKeys.map((k, i) => [k, i]));
  const segList = [
    { bone: 'wrist', a: [0.3, -0.2, -0.5], b: [0.9, -0.3, -5.6] },
    { bone: 'wrist', a: [-1.8, -0.4, -2.0], b: [2.4, -0.4, -2.6] },
    { bone: 'forearm', a: [0.3, -0.4, 2.6], b: [0.5, -0.6, 13] },
    ...segs,
  ];
  const P3 = geo.attributes.position;
  const nv = P3.count;
  const skinIndex = new Uint16Array(nv * 4), skinWeight = new Float32Array(nv * 4), color = new Float32Array(nv * 3);
  const tipPts = [1, 2, 3, 4, 5].map((f) => wp(sk.fingers[f].tip));
  const jointPts = [1, 2, 3, 4, 5].flatMap((f) => [wp(sk.fingers[f].joints[1]), wp(sk.fingers[f].joints[2])]);
  for (let q = 0; q < nv; q++) {
    const x = P3.getX(q), y = P3.getY(q), z = P3.getZ(q);
    const best = new Map();
    for (const sg of segList) {
      const d = distSeg(x, y, z, sg.a, sg.b);
      const w = 1 / Math.pow(Math.max(d, 0.05), 6);
      const b = bi[sg.bone];
      best.set(b, Math.max(best.get(b) || 0, w));
    }
    const arr = [...best.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
    const sum = arr.reduce((a, e) => a + e[1], 0);
    arr.forEach(([b, w], i) => { skinIndex[q * 4 + i] = b; skinWeight[q * 4 + i] = w / sum; });
    // colour: slightly redder at the finger tips and knuckle creases, paler on the palm side
    let red = 0;
    for (const t of tipPts) { const d = Math.hypot(x - t[0], y - t[1], z - t[2]); red += Math.exp(-d * d / 0.9); }
    for (const t of jointPts) { const d = Math.hypot(x - t[0], y - t[1], z - t[2]); red += 0.45 * Math.exp(-d * d / 0.5); }
    red = Math.min(1, red);
    const palmar = y < -0.6 && z < -1 ? 0.06 : 0;
    color[q * 3] = 1.0 + palmar * 0.5;
    color[q * 3 + 1] = 1.0 - 0.13 * red + palmar * 0.4;
    color[q * 3 + 2] = 1.0 - 0.12 * red + palmar * 0.4;
  }
  geo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndex, 4));
  geo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeight, 4));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(color, 3));
  return { geo, boneKeys };
}

function distSeg(x, y, z, a, b) {
  const bx = b[0] - a[0], by = b[1] - a[1], bz = b[2] - a[2];
  const px = x - a[0], py = y - a[1], pz = z - a[2];
  const t = Math.max(0, Math.min(1, (px * bx + py * by + pz * bz) / (bx * bx + by * by + bz * bz)));
  const dx = px - bx * t, dy = py - by * t, dz = pz - bz * t;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function laplacianSmooth(geo, iters) {
  const pos = geo.attributes.position, index = geo.index.array, n = pos.count;
  const nb = Array.from({ length: n }, () => new Set());
  for (let i = 0; i < index.length; i += 3) {
    const a = index[i], b = index[i + 1], c = index[i + 2];
    nb[a].add(b).add(c); nb[b].add(a).add(c); nb[c].add(a).add(b);
  }
  const p = pos.array;
  for (let it = 0; it < iters; it++) {
    const q = new Float32Array(p);
    for (let i = 0; i < n; i++) {
      if (!nb[i].size) continue;
      let sx = 0, sy = 0, sz = 0;
      for (const j of nb[i]) { sx += p[j * 3]; sy += p[j * 3 + 1]; sz += p[j * 3 + 2]; }
      const k = nb[i].size;
      q[i * 3] = p[i * 3] * 0.5 + 0.5 * sx / k;
      q[i * 3 + 1] = p[i * 3 + 1] * 0.5 + 0.5 * sy / k;
      q[i * 3 + 2] = p[i * 3 + 2] * 0.5 + 0.5 * sz / k;
    }
    p.set(q);
  }
  pos.needsUpdate = true;
}

function skinMaterial(tone) {
  const color = new THREE.Color(SKIN_TONES[tone] || tone || SKIN_TONES.light);
  return new THREE.MeshStandardMaterial({
    color, vertexColors: true, roughness: 0.6, metalness: 0,
    normalMap: skinNormalMap(), normalScale: new THREE.Vector2(0.12, 0.12),
  });
}

let _skinNormal = null;
function skinNormalMap() {
  if (_skinNormal) return _skinNormal;
  const S = 256, cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(S, S);
  let seed = 99;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const hgt = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) hgt[i] = rnd();
  for (let pass = 0; pass < 2; pass++) {
    const t = new Float32Array(hgt);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += t[((y + dy + S) % S) * S + ((x + dx + S) % S)];
      hgt[y * S + x] = s / 9;
    }
  }
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const hx = hgt[y * S + (x + 1) % S] - hgt[y * S + (x - 1 + S) % S];
    const hy = hgt[((y + 1) % S) * S + x] - hgt[((y - 1 + S) % S) * S + x];
    const i = (y * S + x) * 4;
    img.data[i] = 128 + hx * 400; img.data[i + 1] = 128 + hy * 400; img.data[i + 2] = 255; img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  _skinNormal = new THREE.CanvasTexture(cv);
  _skinNormal.wrapS = _skinNormal.wrapT = THREE.RepeatWrapping;
  _skinNormal.repeat.set(6, 6);
  return _skinNormal;
}

// box-projected UVs so the pore normal map has something to follow
function addUVs(geo) {
  const p = geo.attributes.position, n = geo.attributes.normal;
  const uv = new Float32Array(p.count * 2);
  for (let i = 0; i < p.count; i++) {
    const ax = Math.abs(n.getX(i)), ay = Math.abs(n.getY(i)), az = Math.abs(n.getZ(i));
    let u, v;
    if (ay >= ax && ay >= az) { u = p.getX(i); v = p.getZ(i); } else if (ax >= az) { u = p.getZ(i); v = p.getY(i); } else { u = p.getX(i); v = p.getY(i); }
    uv[i * 2] = u * 0.18; uv[i * 2 + 1] = v * 0.18;
  }
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
}

export function buildHand(side, handGeo, opts = {}) {
  const s = side === 'R' ? 1 : -1;
  const quality = opts.quality || 'medium';
  if (!geometryCache.has(quality)) {
    const g = buildHandGeometry(handGeo, quality);
    addUVs(g.geo);
    geometryCache.set(quality, g);
  }
  const { geo: rhGeo, boneKeys } = geometryCache.get(quality);
  let geo = rhGeo;
  if (s < 0) {
    geo = rhGeo.clone();
    geo.scale(-1, 1, 1);
    const ix = geo.index.array;
    for (let i = 0; i < ix.length; i += 3) { const t = ix[i + 1]; ix[i + 1] = ix[i + 2]; ix[i + 2] = t; }
    geo.computeVertexNormals();
  }
  const sk = makeSkeleton(s, handGeo);
  const boneFor = (key) => {
    if (key === 'wrist') return sk.root;
    if (key === 'forearm') return sk.forearm;
    const [f, i] = key.split('.').map(Number);
    return sk.fingers[f].joints[i];
  };
  const skin = skinMaterial(opts.skinTone || 'light');
  const mesh = new THREE.SkinnedMesh(geo, skin);
  mesh.frustumCulled = false;
  const group = new THREE.Group();
  group.add(sk.root);
  group.add(mesh);
  setBindPose(sk, s);
  // skeleton bones ordered like boneKeys (the skinIndex attribute refers to these)
  const ordered = boneKeys.map(boneFor);
  const extra = sk.bones.filter((b) => !ordered.includes(b));
  const skeleton = new THREE.Skeleton([...ordered, ...extra]);
  group.updateMatrixWorld(true);
  mesh.bind(skeleton, mesh.matrixWorld);

  // nails (rigid, attached to distal phalanges)
  const nailMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(skin.color).lerp(new THREE.Color('#ffe4dc'), 0.5), roughness: 0.25,
  });
  for (const f of [1, 2, 3, 4, 5]) {
    const fg = sk.fingers[f], L3 = fg.spec.lengths[2], r3 = fg.spec.radii[2] * 0.98;
    const nail = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2), nailMat);
    nail.scale.set(r3 * 0.66, r3 * 0.18, L3 * 0.34);
    nail.position.set(0, r3 * 0.86, -L3 * 0.66);
    nail.rotation.x = 0.08;
    fg.joints[2].add(nail);
  }
  // sleeve on the forearm bone
  if (opts.sleeve !== false) {
    const sleeveMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(opts.sleeveColor || '#101014'), roughness: 0.9 });
    const len = 40;
    const sl = new THREE.CylinderGeometry(3.15, 4.4, len, 36, 1, true);
    sl.rotateX(Math.PI / 2);
    sl.translate(0, 0, 9.5 + len / 2);
    sl.scale(1.12, 0.82, 1);
    const sleeve = new THREE.Mesh(sl, sleeveMat);
    const cuff = new THREE.Mesh(new THREE.TorusGeometry(3.15, 0.42, 12, 40), sleeveMat);
    cuff.scale.set(1.12, 0.82, 1);
    cuff.position.z = 9.5;
    sleeve.add(cuff);
    sleeve.position.set(0, -0.45, 0);
    sk.forearm.add(sleeve);
  }
  group.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  const tips = [1, 2, 3, 4, 5].map((f) => sk.fingers[f].tip);
  return { side, s, root: group, wrist: sk.root, forearm: new THREE.Group(), fingers: sk.fingers, tips, skin, mesh, sk };
}

const _q = new THREE.Quaternion();
const _m = new THREE.Matrix4();
const _target = new THREE.Vector3();
const _pos = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
export function applyHandPose(hand, pose) {
  const s = hand.s, w = hand.wrist;
  w.position.set(pose[0], pose[1], pose[2]);
  w.rotation.set(pose[4], pose[3], pose[5]);     // Euler(x=pitch, y=yaw, z=roll), order YXZ
  const splay = pose[6];
  for (let i = 0; i < 5; i++) {
    const f = i + 1, fg = hand.fingers[f];
    fg.base.position.x = s * fg.spec.base[0] * (1 + splay * SPLAY_K[f]);
    fg.yaw.rotation.y = pose[7 + 4 * i];
    fg.joints[0].rotation.x = -pose[8 + 4 * i];
    fg.joints[1].rotation.x = -pose[9 + 4 * i];
    fg.joints[2].rotation.x = -pose[10 + 4 * i];
  }
  // forearm bone points from the wrist towards a plausible elbow near the body
  const bodyX = hand.bodyX ?? 55;
  _target.set(bodyX + s * 24 + (pose[0] - bodyX) * 0.45, pose[1] + 7, pose[2] + 34);
  w.updateMatrixWorld(true);
  _pos.setFromMatrixPosition(w.matrixWorld);
  _m.lookAt(_target, _pos, _up);                  // +z of the bone towards the elbow
  _q.setFromRotationMatrix(_m);
  const wq = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().extractRotation(w.matrixWorld));
  hand.sk.forearm.quaternion.copy(wq.invert().multiply(_q));
}
