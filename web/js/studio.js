// Piano Studio renderer: one deterministic Three.js scene used both for the
// live preview in the app and for frame-accurate video export.
//
// The scene is a pure function of (performance data, theme, options, video
// time).  Nothing depends on wall-clock time, so rendering frame N always
// produces the same image.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { buildHand, applyHandPose } from './hands.js';
import { Overlay } from './overlay.js';

const C = (c) => new THREE.Color(c);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smooth = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };

export class Studio {
  constructor(canvas, { width, height, pixelRatio = 1, preview = false } = {}) {
    this.canvas = canvas;
    this.preview = preview;
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: false, preserveDrawingBuffer: true, alpha: false, powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 16 / 9, 5, 2000);
    this.width = width; this.height = height;
    this.ready = false;
  }

  // ---------------------------------------------------------------- load
  async load(perf, bakeBuffer) {
    this.perf = perf;
    this.opts = perf.options;
    this.theme = perf.theme;
    this.geo = perf.geometry;
    this.bake = new Float32Array(bakeBuffer);
    this.bakeStride = perf.bake.floatsPerHand * 2;
    this.keyByMidi = new Map(this.geo.keys.map((k) => [k.midi, k]));
    this.constants = perf.constants;
    this._buildScene();
    this.overlay = new Overlay(this.width, this.height, perf, this.theme);
    await this.overlay.ready();
    this._buildPost();
    this.setSize(this.width, this.height);
    this.ready = true;
  }

  dispose() {
    this.renderer.dispose();
  }

  _buildScene() {
    const th = this.theme;
    const scene = this.scene;
    scene.clear();
    scene.background = C(th.background.bottom);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    // Image-based reflections are expensive on CPU rasterisers: only glossy
    // surfaces (black keys, lacquer) sample the environment map.
    this.envIntensity = th.lighting.env;
    this.renderer.toneMappingExposure = th.lighting.exposure;

    // --- lights
    const L = th.lighting;
    scene.add(new THREE.HemisphereLight(C(L.ambient), C(th.background.floor), L.ambientIntensity));
    const key = new THREE.DirectionalLight(C(L.key.color), L.key.intensity);
    key.castShadow = true;
    key.shadow.mapSize.set(this.preview ? 1024 : 2048, this.preview ? 1024 : 2048);
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.02;
    key.shadow.radius = 4;
    this.keyLight = key;
    scene.add(key, key.target);
    const fill = new THREE.DirectionalLight(C(L.fill.color), L.fill.intensity);
    const rim = new THREE.DirectionalLight(C(L.rim.color), L.rim.intensity);
    this.fillLight = fill; this.rimLight = rim;
    scene.add(fill, fill.target, rim, rim.target);

    this._computeRange();
    this._buildPiano();
    this._buildNotes();
    this._buildSparks();
    this.hands = {};
    for (const h of ['R', 'L']) {
      const hand = buildHand(h, this.geo.hand, this.opts.hands || {});
      hand.bodyX = this.keyByMidi.get(62).x;
      hand.root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
      scene.add(hand.root);
      scene.add(hand.forearm);
      this.hands[h] = hand;
    }
    this._buildDebug();
  }

  _computeRange() {
    const keys = this.geo.keys;
    const lo = this.perf.range.lo, hi = this.perf.range.hi;
    const kl = this.keyByMidi.get(lo), kh = this.keyByMidi.get(hi);
    this.xL = kl.x - kl.w / 2;
    this.xR = kh.x + kh.w / 2;
    this.xMin = keys[0].x - keys[0].w / 2 - 4;
    this.xMax = keys[keys.length - 1].x + keys[keys.length - 1].w / 2 + 4;
  }

  // ---------------------------------------------------------------- piano
  _buildPiano() {
    const th = this.theme, g = this.geo;
    const group = new THREE.Group();
    this.pianoGroup = group;
    this.scene.add(group);
    const wkMat = new THREE.MeshStandardMaterial({ color: C(th.piano.whiteKey), roughness: th.piano.keyRoughness });
    const bkMat = new THREE.MeshStandardMaterial({
      color: C(th.piano.blackKey), roughness: 0.22, metalness: 0.0, envMap: this.envTex, envMapIntensity: this.envIntensity * 1.6,
    });
    const W = g.white, B = g.black;
    const whiteGeo = new RoundedBoxGeometry(W.w - W.gap, W.h, W.len, 3, 0.18);
    whiteGeo.translate(0, -W.h / 2, W.len / 2);
    const blackGeo = this._blackKeyGeometry(B.w, B.len, B.top + 1.0);
    this.keyMeshes = new Map();
    for (const k of g.keys) {
      const mat = (k.black ? bkMat : wkMat).clone();
      const mesh = new THREE.Mesh(k.black ? blackGeo : whiteGeo, mat);
      mesh.position.set(k.x, 0, 0);
      mesh.castShadow = true; mesh.receiveShadow = true;
      mesh.userData = { key: k, baseColor: mat.color.clone() };
      group.add(mesh);
      this.keyMeshes.set(k.midi, mesh);
    }
    // key bed + front rail (key slip)
    const bodyMat = this._bodyMaterial();
    this.bodyMat = bodyMat;
    const span = this.xMax - this.xMin;
    const slip = new THREE.Mesh(new RoundedBoxGeometry(span, 3.2, 2.2, 3, 0.4), bodyMat);
    slip.position.set((this.xMin + this.xMax) / 2, -2.9, W.len + 1.3);
    slip.receiveShadow = true;
    group.add(slip);
    const bed = new THREE.Mesh(new THREE.BoxGeometry(span, 1, W.len + 6), new THREE.MeshStandardMaterial({ color: '#050505', roughness: 1 }));
    bed.position.set((this.xMin + this.xMax) / 2, -3.0, W.len / 2);
    group.add(bed);
    // cheek blocks
    for (const side of [-1, 1]) {
      const cheek = new THREE.Mesh(new RoundedBoxGeometry(5.0, 6.0, W.len + 6.5, 4, 0.8), bodyMat);
      cheek.position.set(side < 0 ? this.xMin - 0.3 : this.xMax + 0.3, -0.6, W.len / 2 - 1.5);
      cheek.castShadow = true; cheek.receiveShadow = true;
      group.add(cheek);
    }
    // low back rail (keeps the landing line of the falling notes visible) + felt strip
    const back = new THREE.Mesh(new RoundedBoxGeometry(span + 10, 4.2, 3.4, 4, 0.7), bodyMat);
    back.position.set((this.xMin + this.xMax) / 2, -0.35, -2.3);
    back.castShadow = true; back.receiveShadow = true;
    group.add(back);
    const felt = new THREE.Mesh(new THREE.BoxGeometry(span, 0.42, 0.42),
      new THREE.MeshStandardMaterial({ color: C(th.piano.felt), roughness: 1 }));
    felt.position.set((this.xMin + this.xMax) / 2, 0.2, -0.4);
    group.add(felt);
    // floor (seen below the hands)
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(2000, 2000),
      new THREE.MeshBasicMaterial({ color: C(th.background.floor) }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -24;
        this.scene.add(floor);
  }

  _blackKeyGeometry(w, len, h) {
    // tapered top, slightly rounded: built from a rounded box whose top is narrowed
    const geo = new RoundedBoxGeometry(w, h, len, 3, 0.16);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i);
      const t = (y + h / 2) / h;          // 0 bottom .. 1 top
      pos.setX(i, pos.getX(i) * (1 - 0.16 * t));
      const z = pos.getZ(i);
      if (z > 0) pos.setZ(i, z - 0.5 * t * (z / (len / 2)));  // slanted front
    }
    geo.computeVertexNormals();
    geo.translate(0, this.geo.black.top - h / 2, len / 2);
    return geo;
  }

  _bodyMaterial() {
    const p = this.theme.piano;
    if (p.body === 'wood') {
      const tex = woodTexture(p.bodyColor);
      return new THREE.MeshStandardMaterial({ map: tex, roughness: 0.3, envMap: this.envTex, envMapIntensity: this.envIntensity });
    }
    if (p.body === 'matte') {
      return new THREE.MeshStandardMaterial({ color: C(p.bodyColor), roughness: 0.6 });
    }
    return new THREE.MeshStandardMaterial({ color: C(p.bodyColor), roughness: 0.2, envMap: this.envTex, envMapIntensity: this.envIntensity * 0.8 });
  }

  // ---------------------------------------------------------------- falling notes
  _buildNotes() {
    const th = this.theme, notes = this.perf.notes;
    const plane = new THREE.Group();
    this.notePlane = plane;
    this.scene.add(plane);
    const hingeY = 1.95;
    plane.position.set(0, hingeY, -0.55);

    // background lane panel
    const laneMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: {
        uTop: { value: C(th.background.top) }, uBottom: { value: C(th.background.bottom) },
        uLane: { value: C(th.lanes.color) }, uLaneA: { value: th.lanes.opacity }, uOctA: { value: th.lanes.octaveOpacity },
        uLen: { value: 300 }, uHaze: { value: C(th.background.haze) }, uHazeA: { value: th.background.hazeOpacity },
        uLines: { value: this._laneLineTexture() }, uX0: { value: this.xMin }, uX1: { value: this.xMax },
      },
      vertexShader: `varying vec2 vUv; varying vec3 vW; void main(){ vUv=uv; vec4 w=modelMatrix*vec4(position,1.); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }`,
      fragmentShader: `
        uniform vec3 uTop,uBottom,uLane,uHaze; uniform float uLaneA,uOctA,uLen,uHazeA,uX0,uX1; uniform sampler2D uLines;
        varying vec2 vUv; varying vec3 vW;
        void main(){
          float y=vUv.y;
          vec3 col=mix(uBottom,uTop,smoothstep(0.0,0.55,y));
          col+=uHaze*uHazeA*exp(-y*7.0)*0.6;
          float u=(vW.x-uX0)/(uX1-uX0);
          vec4 ln=texture2D(uLines,vec2(u,0.5));
          float fade=1.0-smoothstep(0.25,0.75,y);
          col=mix(col,uLane,(ln.r*uLaneA+ln.g*uOctA)*fade);
          gl_FragColor=vec4(col,1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    const span = this.xMax - this.xMin;
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(span, 300), laneMat);
    panel.position.set((this.xMin + this.xMax) / 2, 150, -0.05);
    panel.renderOrder = -1;
    plane.add(panel);
    this.lanePanel = panel;

    // instanced note quads
    const n = Math.max(1, notes.length);
    const base = new THREE.PlaneGeometry(1, 1);
    base.translate(0, 0.5, 0);
    const geo = new THREE.InstancedBufferGeometry();
    geo.index = base.index;
    geo.attributes.position = base.attributes.position;
    geo.attributes.uv = base.attributes.uv;
    const aX = new Float32Array(n), aW = new Float32Array(n), aT = new Float32Array(n * 2);
    const aCol = new Float32Array(n * 3), aLab = new Float32Array(n * 2), aVel = new Float32Array(n);
    const handCol = { R: C(this.opts.colors.R), L: C(this.opts.colors.L), '?': C('#9aa0aa') };
    notes.forEach((nt, i) => {
      const k = this.keyByMidi.get(nt.p);
      const w = k.black ? this.geo.black.w * 0.92 : (this.geo.white.w - this.geo.white.gap) * 0.86;
      aX[i] = k.x; aW[i] = w;
      aT[i * 2] = nt.t0; aT[i * 2 + 1] = Math.max(nt.t1, nt.t0 + 0.06);
      const c = handCol[nt.h] || handCol['?'];
      aCol[i * 3] = c.r; aCol[i * 3 + 1] = c.g; aCol[i * 3 + 2] = c.b;
      aLab[i * 2] = nt.f ? nt.f : 0;            // finger label cell (1..5)
      aLab[i * 2 + 1] = 8 + (nt.p - 21);       // note-name cell
      aVel[i] = nt.v / 127;
    });
    geo.setAttribute('aX', new THREE.InstancedBufferAttribute(aX, 1));
    geo.setAttribute('aW', new THREE.InstancedBufferAttribute(aW, 1));
    geo.setAttribute('aT', new THREE.InstancedBufferAttribute(aT, 2));
    geo.setAttribute('aCol', new THREE.InstancedBufferAttribute(aCol, 3));
    geo.setAttribute('aLab', new THREE.InstancedBufferAttribute(aLab, 2));
    geo.setAttribute('aVel', new THREE.InstancedBufferAttribute(aVel, 1));
    geo.instanceCount = notes.length;
    const style = { flat: 0, soft: 1, neon: 2 }[th.notes.style] ?? 1;
    this.noteMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: true,
      uniforms: {
        uTime: { value: 0 }, uSpeed: { value: 20 }, uRadius: { value: th.notes.radius },
        uGlow: { value: th.notes.glow * (this.opts.glow ?? 1) }, uOpacity: { value: th.notes.opacity },
        uStyle: { value: style }, uAtlas: { value: this._labelAtlas() },
        uShowFinger: { value: this.opts.showFingers ? 1 : 0 }, uShowName: { value: this.opts.showNoteNames ? 1 : 0 },
        uMaxLen: { value: 400 },
      },
      vertexShader: `
        attribute float aX, aW, aVel; attribute vec2 aT, aLab; attribute vec3 aCol;
        uniform float uTime, uSpeed, uMaxLen;
        varying vec2 vLocal; varying vec2 vSize; varying vec3 vCol; varying float vActive; varying vec2 vLab; varying float vClip; varying float vVel; varying float vTop;
        void main(){
          float y0=(aT.x-uTime)*uSpeed, y1=(aT.y-uTime)*uSpeed;
          float yc0=max(y0,0.0);                 // consumed at the hit line
          float h=max(y1-yc0,0.0);
          float pad=1.7;
          vSize=vec2(aW,h);
          vec3 p=position;
          float x=aX+(p.x)*(aW+pad*2.0);
          float y=yc0-pad+p.y*(h+pad*2.0);
          vLocal=vec2((p.x)*(aW+pad*2.0), y-yc0);
          vCol=aCol; vLab=aLab; vVel=aVel;
          vActive=step(aT.x,uTime)*step(uTime,aT.y);
          vClip=y0<0.0?1.0:0.0;
          vTop=y;
          if(h<=0.0||y0>uMaxLen){ gl_Position=vec4(2.0,2.0,2.0,1.0); return; }
          gl_Position=projectionMatrix*modelViewMatrix*vec4(x,y,0.02,1.0);
        }`,
      fragmentShader: `
        uniform float uRadius,uGlow,uOpacity,uStyle,uShowFinger,uShowName,uTime; uniform sampler2D uAtlas;
        varying vec2 vLocal; varying vec2 vSize; varying vec3 vCol; varying float vActive; varying vec2 vLab; varying float vClip; varying float vVel; varying float vTop;
        float sdRound(vec2 p, vec2 b, float r){ vec2 q=abs(p)-b+r; return length(max(q,0.0))+min(max(q.x,q.y),0.0)-r; }
        vec4 label(float cell, vec2 uv){
          float cx=mod(cell,16.0), cy=floor(cell/16.0);
          return texture2D(uAtlas, (vec2(cx,cy)+uv)/vec2(16.0,8.0));
        }
        void main(){
          vec2 half_=vSize*0.5;
          vec2 p=vLocal-vec2(0.0,half_.y);
          float r=min(uRadius,min(half_.x,half_.y));
          float d=sdRound(p,half_,r);
          float fill=1.0-smoothstep(-0.06,0.06,d);
          float edge=1.0-smoothstep(0.0,0.12,abs(d+0.12));
          float glow=exp(-max(d,0.0)*2.6)*(1.0-fill);
          vec3 c=vCol;
          float yy=clamp(vLocal.y/max(vSize.y,0.001),0.0,1.0);
          vec3 body;
          if(uStyle<0.5){            // flat editorial
            body=c*(0.92+0.08*yy);
            glow*=0.35;
          } else if(uStyle<1.5){     // soft gradient
            body=mix(c*1.25,c*0.78,yy)+edge*0.18;
          } else {                   // neon: dark core, bright rim
            body=mix(c*0.35,c*0.15,yy)+edge*c*1.9;
          }
          body*=0.85+0.3*vVel;
          float act=vActive;
          body=mix(body,body*1.18+0.04,act);
          float a=fill*uOpacity + glow*uGlow*(0.7+0.5*act);
          vec3 col=body*fill + c*glow*uGlow*(1.1+1.2*act);
          // labels
          float lh=min(vSize.x*1.0, 2.2);
          if(vSize.y>lh*0.95 && fill>0.5){
            vec2 luv=vec2((vLocal.x+vSize.x*0.5)/vSize.x, (vLocal.y-0.12)/ (vSize.x));
            if(uShowFinger>0.5 && vLab.x>0.5 && luv.y>=0.0 && luv.y<=1.0){
              vec4 L=label(vLab.x,vec2(luv.x,1.0-luv.y));
              col=mix(col,vec3(1.0),L.a*0.95); a=max(a,L.a);
            } else if(uShowName>0.5 && luv.y>=0.0 && luv.y<=1.0){
              vec4 L=label(vLab.y,vec2(luv.x,1.0-luv.y));
              col=mix(col,vec3(1.0),L.a*0.9); a=max(a,L.a);
            }
          }
          if(a<0.003) discard;
          gl_FragColor=vec4(col, clamp(a,0.0,1.0));
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    this.noteMesh = new THREE.Mesh(geo, this.noteMat);
    this.noteMesh.frustumCulled = false;
    this.noteMesh.renderOrder = 2;
    plane.add(this.noteMesh);

    // hit-line glow strips per key
    const hitGeo = new THREE.PlaneGeometry(1, 1);
    this.hitMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: {},
      vertexShader: `attribute float aI; attribute vec3 aC; varying vec2 vUv; varying float vI; varying vec3 vC;
        void main(){ vUv=uv; vI=aI; vC=aC; gl_Position=projectionMatrix*modelViewMatrix*instanceMatrix*vec4(position,1.0); }`,
      fragmentShader: `varying vec2 vUv; varying float vI; varying vec3 vC;
        void main(){ float dx=abs(vUv.x-0.5)*2.0; float dy=vUv.y;
          float a=exp(-dy*4.0)*(1.0-dx*dx)*vI; if(a<0.002) discard;
          gl_FragColor=vec4(vC*a*3.2,a);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    const nk = this.geo.keys.length;
    this.hitMesh = new THREE.InstancedMesh(hitGeo, this.hitMat, nk);
    this.hitI = new Float32Array(nk);
    this.hitC = new Float32Array(nk * 3);
    hitGeo.setAttribute('aI', new THREE.InstancedBufferAttribute(this.hitI, 1));
    hitGeo.setAttribute('aC', new THREE.InstancedBufferAttribute(this.hitC, 3));
    const m = new THREE.Matrix4();
    this.geo.keys.forEach((k, i) => {
      const w = k.black ? 3.2 : 4.0;
      m.compose(new THREE.Vector3(k.x, 3.5, 0.03), new THREE.Quaternion(), new THREE.Vector3(w, 7.0, 1));
      this.hitMesh.setMatrixAt(i, m);
    });
    this.hitMesh.frustumCulled = false;
    this.hitMesh.renderOrder = 3;
    plane.add(this.hitMesh);

    // index notes per key for fast key-state queries
    this.notesByKey = new Map();
    notes.forEach((nt) => {
      if (!this.notesByKey.has(nt.p)) this.notesByKey.set(nt.p, []);
      this.notesByKey.get(nt.p).push(nt);
    });
  }

  _laneLineTexture() {
    const W = 4096, cv = document.createElement('canvas');
    cv.width = W; cv.height = 1;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, 1);
    const toU = (x) => ((x - this.xMin) / (this.xMax - this.xMin)) * W;
    for (const k of this.geo.keys) {
      if (k.black) continue;
      const left = k.x - this.geo.white.w / 2;
      const isC = k.name.startsWith('C') && !k.name.startsWith('C#');
      const isF = k.name.startsWith('F') && !k.name.startsWith('F#');
      ctx.fillStyle = isC ? 'rgb(0,255,0)' : isF ? 'rgb(0,110,0)' : 'rgb(255,0,0)';
      ctx.fillRect(Math.round(toU(left)), 0, isC ? 2 : 1, 1);
    }
    const tex = new THREE.CanvasTexture(cv);
    tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
    return tex;
  }

  _labelAtlas() {
    const cell = 96, cv = document.createElement('canvas');
    cv.width = cell * 16; cv.height = cell * 8;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = '#fff';
    const draw = (idx, text, size) => {
      const cx = (idx % 16) * cell + cell / 2, cy = Math.floor(idx / 16) * cell + cell / 2;
      ctx.font = `600 ${size}px Inter, sans-serif`;
      ctx.shadowColor = 'rgba(0,0,0,0.55)'; ctx.shadowBlur = 6;
      ctx.fillText(text, cx, cy + 2);
    };
    for (let f = 1; f <= 5; f++) draw(f, String(f), 66);
    for (const k of this.geo.keys) draw(8 + (k.midi - 21), k.name.replace(/\d+$/, ''), k.name.length > 2 ? 40 : 52);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.NoColorSpace;
    tex.generateMipmaps = true;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.anisotropy = 4;
    return tex;
  }

  // ---------------------------------------------------------------- sparks
  _buildSparks() {
    const notes = this.perf.notes;
    const per = 7;
    const n = Math.max(1, notes.length * per);
    const base = new THREE.PlaneGeometry(1, 1);
    const geo = new THREE.InstancedBufferGeometry();
    geo.index = base.index; geo.attributes.position = base.attributes.position; geo.attributes.uv = base.attributes.uv;
    const aS = new Float32Array(n * 4), aC = new Float32Array(n * 3);
    let rng = mulberry32(1234567);
    const handCol = { R: C(this.opts.colors.R), L: C(this.opts.colors.L), '?': C('#cccccc') };
    let j = 0;
    for (const nt of notes) {
      const k = this.keyByMidi.get(nt.p);
      const c = handCol[nt.h] || handCol['?'];
      for (let q = 0; q < per; q++) {
        aS[j * 4] = k.x + (rng() - 0.5) * (k.black ? 1.0 : 1.6);
        aS[j * 4 + 1] = nt.t0 + rng() * 0.05;
        aS[j * 4 + 2] = rng();
        aS[j * 4 + 3] = rng();
        aC[j * 3] = c.r; aC[j * 3 + 1] = c.g; aC[j * 3 + 2] = c.b;
        j++;
      }
    }
    geo.setAttribute('aS', new THREE.InstancedBufferAttribute(aS, 4));
    geo.setAttribute('aC', new THREE.InstancedBufferAttribute(aC, 3));
    geo.instanceCount = j;
    this.sparkMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uTime: { value: 0 }, uAmt: { value: (this.opts.sparks ?? 1) * Math.min(1.2, this.theme.notes.glow + 0.2) } },
      vertexShader: `attribute vec4 aS; attribute vec3 aC; uniform float uTime; varying vec2 vUv; varying float vA; varying vec3 vC;
        void main(){ float dt=uTime-aS.y; vUv=uv; vC=aC;
          if(dt<0.0||dt>0.9){ gl_Position=vec4(2.,2.,2.,1.); vA=0.; return; }
          float life=dt/0.9;
          float vy=6.0+aS.z*9.0; float vx=(aS.w-0.5)*5.0;
          vec3 c=vec3(aS.x+vx*dt, vy*dt-6.0*dt*dt+0.3, 0.05);
          float s=(0.22+0.25*aS.z)*(1.0-life*0.6);
          vA=(1.0-life)*(1.0-life);
          gl_Position=projectionMatrix*modelViewMatrix*vec4(c+vec3(position.xy*s,0.0),1.0);
        }`,
      fragmentShader: `uniform float uAmt; varying vec2 vUv; varying float vA; varying vec3 vC;
        void main(){ float d=length(vUv-0.5)*2.0; float a=exp(-d*d*4.0)*vA*uAmt; if(a<0.003) discard;
          gl_FragColor=vec4(mix(vC,vec3(1.0),0.35)*a*3.0,a);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    this.sparks = new THREE.Mesh(geo, this.sparkMat);
    this.sparks.frustumCulled = false;
    this.sparks.renderOrder = 4;
    this.notePlane.add(this.sparks);
  }

  _buildDebug() {
    this.debugGroup = new THREE.Group();
    this.debugGroup.visible = !!this.opts.debug;
    this.scene.add(this.debugGroup);
    this.debugTips = {};
    const sph = new THREE.SphereGeometry(0.35, 12, 8);
    for (const h of ['R', 'L']) {
      this.debugTips[h] = [];
      for (let f = 0; f < 5; f++) {
        const m = new THREE.Mesh(sph, new THREE.MeshBasicMaterial({ color: '#00ff88', depthTest: false, transparent: true, opacity: 0.85 }));
        m.renderOrder = 10;
        this.debugGroup.add(m);
        this.debugTips[h].push(m);
      }
    }
  }

  // ---------------------------------------------------------------- post
  _buildPost() {
    const th = this.theme;
    this.cinematic = !!this.opts.cinematic;
    // Fast path (default): scene straight to the canvas with tone mapping,
    // then one blended full-screen quad for vignette, fades and titles.
    const q = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      transparent: true, depthTest: false, depthWrite: false,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      uniforms: { tOverlay: { value: this.overlay.texture }, uVignette: { value: th.post.vignette },
        uFade: { value: 1 }, uAspect: { value: this.width / this.height }, uGrain: { value: th.post.grain }, uSeed: { value: 0 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position.xy,0.0,1.0); }',
      fragmentShader: `uniform sampler2D tOverlay; uniform float uVignette,uFade,uAspect,uGrain,uSeed; varying vec2 vUv;
        float hash(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
        void main(){
          vec2 q=(vUv-0.5)*vec2(uAspect,1.0);
          float v=1.0-uVignette*smoothstep(0.35,1.15,length(q));
          v*=1.0-uGrain*hash(gl_FragCoord.xy+uSeed);
          vec4 o=texture2D(tOverlay,vUv);
          float keep=uFade*v*(1.0-o.a);
          gl_FragColor=vec4(o.rgb*uFade, 1.0-keep);
        }`,
    }));
    q.frustumCulled = false;
    this.quadScene = new THREE.Scene();
    this.quadScene.add(q);
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = q;
    if (!this.cinematic) {
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      return;
    }
    const rt = new THREE.WebGLRenderTarget(this.width, this.height, {
      type: THREE.HalfFloatType, samples: this.opts.msaa ?? 0,
    });
    this.composer = new EffectComposer(this.renderer, rt);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(this.width, this.height),
      th.post.bloomStrength, th.post.bloomRadius, th.post.bloomThreshold);
    if (th.post.bloomStrength > 0.01) this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.finalPass = new ShaderPass({
      uniforms: {
        tDiffuse: { value: null }, tOverlay: { value: this.overlay.texture },
        uVignette: { value: th.post.vignette }, uGrain: { value: th.post.grain }, uFade: { value: 1 },
        uSeed: { value: 0 }, uAspect: { value: this.width / this.height },
      },
      vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `uniform sampler2D tDiffuse, tOverlay; uniform float uVignette,uGrain,uFade,uSeed,uAspect; varying vec2 vUv;
        float hash(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
        void main(){
          vec4 c=texture2D(tDiffuse,vUv);
          vec2 q=(vUv-0.5)*vec2(uAspect,1.0);
          float v=1.0-uVignette*smoothstep(0.35,1.15,length(q));
          vec3 col=c.rgb*v;
          col+= (hash(vUv*1000.0+uSeed)-0.5)*uGrain;
          col*=uFade;
          vec4 o=texture2D(tOverlay,vUv);
          col=col*(1.0-o.a)+o.rgb;
          gl_FragColor=vec4(col,1.0);
        }`,
    });
    this.finalPass.uniforms.tOverlay.value = this.overlay.texture;   // ShaderPass clones uniforms
    this.composer.addPass(this.finalPass);
  }

  setSize(w, h) {
    this.width = w; this.height = h;
    this.renderer.setSize(w, h, false);
    if (this.composer) this.composer.setSize(w, h);
    if (this.finalPass) this.finalPass.uniforms.uAspect.value = w / h;
    if (this.quad) this.quad.material.uniforms.uAspect.value = w / h;
    if (this.overlay) this.overlay.resize(w, h);
    this.camera.aspect = w / h;
    this._frameCamera();
  }

  // ---------------------------------------------------------------- camera
  _frameCamera(intro = 0) {
    const th = this.theme, aspect = this.width / this.height;
    const portrait = aspect < 1;
    const cam = this.camera;
    cam.fov = th.camera.fov * (portrait ? 1.35 : 1);
    const tilt = THREE.MathUtils.degToRad(th.camera.tilt + intro * 8);
    const pad = portrait ? 2.0 : 6.0;
    const width = (this.xR - this.xL) + pad * 2;
    const xc = (this.xL + this.xR) / 2;
    const vfov = THREE.MathUtils.degToRad(cam.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * aspect);
    const dist = (width / 2) / Math.tan(hfov / 2) * (1 + intro * 0.12);
    const target = new THREE.Vector3(xc, 0, 9.0);
    const dir = new THREE.Vector3(0, Math.cos(tilt), Math.sin(tilt));
    cam.position.copy(target).addScaledVector(dir, dist);
    cam.up.set(0, 1, 0);
    cam.lookAt(target);
    cam.near = Math.max(5, dist * 0.2); cam.far = dist * 6;
    cam.updateProjectionMatrix();
    if (this.opts.cameraOverride) {           // debug / alternative camera angles
      const o = this.opts.cameraOverride;
      cam.position.set(...o.pos); cam.lookAt(new THREE.Vector3(...o.target));
      cam.fov = o.fov || cam.fov; cam.near = 1; cam.far = 3000; cam.updateProjectionMatrix();
    }
    // lens shift: put the keyboard in the lower part of the frame (keeps the
    // geometry perspective-correct rather than tilting the camera further)
    const keyScreenY = portrait ? (this.opts.shortsKeyY ?? 0.70) : (this.opts.keyY ?? 0.64);
    const shift = 1 - 2 * keyScreenY;   // desired NDC y of the target
    if (!this.opts.cameraOverride) cam.projectionMatrix.elements[9] = -shift;
    cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
    // note plane faces the camera (no foreshortening of falling notes)
    // plane normal must equal the camera->target direction: rotate by (tilt - 90deg) about x
    this.notePlane.rotation.set(tilt - Math.PI / 2, 0, 0);
    // visible plane length: intersect the top-of-frame ray with the plane
    cam.updateMatrixWorld();
    this.notePlane.updateMatrixWorld();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(0, 1), cam);
    const n = new THREE.Vector3(0, 0, 1).applyQuaternion(this.notePlane.quaternion);
    const pl = new THREE.Plane().setFromNormalAndCoplanarPoint(n, this.notePlane.position);
    const hit = new THREE.Vector3();
    let len = 120;
    if (ray.ray.intersectPlane(pl, hit)) len = hit.sub(this.notePlane.position).length();
    this.visibleLen = len;
    const travel = this.opts.travelTime ?? 3.0;
    this.noteMat.uniforms.uSpeed.value = len / travel;
    this.noteMat.uniforms.uMaxLen.value = len * 1.05;
    // shadows follow the framed range
    const kl = this.keyLight, L = th.lighting;
    kl.position.set(xc + L.key.pos[0], L.key.pos[1], L.key.pos[2] + 10);
    kl.target.position.set(xc, 0, 12);
    const sc = kl.shadow.camera;
    const half = width * 0.6 + 20;
    sc.left = -half; sc.right = half; sc.top = half * 0.6; sc.bottom = -half * 0.6;
    sc.near = 10; sc.far = 600; sc.updateProjectionMatrix();
    this.fillLight.position.set(xc + L.fill.pos[0], L.fill.pos[1], L.fill.pos[2]);
    this.fillLight.target.position.set(xc, 0, 10);
    this.rimLight.position.set(xc + L.rim.pos[0], L.rim.pos[1], L.rim.pos[2]);
    this.rimLight.target.position.set(xc, 0, 10);
  }

  // ---------------------------------------------------------------- per-frame
  programAt(T) {
    const segs = this.perf.program.segments;
    let seg = segs[segs.length - 1];
    for (const s of segs) { if (T >= s.start && T < s.end) { seg = s; break; } }
    const local = T - seg.start;
    let song = seg.songStart ?? 0;
    if (seg.songStart !== undefined) song = seg.songStart + local * (seg.rate ?? 1);
    return { seg, local, song };
  }

  keyDepth(midi, t) {
    const lst = this.notesByKey.get(midi);
    if (!lst) return [0, null];
    const D = this.constants.KEY_DOWN_TIME, U = this.constants.KEY_UP_TIME;
    let best = 0, who = null;
    for (const nt of lst) {
      if (t < nt.on - D || t > nt.off + U) continue;
      const down = clamp((t - (nt.on - D)) / D, 0, 1);
      const up = clamp((t - nt.off) / U, 0, 1);
      const v = Math.min(down, 1 - up);
      if (v > best) { best = v; who = nt; }
    }
    return [best, who];
  }

  render(T) {
    if (!this.ready) return;
    const { seg, local, song } = this.programAt(T);
    const dur = seg.end - seg.start;
    // camera intro move
    let introAmt = 0;
    if (seg.type === 'intro') introAmt = 1 - smooth(local / dur);
    if (Math.abs(introAmt - (this._lastIntro ?? -1)) > 1e-5) { this._frameCamera(introAmt); this._lastIntro = introAmt; }
    // practice mode shows finger numbers
    const fingers = seg.showFingers ?? this.opts.showFingers;
    this.noteMat.uniforms.uShowFinger.value = fingers ? 1 : 0;
    this.noteMat.uniforms.uTime.value = song;
    this.sparkMat.uniforms.uTime.value = song;
    // keys
    const th = this.theme;
    const handCol = { R: C(this.opts.colors.R), L: C(this.opts.colors.L) };
    const tint = th.notes.pressedTint;
    this.geo.keys.forEach((k, i) => {
      const mesh = this.keyMeshes.get(k.midi);
      const [amt, nt] = this.keyDepth(k.midi, song);
      const depth = amt * (k.black ? this.geo.black.depth : this.geo.white.depth);
      mesh.position.y = -depth;
      const mat = mesh.material;
      if (nt && amt > 0) {
        const hc = handCol[nt.h] || handCol.R;
        mat.color.copy(mesh.userData.baseColor).lerp(hc, tint * amt);
        mat.emissive.copy(hc).multiplyScalar(0.18 * amt * (th.notes.glow > 1 ? 1.6 : 1));
        this.hitI[i] = amt * (0.6 + 0.4 * nt.v / 127) * Math.min(1.5, th.notes.glow + 0.3);
        this.hitC[i * 3] = hc.r; this.hitC[i * 3 + 1] = hc.g; this.hitC[i * 3 + 2] = hc.b;
      } else {
        mat.color.copy(mesh.userData.baseColor);
        mat.emissive.setRGB(0, 0, 0);
        this.hitI[i] = 0;
      }
    });
    this.hitMesh.geometry.attributes.aI.needsUpdate = true;
    this.hitMesh.geometry.attributes.aC.needsUpdate = true;
    // hands
    const b = this.perf.bake;
    const fi = clamp((song - b.t0) * b.rate, 0, b.count - 1.0001);
    const i0 = Math.floor(fi), w = fi - i0;
    const fph = b.floatsPerHand;
    ['R', 'L'].forEach((h, hi) => {
      const pose = new Float32Array(fph);
      const o0 = i0 * this.bakeStride + hi * fph, o1 = o0 + this.bakeStride;
      for (let j = 0; j < fph; j++) pose[j] = this.bake[o0 + j] * (1 - w) + this.bake[o1 + j] * w;
      const hand = this.hands[h];
      hand.root.visible = b.visible[h];
      hand.forearm.visible = b.visible[h];
      applyHandPose(hand, pose);
      if (this.debugGroup.visible) {
        hand.root.updateMatrixWorld(true);
        hand.tips.forEach((tip, f) => tip.getWorldPosition(this.debugTips[h][f].position));
      }
    });
    // overlays & post
    const progress = this._progress(T);
    this.overlay.update(T, seg, local, song, progress);
    const fade = this._fade(T), seed = (Math.floor(T * this.perf.program.fps) % 97) * 7.31;
    if (this.cinematic) {
      this.finalPass.uniforms.uFade.value = fade;
      this.finalPass.uniforms.uSeed.value = seed;
      this.composer.render();
    } else {
      const u = this.quad.material.uniforms;
      u.uFade.value = fade; u.uSeed.value = seed;
      this.renderer.autoClear = true;
      this.renderer.render(this.scene, this.camera);
      this.renderer.autoClear = false;
      this.renderer.render(this.quadScene, this.quadCam);
      this.renderer.autoClear = true;
    }
  }

  _progress(T) {
    const segs = this.perf.program.segments.filter((s) => s.type === 'performance');
    if (!segs.length) return 0;
    const s = segs[0];
    return clamp((T - s.start) / (s.end - s.start), 0, 1);
  }

  _fade(T) {
    const P = this.perf.program;
    const fin = Math.min(1, T / 0.8);
    const fout = Math.min(1, (P.duration - T) / 1.0);
    let f = smooth(fin) * smooth(fout);
    // soft dip between segments
    for (const s of P.segments) {
      if (s.dipIn) f *= smooth(Math.abs(T - s.start) / 0.35 + (T < s.start ? 1 : 0));
    }
    return clamp(f, 0, 1);
  }
}

// ------------------------------------------------------------------ helpers
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function woodTexture(base) {
  const cv = document.createElement('canvas');
  cv.width = 1024; cv.height = 256;
  const ctx = cv.getContext('2d');
  const c = new THREE.Color(base);
  const rnd = mulberry32(42);
  const img = ctx.createImageData(cv.width, cv.height);
  const phases = Array.from({ length: 6 }, () => rnd() * 6.28);
  for (let y = 0; y < cv.height; y++) {
    for (let x = 0; x < cv.width; x++) {
      const u = x / cv.width, v = y / cv.height;
      let g = Math.sin((v * 38 + Math.sin(u * 6 + phases[0]) * 1.4 + Math.sin(u * 17 + phases[1]) * 0.35) * 3.1);
      g = 0.5 + 0.5 * g;
      const fine = Math.sin((v * 260 + Math.sin(u * 40 + phases[2]) * 2.0)) * 0.5 + 0.5;
      const k = 0.78 + 0.18 * g + 0.06 * fine;
      const i = (y * cv.width + x) * 4;
      img.data[i] = Math.min(255, c.r * 255 * k * 1.15);
      img.data[i + 1] = Math.min(255, c.g * 255 * k * 1.05);
      img.data[i + 2] = Math.min(255, c.b * 255 * k);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 1);
  return tex;
}
