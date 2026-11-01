import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { board, gpu, macStudio, makeKit, rig, spark, type Kit, type Tick } from "./studio/hw";

export type Variant = "home" | "basics" | "inside" | "files" | "arch" | "train" | "use";

const rand = (a: number, b: number): number => a + Math.random() * (b - a);

const tokenStream = (accent: THREE.Color, accent2: THREE.Color, ticks: Tick[], count: number, target: THREE.Vector3): THREE.Points => {
  const seed = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) seed.set([Math.random(), rand(4, 13), rand(-1.8, 3.8), Math.random()], i * 4);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  geo.setAttribute("seed", new THREE.BufferAttribute(seed, 4));
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uA: { value: accent }, uB: { value: accent2 }, uT: { value: target }, uPx: { value: Math.min(devicePixelRatio, 1.75) } },
    vertexShader: `
      attribute vec4 seed; uniform float uTime; uniform vec3 uT; uniform float uPx; varying float vK; varying float vH;
      void main(){
        float k=fract(seed.x+uTime*(.05+seed.w*.06));
        float r=seed.y*(1.-k)+.3;
        float a=seed.x*6.2831+k*(5.+seed.w*4.)+uTime*.1;
        vec3 p=vec3(cos(a)*r, mix(seed.z,uT.y,pow(k,1.6)), sin(a)*r)+vec3(uT.x,0.,uT.z);
        vK=k; vH=seed.w;
        vec4 mv=modelViewMatrix*vec4(p,1.);
        gl_PointSize=(6.+seed.w*10.)*uPx*(1.-k*.6)/-mv.z*2.;
        gl_Position=projectionMatrix*mv;
      }`,
    fragmentShader: `
      uniform vec3 uA; uniform vec3 uB; varying float vK; varying float vH;
      void main(){
        vec2 c=gl_PointCoord-.5; float d=length(c); float sq=max(abs(c.x),abs(c.y));
        float shape=mix(smoothstep(.5,.1,d), step(sq,.32)*.9, step(.7,vH));
        float a=shape*smoothstep(0.,.12,vK)*smoothstep(1.,.85,vK);
        gl_FragColor=vec4(mix(uB,uA,vK)*1.4,a);
      }`,
  });
  const pts = new THREE.Points(geo, mat);
  pts.frustumCulled = false;
  ticks.push((t) => { mat.uniforms.uTime.value = t; });
  return pts;
};

const dust = (accent: THREE.Color, n: number): THREE.Points => {
  const p = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) p.set([rand(-30, 30), rand(-2, 16), rand(-30, 20)], i * 3);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ color: accent, size: 0.03, transparent: true, opacity: 0.35, depthWrite: false, blending: THREE.AdditiveBlending }));
};

const studioEnv = (renderer: THREE.WebGLRenderer, accent: THREE.Color): THREE.Texture => {
  const env = new THREE.Scene();
  env.add(new THREE.Mesh(new THREE.BoxGeometry(24, 12, 24), new THREE.MeshBasicMaterial({ color: 0x030304, side: THREE.BackSide })));
  const panel = (w: number, h: number, c: THREE.Color, x: number, y: number, z: number): void => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: c, side: THREE.DoubleSide }));
    m.position.set(x, y, z);
    m.lookAt(0, 0, 0);
    env.add(m);
  };
  const white = (k: number): THREE.Color => new THREE.Color(1, 0.97, 0.93).multiplyScalar(k);
  panel(9, 5, white(2.6), 0, 5.8, 1);
  panel(1.2, 7, white(5), -9, 1.5, 4);
  panel(0.8, 7, white(3.2), 8.5, 1.5, -1);
  panel(12, 0.5, white(1.6), 0, 0.6, 10);
  panel(5, 0.9, accent.clone().multiplyScalar(1.3), 2, 1.2, -10);
  const pm = new THREE.PMREMGenerator(renderer);
  const tx = pm.fromScene(env, 0.015).texture;
  pm.dispose();
  return tx;
};

const lens = {
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uRes: { value: new THREE.Vector2(1, 1) } },
  vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `uniform sampler2D tDiffuse; uniform float uTime; uniform vec2 uRes; varying vec2 vUv;
    float h(vec2 p){ return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453); }
    void main(){
      vec2 c=vUv-.5; float r=dot(c,c); vec2 o=c*r*.012;
      vec3 col=vec3(texture2D(tDiffuse,vUv+o).r, texture2D(tDiffuse,vUv).g, texture2D(tDiffuse,vUv-o).b);
      col*=1.-r*1.1;
      col+=(h(vUv*uRes+fract(uTime*7.))-.5)*.028;
      gl_FragColor=vec4(col,1.);
    }`,
};

interface Shot { obj: THREE.Object3D; cam: THREE.Vector3; look: THREE.Vector3; x: number }

const floorPath = (accent: THREE.Color, ticks: Tick[]): THREE.Group => {
  const g = new THREE.Group();
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-6, 0.01, 3), new THREE.Vector3(-3.4, 0.01, 1.6), new THREE.Vector3(-1.2, 0.01, 1.6),
    new THREE.Vector3(1.2, 0.01, 1.7), new THREE.Vector3(3.4, 0.01, 1.4), new THREE.Vector3(6.5, 0.01, -2),
  ]);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uT: { value: 0 }, uC: { value: accent } },
    vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform float uT; uniform vec3 uC; varying vec2 vUv;
      void main(){ float p=fract(vUv.x*3.-uT*.25); float a=.12+pow(p,18.)*2.; a*=smoothstep(0.,.08,vUv.x)*smoothstep(1.,.9,vUv.x); gl_FragColor=vec4(uC*a,a); }`,
  });
  g.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 300, 0.012, 6), mat));
  ticks.push((t) => { mat.uniforms.uT.value = t; });
  return g;
};

const tensor = (accent: THREE.Color, accent2: THREE.Color, ticks: Tick[]): THREE.Group => {
  const g = new THREE.Group();
  const nx = 7, ny = 4, nz = 7, gap = 0.36, n = nx * ny * nz;
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.35, roughness: 0.38 });
  const cubes = new THREE.InstancedMesh(new RoundedBoxGeometry(0.25, 0.25, 0.25, 2, 0.045), mat, n);
  const o = new THREE.Object3D();
  const cell: THREE.Vector3[] = [];
  let i = 0;
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
    const p = new THREE.Vector3((x - (nx - 1) / 2) * gap, 0.4 + y * gap, (z - (nz - 1) / 2) * gap);
    cell.push(p);
    o.position.copy(p);
    o.updateMatrix();
    cubes.setMatrixAt(i++, o.matrix);
  }
  const dark = new THREE.Color(0x101216), c = new THREE.Color();
  for (let j = 0; j < n; j++) cubes.setColorAt(j, dark);
  cubes.castShadow = true;
  g.add(cubes);
  ticks.push((t) => {
    for (let j = 0; j < n; j++) {
      const p = cell[j];
      const w = 0.5 + 0.5 * Math.sin(p.x * 2.4 + p.z * 1.3 - t * 1.5 + Math.sin(p.y * 3 + t * 0.6));
      const k = Math.pow(w, 7);
      c.copy(dark).lerp(j % 3 ? accent : accent2, Math.min(1, k * 1.1)).multiplyScalar(1 + k * 1.8);
      cubes.setColorAt(j, c);
    }
    if (cubes.instanceColor) cubes.instanceColor.needsUpdate = true;
    g.rotation.y = -0.6 + Math.sin(t * 0.14) * 0.25;
  });
  return g;
};

const beam = (from: THREE.Vector3, to: THREE.Vector3, accent: THREE.Color, ticks: Tick[]): { mesh: THREE.Mesh; mat: THREE.ShaderMaterial } => {
  const mid = from.clone().lerp(to, 0.5);
  mid.y += 1.1;
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uT: { value: 0 }, uC: { value: accent }, uA: { value: 0 } },
    vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform float uT; uniform vec3 uC; uniform float uA; varying vec2 vUv;
      void main(){ float p=fract(vUv.x*2.-uT*.9); float a=(.08+pow(p,10.)*2.4)*uA; gl_FragColor=vec4(uC*a,a); }`,
  });
  ticks.push((t) => { mat.uniforms.uT.value = t; });
  return { mesh: new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(from, mid, to), 64, 0.02, 6), mat), mat };
};

const experts = (k: Kit, accent: THREE.Color, accent2: THREE.Color, ticks: Tick[], mobile: boolean): THREE.Group => {
  const g = new THREE.Group();
  const hub = new THREE.Vector3(0, 0, 2.2);
  const router = spark(k, "gold", accent, ticks);
  router.position.copy(hub);
  router.scale.setScalar(1.15);
  g.add(router);
  const N = 8, R = 5.2, glows: THREE.Color[] = [], beams: THREE.ShaderMaterial[] = [];
  const top = hub.clone().setY(0.62);
  for (let i = 0; i < N; i++) {
    const a = -1.2 + i * (2.4 / (N - 1));
    const e = spark(k, "graphite", accent2, ticks);
    e.scale.setScalar(0.56);
    e.position.set(hub.x + Math.sin(a) * R, 0, hub.z - Math.cos(a) * R);
    e.rotation.y = -a * 0.25;
    g.add(e);
    e.traverse((c) => {
      if (c instanceof THREE.Mesh && c.material instanceof THREE.ShaderMaterial) glows.push(c.material.uniforms.uC.value as THREE.Color);
    });
    const b = beam(top, e.position.clone().setY(0.4), accent, ticks);
    g.add(b.mesh);
    beams.push(b.mat);
  }
  g.add(tokenStream(accent, accent2, ticks, mobile ? 420 : 900, top));
  const on = new Set<number>([1, 5]);
  const dim = accent2.clone().multiplyScalar(0.12), lit = accent.clone().multiplyScalar(1.5);
  let next = 0;
  ticks.push((t, dt) => {
    if (t > next) {
      next = t + 1.4;
      on.clear();
      while (on.size < 2) on.add(Math.floor(Math.random() * N));
    }
    const s = Math.min(1, dt * 5);
    glows.forEach((c, i) => { c.lerp(on.has(i) ? lit : dim, s); });
    beams.forEach((m, i) => { m.uniforms.uA.value += ((on.has(i) ? 1 : 0) - m.uniforms.uA.value) * s; });
    g.rotation.y = Math.sin(t * 0.12) * 0.12;
  });
  return g;
};

const build = (v: Variant, k: Kit, accent: THREE.Color, accent2: THREE.Color, ticks: Tick[], mobile: boolean): Shot => {
  const V = (x: number, y: number, z: number): THREE.Vector3 => new THREE.Vector3(x, y, z);
  if (v === "basics") {
    return { obj: tensor(accent, accent2, ticks), cam: V(0, 4.2, 8.2), look: V(0, 0.9, 0), x: 2.3 };
  }
  if (v === "inside") {
    const g = new THREE.Group();
    const b = board(k, accent, ticks, { traces: true, fill: false });
    b.g.scale.setScalar(3);
    g.add(b.g);
    const d = b.die.clone().multiplyScalar(3);
    g.add(tokenStream(accent, accent2, ticks, mobile ? 500 : 1000, d));
    g.rotation.y = -0.3;
    return { obj: g, cam: V(0, 3.4, 5.4), look: V(0.4, 0.3, 0), x: 1.4 };
  }
  if (v === "files") {
    const g = new THREE.Group();
    const b = board(k, accent, ticks, { traces: false, fill: true });
    b.g.position.y = 0.35;
    const c = gpu(k, ticks, 5);
    c.rotation.x = -Math.PI / 2;
    g.add(b.g, c);
    g.rotation.y = -0.5;
    ticks.push((t) => { c.position.y = 1.45 + Math.sin(t * 0.6) * 0.12; g.rotation.y = -0.5 + Math.sin(t * 0.15) * 0.12; });
    return { obj: g, cam: V(0, 4.6, 7.8), look: V(0, 0.9, 0), x: 2.1 };
  }
  if (v === "arch") {
    return { obj: experts(k, accent, accent2, ticks, mobile), cam: V(0, 6.8, 11.5), look: V(0, 0.2, -0.8), x: 2.6 };
  }
  if (v === "train") {
    const g = new THREE.Group();
    const r = rig(k, accent, ticks);
    r.rotation.y = -0.55;
    g.add(r);
    ticks.push((t) => { r.rotation.y = -0.55 + Math.sin(t * 0.12) * 0.12; });
    return { obj: g, cam: V(0, 3.2, 13.5), look: V(0, 1.9, 0), x: 3.0 };
  }
  if (v === "use") {
    const g = new THREE.Group();
    const m = macStudio(k);
    m.position.set(-2.9, 0, 0);
    m.rotation.y = 0.35;
    const s = spark(k, "gold", accent, ticks);
    s.position.set(0, 0, 0.4);
    s.rotation.y = -0.15;
    const c = gpu(k, ticks, 4);
    c.position.set(3.3, 0.78, -0.2);
    c.rotation.y = -0.45;
    g.add(m, s, c, floorPath(accent, ticks));
    ticks.push((t) => { c.position.y = 0.78 + Math.sin(t * 0.8) * 0.04; });
    return { obj: g, cam: V(0, 2.6, 13.5), look: V(0, 0.6, 0), x: 2.6 };
  }
  const g = new THREE.Group();
  const s = spark(k, "gold", accent, ticks);
  s.scale.setScalar(2.1);
  g.add(s);
  g.add(tokenStream(accent, accent2, ticks, mobile ? 600 : 1200, V(0, 0.55, 1.4)));
  ticks.push((t) => { s.rotation.y = -0.55 + Math.sin(t * 0.18) * 0.22; });
  return { obj: g, cam: V(0, 2.3, 9.6), look: V(0, 0.7, 0), x: 2.9 };
};

export const mountScene = (canvas: HTMLCanvasElement, variant: Variant, a: string, b: string): boolean => {
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
  } catch {
    return false;
  }
  const sleep = (): void => {
    const fresh = canvas.cloneNode() as HTMLCanvasElement;
    fresh.classList.remove("on");
    canvas.replaceWith(fresh);
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      io.disconnect();
      if (!mountScene(fresh, variant, a, b)) fresh.classList.add("fallback");
    }, { rootMargin: "200px 0px" });
    io.observe(fresh);
  };
  try {
    run(renderer, canvas, variant, a, b, sleep);
    return true;
  } catch {
    renderer.dispose();
    return false;
  }
};

const release = (x: unknown): void => {
  if (x instanceof THREE.Texture) x.dispose();
  else if (x && typeof x === "object") Object.values(x).forEach((v) => v instanceof THREE.Texture && v.dispose());
};

const run = (renderer: THREE.WebGLRenderer, canvas: HTMLCanvasElement, variant: Variant, a: string, b: string, sleep: () => void): void => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const accent = new THREE.Color(a);
  const accent2 = new THREE.Color(b);
  const mobile = innerWidth < 760;
  renderer.setPixelRatio(Math.min(devicePixelRatio, mobile ? 1.25 : 1.5));
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  const bg = new THREE.Color(0x060607);
  const scene = new THREE.Scene();
  scene.background = bg;
  scene.environment = studioEnv(renderer, accent);
  scene.environmentIntensity = 0.9;
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 140);
  const ticks: Tick[] = [];
  const kit = makeKit();
  const shot = build(variant, kit, accent, accent2, ticks, mobile);
  const root = new THREE.Group();
  scene.add(root);
  root.add(shot.obj);
  const dist = shot.cam.distanceTo(shot.look) * (mobile ? 1.25 : 1);
  scene.fog = new THREE.Fog(bg, dist * 0.95, dist * 2.8);

  const key = new THREE.DirectionalLight(0xfff4e8, 2.4);
  key.position.set(-5, 9, 6);
  key.target = root;
  key.castShadow = true;
  key.shadow.mapSize.set(mobile ? 512 : 1024, mobile ? 512 : 1024);
  key.shadow.camera.left = key.shadow.camera.bottom = -7;
  key.shadow.camera.right = key.shadow.camera.top = 7;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 30;
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.02;
  key.shadow.radius = 4;
  const rim = new THREE.DirectionalLight(accent, 0.6);
  rim.position.set(6, 4, -7);
  rim.target = root;
  const fill = new THREE.DirectionalLight(0xdfe6ff, variant === "train" ? 1.1 : 0.35);
  fill.position.set(6, 2, 8);
  fill.target = root;
  scene.add(key, rim, fill);

  const floorGeo = new THREE.PlaneGeometry(400, 400);
  const top = new THREE.Mesh(floorGeo, new THREE.MeshStandardMaterial({
    color: 0x09090b, roughness: 0.32, metalness: 0.4,
  }));
  top.rotation.x = -Math.PI / 2;
  top.position.y = 0.002;
  top.receiveShadow = true;
  root.add(top);
  root.add(dust(accent, mobile ? 80 : 160));

  const side = mobile ? 0.55 : 1;
  root.position.x = shot.x * side;
  if (mobile) {
    root.position.y = -0.25;
    shot.obj.scale.multiplyScalar(0.78);
  }
  const look = shot.look.clone().add(new THREE.Vector3(shot.x * side * 0.55, mobile ? 1.25 : 0, 0));
  const camBase = shot.cam.clone();
  if (mobile) camBase.multiplyScalar(1.25);

  const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: mobile ? 0 : 2 });
  const composer = new EffectComposer(renderer, target);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.32, 0.55, 0.92);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  const grade = new ShaderPass(lens);
  composer.addPass(grade);

  const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
  const onPointer = (e: PointerEvent): void => { mouse.tx = e.clientX / innerWidth - 0.5; mouse.ty = e.clientY / innerHeight - 0.5; };
  addEventListener("pointermove", onPointer, { passive: true });
  const resize = (): void => {
    const w = canvas.clientWidth || innerWidth;
    const h = canvas.clientHeight || innerHeight;
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    bloom.resolution.set(w / 2, h / 2);
    grade.uniforms.uRes.value.set(w, h);
    camera.aspect = w / h;
    camera.fov = w / h < 0.8 ? 46 : 30;
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  let visible = true;
  let running = false;
  let last = performance.now();
  let t = 4;
  const place = (): void => {
    const sc = Math.min(scrollY / innerHeight, 1.2);
    camera.position.set(camBase.x + mouse.x * 1.2, camBase.y - mouse.y * 0.6 + sc * 1.2, camBase.z - sc * 2);
    camera.lookAt(look);
  };
  let idle = 0;
  let io: IntersectionObserver | undefined;
  const teardown = (): void => {
    visible = false;
    io?.disconnect();
    ro.disconnect();
    removeEventListener("pointermove", onPointer);
    document.removeEventListener("visibilitychange", start);
    scene.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
      const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : [];
      mats.forEach((mat) => { Object.values(mat).forEach(release); mat.dispose(); });
    });
    Object.values(kit).forEach(release);
    scene.environment?.dispose();
    target.dispose();
    bloom.dispose();
    composer.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
    sleep();
  };
  io = new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    clearTimeout(idle);
    if (visible) start();
    else idle = window.setTimeout(teardown, 4000);
  });
  if (reduce) {
    io.observe(canvas);
    ticks.forEach((f) => f(t, 0.016));
    place();
    composer.render();
    canvas.classList.add("on");
    return;
  }
  function start(): void {
    if (running || !visible || document.hidden || reduce) return;
    running = true;
    last = performance.now();
    requestAnimationFrame(frame);
  }
  const frame = (now: number): void => {
    if (!visible || document.hidden) { running = false; return; }
    requestAnimationFrame(frame);
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    t += dt;
    mouse.x += (mouse.tx - mouse.x) * 0.04;
    mouse.y += (mouse.ty - mouse.y) * 0.04;
    ticks.forEach((f) => f(t, dt));
    grade.uniforms.uTime.value = t;
    place();
    composer.render();
    canvas.classList.add("on");
  };
  io.observe(canvas);
  document.addEventListener("visibilitychange", start);
  start();
};
