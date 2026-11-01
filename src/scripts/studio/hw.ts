import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { bead, brushed, chipLabel, dieShot, foam, pcb, perf, radial, rng, type Maps } from "./tex";

export type Tick = (t: number, dt: number) => void;

export interface Kit {
  brushed: Maps;
  bead: Maps;
  foam: Maps;
  perf: Maps;
  pcb: Maps;
  die: THREE.Texture;
  chip: THREE.Texture;
  glow: THREE.Texture;
}

export const makeKit = (): Kit => ({
  brushed: brushed(7),
  bead: bead(5),
  foam: foam(11),
  perf: perf(16, 0.3),
  pcb: pcb(3),
  die: dieShot(9),
  chip: chipLabel(4),
  glow: radial("#fff", "#000"),
});

const V2 = (s: number): THREE.Vector2 => new THREE.Vector2(s, s);

const uvScale = <G extends THREE.BufferGeometry>(g: G, sx: number, sy: number): G => {
  const uv = g.getAttribute("uv");
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * sx, uv.getY(i) * sy);
  uv.needsUpdate = true;
  return g;
};

const rr = <P extends THREE.Path>(p: P, w: number, h: number, r: number, cx = 0, cy = 0): P => {
  const x = cx - w / 2, y = cy - h / 2;
  p.moveTo(x + r, y);
  p.lineTo(x + w - r, y);
  p.absarc(x + w - r, y + r, r, -Math.PI / 2, 0, false);
  p.lineTo(x + w, y + h - r);
  p.absarc(x + w - r, y + h - r, r, 0, Math.PI / 2, false);
  p.lineTo(x + r, y + h);
  p.absarc(x + r, y + h - r, r, Math.PI / 2, Math.PI, false);
  p.lineTo(x, y + r);
  p.absarc(x + r, y + r, r, Math.PI, Math.PI * 1.5, false);
  return p;
};

const mesh = (g: THREE.BufferGeometry, m: THREE.Material | THREE.Material[], x = 0, y = 0, z = 0): THREE.Mesh => {
  const o = new THREE.Mesh(g, m);
  o.position.set(x, y, z);
  return o;
};

export const shadows = (o: THREE.Object3D): void => {
  o.traverse((c) => {
    if (c instanceof THREE.Mesh && !(c.material instanceof THREE.ShaderMaterial) && !(c.material instanceof THREE.MeshBasicMaterial)) {
      c.castShadow = true;
      c.receiveShadow = true;
    }
  });
};

const plastic = (): THREE.MeshPhysicalMaterial =>
  new THREE.MeshPhysicalMaterial({ color: 0x0e0e10, roughness: 0.42, clearcoat: 0.5, clearcoatRoughness: 0.22 });

const rubber = new THREE.MeshStandardMaterial({ color: 0x0a0a0b, roughness: 0.9 });
const black = new THREE.MeshStandardMaterial({ color: 0x040405, roughness: 0.7 });

const passUv = `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`;

export const spark = (k: Kit, finish: "gold" | "graphite", accent: THREE.Color, ticks: Tick[]): THREE.Group => {
  const W = 1.5, H = 0.505, D = 1.5, t = 0.028, b = 0.012;
  const gold = finish === "gold";
  const g = new THREE.Group();
  const body = new THREE.Group();
  body.position.y = 0.022;
  g.add(body);
  const shell = new THREE.MeshPhysicalMaterial({
    color: gold ? 0xc9a676 : 0x3a3c41,
    metalness: 1,
    roughness: 1,
    roughnessMap: k.brushed.rough,
    normalMap: k.brushed.normal,
    normalScale: V2(0.16),
    clearcoat: 0.25,
    clearcoatRoughness: 0.3,
    anisotropy: 0.45,
  });
  const prof = rr(new THREE.Shape(), W - 2 * b, H - 2 * b, 0.034);
  prof.holes.push(rr(new THREE.Path(), W - 2 * b - 2 * t, H - 2 * b - 2 * t, 0.012));
  const sleeve = new THREE.ExtrudeGeometry(prof, { depth: D - 2 * b, bevelEnabled: true, bevelThickness: b, bevelSize: b, bevelSegments: 4, curveSegments: 10 });
  sleeve.translate(0, 0, -(D - 2 * b) / 2);
  body.add(mesh(sleeve, shell, 0, H / 2, 0));
  const fw = W - 2 * b - 2 * t, fh = H - 2 * b - 2 * t;
  const foamMat = new THREE.MeshStandardMaterial({
    color: gold ? 0xc3a270 : 0x33353a,
    metalness: 1,
    roughness: 1,
    map: k.foam.color,
    roughnessMap: k.foam.rough,
    normalMap: k.foam.normal,
    normalScale: V2(1.3),
    alphaMap: k.foam.alpha,
    alphaTest: 0.5,
    side: THREE.DoubleSide,
  });
  const fg = uvScale(new THREE.PlaneGeometry(fw, fh), fw / 0.42, fh / 0.42);
  const front = mesh(fg, foamMat, 0, H / 2, D / 2 - 0.03);
  const back = mesh(fg, foamMat, 0, H / 2, -D / 2 + 0.03);
  back.rotation.y = Math.PI;
  body.add(front, back);
  const glowMat = new THREE.ShaderMaterial({
    uniforms: { uT: { value: 0 }, uC: { value: accent.clone() } },
    vertexShader: passUv,
    fragmentShader: `uniform float uT; uniform vec3 uC; varying vec2 vUv;
      void main(){
        float w=.5+.5*sin(vUv.x*7.-uT*1.4);
        float v=smoothstep(0.,.55,vUv.y)*smoothstep(1.,.45,vUv.y);
        float f=.5+.5*sin(uT*.7);
        gl_FragColor=vec4(uC*(.05+1.3*pow(w,5.)*v*(.55+.45*f)),1.);
      }`,
  });
  body.add(mesh(new THREE.PlaneGeometry(fw, fh), glowMat, 0, H / 2, D / 2 - 0.07));
  body.add(mesh(new THREE.BoxGeometry(fw - 0.01, fh - 0.01, D - 0.15), black, 0, H / 2, 0));
  for (const [x, z] of [[-0.55, -0.55], [0.55, -0.55], [-0.55, 0.55], [0.55, 0.55]])
    g.add(mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.022, 32), rubber, x, 0.011, z));
  ticks.push((tt) => { glowMat.uniforms.uT.value = tt; });
  shadows(g);
  return g;
};

export const macStudio = (k: Kit): THREE.Group => {
  const W = 1.97, H = 0.95, bv = 0.045, baseH = 0.07;
  const g = new THREE.Group();
  const alu = new THREE.MeshPhysicalMaterial({
    color: 0xd5d6d9, metalness: 1, roughness: 1, roughnessMap: k.bead.rough, normalMap: k.bead.normal, normalScale: V2(0.3), clearcoat: 0.1,
  });
  const body = new THREE.ExtrudeGeometry(rr(new THREE.Shape(), W - 2 * bv, W - 2 * bv, 0.3), {
    depth: H - baseH - 2 * bv, bevelEnabled: true, bevelThickness: bv, bevelSize: bv, bevelSegments: 8, curveSegments: 28,
  });
  body.rotateX(-Math.PI / 2);
  body.translate(0, baseH + bv, 0);
  g.add(mesh(body, alu));
  const base = new THREE.ExtrudeGeometry(rr(new THREE.Shape(), W - 0.34, W - 0.34, 0.2), { depth: baseH, bevelEnabled: false, curveSegments: 24 });
  base.rotateX(-Math.PI / 2);
  g.add(mesh(base, new THREE.MeshStandardMaterial({ color: 0x1c1c1e, metalness: 0.6, roughness: 0.5 })));
  const hole = new THREE.MeshStandardMaterial({ color: 0x030303, roughness: 0.6 });
  const port = (w: number, h: number, x: number, y: number): void => {
    g.add(mesh(new RoundedBoxGeometry(w, h, 0.02, 2, Math.min(w, h) / 2.2), hole, x, y, W / 2 - 0.005));
  };
  port(0.09, 0.032, -0.55, 0.3);
  port(0.09, 0.032, -0.4, 0.3);
  port(0.26, 0.022, 0.2, 0.3);
  g.add(mesh(new THREE.CircleGeometry(0.011, 20), new THREE.MeshBasicMaterial({ color: new THREE.Color(1.6, 1.6, 1.6) }), 0.55, 0.3, W / 2 + 0.0012));
  const grill = mesh(
    uvScale(new THREE.PlaneGeometry(1.25, 0.42), 1.25 / 0.3, 0.42 / 0.3),
    new THREE.MeshStandardMaterial({ color: 0xd5d6d9, metalness: 1, roughness: 0.34, map: k.perf.color, normalMap: k.perf.normal, polygonOffset: true, polygonOffsetFactor: -2 }),
    0, 0.62, -W / 2 - 0.0012,
  );
  grill.rotation.y = Math.PI;
  g.add(grill);
  shadows(g);
  return g;
};

const fan = (k: Kit, r: number): THREE.Group => {
  const g = new THREE.Group();
  const pl = plastic();
  const s = new THREE.Shape();
  s.moveTo(r * 0.27, -r * 0.06);
  s.quadraticCurveTo(r * 0.6, -r * 0.24, r * 0.95, -r * 0.1);
  s.lineTo(r * 0.95, r * 0.2);
  s.quadraticCurveTo(r * 0.6, r * 0.12, r * 0.27, r * 0.1);
  s.closePath();
  const blade = new THREE.ExtrudeGeometry(s, { depth: 0.006, bevelEnabled: true, bevelThickness: 0.003, bevelSize: 0.003, bevelSegments: 2, curveSegments: 12 });
  blade.rotateX(0.5);
  for (let i = 0; i < 9; i++) {
    const m = new THREE.Mesh(blade, pl);
    m.rotation.z = (i / 9) * Math.PI * 2;
    g.add(m);
  }
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.27, r * 0.28, 0.05, 48), pl);
  hub.rotation.x = Math.PI / 2;
  g.add(hub);
  const cap = new THREE.Mesh(
    new THREE.CircleGeometry(r * 0.2, 48),
    new THREE.MeshPhysicalMaterial({ color: 0x1a1b1e, metalness: 1, roughness: 0.25, normalMap: k.brushed.normal, normalScale: V2(0.2) }),
  );
  cap.position.z = 0.026;
  g.add(cap);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(r * 0.955, 0.008, 8, 96), pl);
  g.add(ring);
  return g;
};

export const gpu = (k: Kit, ticks: Tick[], spin = 6): THREE.Group => {
  const L = 3.04, Hh = 1.37, T = 0.4, fr = 0.5;
  const g = new THREE.Group();
  const shroud = new THREE.MeshPhysicalMaterial({
    color: 0x2b2c30, metalness: 1, roughness: 1, roughnessMap: k.brushed.rough, normalMap: k.brushed.normal, normalScale: V2(0.14), clearcoat: 0.35, clearcoatRoughness: 0.3,
  });
  const fx = [-0.78, 0.78];
  const cover = rr(new THREE.Shape(), L, Hh, 0.09);
  fx.forEach((x) => { const h = new THREE.Path(); h.absarc(x, 0, fr, 0, Math.PI * 2, true); cover.holes.push(h); });
  const cg = new THREE.ExtrudeGeometry(cover, { depth: 0.03, bevelEnabled: true, bevelThickness: 0.012, bevelSize: 0.012, bevelSegments: 3, curveSegments: 56 });
  g.add(mesh(cg, shroud, 0, 0, T / 2 - 0.042));
  fx.forEach((x) => {
    g.add(mesh(new THREE.TorusGeometry(fr, 0.016, 12, 96), shroud, x, 0, T / 2 - 0.03));
    const f = fan(k, fr - 0.02);
    f.position.set(x, 0, T / 2 - 0.08);
    g.add(f);
    ticks.push((_t, dt) => { f.rotation.z -= dt * spin; });
  });
  const bp = new THREE.ExtrudeGeometry(rr(new THREE.Shape(), L, Hh, 0.09), { depth: 0.018, bevelEnabled: true, bevelThickness: 0.006, bevelSize: 0.006, bevelSegments: 2, curveSegments: 16 });
  g.add(mesh(bp, shroud, 0, 0, -T / 2));
  for (const s of [-1, 1]) g.add(mesh(new RoundedBoxGeometry(0.08, Hh - 0.02, T - 0.03, 2, 0.02), shroud, s * (L / 2 - 0.045), 0, 0));
  const finMat = new THREE.MeshStandardMaterial({ color: 0xa8acb2, metalness: 1, roughness: 0.3 });
  const n = Math.floor((L - 0.2) / 0.022);
  const fins = new THREE.InstancedMesh(new THREE.BoxGeometry(0.005, Hh - 0.12, T - 0.08), finMat, n);
  const o = new THREE.Object3D();
  for (let i = 0; i < n; i++) { o.position.set(-L / 2 + 0.1 + i * 0.022, 0, -0.01); o.updateMatrix(); fins.setMatrixAt(i, o.matrix); }
  g.add(fins);
  const pipe = new THREE.MeshStandardMaterial({ color: 0xc2ab8e, metalness: 1, roughness: 0.22 });
  for (const y of [-0.36, -0.12, 0.12, 0.36]) {
    const p = mesh(new THREE.CylinderGeometry(0.028, 0.028, L - 0.24, 20), pipe, 0, y, -0.02);
    p.rotation.z = Math.PI / 2;
    g.add(p);
  }
  const silver = new THREE.MeshStandardMaterial({ color: 0xc7c9cc, metalness: 1, roughness: 0.32, map: k.perf.color, normalMap: k.perf.normal, alphaMap: k.perf.alpha, alphaTest: 0.5 });
  g.add(mesh(uvScale(new THREE.BoxGeometry(0.012, 1.2, 0.42), 1.7, 4.8), silver, -L / 2 - 0.012, -0.06, 0));
  g.add(mesh(new THREE.BoxGeometry(0.9, 0.12, 0.016), new THREE.MeshStandardMaterial({ color: 0x0f1411, roughness: 0.5 }), -L / 2 + 0.75, -Hh / 2 - 0.05, -T / 2 + 0.08));
  const goldM = new THREE.MeshStandardMaterial({ color: 0xd9ac52, metalness: 1, roughness: 0.18 });
  const pins = new THREE.InstancedMesh(new THREE.BoxGeometry(0.012, 0.08, 0.02), goldM, 40);
  for (let i = 0; i < 40; i++) { o.position.set(-L / 2 + 0.34 + i * 0.021, -Hh / 2 - 0.06, -T / 2 + 0.08); o.updateMatrix(); pins.setMatrixAt(i, o.matrix); }
  g.add(pins);
  g.add(mesh(new RoundedBoxGeometry(0.22, 0.08, 0.11, 2, 0.01), plastic(), 0.35, Hh / 2 + 0.02, -0.02));
  g.add(mesh(new THREE.BoxGeometry(0.5, 0.004, 0.035), new THREE.MeshBasicMaterial({ color: new THREE.Color(1.4, 1.4, 1.4) }), -0.5, Hh / 2 - 0.004, T / 2 - 0.1));
  shadows(g);
  return g;
};

export interface BoardOpts { traces: boolean; fill: boolean }
export interface Board { g: THREE.Group; die: THREE.Vector3 }

export const board = (k: Kit, accent: THREE.Color, ticks: Tick[], opt: BoardOpts): Board => {
  const Lp = 2.8, Dp = 1.45, th = 0.024, top = th;
  const r = rng(21);
  const g = new THREE.Group();
  const pm = new THREE.MeshStandardMaterial({ map: k.pcb.color, normalMap: k.pcb.normal, normalScale: V2(0.6), roughnessMap: k.pcb.rough, roughness: 1, metalness: 0.15 });
  g.add(mesh(uvScale(new THREE.BoxGeometry(Lp, th, Dp), Lp / 1.4, Dp / 1.4), pm, 0, th / 2, 0));
  const dx = 0.25;
  const die = new THREE.Vector3(dx, top + 0.045, 0);
  g.add(mesh(new RoundedBoxGeometry(0.8, 0.03, 0.8, 2, 0.006), new THREE.MeshStandardMaterial({ color: 0x2c3528, roughness: 0.55, metalness: 0.1 }), dx, top + 0.015, 0));
  g.add(mesh(new THREE.BoxGeometry(0.9, 0.012, 0.9), new THREE.MeshStandardMaterial({ color: 0x9da1a6, metalness: 1, roughness: 0.3 }), dx, top + 0.006, 0));
  const dieMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, map: k.die, metalness: 0.35, roughness: 0.1, clearcoat: 1, clearcoatRoughness: 0.05,
    iridescence: 1, iridescenceIOR: 1.7, iridescenceThicknessRange: [220, 780],
  });
  g.add(mesh(new THREE.BoxGeometry(0.5, 0.02, 0.55), dieMat, dx, top + 0.04, 0));
  const tiny = new THREE.InstancedMesh(new THREE.BoxGeometry(0.018, 0.008, 0.01), new THREE.MeshStandardMaterial({ color: 0x8a6c47, roughness: 0.45, metalness: 0.3 }), 360);
  const o = new THREE.Object3D();
  for (let i = 0; i < 360; i++) {
    const ring = i < 120;
    const a = r() * Math.PI * 2;
    const rad = ring ? 0.3 + r() * 0.08 : 0;
    const px = ring ? dx + Math.cos(a) * rad : (r() - 0.5) * Lp * 0.95;
    const pz = ring ? Math.sin(a) * rad : (r() - 0.5) * Dp * 0.95;
    o.position.set(px, top + (ring ? 0.034 : 0.004), pz);
    o.rotation.set(0, r() < 0.5 ? 0 : Math.PI / 2, 0);
    o.updateMatrix();
    tiny.setMatrixAt(i, o.matrix);
  }
  g.add(tiny);
  const side = new THREE.MeshStandardMaterial({ color: 0x121214, roughness: 0.62 });
  const label = new THREE.MeshStandardMaterial({ map: k.chip, roughness: 0.58 });
  const cmats = [side, side, label, side, side, side];
  const chips: THREE.Vector3[] = [];
  for (const x of [-0.45, -0.15, 0.15, 0.45]) { chips.push(new THREE.Vector3(dx + x, 0, -0.6), new THREE.Vector3(dx + x, 0, 0.6)); }
  for (const z of [-0.33, -0.11, 0.11, 0.33]) { chips.push(new THREE.Vector3(dx - 0.65, 0, z), new THREE.Vector3(dx + 0.65, 0, z)); }
  const halos: THREE.MeshBasicMaterial[] = [];
  chips.forEach((c, i) => {
    const vert = Math.abs(c.z) < 0.5;
    const m = mesh(new THREE.BoxGeometry(0.15, 0.016, 0.13), cmats, c.x, top + 0.008, c.z);
    if (vert) m.rotation.y = Math.PI / 2;
    g.add(m);
    if (opt.fill) {
      const hm = new THREE.MeshBasicMaterial({ color: accent, alphaMap: k.glow, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
      const h = mesh(new THREE.PlaneGeometry(0.42, 0.38), hm, c.x, top + 0.0015, c.z);
      h.rotation.x = -Math.PI / 2;
      g.add(h);
      halos[i] = hm;
    }
  });
  const choke = new THREE.MeshStandardMaterial({ color: 0x46474b, roughness: 0.55, metalness: 0.4 });
  for (const x of [-0.7, -0.84]) for (let i = 0; i < 7; i++) g.add(mesh(new RoundedBoxGeometry(0.1, 0.08, 0.1, 2, 0.012), choke, dx + x - 0.2, top + 0.04, -0.54 + i * 0.18));
  const fet = new THREE.MeshStandardMaterial({ color: 0x0c0c0e, roughness: 0.4 });
  for (let i = 0; i < 7; i++) g.add(mesh(new THREE.BoxGeometry(0.06, 0.012, 0.07), fet, dx - 1.19, top + 0.006, -0.54 + i * 0.18));
  const capTop = new THREE.MeshStandardMaterial({ color: 0xb6b8bc, metalness: 1, roughness: 0.3 });
  for (let i = 0; i < 8; i++) g.add(mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.07, 28), capTop, -1.3, top + 0.035, -0.56 + i * 0.16));
  for (let i = 0; i < 5; i++) g.add(mesh(new THREE.BoxGeometry(0.1 + r() * 0.08, 0.012, 0.1 + r() * 0.06), fet, 1.18 + (r() - 0.5) * 0.1, top + 0.006, -0.5 + i * 0.25));
  if (opt.traces) {
    chips.forEach((c, ci) => {
      const vert = Math.abs(c.z) < 0.5;
      for (let l = -2; l <= 2; l++) {
        const off = l * 0.022;
        const y = top + 0.002;
        const p0 = vert ? new THREE.Vector3(c.x + Math.sign(dx - c.x) * 0.075, y, c.z + off) : new THREE.Vector3(c.x + off, y, c.z + Math.sign(-c.z) * 0.075);
        const p2 = vert ? new THREE.Vector3(dx + Math.sign(c.x - dx) * 0.4, y, c.z * 0.5 + off) : new THREE.Vector3(c.x * 0.5 + dx * 0.5 + off, y, Math.sign(c.z) * 0.4);
        const p1 = vert ? new THREE.Vector3((p0.x + p2.x) / 2, y, p0.z) : new THREE.Vector3(p0.x, y, (p0.z + p2.z) / 2);
        const curve = new THREE.CatmullRomCurve3([p0, p1, p2], false, "centripetal");
        const mat = new THREE.ShaderMaterial({
          transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
          uniforms: { uT: { value: 0 }, uC: { value: accent }, uO: { value: r() } },
          vertexShader: passUv,
          fragmentShader: `uniform float uT; uniform vec3 uC; uniform float uO; varying vec2 vUv;
            void main(){ float p=fract(vUv.x-uT*1.1+uO); float a=pow(p,14.)*2.4+.05; gl_FragColor=vec4(uC*a,a); }`,
        });
        g.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 32, 0.0035, 4), mat));
        ticks.push((tt) => { mat.uniforms.uT.value = tt * (0.8 + (ci % 3) * 0.1); });
      }
    });
  }
  if (opt.fill)
    ticks.push((tt) => {
      const lvl = (Math.sin(tt * 0.45) * 0.5 + 0.5) * 17;
      halos.forEach((h, i) => { h.opacity += ((i < lvl ? 0.55 : 0) - h.opacity) * 0.08; });
    });
  shadows(g);
  return { g, die };
};

const ribbon = (curve: THREE.Curve<THREE.Vector3>, w: number, seg: number): THREE.BufferGeometry => {
  const pos: number[] = [], uv: number[] = [], idx: number[] = [];
  const side = new THREE.Vector3(1, 0, 0);
  for (let i = 0; i <= seg; i++) {
    const p = curve.getPointAt(i / seg);
    pos.push(p.x - side.x * w / 2, p.y, p.z, p.x + side.x * w / 2, p.y, p.z);
    uv.push(0, i / seg, 1, i / seg);
    if (i < seg) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
};

export const rig = (k: Kit, accent: THREE.Color, ticks: Tick[]): THREE.Group => {
  const g = new THREE.Group();
  const alu = new THREE.MeshStandardMaterial({ color: 0x9da1a7, metalness: 1, roughness: 0.34, normalMap: k.brushed.normal, normalScale: V2(0.1) });
  const FW = 4.4, FD = 3.4, FH = 3.7, e = 0.1, mid = 0.95;
  const bar = (sx: number, sy: number, sz: number, x: number, y: number, z: number): void => { g.add(mesh(new RoundedBoxGeometry(sx, sy, sz, 2, 0.012), alu, x, y, z)); };
  for (const x of [-1, 1]) for (const z of [-1, 1]) bar(e, FH, e, x * FW / 2, FH / 2, z * FD / 2);
  for (const y of [e / 2, mid, FH]) {
    for (const z of [-1, 1]) bar(FW, e, e, 0, y, z * FD / 2);
    for (const x of [-1, 1]) bar(e, e, FD, x * FW / 2, y, 0);
  }
  bar(FW, e, e, 0, FH, 0);
  const mb = new THREE.MeshStandardMaterial({ map: k.pcb.color, normalMap: k.pcb.normal, roughnessMap: k.pcb.rough, roughness: 1, metalness: 0.15 });
  g.add(mesh(uvScale(new THREE.BoxGeometry(3.05, 0.02, 2.44), 2.2, 1.7), mb, 0.2, mid + 0.06, 0));
  const pl = plastic();
  for (let i = 0; i < 8; i++) {
    const x = -0.5 + (i < 4 ? i * 0.09 : 0.75 + (i - 4) * 0.09);
    g.add(mesh(new RoundedBoxGeometry(0.045, 0.32, 1.33, 2, 0.01), pl, x + 0.2, mid + 0.23, 0.2));
  }
  const cool = new THREE.InstancedMesh(new THREE.BoxGeometry(0.9, 0.006, 0.55), new THREE.MeshStandardMaterial({ color: 0xb0b4ba, metalness: 1, roughness: 0.3 }), 44);
  const o = new THREE.Object3D();
  for (let i = 0; i < 44; i++) { o.position.set(0.53, mid + 0.25 + i * 0.022, 0.2); o.updateMatrix(); cool.setMatrixAt(i, o.matrix); }
  g.add(cool);
  const cf = fan(k, 0.44);
  cf.position.set(0.53, mid + 0.73, 0.52);
  g.add(cf, mesh(new RoundedBoxGeometry(0.98, 0.98, 0.1, 3, 0.05), pl, 0.53, mid + 0.73, 0.54));
  cf.position.z = 0.6;
  ticks.push((_t, dt) => { cf.rotation.z -= dt * 5; });
  const psu = mesh(new RoundedBoxGeometry(1.5, 0.86, 1.6, 3, 0.03), new THREE.MeshStandardMaterial({ color: 0x141416, metalness: 0.5, roughness: 0.5 }), -1.2, 0.54, 0.3);
  g.add(psu);
  const psuGrill = mesh(uvScale(new THREE.CircleGeometry(0.5, 48), 3, 3), new THREE.MeshStandardMaterial({ color: 0x2a2a2d, metalness: 1, roughness: 0.4, map: k.perf.color, normalMap: k.perf.normal }), -1.2, 0.54, 1.101);
  g.add(psuGrill);
  const cardY = FH - 0.06 - 1.37 / 2;
  const xs = [-1.35, -0.45, 0.45, 1.35];
  xs.forEach((x, i) => {
    const c = gpu(k, ticks, 5 + i * 0.4);
    c.rotation.y = -Math.PI / 2;
    c.position.set(x, cardY, -0.05);
    g.add(c);
    const start = new THREE.Vector3(x, cardY - 0.75, 0.9);
    const curve = new THREE.CatmullRomCurve3([start, new THREE.Vector3(x, cardY - 1.05, 1.0), new THREE.Vector3(x * 0.5 + 0.2, mid + 0.55, 0.2), new THREE.Vector3(-0.3 + i * 0.28, mid + 0.1, -0.7)]);
    const rib = mesh(ribbon(curve, 0.26, 40), new THREE.MeshPhysicalMaterial({ color: 0x1a1a1d, roughness: 0.4, metalness: 0.3, side: THREE.DoubleSide, clearcoat: 0.4 }));
    g.add(rib);
    const pw = new THREE.CatmullRomCurve3([new THREE.Vector3(-1.2 + i * 0.1, 0.9, 1.1), new THREE.Vector3(-2.0, 1.6, 1.4), new THREE.Vector3(-1.6 + i * 0.3, FH + 0.05, 1.2), new THREE.Vector3(x, cardY + 0.72, -0.4)]);
    g.add(mesh(new THREE.TubeGeometry(pw, 80, 0.028, 10), new THREE.MeshStandardMaterial({ color: 0x0b0b0c, roughness: 0.75 })));
  });
  const led = new THREE.MeshBasicMaterial({ color: accent.clone().multiplyScalar(2.2) });
  g.add(mesh(new THREE.BoxGeometry(FW - 0.1, 0.012, 0.02), led, 0, mid - e / 2 - 0.008, FD / 2 + 0.045));
  shadows(g);
  return g;
};
