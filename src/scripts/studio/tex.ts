import * as THREE from "three";

export const rng = (seed: number): (() => number) => {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

export interface Maps {
  color: THREE.Texture;
  normal: THREE.Texture;
  rough: THREE.Texture;
  alpha: THREE.Texture;
}

const wrap = <T extends THREE.Texture>(t: T, srgb: boolean): T => {
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.needsUpdate = true;
  return t;
};

const gray = (w: number, h: number, f: (i: number) => number, srgb = false): THREE.DataTexture => {
  const px = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(255, Math.round(f(i) * 255)));
    px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = v;
    px[i * 4 + 3] = 255;
  }
  return wrap(new THREE.DataTexture(px, w, h, THREE.RGBAFormat), srgb);
};

const normal = (hg: Float32Array, w: number, h: number, k: number): THREE.DataTexture => {
  const px = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const l = hg[y * w + ((x - 1 + w) % w)];
      const r = hg[y * w + ((x + 1) % w)];
      const b = hg[((y - 1 + h) % h) * w + x];
      const t = hg[((y + 1) % h) * w + x];
      const nx = (l - r) * k;
      const ny = (b - t) * k;
      const len = Math.hypot(nx, ny, 1);
      const i = (y * w + x) * 4;
      px[i] = (nx / len * 0.5 + 0.5) * 255;
      px[i + 1] = (ny / len * 0.5 + 0.5) * 255;
      px[i + 2] = (1 / len * 0.5 + 0.5) * 255;
      px[i + 3] = 255;
    }
  return wrap(new THREE.DataTexture(px, w, h, THREE.RGBAFormat), false);
};

const solid = gray(4, 4, () => 1);

export const brushed = (seed: number): Maps => {
  const w = 512, h = 512, r = rng(seed), hg = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const base = r() * 0.6;
    let v = r();
    for (let x = 0; x < w; x++) {
      v = (v + (r() - 0.5) * 0.08) * 0.985;
      hg[y * w + x] = base + v * 0.4 + r() * 0.05;
    }
  }
  return {
    color: gray(w, h, (i) => 0.92 + hg[i] * 0.08, true),
    normal: normal(hg, w, h, 1.4),
    rough: gray(w, h, (i) => 0.24 + hg[i] * 0.16),
    alpha: solid,
  };
};

export const bead = (seed: number): Maps => {
  const s = 256, r = rng(seed), raw = new Float32Array(s * s), hg = new Float32Array(s * s);
  for (let i = 0; i < s * s; i++) raw[i] = r();
  for (let y = 0; y < s; y++)
    for (let x = 0; x < s; x++) {
      let a = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) a += raw[((y + dy + s) % s) * s + ((x + dx + s) % s)];
      hg[y * s + x] = a / 9;
    }
  return {
    color: gray(s, s, (i) => 0.96 + hg[i] * 0.04, true),
    normal: normal(hg, s, s, 1.2),
    rough: gray(s, s, (i) => 0.3 + hg[i] * 0.12),
    alpha: solid,
  };
};

export const foam = (seed: number): Maps => {
  const s = 512, r = rng(seed), hg = new Float32Array(s * s).fill(1);
  for (let i = 0; i < 5600; i++) {
    const cx = Math.floor(r() * s), cy = Math.floor(r() * s), rad = 2.4 + r() * r() * 8, R = Math.ceil(rad);
    for (let dy = -R; dy <= R; dy++)
      for (let dx = -R; dx <= R; dx++) {
        const d = Math.hypot(dx, dy) / rad;
        if (d >= 1) continue;
        const k = ((cy + dy + s) % s) * s + ((cx + dx + s) % s);
        const v = d * d;
        if (v < hg[k]) hg[k] = v;
      }
  }
  return {
    color: gray(s, s, (i) => 0.12 + 0.88 * Math.pow(hg[i], 0.6), true),
    normal: normal(hg, s, s, 5),
    rough: gray(s, s, (i) => 0.34 + (1 - hg[i]) * 0.5),
    alpha: gray(s, s, (i) => (hg[i] < 0.05 ? 0 : 1)),
  };
};

export const perf = (cells: number, rad: number): Maps => {
  const s = 256, c = s / cells, hg = new Float32Array(s * s);
  for (let y = 0; y < s; y++)
    for (let x = 0; x < s; x++) {
      const gy = y / c, row = Math.floor(gy), gx = x / c + (row % 2 ? 0.5 : 0);
      const d = Math.hypot(gx - Math.floor(gx) - 0.5, gy - row - 0.5);
      const e = Math.min(1, Math.max(0, (d - (rad - 0.04)) / 0.07));
      hg[y * s + x] = e * e * (3 - 2 * e);
    }
  return {
    color: gray(s, s, (i) => 0.03 + 0.97 * hg[i], true),
    normal: normal(hg, s, s, 3),
    rough: gray(s, s, () => 0.4),
    alpha: gray(s, s, (i) => (hg[i] < 0.5 ? 0 : 1)),
  };
};

const ctx2d = (w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] => {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const x = c.getContext("2d");
  if (!x) throw new Error("2d context unavailable");
  return [c, x];
};

export const pcb = (seed: number): Maps => {
  const s = 1024, r = rng(seed), [c, x] = ctx2d(s, s);
  x.fillStyle = "#0c0f0e";
  x.fillRect(0, 0, s, s);
  x.lineCap = "round";
  x.lineJoin = "round";
  for (let i = 0; i < 520; i++) {
    let px = r() * s, py = r() * s, dir = Math.floor(r() * 8);
    x.beginPath();
    x.moveTo(px, py);
    const n = 2 + Math.floor(r() * 4);
    for (let k = 0; k < n; k++) {
      const len = 20 + r() * 140, a = (dir * Math.PI) / 4;
      px += Math.cos(a) * len;
      py += Math.sin(a) * len;
      x.lineTo(px, py);
      dir = (dir + (r() < 0.5 ? 1 : 7)) % 8;
    }
    x.strokeStyle = `rgb(${22 + r() * 8},${28 + r() * 8},${25 + r() * 6})`;
    x.lineWidth = 1.2 + r() * 2.4;
    x.stroke();
  }
  for (let b = 0; b < 26; b++) {
    const px = r() * s, py = r() * s, horiz = r() < 0.5, len = 120 + r() * 300;
    for (let k = 0; k < 10; k++) {
      const o = k * 5;
      x.beginPath();
      if (horiz) { x.moveTo(px, py + o); x.lineTo(px + len, py + o); x.lineTo(px + len + 30, py + o + 30); }
      else { x.moveTo(px + o, py); x.lineTo(px + o, py + len); x.lineTo(px + o + 30, py + len + 30); }
      x.strokeStyle = "rgb(27,34,30)";
      x.lineWidth = 2;
      x.stroke();
    }
  }
  for (let i = 0; i < 1400; i++) {
    const px = r() * s, py = r() * s, rr = 1.5 + r() * 2;
    x.fillStyle = "#8d7443";
    x.beginPath(); x.arc(px, py, rr, 0, 7); x.fill();
    x.fillStyle = "#050606";
    x.beginPath(); x.arc(px, py, rr * 0.45, 0, 7); x.fill();
  }
  x.font = "10px monospace";
  for (let i = 0; i < 70; i++) {
    x.fillStyle = "rgba(205,208,204,.5)";
    x.fillText(`${"CRLQU"[Math.floor(r() * 5)]}${Math.floor(r() * 900)}`, r() * s, r() * s);
    x.strokeStyle = "rgba(205,208,204,.35)";
    x.lineWidth = 1;
    x.strokeRect(r() * s, r() * s, 10 + r() * 30, 8 + r() * 20);
  }
  const d = x.getImageData(0, 0, s, s).data, hg = new Float32Array(s * s), ro = new Float32Array(s * s);
  for (let i = 0; i < s * s; i++) {
    const R = d[i * 4], G = d[i * 4 + 1], B = d[i * 4 + 2];
    hg[i] = (R + G + B) / 765;
    ro[i] = R > 100 && R > B + 30 ? 0.28 : R > 150 ? 0.62 : 0.5;
  }
  return {
    color: wrap(new THREE.CanvasTexture(c), true),
    normal: normal(hg, s, s, 6),
    rough: gray(s, s, (i) => ro[i]),
    alpha: solid,
  };
};

export const dieShot = (seed: number): THREE.Texture => {
  const s = 1024, r = rng(seed), [c, x] = ctx2d(s, s);
  const pal = ["#141b2e", "#1b1633", "#0f2428", "#241b15", "#17181c", "#1d2238"];
  x.fillStyle = "#0a0c12";
  x.fillRect(0, 0, s, s);
  const split = (px: number, py: number, w: number, h: number, dep: number): void => {
    if (dep > 5 || w < 40 || h < 40 || (dep > 2 && r() < 0.2)) {
      x.fillStyle = pal[Math.floor(r() * pal.length)];
      x.fillRect(px + 2, py + 2, w - 4, h - 4);
      const kind = r();
      if (kind < 0.45) {
        const g = 2 + Math.floor(r() * 3);
        x.fillStyle = "rgba(160,180,220,.07)";
        for (let yy = py + 3; yy < py + h - 3; yy += g) x.fillRect(px + 3, yy, w - 6, 1);
      } else if (kind < 0.7) {
        x.fillStyle = "rgba(200,190,240,.08)";
        for (let yy = py + 4; yy < py + h - 4; yy += 6) for (let xx = px + 4; xx < px + w - 4; xx += 6) x.fillRect(xx, yy, 3, 3);
      }
      return;
    }
    if (w > h ? r() < 0.8 : r() < 0.2) {
      const k = w * (0.3 + r() * 0.4);
      split(px, py, k, h, dep + 1);
      split(px + k, py, w - k, h, dep + 1);
    } else {
      const k = h * (0.3 + r() * 0.4);
      split(px, py, w, k, dep + 1);
      split(px, py + k, w, h - k, dep + 1);
    }
  };
  split(16, 16, s - 32, s - 32, 0);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
};

export const chipLabel = (seed: number): THREE.Texture => {
  const r = rng(seed), [c, x] = ctx2d(256, 224);
  x.fillStyle = "#151517";
  x.fillRect(0, 0, 256, 224);
  x.fillStyle = "rgba(190,190,195,.42)";
  x.font = "600 30px monospace";
  x.fillText("GDDR7", 40, 90);
  x.font = "20px monospace";
  x.fillText(Array.from({ length: 6 }, () => "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"[Math.floor(r() * 33)]).join(""), 40, 126);
  x.fillText("2G32", 40, 156);
  x.beginPath();
  x.arc(26, 26, 7, 0, 7);
  x.fillStyle = "rgba(120,120,125,.5)";
  x.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
};

export const radial = (inner: string, outer: string): THREE.Texture => {
  const [c, x] = ctx2d(256, 256);
  const g = x.createRadialGradient(128, 128, 0, 128, 128, 128);
  g.addColorStop(0, inner);
  g.addColorStop(1, outer);
  x.fillStyle = g;
  x.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(c);
};
