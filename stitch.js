export const D2R = Math.PI / 180;
function qmul(a, b) {
  return [
    a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
    a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
    a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
    a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2]
  ];
}
// Phone orientation to camera rotation; the camera looks out the back of the phone.
export function quatFromDevice(alpha, beta, gamma, screenAngle) {
  const x = beta*D2R/2, y = alpha*D2R/2, z = -gamma*D2R/2;
  const c1 = Math.cos(x), c2 = Math.cos(y), c3 = Math.cos(z), s1 = Math.sin(x), s2 = Math.sin(y), s3 = Math.sin(z);
  let q = [s1*c2*c3 + c1*s2*s3, c1*s2*c3 - s1*c2*s3, c1*c2*s3 - s1*s2*c3, c1*c2*c3 + s1*s2*s3];
  q = qmul(q, [-Math.SQRT1_2, 0, 0, Math.SQRT1_2]);
  const o = -screenAngle*D2R/2;
  return qmul(q, [0, 0, Math.sin(o), Math.cos(o)]);
}
export function quatToMatrix(q) {
  const [x, y, z, w] = q;
  return [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
          2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
          2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)];
}
export const forwardOf = R => [-R[2], -R[5], -R[8]];
export function toLocal(R, d) {
  return [R[0]*d[0]+R[3]*d[1]+R[6]*d[2], R[1]*d[0]+R[4]*d[1]+R[7]*d[2], R[2]*d[0]+R[5]*d[1]+R[8]*d[2]];
}
export function dirFromYawPitch(yaw, pitch) {
  const c = Math.cos(pitch);
  return [c*Math.sin(yaw), Math.sin(pitch), -c*Math.cos(yaw)];
}
export function makeTargets(yaw0, mode = 'six') {
  if (mode === 'six') return [[0, 0], [90, 0], [180, 0], [270, 0], [0, 88], [0, -88]]
    .map(([y, p]) => ({ d: dirFromYawPitch(yaw0 + y*D2R, p*D2R), done: false }));
  const t = [];
  for (let i = 0; i < 16; i++) t.push(dirFromYawPitch(yaw0 + i*22.5*D2R, 0));
  for (let i = 0; i < 10; i++) t.push(dirFromYawPitch(yaw0 + (i*36+18)*D2R, 40*D2R));
  for (let i = 0; i < 10; i++) t.push(dirFromYawPitch(yaw0 + (i*36+18)*D2R, -40*D2R));
  t.push(dirFromYawPitch(yaw0, 88*D2R));
  t.push(dirFromYawPitch(yaw0, -88*D2R));
  return t.map(d => ({ d, done: false }));
}
export function cameraTans(w, h, longFovDeg) {
  const L = Math.tan(longFovDeg*D2R/2);
  return w >= h ? [L, L*h/w] : [L*w/h, L];
}
// frames: [{data, w, h, R, tanX, tanY}] -> RGBA equirectangular image (W x H).
// Each spot is taken from ONE photo; the cuts between photos are moved to where neighbouring photos agree,
// then softened over a few pixels. This avoids the doubled / ghosted look you get from averaging.
export async function stitch(frames, W, H, onProgress) {
  const say = p => { if (onProgress) onProgress(p); };
  const yieldUI = () => new Promise(r => setTimeout(r, 0));
  const F = frames.map(f => ({ ...f, fx: -f.R[2], fy: -f.R[5], fz: -f.R[8], cmin: 1/Math.sqrt(1 + f.tanX*f.tanX + f.tanY*f.tanY) - 0.02 }));
  function proj(f, dx, dy, dz) { // -> [xn, yn] in -1..1 or null
    if (dx*f.fx + dy*f.fy + dz*f.fz < f.cmin) return null;
    const R = f.R, lz = R[2]*dx + R[5]*dy + R[8]*dz; if (lz >= -0.05) return null;
    const xn = (R[0]*dx + R[3]*dy + R[6]*dz)/(-lz)/f.tanX, yn = (R[1]*dx + R[4]*dy + R[7]*dz)/(-lz)/f.tanY;
    return (xn <= -1 || xn >= 1 || yn <= -1 || yn >= 1) ? null : [xn, yn];
  }
  function sampleAt(f, xn, yn, out, o) { // bilinear
    const x = (xn+1)*0.5*f.w - 0.5, y = (1-yn)*0.5*f.h - 0.5;
    const x0 = Math.max(0, Math.min(f.w-2, Math.floor(x))), y0 = Math.max(0, Math.min(f.h-2, Math.floor(y)));
    const ax = Math.max(0, Math.min(1, x-x0)), ay = Math.max(0, Math.min(1, y-y0)), d = f.data;
    const k00 = (y0*f.w+x0)*4, k10 = k00+4, k01 = k00+f.w*4, k11 = k01+4;
    for (let c = 0; c < 3; c++) out[o+c] = (d[k00+c]*(1-ax) + d[k10+c]*ax)*(1-ay) + (d[k01+c]*(1-ax) + d[k11+c]*ax)*ay;
  }
  // 1) low-res grid of candidate photos per cell
  const S = 4, lw = Math.ceil(W/S), lh = Math.ceil(H/S), N = lw*lh;
  const cand = new Array(N), tmp = new Float32Array(3);
  for (let cy = 0; cy < lh; cy++) {
    const th = Math.PI*(cy*S + S/2)/H, st = Math.sin(th), ct = Math.cos(th);
    for (let cx = 0; cx < lw; cx++) {
      const ph = 2*Math.PI*(cx*S + S/2)/W, dx = Math.cos(ph)*st, dy = ct, dz = Math.sin(ph)*st;
      const list = [];
      for (let i = 0; i < F.length; i++) {
        const p = proj(F[i], dx, dy, dz); if (!p) continue;
        sampleAt(F[i], p[0], p[1], tmp, 0);
        list.push({ f: i, w: (1-p[0]*p[0])*(1-p[1]*p[1]), r: tmp[0], g: tmp[1], b: tmp[2] });
      }
      cand[cy*lw + cx] = list;
    }
    if (cy % 16 === 0) { say(0.15*cy/lh); await yieldUI(); }
  }
  // 2) choose one photo per cell: prefer photo centres and what most photos agree on (drops things that moved),
  //    and move cuts to where photos agree
  for (const list of cand) if (list.length >= 3) {
    const med = ['r', 'g', 'b'].map(ch => { const v = list.map(c => c[ch]).sort((a, b) => a - b); return v[v.length >> 1]; });
    for (const c of list) c.dev = Math.abs(c.r-med[0]) + Math.abs(c.g-med[1]) + Math.abs(c.b-med[2]);
  }
  const label = new Int32Array(N).fill(-1);
  for (let k = 0; k < N; k++) { let b = -1, bw = -Infinity; for (const c of cand[k]) { const sc = c.w - (c.dev ? Math.min(c.dev, 400)/300 : 0); if (sc > bw) { bw = sc; b = c.f; } } label[k] = b; }
  const colorOf = (k, f) => { for (const c of cand[k]) if (c.f === f) return c; return null; };
  const diff = (a, b) => Math.abs(a.r-b.r) + Math.abs(a.g-b.g) + Math.abs(a.b-b.b);
  const MISS = 400, DATA = 90;
  const pairCost = (k, q, lk, lq) => {
    if (lk === lq || lq < 0) return 0;
    const a1 = colorOf(k, lk), b1 = colorOf(k, lq), a2 = colorOf(q, lk), b2 = colorOf(q, lq);
    return (a1 && b1 ? diff(a1, b1) : MISS) + (a2 && b2 ? diff(a2, b2) : MISS);
  };
  for (let pass = 0; pass < 8; pass++) {
    const fwd = pass % 2 === 0;
    for (let n = 0; n < N; n++) {
      const k = fwd ? n : N-1-n, list = cand[k]; if (list.length < 2) continue;
      const cx = k % lw, cy = (k - cx)/lw;
      const nb = [cy*lw + (cx+1) % lw, cy*lw + (cx-1+lw) % lw];
      if (cy > 0) nb.push(k - lw); if (cy < lh-1) nb.push(k + lw);
      let best = label[k], bestC = Infinity;
      for (const c of list) {
        let cost = DATA*(1 - c.w) + (c.dev ? 0.6*Math.min(c.dev, 400) : 0);
        for (const q of nb) cost += pairCost(k, q, c.f, label[q]);
        if (cost < bestC) { bestC = cost; best = c.f; }
      }
      label[k] = best;
    }
    say(0.15 + 0.25*(pass+1)/8); await yieldUI();
  }
  // 3) soften each cut over a few pixels
  const R2 = 2, soft = new Array(N);
  for (let cy = 0; cy < lh; cy++) for (let cx = 0; cx < lw; cx++) {
    const k = cy*lw + cx, m = new Map(); let tot = 0;
    for (let oy = -R2; oy <= R2; oy++) {
      const yy = cy + oy; if (yy < 0 || yy >= lh) continue;
      for (let ox = -R2; ox <= R2; ox++) {
        const l = label[yy*lw + (cx+ox+lw) % lw]; if (l < 0 || !colorOf(k, l)) continue;
        m.set(l, (m.get(l) || 0) + 1); tot++;
      }
    }
    soft[k] = tot ? [...m].map(([f, c]) => [f, c/tot]) : null;
  }
  // 4) full-resolution render
  const out = new Uint8ClampedArray(W*H*4), px3 = new Float32Array(3);
  for (let py = 0; py < H; py++) {
    const th = Math.PI*(py+0.5)/H, st = Math.sin(th), ct = Math.cos(th), cy = Math.min(lh-1, (py/S)|0);
    let rs = 0, gs = 0, bs = 0, cnt = 0; const miss = [];
    for (let px = 0; px < W; px++) {
      const ph = 2*Math.PI*(px+0.5)/W, dx = Math.cos(ph)*st, dy = ct, dz = Math.sin(ph)*st;
      const o = (py*W + px)*4, sw = soft[cy*lw + Math.min(lw-1, (px/S)|0)];
      let r = 0, g = 0, b = 0, ws = 0;
      if (sw) for (const [fi, w] of sw) { const p = proj(F[fi], dx, dy, dz); if (!p) continue; sampleAt(F[fi], p[0], p[1], px3, 0); r += px3[0]*w; g += px3[1]*w; b += px3[2]*w; ws += w; }
      if (ws === 0) { // gap: use the nearest photo, stretched
        let nf = null, nd = 0.05; for (const f of F) { const d = dx*f.fx + dy*f.fy + dz*f.fz; if (d > nd) { nd = d; nf = f; } }
        if (nf) {
          const R = nf.R, lz = Math.min(-0.05, R[2]*dx + R[5]*dy + R[8]*dz);
          const xn = Math.max(-0.999, Math.min(0.999, (R[0]*dx + R[3]*dy + R[6]*dz)/(-lz)/nf.tanX));
          const yn = Math.max(-0.999, Math.min(0.999, (R[1]*dx + R[4]*dy + R[7]*dz)/(-lz)/nf.tanY));
          sampleAt(nf, xn, yn, px3, 0); r = px3[0]; g = px3[1]; b = px3[2]; ws = 1;
        }
      }
      if (ws > 0) { out[o] = r/ws; out[o+1] = g/ws; out[o+2] = b/ws; rs += out[o]; gs += out[o+1]; bs += out[o+2]; cnt++; } else miss.push(o);
      out[o+3] = 255;
    }
    const ar = cnt ? rs/cnt : 12, ag = cnt ? gs/cnt : 14, ab = cnt ? bs/cnt : 22;
    for (const o of miss) { out[o] = ar; out[o+1] = ag; out[o+2] = ab; }
    if (py % 32 === 0) { say(0.4 + 0.6*py/H); await yieldUI(); }
  }
  say(1);
  return out;
}
