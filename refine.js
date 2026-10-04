import { D2R, cameraTans, quatToMatrix } from './stitch.js';

const norm3 = v => { const n = Math.hypot(v[0], v[1], v[2]); return [v[0]/n, v[1]/n, v[2]/n]; };
const mulRv = (R, a) => [R[0]*a[0]+R[1]*a[1]+R[2]*a[2], R[3]*a[0]+R[4]*a[1]+R[5]*a[2], R[6]*a[0]+R[7]*a[1]+R[8]*a[2]];
const mulRtv = (R, a) => [R[0]*a[0]+R[3]*a[1]+R[6]*a[2], R[1]*a[0]+R[4]*a[1]+R[7]*a[2], R[2]*a[0]+R[5]*a[1]+R[8]*a[2]];
function matToQuat(R) { // returns [x,y,z,w]
  const t = R[0] + R[4] + R[8]; let x, y, z, w;
  if (t > 0) { const s = 0.5/Math.sqrt(t+1); w = 0.25/s; x = (R[7]-R[5])*s; y = (R[2]-R[6])*s; z = (R[3]-R[1])*s; }
  else if (R[0] > R[4] && R[0] > R[8]) { const s = 2*Math.sqrt(1+R[0]-R[4]-R[8]); w = (R[7]-R[5])/s; x = 0.25*s; y = (R[1]+R[3])/s; z = (R[2]+R[6])/s; }
  else if (R[4] > R[8]) { const s = 2*Math.sqrt(1+R[4]-R[0]-R[8]); w = (R[2]-R[6])/s; x = (R[1]+R[3])/s; y = 0.25*s; z = (R[5]+R[7])/s; }
  else { const s = 2*Math.sqrt(1+R[8]-R[0]-R[4]); w = (R[3]-R[1])/s; x = (R[2]+R[6])/s; y = (R[5]+R[7])/s; z = 0.25*s; }
  return [x, y, z, w];
}
// Best rotation R with R*a ~ b (Horn's quaternion method), S = sum w * a b^T (row-major 3x3)
function hornSolve(S, startR) {
  const [Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz] = S;
  const N = [
    [Sxx+Syy+Szz, Syz-Szy, Szx-Sxz, Sxy-Syx],
    [Syz-Szy, Sxx-Syy-Szz, Sxy+Syx, Szx+Sxz],
    [Szx-Sxz, Sxy+Syx, -Sxx+Syy-Szz, Syz+Szy],
    [Sxy-Syx, Szx+Sxz, Syz+Szy, -Sxx-Syy+Szz]];
  let shift = 0; for (const r of N) for (const v of r) shift += Math.abs(v);
  const q0 = matToQuat(startR); let q = [q0[3], q0[0], q0[1], q0[2]];
  for (let it = 0; it < 80; it++) {
    const n = [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) { n[i] = shift*q[i]; for (let j = 0; j < 4; j++) n[i] += N[i][j]*q[j]; }
    const l = Math.hypot(...n); q = n.map(v => v/l);
  }
  return quatToMatrix([q[1], q[2], q[3], q[0]]);
}
function pixRay(x, y, w, h, tx, ty) { return norm3([(2*(x+0.5)/w - 1)*tx, (1 - 2*(y+0.5)/h)*ty, -1]); }

async function loadCv(src) {
  if (!window.cv) await new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error('Could not load the photo aligner')); document.head.appendChild(s); });
  let cv = window.cv;
  if (cv instanceof Promise) cv = await cv;
  else if (!cv.Mat) await new Promise(r => { cv.onRuntimeInitialized = r; });
  return cv;
}

// frames: [{data, w, h, R, tanX, tanY}] with R from the motion sensor. Returns new frames with corrected R, lens and brightness.
export async function refine(frames, longFovGuess, opts = {}) {
  const say = opts.onProgress || (() => {});
  const cv = opts.cv || await loadCv(opts.cvSrc || 'https://docs.opencv.org/4.x/opencv.js');
  const n = frames.length; if (n < 2) return { frames, fov: longFovGuess, residual: null };
  say('Finding details in your photos');
  const orb = new cv.ORB(1200), empty = new cv.Mat();
  const feats = [];
  for (let i = 0; i < n; i++) {
    const f = frames[i];
    const src = cv.matFromImageData({ data: f.data, width: f.w, height: f.h });
    const g = new cv.Mat(); cv.cvtColor(src, g, cv.COLOR_RGBA2GRAY); src.delete();
    const kp = new cv.KeyPointVector(), desc = new cv.Mat();
    orb.detectAndCompute(g, empty, kp, desc);
    const pts = []; for (let k = 0; k < kp.size(); k++) { const p = kp.get(k).pt; pts.push([p.x, p.y]); }
    kp.delete(); feats.push({ pts, desc, gray: g });
    if (i % 5 === 0) await new Promise(r => setTimeout(r, 0));
  }
  say('Matching overlapping photos');
  const bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
  const fwd = frames.map(f => [-f.R[2], -f.R[5], -f.R[8]]);
  const pairs = [];
  for (let i = 0; i < n; i++) for (let j = i+1; j < n; j++) {
    if (fwd[i][0]*fwd[j][0] + fwd[i][1]*fwd[j][1] + fwd[i][2]*fwd[j][2] < 0.34) continue;
    const A = feats[i], B = feats[j]; if (A.desc.rows < 10 || B.desc.rows < 10) continue;
    const knn = new cv.DMatchVectorVector(); bf.knnMatch(A.desc, B.desc, knn, 2);
    const fi = frames[i], fj = frames[j], [txi, tyi] = cameraTans(fi.w, fi.h, longFovGuess), [txj, tyj] = cameraTans(fj.w, fj.h, longFovGuess);
    const cand = [];
    for (let k = 0; k < knn.size(); k++) {
      const m = knn.get(k); if (m.size() < 2) continue;
      const a = m.get(0), b = m.get(1); if (a.distance > 0.78*b.distance) continue;
      const pa = A.pts[a.queryIdx], pb = B.pts[a.trainIdx];
      const wa = mulRv(fi.R, pixRay(pa[0], pa[1], fi.w, fi.h, txi, tyi)), wb = mulRv(fj.R, pixRay(pb[0], pb[1], fj.w, fj.h, txj, tyj));
      if (wa[0]*wb[0] + wa[1]*wb[1] + wa[2]*wb[2] < Math.cos(15*D2R)) continue;
      cand.push([pa, pb]);
    }
    knn.delete();
    if (cand.length < 12) continue;
    const ma = cv.matFromArray(cand.length, 1, cv.CV_32FC2, cand.flatMap(c => c[0]));
    const mb = cv.matFromArray(cand.length, 1, cv.CV_32FC2, cand.flatMap(c => c[1]));
    const mask = new cv.Mat(); const H = cv.findHomography(ma, mb, cv.RANSAC, 4, mask);
    const inl = []; if (!H.empty()) for (let k = 0; k < cand.length; k++) if (mask.data[k]) inl.push(cand[k]);
    ma.delete(); mb.delete(); mask.delete(); H.delete();
    if (inl.length >= 12) pairs.push({ i, j, m: inl });
    if (pairs.length % 8 === 0) await new Promise(r => setTimeout(r, 0));
  }
  bf.delete(); orb.delete(); empty.delete();
  const deg = new Array(n).fill(0); for (const p of pairs) { deg[p.i] += p.m.length; deg[p.j] += p.m.length; }
  const anchor = deg.indexOf(Math.max(...deg));
  const UP = [0, 1, 0];
  const upLocal = frames.map(f => mulRtv(f.R, UP)); // gravity from the sensor is reliable; heading is not

  function solve(L, sweeps) {
    const tans = frames.map(f => cameraTans(f.w, f.h, L));
    const rays = pairs.map(p => p.m.map(([pa, pb]) => [
      pixRay(pa[0], pa[1], frames[p.i].w, frames[p.i].h, ...tans[p.i]),
      pixRay(pb[0], pb[1], frames[p.j].w, frames[p.j].h, ...tans[p.j])]));
    const R = frames.map(f => f.R.slice());
    for (let s = 0; s < sweeps; s++) for (let i = 0; i < n; i++) {
      if (deg[i] < 12) continue;
      const S = new Array(9).fill(0);
      const add = (a, b, w) => { for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r*3+c] += w*a[r]*b[c]; };
      pairs.forEach((p, k) => {
        if (p.i !== i && p.j !== i) return;
        const other = p.i === i ? p.j : p.i;
        for (const [ra, rb] of rays[k]) { const mine = p.i === i ? ra : rb, theirs = p.i === i ? rb : ra; add(mine, mulRv(R[other], theirs), 1); }
      });
      add(upLocal[i], UP, Math.max(8, deg[i]*0.15));
      if (i === anchor) { const f0 = frames[i].R; for (const e of [[1,0,0],[0,0,1]]) add(e, mulRv(f0, e), 20); }
      R[i] = hornSolve(S, R[i]);
    }
    let sum = 0, cnt = 0;
    pairs.forEach((p, k) => { for (const [ra, rb] of rays[k]) { const a = mulRv(R[p.i], ra), b = mulRv(R[p.j], rb); const e = Math.acos(Math.min(1, a[0]*b[0]+a[1]*b[1]+a[2]*b[2])); sum += Math.min(e, 3*D2R); cnt++; } });
    return { R, tans, err: cnt ? sum/cnt/D2R : 99 };
  }
  say('Lining everything up');
  let best = null, bestL = longFovGuess;
  if (pairs.length) {
    for (let L = 44; L <= 80; L += 2) { const r = solve(L, 12); if (!best || r.err < best.err) { best = r; bestL = L; } await new Promise(r => setTimeout(r, 0)); }
    for (let L = bestL - 1.5; L <= bestL + 1.5; L += 0.5) { const r = solve(L, 12); if (r.err < best.err) { best = r; bestL = L; } }
    best = solve(bestL, 40);
  }
  // brightness matching between overlapping photos
  const box = (g, x, y) => { let s = 0, c = 0; for (let yy = Math.max(0, (y|0)-3); yy <= Math.min(g.rows-1, (y|0)+3); yy++) for (let xx = Math.max(0, (x|0)-3); xx <= Math.min(g.cols-1, (x|0)+3); xx++) { s += g.data[yy*g.cols+xx]; c++; } return s/c + 1; };
  const lg = new Array(n).fill(0);
  const rel = pairs.map(p => { let s = 0; for (const [pa, pb] of p.m) s += Math.log(box(feats[p.i].gray, pa[0], pa[1])) - Math.log(box(feats[p.j].gray, pb[0], pb[1])); return { i: p.i, j: p.j, d: s/p.m.length, w: p.m.length }; });
  for (let it = 0; it < 100; it++) for (let i = 0; i < n; i++) {
    let num = 0, den = 2;
    for (const r of rel) { if (r.i === i) { num += r.w*(lg[r.j] - r.d); den += r.w; } else if (r.j === i) { num += r.w*(lg[r.i] + r.d); den += r.w; } }
    lg[i] = num/den;
  }
  feats.forEach(f => { f.desc.delete(); f.gray.delete(); });
  const out = frames.map((f, i) => {
    const gain = Math.exp(lg[i]), data = new Uint8ClampedArray(f.data);
    if (Math.abs(gain - 1) > 0.01) for (let k = 0; k < data.length; k += 4) { data[k] *= gain; data[k+1] *= gain; data[k+2] *= gain; }
    const R = best ? best.R[i] : f.R, [tanX, tanY] = best ? best.tans[i] : [f.tanX, f.tanY];
    return { ...f, data, R, tanX, tanY };
  });
  return { frames: out, fov: bestL, residual: best ? best.err : null, pairs: pairs.length };
}
