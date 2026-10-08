// Live "does the picture line up with the photos already taken?" check.
// Works on tiny grayscale copies so it can run several times a second on a phone.
export const LW = 72;
export function toGray(rgba, w, h, gw, gh) { // area-average down to gw x gh
  const g = new Float32Array(gw*gh), sx = w/gw, sy = h/gh;
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    let s = 0, n = 0;
    const x0 = (x*sx)|0, x1 = Math.max(x0+1, ((x+1)*sx)|0), y0 = (y*sy)|0, y1 = Math.max(y0+1, ((y+1)*sy)|0);
    for (let yy = y0; yy < y1; yy += 2) for (let xx = x0; xx < x1; xx += 2) { const k = (yy*w + xx)*4; s += rgba[k]*0.3 + rgba[k+1]*0.59 + rgba[k+2]*0.11; n++; }
    g[y*gw + x] = s/n;
  }
  return g;
}
// live: {g, gw, gh, tanX, tanY}; refs: [{g, gw, gh, R, tanX, tanY}]
// Returns {sx, sy, score, n}: the live picture is shifted by (sx, sy) small pixels from where it should be.
export function checkAlign(R, live, refs, S = 9) {
  const { gw, gh } = live, pred = new Float32Array(gw*gh).fill(NaN);
  let n = 0;
  for (let v = 1; v < gh-1; v++) for (let u = 1; u < gw-1; u++) {
    const l0 = ((u+0.5)/gw*2-1)*live.tanX, l1 = (1-(v+0.5)/gh*2)*live.tanY;
    const dx = R[0]*l0 + R[1]*l1 - R[2], dy = R[3]*l0 + R[4]*l1 - R[5], dz = R[6]*l0 + R[7]*l1 - R[8];
    let best = -1, bw = 0;
    for (const f of refs) {
      const Q = f.R, lz = Q[2]*dx + Q[5]*dy + Q[8]*dz; if (lz >= -0.05) continue;
      const xi = (Q[0]*dx + Q[3]*dy + Q[6]*dz)/(-lz)/f.tanX, yi = (Q[1]*dx + Q[4]*dy + Q[7]*dz)/(-lz)/f.tanY;
      if (Math.abs(xi) > 0.96 || Math.abs(yi) > 0.96) continue;
      const w = (1-xi*xi)*(1-yi*yi); if (w <= bw) continue;
      const x = (xi+1)*0.5*f.gw - 0.5, y = (1-yi)*0.5*f.gh - 0.5, x0 = Math.min(f.gw-2, Math.max(0, x|0)), y0 = Math.min(f.gh-2, Math.max(0, y|0)), ax = x-x0, ay = y-y0, G = f.g;
      best = (G[y0*f.gw+x0]*(1-ax) + G[y0*f.gw+x0+1]*ax)*(1-ay) + (G[(y0+1)*f.gw+x0]*(1-ax) + G[(y0+1)*f.gw+x0+1]*ax)*ay; bw = w;
    }
    if (bw > 0) { pred[v*gw + u] = best; n++; }
  }
  if (n < 120) return { sx: 0, sy: 0, dYaw: 0, dPitch: 0, score: 0, n };
  const pts = []; for (let k = 0; k < pred.length; k++) if (!Number.isNaN(pred[k])) pts.push(k);
  let pm = 0; for (const k of pts) pm += pred[k]; pm /= pts.length;
  let pv = 0; for (const k of pts) pv += (pred[k]-pm)**2; if (pv/pts.length < 25) return { sx: 0, sy: 0, dYaw: 0, dPitch: 0, score: 0, n }; // too plain to judge
  const L = live.g; let bestS = -2, bsx = 0, bsy = 0;
  const score = (sx, sy) => {
    let a = 0, b = 0, ab = 0, aa = 0, bb = 0, m = 0;
    for (const k of pts) { const u = k % gw + sx, v = ((k/gw)|0) + sy; if (u < 0 || v < 0 || u >= gw || v >= gh) continue; const p = pred[k], q = L[v*gw+u]; a += p; b += q; ab += p*q; aa += p*p; bb += q*q; m++; }
    if (m < 80) return -2; const c = ab - a*b/m, d = Math.sqrt((aa - a*a/m)*(bb - b*b/m)); return d > 0 ? c/d : -2;
  };
  for (let sy = -S; sy <= S; sy += 2) for (let sx = -S; sx <= S; sx += 2) { const s = score(sx, sy); if (s > bestS) { bestS = s; bsx = sx; bsy = sy; } }
  const cx = bsx, cy = bsy;
  for (let sy = cy-1; sy <= cy+1; sy++) for (let sx = cx-1; sx <= cx+1; sx++) { const s = score(sx, sy); if (s > bestS) { bestS = s; bsx = sx; bsy = sy; } }
  // sub-pixel: fit a parabola through the neighbours
  const sub = (m, c, p) => { const d = m - 2*c + p; return d < 0 ? Math.max(-0.5, Math.min(0.5, 0.5*(m - p)/d)) : 0; };
  const fx = sub(score(bsx-1, bsy), bestS, score(bsx+1, bsy)), fy = sub(score(bsx, bsy-1), bestS, score(bsx, bsy+1));
  // where the overlap sits matters: near the edge of a photo a small turn moves the picture more pixels
  let kx = 0, ky = 0;
  for (const k of pts) { const xn = ((k % gw)+0.5)/gw*2-1, yn = 1-(((k/gw)|0)+0.5)/gh*2; kx += 1 + (xn*live.tanX)**2; ky += 1 + (yn*live.tanY)**2; }
  kx /= pts.length; ky /= pts.length;
  const dYaw = Math.atan((bsx + fx)/(gw/2)*live.tanX/kx), dPitch = Math.atan((bsy + fy)/(gh/2)*live.tanY/ky);
  return { sx: bsx + fx, sy: bsy + fy, dYaw, dPitch, score: bestS, n };
}
// Turn the measured shift into the camera's real direction (small rotation about the camera's own axes).
export function correctR(R, a) {
  const ay = a.dYaw || 0, ax = a.dPitch || 0;
  const cy = Math.cos(ay), sy_ = Math.sin(ay), cx = Math.cos(ax), sx_ = Math.sin(ax);
  const M = [cy, sy_*sx_, sy_*cx, 0, cx, -sx_, -sy_, cy*sx_, cy*cx]; // Ry(ay)*Rx(ax)
  const out = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[i*3+j] = R[i*3]*M[j] + R[i*3+1]*M[3+j] + R[i*3+2]*M[6+j];
  return out;
}
