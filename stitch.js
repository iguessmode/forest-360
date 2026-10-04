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
export function makeTargets(yaw0) {
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
// frames: [{data, w, h, R, tanX, tanY}] -> RGBA equirectangular image (W x H)
export async function stitch(frames, W, H, onProgress) {
  const out = new Uint8ClampedArray(W*H*4);
  const fw = frames.map(f => ({ ...f, fx: -f.R[2], fy: -f.R[5], fz: -f.R[8] }));
  const cosW = new Float64Array(W), sinW = new Float64Array(W);
  for (let px = 0; px < W; px++) { const p = 2*Math.PI*(px+0.5)/W; cosW[px] = Math.cos(p); sinW[px] = Math.sin(p); }
  const covered = new Uint8Array(W);
  for (let py = 0; py < H; py++) {
    const th = Math.PI*(py+0.5)/H, st = Math.sin(th), ct = Math.cos(th);
    let rs = 0, gs = 0, bs = 0, n = 0;
    covered.fill(0);
    for (let px = 0; px < W; px++) {
      const dx = cosW[px]*st, dy = ct, dz = sinW[px]*st;
      let r = 0, g = 0, b = 0, ws = 0, near = null, nearDot = 0.55;
      for (const f of fw) {
        const fd = dx*f.fx + dy*f.fy + dz*f.fz;
        if (fd > nearDot) { nearDot = fd; near = f; }
        if (fd < 0.7) continue;
        const R = f.R;
        const lz = R[2]*dx + R[5]*dy + R[8]*dz;
        if (lz >= -0.05) continue;
        const xn = (R[0]*dx + R[3]*dy + R[6]*dz)/(-lz)/f.tanX;
        const yn = (R[1]*dx + R[4]*dy + R[7]*dz)/(-lz)/f.tanY;
        if (xn <= -1 || xn >= 1 || yn <= -1 || yn >= 1) continue;
        let w = (1-xn*xn)*(1-yn*yn); w = w*w; w = w*w; w = w*w;
        const ix = ((xn+1)*0.5*f.w)|0, iy = ((1-yn)*0.5*f.h)|0, k = (iy*f.w + ix)*4;
        r += f.data[k]*w; g += f.data[k+1]*w; b += f.data[k+2]*w; ws += w;
      }
      if (ws === 0 && near) { // small gap between photos: stretch the closest photo's edge instead of leaving a hole
        const R = near.R, lz = Math.min(-0.05, R[2]*dx + R[5]*dy + R[8]*dz);
        const xn = Math.max(-0.999, Math.min(0.999, (R[0]*dx + R[3]*dy + R[6]*dz)/(-lz)/near.tanX));
        const yn = Math.max(-0.999, Math.min(0.999, (R[1]*dx + R[4]*dy + R[7]*dz)/(-lz)/near.tanY));
        const ix = ((xn+1)*0.5*near.w)|0, iy = ((1-yn)*0.5*near.h)|0, k = (iy*near.w + ix)*4;
        r = near.data[k]; g = near.data[k+1]; b = near.data[k+2]; ws = 1;
      }
      const o = (py*W + px)*4;
      if (ws > 0) { out[o] = r/ws; out[o+1] = g/ws; out[o+2] = b/ws; covered[px] = 1; rs += out[o]; gs += out[o+1]; bs += out[o+2]; n++; }
      out[o+3] = 255;
    }
    const ar = n ? rs/n : 12, ag = n ? gs/n : 14, ab = n ? bs/n : 22;
    for (let px = 0; px < W; px++) if (!covered[px]) { const o = (py*W + px)*4; out[o] = ar; out[o+1] = ag; out[o+2] = ab; }
    if (py % 32 === 0) { if (onProgress) onProgress(py/H); await new Promise(r => setTimeout(r, 0)); }
  }
  if (onProgress) onProgress(1);
  return out;
}
