/**
 * Compact, readable implementations of classic ML algorithms for 2-D teaching datasets.
 * They are exact implementations of the textbook algorithms (not approximations
 * dressed up for animation), sized for tens to hundreds of points.
 */
import { solve, transpose, matMul, matVec, identity, matAdd, matScale, type Matrix, type Vec } from "@/lib/math/linalg";
import { mulberry32, randn, randInt, type RNG } from "@/lib/math/random";

export type Point = { x: number; y: number; label: number };

/* ------------------------------ datasets ------------------------------ */

export type DatasetId = "blobs" | "moons" | "circles" | "xor" | "spiral" | "linear";

export function makeDataset(id: DatasetId, n = 160, noise = 0.15, seed = 7): Point[] {
  const rng = mulberry32(seed);
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    let x = 0, y = 0;
    switch (id) {
      case "blobs": {
        const c = label ? [0.9, 0.7] : [-0.8, -0.6];
        x = c[0] + randn(rng) * (0.35 + noise);
        y = c[1] + randn(rng) * (0.35 + noise);
        break;
      }
      case "linear": {
        x = rng() * 4 - 2;
        y = rng() * 4 - 2;
        const s = x * 0.8 - y + randn(rng) * noise * 2;
        pts.push({ x: x * 0.9, y: y * 0.9, label: s > 0 ? 1 : 0 });
        continue;
      }
      case "moons": {
        const t = rng() * Math.PI;
        if (label === 0) { x = Math.cos(t) - 0.5; y = Math.sin(t) - 0.25; }
        else { x = 1 - Math.cos(t) - 0.5; y = 0.25 - Math.sin(t) + 0.0; }
        x = x * 1.3 + randn(rng) * noise;
        y = y * 1.3 + randn(rng) * noise;
        break;
      }
      case "circles": {
        const t = rng() * 2 * Math.PI;
        const r = label ? 0.6 : 1.6;
        x = r * Math.cos(t) + randn(rng) * noise;
        y = r * Math.sin(t) + randn(rng) * noise;
        break;
      }
      case "xor": {
        x = rng() * 4 - 2;
        y = rng() * 4 - 2;
        const lab = x * y > 0 ? 1 : 0;
        pts.push({ x: x + randn(rng) * noise * 0.5, y: y + randn(rng) * noise * 0.5, label: lab });
        continue;
      }
      case "spiral": {
        const k = Math.floor(i / 2);
        const r = (k / (n / 2)) * 1.9 + 0.1;
        const t = 1.75 * (k / (n / 2)) * 2 * Math.PI + label * Math.PI;
        x = r * Math.sin(t) + randn(rng) * noise * 0.6;
        y = r * Math.cos(t) + randn(rng) * noise * 0.6;
        break;
      }
    }
    pts.push({ x, y, label });
  }
  return pts;
}

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

/* ------------------------------ regression ------------------------------ */

/** Polynomial design matrix [1, x, x², …, x^d]. */
export function polyDesign(xs: number[], degree: number): Matrix {
  return xs.map((x) => Array.from({ length: degree + 1 }, (_, k) => x ** k));
}

/**
 * Ridge / OLS via the normal equations solved as a linear system (no explicit inverse):
 * (XᵀX + λI') w = Xᵀy, where the intercept is not penalised.
 */
export function ridgeFit(X: Matrix, y: Vec, lambda = 0): Vec | null {
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const d = XtX.length;
  const I = identity(d);
  I[0][0] = 0; // don't penalise the bias
  const A = matAdd(XtX, matScale(I, lambda));
  return solve(A, matVec(Xt, y));
}

const softThreshold = (z: number, g: number) => (z > g ? z - g : z < -g ? z + g : 0);

/**
 * Lasso / Elastic Net by cyclic coordinate descent on
 * (1/2n)‖y − Xw‖² + λ(α‖w‖₁ + (1−α)/2 ‖w‖²). Column 0 is the unpenalised intercept.
 */
export function elasticNetFit(X: Matrix, y: Vec, lambda: number, alpha = 1, iters = 500): Vec {
  const n = X.length;
  const d = X[0].length;
  const w = Array(d).fill(0);
  const colSq = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j] * r[j], 0) / n);
  const resid = y.slice();
  for (let it = 0; it < iters; it++) {
    let maxDelta = 0;
    for (let j = 0; j < d; j++) {
      if (colSq[j] === 0) continue;
      let rho = 0;
      for (let i = 0; i < n; i++) rho += X[i][j] * (resid[i] + X[i][j] * w[j]);
      rho /= n;
      const wNew = j === 0 ? rho / colSq[j] : softThreshold(rho, lambda * alpha) / (colSq[j] + lambda * (1 - alpha));
      const delta = wNew - w[j];
      if (delta !== 0) {
        for (let i = 0; i < n; i++) resid[i] -= X[i][j] * delta;
        w[j] = wNew;
        maxDelta = Math.max(maxDelta, Math.abs(delta));
      }
    }
    if (maxDelta < 1e-9) break;
  }
  return w;
}

export const predictLinear = (X: Matrix, w: Vec) => X.map((r) => r.reduce((s, x, j) => s + x * w[j], 0));

/* ------------------------------ logistic regression ------------------------------ */

export interface LogRegState { w: [number, number]; b: number; history: { loss: number }[] }

export function logisticStep(pts: Point[], st: LogRegState, lr: number, l2 = 0): LogRegState {
  let gw0 = 0, gw1 = 0, gb = 0, loss = 0;
  const n = pts.length;
  for (const p of pts) {
    const z = st.w[0] * p.x + st.w[1] * p.y + st.b;
    const a = sigmoid(z);
    const e = a - p.label;
    gw0 += e * p.x; gw1 += e * p.y; gb += e;
    loss += -(p.label * Math.log(Math.max(1e-12, a)) + (1 - p.label) * Math.log(Math.max(1e-12, 1 - a)));
  }
  loss = loss / n + (l2 / 2) * (st.w[0] ** 2 + st.w[1] ** 2);
  const w: [number, number] = [st.w[0] - lr * (gw0 / n + l2 * st.w[0]), st.w[1] - lr * (gw1 / n + l2 * st.w[1])];
  return { w, b: st.b - lr * (gb / n), history: [...st.history, { loss }] };
}

/* ------------------------------ KNN ------------------------------ */

export function knnPredict(pts: Point[], qx: number, qy: number, k: number, metric: "euclidean" | "manhattan" = "euclidean") {
  const d = pts.map((p, i) => ({ i, d: metric === "euclidean" ? Math.hypot(p.x - qx, p.y - qy) : Math.abs(p.x - qx) + Math.abs(p.y - qy) }));
  d.sort((a, b) => a.d - b.d);
  const nb = d.slice(0, k);
  const votes = nb.reduce((s, n) => s + pts[n.i].label, 0);
  return { prob: votes / k, neighbors: nb };
}

/* ------------------------------ Gaussian Naive Bayes ------------------------------ */

export interface GNBModel { classes: { prior: number; mu: [number, number]; var: [number, number] }[] }

export function gnbFit(pts: Point[], varSmoothing = 1e-3): GNBModel {
  const classes = [0, 1].map((c) => {
    const s = pts.filter((p) => p.label === c);
    const n = s.length || 1;
    const mx = s.reduce((a, p) => a + p.x, 0) / n;
    const my = s.reduce((a, p) => a + p.y, 0) / n;
    const vx = s.reduce((a, p) => a + (p.x - mx) ** 2, 0) / n + varSmoothing;
    const vy = s.reduce((a, p) => a + (p.y - my) ** 2, 0) / n + varSmoothing;
    return { prior: s.length / pts.length, mu: [mx, my] as [number, number], var: [vx, vy] as [number, number] };
  });
  return { classes };
}

/** log-space posterior P(y=1 | x) for numerical stability */
export function gnbPredict(m: GNBModel, x: number, y: number): number {
  const ll = m.classes.map((c) => {
    if (c.prior === 0) return -Infinity;
    const lx = -0.5 * Math.log(2 * Math.PI * c.var[0]) - (x - c.mu[0]) ** 2 / (2 * c.var[0]);
    const ly = -0.5 * Math.log(2 * Math.PI * c.var[1]) - (y - c.mu[1]) ** 2 / (2 * c.var[1]);
    return Math.log(c.prior) + lx + ly;
  });
  const mx = Math.max(...ll);
  const e = ll.map((l) => Math.exp(l - mx));
  return e[1] / (e[0] + e[1]);
}

/* ------------------------------ Decision tree (CART) ------------------------------ */

export type TreeNode =
  | { leaf: true; value: number; n: number; impurity: number; depth: number }
  | { leaf: false; feature: 0 | 1; threshold: number; left: TreeNode; right: TreeNode; n: number; impurity: number; depth: number; gain: number };

const gini = (p: number) => 2 * p * (1 - p);
const entropy = (p: number) => (p <= 0 || p >= 1 ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p)));

export function buildTree(
  pts: Point[],
  maxDepth: number,
  minLeaf = 1,
  criterion: "gini" | "entropy" = "gini",
  depth = 0,
  featureSubset?: (rng: RNG) => (0 | 1)[],
  rng?: RNG,
): TreeNode {
  const imp = criterion === "gini" ? gini : entropy;
  const n = pts.length;
  const p = n ? pts.reduce((s, q) => s + q.label, 0) / n : 0;
  const node0 = { n, impurity: imp(p), depth };
  if (depth >= maxDepth || n < 2 * minLeaf || p === 0 || p === 1) return { leaf: true, value: p, ...node0 };
  let best: { f: 0 | 1; t: number; gain: number } | null = null;
  const feats: (0 | 1)[] = featureSubset && rng ? featureSubset(rng) : [0, 1];
  for (const f of feats) {
    const sorted = pts.slice().sort((a, b) => (f === 0 ? a.x - b.x : a.y - b.y));
    let leftPos = 0;
    const totalPos = p * n;
    for (let i = 0; i < n - 1; i++) {
      leftPos += sorted[i].label;
      const v = f === 0 ? sorted[i].x : sorted[i].y;
      const vn = f === 0 ? sorted[i + 1].x : sorted[i + 1].y;
      if (v === vn) continue;
      const nl = i + 1;
      const nr = n - nl;
      if (nl < minLeaf || nr < minLeaf) continue;
      const g = imp(p) - (nl / n) * imp(leftPos / nl) - (nr / n) * imp((totalPos - leftPos) / nr);
      if (!best || g > best.gain + 1e-12) best = { f, t: (v + vn) / 2, gain: g };
    }
  }
  if (!best || best.gain <= 1e-12) return { leaf: true, value: p, ...node0 };
  const L = pts.filter((q) => (best!.f === 0 ? q.x : q.y) <= best!.t);
  const R = pts.filter((q) => (best!.f === 0 ? q.x : q.y) > best!.t);
  return {
    leaf: false,
    feature: best.f,
    threshold: best.t,
    gain: best.gain,
    left: buildTree(L, maxDepth, minLeaf, criterion, depth + 1, featureSubset, rng),
    right: buildTree(R, maxDepth, minLeaf, criterion, depth + 1, featureSubset, rng),
    ...node0,
  };
}

export function treePredict(node: TreeNode, x: number, y: number): number {
  let cur = node;
  while (!cur.leaf) cur = (cur.feature === 0 ? x : y) <= cur.threshold ? cur.left : cur.right;
  return cur.value;
}

export function treeDepth(node: TreeNode): number {
  return node.leaf ? 0 : 1 + Math.max(treeDepth(node.left), treeDepth(node.right));
}
export function treeLeaves(node: TreeNode): number {
  return node.leaf ? 1 : treeLeaves(node.left) + treeLeaves(node.right);
}

/** Random forest: bootstrap samples + random feature choice at each split. */
export function buildForest(pts: Point[], nTrees: number, maxDepth: number, seed = 1) {
  const rng = mulberry32(seed);
  const trees: { tree: TreeNode; oob: Set<number> }[] = [];
  for (let t = 0; t < nTrees; t++) {
    const idx = Array.from({ length: pts.length }, () => randInt(rng, 0, pts.length - 1));
    const inBag = new Set(idx);
    const oob = new Set(pts.map((_, i) => i).filter((i) => !inBag.has(i)));
    const sample = idx.map((i) => pts[i]);
    // with 2 features, max_features = 1 → pick one feature uniformly per split
    const tree = buildTree(sample, maxDepth, 1, "gini", 0, (r) => [r() < 0.5 ? 0 : 1], rng);
    trees.push({ tree, oob });
  }
  return trees;
}

/* ------------------------------ SVM (simplified SMO) ------------------------------ */

export type Kernel = { type: "linear" } | { type: "rbf"; gamma: number } | { type: "poly"; degree: number; coef0: number };

export function kernelFn(k: Kernel) {
  return (a: [number, number], b: [number, number]) => {
    if (k.type === "linear") return a[0] * b[0] + a[1] * b[1];
    if (k.type === "rbf") return Math.exp(-k.gamma * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2));
    return (a[0] * b[0] + a[1] * b[1] + k.coef0) ** k.degree;
  };
}

export interface SVMModel { alpha: number[]; b: number; X: [number, number][]; y: number[]; kernel: Kernel }

/** Platt's SMO (simplified version, Ng's CS229 notes) — exact dual optimisation for small data. */
export function svmFit(pts: Point[], C: number, kernel: Kernel, tol = 1e-3, maxPasses = 12, seed = 3): SVMModel {
  const rng = mulberry32(seed);
  const X = pts.map((p) => [p.x, p.y] as [number, number]);
  const y = pts.map((p) => (p.label === 1 ? 1 : -1));
  const n = X.length;
  const K = kernelFn(kernel);
  const G: number[][] = X.map((a) => X.map((b) => K(a, b)));
  const alpha = Array(n).fill(0);
  let b = 0;
  const f = (i: number) => {
    let s = b;
    for (let j = 0; j < n; j++) if (alpha[j] !== 0) s += alpha[j] * y[j] * G[j][i];
    return s;
  };
  let passes = 0;
  let guard = 0;
  while (passes < maxPasses && guard++ < 400) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const Ei = f(i) - y[i];
      if ((y[i] * Ei < -tol && alpha[i] < C) || (y[i] * Ei > tol && alpha[i] > 0)) {
        let j = Math.floor(rng() * (n - 1));
        if (j >= i) j++;
        const Ej = f(j) - y[j];
        const ai = alpha[i], aj = alpha[j];
        const L = y[i] !== y[j] ? Math.max(0, aj - ai) : Math.max(0, ai + aj - C);
        const H = y[i] !== y[j] ? Math.min(C, C + aj - ai) : Math.min(C, ai + aj);
        if (L === H) continue;
        const eta = 2 * G[i][j] - G[i][i] - G[j][j];
        if (eta >= 0) continue;
        let ajNew = aj - (y[j] * (Ei - Ej)) / eta;
        ajNew = Math.min(H, Math.max(L, ajNew));
        if (Math.abs(ajNew - aj) < 1e-5) continue;
        const aiNew = ai + y[i] * y[j] * (aj - ajNew);
        const b1 = b - Ei - y[i] * (aiNew - ai) * G[i][i] - y[j] * (ajNew - aj) * G[i][j];
        const b2 = b - Ej - y[i] * (aiNew - ai) * G[i][j] - y[j] * (ajNew - aj) * G[j][j];
        alpha[i] = aiNew;
        alpha[j] = ajNew;
        b = aiNew > 0 && aiNew < C ? b1 : ajNew > 0 && ajNew < C ? b2 : (b1 + b2) / 2;
        changed++;
      }
    }
    passes = changed === 0 ? passes + 1 : 0;
  }
  return { alpha, b, X, y, kernel };
}

export function svmDecision(m: SVMModel, x: number, yv: number): number {
  const K = kernelFn(m.kernel);
  let s = m.b;
  for (let i = 0; i < m.X.length; i++) if (m.alpha[i] > 1e-8) s += m.alpha[i] * m.y[i] * K(m.X[i], [x, yv]);
  return s;
}

/* ------------------------------ clustering ------------------------------ */

export type P2 = [number, number];

export function makeClusters(k: number, nPer: number, spread: number, seed: number): P2[] {
  const rng = mulberry32(seed);
  const centers: P2[] = Array.from({ length: k }, (_, i) => {
    const t = (i / k) * 2 * Math.PI + 0.4;
    return [1.4 * Math.cos(t) + (rng() - 0.5) * 0.5, 1.4 * Math.sin(t) + (rng() - 0.5) * 0.5];
  });
  const pts: P2[] = [];
  for (const c of centers) for (let i = 0; i < nPer; i++) pts.push([c[0] + randn(rng) * spread, c[1] + randn(rng) * spread]);
  return pts;
}

const d2 = (a: P2, b: P2) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

export function kmeansInit(pts: P2[], k: number, method: "random" | "kmeans++", seed: number): P2[] {
  const rng = mulberry32(seed);
  if (method === "random") {
    const used = new Set<number>();
    const out: P2[] = [];
    while (out.length < k && used.size < pts.length) {
      const i = Math.floor(rng() * pts.length);
      if (!used.has(i)) { used.add(i); out.push([...pts[i]] as P2); }
    }
    return out;
  }
  const out: P2[] = [[...pts[Math.floor(rng() * pts.length)]] as P2];
  while (out.length < k) {
    const D = pts.map((p) => Math.min(...out.map((c) => d2(p, c))));
    const tot = D.reduce((a, b) => a + b, 0);
    let r = rng() * tot;
    let i = 0;
    for (; i < D.length - 1; i++) { r -= D[i]; if (r <= 0) break; }
    out.push([...pts[i]] as P2);
  }
  return out;
}

export function kmeansAssign(pts: P2[], C: P2[]): number[] {
  return pts.map((p) => {
    let best = 0, bd = Infinity;
    C.forEach((c, j) => { const d = d2(p, c); if (d < bd) { bd = d; best = j; } });
    return best;
  });
}

export function kmeansUpdate(pts: P2[], assign: number[], C: P2[]): P2[] {
  return C.map((c, j) => {
    const mem = pts.filter((_, i) => assign[i] === j);
    if (!mem.length) return c; // empty cluster keeps its centroid
    return [mem.reduce((s, p) => s + p[0], 0) / mem.length, mem.reduce((s, p) => s + p[1], 0) / mem.length] as P2;
  });
}

export const inertia = (pts: P2[], assign: number[], C: P2[]) => pts.reduce((s, p, i) => s + d2(p, C[assign[i]]), 0);

/** DBSCAN. Labels: -1 noise, else cluster id. Also returns point role. */
export function dbscan(pts: P2[], eps: number, minPts: number) {
  const n = pts.length;
  const eps2 = eps * eps;
  const nbrs = pts.map((p) => pts.map((q, j) => (d2(p, q) <= eps2 ? j : -1)).filter((j) => j >= 0));
  const core = nbrs.map((nb) => nb.length >= minPts);
  const label = Array(n).fill(-2); // -2 unvisited
  let cid = 0;
  for (let i = 0; i < n; i++) {
    if (label[i] !== -2) continue;
    if (!core[i]) { label[i] = -1; continue; }
    label[i] = cid;
    const queue = [...nbrs[i]];
    while (queue.length) {
      const j = queue.pop()!;
      if (label[j] === -1) label[j] = cid; // border point
      if (label[j] !== -2) continue;
      label[j] = cid;
      if (core[j]) queue.push(...nbrs[j]);
    }
    cid++;
  }
  const role = label.map((l, i) => (l === -1 ? "noise" : core[i] ? "core" : "border")) as ("core" | "border" | "noise")[];
  return { label, role, clusters: cid };
}

/** Gaussian mixture via EM with full 2×2 covariances. */
export interface GMMComp { pi: number; mu: P2; cov: [[number, number], [number, number]] }

export function gaussPdf2(p: P2, c: GMMComp): number {
  const [[a, b], [, d]] = c.cov;
  const det = a * d - b * b;
  if (det <= 1e-12) return 0;
  const dx = p[0] - c.mu[0], dy = p[1] - c.mu[1];
  const q = (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det;
  return Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det));
}

export function gmmEStep(pts: P2[], comps: GMMComp[]) {
  let ll = 0;
  const R = pts.map((p) => {
    const w = comps.map((c) => c.pi * gaussPdf2(p, c));
    const s = w.reduce((a, b) => a + b, 0) || 1e-300;
    ll += Math.log(s);
    return w.map((x) => x / s);
  });
  return { R, logLik: ll };
}

export function gmmMStep(pts: P2[], R: number[][], reg = 1e-4): GMMComp[] {
  const K = R[0].length;
  const n = pts.length;
  return Array.from({ length: K }, (_, k) => {
    const Nk = R.reduce((s, r) => s + r[k], 0) || 1e-12;
    const mu: P2 = [R.reduce((s, r, i) => s + r[k] * pts[i][0], 0) / Nk, R.reduce((s, r, i) => s + r[k] * pts[i][1], 0) / Nk];
    let a = 0, b = 0, d = 0;
    R.forEach((r, i) => {
      const dx = pts[i][0] - mu[0], dy = pts[i][1] - mu[1];
      a += r[k] * dx * dx; b += r[k] * dx * dy; d += r[k] * dy * dy;
    });
    return { pi: Nk / n, mu, cov: [[a / Nk + reg, b / Nk], [b / Nk, d / Nk + reg]] };
  });
}

/** Agglomerative clustering; returns merge history (like scipy linkage). */
export function agglomerative(pts: P2[], linkage: "single" | "complete" | "average" | "ward") {
  let clusters = pts.map((_, i) => ({ id: i, members: [i] }));
  const merges: { a: number; b: number; dist: number; id: number; size: number }[] = [];
  let next = pts.length;
  const dist = (A: number[], B: number[]) => {
    if (linkage === "ward") {
      const ca = centroid(A), cb = centroid(B);
      return Math.sqrt(((2 * A.length * B.length) / (A.length + B.length)) * d2(ca, cb));
    }
    let agg = linkage === "single" ? Infinity : linkage === "complete" ? -Infinity : 0;
    for (const i of A) for (const j of B) {
      const d = Math.sqrt(d2(pts[i], pts[j]));
      if (linkage === "single") agg = Math.min(agg, d);
      else if (linkage === "complete") agg = Math.max(agg, d);
      else agg += d;
    }
    return linkage === "average" ? agg / (A.length * B.length) : agg;
  };
  const centroid = (A: number[]): P2 => [A.reduce((s, i) => s + pts[i][0], 0) / A.length, A.reduce((s, i) => s + pts[i][1], 0) / A.length];
  while (clusters.length > 1) {
    let bi = 0, bj = 1, bd = Infinity;
    for (let i = 0; i < clusters.length; i++)
      for (let j = i + 1; j < clusters.length; j++) {
        const d = dist(clusters[i].members, clusters[j].members);
        if (d < bd) { bd = d; bi = i; bj = j; }
      }
    const A = clusters[bi], B = clusters[bj];
    const merged = { id: next++, members: [...A.members, ...B.members] };
    merges.push({ a: A.id, b: B.id, dist: bd, id: merged.id, size: merged.members.length });
    clusters = clusters.filter((_, k) => k !== bi && k !== bj);
    clusters.push(merged);
  }
  return merges;
}

/** Cut a linkage into k flat clusters. */
export function cutTree(n: number, merges: { a: number; b: number; id: number }[], k: number): number[] {
  const parent = new Map<number, number[]>();
  for (let i = 0; i < n; i++) parent.set(i, [i]);
  const active = new Set(Array.from({ length: n }, (_, i) => i));
  for (const m of merges) {
    if (active.size <= k) break;
    parent.set(m.id, [...parent.get(m.a)!, ...parent.get(m.b)!]);
    active.delete(m.a); active.delete(m.b); active.add(m.id);
  }
  const lab = Array(n).fill(0);
  [...active].forEach((cid, c) => parent.get(cid)!.forEach((i) => (lab[i] = c)));
  return lab;
}

/* ------------------------------ gradient boosting (1-D regression) ------------------------------ */

export interface Stump { t: number; left: number; right: number }

export function fitStump(xs: number[], r: number[]): Stump {
  const idx = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const n = xs.length;
  const total = r.reduce((a, b) => a + b, 0);
  let best: Stump = { t: Infinity, left: total / n, right: total / n };
  let bestSse = Infinity;
  let ls = 0;
  for (let k = 0; k < n - 1; k++) {
    ls += r[idx[k]];
    if (xs[idx[k]] === xs[idx[k + 1]]) continue;
    const nl = k + 1, nr = n - nl;
    const ml = ls / nl, mr = (total - ls) / nr;
    // SSE = Σr² − nl·ml² − nr·mr² ; minimise the negative part
    const sse = -(nl * ml * ml + nr * mr * mr);
    if (sse < bestSse) { bestSse = sse; best = { t: (xs[idx[k]] + xs[idx[k + 1]]) / 2, left: ml, right: mr }; }
  }
  return best;
}

export const stumpPredict = (s: Stump, x: number) => (x <= s.t ? s.left : s.right);

export function gradientBoost(xs: number[], ys: number[], nStages: number, lr: number) {
  const f0 = ys.reduce((a, b) => a + b, 0) / ys.length;
  const stumps: Stump[] = [];
  let pred = ys.map(() => f0);
  const stages = [{ pred: pred.slice(), resid: ys.map((y, i) => y - pred[i]) }];
  for (let m = 0; m < nStages; m++) {
    const resid = ys.map((y, i) => y - pred[i]); // negative gradient of ½(y−F)²
    const s = fitStump(xs, resid);
    stumps.push(s);
    pred = pred.map((p, i) => p + lr * stumpPredict(s, xs[i]));
    stages.push({ pred: pred.slice(), resid: ys.map((y, i) => y - pred[i]) });
  }
  return { f0, stumps, stages, predict: (x: number, upTo = stumps.length) => stumps.slice(0, upTo).reduce((s, st) => s + lr * stumpPredict(st, x), f0) };
}

/** AdaBoost with axis-aligned stumps on 2-D points (labels ±1). */
export function adaboost(pts: Point[], rounds: number) {
  const n = pts.length;
  let w = Array(n).fill(1 / n);
  const y = pts.map((p) => (p.label ? 1 : -1));
  const learners: { f: 0 | 1; t: number; s: 1 | -1; alpha: number; err: number }[] = [];
  const weights: number[][] = [w.slice()];
  for (let r = 0; r < rounds; r++) {
    let best = { f: 0 as 0 | 1, t: 0, s: 1 as 1 | -1, err: Infinity };
    for (const f of [0, 1] as const) {
      const vals = [...new Set(pts.map((p) => (f === 0 ? p.x : p.y)))].sort((a, b) => a - b);
      for (let k = 0; k < vals.length - 1; k++) {
        const t = (vals[k] + vals[k + 1]) / 2;
        for (const s of [1, -1] as const) {
          let err = 0;
          pts.forEach((p, i) => { const h = ((f === 0 ? p.x : p.y) > t ? 1 : -1) * s; if (h !== y[i]) err += w[i]; });
          if (err < best.err) best = { f, t, s, err };
        }
      }
    }
    const err = Math.min(1 - 1e-10, Math.max(1e-10, best.err));
    const alpha = 0.5 * Math.log((1 - err) / err);
    learners.push({ ...best, alpha, err });
    w = w.map((wi, i) => { const p = pts[i]; const h = ((best.f === 0 ? p.x : p.y) > best.t ? 1 : -1) * best.s; return wi * Math.exp(-alpha * y[i] * h); });
    const Z = w.reduce((a, b) => a + b, 0);
    w = w.map((wi) => wi / Z);
    weights.push(w.slice());
  }
  const score = (x: number, yv: number, upTo = learners.length) =>
    learners.slice(0, upTo).reduce((s, l) => s + l.alpha * ((l.f === 0 ? x : yv) > l.t ? 1 : -1) * l.s, 0);
  return { learners, weights, score };
}
