/**
 * Support vector machine without a bias term, as in the IITM MLT notes: the separating
 * hyperplane passes through the origin (a constant feature can be appended to recover a bias).
 *
 *   primal:  min ½‖w‖² + C Σ ξᵢ   s.t.  yᵢ(wᵀxᵢ) ≥ 1 − ξᵢ,  ξᵢ ≥ 0        (C = ∞: hard margin)
 *   dual:    max Σ αᵢ − ½ αᵀ Y Xᵀ X Y α   s.t.  0 ≤ αᵢ ≤ C,   w = Σ αᵢ yᵢ xᵢ
 *
 * Solved by dual coordinate ascent: each αᵢ is maximised exactly with the others fixed,
 *   αᵢ ← clip(αᵢ + (1 − yᵢ wᵀxᵢ) / ‖xᵢ‖², 0, C),
 * which keeps w in sync and converges to the global optimum of this concave problem.
 */
export interface OriginSVM {
  w: number[];
  alpha: number[];
  /** dual objective after every sweep */
  dual: number[];
  /** primal objective after every sweep (∞ while some hard-margin constraint is violated) */
  primal: number[];
  /** weight vector after every sweep */
  wHist: number[][];
  sweeps: number;
  converged: boolean;
}

export const dualObjective = (alpha: number[], w: number[]) => alpha.reduce((s, a) => s + a, 0) - 0.5 * w.reduce((s, v) => s + v * v, 0);

export function primalObjective(X: number[][], y: number[], w: number[], C: number): number {
  let slack = 0;
  for (let i = 0; i < X.length; i++) slack += Math.max(0, 1 - y[i] * X[i].reduce((s, v, j) => s + v * w[j], 0));
  const reg = 0.5 * w.reduce((s, v) => s + v * v, 0);
  if (!Number.isFinite(C)) return slack > 1e-6 ? Infinity : reg;
  return reg + C * slack;
}

export function svmOrigin(X: number[][], y: number[], C = Infinity, maxSweeps = 2000, tol = 1e-9, alpha0?: number[]): OriginSVM {
  const n = X.length, d = X[0]?.length ?? 0;
  const alpha = alpha0 ? alpha0.slice() : Array(n).fill(0);
  const w = Array(d).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) w[j] += alpha[i] * y[i] * X[i][j];
  const sq = X.map((x) => x.reduce((s, v) => s + v * v, 0));
  const dual: number[] = [], primal: number[] = [], wHist: number[][] = [];
  let sweeps = 0, converged = false;
  for (; sweeps < maxSweeps; sweeps++) {
    let change = 0;
    for (let i = 0; i < n; i++) {
      if (sq[i] < 1e-12) continue;
      const g = 1 - y[i] * X[i].reduce((s, v, j) => s + v * w[j], 0);
      const a = Math.min(C, Math.max(0, alpha[i] + g / sq[i]));
      const da = a - alpha[i];
      if (da !== 0) {
        alpha[i] = a;
        for (let j = 0; j < d; j++) w[j] += da * y[i] * X[i][j];
        change = Math.max(change, Math.abs(da));
      }
    }
    dual.push(dualObjective(alpha, w));
    primal.push(primalObjective(X, y, w, C));
    wHist.push(w.slice());
    if (change < tol) { converged = true; sweeps++; break; }
  }
  return { w, alpha, dual, primal, wHist, sweeps, converged };
}

/** Largest margin achievable by a hyperplane through the origin: γ* = 1 / ‖w*‖ (0 if the data are not separable that way). */
export function maxMarginOrigin(X: number[][], y: number[]): { gamma: number; wUnit: number[] } {
  const r = svmOrigin(X, y, Infinity, 4000, 1e-10);
  const norm = Math.hypot(...r.w);
  const ok = norm > 0 && X.every((x, i) => y[i] * x.reduce((s, v, j) => s + v * r.w[j], 0) > 1 - 1e-4);
  return ok ? { gamma: 1 / norm, wUnit: r.w.map((v) => v / norm) } : { gamma: 0, wUnit: r.w.map((v) => v / (norm || 1)) };
}
