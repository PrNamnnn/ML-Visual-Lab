/**
 * Small, dependency-free linear algebra utilities.
 * Matrices are row-major arrays of rows: Matrix[i][j] = row i, column j.
 * Algorithms favour clarity + numerical stability (partial pivoting, Jacobi
 * rotations) over raw speed — inputs here are tiny educational examples.
 */
export type Vec = number[];
export type Matrix = number[][];

export const EPS = 1e-10;

/* ----------------------------- vectors ----------------------------- */

export const add = (a: Vec, b: Vec): Vec => a.map((x, i) => x + b[i]);
export const sub = (a: Vec, b: Vec): Vec => a.map((x, i) => x - b[i]);
export const scale = (a: Vec, k: number): Vec => a.map((x) => x * k);
export const dot = (a: Vec, b: Vec): number => a.reduce((s, x, i) => s + x * b[i], 0);
export const norm2 = (a: Vec): number => Math.hypot(...a);
export const norm1 = (a: Vec): number => a.reduce((s, x) => s + Math.abs(x), 0);
export const normInf = (a: Vec): number => a.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
export const normP = (a: Vec, p: number): number => {
  if (p === Infinity) return normInf(a);
  return Math.pow(a.reduce((s, x) => s + Math.pow(Math.abs(x), p), 0), 1 / p);
};

export function unit(a: Vec): Vec | null {
  const n = norm2(a);
  return n < EPS ? null : scale(a, 1 / n);
}

export function cross(a: Vec, b: Vec): Vec {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Angle in radians between two vectors; null if either is the zero vector. */
export function angleBetween(a: Vec, b: Vec): number | null {
  const na = norm2(a);
  const nb = norm2(b);
  if (na < EPS || nb < EPS) return null;
  const c = Math.min(1, Math.max(-1, dot(a, b) / (na * nb)));
  return Math.acos(c);
}

/** Vector projection of a onto b; null when b is zero. */
export function project(a: Vec, b: Vec): Vec | null {
  const bb = dot(b, b);
  if (bb < EPS) return null;
  return scale(b, dot(a, b) / bb);
}

export const euclidean = (a: Vec, b: Vec) => norm2(sub(a, b));
export const manhattan = (a: Vec, b: Vec) => norm1(sub(a, b));
export const chebyshev = (a: Vec, b: Vec) => normInf(sub(a, b));
export const minkowski = (a: Vec, b: Vec, p: number) => normP(sub(a, b), p);

export function cosineSimilarity(a: Vec, b: Vec): number | null {
  const na = norm2(a);
  const nb = norm2(b);
  if (na < EPS || nb < EPS) return null;
  return dot(a, b) / (na * nb);
}

/* ----------------------------- matrices ----------------------------- */

export const shape = (A: Matrix): [number, number] => [A.length, A[0]?.length ?? 0];
export const zeros = (r: number, c: number): Matrix => Array.from({ length: r }, () => Array(c).fill(0));
export const identity = (n: number): Matrix =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
export const clone = (A: Matrix): Matrix => A.map((r) => r.slice());
export const transpose = (A: Matrix): Matrix => (A[0] ?? []).map((_, j) => A.map((r) => r[j]));

export function matAdd(A: Matrix, B: Matrix): Matrix {
  assertSameShape(A, B);
  return A.map((r, i) => r.map((x, j) => x + B[i][j]));
}
export function matSub(A: Matrix, B: Matrix): Matrix {
  assertSameShape(A, B);
  return A.map((r, i) => r.map((x, j) => x - B[i][j]));
}
export const matScale = (A: Matrix, k: number): Matrix => A.map((r) => r.map((x) => x * k));

export function hadamard(A: Matrix, B: Matrix): Matrix {
  assertSameShape(A, B);
  return A.map((r, i) => r.map((x, j) => x * B[i][j]));
}

function assertSameShape(A: Matrix, B: Matrix) {
  const [r1, c1] = shape(A);
  const [r2, c2] = shape(B);
  if (r1 !== r2 || c1 !== c2) throw new Error(`Shape mismatch: ${r1}×${c1} vs ${r2}×${c2}`);
}

export function matMul(A: Matrix, B: Matrix): Matrix {
  const [m, n] = shape(A);
  const [n2, p] = shape(B);
  if (n !== n2) throw new Error(`Cannot multiply ${m}×${n} by ${n2}×${p}`);
  const C = zeros(m, p);
  for (let i = 0; i < m; i++)
    for (let k = 0; k < n; k++) {
      const a = A[i][k];
      for (let j = 0; j < p; j++) C[i][j] += a * B[k][j];
    }
  return C;
}

export function matVec(A: Matrix, x: Vec): Vec {
  if ((A[0]?.length ?? 0) !== x.length) throw new Error("Dimension mismatch in A·x");
  return A.map((r) => dot(r, x));
}

export function matPow(A: Matrix, k: number): Matrix {
  if (!Number.isInteger(k) || k < 0) throw new Error("Power must be a non-negative integer");
  let R = identity(A.length);
  let B = clone(A);
  let e = k;
  while (e > 0) {
    if (e & 1) R = matMul(R, B);
    B = matMul(B, B);
    e >>= 1;
  }
  return R;
}

export const trace = (A: Matrix): number => A.reduce((s, r, i) => s + r[i], 0);

export function isSymmetric(A: Matrix, tol = 1e-9): boolean {
  const n = A.length;
  if (n !== A[0]?.length) return false;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Math.abs(A[i][j] - A[j][i]) > tol) return false;
  return true;
}

export function isOrthogonal(A: Matrix, tol = 1e-9): boolean {
  const n = A.length;
  if (n !== A[0]?.length) return false;
  const P = matMul(transpose(A), A);
  const I = identity(n);
  return P.every((r, i) => r.every((x, j) => Math.abs(x - I[i][j]) < tol));
}

export function sparsity(A: Matrix): number {
  const total = A.length * (A[0]?.length ?? 0);
  if (!total) return 0;
  const zerosCount = A.flat().filter((x) => x === 0).length;
  return zerosCount / total;
}

/* ------------------------ determinants & inverse ------------------------ */

export const det2 = (A: Matrix): number => A[0][0] * A[1][1] - A[0][1] * A[1][0];

export function det3(A: Matrix): number {
  const [a, b, c] = A[0];
  const [d, e, f] = A[1];
  const [g, h, i] = A[2];
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/** Determinant via LU with partial pivoting (O(n³)). */
export function det(A: Matrix): number {
  const n = A.length;
  if (n !== A[0]?.length) throw new Error("Determinant requires a square matrix");
  if (n === 1) return A[0][0];
  if (n === 2) return det2(A);
  const M = clone(A);
  let d = 1;
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    if (Math.abs(M[p][k]) < EPS) return 0;
    if (p !== k) {
      [M[p], M[k]] = [M[k], M[p]];
      d = -d;
    }
    d *= M[k][k];
    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / M[k][k];
      for (let j = k; j < n; j++) M[i][j] -= f * M[k][j];
    }
  }
  return d;
}

export function minor(A: Matrix, r: number, c: number): Matrix {
  return A.filter((_, i) => i !== r).map((row) => row.filter((_, j) => j !== c));
}

export function cofactorMatrix(A: Matrix): Matrix {
  const n = A.length;
  if (n === 1) return [[1]];
  return A.map((row, i) => row.map((_, j) => ((i + j) % 2 === 0 ? 1 : -1) * det(minor(A, i, j))));
}

export const adjugate = (A: Matrix): Matrix => transpose(cofactorMatrix(A));

/** Inverse via Gauss–Jordan with partial pivoting. Returns null if singular. */
export function inverse(A: Matrix): Matrix | null {
  const n = A.length;
  if (n !== A[0]?.length) throw new Error("Inverse requires a square matrix");
  const M = A.map((r, i) => [...r, ...identity(n)[i]]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    if (Math.abs(M[p][k]) < EPS) return null;
    [M[p], M[k]] = [M[k], M[p]];
    const pivot = M[k][k];
    for (let j = 0; j < 2 * n; j++) M[k][j] /= pivot;
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const f = M[i][k];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[i][j] -= f * M[k][j];
    }
  }
  return M.map((r) => r.slice(n));
}

/** Solve A x = b by Gaussian elimination with partial pivoting (no explicit inverse). */
export function solve(A: Matrix, b: Vec): Vec | null {
  const n = A.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    if (Math.abs(M[p][k]) < EPS) return null;
    [M[p], M[k]] = [M[k], M[p]];
    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / M[k][k];
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
    }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

/* ----------------------- row reduction (step-by-step) ----------------------- */

export interface RowOpStep {
  description: string;
  tex: string;
  matrix: Matrix;
  pivot?: [number, number];
}

const fmt = (x: number) => {
  const r = Math.round(x * 1000) / 1000;
  return Object.is(r, -0) ? "0" : String(r);
};

/**
 * Reduce to reduced row echelon form, recording every elementary row operation.
 * `augmentedCols` marks how many trailing columns are the right-hand side (they are
 * never used as pivot columns).
 */
export function rrefSteps(A: Matrix, augmentedCols = 0): { steps: RowOpStep[]; result: Matrix; pivots: number[] } {
  const M = clone(A);
  const rows = M.length;
  const cols = M[0].length;
  const steps: RowOpStep[] = [{ description: "Start", tex: "\\text{start}", matrix: clone(M) }];
  const pivots: number[] = [];
  let r = 0;
  for (let c = 0; c < cols - augmentedCols && r < rows; c++) {
    let p = r;
    for (let i = r + 1; i < rows; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
    if (Math.abs(M[p][c]) < EPS) continue;
    if (p !== r) {
      [M[p], M[r]] = [M[r], M[p]];
      steps.push({ description: `Swap R${r + 1} ↔ R${p + 1} (partial pivoting)`, tex: `R_{${r + 1}} \\leftrightarrow R_{${p + 1}}`, matrix: clone(M), pivot: [r, c] });
    }
    const pv = M[r][c];
    if (Math.abs(pv - 1) > EPS) {
      for (let j = 0; j < cols; j++) M[r][j] /= pv;
      steps.push({ description: `Scale R${r + 1} by 1/${fmt(pv)} so the pivot is 1`, tex: `R_{${r + 1}} \\leftarrow \\tfrac{1}{${fmt(pv)}} R_{${r + 1}}`, matrix: clone(M), pivot: [r, c] });
    }
    for (let i = 0; i < rows; i++) {
      if (i === r) continue;
      const f = M[i][c];
      if (Math.abs(f) < EPS) continue;
      for (let j = 0; j < cols; j++) M[i][j] -= f * M[r][j];
      steps.push({
        description: `Eliminate column ${c + 1} in R${i + 1}: R${i + 1} − (${fmt(f)})·R${r + 1}`,
        tex: `R_{${i + 1}} \\leftarrow R_{${i + 1}} - (${fmt(f)})\\,R_{${r + 1}}`,
        matrix: clone(M),
        pivot: [r, c],
      });
    }
    pivots.push(c);
    r++;
  }
  // clean tiny values
  for (const row of M) for (let j = 0; j < cols; j++) if (Math.abs(row[j]) < 1e-12) row[j] = 0;
  return { steps, result: M, pivots };
}

export function rank(A: Matrix): number {
  return rrefSteps(A).pivots.length;
}

/** Basis for the null space of A (from RREF free variables). */
export function nullSpace(A: Matrix): Vec[] {
  const { result, pivots } = rrefSteps(A);
  const n = A[0].length;
  const free = Array.from({ length: n }, (_, j) => j).filter((j) => !pivots.includes(j));
  return free.map((f) => {
    const v = Array(n).fill(0);
    v[f] = 1;
    pivots.forEach((pc, row) => {
      v[pc] = -result[row][f];
    });
    return v;
  });
}

/** Doolittle LU with partial pivoting: PA = LU. */
export function lu(A: Matrix): { L: Matrix; U: Matrix; P: Matrix } | null {
  const n = A.length;
  const U = clone(A);
  const L = identity(n);
  const P = identity(n);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(U[i][k]) > Math.abs(U[p][k])) p = i;
    if (Math.abs(U[p][k]) < EPS) return null;
    if (p !== k) {
      [U[p], U[k]] = [U[k], U[p]];
      [P[p], P[k]] = [P[k], P[p]];
      for (let j = 0; j < k; j++) [L[p][j], L[k][j]] = [L[k][j], L[p][j]];
    }
    for (let i = k + 1; i < n; i++) {
      const f = U[i][k] / U[k][k];
      L[i][k] = f;
      for (let j = k; j < n; j++) U[i][j] -= f * U[k][j];
    }
  }
  return { L, U, P };
}

/* ----------------------------- eigen ----------------------------- */

export interface Eigen2 {
  /** real eigenvalues if discriminant ≥ 0 */
  real: boolean;
  values: [number, number];
  /** imaginary part (±) when complex */
  imag: number;
  vectors: [Vec, Vec] | null;
}

/** Closed-form eigen-decomposition of a 2×2 matrix via the characteristic polynomial. */
export function eigen2(A: Matrix): Eigen2 {
  const [[a, b], [c, d]] = A;
  const tr = a + d;
  const dt = a * d - b * c;
  const disc = tr * tr - 4 * dt;
  if (disc < -1e-12) {
    return { real: false, values: [tr / 2, tr / 2], imag: Math.sqrt(-disc) / 2, vectors: null };
  }
  const s = Math.sqrt(Math.max(0, disc));
  const l1 = (tr + s) / 2;
  const l2 = (tr - s) / 2;
  const vecFor = (l: number): Vec => {
    // (A - λI)v = 0 → choose the better-conditioned row
    if (Math.abs(b) > 1e-12 || Math.abs(l - a) > 1e-12) {
      const v = Math.abs(b) >= Math.abs(c) ? [b, l - a] : [l - d, c];
      const u = unit(v);
      if (u) return u;
    }
    if (Math.abs(c) > 1e-12) return unit([l - d, c]) ?? [1, 0];
    // diagonal matrix
    return Math.abs(a - l) < Math.abs(d - l) ? [1, 0] : [0, 1];
  };
  let v1 = vecFor(l1);
  let v2 = vecFor(l2);
  if (Math.abs(l1 - l2) < 1e-12) {
    // repeated eigenvalue: A - λI may be zero (every vector is an eigenvector)
    const zero = Math.abs(a - l1) < 1e-12 && Math.abs(b) < 1e-12 && Math.abs(c) < 1e-12 && Math.abs(d - l1) < 1e-12;
    if (zero) {
      v1 = [1, 0];
      v2 = [0, 1];
    } else v2 = v1; // defective
  }
  return { real: true, values: [l1, l2], imag: 0, vectors: [v1, v2] };
}

/** Jacobi eigenvalue algorithm for symmetric matrices. Returns eigenpairs sorted by eigenvalue desc. */
export function eigSym(S: Matrix, maxIter = 100): { values: number[]; vectors: Matrix } {
  const n = S.length;
  const A = clone(S);
  let V = identity(n);
  for (let iter = 0; iter < maxIter; iter++) {
    let off = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (Math.abs(A[i][j]) > off) {
          off = Math.abs(A[i][j]);
          p = i;
          q = j;
        }
    if (off < 1e-12) break;
    const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
    const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    const J = identity(n);
    J[p][p] = c;
    J[q][q] = c;
    J[p][q] = s;
    J[q][p] = -s;
    const Jt = transpose(J);
    const B = matMul(matMul(Jt, A), J);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i][j] = B[i][j];
    V = matMul(V, J);
  }
  const pairs = A.map((_, i) => ({ value: A[i][i], vec: V.map((r) => r[i]) }));
  pairs.sort((x, y) => y.value - x.value);
  // sign convention: largest-magnitude component positive
  for (const pr of pairs) {
    const k = pr.vec.reduce((bi, x, i, arr) => (Math.abs(x) > Math.abs(arr[bi]) ? i : bi), 0);
    if (pr.vec[k] < 0) pr.vec = pr.vec.map((x) => -x);
  }
  return { values: pairs.map((p) => p.value), vectors: transpose(pairs.map((p) => p.vec)) };
}

/** SVD for small matrices via eigen-decomposition of AᵀA. A = U Σ Vᵀ. */
export function svd(A: Matrix): { U: Matrix; S: number[]; V: Matrix } {
  const [m, n] = shape(A);
  const AtA = matMul(transpose(A), A);
  const { values, vectors: V } = eigSym(AtA, 200);
  const S = values.map((x) => Math.sqrt(Math.max(0, x)));
  const k = Math.min(m, n);
  const Ucols: Vec[] = [];
  for (let i = 0; i < k; i++) {
    const vi = V.map((r) => r[i]);
    if (S[i] > 1e-10) Ucols.push(scale(matVec(A, vi), 1 / S[i]));
    else {
      // complete with a vector orthogonal to previous columns (Gram–Schmidt on basis vectors)
      for (let e = 0; e < m; e++) {
        let cand: number[] = Array.from({ length: m }, (_, j) => (j === e ? 1 : 0));
        for (const u of Ucols) cand = sub(cand, scale(u, dot(cand, u)));
        const nu = unit(cand);
        if (nu && norm2(cand) > 1e-6) {
          Ucols.push(nu);
          break;
        }
      }
    }
  }
  return { U: transpose(Ucols), S: S.slice(0, k), V };
}

/** 2×2 SVD returning rotation angles, useful for animating Vᵀ → Σ → U. */
export function svd2(A: Matrix) {
  const { U, S, V } = svd(A);
  // make U and V proper rotations where possible (det +1) by flipping signs consistently
  const Uf = clone(U);
  const Vf = clone(V);
  if (det2(Vf) < 0) {
    Vf[0][1] *= -1;
    Vf[1][1] *= -1;
    Uf[0][1] *= -1;
    Uf[1][1] *= -1;
  }
  const s2 = S.slice();
  if (det2(Uf) < 0) {
    // det(A) < 0: absorb the reflection into Σ's second value
    Uf[0][1] *= -1;
    Uf[1][1] *= -1;
    s2[1] = -s2[1];
  }
  const angle = (R: Matrix) => Math.atan2(R[1][0], R[0][0]);
  return { U: Uf, S: s2, V: Vf, thetaU: angle(Uf), thetaV: angle(Vf) };
}

/** Thin QR via modified Gram–Schmidt. */
export function qr(A: Matrix): { Q: Matrix; R: Matrix } {
  const [m, n] = shape(A);
  const Q = zeros(m, n);
  const R = zeros(n, n);
  const V = transpose(A); // columns as rows
  for (let j = 0; j < n; j++) {
    let v = V[j].slice();
    for (let i = 0; i < j; i++) {
      const qi = Q.map((r) => r[i]);
      R[i][j] = dot(qi, v);
      v = sub(v, scale(qi, R[i][j]));
    }
    R[j][j] = norm2(v);
    const q = R[j][j] > EPS ? scale(v, 1 / R[j][j]) : v.map(() => 0);
    for (let i = 0; i < m; i++) Q[i][j] = q[i];
  }
  return { Q, R };
}

/** 2-norm condition number σ_max / σ_min. Infinity for singular matrices. */
export function conditionNumber(A: Matrix): number {
  const { S } = svd(A);
  const min = Math.min(...S);
  return min < 1e-14 ? Infinity : Math.max(...S) / min;
}

/** Moore–Penrose pseudoinverse via SVD. */
export function pinv(A: Matrix, tol = 1e-10): Matrix {
  // thin SVD: A = U_k Σ_k V_kᵀ  ⇒  A⁺ = V_k Σ_k⁺ U_kᵀ
  const { U, S, V } = svd(A);
  const k = S.length;
  const Vk = V.map((r) => r.slice(0, k));
  const D = zeros(k, k);
  S.forEach((s, i) => {
    if (s > tol) D[i][i] = 1 / s;
  });
  return matMul(matMul(Vk, D), transpose(U));
}

export function covarianceMatrix(X: Matrix, sample = true): Matrix {
  const n = X.length;
  const d = X[0].length;
  const mu = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const C = zeros(d, d);
  for (const r of X)
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += (r[i] - mu[i]) * (r[j] - mu[j]);
  const denom = sample ? n - 1 : n;
  return matScale(C, 1 / Math.max(1, denom));
}

export function gram(X: Matrix): Matrix {
  return matMul(X, transpose(X));
}

export function isPositiveDefinite(A: Matrix): boolean {
  if (!isSymmetric(A)) return false;
  return eigSym(A).values.every((v) => v > 1e-12);
}
