/** Descriptive statistics with explicit handling of degenerate inputs. */

export const sum = (x: number[]) => x.reduce((a, b) => a + b, 0);
export const mean = (x: number[]) => (x.length ? sum(x) / x.length : NaN);

/** Variance; sample (n−1) by default. Returns NaN when undefined. */
export function variance(x: number[], sample = true): number {
  const n = x.length;
  if (n === 0 || (sample && n < 2)) return NaN;
  const m = mean(x);
  // two-pass algorithm for numerical stability
  let s = 0;
  let c = 0;
  for (const v of x) {
    s += (v - m) ** 2;
    c += v - m;
  }
  return (s - (c * c) / n) / (sample ? n - 1 : n);
}

export const std = (x: number[], sample = true) => Math.sqrt(variance(x, sample));

export function quantileSorted(sorted: number[], q: number): number {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function quantile(x: number[], q: number): number {
  return quantileSorted(x.slice().sort((a, b) => a - b), q);
}

export const median = (x: number[]) => quantile(x, 0.5);

export function mode(x: number[]): number {
  const counts = new Map<number, number>();
  for (const v of x) counts.set(v, (counts.get(v) ?? 0) + 1);
  let best = NaN;
  let bc = -1;
  for (const [v, c] of counts) if (c > bc || (c === bc && v < best)) [best, bc] = [v, c];
  return best;
}

/** Adjusted Fisher–Pearson sample skewness (as in pandas / Excel SKEW). */
export function skewness(x: number[]): number {
  const n = x.length;
  if (n < 3) return NaN;
  const m = mean(x);
  const m2 = sum(x.map((v) => (v - m) ** 2)) / n;
  const m3 = sum(x.map((v) => (v - m) ** 3)) / n;
  if (m2 < 1e-300) return 0;
  const g1 = m3 / Math.pow(m2, 1.5);
  return (Math.sqrt(n * (n - 1)) / (n - 2)) * g1;
}

/** Excess kurtosis (population moment estimator g₂ = m₄/m₂² − 3). */
export function kurtosis(x: number[]): number {
  const n = x.length;
  if (n < 4) return NaN;
  const m = mean(x);
  const m2 = sum(x.map((v) => (v - m) ** 2)) / n;
  const m4 = sum(x.map((v) => (v - m) ** 4)) / n;
  if (m2 < 1e-300) return 0;
  return m4 / (m2 * m2) - 3;
}

export function covariance(x: number[], y: number[], sample = true): number {
  const n = x.length;
  if (n !== y.length || n < (sample ? 2 : 1)) return NaN;
  const mx = mean(x);
  const my = mean(y);
  let s = 0;
  for (let i = 0; i < n; i++) s += (x[i] - mx) * (y[i] - my);
  return s / (sample ? n - 1 : n);
}

export function pearson(x: number[], y: number[]): number {
  const sx = std(x);
  const sy = std(y);
  if (!(sx > 0) || !(sy > 0)) return NaN;
  return covariance(x, y) / (sx * sy);
}

/** Average ranks (ties receive the mean of their positions), 1-based. */
export function ranks(x: number[]): number[] {
  const idx = x.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const r = Array(x.length).fill(0);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
    i = j + 1;
  }
  return r;
}

export const spearman = (x: number[], y: number[]) => pearson(ranks(x), ranks(y));

export function zScores(x: number[]): number[] {
  const m = mean(x);
  const s = std(x);
  if (!(s > 0)) return x.map(() => 0);
  return x.map((v) => (v - m) / s);
}

/** Median absolute deviation. */
export function mad(x: number[]): number {
  const med = median(x);
  return median(x.map((v) => Math.abs(v - med)));
}

export function iqrBounds(x: number[], k = 1.5) {
  const s = x.slice().sort((a, b) => a - b);
  const q1 = quantileSorted(s, 0.25);
  const q3 = quantileSorted(s, 0.75);
  const iqr = q3 - q1;
  return { q1, q3, iqr, lower: q1 - k * iqr, upper: q3 + k * iqr, median: quantileSorted(s, 0.5), min: s[0], max: s[s.length - 1] };
}

/** Histogram with fixed bin edges. Returns counts and density-normalised heights. */
export function histogram(x: number[], lo: number, hi: number, bins: number) {
  const counts = Array(bins).fill(0);
  const w = (hi - lo) / bins;
  for (const v of x) {
    if (!Number.isFinite(v) || v < lo || v > hi) continue;
    const b = Math.min(bins - 1, Math.floor((v - lo) / w));
    counts[b]++;
  }
  const n = x.length || 1;
  return counts.map((c, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, count: c, density: c / (n * w) }));
}

export function extent(x: number[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of x) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}
