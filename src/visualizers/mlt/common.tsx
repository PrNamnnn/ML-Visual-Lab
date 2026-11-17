"use client";
import type { CSSProperties, ReactNode } from "react";
import { Plot, Polyline, Dot, C, usePlot, HeatmapCanvas, useThemeRGB, type Pt, type PlotCtx } from "@/components/visualization/Plot";
import { mulberry32, randn, type RNG } from "@/lib/math/random";

export type V2 = [number, number];

/** The line { x : w·x + b = 0 } drawn across the visible plot area. */
export function LineWB({ w, b = 0, style }: { w: Pt; b?: number; style?: CSSProperties }) {
  const { sx, sy } = usePlot();
  const n2 = w[0] * w[0] + w[1] * w[1];
  if (n2 < 1e-12) return null;
  const [x0, x1] = sx.domain(), [y0, y1] = sy.domain();
  const L = Math.hypot(x1 - x0, y1 - y0) * 2;
  const p: V2 = [(-b * w[0]) / n2, (-b * w[1]) / n2];
  const n = Math.sqrt(n2), d: V2 = [-w[1] / n, w[0] / n];
  return <Polyline pts={[[p[0] - L * d[0], p[1] - L * d[1]], [p[0] + L * d[0], p[1] + L * d[1]]]} style={{ stroke: C.fg, strokeWidth: 1.8, ...style }} />;
}

/** Small line chart of one or more series against the step index (0, 1, 2, …). */
export function Trace({
  series, yDomain, xMax, marker, xLabel = "step", yLabel, ariaLabel, minH = 130, maxH = 190, aspect = 0.42, children,
}: {
  series: { pts: number[]; color: string; dashed?: boolean; dots?: boolean }[];
  yDomain?: [number, number]; xMax?: number; marker?: number; xLabel?: string; yLabel?: string; ariaLabel: string;
  minH?: number; maxH?: number; aspect?: number; children?: ReactNode;
}) {
  const all = series.flatMap((s) => s.pts).filter(Number.isFinite);
  const lo = yDomain?.[0] ?? Math.min(0, ...all), hi0 = yDomain?.[1] ?? Math.max(1e-9, ...all);
  const hi = hi0 > lo ? hi0 : lo + 1;
  const pad = yDomain ? 0 : (hi - lo) * 0.08;
  const len = Math.max(1, xMax ?? Math.max(...series.map((s) => s.pts.length - 1)));
  return (
    <Plot x={[0, len]} y={[lo - (lo < 0 ? pad : 0), hi + pad]} aspect={aspect} minH={minH} maxH={maxH} margin={{ l: 44, b: 24, t: 10 }} xLabel={xLabel} yLabel={yLabel} ariaLabel={ariaLabel}>
      {series.map((s, i) => (
        <g key={i}>
          <Polyline pts={s.pts.map((v, t) => [t, v] as V2).filter((p) => Number.isFinite(p[1]))} style={{ stroke: s.color, strokeWidth: 1.8, strokeDasharray: s.dashed ? "5 4" : undefined }} />
          {s.dots && s.pts.map((v, t) => (Number.isFinite(v) ? <Dot key={t} p={[t, v]} r={2.4} color={s.color} /> : null))}
        </g>
      ))}
      {marker !== undefined && <Polyline pts={[[marker, lo - 10 * (hi - lo)], [marker, hi + 10 * (hi - lo)]]} style={{ stroke: C.a("fg", 0.35), strokeWidth: 1, strokeDasharray: "3 3" }} />}
      {children}
    </Plot>
  );
}

/** n correlated Gaussian points with zero mean (rotation `angle`, standard deviations s1 ≥ s2 along / across it). */
export function gaussCloud(rng: RNG, n: number, angle: number, s1: number, s2: number, mean: V2 = [0, 0]): V2[] {
  const c = Math.cos(angle), s = Math.sin(angle);
  return Array.from({ length: n }, () => {
    const a = randn(rng) * s1, b = randn(rng) * s2;
    return [mean[0] + c * a - s * b, mean[1] + s * a + c * b] as V2;
  });
}

export const meanOf = (X: readonly Pt[]): V2 => [X.reduce((s, p) => s + p[0], 0) / X.length, X.reduce((s, p) => s + p[1], 0) / X.length];

/** C = (1/n) Σ xᵢxᵢᵀ for 2-D points (no centring is applied: pass centred data for a covariance). */
export function secondMoment(X: readonly Pt[]): [[number, number], [number, number]] {
  const n = X.length || 1;
  let a = 0, b = 0, d = 0;
  for (const p of X) { a += p[0] * p[0]; b += p[0] * p[1]; d += p[1] * p[1]; }
  return [[a / n, b / n], [b / n, d / n]];
}

/** Eigen-decomposition of a symmetric 2 × 2 matrix, eigenvalues descending, unit eigenvectors. */
export function eigSym2(M: readonly (readonly number[])[]): { values: V2; vectors: [V2, V2] } {
  const a = M[0][0], b = M[0][1], d = M[1][1];
  const tr = a + d, disc = Math.sqrt(Math.max(0, ((a - d) * (a - d)) / 4 + b * b));
  const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
  let v1: V2;
  if (Math.abs(b) > 1e-12) v1 = [l1 - d, b];
  else v1 = a >= d ? [1, 0] : [0, 1];
  const n = Math.hypot(v1[0], v1[1]) || 1;
  v1 = [v1[0] / n, v1[1] / n];
  if (v1[0] < 0 || (v1[0] === 0 && v1[1] < 0)) v1 = [-v1[0], -v1[1]];
  return { values: [l1, l2], vectors: [v1, [-v1[1], v1[0]]] };
}

export const seeded = (seed: number) => mulberry32(seed);

/** Two-class data that is linearly separable through the origin with roughly the requested margin. Labels are ±1. */
export function separableData(seed: number, nPer: number, margin: number, angle = 0.6): { x: V2; y: 1 | -1 }[] {
  const rng = mulberry32(seed);
  const w: V2 = [Math.cos(angle), Math.sin(angle)], t: V2 = [-w[1], w[0]];
  const out: { x: V2; y: 1 | -1 }[] = [];
  for (const y of [1, -1] as const) {
    for (let i = 0; i < nPer; i++) {
      const along = (rng() * 2 - 1) * 2.1;
      const off = margin + rng() * 1.5;
      out.push({ x: [y * off * w[0] + along * t[0], y * off * w[1] + along * t[1]], y });
    }
  }
  return out;
}

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
export const log2 = (x: number) => Math.log(x) / Math.LN2;
/** Binary entropy in bits. */
export const H2 = (p: number) => (p <= 0 || p >= 1 ? 0 : -(p * log2(p) + (1 - p) * log2(1 - p)));

/**
 * Shaded scalar field with iso-lines at integer values of `level(f(x, y))`.
 * Lines are drawn at a constant on-screen width (the level function is differenced between
 * neighbouring samples), and fade out where levels would be closer than a sample apart, so
 * steep regions shade smoothly instead of producing moire patterns.
 */
export function ContourField({ ctx, f, level, deps, signed = false }: {
  ctx: PlotCtx; f: (x: number, y: number) => number; level: (v: number) => number; deps: unknown[]; signed?: boolean;
}) {
  const th = useThemeRGB();
  const res = 2;
  const dx = Math.abs(ctx.sx.invert(res) - ctx.sx.invert(0)), dy = Math.abs(ctx.sy.invert(res) - ctx.sy.invert(0));
  return (
    <HeatmapCanvas ctx={ctx} res={res} deps={[...deps, dx, dy, signed, th.blue[0], th.panel[0]]} f={(x, y) => {
      const v = f(x, y), t = level(v);
      const g = Math.hypot(level(f(x + dx, y)) - t, level(f(x, y + dy)) - t); // change of level per sample
      const d = Math.abs(t - Math.round(t));
      const line = g < 0.45 ? Math.max(0, 1 - d / Math.max(1e-9, 0.8 * g)) : 0;
      const c = signed && v < 0 ? th.pink : th.blue, p = th.panel;
      const k = Math.min(0.4, 0.05 + 0.028 * Math.abs(t)) + 0.26 * line;
      return [p[0] + (c[0] - p[0]) * k, p[1] + (c[1] - p[1]) * k, p[2] + (c[2] - p[2]) * k, 255];
    }} />
  );
}
