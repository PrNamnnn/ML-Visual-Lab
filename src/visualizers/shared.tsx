"use client";
import type { ReactNode } from "react";
import { C } from "@/components/visualization/Plot";
import { fmt, cn } from "@/lib/utilities/format";

/** Categorical colours for class / cluster labels (theme tokens). */
export const CLS = [C.blue, C.pink, C.green, C.amber, C.red, C.muted];
export const clsColor = (i: number) => (i < 0 ? C.faint : CLS[i % CLS.length]);

/** Symmetric top-k eigenpairs by power iteration with deflation (fine for PSD matrices up to a few hundred rows). */
export function topEig(M: number[][], k: number, iters = 300): { values: number[]; vectors: number[][] } {
  const n = M.length;
  const A = M.map((r) => r.slice());
  const values: number[] = [];
  const vectors: number[][] = [];
  for (let c = 0; c < k; c++) {
    let v = Array.from({ length: n }, (_, i) => Math.sin(1.3 * i + c + 1) + 0.3);
    let lam = 0;
    for (let it = 0; it < iters; it++) {
      const w = A.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
      const nw = Math.hypot(...w) || 1;
      v = w.map((x) => x / nw);
      lam = nw;
    }
    // Rayleigh quotient gives the signed eigenvalue
    const Av = A.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
    lam = Av.reduce((s, x, i) => s + x * v[i], 0);
    values.push(lam);
    vectors.push(v);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i][j] -= lam * v[i] * v[j];
  }
  return { values, vectors };
}

/** Small matrix heatmap made of divs; diverging (blue/pink) when `signed`. */
export function Heat({ M, rowLabels, colLabels, signed = false, digits = 1, cell = 34, max, highlight }: {
  M: number[][]; rowLabels?: string[]; colLabels?: string[]; signed?: boolean; digits?: number; cell?: number; max?: number;
  highlight?: (i: number, j: number) => boolean;
}) {
  const m = max ?? Math.max(1e-9, ...M.flat().map(Math.abs));
  return (
    <div className="overflow-x-auto">
      <div className="inline-grid gap-[2px]" style={{ gridTemplateColumns: `${rowLabels ? "auto " : ""}repeat(${M[0]?.length ?? 0}, ${cell}px)` }}>
        {colLabels && <>{rowLabels && <span />}{colLabels.map((c, j) => <span key={j} className="truncate px-0.5 text-center text-[10.5px] text-faint" title={c}>{c}</span>)}</>}
        {M.map((row, i) => (
          <div key={i} className="contents">
            {rowLabels && <span className="flex items-center justify-end pr-1.5 text-[11px] text-faint">{rowLabels[i]}</span>}
            {row.map((v, j) => {
              const t = Math.min(1, Math.abs(v) / m);
              const bg = !Number.isFinite(v) ? "rgb(var(--raised))" : signed && v < 0 ? `rgb(var(--pink) / ${0.08 + 0.7 * t})` : `rgb(var(--blue) / ${0.08 + 0.7 * t})`;
              return (
                <div key={j} className={cn("num flex items-center justify-center rounded-[3px] text-[10.5px] text-fg", highlight?.(i, j) && "ring-2 ring-fg")} style={{ height: cell * 0.8, background: bg }}>
                  {Number.isFinite(v) ? fmt(v, digits) : "?"}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Compact data table. */
export function MiniTable({ head, rows, rowClass }: { head: ReactNode[]; rows: ReactNode[][]; rowClass?: (i: number) => string | undefined }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-line">
      <table className="w-full text-left text-[12px]">
        <thead className="bg-raised text-faint"><tr>{head.map((h, i) => <th key={i} className="whitespace-nowrap px-2.5 py-1.5 font-medium">{h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={cn("border-t border-line", rowClass?.(i))}>
              {r.map((c, j) => <td key={j} className="num whitespace-nowrap px-2.5 py-1.5 text-muted">{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Horizontal labelled bar list (value in [0, max]). */
export function BarList({ items, max, digits = 3 }: { items: { label: string; value: number; color?: string; note?: string }[]; max?: number; digits?: number }) {
  const m = max ?? Math.max(1e-9, ...items.map((i) => Math.abs(i.value)));
  return (
    <div className="space-y-1.5">
      {items.map((it) => (
        <div key={it.label} className="grid grid-cols-[minmax(90px,140px)_1fr_auto] items-center gap-2 text-[12px]">
          <span className="truncate text-muted" title={it.label}>{it.label}</span>
          <div className="h-3.5 overflow-hidden rounded-sm bg-raised">
            <div className="h-full rounded-sm transition-[width] duration-300" style={{ width: `${Math.min(100, (Math.abs(it.value) / m) * 100)}%`, background: it.color ?? C.blue }} />
          </div>
          <span className="num w-20 text-right text-fg">{fmt(it.value, digits)}{it.note ? ` ${it.note}` : ""}</span>
        </div>
      ))}
    </div>
  );
}

/** Squared Euclidean distance. */
export const sqd = (a: readonly number[], b: readonly number[]) => a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0);

/** Cyclic Jacobi eigendecomposition of a symmetric matrix (exact to ~1e-10; fine up to a few hundred rows). Ascending order. */
export function jacobiEig(S: number[][], sweeps = 12): { values: number[]; vectors: number[][] } {
  const n = S.length, A = S.map((r) => r.slice());
  const V: number[][] = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0)));
  for (let sw = 0; sw < sweeps; sw++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-14) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const order = A.map((r, i) => ({ v: r[i], i })).sort((a, b) => a.v - b.v);
  return { values: order.map((o) => o.v), vectors: order.map((o) => V.map((r) => r[o.i])) };
}
