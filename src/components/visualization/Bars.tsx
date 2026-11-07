"use client";
import { usePlot, C } from "./Plot";

/** Histogram bars in data coordinates. */
export function Bars({ bins, color = C.a("blue", 0.35), stroke = C.a("blue", 0.7), field = "density" }: { bins: { x0: number; x1: number; density: number; count: number }[]; color?: string; stroke?: string; field?: "density" | "count" }) {
  const { sx, sy } = usePlot();
  return (
    <g>
      {bins.map((b, i) => {
        const v = field === "density" ? b.density : b.count;
        if (!v) return null;
        return <rect key={i} x={sx(b.x0) + 0.5} width={Math.max(0, sx(b.x1) - sx(b.x0) - 1)} y={sy(v)} height={Math.max(0, sy(0) - sy(v))} style={{ fill: color, stroke, strokeWidth: 0.6 }} />;
      })}
    </g>
  );
}

/** Vertical stems (for PMFs). */
export function Stems({ pts, color = C.blue, width = 0.6 }: { pts: [number, number][]; color?: string; width?: number }) {
  const { sx, sy } = usePlot();
  const w = Math.max(2, Math.min(22, (sx(1) - sx(0)) * width));
  return (
    <g>
      {pts.map(([x, y], i) => (
        <rect key={i} x={sx(x) - w / 2} width={w} y={sy(Math.max(0, y))} height={Math.abs(sy(0) - sy(y))} style={{ fill: color }} rx={1} />
      ))}
    </g>
  );
}

/** Area under a function between a and b. */
export function AreaUnder({ f, a, b, color = C.a("blue", 0.25), n = 160 }: { f: (x: number) => number; a: number; b: number; color?: string; n?: number }) {
  const { sx, sy } = usePlot();
  if (!(b > a)) return null;
  let d = `M${sx(a)},${sy(0)}`;
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    const y = f(x);
    d += `L${sx(x)},${sy(Number.isFinite(y) ? y : 0)}`;
  }
  d += `L${sx(b)},${sy(0)}Z`;
  return <path d={d} style={{ fill: color }} />;
}

export function VLine({ x, color = C.muted, dashed = true, label }: { x: number; color?: string; dashed?: boolean; label?: string }) {
  const { sx, inner } = usePlot();
  return (
    <g>
      <line x1={sx(x)} x2={sx(x)} y1={inner.top} y2={inner.bottom} style={{ stroke: color, strokeDasharray: dashed ? "4 3" : undefined, strokeWidth: 1.2 }} />
      {label && <text x={sx(x) + 4} y={inner.top + 12} style={{ fill: color, fontSize: 11 }}>{label}</text>}
    </g>
  );
}
