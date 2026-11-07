"use client";
import { createContext, useCallback, useContext, useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { scaleLinear, type ScaleLinear } from "d3-scale";
import { line as d3line, curveLinear } from "d3-shape";
import { useSize } from "@/hooks/useSize";

export type Pt = readonly number[];

/** CSS colour strings usable in SVG `style` props (theme-aware via CSS variables). */
export const C = {
  fg: "rgb(var(--fg))",
  muted: "rgb(var(--muted))",
  faint: "rgb(var(--faint))",
  line: "rgb(var(--line))",
  lineStrong: "rgb(var(--line-strong))",
  blue: "rgb(var(--blue))",
  pink: "rgb(var(--pink))",
  green: "rgb(var(--green))",
  amber: "rgb(var(--amber))",
  red: "rgb(var(--red))",
  grid: "var(--c-grid)",
  axis: "var(--c-axis)",
  panel: "rgb(var(--panel))",
  bg: "rgb(var(--bg))",
  a: (name: "blue" | "pink" | "green" | "amber" | "red" | "fg" | "muted", alpha: number) => `rgb(var(--${name}) / ${alpha})`,
};

/** Resolved RGB triples for canvas drawing; re-read when the theme attribute changes. */
export function useThemeRGB() {
  const read = () => {
    if (typeof window === "undefined") return null;
    const cs = getComputedStyle(document.documentElement);
    const g = (n: string) => cs.getPropertyValue(`--${n}`).trim().split(/\s+/).map(Number) as [number, number, number];
    return { blue: g("blue"), pink: g("pink"), green: g("green"), amber: g("amber"), red: g("red"), fg: g("fg"), bg: g("bg"), panel: g("panel"), muted: g("muted"), line: g("line") };
  };
  const [rgb, setRgb] = useState<ReturnType<typeof read>>(null);
  useEffect(() => {
    setRgb(read());
    const mo = new MutationObserver(() => setRgb(read()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => mo.disconnect();
  }, []);
  return rgb ?? { blue: [90, 169, 255], pink: [255, 111, 174], green: [110, 207, 151], amber: [230, 190, 100], red: [240, 104, 104], fg: [230, 232, 236], bg: [11, 12, 14], panel: [17, 18, 21], muted: [150, 155, 166], line: [38, 40, 46] };
}

export interface PlotCtx {
  sx: ScaleLinear<number, number>;
  sy: ScaleLinear<number, number>;
  width: number;
  height: number;
  inner: { left: number; top: number; right: number; bottom: number };
  /** convert client pointer coordinates to data coordinates */
  toData: (clientX: number, clientY: number) => [number, number];
  svg: SVGSVGElement | null;
}

const Ctx = createContext<PlotCtx | null>(null);
export const usePlot = () => {
  const c = useContext(Ctx);
  if (!c) throw new Error("usePlot must be used inside <Plot>");
  return c;
};

export interface PlotProps {
  x: [number, number];
  y: [number, number];
  /** keep 1:1 data aspect ratio (expands the shorter domain) */
  equal?: boolean;
  aspect?: number;
  minH?: number;
  maxH?: number;
  margin?: Partial<{ l: number; r: number; t: number; b: number }>;
  grid?: boolean;
  axes?: boolean;
  /** draw x=0 / y=0 axis lines through the origin rather than at the edges */
  originAxes?: boolean;
  xLabel?: string;
  yLabel?: string;
  ticks?: number;
  children?: ReactNode;
  onPointerDown?: (p: [number, number], e: React.PointerEvent) => void;
  ariaLabel: string;
  className?: string;
  /** absolutely positioned layer under the SVG (e.g. canvas heatmap) */
  underlay?: (ctx: PlotCtx) => ReactNode;
}

export function Plot({
  x, y, equal, aspect = 0.62, minH, maxH, margin, grid = true, axes = true, originAxes = false, xLabel, yLabel, ticks = 6,
  children, onPointerDown, ariaLabel, className, underlay,
}: PlotProps) {
  const { ref, width, height } = useSize<HTMLDivElement>(aspect, minH, maxH);
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = "clip" + useId().replace(/[^a-zA-Z0-9]/g, "");
  const m = { l: margin?.l ?? 40, r: margin?.r ?? 14, t: margin?.t ?? 12, b: margin?.b ?? 30 };
  const iw = Math.max(10, width - m.l - m.r);
  const ih = Math.max(10, height - m.t - m.b);

  let xd = x;
  let yd = y;
  if (equal) {
    const xs = (x[1] - x[0]) / iw;
    const ys = (y[1] - y[0]) / ih;
    if (xs > ys) {
      const c = (y[0] + y[1]) / 2;
      const half = (xs * ih) / 2;
      yd = [c - half, c + half];
    } else {
      const c = (x[0] + x[1]) / 2;
      const half = (ys * iw) / 2;
      xd = [c - half, c + half];
    }
  }
  const sx = useMemo(() => scaleLinear().domain(xd).range([m.l, m.l + iw]), [xd[0], xd[1], m.l, iw]); // eslint-disable-line
  const sy = useMemo(() => scaleLinear().domain(yd).range([m.t + ih, m.t]), [yd[0], yd[1], m.t, ih]); // eslint-disable-line

  const toData = useCallback(
    (cx: number, cy: number): [number, number] => {
      const r = svgRef.current?.getBoundingClientRect();
      if (!r) return [0, 0];
      return [sx.invert(cx - r.left), sy.invert(cy - r.top)];
    },
    [sx, sy],
  );

  const ctx: PlotCtx = { sx, sy, width, height, inner: { left: m.l, top: m.t, right: m.l + iw, bottom: m.t + ih }, toData, svg: svgRef.current };
  const xt = sx.ticks(Math.max(2, Math.round((ticks * iw) / 560)));
  const yt = sy.ticks(Math.max(2, Math.round((ticks * ih) / 360)));
  const tf = (v: number) => {
    const s = Math.abs(v) >= 1000 ? v.toExponential(0) : String(Number(v.toFixed(3)));
    return s.replace("-", "−");
  };
  const ox = originAxes && xd[0] < 0 && xd[1] > 0 ? sx(0) : null;
  const oy = originAxes && yd[0] < 0 && yd[1] > 0 ? sy(0) : null;

  return (
    <div ref={ref} className={"relative w-full select-none " + (className ?? "")}>
      {underlay && <div className="pointer-events-none absolute inset-0">{underlay(ctx)}</div>}
      <svg
        ref={svgRef}
        width={width}
        height={height}
        role="img"
        aria-label={ariaLabel}
        className="relative block touch-none"
        onPointerDown={onPointerDown ? (e) => onPointerDown(toData(e.clientX, e.clientY), e) : undefined}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={m.l} y={m.t} width={iw} height={ih} />
          </clipPath>
          {(["blue", "pink", "green", "amber", "fg", "muted", "red"] as const).map((c) => (
            <marker key={c} id={`arrow-${c}`} viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: c === "fg" ? C.fg : c === "muted" ? C.muted : `rgb(var(--${c}))` }} />
            </marker>
          ))}
        </defs>
        {grid && (
          <g>
            {xt.map((t) => <line key={`gx${t}`} x1={sx(t)} x2={sx(t)} y1={m.t} y2={m.t + ih} style={{ stroke: C.grid }} />)}
            {yt.map((t) => <line key={`gy${t}`} y1={sy(t)} y2={sy(t)} x1={m.l} x2={m.l + iw} style={{ stroke: C.grid }} />)}
          </g>
        )}
        {axes && (
          <g className="text-[10.5px]" style={{ fill: C.faint }}>
            <line x1={m.l} x2={m.l + iw} y1={oy ?? m.t + ih} y2={oy ?? m.t + ih} style={{ stroke: C.axis }} />
            <line y1={m.t} y2={m.t + ih} x1={ox ?? m.l} x2={ox ?? m.l} style={{ stroke: C.axis }} />
            {xt.map((t) => (
              <text key={`tx${t}`} x={sx(t)} y={m.t + ih + 16} textAnchor="middle" className="num">{tf(t)}</text>
            ))}
            {yt.map((t) => (
              <text key={`ty${t}`} x={m.l - 6} y={sy(t) + 3.5} textAnchor="end" className="num">{tf(t)}</text>
            ))}
            {xLabel && <text x={m.l + iw} y={m.t + ih - 6} textAnchor="end" style={{ fill: C.muted }}>{xLabel}</text>}
            {yLabel && <text x={m.l + 6} y={m.t + 12} style={{ fill: C.muted }}>{yLabel}</text>}
          </g>
        )}
        <Ctx.Provider value={ctx}>
          <g clipPath={`url(#${clipId})`}>{children}</g>
        </Ctx.Provider>
      </svg>
    </div>
  );
}

/** Polyline of a function sampled across the visible x-domain; breaks at non-finite values. */
export function FnPath({ f, n = 300, style, domain }: { f: (x: number) => number; n?: number; style?: React.CSSProperties; domain?: [number, number] }) {
  const { sx, sy } = usePlot();
  const [a, b] = domain ?? (sx.domain() as [number, number]);
  const [ylo, yhi] = sy.domain();
  const span = yhi - ylo;
  const pts: ([number, number] | null)[] = [];
  for (let i = 0; i <= n; i++) {
    const xv = a + ((b - a) * i) / n;
    const yv = f(xv);
    pts.push(Number.isFinite(yv) && yv > ylo - 5 * span && yv < yhi + 5 * span ? [xv, yv] : null);
  }
  const gen = d3line<[number, number] | null>()
    .defined((d) => d !== null)
    .x((d) => sx(d![0]))
    .y((d) => sy(d![1]))
    .curve(curveLinear);
  return <path d={gen(pts) ?? ""} fill="none" style={{ stroke: C.blue, strokeWidth: 2, ...style }} />;
}

export function Polyline({ pts, style, closed }: { pts: readonly Pt[]; style?: React.CSSProperties; closed?: boolean }) {
  const { sx, sy } = usePlot();
  if (!pts.length) return null;
  const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0])},${sy(p[1])}`).join("") + (closed ? "Z" : "");
  return <path d={d} fill="none" style={{ stroke: C.blue, strokeWidth: 1.5, ...style }} />;
}

export function Arrow({ from, to, color = "blue", width = 2.2, dashed, label, labelOffset = [6, -6] }: {
  from: Pt; to: Pt; color?: "blue" | "pink" | "green" | "amber" | "fg" | "muted" | "red";
  width?: number; dashed?: boolean; label?: string; labelOffset?: [number, number];
}) {
  const { sx, sy } = usePlot();
  const stroke = color === "fg" ? C.fg : color === "muted" ? C.muted : `rgb(var(--${color}))`;
  const len = Math.hypot(sx(to[0]) - sx(from[0]), sy(to[1]) - sy(from[1]));
  return (
    <g>
      <line
        x1={sx(from[0])} y1={sy(from[1])} x2={sx(to[0])} y2={sy(to[1])}
        markerEnd={len > 4 ? `url(#arrow-${color})` : undefined}
        style={{ stroke, strokeWidth: width, strokeDasharray: dashed ? "5 4" : undefined }}
      />
      {label && (
        <text x={sx(to[0]) + labelOffset[0]} y={sy(to[1]) + labelOffset[1]} className="text-[12px] font-medium" style={{ fill: stroke }}>
          {label}
        </text>
      )}
    </g>
  );
}

/** Draggable handle in data coordinates. */
export function Handle({ p, onDrag, color = "blue", r = 7, label }: { p: Pt; onDrag: (p: [number, number]) => void; color?: string; r?: number; label?: string }) {
  const { sx, sy, toData } = usePlot();
  const dragging = useRef(false);
  const fill = color.startsWith("rgb") || color.startsWith("#") ? color : `rgb(var(--${color}))`;
  return (
    <g
      tabIndex={0}
      role="slider"
      aria-label={label ?? "draggable point"}
      aria-valuetext={`(${p[0].toFixed(2)}, ${p[1].toFixed(2)})`}
      style={{ cursor: "grab" }}
      onPointerDown={(e) => {
        e.stopPropagation();
        dragging.current = true;
        (e.target as Element).setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (dragging.current) onDrag(toData(e.clientX, e.clientY));
      }}
      onPointerUp={() => (dragging.current = false)}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 0.5 : 0.1;
        const d: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
        if (d[e.key]) {
          e.preventDefault();
          onDrag([p[0] + d[e.key][0], p[1] + d[e.key][1]]);
        }
      }}
    >
      <circle cx={sx(p[0])} cy={sy(p[1])} r={r + 8} fill="transparent" />
      <circle cx={sx(p[0])} cy={sy(p[1])} r={r} style={{ fill, stroke: C.bg, strokeWidth: 2 }} />
    </g>
  );
}

export function Dot({ p, r = 3.5, color = C.fg, stroke, opacity = 1 }: { p: Pt; r?: number; color?: string; stroke?: string; opacity?: number }) {
  const { sx, sy } = usePlot();
  return <circle cx={sx(p[0])} cy={sy(p[1])} r={r} style={{ fill: color, stroke, opacity }} />;
}

export function Label({ p, children, color = C.muted, dx = 0, dy = 0, anchor = "start", size = 11.5 }: { p: Pt; children: ReactNode; color?: string; dx?: number; dy?: number; anchor?: "start" | "middle" | "end"; size?: number }) {
  const { sx, sy } = usePlot();
  return (
    <text x={sx(p[0]) + dx} y={sy(p[1]) + dy} textAnchor={anchor} style={{ fill: color, fontSize: size }}>
      {children}
    </text>
  );
}

/**
 * Canvas heatmap filling the plot's inner area. `f(x, y)` returns an [r,g,b,a] colour (0-255).
 * Rendered at reduced resolution and scaled up for speed.
 */
export function HeatmapCanvas({ ctx, f, res = 3, deps }: { ctx: PlotCtx; f: (x: number, y: number) => [number, number, number, number]; res?: number; deps: unknown[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { inner, sx, sy } = ctx;
  const w = Math.max(1, Math.ceil((inner.right - inner.left) / res));
  const h = Math.max(1, Math.ceil((inner.bottom - inner.top) / res));
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const g = c.getContext("2d");
    if (!g) return;
    const img = g.createImageData(w, h);
    for (let j = 0; j < h; j++) {
      const yv = sy.invert(inner.top + (j + 0.5) * res);
      for (let i = 0; i < w; i++) {
        const xv = sx.invert(inner.left + (i + 0.5) * res);
        const [r, gg, b, a] = f(xv, yv);
        const k = (j * w + i) * 4;
        img.data[k] = r;
        img.data[k + 1] = gg;
        img.data[k + 2] = b;
        img.data[k + 3] = a;
      }
    }
    g.putImageData(img, 0, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w, h, inner.left, inner.top, res, ...deps]);
  return (
    <canvas
      ref={ref}
      width={w}
      height={h}
      aria-hidden
      style={{ position: "absolute", left: inner.left, top: inner.top, width: inner.right - inner.left, height: inner.bottom - inner.top, imageRendering: "auto" }}
    />
  );
}
