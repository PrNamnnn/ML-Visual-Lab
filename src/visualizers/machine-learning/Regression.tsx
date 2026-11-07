"use client";
import { useMemo, useState } from "react";
import { Plot, Handle, C, Polyline, HeatmapCanvas, Dot, usePlot } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Toggle, Segmented, Presets } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { regressionMetrics } from "@/lib/ml/metrics";
import { fmt, tex } from "@/lib/utilities/format";

type P = [number, number];

function makePts(seed: number, n = 12, outlier = false): P[] {
  const rng = mulberry32(seed);
  const pts: P[] = Array.from({ length: n }, (_, i) => {
    const x = -3.5 + (7 * (i + rng() * 0.6)) / n;
    return [Math.round(x * 10) / 10, Math.round((0.7 * x + 0.5 + randn(rng) * 0.7) * 10) / 10];
  });
  if (outlier) pts.push([2.5, -3]);
  return pts;
}

function ols(pts: P[]) {
  const n = pts.length;
  const mx = pts.reduce((s, p) => s + p[0], 0) / n;
  const my = pts.reduce((s, p) => s + p[1], 0) / n;
  let sxy = 0, sxx = 0;
  for (const [x, y] of pts) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; }
  const w = sxx > 1e-12 ? sxy / sxx : 0;
  return { w, b: my - w * mx, mx, my, sxy, sxx };
}

function Squares({ pts, w, b }: { pts: P[]; w: number; b: number }) {
  const { sx, sy } = usePlot();
  return (
    <g>
      {pts.map(([x, y], i) => {
        const yh = w * x + b;
        const side = Math.abs(sy(y) - sy(yh));
        const top = Math.min(sy(y), sy(yh));
        return <rect key={i} x={sx(x)} y={top} width={side} height={side} style={{ fill: C.a("pink", 0.13), stroke: C.a("pink", 0.5), strokeWidth: 0.8 }} />;
      })}
    </g>
  );
}

export default function Regression({ mode: initial = "residuals" }: { mode?: "residuals" | "fit" | "loss" }) {
  const [pts, setPts] = useState<P[]>(() => makePts(4));
  const [manual, setManual] = useState(initial === "fit" || initial === "loss");
  const [w, setW] = useState(0.2);
  const [b, setB] = useState(1.5);
  const [view, setView] = useState<"residuals" | "loss">(initial === "loss" ? "loss" : "residuals");
  const [squares, setSquares] = useState(initial === "residuals");
  const fit = useMemo(() => ols(pts), [pts]);
  const W = manual ? w : fit.w;
  const B = manual ? b : fit.b;
  const yhat = pts.map(([x]) => W * x + B);
  const m = regressionMetrics(pts.map((p) => p[1]), yhat, 1);
  const mOls = regressionMetrics(pts.map((p) => p[1]), pts.map(([x]) => fit.w * x + fit.b), 1);
  const mse = (ww: number, bb: number) => pts.reduce((s, [x, y]) => s + (y - ww * x - bb) ** 2, 0) / pts.length;
  const lossMax = Math.max(mse(-2, -4), mse(3, 5), mse(-2, 5), mse(3, -4));

  return (
    <LabShell
      wideControls
      viz={
        <div className="grid gap-3 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
          <div className="space-y-2">
            <Plot x={[-4.5, 4.5]} y={[-4.5, 4.5]} aspect={0.75} maxH={460} ariaLabel="scatter plot with regression line and residuals"
              onPointerDown={(p, e) => { if (e.shiftKey || e.altKey) return; if (pts.length < 40) setPts([...pts, [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]]); }}>
              {squares && <Squares pts={pts} w={W} b={B} />}
              {pts.map(([x, y], i) => <Polyline key={`res${i}`} pts={[[x, y], [x, W * x + B]]} style={{ stroke: C.pink, strokeWidth: 1.2 }} />)}
              <Polyline pts={[[-6, W * -6 + B], [6, W * 6 + B]]} style={{ stroke: C.blue, strokeWidth: 2.4 }} />
              {manual && <Polyline pts={[[-6, fit.w * -6 + fit.b], [6, fit.w * 6 + fit.b]]} style={{ stroke: C.green, strokeWidth: 1.2, strokeDasharray: "5 4" }} />}
              {pts.map((p, i) => (
                <g key={i} onDoubleClick={() => setPts(pts.filter((_, j) => j !== i))}>
                  <Handle p={p} r={5} color="fg" label={`data point ${i + 1}`} onDrag={(q) => setPts(pts.map((o, j) => (j === i ? [Math.round(q[0] * 10) / 10, Math.round(q[1] * 10) / 10] : o)))} />
                </g>
              ))}
            </Plot>
            <Legend items={[{ label: manual ? "your line" : "least-squares line", color: C.blue }, ...(manual ? [{ label: "least-squares optimum", color: C.green, dashed: true }] : []), { label: "residuals yᵢ − ŷᵢ", color: C.pink }]} />
          </div>
          {view === "loss" && (
            <div className="space-y-2">
              <Plot x={[-2, 3]} y={[-4, 5]} aspect={0.95} maxH={420} xLabel="w (slope)" yLabel="b (intercept)" ariaLabel="MSE loss surface over slope and intercept"
                onPointerDown={(p) => { setManual(true); setW(Math.round(p[0] * 100) / 100); setB(Math.round(p[1] * 100) / 100); }}
                underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={4} deps={[pts]} f={(ww, bb) => { const v = Math.log1p(mse(ww, bb)) / Math.log1p(lossMax); const k = 1 - Math.min(1, v); return [40 + 50 * k, 70 + 100 * k, 110 + 140 * k, 230]; }} />}>
                <Dot p={[fit.w, fit.b]} r={5} color={C.green} />
                <Dot p={[W, B]} r={5} color={C.pink} stroke={C.bg} />
              </Plot>
              <p className="text-[12px] text-faint">MSE over (w, b): brighter = lower loss. It is a convex bowl with a single minimum (green). Click to move your line&apos;s parameters.</p>
            </div>
          )}
        </div>
      }
      controls={
        <>
          <Segmented value={view} onChange={setView} options={[{ value: "residuals", label: "Data view" }, { value: "loss", label: "+ Loss surface" }]} />
          <Toggle label="Set w and b manually" checked={manual} onChange={setManual} />
          {manual && (
            <>
              <ParameterSlider label="slope w" value={w} min={-2} max={3} step={0.01} onChange={setW} />
              <ParameterSlider label="intercept b" value={b} min={-4} max={5} step={0.01} onChange={setB} />
            </>
          )}
          <Toggle label="Show squared residuals" checked={squares} onChange={setSquares} />
          <Presets presets={[{ label: "Linear", value: 4 }, { label: "Noisy", value: 9 }, { label: "With outlier", value: -1 }]} onPick={(s) => setPts(s === -1 ? makePts(4, 12, true) : makePts(s, s === 9 ? 18 : 12))} />
          <p className="text-[12px] leading-relaxed text-faint">Click empty space to add a point, drag points to move them, double-click a point to delete it.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\hat y=wx+b", tex: true, value: `${fmt(W)}x ${B >= 0 ? "+" : "−"} ${fmt(Math.abs(B))}` },
        { label: "\\text{MSE}", tex: true, value: fmt(m.mse), color: manual && m.mse > mOls.mse + 1e-9 ? C.amber : C.green },
        { label: "\\text{SSE}", tex: true, value: fmt(m.sse) },
        { label: "R^2", tex: true, value: fmt(m.r2) },
        { label: "\\text{OLS MSE}", tex: true, value: fmt(mOls.mse) },
        { label: "n", tex: true, value: pts.length },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `\\bar x=${tex(fit.mx)},\\ \\bar y=${tex(fit.my)}` },
        { tex: `w^*=\\dfrac{\\sum(x_i-\\bar x)(y_i-\\bar y)}{\\sum(x_i-\\bar x)^2}=\\dfrac{${tex(fit.sxy)}}{${tex(fit.sxx)}}=${tex(fit.w)}` },
        { tex: `b^*=\\bar y-w^*\\bar x=${tex(fit.b)}` },
        { tex: `\\text{MSE}=\\tfrac1n\\sum(y_i-\\hat y_i)^2=${tex(m.mse)}` },
      ]} />}
    />
  );
}
