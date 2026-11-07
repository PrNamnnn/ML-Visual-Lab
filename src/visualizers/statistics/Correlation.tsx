"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, C, Polyline } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle } from "@/components/controls";
import { mulberry32, randBivariate, randn } from "@/lib/math/random";
import { pearson, spearman, covariance, mean, std } from "@/lib/statistics/descriptive";
import { fmt, tex } from "@/lib/utilities/format";

type Shape = "linear" | "monotone" | "quadratic" | "simpson";

function fitLine(xs: number[], ys: number[]) {
  const mx = mean(xs), my = mean(ys);
  const c = covariance(xs, ys), v = std(xs) ** 2;
  const w = v > 0 ? c / v : 0;
  return { w, b: my - w * mx };
}

export default function Correlation({ shape: initial = "linear" }: { shape?: Shape }) {
  const [shape, setShape] = useState<Shape>(initial);
  const [rho, setRho] = useState(0.7);
  const [n, setN] = useState(80);
  const [outlier, setOutlier] = useState(false);
  const [byGroup, setByGroup] = useState(true);
  const [seed, setSeed] = useState(4);
  const pts = useMemo(() => {
    const rng = mulberry32(seed);
    const out: { x: number; y: number; g: number }[] = [];
    if (shape === "linear") for (let i = 0; i < n; i++) { const [x, y] = randBivariate(rng, 0, 0, 1, 1, rho); out.push({ x, y, g: 0 }); }
    if (shape === "monotone") for (let i = 0; i < n; i++) { const x = rng() * 4 - 2; out.push({ x, y: Math.exp(1.2 * x) / 3 - 1.5 + randn(rng) * 0.15 * (1 - Math.abs(rho)) * 3, g: 0 }); }
    if (shape === "quadratic") for (let i = 0; i < n; i++) { const x = rng() * 4 - 2; out.push({ x, y: x * x - 1.3 + randn(rng) * 0.3, g: 0 }); }
    if (shape === "simpson") for (let g = 0; g < 3; g++) for (let i = 0; i < Math.round(n / 3); i++) { const x = rng() * 1.2 + g * 1.1 - 1.8; out.push({ x, y: -0.9 * (x - (g * 1.1 - 1.2)) + g * 1.3 - 1.6 + randn(rng) * 0.25, g }); }
    if (outlier) out.push({ x: 2.6, y: shape === "linear" && rho > 0 ? -2.6 : 2.8, g: 0 });
    return out;
  }, [shape, rho, n, outlier, seed]);
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const r = pearson(xs, ys), rs = spearman(xs, ys);
  const L = fitLine(xs, ys);
  const groups = shape === "simpson" ? [0, 1, 2].map((g) => { const s = pts.filter((p) => p.g === g); return { g, r: pearson(s.map((p) => p.x), s.map((p) => p.y)), ...fitLine(s.map((p) => p.x), s.map((p) => p.y)), x0: Math.min(...s.map((p) => p.x)), x1: Math.max(...s.map((p) => p.x)) }; }) : [];
  const gc = [C.blue, C.pink, C.amber];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-3, 3]} equal aspect={0.75} maxH={460} originAxes ariaLabel="scatter plot for correlation">
            {pts.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={i === pts.length - 1 && outlier ? 6 : 3.2} color={shape === "simpson" && byGroup ? gc[p.g] : i === pts.length - 1 && outlier ? C.red : C.a("blue", 0.85)} />)}
            <Polyline pts={[[-4, L.w * -4 + L.b], [4, L.w * 4 + L.b]]} style={{ stroke: C.fg, strokeWidth: 1.6, strokeDasharray: shape === "simpson" ? "5 4" : undefined }} />
            {shape === "simpson" && byGroup && groups.map((g) => <Polyline key={g.g} pts={[[g.x0, g.w * g.x0 + g.b], [g.x1, g.w * g.x1 + g.b]]} style={{ stroke: gc[g.g], strokeWidth: 2.2 }} />)}
          </Plot>
          <Legend items={[{ label: shape === "simpson" ? "pooled trend (ignoring group)" : "least-squares line", color: C.fg, dashed: shape === "simpson" }, ...(shape === "simpson" && byGroup ? [{ label: "within-group trends", color: C.pink }] : []), ...(outlier ? [{ label: "outlier", color: C.red }] : [])]} />
        </div>
      }
      controls={
        <>
          <Select label="Relationship" value={shape} onChange={setShape} options={[{ value: "linear", label: "Linear (bivariate normal)" }, { value: "monotone", label: "Monotone non-linear" }, { value: "quadratic", label: "U-shaped (non-monotone)" }, { value: "simpson", label: "Simpson's paradox" }]} />
          {(shape === "linear" || shape === "monotone") && <ParameterSlider label={shape === "linear" ? "population ρ" : "signal strength"} value={rho} min={shape === "linear" ? -0.99 : 0} max={0.99} step={0.01} onChange={setRho} />}
          <ParameterSlider label="n" value={n} min={10} max={300} step={5} onChange={setN} />
          {shape !== "simpson" && <Toggle label="Add an outlier" checked={outlier} onChange={setOutlier} />}
          {shape === "simpson" && <Toggle label="Colour by group (confounder)" checked={byGroup} onChange={setByGroup} />}
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New sample</button>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{Pearson }r", tex: true, value: fmt(r), color: C.blue },
        { label: "\\text{Spearman }\\rho_s", tex: true, value: fmt(rs), color: C.pink },
        { label: "\\operatorname{Cov}(X,Y)", tex: true, value: fmt(covariance(xs, ys)) },
        { label: "r^2", tex: true, value: fmt(r * r) },
        ...groups.map((g) => ({ label: `r_{\\text{group }${g.g + 1}}`, tex: true, value: fmt(g.r), color: gc[g.g] })),
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `r=\\dfrac{\\operatorname{Cov}(X,Y)}{s_Xs_Y}=\\dfrac{${tex(covariance(xs, ys))}}{${tex(std(xs))}\\cdot${tex(std(ys))}}=${tex(r)}` },
        { tex: `\\rho_s=r\\big(\\operatorname{rank}(X),\\operatorname{rank}(Y)\\big)=${tex(rs)}`, note: "Pearson on ranks" },
      ]} />}
      note={shape === "quadratic" ? "Correlation measures linear (Pearson) or monotone (Spearman) association only: a perfect U-shaped dependence can have r ≈ 0." : shape === "simpson" ? "Within every group the trend is negative, but pooled data shows a positive trend because group membership (a confounder) shifts both x and y. Correlation is not causation." : "Spearman uses ranks, so it is robust to outliers and captures any monotone relationship."}
    />
  );
}
