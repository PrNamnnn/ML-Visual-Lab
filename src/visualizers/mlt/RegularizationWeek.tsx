"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, Polyline, FnPath, C, usePlot, type PlotCtx } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider } from "@/components/controls";
import { mulberry32, randn, shuffle } from "@/lib/math/random";
import { polyDesign, ridgeFit, predictLinear } from "@/lib/ml/models";
import { fmt, tex, texMatrix } from "@/lib/utilities/format";
import { MiniTable } from "@/visualizers/shared";
import { Trace, ContourField, eigSym2, type V2 } from "./common";

type Mode = "ridge" | "lasso" | "map";
type M2 = [[number, number], [number, number]];

const soft = (z: number, g: number) => (z > g ? z - g : z < -g ? z + g : 0);
const solve2 = (A: M2, b: V2, lam = 0): V2 => {
  const a = A[0][0] + lam, d = A[1][1] + lam, o = A[0][1], det = a * d - o * o;
  return [(d * b[0] - o * b[1]) / det, (a * b[1] - o * b[0]) / det];
};
/** argmin ½wᵀAw − bᵀw + λ‖w‖₁ by cyclic coordinate descent (exact in a few sweeps for two weights). */
function lasso2(A: M2, b: V2, lam: number): V2 {
  const w: V2 = [0, 0];
  for (let it = 0; it < 400; it++) {
    const w0 = soft(b[0] - A[0][1] * w[1], lam) / A[0][0];
    const w1 = soft(b[1] - A[0][1] * w0, lam) / A[1][1];
    const moved = Math.abs(w0 - w[0]) + Math.abs(w1 - w[1]);
    w[0] = w0; w[1] = w1;
    if (moved < 1e-12) break;
  }
  return w;
}

function Field({ ctx, f, scale, deps }: { ctx: PlotCtx; f: (a: number, b: number) => number; scale: number; deps: unknown[] }) {
  return <ContourField ctx={ctx} f={f} level={(v) => Math.sqrt(Math.max(0, v)) / scale} deps={[...deps, scale]} />;
}

function Ball({ r, kind }: { r: number; kind: "l1" | "l2" }) {
  const { sx, sy } = usePlot();
  if (!(r > 0)) return null;
  const style = { fill: C.a("amber", 0.1), stroke: C.amber, strokeWidth: 1.6 };
  if (kind === "l2") return <ellipse cx={sx(0)} cy={sy(0)} rx={Math.abs(sx(r) - sx(0))} ry={Math.abs(sy(r) - sy(0))} style={style} />;
  const P: V2[] = [[r, 0], [0, r], [-r, 0], [0, -r]];
  return <polygon points={P.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} style={style} />;
}

export default function RegularizationWeek({ mode = "ridge" }: { mode?: Mode }) {
  const [rho, setRho] = useState(0.6);
  const [logLam, setLogLam] = useState(0.6);
  const [sigma, setSigma] = useState(1);
  const [gamma, setGamma] = useState(0.5);
  const [seed, setSeed] = useState(2);
  const n = 40;
  // an actual dataset with two correlated features (no intercept, as in the notes: w ∈ ℝ²)
  const data = useMemo(() => {
    const r = mulberry32(seed * 19 + 1);
    const X: V2[] = [], y: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = randn(r), b = rho * a + Math.sqrt(1 - rho * rho) * randn(r);
      X.push([a, b]);
      y.push(1.8 * a + 0.5 * b + randn(r) * (mode === "map" ? sigma : 1));
    }
    const A: M2 = [[0, 0], [0, 0]], bb: V2 = [0, 0];
    X.forEach((x, i) => { A[0][0] += x[0] * x[0]; A[0][1] += x[0] * x[1]; A[1][1] += x[1] * x[1]; bb[0] += x[0] * y[i]; bb[1] += x[1] * y[i]; });
    A[1][0] = A[0][1];
    return { A, b: bb, yy: y.reduce((s, v) => s + v * v, 0) };
  }, [rho, seed, sigma, mode]);
  const { A, b } = data;
  const lam = mode === "map" ? (sigma * sigma) / (gamma * gamma) : 10 ** logLam;
  const ols = solve2(A, b);
  const loss = (w0: number, w1: number) => 0.5 * (A[0][0] * w0 * w0 + 2 * A[0][1] * w0 * w1 + A[1][1] * w1 * w1) - b[0] * w0 - b[1] * w1 + 0.5 * data.yy;
  const Lmin = loss(ols[0], ols[1]);
  const w = mode === "lasso" ? lasso2(A, b, lam) : solve2(A, b, lam);
  const grid = Array.from({ length: 61 }, (_, i) => -1 + i * 0.07); // log10 λ from −1 to 3.2
  const path = useMemo(() => grid.map((g) => (mode === "lasso" ? lasso2(A, b, 10 ** g) : solve2(A, b, 10 ** g))), [A, b, mode]); // eslint-disable-line react-hooks/exhaustive-deps
  const eig = eigSym2(A);
  const norm = mode === "lasso" ? Math.abs(w[0]) + Math.abs(w[1]) : Math.hypot(w[0], w[1]);
  const zero = (v: number) => Math.abs(v) < 1e-9;
  const mark = Math.max(0, Math.min(60, (Math.log10(lam) + 1) / 0.07));
  const R = Math.max(2.6, Math.abs(ols[0]) * 1.3, Math.abs(ols[1]) * 1.6);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-R * 0.5, R]} y={[-R * 0.62, R * 0.62]} equal originAxes aspect={0.7} maxH={430} xLabel="w₁" yLabel="w₂" ariaLabel={`${mode} solution in weight space`}
            underlay={(ctx) => <Field ctx={ctx} f={(a, c) => loss(a, c) - Lmin} scale={mode === "map" ? 2.2 * sigma : 2.2} deps={[A, b, sigma, mode]} />}>
            {mode === "map" && [1, 2].map((m) => <Ball key={m} r={m * gamma} kind="l2" />)}
            {mode !== "map" && <Ball r={norm} kind={mode === "lasso" ? "l1" : "l2"} />}
            <Polyline pts={[ols, ...path]} style={{ stroke: C.a("pink", 0.7), strokeWidth: 1.4, strokeDasharray: "4 3" }} />
            <Dot p={ols} r={5.5} color={C.green} stroke={C.bg} />
            <Dot p={w} r={6} color={C.pink} stroke={C.bg} />
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Weights along the regularisation path: w₁ (blue) and w₂ (green) against log₁₀ λ from −1 to 3.2</div>
            <Trace ariaLabel="coefficient paths versus regularisation strength" xLabel="grid index of log₁₀ λ" marker={mark} series={[{ pts: path.map((p) => p[0]), color: C.blue }, { pts: path.map((p) => p[1]), color: C.green }, { pts: path.map(() => 0), color: C.a("fg", 0.3), dashed: true }]} />
          </div>
          <Legend items={[{ label: "contours of the squared loss", color: C.a("blue", 0.6) }, { label: mode === "map" ? "prior: rings at γ and 2γ" : mode === "lasso" ? "L1 ball ‖w‖₁ = ‖ŵ‖₁" : "L2 ball ‖w‖₂ = ‖ŵ‖₂", color: C.amber }, { label: "least-squares / maximum-likelihood solution", color: C.green }, { label: mode === "map" ? "MAP estimate" : `${mode === "lasso" ? "LASSO" : "ridge"} solution and its path`, color: C.pink }]} />
        </div>
      }
      controls={
        <>
          {mode === "map" ? (
            <>
              <ParameterSlider label="noise standard deviation σ" value={sigma} min={0.3} max={3} step={0.05} onChange={setSigma} />
              <ParameterSlider label="prior standard deviation γ" value={gamma} min={0.1} max={3} step={0.05} onChange={setGamma} />
            </>
          ) : (
            <ParameterSlider label="regularisation strength λ" value={logLam} min={-1} max={3.2} step={0.05} onChange={setLogLam} format={() => fmt(lam, 3)} />
          )}
          <ParameterSlider label="correlation between the two features" value={rho} min={-0.95} max={0.95} step={0.05} onChange={setRho} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New dataset</button>
          <p className="text-[12.5px] leading-relaxed text-muted">
            {mode === "lasso" ? (zero(w[0]) || zero(w[1]) ? "A weight is exactly zero: the loss contour touches the diamond at a corner, which lies on an axis." : "Increase λ until the solution reaches a corner of the diamond: one weight becomes exactly zero while the other is still non-zero.") : mode === "ridge" ? "The ridge solution is where a loss contour first touches the disc. Both weights shrink smoothly towards zero but neither reaches it." : "A small γ says “weights are probably near zero” and pulls the estimate strongly; a large σ says “the data are noisy” and has the same effect."}
          </p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\lambda`, tex: true, value: fmt(lam, 4) },
        { label: mode === "map" ? String.raw`\hat{\mathbf w}_{\text{MAP}}` : mode === "lasso" ? String.raw`\hat{\mathbf w}_{\text{LASSO}}` : String.raw`\hat{\mathbf w}_{R}`, tex: true, value: `(${fmt(w[0], 4)}, ${fmt(w[1], 4)})`, color: C.pink },
        { label: String.raw`\hat{\mathbf w}_{\text{ML}}`, tex: true, value: `(${fmt(ols[0], 4)}, ${fmt(ols[1], 4)})`, color: C.green },
        { label: mode === "lasso" ? String.raw`\|\hat{\mathbf w}\|_1` : String.raw`\|\hat{\mathbf w}\|_2`, tex: true, value: fmt(norm, 4), color: C.amber },
        { label: String.raw`\text{loss increase over ML}`, tex: true, value: fmt(loss(w[0], w[1]) - Lmin, 4) },
        { label: mode === "lasso" ? String.raw`\text{zero weights}` : String.raw`\operatorname{tr}\big((XX^\top+\lambda I)^{-1}\big)`, tex: true, value: mode === "lasso" ? [w[0], w[1]].filter(zero).length : fmt(1 / (eig.values[0] + lam) + 1 / (eig.values[1] + lam), 5) },
      ]} />}
      calc={<CalcSteps steps={mode === "lasso" ? [
        { tex: String.raw`\min_{\mathbf w}\ \tfrac12\|X^\top\mathbf w-\mathbf y\|^2+\lambda\|\mathbf w\|_1,\qquad\lambda=${tex(lam, 3)}` },
        { tex: String.raw`w_1\leftarrow\frac{S\big((X\mathbf y)_1-(XX^\top)_{12}w_2,\ \lambda\big)}{(XX^\top)_{11}}=\frac{S(${tex(b[0] - A[0][1] * w[1], 3)},\ ${tex(lam, 3)})}{${tex(A[0][0], 3)}}=${tex(w[0], 4)}`, note: "coordinate update with soft-thresholding S(z, λ) = sign(z)·max(|z| − λ, 0)" },
        { tex: String.raw`w_2\leftarrow\frac{S(${tex(b[1] - A[0][1] * w[0], 3)},\ ${tex(lam, 3)})}{${tex(A[1][1], 3)}}=${tex(w[1], 4)}`, note: zero(w[1]) ? "|z| ≤ λ, so the weight is set to exactly 0" : undefined },
      ] : [
        { tex: mode === "map" ? String.raw`\lambda=\frac{\sigma^2}{\gamma^2}=\frac{${tex(sigma * sigma, 3)}}{${tex(gamma * gamma, 3)}}=${tex(lam, 4)}` : String.raw`\min_{\mathbf w}\ \tfrac12\|X^\top\mathbf w-\mathbf y\|^2+\tfrac{\lambda}{2}\|\mathbf w\|_2^2,\qquad\lambda=${tex(lam, 3)}`, note: mode === "map" ? "the ridge strength implied by the prior and the noise" : undefined },
        { tex: String.raw`\hat{\mathbf w}=(XX^\top+\lambda I)^{-1}X\mathbf y=${texMatrix([[A[0][0] + lam, A[0][1]], [A[1][0], A[1][1] + lam]], 2)}^{-1}${texMatrix([[b[0]], [b[1]]], 2)}=${texMatrix([[w[0]], [w[1]]], 4)}` },
        { tex: String.raw`\text{eigenvalues of }XX^\top:\ ${tex(eig.values[0], 2)},\ ${tex(eig.values[1], 2)}\ \longrightarrow\ \text{shrink factors }\frac{\lambda_i}{\lambda_i+\lambda}=${tex(eig.values[0] / (eig.values[0] + lam), 3)},\ ${tex(eig.values[1] / (eig.values[1] + lam), 3)}`, note: "directions with little data support (small eigenvalue) are shrunk the most" },
      ]} />}
      note={mode === "lasso" ? "With strongly correlated features the loss valley is long and narrow, and LASSO tends to keep one feature and drop the other. Which one it keeps can flip between datasets." : mode === "ridge" ? "Raise the feature correlation: the smaller eigenvalue of XXᵀ collapses, the maximum-likelihood solution becomes unstable from dataset to dataset, and the trace readout shows how much λ stabilises it." : "The MAP estimate maximises prior × likelihood. Taking logs turns the product into the ridge objective, so the pink point coincides with ridge regression at λ = σ²/γ²."}
    />
  );
}

/* ------------------------------------------------------------------ k-fold cross-validation for choosing λ */
const truth = (x: number) => Math.sin(2.6 * x) + 0.4 * x;

export function CrossValidation() {
  const [k, setK] = useState(5);
  const [fold, setFold] = useState(0);
  const [logLam, setLogLam] = useState(-3);
  const [seed, setSeed] = useState(1);
  const n = 30, DEG = 9;
  const data = useMemo(() => {
    const r = mulberry32(seed * 7 + 2);
    const pts = Array.from({ length: n }, () => { const x = r() * 2 - 1; return [x, truth(x) + 0.28 * randn(r)] as V2; });
    return shuffle(pts, mulberry32(seed + 40));
  }, [seed]);
  const folds = useMemo(() => Array.from({ length: k }, (_, f) => data.map((_, i) => i).filter((i) => i % k === f)), [data, k]);
  const fitOn = (idx: number[], lam: number) => {
    const w = ridgeFit(polyDesign(idx.map((i) => data[i][0]), DEG), idx.map((i) => data[i][1]), lam) ?? Array(DEG + 1).fill(0);
    return (x: number) => predictLinear(polyDesign([x], DEG), w)[0];
  };
  const cvAt = (lam: number) => {
    const errs = folds.map((val) => {
      const tr = data.map((_, i) => i).filter((i) => !val.includes(i));
      const f = fitOn(tr, lam);
      return { val: val.reduce((s, i) => s + (data[i][1] - f(data[i][0])) ** 2, 0) / val.length, train: tr.reduce((s, i) => s + (data[i][1] - f(data[i][0])) ** 2, 0) / tr.length };
    });
    return { errs, cv: errs.reduce((s, e) => s + e.val, 0) / k, train: errs.reduce((s, e) => s + e.train, 0) / k };
  };
  const grid = useMemo(() => Array.from({ length: 41 }, (_, i) => -8 + i * 0.25), []);
  const curve = useMemo(() => grid.map((g) => cvAt(10 ** g)), [folds]); // eslint-disable-line react-hooks/exhaustive-deps
  const lam = 10 ** logLam;
  const cur = useMemo(() => cvAt(lam), [folds, lam]); // eslint-disable-line react-hooks/exhaustive-deps
  const bestI = curve.reduce((b, c, i) => (c.cv < curve[b].cv ? i : b), 0);
  const f = Math.min(fold, k - 1);
  const val = folds[f];
  const model = useMemo(() => fitOn(data.map((_, i) => i).filter((i) => !val.includes(i)), lam), [data, val, lam]); // eslint-disable-line react-hooks/exhaustive-deps
  const cap = Math.min(1.2, Math.max(...curve.map((c) => Math.min(1.2, c.cv))) * 1.05);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <div className="flex gap-[3px]" aria-label="folds">
            {Array.from({ length: k }, (_, j) => (
              <button key={j} onClick={() => setFold(j)} aria-label={`show fold ${j + 1}`} className="h-6 flex-1 rounded-sm text-[10.5px] font-medium" style={{ background: j === f ? "rgb(var(--pink) / 0.55)" : "rgb(var(--blue) / 0.22)", color: "rgb(var(--fg))" }}>{k <= 15 ? j + 1 : ""}</button>
            ))}
          </div>
          <Plot x={[-1.05, 1.05]} y={[-2, 2.2]} aspect={0.48} maxH={340} xLabel="x" yLabel="y" ariaLabel="model trained on k minus one folds and validated on the held-out fold">
            <FnPath f={truth} style={{ stroke: C.a("green", 0.6), strokeWidth: 1.2, strokeDasharray: "5 4" }} />
            <FnPath f={model} n={400} style={{ stroke: C.blue, strokeWidth: 2 }} />
            {val.map((i) => <Polyline key={`e${i}`} pts={[data[i], [data[i][0], model(data[i][0])]]} style={{ stroke: C.a("pink", 0.8), strokeWidth: 1.3 }} />)}
            {data.map((p, i) => <Dot key={i} p={p} r={val.includes(i) ? 4.6 : 3.2} color={val.includes(i) ? C.pink : C.a("blue", 0.75)} stroke={C.bg} />)}
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Average validation error (pink) and training error (blue) for log₁₀ λ from −8 to 2</div>
            <Trace ariaLabel="cross-validation and training error versus regularisation strength" xLabel="grid index of log₁₀ λ" marker={(logLam + 8) / 0.25} yDomain={[0, cap]} series={[{ pts: curve.map((c) => Math.min(cap, c.cv)), color: C.pink, dots: true }, { pts: curve.map((c) => c.train), color: C.blue }]}>
              <Dot p={[bestI, curve[bestI].cv]} r={5} color={C.green} stroke={C.bg} />
            </Trace>
          </div>
          <Legend items={[{ label: `training folds (${n - val.length} points)`, color: C.blue }, { label: `validation fold ${f + 1} (${val.length} point${val.length > 1 ? "s" : ""})`, color: C.pink }, { label: "true function", color: C.green, dashed: true }, { label: "λ with the lowest CV error", color: C.green }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="number of folds k" value={k} min={2} max={n} step={1} onChange={(v) => { setK(v); setFold(0); }} format={(v) => (v === n ? `${v} (leave-one-out)` : String(v))} />
          <ParameterSlider label="fold held out for validation" value={f + 1} min={1} max={k} step={1} onChange={(v) => setFold(v - 1)} />
          <ParameterSlider label="regularisation log₁₀ λ" value={logLam} min={-8} max={2} step={0.25} onChange={setLogLam} />
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={() => setLogLam(grid[bestI])}>Use best λ</button>
            <button className="btn flex-1" onClick={() => setSeed((s) => s + 1)}>New data</button>
          </div>
          {k <= 10 && <MiniTable head={["fold", "validation MSE", "training MSE"]} rows={cur.errs.map((e, j) => [j + 1, fmt(e.val, 4), fmt(e.train, 4)])} rowClass={(j) => (j === f ? "bg-raised" : undefined)} />}
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{CV}(\lambda)`, tex: true, value: fmt(cur.cv, 4), color: C.pink },
        { label: String.raw`\text{mean training MSE}`, tex: true, value: fmt(cur.train, 4), color: C.blue },
        { label: String.raw`\lambda^*=\arg\min\text{CV}`, tex: true, value: (10 ** grid[bestI]).toExponential(1), color: C.green },
        { label: String.raw`\text{CV}(\lambda^*)`, tex: true, value: fmt(curve[bestI].cv, 4) },
        { label: String.raw`\text{models trained per }\lambda`, tex: true, value: k },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{CV}(\lambda)=\frac1k\sum_{j=1}^{k}\frac{1}{|F_j|}\sum_{i\in F_j}\big(y_i-\hat{\mathbf w}_{-j}(\lambda)^\top\phi(x_i)\big)^2`, note: "ŵ₋ⱼ is trained without fold j" },
        { tex: String.raw`\text{CV}(${tex(lam, 4)})=\frac{1}{${k}}\big(${cur.errs.slice(0, 4).map((e) => tex(e.val, 3)).join("+")}${k > 4 ? String.raw`+\cdots` : ""}\big)=${tex(cur.cv, 4)}` },
        { tex: String.raw`\text{total fits for the whole curve}=k\times|\text{grid}|=${k}\times41=${k * 41}`, note: k === n ? "leave-one-out: n trainings per value of λ" : undefined },
      ]} />}
      note={`The model is a degree-${DEG} polynomial fitted by ridge regression, so λ controls its effective complexity. Tiny λ overfits (low training error, high validation error); large λ underfits (both high). Training error alone would always pick the smallest λ.`}
    />
  );
}
