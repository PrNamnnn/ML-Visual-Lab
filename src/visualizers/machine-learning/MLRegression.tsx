"use client";
import { useMemo, useState } from "react";
import { Plot, FnPath, Dot, C, Polyline, HeatmapCanvas, useThemeRGB } from "@/components/visualization/Plot";
import { Stems } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Toggle, AnimationControls } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { polyDesign, ridgeFit, makeDataset, knnPredict, kmeansAssign, kmeansUpdate, type P2 } from "@/lib/ml/models";
import { useTicker } from "@/hooks/useSize";
import { fmt, pct, tex } from "@/lib/utilities/format";

export function poissonSample(rng: () => number, lam: number) {
  const L = Math.exp(-lam);
  let k = 0, p = 1;
  do { k++; p *= rng(); } while (p > L);
  return k - 1;
}
const logFact = (k: number) => { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; };

/* ------------------------------ Poisson regression ------------------------------ */
export function PoissonViz() {
  const [x0, setX0] = useState(3);
  const [showOls, setShowOls] = useState(true);
  const data = useMemo(() => { const r = mulberry32(12); return Array.from({ length: 70 }, () => { const x = 4 * r(); return [x, poissonSample(r, Math.exp(0.2 + 0.55 * x))] as [number, number]; }); }, []);
  // Newton–Raphson / IRLS on the Poisson log-likelihood
  const fit = useMemo(() => {
    let b0 = Math.log(data.reduce((s, d) => s + d[1], 0) / data.length), b1 = 0;
    for (let it = 0; it < 30; it++) {
      let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
      for (const [x, y] of data) { const mu = Math.exp(b0 + b1 * x); g0 += y - mu; g1 += (y - mu) * x; h00 += mu; h01 += mu * x; h11 += mu * x * x; }
      const det = h00 * h11 - h01 * h01;
      b0 += (h11 * g0 - h01 * g1) / det; b1 += (-h01 * g0 + h00 * g1) / det;
    }
    const dev = 2 * data.reduce((s, [x, y]) => { const mu = Math.exp(b0 + b1 * x); return s + (y > 0 ? y * Math.log(y / mu) : 0) - (y - mu); }, 0);
    return { b0, b1, dev };
  }, [data]);
  const ols = useMemo(() => { const n = data.length, mx = data.reduce((s, d) => s + d[0], 0) / n, my = data.reduce((s, d) => s + d[1], 0) / n; const b1 = data.reduce((s, d) => s + (d[0] - mx) * (d[1] - my), 0) / data.reduce((s, d) => s + (d[0] - mx) ** 2, 0); return { b0: my - b1 * mx, b1 }; }, [data]);
  const mu0 = Math.exp(fit.b0 + fit.b1 * x0);
  const pmf = Array.from({ length: 31 }, (_, k) => [k, Math.exp(k * Math.log(mu0) - mu0 - logFact(k))] as [number, number]);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[0, 4]} y={[-2, 26]} aspect={0.5} maxH={330} xLabel="x (e.g. hours of exposure)" yLabel="count y" ariaLabel="count data with Poisson regression fit">
            {data.map((d, i) => <Dot key={i} p={d} r={3} color={C.blue} opacity={0.75} />)}
            <FnPath f={(x) => Math.exp(fit.b0 + fit.b1 * x)} style={{ stroke: C.pink, strokeWidth: 2.4 }} />
            {showOls && <FnPath f={(x) => ols.b0 + ols.b1 * x} style={{ stroke: C.amber, strokeWidth: 1.5, strokeDasharray: "5 4" }} />}
            <Polyline pts={[[x0, -2], [x0, 26]]} style={{ stroke: C.a("fg", 0.3), strokeDasharray: "3 3" }} />
          </Plot>
          <Plot x={[-0.5, 30.5]} y={[0, Math.max(...pmf.map((p) => p[1])) * 1.15]} aspect={0.25} minH={120} maxH={170} xLabel={`P(Y = k | x = ${fmt(x0, 2)})`} ariaLabel="Poisson probability mass function at the chosen x">
            <Stems pts={pmf} color={C.pink} />
          </Plot>
          <Legend items={[{ label: "Poisson GLM mean μ(x) = exp(β₀ + β₁x)", color: C.pink }, ...(showOls ? [{ label: "ordinary least squares line", color: C.amber, dashed: true }] : [])]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="inspect x" value={x0} min={0} max={4} step={0.05} onChange={setX0} />
          <Toggle label="Show the OLS line" checked={showOls} onChange={setShowOls} />
          <p className="text-[12px] leading-relaxed text-faint">Counts cannot be negative and their spread grows with their mean. A Poisson GLM models log μ linearly, so predictions are always positive and the variance equals the mean.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\hat\\beta_0,\\ \\hat\\beta_1", tex: true, value: `${fmt(fit.b0, 3)}, ${fmt(fit.b1, 3)}`, hint: "true values 0.2, 0.55" },
        { label: "e^{\\hat\\beta_1}", tex: true, value: fmt(Math.exp(fit.b1), 3), hint: "multiplicative change in the expected count per unit x" },
        { label: "\\mu(x)", tex: true, value: fmt(mu0, 3), color: C.pink },
        { label: "\\text{deviance}", tex: true, value: fmt(fit.dev, 2) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`Y\mid x\sim\text{Poisson}(\mu),\quad\log\mu=\beta_0+\beta_1x` },
        { tex: String.raw`\mu(${tex(x0, 2)})=e^{${tex(fit.b0, 3)}+${tex(fit.b1, 3)}\cdot${tex(x0, 2)}}=${tex(mu0, 3)}`, note: "fitted by Newton–Raphson (IRLS) on the log-likelihood Σ yᵢ log μᵢ − μᵢ" },
      ]} />}
    />
  );
}

/* ------------------------------ quantile regression ------------------------------ */
function quantileLine(data: [number, number][], tau: number) {
  // the optimum of the pinball loss passes through (at least) two data points: search all pairs
  const loss = (a: number, b: number) => data.reduce((s, [x, y]) => { const r = y - (a + b * x); return s + (r >= 0 ? tau * r : (tau - 1) * r); }, 0);
  let best = { a: 0, b: 0, l: Infinity };
  for (let i = 0; i < data.length; i++) for (let j = i + 1; j < data.length; j++) {
    const dx = data[j][0] - data[i][0];
    if (Math.abs(dx) < 1e-9) continue;
    const b = (data[j][1] - data[i][1]) / dx, a = data[i][1] - b * data[i][0];
    const l = loss(a, b);
    if (l < best.l) best = { a, b, l };
  }
  return best;
}
export function QuantileViz() {
  const [tau, setTau] = useState(0.9);
  const data = useMemo(() => { const r = mulberry32(5); return Array.from({ length: 60 }, () => { const x = 10 * r(); return [x, 2 + 0.6 * x + (0.3 + 0.25 * x) * randn(r)] as [number, number]; }); }, []);
  const fit = useMemo(() => quantileLine(data, tau), [data, tau]);
  const ref = useMemo(() => [0.1, 0.5, 0.9].map((t) => quantileLine(data, t)), [data]);
  const below = data.filter(([x, y]) => y < fit.a + fit.b * x + 1e-9).length / data.length;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, 10]} y={[-2, 16]} aspect={0.55} maxH={380} xLabel="x" yLabel="y (noise grows with x)" ariaLabel="quantile regression lines">
            {ref.map((f, i) => <FnPath key={i} f={(x) => f.a + f.b * x} style={{ stroke: C.a("fg", 0.18), strokeWidth: 1, strokeDasharray: "4 3" }} />)}
            {data.map(([x, y], i) => <Dot key={i} p={[x, y]} r={3} color={y < fit.a + fit.b * x + 1e-9 ? C.blue : C.pink} opacity={0.85} />)}
            <FnPath f={(x) => fit.a + fit.b * x} style={{ stroke: C.green, strokeWidth: 2.6 }} />
          </Plot>
          <Legend items={[{ label: `τ = ${fmt(tau, 2)} quantile line`, color: C.green }, { label: "points below", color: C.blue }, { label: "points above", color: C.pink }, { label: "τ = 0.1, 0.5, 0.9 for reference", color: C.a("fg", 0.3), dashed: true }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="quantile τ" value={tau} min={0.05} max={0.95} step={0.05} onChange={setTau} />
          <p className="text-[12px] leading-relaxed text-faint">The τ = 0.9 line is the value that 90 % of outcomes fall below at each x. Because the noise widens with x, the upper and lower quantile lines fan out — something a single mean line cannot show.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{intercept},\\ \\text{slope}", tex: true, value: `${fmt(fit.a, 3)}, ${fmt(fit.b, 3)}` },
        { label: "\\text{share of points below}", tex: true, value: pct(below), color: C.blue, hint: "≈ τ for the optimal line" },
        { label: "\\text{pinball loss}", tex: true, value: fmt(fit.l / data.length, 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\rho_\tau(r)=\begin{cases}\tau\,r & r\ge0\\(\tau-1)\,r & r<0\end{cases},\qquad \min_{a,b}\sum_i\rho_{${tex(tau, 2)}}(y_i-a-bx_i)`, note: "under-predictions cost τ per unit, over-predictions cost 1 − τ" },
        { tex: String.raw`\text{optimum passes through two data points}\Rightarrow\text{exact search over all pairs}` },
      ]} />}
    />
  );
}

/* ------------------------------ robust regression ------------------------------ */
export function RobustViz() {
  const [nOut, setNOut] = useState(6);
  const [delta, setDelta] = useState(1);
  const [thr, setThr] = useState(1);
  const base = useMemo(() => { const r = mulberry32(9); return Array.from({ length: 40 }, () => { const x = 10 * r(); return [x, 1 + 0.8 * x + 0.5 * randn(r)] as [number, number]; }); }, []);
  const outl = useMemo(() => { const r = mulberry32(19); return Array.from({ length: 15 }, () => [7 + 3 * r(), 0.5 + 2 * r()] as [number, number]); }, []);
  const data = [...base, ...outl.slice(0, nOut)];
  const wls = (w: number[]) => {
    const sw = w.reduce((s, v) => s + v, 0), mx = data.reduce((s, d, i) => s + w[i] * d[0], 0) / sw, my = data.reduce((s, d, i) => s + w[i] * d[1], 0) / sw;
    const b = data.reduce((s, d, i) => s + w[i] * (d[0] - mx) * (d[1] - my), 0) / data.reduce((s, d, i) => s + w[i] * (d[0] - mx) ** 2, 0);
    return { a: my - b * mx, b };
  };
  const ols = wls(data.map(() => 1));
  // Huber via iteratively reweighted least squares: weight 1 inside δ, δ/|r| outside
  let hub = ols;
  for (let it = 0; it < 50; it++) hub = wls(data.map(([x, y]) => { const r = Math.abs(y - hub.a - hub.b * x); return r <= delta ? 1 : delta / r; }));
  // RANSAC: fit lines through random pairs, keep the one with most inliers, refit on inliers
  const ransac = useMemo(() => {
    const r = mulberry32(2);
    let best = { inl: [] as number[], a: 0, b: 0 };
    for (let t = 0; t < 200; t++) {
      const i = Math.floor(r() * data.length), j = Math.floor(r() * data.length);
      if (i === j || data[i][0] === data[j][0]) continue;
      const b = (data[j][1] - data[i][1]) / (data[j][0] - data[i][0]), a = data[i][1] - b * data[i][0];
      const inl = data.map((d, k) => (Math.abs(d[1] - a - b * d[0]) < thr ? k : -1)).filter((k) => k >= 0);
      if (inl.length > best.inl.length) best = { inl, a, b };
    }
    const fit = wls(data.map((_, k) => (best.inl.includes(k) ? 1 : 0)));
    return { ...fit, inl: best.inl };
  }, [nOut, thr]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, 10]} y={[-1, 11]} aspect={0.55} maxH={380} xLabel="x" yLabel="y" ariaLabel="ordinary, Huber and RANSAC regression with outliers">
            {data.map((d, i) => <Dot key={i} p={d} r={i >= base.length ? 4.5 : 3} color={i >= base.length ? C.red : ransac.inl.includes(i) ? C.blue : C.a("blue", 0.4)} />)}
            <FnPath f={(x) => ols.a + ols.b * x} style={{ stroke: C.amber, strokeWidth: 2, strokeDasharray: "6 4" }} />
            <FnPath f={(x) => hub.a + hub.b * x} style={{ stroke: C.pink, strokeWidth: 2.2 }} />
            <FnPath f={(x) => ransac.a + ransac.b * x} style={{ stroke: C.green, strokeWidth: 2.2 }} />
          </Plot>
          <Legend items={[{ label: "OLS (squared loss)", color: C.amber, dashed: true }, { label: "Huber", color: C.pink }, { label: "RANSAC", color: C.green }, { label: "outliers", color: C.red }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="number of outliers" value={nOut} min={0} max={15} step={1} onChange={setNOut} />
          <ParameterSlider label="Huber δ" value={delta} min={0.1} max={5} step={0.1} onChange={setDelta} />
          <ParameterSlider label="RANSAC inlier threshold" value={thr} min={0.2} max={4} step={0.1} onChange={setThr} />
          <p className="text-[12px] leading-relaxed text-faint">Squared loss lets a few far-away points drag the line. Huber loss grows only linearly beyond δ; RANSAC ignores points that disagree with the best consensus line entirely.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{true slope}", tex: true, value: "0.800" },
        { label: "\\hat b_{\\text{OLS}}", tex: true, value: fmt(ols.b, 3), color: C.amber },
        { label: "\\hat b_{\\text{Huber}}", tex: true, value: fmt(hub.b, 3), color: C.pink },
        { label: "\\hat b_{\\text{RANSAC}}", tex: true, value: fmt(ransac.b, 3), color: C.green },
        { label: "\\text{RANSAC inliers}", tex: true, value: `${ransac.inl.length} / ${data.length}` },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`L_\delta(r)=\begin{cases}\tfrac12r^2 & |r|\le${tex(delta, 1)}\\ ${tex(delta, 1)}\,(|r|-\tfrac{${tex(delta, 1)}}{2}) & |r|>${tex(delta, 1)}\end{cases}`, note: "solved by iteratively reweighted least squares, weight min(1, δ/|r|)" },
      ]} />}
    />
  );
}

/* ------------------------------ train / validation / test ------------------------------ */
const truthTVT = (x: number) => Math.sin(2.2 * x) + 0.3 * x;
export function SplitViz() {
  const [trainPct, setTrainPct] = useState(60);
  const [valPct, setValPct] = useState(20);
  const [peek, setPeek] = useState(false);
  const [seed, setSeed] = useState(4);
  const data = useMemo(() => { const r = mulberry32(seed); return Array.from({ length: 60 }, () => { const x = 2.8 * r() - 1.4; return [x, truthTVT(x) + 0.3 * randn(r)] as [number, number]; }); }, [seed]);
  const nTr = Math.round((trainPct / 100) * 60), nVa = Math.min(60 - nTr - 2, Math.round((valPct / 100) * 60));
  const tr = data.slice(0, nTr), va = data.slice(nTr, nTr + nVa), te = data.slice(nTr + nVa);
  const mse = (w: number[] | null, d: [number, number][]) => (w ? d.reduce((s, [x, y]) => s + (y - w.reduce((a, c, k) => a + c * x ** k, 0)) ** 2, 0) / Math.max(1, d.length) : NaN);
  const curves = useMemo(() => Array.from({ length: 10 }, (_, i) => {
    const deg = i + 1;
    const w = ridgeFit(polyDesign(tr.map((p) => p[0]), deg), tr.map((p) => p[1]), 1e-6);
    return { deg, w, tr: mse(w, tr), va: mse(w, va), te: mse(w, te) };
  }), [tr, va, te]);  
  const pick = curves.reduce((b, c) => ((peek ? c.te : c.va) < (peek ? b.te : b.va) ? c : b));
  const ymax = Math.min(1.5, Math.max(...curves.map((c) => Math.max(c.tr, c.va, c.te)).filter(Number.isFinite)));
  return (
    <LabShell
      viz={
        <div className="grid gap-3 lg:grid-cols-2">
          <Plot x={[-1.5, 1.5]} y={[-2.2, 2.2]} aspect={0.75} maxH={300} ariaLabel="data coloured by split with the selected model">
            {pick.w && <FnPath f={(x) => pick.w!.reduce((a, c, k) => a + c * x ** k, 0)} style={{ stroke: C.fg, strokeWidth: 2 }} />}
            {tr.map((p, i) => <Dot key={"t" + i} p={p} r={3.2} color={C.blue} />)}
            {va.map((p, i) => <Dot key={"v" + i} p={p} r={3.2} color={C.amber} />)}
            {te.map((p, i) => <Dot key={"e" + i} p={p} r={3.2} color={C.pink} />)}
          </Plot>
          <Plot x={[1, 10]} y={[0, ymax * 1.05]} aspect={0.75} maxH={300} xLabel="polynomial degree" yLabel="MSE" ariaLabel="error curves for train, validation and test">
            <Polyline pts={curves.map((c) => [c.deg, Math.min(c.tr, ymax)])} style={{ stroke: C.blue, strokeWidth: 2 }} />
            <Polyline pts={curves.map((c) => [c.deg, Math.min(c.va, ymax)])} style={{ stroke: C.amber, strokeWidth: 2 }} />
            <Polyline pts={curves.map((c) => [c.deg, Math.min(c.te, ymax)])} style={{ stroke: C.pink, strokeWidth: 1.4, strokeDasharray: "4 3" }} />
            <Dot p={[pick.deg, Math.min(peek ? pick.te : pick.va, ymax)]} r={6} color={C.green} />
          </Plot>
          <div className="lg:col-span-2"><Legend items={[{ label: `train (${tr.length})`, color: C.blue }, { label: `validation (${va.length})`, color: C.amber }, { label: `test (${te.length})`, color: C.pink, dashed: true }, { label: "selected degree", color: C.green }]} /></div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="train %" value={trainPct} min={20} max={80} step={5} onChange={setTrainPct} />
          <ParameterSlider label="validation %" value={valPct} min={5} max={40} step={5} onChange={setValPct} />
          <Toggle label="Cheat: choose the degree on the test set" checked={peek} onChange={setPeek} />
          <button className="btn" onClick={() => setSeed((s) => s + 1)}>New random dataset</button>
          <p className="text-[12px] leading-relaxed text-faint">Train fits the weights, validation chooses the degree, and test is looked at once to report the final error. Choosing on the test set makes that final number optimistic.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{chosen degree}", tex: true, value: pick.deg, color: C.green },
        { label: "\\text{train MSE}", tex: true, value: fmt(pick.tr, 4), color: C.blue },
        { label: "\\text{val MSE}", tex: true, value: fmt(pick.va, 4), color: C.amber },
        { label: "\\text{test MSE (reported)}", tex: true, value: fmt(pick.te, 4), color: C.pink },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`d^*=\arg\min_d\ \text{MSE}_{\text{${peek ? "test" : "val"}}}(\hat f_d),\qquad \text{report }\text{MSE}_{\text{test}}(\hat f_{d^*})=${tex(pick.te, 4)}`, note: peek ? "the test set was used for a decision, so it is no longer an unbiased estimate" : "the test set was not used for any decision" }]} />}
    />
  );
}

/* ------------------------------ learning paradigms (label propagation) ------------------------------ */
type Paradigm = "supervised" | "unsupervised" | "semi";
export function ParadigmViz() {
  const [mode, setMode] = useState<Paradigm>("semi");
  const [nLab, setNLab] = useState(4);
  const [iter, setIter] = useState(0);
  const [playing, setPlaying] = useState(false);
  const th = useThemeRGB();
  const pts = useMemo(() => makeDataset("moons", 160, 0.12, 3), []);
  const labelled = useMemo(() => { const out: number[] = []; for (let c = 0; c < 2; c++) out.push(...pts.map((p, i) => (p.label === c ? i : -1)).filter((i) => i >= 0).slice(0, Math.max(1, Math.round(nLab / 2)))); return out; }, [pts, nLab]);
  // label propagation on a k-NN graph: F ← S F, then clamp the labelled points
  const graph = useMemo(() => pts.map((p, i) => pts.map((q, j) => ({ j, d: (p.x - q.x) ** 2 + (p.y - q.y) ** 2 })).filter((o) => o.j !== i).sort((a, b) => a.d - b.d).slice(0, 7).map((o) => o.j)), [pts]);
  const F = useMemo(() => {
    let f = pts.map((_, i) => (labelled.includes(i) ? pts[i].label : 0.5));
    for (let t = 0; t < iter; t++) f = f.map((v, i) => (labelled.includes(i) ? pts[i].label : graph[i].reduce((s, j) => s + f[j], 0) / graph[i].length));
    return f;
  }, [pts, labelled, graph, iter]);
  const km = useMemo(() => {
    const P: P2[] = pts.map((p) => [p.x, p.y]);
    let Cc: P2[] = [[-1, 0.5], [1.5, -0.2]];
    let a = kmeansAssign(P, Cc);
    for (let t = 0; t < 20; t++) { Cc = kmeansUpdate(P, a, Cc); a = kmeansAssign(P, Cc); }
    return a;
  }, [pts]);
  useTicker(playing && iter < 60, 120, () => setIter((i) => i + 1));
  const pred = (i: number) => (mode === "semi" ? (F[i] >= 0.5 ? 1 : 0) : mode === "unsupervised" ? km[i] : pts[i].label);
  const acc = pts.reduce((s, p, i) => s + (pred(i) === p.label ? 1 : 0), 0) / pts.length;
  const accU = Math.max(acc, 1 - acc); // cluster ids are arbitrary
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-1.6, 2.6]} y={[-1.2, 1.6]} equal aspect={0.62} maxH={420} ariaLabel={`${mode} learning on the two-moons data`}
            underlay={mode === "supervised" ? (ctx) => <HeatmapCanvas ctx={ctx} res={5} deps={[mode, th]} f={(x, y) => { const pr = knnPredict(pts, x, y, 7).prob; const c = pr >= 0.5 ? th.pink : th.blue; const k = 0.12 + 0.3 * Math.abs(pr - 0.5) * 2; return [th.panel[0] + (c[0] - th.panel[0]) * k, th.panel[1] + (c[1] - th.panel[1]) * k, th.panel[2] + (c[2] - th.panel[2]) * k, 255]; }} /> : undefined}>
            {pts.map((p, i) => {
              const isLab = labelled.includes(i);
              const color = mode === "supervised" ? (p.label ? C.pink : C.blue) : mode === "unsupervised" ? (km[i] ? C.green : C.amber) : F[i] === 0.5 ? C.faint : F[i] > 0.5 ? C.pink : C.blue;
              return <Dot key={i} p={[p.x, p.y]} r={mode === "semi" && isLab ? 6.5 : 3.4} color={color} stroke={mode === "semi" && isLab ? C.fg : undefined} opacity={mode === "semi" && !isLab ? 0.35 + 0.65 * Math.abs(F[i] - 0.5) * 2 : 1} />;
            })}
          </Plot>
          <Legend items={mode === "semi" ? [{ label: "labelled points (ringed)", color: C.fg }, { label: "propagated class 0 / 1", color: C.blue }, { label: "no information yet", color: C.faint }] : mode === "unsupervised" ? [{ label: "k-means cluster A", color: C.amber }, { label: "k-means cluster B", color: C.green }] : [{ label: "class 0", color: C.blue }, { label: "class 1", color: C.pink }, { label: "7-NN decision regions", color: C.muted }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Paradigm" value={mode} onChange={(m) => { setMode(m); setIter(0); }} options={[{ value: "supervised", label: "Supervised" }, { value: "unsupervised", label: "Unsupervised" }, { value: "semi", label: "Semi-supervised" }]} />
          {mode === "semi" && <>
            <ParameterSlider label="labelled points" value={nLab} min={2} max={20} step={2} onChange={(v) => { setNLab(v); setIter(0); }} />
            <AnimationControls playing={playing} onPlay={() => setPlaying((p) => !p)} onStep={() => setIter((i) => i + 1)} onReset={() => { setIter(0); setPlaying(false); }} />
          </>}
          <p className="text-[12px] leading-relaxed text-faint">{mode === "supervised" ? "Every point has a label; the model learns the boundary directly." : mode === "unsupervised" ? "No labels at all: k-means groups by distance and cuts each moon in half, because 'closest centre' is not the structure here." : "Only a few points are labelled. Labels spread along the nearest-neighbour graph, following the shape of the unlabelled data."}</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{labels used}", tex: true, value: mode === "supervised" ? pts.length : mode === "unsupervised" ? 0 : labelled.length },
        { label: "\\text{agreement with true classes}", tex: true, value: pct(mode === "unsupervised" ? accU : acc), hint: "for clusters, the better of the two possible id matchings" },
        ...(mode === "semi" ? [{ label: "\\text{propagation steps}", tex: true, value: iter }] : []),
      ]} />}
      calc={<CalcSteps steps={[mode === "semi"
        ? { tex: String.raw`f_i\leftarrow\frac{1}{|N_7(i)|}\sum_{j\in N_7(i)}f_j\ \ (\text{unlabelled }i),\qquad f_i=y_i\ (\text{labelled})`, note: "label propagation on a 7-nearest-neighbour graph" }
        : mode === "unsupervised" ? { tex: String.raw`\min_{\boldsymbol\mu_1,\boldsymbol\mu_2}\sum_i\min_k\|\mathbf x_i-\boldsymbol\mu_k\|^2`, note: "k-means objective — uses no labels" }
        : { tex: String.raw`\hat y(\mathbf x)=\text{majority label of the 7 nearest training points}` }]} />}
    />
  );
}
