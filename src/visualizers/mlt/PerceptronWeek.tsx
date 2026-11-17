"use client";
import { useMemo, useState } from "react";
import { Plot, Arrow, Dot, Polyline, FnPath, HeatmapCanvas, C, useThemeRGB } from "@/components/visualization/Plot";
import { VLine } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend, Warn } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, AnimationControls } from "@/components/controls";
import { useTicker } from "@/hooks/useSize";
import { mulberry32, randn } from "@/lib/math/random";
import { maxMarginOrigin } from "@/lib/ml/svmOrigin";
import { fmt, tex, texMatrix } from "@/lib/utilities/format";
import { Trace, LineWB, separableData, sigmoid, type V2 } from "./common";
import { HalfPlanes, predict, type LP } from "./ClassifierWeek";

const POS = C.pink, NEG = C.blue;
const dotw = (w: V2, x: V2) => w[0] * x[0] + w[1] * x[1];

interface PStep { w: V2; i: number; before: number }
/** Runs the perceptron to convergence, visiting points cyclically. Returns every update. */
function runPerceptron(D: LP[], cap = 5000): PStep[] {
  let w: V2 = [0, 0];
  const steps: PStep[] = [];
  let clean = 0, i = 0;
  while (clean < D.length && steps.length < cap) {
    const p = D[i];
    if (predict(w, p.x) !== p.y) {
      const before = p.y * dotw(w, p.x);
      w = [w[0] + p.y * p.x[0], w[1] + p.y * p.x[1]];
      steps.push({ w, i, before });
      clean = 0;
    } else clean++;
    i = (i + 1) % D.length;
  }
  return steps;
}

/* ------------------------------------------------------------------ the algorithm, one update at a time */
function PerceptronRun() {
  const [margin, setMargin] = useState(0.45);
  const [seed, setSeed] = useState(4);
  const [l, setL] = useState(0);
  const [playing, setPlaying] = useState(false);
  const D = useMemo(() => separableData(seed, 10, margin, 0.8), [seed, margin]);
  const steps = useMemo(() => runPerceptron(D), [D]);
  const k = Math.min(l, steps.length);
  useTicker(playing && k < steps.length, 700, () => setL((v) => v + 1));
  const w: V2 = k === 0 ? [0, 0] : steps[k - 1].w;
  const prev: V2 = k <= 1 ? [0, 0] : steps[k - 2].w;
  const last = k > 0 ? steps[k - 1] : null;
  const mistakes = D.filter((p) => predict(w, p.x) !== p.y).length;
  const trace = [D.filter((p) => predict([0, 0], p.x) !== p.y).length, ...steps.slice(0, k).map((s) => D.filter((p) => predict(s.w, p.x) !== p.y).length)];
  const done = k >= steps.length;
  const lp = last ? D[last.i] : null;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-5, 5]} y={[-4.2, 4.2]} equal originAxes aspect={0.74} maxH={440} ariaLabel="perceptron decision boundary after the current update" underlay={(ctx) => (k > 0 ? <HalfPlanes ctx={ctx} w={w} /> : null)}>
            {k > 0 && <LineWB w={w} />}
            {D.map((p, j) => { const bad = predict(w, p.x) !== p.y; return <Dot key={j} p={p.x} r={bad ? 6 : 4.2} color={p.y > 0 ? POS : NEG} stroke={last && j === last.i ? C.amber : bad ? C.fg : C.bg} />; })}
            {lp && <Arrow from={prev} to={w} color="amber" label="+ yᵢxᵢ" dashed />}
            {k > 1 && <Arrow from={[0, 0]} to={prev} color="muted" width={1.4} />}
            {k > 0 && <Arrow from={[0, 0]} to={w} color="fg" label="w" />}
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Training points misclassified after each update — not monotone, but it reaches zero</div>
            <Trace ariaLabel="number of misclassified points after each perceptron update" xLabel="update l" xMax={Math.max(6, steps.length)} series={[{ pts: trace, color: C.pink, dots: true }]} yDomain={[0, Math.max(...trace, 1) + 1]} />
          </div>
          <Legend items={[{ label: "class +1", color: POS }, { label: "class −1", color: NEG }, { label: "point used in the last update and the change in w", color: C.amber }, { label: "currently misclassified (ringed)", color: C.fg }]} />
        </div>
      }
      controls={
        <>
          <AnimationControls playing={playing && !done} onPlay={() => { if (done) setL(0); setPlaying(!playing); }} onStep={() => setL(k + 1)} canStep={!done} onReset={() => { setPlaying(false); setL(0); }} />
          <ParameterSlider label="update number l" value={k} min={0} max={steps.length} step={1} onChange={(v) => { setPlaying(false); setL(v); }} />
          <ParameterSlider label="gap built into the data" value={margin} min={0.08} max={1.5} step={0.02} onChange={(v) => { setMargin(v); setL(0); setPlaying(false); }} />
          <button className="btn w-full" onClick={() => { setSeed((s) => s + 1); setL(0); setPlaying(false); }}>New sample</button>
          <p className="text-[12.5px] leading-relaxed text-muted">{k === 0 ? "Start: w = 0 predicts +1 everywhere (sign(0) is taken as +1), so every −1 point is a mistake." : done ? `Converged after ${steps.length} update${steps.length === 1 ? "" : "s"}: every point is on the correct side.` : "The points are visited in a fixed cyclic order; the first misclassified one triggers the next update."}</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{updates so far}`, tex: true, value: `${k} of ${steps.length}` },
        { label: String.raw`\mathbf w`, tex: true, value: `(${fmt(w[0], 3)}, ${fmt(w[1], 3)})` },
        { label: String.raw`\|\mathbf w\|`, tex: true, value: fmt(Math.hypot(w[0], w[1]), 3) },
        { label: String.raw`\text{misclassified now}`, tex: true, value: mistakes, color: mistakes ? C.pink : C.green },
        { label: String.raw`\text{status}`, tex: true, value: done ? "converged" : "running", color: done ? C.green : undefined },
      ]} />}
      calc={lp && last ? <CalcSteps title={`Update ${k}`} steps={[
        { tex: String.raw`\text{mistake on }\mathbf x_i=${texMatrix([[lp.x[0]], [lp.x[1]]], 2)},\ y_i=${lp.y > 0 ? "+1" : "-1"}:\qquad y_i\,\mathbf w^\top\mathbf x_i=${tex(last.before, 3)}\le0` },
        { tex: String.raw`\mathbf w\leftarrow\mathbf w+y_i\mathbf x_i=${texMatrix([[prev[0]], [prev[1]]], 2)}${lp.y > 0 ? "+" : "-"}${texMatrix([[lp.x[0]], [lp.x[1]]], 2)}=${texMatrix([[w[0]], [w[1]]], 2)}` },
        { tex: String.raw`y_i\,\mathbf w_{\text{new}}^\top\mathbf x_i=y_i\,\mathbf w^\top\mathbf x_i+\|\mathbf x_i\|^2=${tex(last.before, 3)}+${tex(lp.x[0] ** 2 + lp.x[1] ** 2, 3)}=${tex(lp.y * dotw(w, lp.x), 3)}`, note: lp.y * dotw(w, lp.x) > 0 ? "this point is now classified correctly" : "better than before, but still on the wrong side" },
      ]} /> : <CalcSteps steps={[{ tex: String.raw`\mathbf w^0=\mathbf 0;\quad\text{while some }i\text{ has }\operatorname{sign}(\mathbf w^\top\mathbf x_i)\neq y_i:\quad\mathbf w\leftarrow\mathbf w+y_i\mathbf x_i` }]} />}
      note="Each update rotates w towards a misclassified positive point or away from a misclassified negative one. One update can break points that were correct before, which is why the mistake count wiggles on its way down."
    />
  );
}

/* ------------------------------------------------------------------ the mistake bound R²/γ² */
function PerceptronBound() {
  const [margin, setMargin] = useState(0.35);
  const [seed, setSeed] = useState(6);
  const D = useMemo(() => separableData(seed, 12, margin, 0.8), [seed, margin]);
  const steps = useMemo(() => runPerceptron(D), [D]);
  const opt = useMemo(() => maxMarginOrigin(D.map((p) => p.x), D.map((p) => p.y)), [D]);
  const R = Math.max(...D.map((p) => Math.hypot(p.x[0], p.x[1])));
  const g = opt.gamma, ws = opt.wUnit as V2;
  const bound = (R * R) / (g * g);
  const m = steps.length;
  const norm2 = [0, ...steps.map((s) => s.w[0] ** 2 + s.w[1] ** 2)];
  const align = [0, ...steps.map((s) => dotw(s.w, ws))];
  const wF: V2 = m ? steps[m - 1].w : [0, 0];
  const cos = m ? align[m] / Math.sqrt(norm2[m]) : 0;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div className="grid gap-3 md:grid-cols-[1.15fr_1fr]">
            <Plot x={[-5, 5]} y={[-4.2, 4.2]} equal originAxes aspect={0.86} maxH={380} ariaLabel="data, maximum-margin direction and the perceptron solution">
              <LineWB w={ws} style={{ stroke: C.a("green", 0.9), strokeWidth: 1.4, strokeDasharray: "7 4" }} />
              <LineWB w={ws} b={-g} style={{ stroke: C.a("green", 0.4), strokeWidth: 1, strokeDasharray: "3 4" }} />
              <LineWB w={ws} b={g} style={{ stroke: C.a("green", 0.4), strokeWidth: 1, strokeDasharray: "3 4" }} />
              {m > 0 && <LineWB w={wF} />}
              <Polyline pts={Array.from({ length: 73 }, (_, i) => [R * Math.cos((i * Math.PI) / 36), R * Math.sin((i * Math.PI) / 36)] as V2)} style={{ stroke: C.a("amber", 0.6), strokeWidth: 1, strokeDasharray: "2 4" }} />
              {D.map((p, j) => <Dot key={j} p={p.x} r={4} color={p.y > 0 ? POS : NEG} stroke={C.bg} />)}
              <Arrow from={[0, 0]} to={[ws[0] * 1.6, ws[1] * 1.6]} color="green" label="w*" />
            </Plot>
            <div className="space-y-3">
              <div>
                <div className="mb-1 text-[12px] text-faint">Upper bound: ‖w<sub>l</sub>‖² (pink) stays below l·R² (dashed)</div>
                <Trace ariaLabel="squared norm of w and its upper bound" xLabel="mistakes l" xMax={Math.max(4, m)} series={[{ pts: norm2, color: C.pink, dots: m < 40 }, { pts: norm2.map((_, i) => i * R * R), color: C.a("fg", 0.45), dashed: true }]} />
              </div>
              <div>
                <div className="mb-1 text-[12px] text-faint">Lower bound: w<sub>l</sub>ᵀw* (green) stays above l·γ (dashed)</div>
                <Trace ariaLabel="alignment with w star and its lower bound" xLabel="mistakes l" xMax={Math.max(4, m)} series={[{ pts: align, color: C.green, dots: m < 40 }, { pts: align.map((_, i) => i * g), color: C.a("fg", 0.45), dashed: true }]} />
              </div>
            </div>
          </div>
          <Legend items={[{ label: "maximum-margin separator w* and its margin γ", color: C.green, dashed: true }, { label: "separator found by the perceptron", color: C.fg }, { label: "circle of radius R", color: C.amber, dashed: true }]} />
          {m > bound + 1e-9 && <Warn>The mistake count exceeds the bound — this should be impossible and would indicate a bug.</Warn>}
        </div>
      }
      controls={
        <>
          <ParameterSlider label="gap built into the data" value={margin} min={0.05} max={1.5} step={0.01} onChange={setMargin} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New sample</button>
          <p className="text-[12.5px] leading-relaxed text-muted">Shrink the gap: γ falls, the bound R²/γ² rises like 1/γ², and the perceptron typically needs more updates. The bound holds for every dataset and every visiting order; the actual count is usually far below it.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{mistakes made}`, tex: true, value: m, color: C.pink },
        { label: String.raw`\text{bound }R^2/\gamma^2`, tex: true, value: fmt(bound, 2), color: C.green },
        { label: "R", tex: true, value: fmt(R, 4), color: C.amber },
        { label: String.raw`\gamma`, tex: true, value: fmt(g, 4) },
        { label: String.raw`\cos\angle(\mathbf w_l,\mathbf w^*)`, tex: true, value: fmt(cos, 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\|\mathbf w_l\|^2=\|\mathbf w_{l-1}\|^2+2\,y\,\mathbf w_{l-1}^\top\mathbf x+\|\mathbf x\|^2\le\|\mathbf w_{l-1}\|^2+R^2\ \Rightarrow\ \|\mathbf w_l\|^2\le lR^2`, note: `here: ${fmt(norm2[m], 2)} ≤ ${m} × ${fmt(R * R, 2)} = ${fmt(m * R * R, 2)}` },
        { tex: String.raw`\mathbf w_l^\top\mathbf w^*=\mathbf w_{l-1}^\top\mathbf w^*+y\,\mathbf x^\top\mathbf w^*\ge\mathbf w_{l-1}^\top\mathbf w^*+\gamma\ \Rightarrow\ \mathbf w_l^\top\mathbf w^*\ge l\gamma`, note: `here: ${fmt(align[m], 2)} ≥ ${m} × ${fmt(g, 3)} = ${fmt(m * g, 2)}` },
        { tex: String.raw`l^2\gamma^2\le(\mathbf w_l^\top\mathbf w^*)^2\le\|\mathbf w_l\|^2\|\mathbf w^*\|^2\le lR^2\ \Rightarrow\ l\le\frac{R^2}{\gamma^2}=\frac{${tex(R * R, 2)}}{${tex(g * g, 4)}}=${tex(bound, 2)}`, note: `actual mistakes: ${m}` },
      ]} />}
      note="Both traces are measured from the actual run. The norm grows at most linearly while the alignment with w* grows at least linearly; a vector cannot keep getting longer along w* faster than its own length allows, and that contradiction caps the number of updates."
    />
  );
}

export default function Perceptron({ mode = "run" }: { mode?: "run" | "bound" }) {
  return mode === "bound" ? <PerceptronBound /> : <PerceptronRun />;
}

/* ------------------------------------------------------------------ logistic regression: the sigmoid */
function SigmoidViz() {
  const [w, setW] = useState(1.2);
  const [b, setB] = useState(-0.5);
  const [q, setQ] = useState(1.5);
  const D = useMemo(() => { const r = mulberry32(31); return Array.from({ length: 22 }, () => { const x = r() * 8 - 4; return { x, y: r() < sigmoid(1.6 * x - 0.4) ? 1 : 0 }; }); }, []);
  const z = w * q + b, pq = sigmoid(z);
  const ll = D.reduce((s, d) => { const p = sigmoid(w * d.x + b); return s + (d.y ? Math.log(Math.max(1e-12, p)) : Math.log(Math.max(1e-12, 1 - p))); }, 0);
  const acc = D.filter((d) => (sigmoid(w * d.x + b) >= 0.5 ? 1 : 0) === d.y).length / D.length;
  const bd = Math.abs(w) > 1e-6 ? -b / w : NaN;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-4.5, 4.5]} y={[-0.12, 1.12]} aspect={0.46} maxH={340} xLabel="x" yLabel="P(y = 1 | x)" ticks={6} ariaLabel="sigmoid probability curve over one feature" onPointerDown={(p) => setQ(Number(p[0].toFixed(2)))}>
            <Polyline pts={[[-9, 0.5], [9, 0.5]]} style={{ stroke: C.a("fg", 0.25), strokeWidth: 1, strokeDasharray: "3 4" }} />
            <FnPath f={(x) => sigmoid(w * x + b)} n={400} style={{ stroke: C.blue, strokeWidth: 2.4 }} />
            {D.map((d, i) => <Polyline key={`r${i}`} pts={[[d.x, d.y], [d.x, sigmoid(w * d.x + b)]]} style={{ stroke: C.a(d.y ? "pink" : "green", 0.35), strokeWidth: 1 }} />)}
            {D.map((d, i) => <Dot key={i} p={[d.x, d.y]} r={4.2} color={d.y ? C.pink : C.green} stroke={C.bg} />)}
            {Number.isFinite(bd) && <VLine x={bd} color={C.a("fg", 0.6)} label="wx + b = 0" />}
            <Dot p={[q, pq]} r={5.5} color={C.amber} stroke={C.bg} />
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">The linear score z = wx + b that is fed into the sigmoid (also the log-odds)</div>
            <Plot x={[-4.5, 4.5]} y={[-8, 8]} aspect={0.22} minH={110} maxH={150} xLabel="x" yLabel="z" ticks={4} ariaLabel="linear score before the sigmoid">
              <Polyline pts={[[-9, 0], [9, 0]]} style={{ stroke: C.a("fg", 0.25), strokeWidth: 1 }} />
              <FnPath f={(x) => w * x + b} style={{ stroke: C.a("blue", 0.8), strokeWidth: 1.8 }} />
              <Dot p={[q, z]} r={4.5} color={C.amber} stroke={C.bg} />
            </Plot>
          </div>
          <Legend items={[{ label: "σ(wx + b)", color: C.blue }, { label: "examples with y = 1", color: C.pink }, { label: "examples with y = 0", color: C.green }, { label: "query point (click the plot)", color: C.amber }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="weight w" value={w} min={-5} max={5} step={0.05} onChange={setW} />
          <ParameterSlider label="bias b" value={b} min={-5} max={5} step={0.05} onChange={setB} />
          <ParameterSlider label="query x" value={q} min={-4.5} max={4.5} step={0.05} onChange={setQ} />
          <p className="text-[12.5px] leading-relaxed text-muted">|w| sets how sharply the probability switches from 0 to 1; b slides the switching point. As |w| grows the curve approaches the hard step of the perceptron.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "z=wx+b", tex: true, value: fmt(z, 4) },
        { label: String.raw`\sigma(z)=P(y=1\mid x)`, tex: true, value: fmt(pq, 4), color: C.amber },
        { label: String.raw`\text{odds }\tfrac{P}{1-P}=e^{z}`, tex: true, value: fmt(Math.exp(z), 4) },
        { label: String.raw`\sigma'(z)=\sigma(1-\sigma)`, tex: true, value: fmt(pq * (1 - pq), 4) },
        { label: String.raw`\ell(w,b)\ \text{on the data}`, tex: true, value: fmt(ll, 3) },
        { label: String.raw`\text{accuracy at threshold }0.5`, tex: true, value: fmt(acc, 3) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`z=wx+b=(${tex(w, 2)})(${tex(q, 2)})+(${tex(b, 2)})=${tex(z, 4)}` },
        { tex: String.raw`\sigma(z)=\frac{1}{1+e^{-z}}=\frac{1}{1+${tex(Math.exp(-z), 4)}}=${tex(pq, 4)},\qquad\sigma(-z)=1-\sigma(z)=${tex(1 - pq, 4)}` },
        { tex: String.raw`\log\frac{P(y=1\mid x)}{P(y=0\mid x)}=\log\frac{${tex(pq, 4)}}{${tex(1 - pq, 4)}}=${tex(Math.log(pq / (1 - pq)), 4)}=z`, note: "the log-odds are linear in x" },
      ]} />}
      note="Logistic regression keeps the linear score of the perceptron but passes it through a smooth, monotone squashing function, so the model outputs a probability instead of a hard label and its fit can be judged by likelihood."
    />
  );
}

/* ------------------------------------------------------------------ gradient ascent on the log-likelihood */
function logisticData(kind: "overlap" | "separable", seed: number) {
  if (kind === "separable") return separableData(seed, 12, 0.5, 0.9).map((p) => ({ x: p.x, y: p.y > 0 ? 1 : 0 }));
  const r = mulberry32(seed * 7 + 5);
  return Array.from({ length: 30 }, (_, i) => { const c = i % 2; return { x: [(c ? 1 : -1) * 0.9 + randn(r) * 1.1, (c ? 1 : -1) * 0.6 + randn(r) * 1.1] as V2, y: c }; });
}

function AscentViz() {
  const [kind, setKind] = useState<"overlap" | "separable">("overlap");
  const [eta, setEta] = useState(0.05);
  const [seed, setSeed] = useState(1);
  const [T, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [sel, setSel] = useState(0);
  const th = useThemeRGB();
  const MAXT = 200;
  const D = useMemo(() => logisticData(kind, seed), [kind, seed]);
  const run = useMemo(() => {
    const ws: V2[] = [[0, 0]], gs: V2[] = [], lls: number[] = [];
    const llOf = (w: V2) => D.reduce((s, d) => { const z = dotw(w, d.x); return s + d.y * z - (z > 30 ? z : Math.log(1 + Math.exp(z))); }, 0);
    for (let t = 0; t <= MAXT; t++) {
      const w = ws[t];
      const g: V2 = [0, 0];
      for (const d of D) { const e = d.y - sigmoid(dotw(w, d.x)); g[0] += e * d.x[0]; g[1] += e * d.x[1]; }
      gs.push(g); lls.push(llOf(w));
      ws.push([w[0] + eta * g[0], w[1] + eta * g[1]]);
    }
    return { ws, gs, lls };
  }, [D, eta]);
  useTicker(playing, 70, () => setT((t) => { if (t >= MAXT) { setPlaying(false); return MAXT; } return t + 1; }));
  const w = run.ws[T], g = run.gs[T];
  const i = Math.min(sel, D.length - 1), d = D[i];
  const pi = sigmoid(dotw(w, d.x));
  const acc = D.filter((p) => (dotw(w, p.x) >= 0 ? 1 : 0) === p.y).length / D.length;
  const norms = run.ws.slice(0, MAXT + 1).map((v) => Math.hypot(v[0], v[1]));
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-4.6, 4.6]} y={[-3.9, 3.9]} equal originAxes aspect={0.72} maxH={420} ariaLabel="logistic regression probability map during gradient ascent"
            onPointerDown={(q) => { let b = 0, bd = Infinity; D.forEach((p, j) => { const dd = (p.x[0] - q[0]) ** 2 + (p.x[1] - q[1]) ** 2; if (dd < bd) { bd = dd; b = j; } }); setSel(b); }}
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={4} deps={[w[0], w[1], th.blue[0], th.panel[0]]} f={(x, y) => { const p = sigmoid(w[0] * x + w[1] * y), c = p >= 0.5 ? th.pink : th.blue, a = 0.06 + 0.3 * Math.abs(p - 0.5) * 2, bg = th.panel; return [bg[0] + (c[0] - bg[0]) * a, bg[1] + (c[1] - bg[1]) * a, bg[2] + (c[2] - bg[2]) * a, 255]; }} />}>
            {Math.hypot(w[0], w[1]) > 1e-9 && <LineWB w={w} />}
            {D.map((p, j) => <Dot key={j} p={p.x} r={4.2} color={p.y ? POS : NEG} stroke={j === i ? C.amber : C.bg} />)}
            <Arrow from={[0, 0]} to={[w[0], w[1]]} color="fg" label="w" />
            <Arrow from={w} to={[w[0] + eta * g[0], w[1] + eta * g[1]]} color="amber" label="η∇ℓ" />
          </Plot>
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <div className="mb-1 text-[12px] text-faint">Log-likelihood ℓ(wₜ)</div>
              <Trace ariaLabel="log-likelihood per gradient ascent step" xLabel="step t" xMax={MAXT} marker={T} series={[{ pts: run.lls, color: C.green }]} yDomain={[Math.min(...run.lls) - 0.5, Math.min(0, Math.max(...run.lls) + 0.5)]} />
            </div>
            <div>
              <div className="mb-1 text-[12px] text-faint">Length of the weight vector ‖wₜ‖</div>
              <Trace ariaLabel="norm of the weights per step" xLabel="step t" xMax={MAXT} marker={T} series={[{ pts: norms, color: C.blue }]} />
            </div>
          </div>
          <Legend items={[{ label: "y = 1", color: POS }, { label: "y = 0", color: NEG }, { label: "boundary P = ½; shading = predicted probability", color: C.fg }, { label: "next step η∇ℓ", color: C.amber }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Dataset" value={kind} onChange={(v) => { setKind(v); setT(0); setPlaying(false); }} options={[{ value: "overlap", label: "Overlapping" }, { value: "separable", label: "Separable" }]} />
          <ParameterSlider label="step size η" value={eta} min={0.005} max={0.3} step={0.005} onChange={setEta} />
          <ParameterSlider label="step t" value={T} min={0} max={MAXT} step={1} onChange={(v) => { setPlaying(false); setT(v); }} />
          <AnimationControls playing={playing} onPlay={() => { if (!playing && T >= MAXT) setT(0); setPlaying(!playing); }} onStep={() => setT((t) => Math.min(MAXT, t + 1))} onReset={() => { setPlaying(false); setT(0); }} />
          <button className="btn w-full" onClick={() => { setSeed((s) => s + 1); setT(0); setPlaying(false); }}>New sample</button>
          {kind === "separable" && <p className="text-[12.5px] leading-relaxed text-muted">With separable data the likelihood has no finite maximiser: scaling w up always raises it, so ‖w‖ keeps growing. A penalty on ‖w‖² fixes this.</p>}
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\mathbf w_t`, tex: true, value: `(${fmt(w[0], 3)}, ${fmt(w[1], 3)})` },
        { label: String.raw`\ell(\mathbf w_t)`, tex: true, value: fmt(run.lls[T], 4), color: C.green },
        { label: String.raw`\nabla\ell(\mathbf w_t)`, tex: true, value: `(${fmt(g[0], 3)}, ${fmt(g[1], 3)})`, color: C.amber },
        { label: String.raw`\|\nabla\ell\|`, tex: true, value: fmt(Math.hypot(g[0], g[1]), 4) },
        { label: String.raw`\text{training accuracy}`, tex: true, value: fmt(acc, 3) },
      ]} />}
      calc={<CalcSteps title="Live calculation — contribution of the selected point (amber ring)" steps={[
        { tex: String.raw`\sigma(\mathbf w^\top\mathbf x_i)=\sigma(${tex(dotw(w, d.x), 3)})=${tex(pi, 4)},\qquad y_i=${d.y}\ \Rightarrow\ y_i-\sigma_i=${tex(d.y - pi, 4)}` },
        { tex: String.raw`(y_i-\sigma_i)\,\mathbf x_i=${tex(d.y - pi, 4)}${texMatrix([[d.x[0]], [d.x[1]]], 2)}=${texMatrix([[(d.y - pi) * d.x[0]], [(d.y - pi) * d.x[1]]], 3)}`, note: "confidently correct points contribute almost nothing" },
        { tex: String.raw`\nabla\ell(\mathbf w)=\sum_{i=1}^{n}(y_i-\sigma_i)\,\mathbf x_i=${texMatrix([[g[0]], [g[1]]], 3)},\qquad\mathbf w\leftarrow\mathbf w+\eta\nabla\ell(\mathbf w)` },
      ]} />}
      note="The gradient is a sum of the data points weighted by their errors y − σ: each misfit point pulls w towards itself (y = 1) or pushes it away (y = 0). The log-likelihood is concave, so gradient ascent cannot get stuck in a poor local maximum."
    />
  );
}

export function Logistic({ mode = "sigmoid" }: { mode?: "sigmoid" | "ascent" }) {
  return mode === "ascent" ? <AscentViz /> : <SigmoidViz />;
}
