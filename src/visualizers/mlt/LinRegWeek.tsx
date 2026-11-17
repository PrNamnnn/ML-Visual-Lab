"use client";
import { useMemo, useState } from "react";
import { Plot, Arrow, Dot, Handle, Polyline, FnPath, C, type PlotCtx } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend, Warn } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, AnimationControls, Presets, Select } from "@/components/controls";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { useTicker } from "@/hooks/useSize";
import { mulberry32, randn } from "@/lib/math/random";
import { pinv, matVec, solve } from "@/lib/math/linalg";
import { fmt, tex, texMatrix, clamp } from "@/lib/utilities/format";
import { Heat, BarList } from "@/visualizers/shared";
import { Trace, ContourField, eigSym2, type V2 } from "./common";

type Mode = "supervised" | "normal" | "gd" | "views";
const WORKED: V2[] = [[1, 1], [2, 2], [3, 2]];

/** Sufficient statistics of 1-D data with a bias feature: X is 2 × n with rows (1, …, 1) and (x₁, …, xₙ). */
function stats(D: V2[]) {
  const n = D.length;
  const sx = D.reduce((s, p) => s + p[0], 0), sxx = D.reduce((s, p) => s + p[0] * p[0], 0);
  const sy = D.reduce((s, p) => s + p[1], 0), sxy = D.reduce((s, p) => s + p[0] * p[1], 0);
  const A: [[number, number], [number, number]] = [[n, sx], [sx, sxx]];
  const b: V2 = [sy, sxy];
  const det = n * sxx - sx * sx;
  const w: V2 | null = Math.abs(det) > 1e-10 ? [(sxx * sy - sx * sxy) / det, (n * sxy - sx * sy) / det] : null;
  const loss = (w0: number, w1: number) => 0.5 * D.reduce((s, p) => s + (w0 + w1 * p[0] - p[1]) ** 2, 0);
  const grad = (w0: number, w1: number): V2 => [A[0][0] * w0 + A[0][1] * w1 - b[0], A[1][0] * w0 + A[1][1] * w1 - b[1]];
  return { n, A, b, det, w, loss, grad };
}

/** Shaded loss surface with iso-lines at equal steps of sqrt(L - L_min): evenly spaced rings for a quadratic bowl. */
function LossField({ ctx, loss, min, scale, deps }: { ctx: PlotCtx; loss: (a: number, b: number) => number; min: number; scale: number; deps: unknown[] }) {
  return <ContourField ctx={ctx} f={loss} level={(v) => Math.sqrt(Math.max(0, v - min)) / scale} deps={[...deps, min, scale]} />;
}

/* ------------------------------------------------------------------ regression versus classification */
function SupervisedViz() {
  const [task, setTask] = useState<"regression" | "classification">("regression");
  const [w, setW] = useState(0.2);
  const [b, setB] = useState(2.5);
  const [t, setT] = useState(3);
  const data = useMemo(() => {
    const r = mulberry32(21);
    return Array.from({ length: 16 }, (_, i) => { const x = 0.5 + (9 * (i + r() * 0.8)) / 16; return { x, y: 1 + 0.55 * x + randn(r) * 0.7, c: (x + randn(r) * 1.3 > 5 ? 1 : -1) as 1 | -1 }; });
  }, []);
  const D: V2[] = data.map((d) => [d.x, d.y]);
  const st = stats(D);
  const sse = st.loss(b, w);
  const errAt = (thr: number) => data.filter((d) => (d.x > thr ? 1 : -1) !== d.c).length;
  const err = errAt(t);
  const errCurve = Array.from({ length: 101 }, (_, i) => errAt(i / 10) / data.length);
  const bestT = errCurve.indexOf(Math.min(...errCurve)) / 10;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          {task === "regression" ? (
            <Plot x={[0, 10]} y={[-1, 9]} aspect={0.55} maxH={400} xLabel="feature x" yLabel="label y ∈ ℝ" ariaLabel="regression: real-valued labels and a fitted line">
              {data.map((d, i) => <Polyline key={`r${i}`} pts={[[d.x, d.y], [d.x, b + w * d.x]]} style={{ stroke: C.a("pink", 0.7), strokeWidth: 1.2 }} />)}
              <FnPath f={(x) => b + w * x} style={{ stroke: C.blue, strokeWidth: 2.2 }} />
              {st.w && <FnPath f={(x) => st.w![0] + st.w![1] * x} style={{ stroke: C.a("green", 0.7), strokeWidth: 1.2, strokeDasharray: "5 4" }} />}
              {data.map((d, i) => <Dot key={i} p={[d.x, d.y]} r={4} color={C.fg} stroke={C.bg} />)}
            </Plot>
          ) : (
            <>
              <Plot x={[0, 10]} y={[-1.6, 1.6]} aspect={0.34} minH={170} maxH={250} xLabel="feature x" yLabel="label y ∈ {−1, +1}" ticks={4} ariaLabel="classification: labels in a finite set and a threshold classifier">
                <Polyline pts={[[0, -1], [t, -1], [t, 1], [10, 1]]} style={{ stroke: C.blue, strokeWidth: 2.2 }} />
                {data.map((d, i) => { const bad = (d.x > t ? 1 : -1) !== d.c; return <Dot key={i} p={[d.x, d.c]} r={bad ? 6 : 4.5} color={d.c > 0 ? C.pink : C.green} stroke={bad ? C.fg : C.bg} />; })}
              </Plot>
              <div>
                <div className="mb-1 text-[12px] text-faint">Fraction misclassified as the threshold moves — a staircase, flat almost everywhere</div>
                <Trace ariaLabel="zero-one error versus threshold" xLabel="threshold × 10" series={[{ pts: errCurve, color: C.pink }]} marker={t * 10} yDomain={[0, 0.7]} />
              </div>
            </>
          )}
          <Legend items={task === "regression" ? [{ label: "model f(x) = b + w·x", color: C.blue }, { label: "residuals f(xᵢ) − yᵢ", color: C.pink }, { label: "least-squares line", color: C.green, dashed: true }] : [{ label: "class +1", color: C.pink }, { label: "class −1", color: C.green }, { label: "classifier h(x) = sign(x − t); mistakes are ringed", color: C.blue }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Supervised task" value={task} onChange={setTask} options={[{ value: "regression", label: "Regression" }, { value: "classification", label: "Classification" }]} />
          {task === "regression" ? (
            <>
              <ParameterSlider label="slope w" value={w} min={-1} max={2} step={0.01} onChange={setW} />
              <ParameterSlider label="intercept b" value={b} min={-2} max={8} step={0.05} onChange={setB} />
              <button className="btn w-full" onClick={() => { if (st.w) { setB(Number(st.w[0].toFixed(2))); setW(Number(st.w[1].toFixed(2))); } }}>Minimise the loss</button>
            </>
          ) : (
            <>
              <ParameterSlider label="threshold t" value={t} min={0} max={10} step={0.1} onChange={setT} />
              <button className="btn w-full" onClick={() => setT(bestT)}>Minimise the error</button>
            </>
          )}
          <p className="text-[12.5px] leading-relaxed text-muted">Same inputs, two kinds of label. The label set decides the loss: squared error for real-valued labels, a count of mistakes for categories.</p>
        </>
      }
      readouts={<Readouts items={task === "regression" ? [
        { label: String.raw`\text{SSE}=\tfrac12\sum(f(x_i)-y_i)^2`, tex: true, value: fmt(sse, 3), color: C.pink },
        { label: String.raw`\text{RMSE}`, tex: true, value: fmt(Math.sqrt((2 * sse) / data.length), 3) },
        { label: String.raw`\text{minimum SSE}`, tex: true, value: st.w ? fmt(st.loss(st.w[0], st.w[1]), 3) : "—", color: C.green },
        { label: "n", tex: true, value: data.length },
      ] : [
        { label: String.raw`\text{mistakes}`, tex: true, value: `${err} of ${data.length}`, color: C.pink },
        { label: String.raw`\text{0-1 error}`, tex: true, value: fmt(err / data.length, 3) },
        { label: String.raw`\text{minimum error}`, tex: true, value: fmt(Math.min(...errCurve), 3), color: C.green },
        { label: String.raw`\text{accuracy}`, tex: true, value: fmt(1 - err / data.length, 3) },
      ]} />}
      calc={<CalcSteps steps={task === "regression" ? [
        { tex: String.raw`D=\{(\mathbf x_i,y_i)\}_{i=1}^{n},\quad y_i\in\mathbb R,\quad f(x)=${tex(b, 2)}+${tex(w, 2)}\,x` },
        { tex: String.raw`L(f)=\frac12\sum_{i=1}^{${data.length}}(f(x_i)-y_i)^2=${tex(sse, 3)}`, note: "smooth in (w, b): gradients are available" },
      ] : [
        { tex: String.raw`D=\{(\mathbf x_i,y_i)\}_{i=1}^{n},\quad y_i\in\{-1,+1\},\quad h(x)=\operatorname{sign}(x-${tex(t, 1)})` },
        { tex: String.raw`L(h)=\frac1n\sum_{i=1}^{${data.length}}\mathbb 1[h(x_i)\neq y_i]=\frac{${err}}{${data.length}}=${tex(err / data.length, 3)}`, note: "piecewise constant in t: the gradient is zero wherever it exists" },
      ]} />}
      note="Both problems learn a function from labelled examples and are judged on unseen data. Regression can be solved with calculus; the zero-one error needs different tools, which is where Weeks 7 to 12 go."
    />
  );
}

/* ------------------------------------------------------------------ normal equations and the loss surface */
function NormalViz() {
  const [D, setD] = useState<V2[]>(WORKED);
  const [cand, setCand] = useState<V2>([2, -0.3]);
  const st = useMemo(() => stats(D), [D]);
  const g = st.grad(cand[0], cand[1]);
  const Lc = st.loss(cand[0], cand[1]);
  const Lmin = st.w ? st.loss(st.w[0], st.w[1]) : 0;
  const X = [D.map(() => 1), D.map((p) => p[0])];
  const gn = Math.hypot(g[0], g[1]) || 1;
  const inv = st.w ? [[st.A[1][1] / st.det, -st.A[0][1] / st.det], [-st.A[1][0] / st.det, st.A[0][0] / st.det]] : null;
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <div className="mb-1 text-[12px] text-faint">Data space — drag the points</div>
              <Plot x={[0, 4.5]} y={[-0.5, 4]} aspect={0.95} maxH={360} xLabel="x" yLabel="y" ariaLabel="data points with candidate and optimal regression lines">
                {D.map((p, i) => <Polyline key={`r${i}`} pts={[p, [p[0], cand[0] + cand[1] * p[0]]]} style={{ stroke: C.a("pink", 0.75), strokeWidth: 1.2 }} />)}
                <FnPath f={(x) => cand[0] + cand[1] * x} style={{ stroke: C.pink, strokeWidth: 2 }} />
                {st.w && <FnPath f={(x) => st.w![0] + st.w![1] * x} style={{ stroke: C.green, strokeWidth: 2 }} />}
                {D.map((p, i) => <Handle key={i} p={p} color="blue" label={`data point ${i + 1}`} onDrag={(q) => setD(D.map((o, j) => (j === i ? [clamp(q[0], 0.1, 4.4), clamp(q[1], -0.4, 3.9)] : o)))} />)}
              </Plot>
            </div>
            <div>
              <div className="mb-1 text-[12px] text-faint">Parameter space — drag the candidate w</div>
              <Plot x={[-2, 3.5]} y={[-1.5, 2.5]} aspect={0.95} maxH={360} xLabel="w₀ (intercept)" yLabel="w₁ (slope)" ariaLabel="contours of the squared loss over the two weights"
                underlay={(ctx) => <LossField ctx={ctx} loss={st.loss} min={Lmin} scale={0.9} deps={[D]} />}>
                <Arrow from={cand} to={[cand[0] - (0.7 * g[0]) / gn, cand[1] - (0.7 * g[1]) / gn]} color="amber" label="−∇L" />
                {st.w && <Dot p={st.w} r={6} color={C.green} stroke={C.bg} />}
                <Handle p={cand} color="pink" label="candidate weights" onDrag={(q) => setCand([clamp(q[0], -1.9, 3.4), clamp(q[1], -1.4, 2.4)])} />
              </Plot>
            </div>
          </div>
          <Legend items={[{ label: "candidate w and its line", color: C.pink }, { label: "solution w* of the normal equations", color: C.green }, { label: "negative gradient (direction of steepest descent)", color: C.amber }]} />
          {!st.w && <Warn>All x values coincide, so XXᵀ is singular: infinitely many lines fit equally well. The pseudoinverse would return the minimum-norm one.</Warn>}
        </div>
      }
      controls={
        <>
          <div className="flex flex-wrap items-start gap-4">
            <MatrixView M={X} label={<span>X (2 × {D.length})</span>} digits={2} cell={44} heat={false} />
            <MatrixView M={st.A} label={<span>XXᵀ</span>} digits={2} cell={50} />
            <MatrixView M={[[st.b[0]], [st.b[1]]]} label={<span>Xy</span>} digits={2} cell={50} />
          </div>
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={() => setD([...D, [Number((0.6 + 3.2 * ((D.length * 0.37) % 1)).toFixed(2)), Number((0.8 + 2 * ((D.length * 0.61) % 1)).toFixed(2))]])} disabled={D.length >= 8}>Add point</button>
            <button className="btn flex-1" onClick={() => setD(D.slice(0, -1))} disabled={D.length <= 2}>Remove</button>
          </div>
          <Presets presets={[{ label: "Worked example", value: WORKED }, { label: "Steep", value: [[1, 0.4], [1.8, 1.9], [2.6, 3.4]] as V2[] }, { label: "With an outlier", value: [[0.8, 0.9], [1.6, 1.4], [2.4, 1.9], [3.2, 2.4], [3.9, 0.3]] as V2[] }]} onPick={setD} />
          <button className="btn w-full" onClick={() => st.w && setCand(st.w)}>Move candidate to w*</button>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`L(\mathbf w)`, tex: true, value: fmt(Lc, 4), color: C.pink },
        { label: String.raw`L(\mathbf w^*)`, tex: true, value: fmt(Lmin, 4), color: C.green },
        { label: String.raw`\nabla L(\mathbf w)`, tex: true, value: `(${fmt(g[0], 3)}, ${fmt(g[1], 3)})`, color: C.amber },
        { label: String.raw`\mathbf w^*`, tex: true, value: st.w ? `(${fmt(st.w[0], 4)}, ${fmt(st.w[1], 4)})` : "not unique", color: C.green },
        { label: String.raw`\det(XX^\top)`, tex: true, value: fmt(st.det, 3) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\nabla L(\mathbf w)=XX^\top\mathbf w-X\mathbf y=${texMatrix(st.A, 2)}${texMatrix([[cand[0]], [cand[1]]], 2)}-${texMatrix([[st.b[0]], [st.b[1]]], 2)}=${texMatrix([[g[0]], [g[1]]], 3)}` },
        ...(st.w && inv ? [
          { tex: String.raw`XX^\top\mathbf w=X\mathbf y\ \Rightarrow\ \mathbf w^*=(XX^\top)^{-1}X\mathbf y=${texMatrix(inv, 3)}${texMatrix([[st.b[0]], [st.b[1]]], 2)}=${texMatrix([[st.w[0]], [st.w[1]]], 4)}`, note: "normal equations" },
          { tex: String.raw`\nabla L(\mathbf w^*)=${texMatrix([[st.grad(st.w[0], st.w[1])[0]], [st.grad(st.w[0], st.w[1])[1]]], 4)}`, note: "the gradient vanishes at the solution" },
        ] : []),
      ]} />}
      note="Each data point contributes one bowl-shaped term to L, so the total is a single convex bowl. Its lowest point is where the gradient is zero, which is exactly the linear system XXᵀw = Xy."
    />
  );
}

/* ------------------------------------------------------------------ gradient descent and SGD */
function GDViz() {
  const [n, setN] = useState(20);
  const [eta, setEta] = useState(0.5);
  const [batch, setBatch] = useState(2);
  const [T, setT] = useState(0);
  const [seed, setSeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const D = useMemo(() => { const r = mulberry32(8); return Array.from({ length: n }, () => { const x = r() * 4 - 2; return [x, 0.8 + 1.1 * x + randn(r) * 0.55] as V2; }); }, [n]);
  const st = useMemo(() => stats(D), [D]);
  const lamMax = eigSym2(st.A).values[0];
  const step = (eta * 2) / lamMax; // slider is a fraction of the stability limit 2/λ_max
  const MAXT = 150;
  const run = useMemo(() => {
    const w0: V2 = [-1.6, -0.9];
    const gd: V2[] = [w0], sgd: V2[] = [w0];
    const r = mulberry32(seed * 91 + batch);
    for (let t = 0; t < MAXT; t++) {
      const a = gd[t], g = st.grad(a[0], a[1]);
      gd.push([a[0] - step * g[0], a[1] - step * g[1]]);
      const s = sgd[t];
      let g0 = 0, g1 = 0;
      for (let j = 0; j < batch; j++) { const p = D[Math.floor(r() * D.length)]; const e = s[0] + s[1] * p[0] - p[1]; g0 += e; g1 += e * p[0]; }
      // scale the mini-batch sum by n/|B| so it is an unbiased estimate of the full gradient
      const k = D.length / batch;
      sgd.push([s[0] - step * k * g0, s[1] - step * k * g1]);
    }
    return { gd, sgd };
  }, [D, st, step, batch, seed]);
  useTicker(playing, 90, () => setT((t) => { if (t >= MAXT) { setPlaying(false); return MAXT; } return t + 1; }));
  const Lmin = st.w ? st.loss(st.w[0], st.w[1]) : 0;
  const cap = (v: number) => (Number.isFinite(v) ? Math.min(v, 1e6) : 1e6);
  const Lgd = run.gd.slice(0, T + 1).map((w) => cap(st.loss(w[0], w[1]) - Lmin));
  const Lsgd = run.sgd.slice(0, T + 1).map((w) => cap(st.loss(w[0], w[1]) - Lmin));
  const diverging = eta >= 1;
  const inView = (w: V2) => Math.abs(w[0]) < 50 && Math.abs(w[1]) < 50;
  const wg = run.gd[T], ws = run.sgd[T];
  const avg: V2 = [run.sgd.slice(Math.floor(T / 2), T + 1).reduce((s, w) => s + w[0], 0) / (T - Math.floor(T / 2) + 1), run.sgd.slice(Math.floor(T / 2), T + 1).reduce((s, w) => s + w[1], 0) / (T - Math.floor(T / 2) + 1)];
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-2.4, 2.8]} y={[-1.6, 2.6]} aspect={0.66} maxH={420} xLabel="w₀" yLabel="w₁" ariaLabel="gradient descent and stochastic gradient descent trajectories on the loss contours"
            underlay={(ctx) => <LossField ctx={ctx} loss={st.loss} min={Lmin} scale={2.2} deps={[D]} />}>
            <Polyline pts={run.sgd.slice(0, T + 1).filter(inView)} style={{ stroke: C.pink, strokeWidth: 1.3 }} />
            <Polyline pts={run.gd.slice(0, T + 1).filter(inView)} style={{ stroke: C.amber, strokeWidth: 2 }} />
            {st.w && <Dot p={st.w} r={6} color={C.green} stroke={C.bg} />}
            {inView(ws) && <Dot p={ws} r={4.5} color={C.pink} stroke={C.bg} />}
            {inView(wg) && <Dot p={wg} r={4.5} color={C.amber} stroke={C.bg} />}
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Excess loss L(wₜ) − L(w*) per step</div>
            <Trace ariaLabel="excess loss per iteration for GD and SGD" xLabel="step t" xMax={MAXT} yDomain={[0, Math.max(1, Math.min(60, Math.max(...Lgd.slice(0, 3), ...Lsgd.slice(0, 3)) * 1.1))]} series={[{ pts: Lsgd, color: C.pink }, { pts: Lgd, color: C.amber }]} />
          </div>
          <Legend items={[{ label: "gradient descent (all n points per step)", color: C.amber }, { label: `SGD (batch of ${batch})`, color: C.pink }, { label: "w* from the normal equations", color: C.green }]} />
          {diverging && <Warn>η is at or beyond 2/λ_max: along the steepest direction the error is multiplied by |1 − ηλ_max| ≥ 1 every step, so gradient descent no longer converges.</Warn>}
        </div>
      }
      controls={
        <>
          <ParameterSlider label="step size η, as a fraction of 2/λ_max" value={eta} min={0.02} max={1.1} step={0.01} onChange={(v) => { setEta(v); }} format={(v) => `${fmt(v, 2)} → η = ${fmt((v * 2) / lamMax, 4)}`} />
          <ParameterSlider label="SGD batch size |B|" value={batch} min={1} max={n} step={1} onChange={setBatch} />
          <ParameterSlider label="data points n" value={n} min={4} max={60} step={1} onChange={(v) => { setN(v); setBatch(Math.min(batch, v)); }} />
          <ParameterSlider label="step t" value={T} min={0} max={MAXT} step={1} onChange={(v) => { setPlaying(false); setT(v); }} />
          <AnimationControls playing={playing} onPlay={() => { if (!playing && T >= MAXT) setT(0); setPlaying(!playing); }} onStep={() => setT((t) => Math.min(MAXT, t + 1))} onReset={() => { setPlaying(false); setT(0); }} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>Resample the SGD batches</button>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\mathbf w_t\ \text{(GD)}`, tex: true, value: `(${fmt(wg[0], 3)}, ${fmt(wg[1], 3)})`, color: C.amber },
        { label: String.raw`\mathbf w_t\ \text{(SGD)}`, tex: true, value: `(${fmt(ws[0], 3)}, ${fmt(ws[1], 3)})`, color: C.pink },
        { label: String.raw`\text{SGD average of last half}`, tex: true, value: `(${fmt(avg[0], 3)}, ${fmt(avg[1], 3)})` },
        { label: String.raw`\mathbf w^*`, tex: true, value: st.w ? `(${fmt(st.w[0], 3)}, ${fmt(st.w[1], 3)})` : "—", color: C.green },
        { label: String.raw`\lambda_{\max}(XX^\top)`, tex: true, value: fmt(lamMax, 3) },
        { label: String.raw`2/\lambda_{\max}`, tex: true, value: fmt(2 / lamMax, 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{GD: }\mathbf w_{t+1}=\mathbf w_t-\eta\,(XX^\top\mathbf w_t-X\mathbf y),\qquad\eta=${tex(step, 4)}`, note: `cost per step: all ${n} points` },
        { tex: String.raw`\text{SGD: }\mathbf w_{t+1}=\mathbf w_t-\eta\,\frac{n}{|B|}\sum_{i\in B}(\mathbf w_t^\top\mathbf x_i-y_i)\,\mathbf x_i,\qquad|B|=${batch}`, note: "random batch: an unbiased but noisy gradient" },
        { tex: String.raw`\mathbf w_{t+1}-\mathbf w^*=(I-\eta XX^\top)(\mathbf w_t-\mathbf w^*)\ \Rightarrow\ \text{converges iff }\eta<\frac{2}{\lambda_{\max}}=${tex(2 / lamMax, 4)}` },
      ]} />}
      note="With a constant step SGD does not settle: it keeps bouncing in a neighbourhood of w* whose size grows with η and shrinks with the batch size. Averaging the later iterates, or decaying η, recovers an accurate answer."
    />
  );
}

/* ------------------------------------------------------------------ projection view and likelihood view */
function ProjectionView() {
  const [a, setA] = useState<V2>([1, 2]);
  const [y, setY] = useState<V2>([2, 3]);
  const [w, setW] = useState(0.8);
  const aa = a[0] * a[0] + a[1] * a[1];
  const wStar = aa > 1e-9 ? (a[0] * y[0] + a[1] * y[1]) / aa : 0;
  const yhat: V2 = [wStar * a[0], wStar * a[1]];
  const cand: V2 = [w * a[0], w * a[1]];
  const res: V2 = [y[0] - yhat[0], y[1] - yhat[1]];
  const err = (ww: number) => (y[0] - ww * a[0]) ** 2 + (y[1] - ww * a[1]) ** 2;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-2, 5]} y={[-1.5, 4.5]} equal originAxes aspect={0.74} maxH={440} xLabel="label of data point 1" yLabel="label of data point 2" ariaLabel="projection of the label vector onto the span of the feature vector">
            <Polyline pts={[[-20 * a[0], -20 * a[1]], [20 * a[0], 20 * a[1]]]} style={{ stroke: C.a("blue", 0.35), strokeWidth: 1.2 }} />
            <Polyline pts={[y, cand]} style={{ stroke: C.a("fg", 0.45), strokeWidth: 1.2, strokeDasharray: "4 3" }} />
            <Polyline pts={[y, yhat]} style={{ stroke: C.pink, strokeWidth: 1.8 }} />
            <Arrow from={[0, 0]} to={a} color="blue" label="feature row (x₁, x₂)" />
            <Arrow from={[0, 0]} to={yhat} color="green" label="ŷ = Xᵀw*" labelOffset={[8, 14]} />
            <Arrow from={[0, 0]} to={y} color="amber" label="y" />
            <Dot p={cand} r={5} color={C.fg} stroke={C.bg} />
            <Handle p={a} color="blue" label="feature vector" onDrag={(q) => setA([clamp(q[0], -1.8, 4.8), clamp(q[1], -1.3, 4.3)])} />
            <Handle p={y} color="amber" label="label vector" onDrag={(q) => setY([clamp(q[0], -1.8, 4.8), clamp(q[1], -1.3, 4.3)])} />
          </Plot>
          <Legend items={[{ label: "all achievable predictions: multiples of the feature row", color: C.a("blue", 0.6) }, { label: "residual y − ŷ, perpendicular to the line", color: C.pink }, { label: "prediction for the candidate w", color: C.fg }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="candidate w" value={w} min={-1} max={3} step={0.01} onChange={setW} />
          <button className="btn w-full" onClick={() => setW(Number(wStar.toFixed(2)))}>Move candidate to w*</button>
          <p className="text-[12.5px] leading-relaxed text-muted">Two data points (n = 2) with one feature (d = 1). The axes are the data points, not the features: this is the ℝⁿ view, where the whole dataset is two vectors.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "w^*", tex: true, value: fmt(wStar, 4), color: C.green },
        { label: String.raw`\|\mathbf y-X^\top w^*\|^2`, tex: true, value: fmt(err(wStar), 4), color: C.pink },
        { label: String.raw`\|\mathbf y-X^\top w\|^2`, tex: true, value: fmt(err(w), 4) },
        { label: String.raw`X(\mathbf y-\hat{\mathbf y})`, tex: true, value: fmt(a[0] * res[0] + a[1] * res[1], 6) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`X=\begin{bmatrix}${tex(a[0], 2)}&${tex(a[1], 2)}\end{bmatrix},\quad\mathbf y=${texMatrix([[y[0]], [y[1]]], 2)},\quad w^*=\frac{X\mathbf y}{XX^\top}=\frac{${tex(a[0] * y[0] + a[1] * y[1], 3)}}{${tex(aa, 3)}}=${tex(wStar, 4)}` },
        { tex: String.raw`\hat{\mathbf y}=X^\top w^*=${texMatrix([[yhat[0]], [yhat[1]]], 3)},\qquad\mathbf y-\hat{\mathbf y}=${texMatrix([[res[0]], [res[1]]], 3)}` },
        { tex: String.raw`X(\mathbf y-\hat{\mathbf y})=${tex(a[0], 2)}\cdot${tex(res[0], 3)}+${tex(a[1], 2)}\cdot${tex(res[1], 3)}=${tex(a[0] * res[0] + a[1] * res[1], 5)}`, note: "orthogonality is the normal equation X(y − Xᵀw) = 0" },
      ]} />}
      note="Least squares picks, among all predictions the model can produce, the one closest to y. The closest point of a subspace is the orthogonal projection, so the residual must be perpendicular to every feature row."
    />
  );
}

function LikelihoodView() {
  const [w, setW] = useState(0.4);
  const [sigma, setSigma] = useState(0.7);
  const D = useMemo(() => { const r = mulberry32(5); return Array.from({ length: 14 }, (_, i) => { const x = 0.4 + i * 0.4; return [x, 0.9 * x + randn(r) * 0.7] as V2; }); }, []);
  const sse = (ww: number) => D.reduce((s, p) => s + (p[1] - ww * p[0]) ** 2, 0);
  const ll = (ww: number) => -0.5 * D.length * Math.log(2 * Math.PI * sigma * sigma) - sse(ww) / (2 * sigma * sigma);
  const wStar = D.reduce((s, p) => s + p[0] * p[1], 0) / D.reduce((s, p) => s + p[0] * p[0], 0);
  const grid = Array.from({ length: 121 }, (_, i) => -0.2 + i * 0.015);
  const lls = grid.map(ll), sses = grid.map((g) => 0.5 * sse(g));
  const mark = Math.round((w + 0.2) / 0.015);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[0, 6.4]} y={[-1.5, 7.5]} aspect={0.5} maxH={340} xLabel="x" yLabel="y" ariaLabel="data with the model line and its Gaussian noise band">
            <Polyline pts={[[0, 2 * sigma], [6.4, w * 6.4 + 2 * sigma], [6.4, w * 6.4 - 2 * sigma], [0, -2 * sigma]]} closed style={{ stroke: "none", fill: C.a("blue", 0.1) }} />
            <Polyline pts={[[0, sigma], [6.4, w * 6.4 + sigma], [6.4, w * 6.4 - sigma], [0, -sigma]]} closed style={{ stroke: "none", fill: C.a("blue", 0.12) }} />
            {D.map((p, i) => <Polyline key={`r${i}`} pts={[p, [p[0], w * p[0]]]} style={{ stroke: C.a("pink", 0.7), strokeWidth: 1.1 }} />)}
            <FnPath f={(x) => w * x} style={{ stroke: C.blue, strokeWidth: 2 }} />
            <FnPath f={(x) => wStar * x} style={{ stroke: C.a("green", 0.8), strokeWidth: 1.2, strokeDasharray: "5 4" }} />
            {D.map((p, i) => <Dot key={i} p={p} r={3.8} color={C.fg} stroke={C.bg} />)}
          </Plot>
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <div className="mb-1 text-[12px] text-faint">Log-likelihood ℓ(w) — to be maximised</div>
              <Trace ariaLabel="log-likelihood as a function of w" xLabel="grid index of w" series={[{ pts: lls, color: C.green }]} marker={mark} yDomain={[Math.max(...lls) - 60, Math.max(...lls) + 3]} />
            </div>
            <div>
              <div className="mb-1 text-[12px] text-faint">Squared error ½Σ(yᵢ − wxᵢ)² — to be minimised</div>
              <Trace ariaLabel="sum of squared errors as a function of w" xLabel="grid index of w" series={[{ pts: sses, color: C.pink }]} marker={mark} yDomain={[0, Math.min(...sses) + 60 * sigma * sigma]} />
            </div>
          </div>
          <Legend items={[{ label: "mean wᵀx of the model", color: C.blue }, { label: "±σ and ±2σ noise bands", color: C.a("blue", 0.5) }, { label: "maximum-likelihood line", color: C.green, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="weight w" value={w} min={-0.2} max={1.6} step={0.015} onChange={setW} />
          <ParameterSlider label="noise standard deviation σ" value={sigma} min={0.2} max={2} step={0.05} onChange={setSigma} />
          <button className="btn w-full" onClick={() => setW(-0.2 + 0.015 * Math.round((wStar + 0.2) / 0.015))}>Maximise the likelihood</button>
          <p className="text-[12.5px] leading-relaxed text-muted">Changing σ rescales the log-likelihood but never moves its peak: the best w does not depend on the noise level.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\ell(w)`, tex: true, value: fmt(ll(w), 3), color: C.green },
        { label: String.raw`\tfrac12\sum(y_i-wx_i)^2`, tex: true, value: fmt(0.5 * sse(w), 3), color: C.pink },
        { label: String.raw`w_{\text{ML}}=w_{\text{LS}}`, tex: true, value: fmt(wStar, 4) },
        { label: String.raw`\hat\sigma^2_{\text{ML}}=\tfrac1n\text{SSE}(w_{\text{ML}})`, tex: true, value: fmt(sse(wStar) / D.length, 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`y_i=w\,x_i+\varepsilon_i,\quad\varepsilon_i\sim\mathcal N(0,\sigma^2)\ \Rightarrow\ y_i\mid x_i\sim\mathcal N(w\,x_i,\sigma^2)` },
        { tex: String.raw`\ell(w)=-\frac n2\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_i(y_i-w\,x_i)^2=${tex(-0.5 * D.length * Math.log(2 * Math.PI * sigma * sigma), 3)}-\frac{${tex(sse(w), 3)}}{${tex(2 * sigma * sigma, 3)}}=${tex(ll(w), 3)}` },
        { tex: String.raw`\arg\max_w\ell(w)=\arg\min_w\sum_i(y_i-w\,x_i)^2=${tex(wStar, 4)}`, note: "the first term and the factor 1/2σ² do not involve w" },
      ]} />}
      note="The two small plots are mirror images: wherever the squared error falls, the log-likelihood rises by the same amount divided by σ². Least squares is maximum likelihood under Gaussian noise."
    />
  );
}

function ViewsViz() {
  const [view, setView] = useState<"geometric" | "probabilistic">("geometric");
  return (
    <div className="space-y-3">
      <div className="max-w-md"><Segmented label="Perspective" value={view} onChange={setView} options={[{ value: "geometric", label: "Geometric (projection)" }, { value: "probabilistic", label: "Probabilistic (likelihood)" }]} /></div>
      {view === "geometric" ? <ProjectionView /> : <LikelihoodView />}
    </div>
  );
}

export default function LinRegWeek({ mode = "normal" }: { mode?: Mode }) {
  if (mode === "supervised") return <SupervisedViz />;
  if (mode === "gd") return <GDViz />;
  if (mode === "views") return <ViewsViz />;
  return <NormalViz />;
}

/* ------------------------------------------------------------------ kernel regression */
type KType = "poly" | "gauss";
const DATASETS: Record<string, V2[]> = {
  curve: [[-2.4, 0.3], [-1.6, -0.9], [-0.7, -0.2], [0.2, 1.1], [1.1, 0.9], [1.9, -0.4], [2.6, 0.6]],
  quadratic: [[0, 1], [1, 2], [2, 5]],
  step: [[-2.5, -1], [-1.5, -1], [-0.5, -1], [0.5, 1], [1.5, 1], [2.5, 1]],
};

export function KernelRegression() {
  const [D, setD] = useState<V2[]>(DATASETS.curve);
  const [type, setType] = useState<KType>("gauss");
  const [p, setP] = useState(2);
  const [sigma, setSigma] = useState(0.8);
  const [logLam, setLogLam] = useState(-6);
  const [q, setQ] = useState(0.6);
  const lam = logLam <= -6 ? 0 : 10 ** logLam;
  const k = (a: number, b: number) => (type === "poly" ? (1 + a * b) ** p : Math.exp(-((a - b) ** 2) / (2 * sigma * sigma)));
  const fit = useMemo(() => {
    const n = D.length;
    const K = D.map((a) => D.map((b) => k(a[0], b[0])));
    const y = D.map((d) => d[1]);
    let alpha: number[];
    let exact = true;
    if (lam > 0) alpha = solve(K.map((r, i) => r.map((v, j) => v + (i === j ? lam : 0))), y) ?? matVec(pinv(K), y);
    else { alpha = matVec(pinv(K, 1e-9), y); exact = false; }
    const pred = (x: number) => D.reduce((s, d, i) => s + alpha[i] * k(d[0], x), 0);
    const train = Math.sqrt(D.reduce((s, d) => s + (pred(d[0]) - d[1]) ** 2, 0) / n);
    return { K, alpha, pred, train, exact };
  }, [D, type, p, sigma, lam]); // eslint-disable-line react-hooks/exhaustive-deps
  const terms = D.map((d, i) => fit.alpha[i] * k(d[0], q));
  const maxA = Math.max(1e-9, ...fit.alpha.map(Math.abs));
  const big = maxA > 1e4;
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <Plot x={[-3.2, 3.2]} y={[-2.6, 2.8]} aspect={0.58} maxH={420} xLabel="x" yLabel="y" ariaLabel="kernel regression prediction through the data" onPointerDown={(pt) => setQ(Number(clamp(pt[0], -3.1, 3.1).toFixed(2)))}>
            {type === "gauss" && D.map((d, i) => <FnPath key={`t${i}`} f={(x) => fit.alpha[i] * k(d[0], x)} n={200} style={{ stroke: C.a("blue", 0.3), strokeWidth: 1 }} />)}
            <FnPath f={fit.pred} n={500} style={{ stroke: C.blue, strokeWidth: 2.2 }} />
            <Polyline pts={[[q, -9], [q, 9]]} style={{ stroke: C.a("fg", 0.4), strokeWidth: 1, strokeDasharray: "3 3" }} />
            <Dot p={[q, fit.pred(q)]} r={5} color={C.pink} stroke={C.bg} />
            {D.map((d, i) => <Handle key={i} p={d} color="amber" label={`training point ${i + 1}`} onDrag={(pt) => setD(D.map((o, j) => (j === i ? [clamp(pt[0], -3, 3), clamp(pt[1], -2.4, 2.6)] : o)))} />)}
          </Plot>
          <Legend items={[{ label: "prediction f(x) = Σ αᵢ k(xᵢ, x)", color: C.blue }, { label: "training points (drag)", color: C.amber }, ...(type === "gauss" ? [{ label: "individual terms αᵢ k(xᵢ, ·)", color: C.a("blue", 0.45) }] : []), { label: "test point", color: C.pink }]} />
          {big && <Warn>The coefficients are enormous (max |αᵢ| = {fmt(maxA)}): K is nearly singular for this kernel and these points, so exact interpolation is numerically fragile. Add a little regularisation.</Warn>}
        </div>
      }
      controls={
        <>
          <Segmented label="Kernel" value={type} onChange={setType} options={[{ value: "gauss", label: "Gaussian" }, { value: "poly", label: "Polynomial" }]} />
          {type === "poly" ? <ParameterSlider label="degree p" value={p} min={1} max={8} step={1} onChange={setP} /> : <ParameterSlider label="width σ" value={sigma} min={0.15} max={3} step={0.05} onChange={setSigma} />}
          <ParameterSlider label="regularisation λ" value={logLam} min={-6} max={1} step={0.25} onChange={setLogLam} format={() => (lam === 0 ? "0 (interpolate)" : lam.toExponential(1))} />
          <Select label="Training set" value="" onChange={(v) => v && setD(DATASETS[v])} options={[{ value: "", label: "Choose a preset…" }, { value: "curve", label: "Seven points on a curve" }, { value: "quadratic", label: "Worked example (y = 1 + x²)" }, { value: "step", label: "Step function" }]} />
          <div>
            <div className="label mb-1.5">Kernel matrix K</div>
            <Heat M={fit.K} digits={2} cell={Math.max(26, 44 - 2 * D.length)} />
          </div>
          <div>
            <div className="label mb-1.5">Coefficients α</div>
            <BarList digits={3} items={fit.alpha.map((a, i) => ({ label: `α${i + 1}  (x = ${fmt(D[i][0], 2)})`, value: a, color: a >= 0 ? C.blue : C.pink }))} />
          </div>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`f(x_{\text{test}})`, tex: true, value: fmt(fit.pred(q), 4), color: C.pink },
        { label: String.raw`x_{\text{test}}`, tex: true, value: fmt(q, 2) },
        { label: String.raw`\text{training RMSE}`, tex: true, value: fmt(fit.train, 5) },
        { label: String.raw`n\ (\text{size of }K)`, tex: true, value: `${D.length} × ${D.length}` },
        { label: String.raw`\max|\alpha_i|`, tex: true, value: fmt(maxA, 3) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: type === "poly" ? String.raw`K_{ij}=k(x_i,x_j)=(1+x_ix_j)^{${p}}` : String.raw`K_{ij}=k(x_i,x_j)=\exp\!\Big(-\frac{(x_i-x_j)^2}{2\cdot${tex(sigma, 2)}^2}\Big)` },
        { tex: lam > 0 ? String.raw`\boldsymbol\alpha=(K+\lambda I)^{-1}\mathbf y,\quad\lambda=${tex(lam, 4)}` : String.raw`\boldsymbol\alpha=K^{\dagger}\mathbf y`, note: lam > 0 ? "kernel ridge regression" : "λ = 0: pseudoinverse; equals K⁻¹y whenever K is invertible" },
        { tex: String.raw`f(${tex(q, 2)})=\sum_{i=1}^{${D.length}}\alpha_i\,k(x_i,${tex(q, 2)})=${terms.slice(0, 4).map((v) => `(${tex(v, 3)})`).join("+")}${terms.length > 4 ? String.raw`+\cdots` : ""}=${tex(fit.pred(q), 4)}`, note: "every training point takes part in every prediction" },
      ]} />}
      note={type === "poly" ? `A polynomial kernel of degree ${p} can represent polynomials up to degree ${p}. With ${D.length} points it interpolates them exactly only when p ≥ ${D.length - 1}; otherwise the result is the least-squares polynomial fit.` : "A narrow Gaussian kernel makes each point influence only its neighbourhood, so the curve passes through the data and returns to zero in between. A wide one produces smooth curves but an ill-conditioned K."}
    />
  );
}
