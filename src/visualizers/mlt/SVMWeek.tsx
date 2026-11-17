"use client";
import { useMemo, useState } from "react";
import { Plot, Arrow, Dot, Handle, Polyline, FnPath, C } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend, Warn } from "@/components/visualization/LabShell";
import { ParameterSlider, AnimationControls } from "@/components/controls";
import { useTicker } from "@/hooks/useSize";
import { mulberry32, randn } from "@/lib/math/random";
import { svmOrigin } from "@/lib/ml/svmOrigin";
import { fmt, tex, clamp } from "@/lib/utilities/format";
import { MiniTable, BarList } from "@/visualizers/shared";
import { Trace, LineWB, separableData, type V2 } from "./common";
import { HalfPlanes, type LP } from "./ClassifierWeek";

type Mode = "margin" | "dual" | "sv" | "soft" | "soft-dual";
const POS = C.pink, NEG = C.blue;
const dotw = (w: readonly number[], x: V2) => w[0] * x[0] + w[1] * x[1];
const XY = (D: LP[]) => ({ X: D.map((p) => [p.x[0], p.x[1]]), y: D.map((p) => p.y as number) });

function Margins({ w }: { w: V2 }) {
  return (
    <>
      <LineWB w={w} b={-1} style={{ stroke: C.a("fg", 0.5), strokeWidth: 1.1, strokeDasharray: "5 4" }} />
      <LineWB w={w} b={1} style={{ stroke: C.a("fg", 0.5), strokeWidth: 1.1, strokeDasharray: "5 4" }} />
      <LineWB w={w} />
    </>
  );
}

/* ------------------------------------------------------------------ functional and geometric margin */
function MarginView() {
  const [seed, setSeed] = useState(2);
  const [w, setW] = useState<V2>([1.6, 0.4]);
  const D = useMemo(() => separableData(seed, 9, 0.6, 0.75), [seed]);
  const opt = useMemo(() => { const { X, y } = XY(D); return svmOrigin(X, y); }, [D]);
  const ws = opt.w as V2;
  const fm = D.map((p) => p.y * dotw(w, p.x));
  const func = Math.min(...fm), nw = Math.hypot(w[0], w[1]) || 1e-9;
  const iMin = fm.indexOf(func);
  const feasible = func >= 1 - 1e-9;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-5, 5]} y={[-4.2, 4.2]} equal originAxes aspect={0.74} maxH={460} ariaLabel="hyperplane with its supporting hyperplanes" underlay={(ctx) => <HalfPlanes ctx={ctx} w={w} />}>
            <Margins w={w} />
            {D.map((p, j) => <Dot key={j} p={p.x} r={fm[j] < 1 - 1e-9 ? 5.5 : 4.2} color={p.y > 0 ? POS : NEG} stroke={j === iMin ? C.amber : fm[j] < 1 - 1e-9 ? C.fg : C.bg} />)}
            <Arrow from={[0, 0]} to={w} color="fg" label="w" />
            <Handle p={w} color="fg" label="weight vector w" onDrag={(q) => setW([clamp(q[0], -4.8, 4.8), clamp(q[1], -4, 4)])} />
          </Plot>
          <Legend items={[{ label: "wᵀx = 0", color: C.fg }, { label: "supporting hyperplanes wᵀx = ±1", color: C.a("fg", 0.6), dashed: true }, { label: "closest point", color: C.amber }, { label: "points violating yᵢwᵀxᵢ ≥ 1 (ringed)", color: C.muted }]} />
          {!feasible && func > 0 && <Warn>w separates the classes but some points lie between the dashed lines, so the constraints yᵢwᵀxᵢ ≥ 1 fail. Lengthening w (same direction) pulls the dashed lines inwards until they are satisfied.</Warn>}
        </div>
      }
      controls={
        <>
          <button className="btn w-full" disabled={func <= 0} onClick={() => setW([w[0] / func, w[1] / func])}>Rescale w so the closest point has functional margin 1</button>
          <button className="btn w-full" onClick={() => setW([w[0] * 2, w[1] * 2])}>Double w (same boundary)</button>
          <button className="btn w-full" onClick={() => setW([ws[0], ws[1]])}>Move to the optimal w*</button>
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New sample</button>
          <p className="text-[12.5px] leading-relaxed text-muted">Dragging w along its own direction changes the functional margin but not the geometric one. Fixing the functional margin at 1 removes that freedom; then a wide margin is the same thing as a short w.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{functional margin }\min_iy_i\mathbf w^\top\mathbf x_i`, tex: true, value: fmt(func, 4), color: C.amber },
        { label: String.raw`\text{geometric margin}`, tex: true, value: fmt(func / nw, 4) },
        { label: String.raw`\|\mathbf w\|`, tex: true, value: fmt(nw, 4) },
        { label: String.raw`\tfrac12\|\mathbf w\|^2`, tex: true, value: fmt(0.5 * nw * nw, 4), color: feasible ? C.fg : C.faint },
        { label: String.raw`\text{feasible?}`, tex: true, value: feasible ? "yes" : "no", color: feasible ? C.green : C.red },
        { label: String.raw`\tfrac12\|\mathbf w^*\|^2\ \text{(optimum)}`, tex: true, value: fmt(0.5 * (ws[0] ** 2 + ws[1] ** 2), 4), color: C.green },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{closest point }\mathbf x=(${tex(D[iMin].x[0], 2)},${tex(D[iMin].x[1], 2)}),\ y=${D[iMin].y > 0 ? "+1" : "-1"}:\quad y\,\mathbf w^\top\mathbf x=${tex(func, 4)}`, note: "functional margin: depends on the length of w" },
        { tex: String.raw`\frac{y\,\mathbf w^\top\mathbf x}{\|\mathbf w\|}=\frac{${tex(func, 4)}}{${tex(nw, 4)}}=${tex(func / nw, 4)}`, note: "geometric margin: the actual distance to the boundary" },
        { tex: String.raw`\text{if }\min_iy_i\mathbf w^\top\mathbf x_i=1:\quad\text{margin}=\frac{1}{\|\mathbf w\|}=${tex(1 / nw, 4)},\quad\text{distance between dashed lines}=\frac{2}{\|\mathbf w\|}=${tex(2 / nw, 4)}` },
        { tex: String.raw`\max_{\mathbf w}\frac{1}{\|\mathbf w\|}\ \equiv\ \min_{\mathbf w}\tfrac12\|\mathbf w\|^2\quad\text{s.t.}\quad y_i\,\mathbf w^\top\mathbf x_i\ge1\ \ \forall i` },
      ]} />}
      note="Among all feasible w the shortest one has its dashed lines as far apart as the data allow. That is the hard-margin SVM: the unique maximum-margin separator."
    />
  );
}

/* ------------------------------------------------------------------ dual coordinate ascent */
function DualView() {
  const [seed, setSeed] = useState(5);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const MAXS = 40;
  const D = useMemo(() => separableData(seed, 4, 0.5, 0.75), [seed]);
  const { X, y } = useMemo(() => XY(D), [D]);
  const full = useMemo(() => svmOrigin(X, y, Infinity, MAXS, 0), [X, y]);
  const cur = useMemo(() => (t === 0 ? { w: [0, 0], alpha: X.map(() => 0) } : svmOrigin(X, y, Infinity, t, 0)), [X, y, t]);
  useTicker(playing, 450, () => setT((v) => { if (v >= MAXS) { setPlaying(false); return MAXS; } return v + 1; }));
  const w = cur.w as V2;
  const dualVals = [0, ...full.dual];
  // a feasible primal point at every sweep: rescale w so that the smallest functional margin equals 1
  const primalVals = [NaN, ...full.wHist.map((wv) => { const m = Math.min(...D.map((p) => p.y * dotw(wv, p.x))); return m > 0 ? 0.5 * (wv[0] ** 2 + wv[1] ** 2) / (m * m) : NaN; })];
  const gap = primalVals[t] - dualVals[t];
  const top = Math.max(...dualVals.filter(Number.isFinite), ...primalVals.filter(Number.isFinite)) * 1.25;
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <Plot x={[-5, 5]} y={[-4.2, 4.2]} equal originAxes aspect={0.7} maxH={400} ariaLabel="separator built from the current dual variables">
            {Math.hypot(w[0], w[1]) > 1e-9 && <Margins w={w} />}
            {D.map((p, j) => <Dot key={j} p={p.x} r={4 + 5 * Math.sqrt(cur.alpha[j] / Math.max(1e-9, ...cur.alpha))} color={p.y > 0 ? POS : NEG} stroke={cur.alpha[j] > 1e-6 ? C.fg : C.bg} />)}
            <Arrow from={[0, 0]} to={w} color="fg" label="w = Σ αᵢyᵢxᵢ" />
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Dual objective (green, rises) and the primal objective of the rescaled w (pink, falls): they meet at the optimum</div>
            <Trace ariaLabel="primal and dual objective values per sweep" xLabel="sweep" xMax={MAXS} marker={t} yDomain={[0, top]} series={[{ pts: primalVals.map((v) => Math.min(v, top)), color: C.pink, dots: true }, { pts: dualVals, color: C.green, dots: true }]} />
          </div>
          <Legend items={[{ label: "dot size ∝ αᵢ; ringed points have αᵢ > 0", color: C.fg }, { label: "dual value g(α)", color: C.green }, { label: "primal value ½‖w‖² of a feasible w", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="sweeps of coordinate ascent" value={t} min={0} max={MAXS} step={1} onChange={(v) => { setPlaying(false); setT(v); }} />
          <AnimationControls playing={playing} onPlay={() => { if (!playing && t >= MAXS) setT(0); setPlaying(!playing); }} onStep={() => setT((v) => Math.min(MAXS, v + 1))} onReset={() => { setPlaying(false); setT(0); }} />
          <button className="btn w-full" onClick={() => { setSeed((s) => s + 1); setT(0); setPlaying(false); }}>New sample</button>
          <div>
            <div className="label mb-1.5">Dual variables α (one per data point)</div>
            <BarList digits={4} items={cur.alpha.map((a, j) => ({ label: `α${j + 1}  (y = ${D[j].y > 0 ? "+1" : "−1"})`, value: a, color: D[j].y > 0 ? POS : NEG }))} />
          </div>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`g(\boldsymbol\alpha)=\boldsymbol\alpha^\top\mathbf 1-\tfrac12\boldsymbol\alpha^\top Y X^\top XY\boldsymbol\alpha`, tex: true, value: fmt(dualVals[t], 5), color: C.green },
        { label: String.raw`\tfrac12\|\mathbf w_{\text{feasible}}\|^2`, tex: true, value: Number.isFinite(primalVals[t]) ? fmt(primalVals[t], 5) : "— (w = 0)", color: C.pink },
        { label: String.raw`\text{duality gap}`, tex: true, value: Number.isFinite(gap) ? fmt(gap, 6) : "—" },
        { label: String.raw`\mathbf w`, tex: true, value: `(${fmt(w[0], 4)}, ${fmt(w[1], 4)})` },
        { label: String.raw`\text{non-zero }\alpha_i`, tex: true, value: cur.alpha.filter((a) => a > 1e-6).length },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\mathcal L(\mathbf w,\boldsymbol\alpha)=\tfrac12\|\mathbf w\|^2+\sum_i\alpha_i\,(1-y_i\mathbf w^\top\mathbf x_i),\qquad\alpha_i\ge0` },
        { tex: String.raw`\nabla_{\mathbf w}\mathcal L=\mathbf 0\ \Rightarrow\ \mathbf w=\sum_i\alpha_iy_i\mathbf x_i=XY\boldsymbol\alpha=(${tex(w[0], 4)},\ ${tex(w[1], 4)})` },
        { tex: String.raw`\alpha_i\leftarrow\max\!\Big(0,\ \alpha_i+\frac{1-y_i\mathbf w^\top\mathbf x_i}{\|\mathbf x_i\|^2}\Big)`, note: "exact maximisation of g over one coordinate; a sweep visits every i once" },
        { tex: String.raw`\text{weak duality: }g(\boldsymbol\alpha)\le\tfrac12\|\mathbf w\|^2\ \text{for every feasible pair}` + (Number.isFinite(primalVals[t]) ? String.raw`;\quad\text{here }${tex(dualVals[t], 4)}\le${tex(primalVals[t], 4)}` : ""), note: Number.isFinite(primalVals[t]) ? undefined : "press Step to run the first sweep" },
      ]} />}
      note="The dual has one variable per data point and only the simple constraints αᵢ ≥ 0, which is why it can be solved one coordinate at a time. Any dual value is a lower bound on any feasible primal value; because the problem is convex the two bounds close completely (strong duality)."
    />
  );
}

/* ------------------------------------------------------------------ support vectors and complementary slackness */
function SupportView() {
  const [D, setD] = useState<LP[]>([
    { x: [1.2, 1.4], y: 1 }, { x: [2.6, 0.4], y: 1 }, { x: [3.2, 2.6], y: 1 }, { x: [0.6, 3], y: 1 },
    { x: [-1.4, -0.8], y: -1 }, { x: [-0.4, -2.2], y: -1 }, { x: [-3, -1.6], y: -1 }, { x: [-2.4, -3.2], y: -1 },
  ]);
  const sol = useMemo(() => { const { X, y } = XY(D); return svmOrigin(X, y, Infinity, 3000, 1e-10); }, [D]);
  const w = sol.w as V2;
  const fm = D.map((p) => p.y * dotw(w, p.x));
  const ok = sol.converged && Math.min(...fm) > 1 - 1e-4;
  const sv = sol.alpha.map((a) => ok && a > 1e-6);
  const nw = Math.hypot(w[0], w[1]) || 1e-9;
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <Plot x={[-5, 5]} y={[-4.2, 4.2]} equal originAxes aspect={0.74} maxH={460} ariaLabel="maximum margin separator and its support vectors" underlay={(ctx) => (ok ? <HalfPlanes ctx={ctx} w={w} /> : null)}>
            {ok && <Margins w={w} />}
            {ok && D.map((p, j) => (sv[j] ? <Polyline key={`s${j}`} pts={[[0, 0], [sol.alpha[j] * p.y * p.x[0], sol.alpha[j] * p.y * p.x[1]]]} style={{ stroke: C.a("amber", 0.8), strokeWidth: 1.6 }} /> : null))}
            {ok && <Arrow from={[0, 0]} to={w} color="fg" label="w*" />}
            {D.map((p, j) => <Dot key={`r${j}`} p={p.x} r={sv[j] ? 11 : 0} color="transparent" stroke={C.amber} />)}
            {D.map((p, j) => <Handle key={j} p={p.x} color={p.y > 0 ? "pink" : "blue"} r={6} label={`point ${j + 1}, class ${p.y > 0 ? "+1" : "−1"}`} onDrag={(q) => setD(D.map((o, t) => (t === j ? { ...o, x: [clamp(q[0], -4.8, 4.8), clamp(q[1], -4, 4)] } : o)))} />)}
          </Plot>
          <Legend items={[{ label: "class +1 (drag any point)", color: POS }, { label: "class −1", color: NEG }, { label: "support vectors: αᵢ > 0, on a dashed line", color: C.amber }, { label: "their contributions αᵢyᵢxᵢ, which add up to w*", color: C.a("amber", 0.8) }]} />
          {!ok && <Warn>These points cannot be separated by a line through the origin, so the hard-margin problem has no feasible w. Move the points apart, or see Week 11 for the soft margin.</Warn>}
        </div>
      }
      controls={
        <>
          <MiniTable head={["i", "yᵢ", "yᵢ w*ᵀxᵢ", "αᵢ", "αᵢ (1 − yᵢ w*ᵀxᵢ)"]} rows={D.map((p, j) => [j + 1, p.y > 0 ? "+1" : "−1", ok ? fmt(fm[j], 4) : "—", ok ? fmt(sol.alpha[j], 4) : "—", ok ? fmt(sol.alpha[j] * (1 - fm[j]), 5) : "—"])} rowClass={(j) => (sv[j] ? "bg-raised" : undefined)} />
          <p className="text-[12.5px] leading-relaxed text-muted">Drag a point that is not ringed: as long as it stays outside the dashed lines, w* does not move at all. Drag a ringed point and the whole solution follows it.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\mathbf w^*`, tex: true, value: ok ? `(${fmt(w[0], 4)}, ${fmt(w[1], 4)})` : "—" },
        { label: String.raw`\text{margin }1/\|\mathbf w^*\|`, tex: true, value: ok ? fmt(1 / nw, 4) : "—" },
        { label: String.raw`\text{support vectors}`, tex: true, value: ok ? `${sv.filter(Boolean).length} of ${D.length}` : "—", color: C.amber },
        { label: String.raw`\textstyle\sum_i\alpha_i^*`, tex: true, value: ok ? fmt(sol.alpha.reduce((a, b) => a + b, 0), 4) : "—" },
        { label: String.raw`\|\mathbf w^*\|^2`, tex: true, value: ok ? fmt(nw * nw, 4) : "—" },
      ]} />}
      calc={ok ? <CalcSteps steps={[
        { tex: String.raw`\alpha_i^*\,\big(1-y_i\,\mathbf w^{*\top}\mathbf x_i\big)=0\quad\forall i`, note: "complementary slackness: for each point at least one factor is zero" },
        { tex: String.raw`\alpha_i^*>0\ \Rightarrow\ y_i\,\mathbf w^{*\top}\mathbf x_i=1\ \ (\text{on a supporting hyperplane}),\qquad y_i\,\mathbf w^{*\top}\mathbf x_i>1\ \Rightarrow\ \alpha_i^*=0` },
        { tex: String.raw`\mathbf w^*=\sum_{i\in SV}\alpha_i^*y_i\mathbf x_i=${D.map((p, j) => (sv[j] ? `${tex(sol.alpha[j], 3)}\\cdot(${p.y > 0 ? "+1" : "-1"})\\begin{bmatrix}${tex(p.x[0], 2)}\\\\${tex(p.x[1], 2)}\\end{bmatrix}` : "")).filter(Boolean).join("+")}=\begin{bmatrix}${tex(w[0], 3)}\\${tex(w[1], 3)}\end{bmatrix}` },
        { tex: String.raw`\textstyle\sum_i\alpha_i^*=${tex(sol.alpha.reduce((a, b) => a + b, 0), 4)}=\|\mathbf w^*\|^2=${tex(nw * nw, 4)}`, note: "consequence of strong duality: Σα − ½‖w‖² = ½‖w‖²" },
      ]} /> : undefined}
      note="The solution is a combination of the support vectors only. Every other training point could be deleted without changing the classifier, which is the “sparse” nature of the SVM."
    />
  );
}

/* ------------------------------------------------------------------ soft margin */
function softData(seed: number): LP[] {
  const r = mulberry32(seed * 13 + 2);
  const out: LP[] = [];
  for (const y of [1, -1] as const) for (let i = 0; i < 14; i++) out.push({ x: [y * 1.25 + randn(r) * 0.95, y * 0.85 + randn(r) * 0.95], y });
  return out;
}

function SoftView({ dual }: { dual: boolean }) {
  const [seed, setSeed] = useState(3);
  const [logC, setLogC] = useState(0);
  const [sel, setSel] = useState(0);
  const Cc = 10 ** logC;
  const D = useMemo(() => softData(seed), [seed]);
  const sol = useMemo(() => { const { X, y } = XY(D); return svmOrigin(X, y, Cc, 4000, 1e-10); }, [D, Cc]);
  const w = sol.w as V2, nw2 = w[0] ** 2 + w[1] ** 2, nw = Math.sqrt(nw2) || 1e-9;
  const u = D.map((p) => p.y * dotw(w, p.x));
  const xi = u.map((v) => Math.max(0, 1 - v));
  const sumXi = xi.reduce((a, b) => a + b, 0);
  const tolA = 1e-5 * Math.max(1, Cc);
  const cat = sol.alpha.map((a) => (a <= tolA ? 0 : a >= Cc - tolA ? 2 : 1));
  const i = Math.min(sel, D.length - 1);
  const group = (k: number) => { const idx = D.map((_, j) => j).filter((j) => cat[j] === k); return { n: idx.length, lo: idx.length ? Math.min(...idx.map((j) => u[j])) : NaN, hi: idx.length ? Math.max(...idx.map((j) => u[j])) : NaN }; };
  const g0 = group(0), g1 = group(1), g2 = group(2);
  const ring = (j: number) => (dual ? (cat[j] === 1 ? C.green : cat[j] === 2 ? C.amber : C.bg) : xi[j] > 1 ? C.fg : C.bg);
  return (
    <LabShell
      wideControls={dual}
      viz={
        <div className="space-y-3">
          <Plot x={[-5, 5]} y={[-4.2, 4.2]} equal originAxes aspect={0.72} maxH={430} ariaLabel="soft margin separator with slack variables" underlay={(ctx) => <HalfPlanes ctx={ctx} w={w} />}
            onPointerDown={(q) => { let b = 0, bd = Infinity; D.forEach((p, j) => { const d = (p.x[0] - q[0]) ** 2 + (p.x[1] - q[1]) ** 2; if (d < bd) { bd = d; b = j; } }); setSel(b); }}>
            <Margins w={w} />
            {D.map((p, j) => (xi[j] > 1e-6 ? <Polyline key={`x${j}`} pts={[p.x, [p.x[0] + ((p.y - dotw(w, p.x)) / nw2) * w[0], p.x[1] + ((p.y - dotw(w, p.x)) / nw2) * w[1]]]} style={{ stroke: C.a("amber", 0.85), strokeWidth: 1.5 }} /> : null))}
            {D.map((p, j) => <Dot key={j} p={p.x} r={j === i ? 6.5 : cat[j] > 0 && dual ? 5.4 : 4.2} color={p.y > 0 ? POS : NEG} stroke={j === i ? C.fg : ring(j)} />)}
          </Plot>
          {!dual && (
            <div>
              <div className="mb-1 text-[12px] text-faint">Every training point on the hinge-loss curve: horizontal position u = yᵢwᵀxᵢ, height ξᵢ = max(0, 1 − u)</div>
              <Plot x={[-3, 5]} y={[0, 4]} aspect={0.26} minH={130} maxH={180} xLabel="u = y·wᵀx" yLabel="ξ" ticks={5} ariaLabel="hinge loss with the training points placed on it">
                <FnPath f={(v) => Math.max(0, 1 - v)} n={200} style={{ stroke: C.a("fg", 0.6), strokeWidth: 1.6 }} />
                <Polyline pts={[[0, 0], [0, 9]]} style={{ stroke: C.a("fg", 0.2), strokeWidth: 1 }} />
                {D.map((p, j) => <Dot key={j} p={[u[j], xi[j]]} r={j === i ? 5.5 : 3.6} color={p.y > 0 ? POS : NEG} stroke={j === i ? C.fg : C.bg} />)}
              </Plot>
            </div>
          )}
          <Legend items={dual ? [{ label: "αᵢ = 0: outside the margin", color: C.muted }, { label: "0 < αᵢ < C: exactly on the margin", color: C.green }, { label: "αᵢ = C: inside the margin or misclassified", color: C.amber }, { label: "selected point", color: C.fg }] : [{ label: "class +1", color: POS }, { label: "class −1", color: NEG }, { label: "slack ξᵢ (drawn as a distance ξᵢ/‖w‖)", color: C.amber }, { label: "misclassified (ξᵢ > 1) are ringed", color: C.fg }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="penalty C" value={logC} min={-2.5} max={2.5} step={0.05} onChange={setLogC} format={() => fmt(Cc, 3)} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New sample</button>
          {dual ? (
            <MiniTable head={["dual variable", "points", "range of yᵢwᵀxᵢ", "must satisfy"]} rows={[
              ["αᵢ = 0", g0.n, g0.n ? `${fmt(g0.lo, 3)} … ${fmt(g0.hi, 3)}` : "—", "≥ 1"],
              ["0 < αᵢ < C", g1.n, g1.n ? `${fmt(g1.lo, 3)} … ${fmt(g1.hi, 3)}` : "—", "= 1"],
              ["αᵢ = C", g2.n, g2.n ? `${fmt(g2.lo, 3)} … ${fmt(g2.hi, 3)}` : "—", "≤ 1"],
            ]} />
          ) : (
            <p className="text-[12.5px] leading-relaxed text-muted">Small C tolerates violations and buys a wide margin; large C punishes every violation and narrows the margin until, for separable data, it becomes the hard-margin SVM.</p>
          )}
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\tfrac12\|\mathbf w\|^2`, tex: true, value: fmt(0.5 * nw2, 4) },
        { label: String.raw`C\textstyle\sum_i\xi_i`, tex: true, value: fmt(Cc * sumXi, 4), color: C.amber },
        { label: String.raw`\text{objective}`, tex: true, value: fmt(0.5 * nw2 + Cc * sumXi, 4) },
        { label: String.raw`\text{margin width }2/\|\mathbf w\|`, tex: true, value: fmt(2 / nw, 4) },
        { label: String.raw`\text{points with }\xi_i>0`, tex: true, value: xi.filter((v) => v > 1e-6).length },
        { label: String.raw`\text{misclassified }(\xi_i>1)`, tex: true, value: xi.filter((v) => v > 1).length },
      ]} />}
      calc={<CalcSteps title={`Live calculation for the selected point ${i + 1} (click a point)`} steps={dual ? [
        { tex: String.raw`y_i\mathbf w^\top\mathbf x_i=${tex(u[i], 4)},\quad\xi_i=${tex(xi[i], 4)},\quad\alpha_i=${tex(sol.alpha[i], 4)},\quad\beta_i=C-\alpha_i=${tex(Cc - sol.alpha[i], 4)}` },
        { tex: String.raw`\alpha_i\,(1-\xi_i-y_i\mathbf w^\top\mathbf x_i)=${tex(sol.alpha[i] * (1 - xi[i] - u[i]), 5)},\qquad\beta_i\,\xi_i=${tex((Cc - sol.alpha[i]) * xi[i], 5)}`, note: "both complementary-slackness products vanish (up to solver tolerance)" },
        { tex: cat[i] === 0 ? String.raw`\alpha_i=0\Rightarrow\beta_i=C>0\Rightarrow\xi_i=0\Rightarrow y_i\mathbf w^\top\mathbf x_i\ge1` : cat[i] === 1 ? String.raw`0<\alpha_i<C\Rightarrow\beta_i>0\Rightarrow\xi_i=0\ \text{and}\ y_i\mathbf w^\top\mathbf x_i=1-\xi_i=1` : String.raw`\alpha_i=C\Rightarrow y_i\mathbf w^\top\mathbf x_i=1-\xi_i\le1`, note: cat[i] === 0 ? "not a support vector" : cat[i] === 1 ? "support vector on the margin" : "support vector that violates the margin" },
      ] : [
        { tex: String.raw`\xi_i=\max(0,\ 1-y_i\mathbf w^\top\mathbf x_i)=\max(0,\ 1-(${tex(u[i], 4)}))=${tex(xi[i], 4)}`, note: xi[i] === 0 ? "outside the margin: no penalty" : xi[i] <= 1 ? "inside the margin but on the correct side" : "on the wrong side of the boundary" },
        { tex: String.raw`\min_{\mathbf w,\boldsymbol\xi}\ \tfrac12\|\mathbf w\|^2+C\sum_i\xi_i=${tex(0.5 * nw2, 4)}+${tex(Cc, 3)}\times${tex(sumXi, 4)}=${tex(0.5 * nw2 + Cc * sumXi, 4)}` },
        { tex: String.raw`\equiv\ \min_{\mathbf w}\ \tfrac12\|\mathbf w\|^2+C\sum_i\max(0,\ 1-y_i\mathbf w^\top\mathbf x_i)`, note: "the slack variables can be eliminated: this is hinge loss with an L2 penalty" },
      ]} />}
      note={dual ? "The table checks the three scenarios on the solved problem: the middle group sits at 1 and the other two respect their inequalities. Only points in the last two groups are support vectors, and w = Σ αᵢyᵢxᵢ is built from them alone." : "The optimum balances two terms. Each amber segment costs C per unit of slack, while shrinking ‖w‖ widens the margin but creates more slack."}
    />
  );
}

export default function SVMWeek({ mode = "margin" }: { mode?: Mode }) {
  if (mode === "dual") return <DualView />;
  if (mode === "sv") return <SupportView />;
  if (mode === "soft") return <SoftView dual={false} />;
  if (mode === "soft-dual") return <SoftView dual />;
  return <MarginView />;
}
