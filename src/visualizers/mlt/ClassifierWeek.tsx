"use client";
import { useMemo, useState } from "react";
import { Plot, Arrow, Dot, Handle, Polyline, FnPath, HeatmapCanvas, C, useThemeRGB, type PlotCtx } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Select } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { maxMarginOrigin } from "@/lib/ml/svmOrigin";
import { fmt, tex, clamp } from "@/lib/utilities/format";
import { MiniTable } from "@/visualizers/shared";
import { Trace, LineWB, separableData, H2, type V2 } from "./common";

export interface LP { x: V2; y: 1 | -1 }
const POS = C.pink, NEG = C.blue;
const dotw = (w: V2, x: V2) => w[0] * x[0] + w[1] * x[1];
/** Prediction convention of the notes: +1 when wᵀx ≥ 0, −1 otherwise. */
export const predict = (w: V2, x: V2): 1 | -1 => (dotw(w, x) >= 0 ? 1 : -1);

/** Light shading of the two half-planes of a linear classifier. */
export function HalfPlanes({ ctx, w, b = 0 }: { ctx: PlotCtx; w: V2; b?: number }) {
  const th = useThemeRGB();
  return (
    <HeatmapCanvas ctx={ctx} res={4} deps={[w[0], w[1], b, th.blue[0], th.panel[0]]} f={(x, y) => {
      const c = w[0] * x + w[1] * y + b >= 0 ? th.pink : th.blue, p = th.panel, a = 0.1;
      return [p[0] + (c[0] - p[0]) * a, p[1] + (c[1] - p[1]) * a, p[2] + (c[2] - p[2]) * a, 255];
    }} />
  );
}

function noisyData(kind: "noisy" | "separable" | "outlier", seed: number): LP[] {
  if (kind === "separable") return separableData(seed, 12, 0.5, 0.9);
  const r = mulberry32(seed * 3 + 11);
  const out: LP[] = [];
  for (const y of [1, -1] as const) for (let i = 0; i < 13; i++) out.push({ x: [y * 1.0 + randn(r) * 0.95, y * 0.7 + randn(r) * 0.95], y });
  if (kind === "outlier") for (let i = 0; i < 4; i++) out.push({ x: [3.4 + randn(r) * 0.2, -2.6 + randn(r) * 0.2], y: 1 });
  return out;
}

/* ------------------------------------------------------------------ zero-one loss of a linear classifier */
function ZeroOneViz() {
  const [kind, setKind] = useState<"noisy" | "separable" | "outlier">("noisy");
  const [seed, setSeed] = useState(2);
  const [w, setW] = useState<V2>([1.4, 0.2]);
  const [sel, setSel] = useState(0);
  const D = useMemo(() => noisyData(kind, seed), [kind, seed]);
  const mistakes = (v: V2) => D.filter((p) => predict(v, p.x) !== p.y).length;
  const m = mistakes(w);
  const curve = useMemo(() => Array.from({ length: 361 }, (_, d) => mistakes([Math.cos((d * Math.PI) / 180), Math.sin((d * Math.PI) / 180)]) / D.length), [D]); // eslint-disable-line react-hooks/exhaustive-deps
  const deg = ((Math.atan2(w[1], w[0]) * 180) / Math.PI + 360) % 360;
  // least-squares classifier: regress the ±1 labels on x, then take the sign
  const ls = useMemo((): V2 => {
    let a = 0, b = 0, d = 0, p = 0, q = 0;
    for (const pt of D) { a += pt.x[0] ** 2; b += pt.x[0] * pt.x[1]; d += pt.x[1] ** 2; p += pt.x[0] * pt.y; q += pt.x[1] * pt.y; }
    const det = a * d - b * b || 1;
    return [(d * p - b * q) / det, (a * q - b * p) / det];
  }, [D]);
  const i = Math.min(sel, D.length - 1), pt = D[i];
  const best = Math.min(...curve);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-4.2, 4.2]} y={[-3.6, 3.6]} equal originAxes aspect={0.72} maxH={440} ariaLabel="linear classifier through the origin with its mistakes" underlay={(ctx) => <HalfPlanes ctx={ctx} w={w} />}
            onPointerDown={(q) => { let b = 0, bd = Infinity; D.forEach((p, j) => { const d = (p.x[0] - q[0]) ** 2 + (p.x[1] - q[1]) ** 2; if (d < bd) { bd = d; b = j; } }); setSel(b); }}>
            <LineWB w={ls} style={{ stroke: C.a("green", 0.8), strokeWidth: 1.3, strokeDasharray: "6 4" }} />
            <LineWB w={w} />
            {D.map((p, j) => { const bad = predict(w, p.x) !== p.y; return <Dot key={j} p={p.x} r={bad ? 6 : 4} color={p.y > 0 ? POS : NEG} stroke={j === i ? C.amber : bad ? C.fg : C.bg} />; })}
            <Arrow from={[0, 0]} to={w} color="fg" label="w" />
            <Handle p={w} color="fg" label="weight vector w" onDrag={(q) => setW([clamp(q[0], -4, 4), clamp(q[1], -3.4, 3.4)])} />
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Zero-one loss as w rotates through 360° — a staircase with flat steps and several separate minima</div>
            <Trace ariaLabel="zero-one loss versus direction of w" xLabel="direction of w (degrees)" xMax={360} marker={deg} yDomain={[0, 1]} series={[{ pts: curve, color: C.pink }]} />
          </div>
          <Legend items={[{ label: "class +1", color: POS }, { label: "class −1", color: NEG }, { label: "decision boundary wᵀx = 0 (mistakes ringed)", color: C.fg }, { label: "least-squares classifier", color: C.green, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <Select label="Dataset" value={kind} onChange={(v) => { setKind(v); setSel(0); }} options={[{ value: "noisy", label: "Overlapping classes" }, { value: "separable", label: "Linearly separable" }, { value: "outlier", label: "With far-away correct points" }]} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New sample</button>
          <button className="btn w-full" onClick={() => { const d = curve.indexOf(best); setW([1.5 * Math.cos((d * Math.PI) / 180), 1.5 * Math.sin((d * Math.PI) / 180)]); }}>Jump to a minimiser (grid search over 360 directions)</button>
          <p className="text-[12.5px] leading-relaxed text-muted">Drag the tip of w. Its length does not matter: only the direction changes the predictions, because sign(wᵀx) = sign(c·wᵀx) for any c &gt; 0.</p>
          {kind === "outlier" && <p className="text-[12.5px] leading-relaxed text-muted">The four points far to the lower right are correctly on the positive side, yet squared error penalises them for being “too correct” and tilts the dashed least-squares boundary.</p>}
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{mistakes}`, tex: true, value: `${m} of ${D.length}`, color: C.pink },
        { label: String.raw`L_{0\text{-}1}(\mathbf w)`, tex: true, value: fmt(m / D.length, 4), color: C.pink },
        { label: String.raw`\min_{\mathbf w}L_{0\text{-}1}`, tex: true, value: fmt(best, 4) },
        { label: String.raw`L_{0\text{-}1}(\mathbf w_{\text{LS}})`, tex: true, value: fmt(mistakes(ls) / D.length, 4), color: C.green },
        { label: String.raw`\text{direction of }\mathbf w`, tex: true, value: `${fmt(deg, 1)}°` },
      ]} />}
      calc={<CalcSteps title="Live calculation for the selected point (amber ring; click a point)" steps={[
        { tex: String.raw`\mathbf w^\top\mathbf x=(${tex(w[0], 2)})(${tex(pt.x[0], 2)})+(${tex(w[1], 2)})(${tex(pt.x[1], 2)})=${tex(dotw(w, pt.x), 3)}\ \Rightarrow\ h(\mathbf x)=${predict(w, pt.x) > 0 ? "+1" : "-1"},\quad y=${pt.y > 0 ? "+1" : "-1"}`, note: predict(w, pt.x) === pt.y ? "correct: contributes 0" : "mistake: contributes 1" },
        { tex: String.raw`L_{0\text{-}1}(\mathbf w)=\frac1n\sum_{i=1}^{n}\mathbb 1[\operatorname{sign}(\mathbf w^\top\mathbf x_i)\neq y_i]=\frac{${m}}{${D.length}}=${tex(m / D.length, 4)}` },
      ]} />}
      note="Nudging w almost never changes the loss, so its gradient is zero wherever it is defined and gradient descent has nothing to follow. Minimising it exactly over linear classifiers is NP-hard in general, which motivates the surrogate losses of the later weeks."
    />
  );
}

/* ------------------------------------------------------------------ linear separability and margin */
function MarginViz() {
  const [margin, setMargin] = useState(0.7);
  const [seed, setSeed] = useState(3);
  const [w, setW] = useState<V2>([1.3, 0.6]);
  const D = useMemo(() => separableData(seed, 10, margin, 0.7), [seed, margin]);
  const nw = Math.hypot(w[0], w[1]) || 1;
  const dist = D.map((p) => (p.y * dotw(w, p.x)) / nw);
  const gamma = Math.min(...dist);
  const iMin = dist.indexOf(gamma);
  const opt = useMemo(() => maxMarginOrigin(D.map((p) => p.x), D.map((p) => p.y)), [D]);
  const R = Math.max(...D.map((p) => Math.hypot(p.x[0], p.x[1])));
  const sep = gamma > 0;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-4.6, 4.6]} y={[-3.9, 3.9]} equal originAxes aspect={0.74} maxH={460} ariaLabel="separating hyperplane and its margin" underlay={(ctx) => <HalfPlanes ctx={ctx} w={w} />}>
            {sep && <LineWB w={w} b={-gamma * nw} style={{ stroke: C.a("fg", 0.45), strokeWidth: 1, strokeDasharray: "4 4" }} />}
            {sep && <LineWB w={w} b={gamma * nw} style={{ stroke: C.a("fg", 0.45), strokeWidth: 1, strokeDasharray: "4 4" }} />}
            <LineWB w={opt.wUnit} style={{ stroke: C.a("green", 0.8), strokeWidth: 1.3, strokeDasharray: "7 4" }} />
            <LineWB w={w} />
            {D.map((p, j) => { const foot: V2 = [p.x[0] - (dotw(w, p.x) / (nw * nw)) * w[0], p.x[1] - (dotw(w, p.x) / (nw * nw)) * w[1]]; return j === iMin ? <Polyline key={`d${j}`} pts={[p.x, foot]} style={{ stroke: C.amber, strokeWidth: 2 }} /> : null; })}
            {D.map((p, j) => <Dot key={j} p={p.x} r={dist[j] < 0 ? 6 : 4.2} color={p.y > 0 ? POS : NEG} stroke={j === iMin ? C.amber : dist[j] < 0 ? C.fg : C.bg} />)}
            <Arrow from={[0, 0]} to={w} color="fg" label="w" />
            <Handle p={w} color="fg" label="weight vector w" onDrag={(q) => setW([clamp(q[0], -4.4, 4.4), clamp(q[1], -3.7, 3.7)])} />
          </Plot>
          <Legend items={[{ label: "boundary wᵀx = 0", color: C.fg }, { label: "closest point and its distance γ(w)", color: C.amber }, { label: "margin band", color: C.a("fg", 0.5), dashed: true }, { label: "maximum-margin boundary", color: C.green, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="gap built into the data" value={margin} min={0.1} max={1.6} step={0.05} onChange={setMargin} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New sample</button>
          <button className="btn w-full" onClick={() => setW([opt.wUnit[0] * 1.5, opt.wUnit[1] * 1.5])}>Move w to the maximum-margin direction</button>
          <p className="text-[12.5px] leading-relaxed text-muted">Many directions separate the classes. Each has its own margin γ(w): the distance from the boundary to the nearest point. The dataset is “linearly separable with margin γ” for the best of them.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{separates the data?}`, tex: true, value: sep ? "yes" : "no", color: sep ? C.green : C.red },
        { label: String.raw`\gamma(\mathbf w)=\min_i\frac{y_i\mathbf w^\top\mathbf x_i}{\|\mathbf w\|}`, tex: true, value: fmt(gamma, 4), color: C.amber },
        { label: String.raw`\gamma^*=\max_{\mathbf w}\gamma(\mathbf w)`, tex: true, value: fmt(opt.gamma, 4), color: C.green },
        { label: String.raw`R=\max_i\|\mathbf x_i\|`, tex: true, value: fmt(R, 4) },
        { label: String.raw`R^2/\gamma^{*2}`, tex: true, value: fmt((R * R) / (opt.gamma * opt.gamma), 2) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{closest point: }\mathbf x=(${tex(D[iMin].x[0], 2)},${tex(D[iMin].x[1], 2)}),\ y=${D[iMin].y > 0 ? "+1" : "-1"}:\quad\frac{y\,\mathbf w^\top\mathbf x}{\|\mathbf w\|}=\frac{${tex(D[iMin].y * dotw(w, D[iMin].x), 3)}}{${tex(nw, 3)}}=${tex(gamma, 4)}`, note: sep ? "distance to the boundary" : "negative: this point is on the wrong side" },
        { tex: String.raw`\text{separable with margin }\gamma\iff\exists\,\mathbf w^*,\ \|\mathbf w^*\|=1:\ \ y_i\,\mathbf w^{*\top}\mathbf x_i\ge\gamma\ \ \forall i` },
        { tex: String.raw`\mathbf w^*=(${tex(opt.wUnit[0], 3)},\ ${tex(opt.wUnit[1], 3)}),\qquad\gamma^*=${tex(opt.gamma, 4)}`, note: "found by solving the hard-margin SVM of Week 10" },
      ]} />}
      note="The margin is a property of the data and the direction together. The ratio R²/γ*² in the last readout is the perceptron's mistake bound, proved two lessons later: wide margins mean few mistakes."
    />
  );
}

export default function LinearClassifier({ mode = "zero-one" }: { mode?: "zero-one" | "margin" }) {
  return mode === "margin" ? <MarginViz /> : <ZeroOneViz />;
}

/* ------------------------------------------------------------------ decision stump: entropy and information gain */
function stumpData(kind: "bands" | "blobs" | "xor", seed: number): LP[] {
  const r = mulberry32(seed * 5 + 3);
  return Array.from({ length: 40 }, (_, i) => {
    const x: V2 = [r() * 6 - 3, r() * 6 - 3];
    let y: 1 | -1;
    if (kind === "bands") y = x[0] + randn(r) * 0.45 > 0.4 ? 1 : -1;
    else if (kind === "xor") y = x[0] * x[1] > 0 ? 1 : -1;
    else { const c = i % 2 ? 1 : -1; x[0] = c * 1.1 + randn(r) * 0.9; x[1] = c * 0.9 + randn(r) * 0.9; y = c as 1 | -1; }
    return { x, y };
  });
}

export function Stump() {
  const [kind, setKind] = useState<"bands" | "blobs" | "xor">("blobs");
  const [seed, setSeed] = useState(1);
  const [feat, setFeat] = useState<0 | 1>(0);
  const [t, setT] = useState(-1);
  const D = useMemo(() => stumpData(kind, seed), [kind, seed]);
  const split = (f: 0 | 1, thr: number) => {
    const L = D.filter((p) => p.x[f] <= thr), Rr = D.filter((p) => p.x[f] > thr);
    const pos = (S: LP[]) => S.filter((p) => p.y > 0).length;
    const pP = pos(D) / D.length, pL = L.length ? pos(L) / L.length : 0, pR = Rr.length ? pos(Rr) / Rr.length : 0;
    const g = L.length / D.length;
    return { nL: L.length, nR: Rr.length, posL: pos(L), posR: pos(Rr), pP, pL, pR, g, HP: H2(pP), HL: H2(pL), HR: H2(pR), IG: H2(pP) - (g * H2(pL) + (1 - g) * H2(pR)) };
  };
  const s = split(feat, t);
  const grid = Array.from({ length: 121 }, (_, i) => -3 + i * 0.05);
  const curves = useMemo(() => ([0, 1] as const).map((f) => grid.map((g) => split(f, g).IG)), [D]); // eslint-disable-line react-hooks/exhaustive-deps
  const bestOf = (f: 0 | 1) => { const c = curves[f]; const i = c.indexOf(Math.max(...c)); return { t: grid[i], ig: c[i] }; };
  const b0 = bestOf(0), b1 = bestOf(1);
  const leaf = (p: number) => (p >= 0.5 ? "+1" : "−1");
  const err = (D.filter((p) => (p.x[feat] <= t ? (s.pL >= 0.5 ? 1 : -1) : s.pR >= 0.5 ? 1 : -1) !== p.y).length) / D.length;
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-[1.25fr_1fr]">
            <Plot x={[-3.3, 3.3]} y={[-3.3, 3.3]} equal aspect={1} maxH={380} xLabel="x₁" yLabel="x₂" ariaLabel="dataset split by one threshold question">
              <Polyline pts={feat === 0 ? [[t, -9], [t, 9]] : [[-9, t], [9, t]]} style={{ stroke: C.fg, strokeWidth: 2 }} />
              {D.map((p, j) => <Dot key={j} p={p.x} r={4.2} color={p.y > 0 ? POS : NEG} stroke={C.bg} />)}
            </Plot>
            <div>
              <div className="mb-1 text-[12px] text-faint">Entropy H(p) and the three nodes</div>
              <Plot x={[0, 1]} y={[0, 1.08]} aspect={0.82} maxH={300} xLabel="fraction of +1 in the node, p" yLabel="H(p) bits" ticks={5} ariaLabel="binary entropy with the parent and child nodes marked">
                <FnPath f={H2} domain={[0.0005, 0.9995]} n={300} style={{ stroke: C.a("fg", 0.6), strokeWidth: 1.6 }} />
                <Dot p={[s.pP, s.HP]} r={6} color={C.fg} stroke={C.bg} />
                {s.nL > 0 && <Dot p={[s.pL, s.HL]} r={5} color={C.green} stroke={C.bg} />}
                {s.nR > 0 && <Dot p={[s.pR, s.HR]} r={5} color={C.amber} stroke={C.bg} />}
              </Plot>
            </div>
          </div>
          <div>
            <div className="mb-1 text-[12px] text-faint">Information gain of the question “x{feat === 0 ? "₁" : "₂"} ≤ θ ?” for every threshold θ from −3 to 3</div>
            <Trace ariaLabel="information gain versus threshold" xLabel="grid index of θ" marker={(t + 3) / 0.05} yDomain={[0, Math.max(0.2, Math.max(b0.ig, b1.ig) * 1.1)]} series={[{ pts: curves[feat], color: C.pink }, { pts: curves[feat === 0 ? 1 : 0], color: C.a("fg", 0.25), dashed: true }]} />
          </div>
          <Legend items={[{ label: "parent node", color: C.fg }, { label: "yes branch (≤ θ)", color: C.green }, { label: "no branch (> θ)", color: C.amber }, { label: "information gain, chosen feature", color: C.pink }, { label: "the other feature", color: C.a("fg", 0.4), dashed: true }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Feature asked about" value={String(feat)} onChange={(v) => setFeat(Number(v) as 0 | 1)} options={[{ value: "0", label: "x₁" }, { value: "1", label: "x₂" }]} />
          <ParameterSlider label="threshold θ" value={t} min={-3} max={3} step={0.05} onChange={setT} />
          <button className="btn w-full" onClick={() => { const b = b0.ig >= b1.ig ? { f: 0 as const, ...b0 } : { f: 1 as const, ...b1 }; setFeat(b.f); setT(Number(b.t.toFixed(2))); }}>Best question over both features</button>
          <Select label="Dataset" value={kind} onChange={setKind} options={[{ value: "blobs", label: "Two blobs" }, { value: "bands", label: "Split along x₁" }, { value: "xor", label: "XOR (no single question helps)" }]} />
          <button className="btn w-full" onClick={() => setSeed((v) => v + 1)}>New sample</button>
          <MiniTable head={["node", "points", "+1", "−1", "p", "H(p)", "predicts"]} rows={[
            ["parent", D.length, D.filter((p) => p.y > 0).length, D.filter((p) => p.y < 0).length, fmt(s.pP, 3), fmt(s.HP, 4), leaf(s.pP)],
            ["yes (≤ θ)", s.nL, s.posL, s.nL - s.posL, s.nL ? fmt(s.pL, 3) : "—", fmt(s.HL, 4), s.nL ? leaf(s.pL) : "—"],
            ["no (> θ)", s.nR, s.posR, s.nR - s.posR, s.nR ? fmt(s.pR, 3) : "—", fmt(s.HR, 4), s.nR ? leaf(s.pR) : "—"],
          ]} />
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{information gain}`, tex: true, value: fmt(s.IG, 4), color: C.pink },
        { label: String.raw`\gamma=|D_{\text{yes}}|/|D|`, tex: true, value: fmt(s.g, 3) },
        { label: String.raw`\text{stump training error}`, tex: true, value: fmt(err, 3) },
        { label: String.raw`\text{best IG on }x_1`, tex: true, value: `${fmt(b0.ig, 4)} at θ = ${fmt(b0.t, 2)}` },
        { label: String.raw`\text{best IG on }x_2`, tex: true, value: `${fmt(b1.ig, 4)} at θ = ${fmt(b1.t, 2)}` },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`H(D)=-p\log_2p-(1-p)\log_2(1-p)\ \text{with}\ p=${tex(s.pP, 3)}:\quad H(D)=${tex(s.HP, 4)}` },
        { tex: String.raw`H(D_{\text{yes}})=H(${tex(s.pL, 3)})=${tex(s.HL, 4)},\qquad H(D_{\text{no}})=H(${tex(s.pR, 3)})=${tex(s.HR, 4)}` },
        { tex: String.raw`IG=H(D)-\big[\gamma H(D_{\text{yes}})+(1-\gamma)H(D_{\text{no}})\big]=${tex(s.HP, 4)}-\big[${tex(s.g, 3)}\cdot${tex(s.HL, 4)}+${tex(1 - s.g, 3)}\cdot${tex(s.HR, 4)}\big]=${tex(s.IG, 4)}` },
      ]} />}
      note={kind === "xor" ? "For XOR every single question leaves both branches about half positive, so the gain is near zero everywhere. A second level of questions is needed: this is why trees are grown deeper than one stump." : "A good question sends the two children towards the ends of the entropy curve (pure nodes), far below the parent. Information gain is exactly that drop, weighted by how many points go each way."}
    />
  );
}
