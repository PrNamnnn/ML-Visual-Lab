"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, Polyline, HeatmapCanvas, C, useThemeRGB } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, AnimationControls, Select, Toggle } from "@/components/controls";
import { useTicker } from "@/hooks/useSize";
import { mulberry32, randInt } from "@/lib/math/random";
import { makeDataset, buildTree, treePredict, adaboost, type DatasetId, type Point, type TreeNode } from "@/lib/ml/models";
import { fmt, pct, tex, cn } from "@/lib/utilities/format";
import { Trace } from "./common";

/* ------------------------------------------------------------------ bootstrapping and bagged trees */
export default function Bagging({ forest = false }: { forest?: boolean }) {
  const [ds, setDs] = useState<DatasetId>("moons");
  const [B, setB] = useState(forest ? 25 : 15);
  const [depth, setDepth] = useState(8);
  const [subset, setSubset] = useState(forest);
  const [bag, setBag] = useState(0);
  const [view, setView] = useState<"ensemble" | "single">("ensemble");
  const [nSmall, setNSmall] = useState(20);
  const [seed, setSeed] = useState(1);
  const th = useThemeRGB();
  const MAXB = 60;
  const train = useMemo(() => makeDataset(ds, 120, 0.3, 7), [ds]);
  const test = useMemo(() => makeDataset(ds, 600, 0.3, 991), [ds]);
  const n = train.length;
  // all bags are drawn once; using the first B of them makes the B slider a pure "add more trees" control
  const bags = useMemo(() => {
    const r = mulberry32(seed * 77 + 5);
    return Array.from({ length: MAXB }, () => Array.from({ length: n }, () => randInt(r, 0, n - 1)));
  }, [seed, n]);
  const trees = useMemo(() => {
    const r = mulberry32(seed * 9 + 1);
    return bags.map((idx) => buildTree(idx.map((i) => train[i]), depth, 1, "entropy", 0, subset ? (g) => [g() < 0.5 ? 0 : 1] : undefined, subset ? r : undefined));
  }, [bags, train, depth, subset, seed]);
  const vote = (x: number, y: number, upTo: number) => { let s = 0; for (let t = 0; t < upTo; t++) s += treePredict(trees[t], x, y) >= 0.5 ? 1 : 0; return s / upTo; };
  const bi = Math.min(bag, B - 1);
  const predictOne = (t: TreeNode) => (x: number, y: number) => (treePredict(t, x, y) >= 0.5 ? 1 : 0);
  const f = view === "ensemble" ? (x: number, y: number) => vote(x, y, B) : predictOne(trees[bi]);
  const accOf = (g: (x: number, y: number) => number, pts: Point[]) => pts.filter((p) => (g(p.x, p.y) >= 0.5 ? 1 : 0) === p.label).length / pts.length;
  const curve = useMemo(() => {
    const votes = test.map(() => 0);
    return [NaN, ...trees.map((t, b) => { test.forEach((p, i) => { votes[i] += treePredict(t, p.x, p.y) >= 0.5 ? 1 : 0; }); return 1 - test.filter((p, i) => (votes[i] / (b + 1) >= 0.5 ? 1 : 0) === p.label).length / test.length; })];
  }, [trees, test]);
  const singleErr = useMemo(() => trees.slice(0, 30).reduce((s, t) => s + (1 - accOf(predictOne(t), test)), 0) / 30, [trees, test]);
  // the small bootstrap demo
  const small = useMemo(() => { const r = mulberry32(seed * 131 + nSmall); return Array.from({ length: nSmall }, () => randInt(r, 0, nSmall - 1)); }, [seed, nSmall]);
  const counts = Array.from({ length: nSmall }, (_, i) => small.filter((v) => v === i).length);
  const uniq = counts.filter((c) => c > 0).length;
  const theory = 1 - (1 - 1 / nSmall) ** nSmall;
  const inBag = new Set(bags[bi]);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-4">
          <div>
            <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2 text-[12px] text-faint">
              <span>One bootstrap bag from a dataset of {nSmall} points: how many times each point was drawn</span>
              <span className="num text-muted">{uniq} of {nSmall} distinct = {pct(uniq / nSmall, 1)}; expected {pct(theory, 1)}</span>
            </div>
            <div className="flex flex-wrap gap-[3px]">
              {counts.map((c, i) => <span key={i} className={cn("num flex h-6 w-6 items-center justify-center rounded-[3px] text-[11px]", c === 0 ? "border border-dashed border-line-strong text-faint" : "text-fg")} style={c ? { background: `rgb(var(--blue) / ${Math.min(0.85, 0.2 + 0.2 * c)})` } : undefined} title={`point ${i + 1}: drawn ${c} time${c === 1 ? "" : "s"}`}>{c}</span>)}
            </div>
          </div>
          <Plot x={[-2.8, 2.8]} y={[-2.5, 2.5]} equal aspect={0.7} maxH={420} ariaLabel="decision regions of the ensemble or of one tree"
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={4} deps={[trees, B, view, bi, th.blue[0], th.panel[0]]} f={(x, y) => { const p = f(x, y), c = p >= 0.5 ? th.pink : th.blue, a = 0.1 + 0.3 * Math.abs(p - 0.5) * 2, bg = th.panel; return [bg[0] + (c[0] - bg[0]) * a, bg[1] + (c[1] - bg[1]) * a, bg[2] + (c[2] - bg[2]) * a, 255]; }} />}>
            {train.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={3.4} color={p.label ? C.pink : C.blue} opacity={view === "single" && !inBag.has(i) ? 0.25 : 1} stroke={C.bg} />)}
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Test error of the majority vote as trees are added (600 fresh points); dashed: average error of a single tree</div>
            <Trace ariaLabel="test error versus number of trees" xLabel="number of trees B" xMax={MAXB} marker={B} series={[{ pts: curve, color: C.pink }, { pts: curve.map(() => singleErr), color: C.a("fg", 0.4), dashed: true }]} yDomain={[0, Math.max(0.3, singleErr * 1.4)]} />
          </div>
          <Legend items={[{ label: "class 0", color: C.blue }, { label: "class 1", color: C.pink }, { label: view === "ensemble" ? `shading: fraction of the ${B} trees voting class 1` : `tree ${bi + 1}; faded points are out-of-bag for it`, color: C.muted }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Decision regions of" value={view} onChange={setView} options={[{ value: "ensemble", label: "Majority vote" }, { value: "single", label: "One tree" }]} />
          <ParameterSlider label="number of bags / trees B" value={B} min={1} max={MAXB} step={1} onChange={setB} />
          {view === "single" && <ParameterSlider label="tree shown" value={bi + 1} min={1} max={B} step={1} onChange={(v) => setBag(v - 1)} />}
          <ParameterSlider label="maximum tree depth" value={depth} min={1} max={12} step={1} onChange={setDepth} />
          <Toggle label="Random feature at each split (random forest)" checked={subset} onChange={setSubset} />
          <Select label="Dataset" value={ds} onChange={setDs} options={[{ value: "moons", label: "Moons" }, { value: "circles", label: "Circles" }, { value: "blobs", label: "Blobs" }, { value: "xor", label: "XOR" }, { value: "spiral", label: "Spiral" }]} />
          <ParameterSlider label="size n of the bootstrap demo" value={nSmall} min={5} max={100} step={1} onChange={setNSmall} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>Draw new bags</button>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{test error, vote of }B`, tex: true, value: fmt(curve[B], 4), color: C.pink },
        { label: String.raw`\text{test error, average single tree}`, tex: true, value: fmt(singleErr, 4) },
        { label: String.raw`\text{train error, vote}`, tex: true, value: fmt(1 - accOf((x, y) => vote(x, y, B), train), 4) },
        { label: String.raw`\text{distinct points in bag}`, tex: true, value: `${new Set(bags[bi]).size} of ${n}` },
        { label: String.raw`1-(1-\tfrac1n)^n`, tex: true, value: fmt(1 - (1 - 1 / n) ** n, 4) },
        { label: String.raw`1-1/e`, tex: true, value: fmt(1 - Math.exp(-1), 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`P(\text{point }i\text{ is never drawn})=\Big(1-\frac1n\Big)^{n}=\Big(1-\frac{1}{${nSmall}}\Big)^{${nSmall}}=${tex((1 - 1 / nSmall) ** nSmall, 4)}\ \xrightarrow{\,n\to\infty\,}\ \frac1e\approx0.3679` },
        { tex: String.raw`\text{expected fraction of distinct points}=1-${tex((1 - 1 / nSmall) ** nSmall, 4)}=${tex(theory, 4)};\qquad\text{this bag: }\frac{${uniq}}{${nSmall}}=${tex(uniq / nSmall, 4)}` },
        { tex: String.raw`H(\mathbf x)=\operatorname{majority}\{h_1(\mathbf x),\dots,h_{${B}}(\mathbf x)\},\qquad h_b\ \text{trained on bag }b` },
        ...(subset ? [{ tex: String.raw`\operatorname{Var}\Big(\frac1B\sum_bh_b\Big)=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2`, note: "restricting each split to a random feature lowers the correlation ρ between trees" }] : []),
      ]} />}
      note={subset ? "A random forest is bagging plus one more source of randomness: each split may only use a random subset of the features (here one of the two). The trees disagree more, so their average is more stable." : "Deep trees fit their own bag almost perfectly and differ a lot from bag to bag (high variance). Voting averages those differences away: the ensemble's test error falls below that of a typical single tree and then levels off."}
    />
  );
}

/* ------------------------------------------------------------------ AdaBoost, round by round */
export function AdaBoostSteps() {
  const [ds, setDs] = useState<DatasetId>("circles");
  const [t, setT] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [sel, setSel] = useState(0);
  const th = useThemeRGB();
  const T = 40;
  const pts = useMemo(() => makeDataset(ds, 60, ds === "circles" ? 0.18 : 0.3, 11), [ds]);
  const ab = useMemo(() => adaboost(pts, T), [pts]);
  useTicker(playing, 600, () => setT((v) => { if (v >= T) { setPlaying(false); return T; } return v + 1; }));
  const n = pts.length;
  const y = pts.map((p) => (p.label ? 1 : -1));
  const L = ab.learners[t - 1];
  const h = (l: (typeof ab.learners)[0], p: Point) => ((l.f === 0 ? p.x : p.y) > l.t ? 1 : -1) * l.s;
  const Dt = ab.weights[t - 1], Dn = ab.weights[t];
  const wrong = pts.map((p, i) => h(L, p) !== y[i]);
  const trainErr = Array.from({ length: T + 1 }, (_, k) => (k === 0 ? NaN : pts.filter((p, i) => (ab.score(p.x, p.y, k) >= 0 ? 1 : -1) !== y[i]).length / n));
  const bound = Array.from({ length: T + 1 }, (_, k) => (k === 0 ? NaN : ab.learners.slice(0, k).reduce((s, l) => s * 2 * Math.sqrt(l.err * (1 - l.err)), 1)));
  const wMax = Math.max(...Dt);
  const i = Math.min(sel, n - 1);
  const Z = Dt.reduce((s, d, j) => s + d * Math.exp(-L.alpha * y[j] * h(L, pts[j])), 0);
  const wrongMassAfter = Dn.reduce((s, d, j) => s + (wrong[j] ? d : 0), 0);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-2.8, 2.8]} y={[-2.5, 2.5]} equal aspect={0.7} maxH={430} ariaLabel="AdaBoost ensemble regions, current stump and example weights"
            onPointerDown={(q) => { let b = 0, bd = Infinity; pts.forEach((p, j) => { const d = (p.x - q[0]) ** 2 + (p.y - q[1]) ** 2; if (d < bd) { bd = d; b = j; } }); setSel(b); }}
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={4} deps={[ab, t, th.blue[0], th.panel[0]]} f={(x, yy) => { const s = ab.score(x, yy, t), c = s >= 0 ? th.pink : th.blue, bg = th.panel, a = 0.14; return [bg[0] + (c[0] - bg[0]) * a, bg[1] + (c[1] - bg[1]) * a, bg[2] + (c[2] - bg[2]) * a, 255]; }} />}>
            <Polyline pts={L.f === 0 ? [[L.t, -9], [L.t, 9]] : [[-9, L.t], [9, L.t]]} style={{ stroke: C.fg, strokeWidth: 2 }} />
            {pts.map((p, j) => <Dot key={j} p={[p.x, p.y]} r={2.5 + 9 * Math.sqrt(Dt[j] / wMax)} color={p.label ? C.a("pink", 0.85) : C.a("blue", 0.85)} stroke={j === i ? C.amber : wrong[j] ? C.fg : C.bg} />)}
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Training error of the weighted vote (pink) and its upper bound ∏ 2√(εₜ(1 − εₜ)) (dashed)</div>
            <Trace ariaLabel="AdaBoost training error and bound per round" xLabel="round t" xMax={T} marker={t} yDomain={[0, 1]} series={[{ pts: bound, color: C.a("fg", 0.45), dashed: true }, { pts: trainErr, color: C.pink, dots: true }]} />
          </div>
          <Legend items={[{ label: "dot area ∝ weight Dₜ(i) before this round", color: C.muted }, { label: `weak learner hₜ (stump on x${L.f === 0 ? "₁" : "₂"})`, color: C.fg }, { label: "misclassified by hₜ (ringed)", color: C.fg }, { label: "shading: prediction of the vote after t rounds", color: C.a("pink", 0.6) }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="round t" value={t} min={1} max={T} step={1} onChange={(v) => { setPlaying(false); setT(v); }} />
          <AnimationControls playing={playing} onPlay={() => { if (!playing && t >= T) setT(1); setPlaying(!playing); }} onStep={() => setT((v) => Math.min(T, v + 1))} onReset={() => { setPlaying(false); setT(1); }} />
          <Select label="Dataset" value={ds} onChange={(v) => { setDs(v); setT(1); setPlaying(false); }} options={[{ value: "circles", label: "Circles" }, { value: "moons", label: "Moons" }, { value: "xor", label: "XOR" }, { value: "blobs", label: "Blobs" }]} />
          <p className="text-[12.5px] leading-relaxed text-muted">Every weak learner is a single threshold on one coordinate. After each round the points it got wrong grow and the others shrink, so the next stump is forced to deal with them.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\varepsilon_t`, tex: true, value: fmt(L.err, 4) },
        { label: String.raw`\alpha_t=\tfrac12\ln\tfrac{1-\varepsilon_t}{\varepsilon_t}`, tex: true, value: fmt(L.alpha, 4) },
        { label: String.raw`\text{training error of the vote}`, tex: true, value: fmt(trainErr[t], 4), color: C.pink },
        { label: String.raw`\text{bound}`, tex: true, value: fmt(bound[t], 4) },
        { label: String.raw`\text{weight on }h_t\text{'s mistakes after update}`, tex: true, value: fmt(wrongMassAfter, 4) },
        { label: String.raw`\max_iD_t(i)\,/\,\tfrac1n`, tex: true, value: fmt(wMax * n, 2) },
      ]} />}
      calc={<CalcSteps title={`Round ${t} — selected point ${i + 1} (amber ring)`} steps={[
        { tex: String.raw`\varepsilon_t=\sum_{i:\,h_t(\mathbf x_i)\neq y_i}D_t(i)=${tex(L.err, 4)}\ \Rightarrow\ \alpha_t=\tfrac12\ln\frac{${tex(1 - L.err, 4)}}{${tex(L.err, 4)}}=${tex(L.alpha, 4)}` },
        { tex: String.raw`D_{t+1}(i)=\frac{D_t(i)\,e^{-\alpha_ty_ih_t(\mathbf x_i)}}{Z_t}=\frac{${tex(Dt[i], 5)}\times e^{${wrong[i] ? "+" : "-"}${tex(L.alpha, 3)}}}{${tex(Z, 4)}}=${tex(Dn[i], 5)}`, note: wrong[i] ? "misclassified by hₜ: weight goes up" : "classified correctly by hₜ: weight goes down" },
        { tex: String.raw`H_t(\mathbf x)=\operatorname{sign}\Big(\sum_{s=1}^{${t}}\alpha_sh_s(\mathbf x)\Big)`, note: `after the update, hₜ's mistakes carry weight ${fmt(wrongMassAfter, 3)} — exactly one half, so hₜ is no better than chance on the new weights` },
      ]} />}
      note="No single stump can separate these classes, yet the weighted vote does. The training error can go up briefly but the dashed bound falls by a factor 2√(ε(1 − ε)) < 1 every round, which forces the error to zero."
    />
  );
}
