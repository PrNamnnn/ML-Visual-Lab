"use client";
import { useMemo, useState } from "react";
import { Plot, HeatmapCanvas, Dot, C, useThemeRGB, Polyline, usePlot } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented, Toggle } from "@/components/controls";
import {
  makeDataset, type DatasetId, type Point, logisticStep, sigmoid, knnPredict, gnbFit, gnbPredict, buildTree, treePredict, treeDepth, treeLeaves,
  buildForest, svmFit, svmDecision, adaboost, type TreeNode,
} from "@/lib/ml/models";
import { fmt, tex } from "@/lib/utilities/format";

export type Algo = "logistic" | "knn" | "naive-bayes" | "tree" | "forest" | "svm" | "adaboost";

const ALGOS: { value: Algo; label: string }[] = [
  { value: "logistic", label: "Logistic regression" },
  { value: "knn", label: "K-nearest neighbours" },
  { value: "naive-bayes", label: "Gaussian Naive Bayes" },
  { value: "tree", label: "Decision tree (CART)" },
  { value: "forest", label: "Random forest" },
  { value: "svm", label: "Support vector machine" },
  { value: "adaboost", label: "AdaBoost (stumps)" },
];

function TreeSplits({ node, box }: { node: TreeNode; box: [number, number, number, number] }) {
  if (node.leaf) return null;
  const [x0, x1, y0, y1] = box;
  const line: [number, number][] = node.feature === 0 ? [[node.threshold, y0], [node.threshold, y1]] : [[x0, node.threshold], [x1, node.threshold]];
  const lb: [number, number, number, number] = node.feature === 0 ? [x0, node.threshold, y0, y1] : [x0, x1, y0, node.threshold];
  const rb: [number, number, number, number] = node.feature === 0 ? [node.threshold, x1, y0, y1] : [x0, x1, node.threshold, y1];
  return (
    <g>
      <Polyline pts={line} style={{ stroke: C.a("fg", Math.max(0.25, 0.9 - node.depth * 0.15)), strokeWidth: Math.max(0.8, 2.4 - node.depth * 0.4) }} />
      <TreeSplits node={node.left} box={lb} />
      <TreeSplits node={node.right} box={rb} />
    </g>
  );
}

function Ellipse({ mu, v, color }: { mu: [number, number]; v: [number, number]; color: string }) {
  const { sx, sy } = usePlot();
  return <ellipse cx={sx(mu[0])} cy={sy(mu[1])} rx={Math.abs(sx(2 * Math.sqrt(v[0])) - sx(0))} ry={Math.abs(sy(2 * Math.sqrt(v[1])) - sy(0))} style={{ fill: "none", stroke: color, strokeWidth: 1.4, strokeDasharray: "5 3" }} />;
}

export default function Classifier({ algo: initial = "knn", lab = false, dataset = "moons" }: { algo?: Algo; lab?: boolean; dataset?: DatasetId }) {
  const [algo, setAlgo] = useState<Algo>(initial);
  const [ds, setDs] = useState<DatasetId>(dataset);
  const [noise, setNoise] = useState(0.2);
  const [extra, setExtra] = useState<Point[]>([]);
  const [addClass, setAddClass] = useState<0 | 1>(1);
  const [k, setK] = useState(7);
  const [depth, setDepth] = useState(4);
  const [nTrees, setNTrees] = useState(25);
  const [C_, setC] = useState(1);
  const [kernel, setKernel] = useState<"linear" | "rbf" | "poly">("rbf");
  const [gamma, setGamma] = useState(1);
  const [rounds, setRounds] = useState(20);
  const [lr, setLr] = useState(0.5);
  const [l2, setL2] = useState(0);
  const [query, setQuery] = useState<[number, number] | null>(null);
  const [showProb, setShowProb] = useState(true);
  const th = useThemeRGB();
  const pts = useMemo(() => [...makeDataset(ds, 140, noise, 7), ...extra], [ds, noise, extra]);

  const model = useMemo(() => {
    switch (algo) {
      case "logistic": {
        let st = { w: [0, 0] as [number, number], b: 0, history: [] as { loss: number }[] };
        for (let i = 0; i < 400; i++) st = logisticStep(pts, st, lr, l2);
        return { predict: (x: number, y: number) => sigmoid(st.w[0] * x + st.w[1] * y + st.b), info: st };
      }
      case "knn": return { predict: (x: number, y: number) => knnPredict(pts, x, y, k).prob, info: null };
      case "naive-bayes": { const m = gnbFit(pts); return { predict: (x: number, y: number) => gnbPredict(m, x, y), info: m }; }
      case "tree": { const t = buildTree(pts, depth, 1); return { predict: (x: number, y: number) => treePredict(t, x, y), info: t }; }
      case "forest": { const f = buildForest(pts, nTrees, depth, 3); return { predict: (x: number, y: number) => f.reduce((s, t) => s + (treePredict(t.tree, x, y) >= 0.5 ? 1 : 0), 0) / f.length, info: f }; }
      case "svm": {
        const m = svmFit(pts, C_, kernel === "linear" ? { type: "linear" } : kernel === "rbf" ? { type: "rbf", gamma } : { type: "poly", degree: 3, coef0: 1 });
        return { predict: (x: number, y: number) => sigmoid(2 * svmDecision(m, x, y)), info: m, decision: (x: number, y: number) => svmDecision(m, x, y) };
      }
      case "adaboost": { const a = adaboost(pts, rounds); return { predict: (x: number, y: number) => sigmoid(2 * a.score(x, y)), info: a }; }
    }
  }, [algo, pts, k, depth, nTrees, C_, kernel, gamma, rounds, lr, l2]);

  const acc = pts.filter((p) => (model.predict(p.x, p.y) >= 0.5 ? 1 : 0) === p.label).length / pts.length;
  const svm = algo === "svm" ? (model.info as ReturnType<typeof svmFit>) : null;
  const nSV = svm ? svm.alpha.filter((a) => a > 1e-6).length : 0;
  const decision = (model as { decision?: (x: number, y: number) => number }).decision;
  const nbInfo = algo === "naive-bayes" ? (model.info as ReturnType<typeof gnbFit>) : null;
  const q = query ? knnPredict(pts, query[0], query[1], k) : null;

  const deps = [algo, pts, k, depth, nTrees, C_, kernel, gamma, rounds, lr, l2, showProb, th.blue[0]];
  const steps: { tex: string; note?: string }[] = [];
  switch (algo) {
    case "logistic": { const st = model.info as { w: number[]; b: number; history: { loss: number }[] }; steps.push({ tex: `P(y=1\\mid\\mathbf x)=\\sigma(${tex(st.w[0])}x_1+${tex(st.w[1])}x_2+${tex(st.b)})` }, { tex: `\\text{log-loss after 400 GD steps}=${tex(st.history[st.history.length - 1].loss, 4)}` }); break; }
    case "knn": steps.push({ tex: `\\hat P(y=1\\mid\\mathbf x)=\\tfrac1k\\sum_{i\\in N_k(\\mathbf x)}y_i,\\quad k=${k}` }); if (q) steps.push({ tex: `\\text{query: }${q.neighbors.filter((nb) => pts[nb.i].label === 1).length}/${k}\\text{ neighbours in class 1}\\Rightarrow\\hat P=${tex(q.prob)}` }); break;
    case "naive-bayes": steps.push({ tex: "P(y\\mid x_1,x_2)\\propto P(y)\\,\\mathcal N(x_1;\\mu_{y1},\\sigma^2_{y1})\\,\\mathcal N(x_2;\\mu_{y2},\\sigma^2_{y2})", note: "features assumed independent given the class" }); break;
    case "tree": { const t = model.info as TreeNode; steps.push({ tex: `\\text{Gini}(p)=2p(1-p);\\ \\text{split maximises }\\Delta G=G(\\text{parent})-\\sum\\tfrac{n_c}{n}G(c)`, note: `depth ${treeDepth(t)}, ${treeLeaves(t)} leaves` }); if (!t.leaf) steps.push({ tex: `\\text{root split: }x_{${t.feature + 1}}\\le${tex(t.threshold)},\\ \\Delta G=${tex(t.gain, 4)}` }); break; }
    case "forest": steps.push({ tex: `\\hat y=\\text{majority vote of }${nTrees}\\text{ trees, each on a bootstrap sample with random feature per split}` }); break;
    case "svm": steps.push({ tex: `\\min_{\\mathbf w,b}\\tfrac12\\|\\mathbf w\\|^2+C\\sum_i\\xi_i\\ \\text{s.t.}\\ y_i(\\mathbf w^\\top\\phi(\\mathbf x_i)+b)\\ge1-\\xi_i,\\quad C=${tex(C_)}` }, { tex: `f(\\mathbf x)=\\sum_i\\alpha_iy_iK(\\mathbf x_i,\\mathbf x)+b,\\ \\ ${nSV}\\text{ support vectors}`, note: kernel === "rbf" ? `K = exp(−${fmt(gamma)}‖x − x′‖²)` : kernel === "poly" ? "K = (x·x′ + 1)³" : "K = x·x′" }); break;
    case "adaboost": { const a = model.info as ReturnType<typeof adaboost>; const l0 = a.learners[0]; steps.push({ tex: `F(\\mathbf x)=\\sum_{m=1}^{${rounds}}\\alpha_mh_m(\\mathbf x),\\quad\\alpha_m=\\tfrac12\\ln\\tfrac{1-\\varepsilon_m}{\\varepsilon_m}` }); if (l0) steps.push({ tex: `h_1:\\ x_{${l0.f + 1}}>${tex(l0.t)},\\ \\varepsilon_1=${tex(l0.err)},\\ \\alpha_1=${tex(l0.alpha)}` }); break; }
  }

  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <Plot x={[-2.6, 2.6]} y={[-2.4, 2.4]} equal aspect={0.75} maxH={500} ariaLabel={`${algo} decision regions`}
            onPointerDown={(p, e) => { if (algo === "knn" && !e.shiftKey) setQuery(p); else setExtra([...extra, { x: p[0], y: p[1], label: addClass }]); }}
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={4} deps={deps} f={(x, y) => {
              const pr = model.predict(x, y);
              const v = showProb ? pr : pr >= 0.5 ? 1 : 0;
              const c = v >= 0.5 ? th.pink : th.blue;
              const a = Math.abs(v - 0.5) * 2;
              const base = th.panel;
              const k2 = 0.12 + 0.33 * a;
              return [base[0] + (c[0] - base[0]) * k2, base[1] + (c[1] - base[1]) * k2, base[2] + (c[2] - base[2]) * k2, 255];
            }} />}>
            {algo === "tree" && <TreeSplits node={model.info as TreeNode} box={[-4, 4, -4, 4]} />}
            {nbInfo && nbInfo.classes.map((c, i) => <Ellipse key={i} mu={c.mu} v={c.var} color={i ? C.pink : C.blue} />)}
            {decision && <SvmMargins f={decision} />}
            {q && q.neighbors.map((nb) => <Polyline key={nb.i} pts={[query!, [pts[nb.i].x, pts[nb.i].y]]} style={{ stroke: C.a("fg", 0.45), strokeWidth: 1 }} />)}
            {pts.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={svm && svm.alpha[i] > 1e-6 ? 5 : 3.4} color={p.label ? C.pink : C.blue} stroke={svm && svm.alpha[i] > 1e-6 ? C.fg : C.bg} />)}
            {query && algo === "knn" && <Dot p={query} r={6} color={C.fg} stroke={C.bg} />}
          </Plot>
          <Legend items={[{ label: "class 0", color: C.blue }, { label: "class 1", color: C.pink }, ...(svm ? [{ label: "support vectors (ringed); dashed = margins f(x) = ±1", color: C.fg }] : []), ...(nbInfo ? [{ label: "per-class 2σ (axis-aligned)", color: C.muted, dashed: true }] : [])]} />
        </div>
      }
      controls={
        <>
          {lab && <Select label="Algorithm" value={algo} onChange={(v) => { setAlgo(v); setQuery(null); }} options={ALGOS} />}
          <Select label="Dataset" value={ds} onChange={(v) => { setDs(v); setExtra([]); }} options={[{ value: "blobs", label: "Blobs" }, { value: "linear", label: "Linearly separable" }, { value: "moons", label: "Moons" }, { value: "circles", label: "Circles" }, { value: "xor", label: "XOR" }, { value: "spiral", label: "Spiral" }]} />
          <ParameterSlider label="noise" value={noise} min={0} max={0.6} step={0.01} onChange={setNoise} />
          {algo === "logistic" && <><ParameterSlider label="learning rate" value={lr} min={0.05} max={2} step={0.05} onChange={setLr} /><ParameterSlider label="L2 penalty" value={l2} min={0} max={0.5} step={0.01} onChange={setL2} /></>}
          {algo === "knn" && <ParameterSlider label="k" value={k} min={1} max={51} step={2} onChange={setK} />}
          {(algo === "tree" || algo === "forest") && <ParameterSlider label="max depth" value={depth} min={1} max={12} step={1} onChange={setDepth} />}
          {algo === "forest" && <ParameterSlider label="trees" value={nTrees} min={1} max={60} step={1} onChange={setNTrees} />}
          {algo === "svm" && (
            <>
              <Segmented value={kernel} onChange={setKernel} options={[{ value: "linear", label: "Linear" }, { value: "rbf", label: "RBF" }, { value: "poly", label: "Poly-3" }]} />
              <ParameterSlider label="C (penalty on violations)" value={C_} min={0.05} max={20} step={0.05} onChange={setC} />
              {kernel === "rbf" && <ParameterSlider label="γ (RBF width⁻¹)" value={gamma} min={0.1} max={10} step={0.1} onChange={setGamma} />}
            </>
          )}
          {algo === "adaboost" && <ParameterSlider label="boosting rounds" value={rounds} min={1} max={60} step={1} onChange={setRounds} />}
          <Toggle label="Shade by probability" checked={showProb} onChange={setShowProb} />
          <div className="space-y-1">
            <Segmented label={algo === "knn" ? "Shift-click adds a point of class" : "Click adds a point of class"} value={String(addClass)} onChange={(v) => setAddClass(Number(v) as 0 | 1)} options={[{ value: "0", label: "class 0" }, { value: "1", label: "class 1" }]} />
            {extra.length > 0 && <button className="btn w-full" onClick={() => setExtra([])}>Remove {extra.length} added point(s)</button>}
          </div>
          {algo === "knn" && <p className="text-[12px] text-faint">Click to place a query point and see its k nearest neighbours.</p>}
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{training accuracy}", tex: true, value: `${fmt(acc * 100, 1)}%` },
        { label: "n", tex: true, value: pts.length },
        ...(algo === "tree" ? [{ label: "\\text{depth / leaves}", tex: true, value: `${treeDepth(model.info as TreeNode)} / ${treeLeaves(model.info as TreeNode)}` }] : []),
        ...(svm ? [{ label: "\\text{support vectors}", tex: true, value: nSV }] : []),
      ]} />}
      calc={<CalcSteps steps={steps} />}
      note="Training accuracy alone rewards overfitting: a depth-12 tree or k = 1 reaches ~100% here but produces jagged boundaries that generalise poorly."
    />
  );
}

function SvmMargins({ f }: { f: (x: number, y: number) => number }) {
  const { sx, sy } = usePlot();
  const [x0, x1] = sx.domain(), [y0, y1] = sy.domain();
  const N = 60;
  const segs: string[] = [];
  for (const L of [-1, 0, 1]) {
    const xs = Array.from({ length: N + 1 }, (_, i) => x0 + ((x1 - x0) * i) / N);
    const ys = Array.from({ length: N + 1 }, (_, j) => y0 + ((y1 - y0) * j) / N);
    const v = ys.map((y) => xs.map((x) => f(x, y) - L));
    let d = "";
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const a = v[j][i], b = v[j][i + 1], c = v[j + 1][i + 1], e = v[j + 1][i];
      const P: [number, number][] = [];
      const ed = (p: number, q: number, xa: number, ya: number, xb: number, yb: number) => { if ((p > 0) !== (q > 0)) { const t = p / (p - q); P.push([xa + (xb - xa) * t, ya + (yb - ya) * t]); } };
      ed(a, b, xs[i], ys[j], xs[i + 1], ys[j]); ed(b, c, xs[i + 1], ys[j], xs[i + 1], ys[j + 1]); ed(c, e, xs[i + 1], ys[j + 1], xs[i], ys[j + 1]); ed(e, a, xs[i], ys[j + 1], xs[i], ys[j]);
      if (P.length >= 2) d += `M${sx(P[0][0])},${sy(P[0][1])}L${sx(P[1][0])},${sy(P[1][1])}`;
    }
    segs.push(d);
  }
  return (
    <g>
      <path d={segs[0]} style={{ stroke: C.a("fg", 0.55), strokeWidth: 1, strokeDasharray: "4 3", fill: "none" }} />
      <path d={segs[1]} style={{ stroke: C.fg, strokeWidth: 1.8, fill: "none" }} />
      <path d={segs[2]} style={{ stroke: C.a("fg", 0.55), strokeWidth: 1, strokeDasharray: "4 3", fill: "none" }} />
    </g>
  );
}
