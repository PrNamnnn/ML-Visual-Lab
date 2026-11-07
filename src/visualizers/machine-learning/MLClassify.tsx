"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, C, Polyline, HeatmapCanvas, useThemeRGB } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { makeDataset, knnPredict, buildTree, treePredict, sigmoid, type Point } from "@/lib/ml/models";
import { fmt, pct } from "@/lib/utilities/format";
import { BarList } from "@/visualizers/shared";

type RGB = [number, number, number];
const mix = (base: RGB, c: RGB, k: number): [number, number, number, number] => [base[0] + (c[0] - base[0]) * k, base[1] + (c[1] - base[1]) * k, base[2] + (c[2] - base[2]) * k, 255];

/** Binary logistic regression on arbitrary feature vectors by batch gradient descent. */
export function fitLogistic(X: number[][], y: number[], iters = 400, lr = 0.5, l2 = 1e-3) {
  const p = X[0].length;
  const w = Array(p + 1).fill(0);
  for (let it = 0; it < iters; it++) {
    const g = Array(p + 1).fill(0);
    X.forEach((x, i) => {
      const e = sigmoid(w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0)) - y[i];
      g[0] += e; x.forEach((v, j) => (g[j + 1] += e * v));
    });
    for (let j = 0; j <= p; j++) w[j] -= lr * (g[j] / X.length + (j ? l2 * w[j] : 0));
  }
  return (x: number[]) => sigmoid(w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0));
}

/* ------------------------------ multiclass: one-vs-rest and one-vs-one ------------------------------ */
export function MulticlassViz() {
  const [strategy, setStrategy] = useState<"ovr" | "ovo">("ovr");
  const [spread, setSpread] = useState(0.55);
  const th = useThemeRGB();
  const data = useMemo(() => {
    const r = mulberry32(6);
    const centres = [[-1.3, -0.8], [1.3, -0.8], [0, 1.2]];
    return centres.flatMap((m, c) => Array.from({ length: 40 }, () => ({ x: m[0] + spread * randn(r), y: m[1] + spread * randn(r), label: c })));
  }, [spread]);
  const models = useMemo(() => {
    const X = data.map((p) => [p.x, p.y]);
    const ovr = [0, 1, 2].map((c) => fitLogistic(X, data.map((p) => (p.label === c ? 1 : 0))));
    const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
    const ovo = pairs.map(([a, b]) => { const sub = data.filter((p) => p.label === a || p.label === b); return { a, b, f: fitLogistic(sub.map((p) => [p.x, p.y]), sub.map((p) => (p.label === b ? 1 : 0))) }; });
    return { ovr, ovo };
  }, [data]);
  const predict = (x: number, y: number) => {
    if (strategy === "ovr") {
      const s = models.ovr.map((f) => f([x, y]));
      const best = s.indexOf(Math.max(...s));
      const positives = s.filter((v) => v > 0.5).length;
      return { c: best, ambiguous: positives !== 1, conf: s[best] };
    }
    const votes = [0, 0, 0];
    models.ovo.forEach((m) => votes[m.f([x, y]) > 0.5 ? m.b : m.a]++);
    const best = votes.indexOf(Math.max(...votes));
    return { c: best, ambiguous: votes.filter((v) => v === votes[best]).length > 1, conf: votes[best] / 2 };
  };
  const acc = data.reduce((s, p) => s + (predict(p.x, p.y).c === p.label ? 1 : 0), 0) / data.length;
  const cols: RGB[] = [th.blue, th.pink, th.green];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-2.6, 2.8]} equal aspect={0.7} maxH={440} ariaLabel={`${strategy} multiclass decision regions`}
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={4} deps={[strategy, data, th]} f={(x, y) => { const p = predict(x, y); return p.ambiguous ? mix(th.panel, th.fg, 0.08) : mix(th.panel, cols[p.c], 0.28); }} />}>
            {data.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={3.4} color={[C.blue, C.pink, C.green][p.label]} stroke={C.bg} />)}
          </Plot>
          <Legend items={[{ label: "class A", color: C.blue }, { label: "class B", color: C.pink }, { label: "class C", color: C.green }, { label: strategy === "ovr" ? "zero or several classifiers say 'mine' (resolved by highest score)" : "tied votes", color: C.faint }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Strategy" value={strategy} onChange={setStrategy} options={[{ value: "ovr", label: "One-vs-Rest" }, { value: "ovo", label: "One-vs-One" }]} />
          <ParameterSlider label="class spread" value={spread} min={0.2} max={1.2} step={0.05} onChange={setSpread} />
          <p className="text-[12px] leading-relaxed text-faint">{strategy === "ovr" ? "Three binary logistic regressions, each 'class k vs everything else'. The predicted class is the one with the highest score." : "Three binary models, one per pair of classes (K(K−1)/2 in general). Each votes; the class with most votes wins."} The grey cells show where the binary decisions disagree.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{binary models}", tex: true, value: 3 },
        { label: "\\text{training accuracy}", tex: true, value: pct(acc) },
      ]} />}
      calc={<CalcSteps steps={[strategy === "ovr"
        ? { tex: String.raw`\hat y=\arg\max_k\ \sigma(\mathbf w_k^\top\mathbf x+b_k),\qquad \text{model }k\text{ trained on }y=k\text{ vs }y\ne k` }
        : { tex: String.raw`\hat y=\arg\max_k\ \#\{\text{pairwise models voting for }k\},\qquad \binom{K}{2}=3\text{ models}` }]} />}
    />
  );
}

/* ------------------------------ voting and stacking ------------------------------ */
type Combiner = "hard" | "soft" | "stack";
export function EnsembleViz({ mode = "soft" }: { mode?: Combiner }) {
  const [comb, setComb] = useState<Combiner>(mode);
  const th = useThemeRGB();
  const train = useMemo(() => makeDataset("moons", 160, 0.28, 4), []);
  const test = useMemo(() => makeDataset("moons", 400, 0.28, 44), []);
  const fitBase = (pts: Point[]) => {
    const feats = (x: number, y: number) => [x, y, x * x, x * y, y * y, x ** 3 / 2, y ** 3 / 2];
    const lg = fitLogistic(pts.map((p) => feats(p.x, p.y)), pts.map((p) => p.label), 1500, 0.5);
    const tree = buildTree(pts, 3, 4);
    return [(x: number, y: number) => lg(feats(x, y)), (x: number, y: number) => knnPredict(pts, x, y, 15).prob, (x: number, y: number) => treePredict(tree, x, y)];
  };
  const base = useMemo(() => fitBase(train), [train]);  
  // stacking: out-of-fold base predictions (5 folds) train the meta-model, so it never sees leaked predictions
  const meta = useMemo(() => {
    const oof: number[][] = train.map(() => [0, 0, 0]);
    for (let k = 0; k < 5; k++) {
      const rest = train.filter((_, i) => i % 5 !== k);
      const ms = fitBase(rest);
      train.forEach((p, i) => { if (i % 5 === k) oof[i] = ms.map((m) => m(p.x, p.y)); });
    }
    return fitLogistic(oof.map((r) => r.map((v) => 4 * (v - 0.5))), train.map((p) => p.label), 800, 0.5);
  }, [train]);  
  const ens = (x: number, y: number) => {
    const ps = base.map((m) => m(x, y));
    if (comb === "hard") return ps.filter((p) => p >= 0.5).length / 3;
    if (comb === "soft") return ps.reduce((s, p) => s + p, 0) / 3;
    return meta(ps.map((v) => 4 * (v - 0.5)));
  };
  const accOf = (f: (x: number, y: number) => number) => test.reduce((s, p) => s + ((f(p.x, p.y) >= 0.5 ? 1 : 0) === p.label ? 1 : 0), 0) / test.length;
  const accs = useMemo(() => base.map(accOf), [base]); // eslint-disable-line react-hooks/exhaustive-deps
  const accE = accOf(ens);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-1.8, 2.8]} y={[-1.4, 1.8]} equal aspect={0.6} maxH={400} ariaLabel="ensemble decision regions"
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={5} deps={[comb, th]} f={(x, y) => { const p = ens(x, y); return mix(th.panel, p >= 0.5 ? th.pink : th.blue, 0.1 + 0.32 * Math.abs(p - 0.5) * 2); }} />}>
            {train.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={3.2} color={p.label ? C.pink : C.blue} stroke={C.bg} />)}
          </Plot>
          <BarList max={1} items={[
            { label: "logistic (cubic features)", value: accs[0], color: C.a("fg", 0.35) },
            { label: "15-NN", value: accs[1], color: C.a("fg", 0.35) },
            { label: "tree (depth 3)", value: accs[2], color: C.a("fg", 0.35) },
            { label: comb === "stack" ? "stacked ensemble" : `${comb} voting`, value: accE, color: C.green },
          ]} />
          <p className="text-[12px] text-faint">Test accuracy on 400 fresh points.</p>
        </div>
      }
      controls={
        <>
          <Segmented label="Combiner" value={comb} onChange={setComb} options={[{ value: "hard", label: "Hard vote" }, { value: "soft", label: "Soft vote" }, { value: "stack", label: "Stacking" }]} />
          <p className="text-[12px] leading-relaxed text-faint">{comb === "hard" ? "Each model casts one vote for a class; the majority wins." : comb === "soft" ? "Average the predicted probabilities, then threshold at 0.5. Uses each model's confidence." : "A logistic-regression meta-model learns how much to trust each base model, trained on 5-fold out-of-fold predictions so it never sees predictions made on a model's own training data."}</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{best single model}", tex: true, value: pct(Math.max(...accs)) },
        { label: "\\text{ensemble}", tex: true, value: pct(accE), color: C.green },
      ]} />}
      calc={<CalcSteps steps={[comb === "hard"
        ? { tex: String.raw`\hat y=\operatorname{mode}\{\hat y_1,\hat y_2,\hat y_3\}` }
        : comb === "soft" ? { tex: String.raw`\hat p=\tfrac13(\hat p_1+\hat p_2+\hat p_3),\quad \hat y=[\hat p\ge0.5]` }
        : { tex: String.raw`\hat p=\sigma\big(w_0+\textstyle\sum_m w_m\,\hat p_m^{\text{(out-of-fold)}}\big)` }]} />}
    />
  );
}

/* ------------------------------ LDA vs QDA ------------------------------ */
type G2 = { mu: [number, number]; S: [[number, number], [number, number]] };
const inv2 = (S: G2["S"]): G2["S"] => { const d = S[0][0] * S[1][1] - S[0][1] * S[1][0]; return [[S[1][1] / d, -S[0][1] / d], [-S[1][0] / d, S[0][0] / d]]; };
const det2 = (S: G2["S"]) => S[0][0] * S[1][1] - S[0][1] * S[1][0];
const logN = (x: number, y: number, g: G2) => { const I = inv2(g.S), dx = x - g.mu[0], dy = y - g.mu[1]; return -0.5 * (dx * (I[0][0] * dx + I[0][1] * dy) + dy * (I[1][0] * dx + I[1][1] * dy)) - 0.5 * Math.log(det2(g.S)); };
function ellipse(g: G2): [number, number][] {
  const a = g.S[0][0], b = g.S[0][1], c = g.S[1][1];
  const l11 = Math.sqrt(a), l21 = b / l11, l22 = Math.sqrt(Math.max(1e-9, c - l21 * l21)); // Cholesky
  return Array.from({ length: 61 }, (_, i) => { const t = (i / 60) * 2 * Math.PI, u = 2 * Math.cos(t), v = 2 * Math.sin(t); return [g.mu[0] + l11 * u, g.mu[1] + l21 * u + l22 * v]; });
}
export function DiscriminantViz() {
  const [model, setModel] = useState<"lda" | "qda">("qda");
  const [ratio, setRatio] = useState(3);
  const th = useThemeRGB();
  const gen = (seed: number, n: number) => {
    const r = mulberry32(seed);
    const out: Point[] = [];
    for (let i = 0; i < n; i++) out.push({ x: -0.6 + 0.45 * randn(r), y: 0.45 * randn(r), label: 0 });
    for (let i = 0; i < n; i++) { const u = randn(r) * 0.45 * ratio, v = randn(r) * 0.3; out.push({ x: 0.9 + 0.6 * u + 0.5 * v, y: 0.5 * u - 0.6 * v, label: 1 }); }
    return out;
  };
  const train = useMemo(() => gen(3, 60), [ratio]); // eslint-disable-line react-hooks/exhaustive-deps
  const test = useMemo(() => gen(33, 300), [ratio]); // eslint-disable-line react-hooks/exhaustive-deps
  const fit = useMemo(() => {
    const g = [0, 1].map((l) => {
      const P = train.filter((p) => p.label === l), n = P.length;
      const mu: [number, number] = [P.reduce((s, p) => s + p.x, 0) / n, P.reduce((s, p) => s + p.y, 0) / n];
      const S: G2["S"] = [[0, 0], [0, 0]];
      P.forEach((p) => { const dx = p.x - mu[0], dy = p.y - mu[1]; S[0][0] += dx * dx / (n - 1); S[0][1] += dx * dy / (n - 1); S[1][1] += dy * dy / (n - 1); });
      S[1][0] = S[0][1];
      return { mu, S, n };
    });
    const N = g[0].n + g[1].n;
    const pooled: G2["S"] = [0, 1].map((i) => [0, 1].map((j) => ((g[0].n - 1) * g[0].S[i][j] + (g[1].n - 1) * g[1].S[i][j]) / (N - 2))) as G2["S"];
    return { g, pooled };
  }, [train]);
  const comps: G2[] = fit.g.map((c) => ({ mu: c.mu, S: model === "lda" ? fit.pooled : c.S }));
  const post = (x: number, y: number) => { const a = logN(x, y, comps[0]), b = logN(x, y, comps[1]); return 1 / (1 + Math.exp(a - b)); }; // equal priors
  const acc = test.reduce((s, p) => s + ((post(p.x, p.y) >= 0.5 ? 1 : 0) === p.label ? 1 : 0), 0) / test.length;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-2.5, 3.5]} y={[-2.6, 2.6]} equal aspect={0.7} maxH={440} ariaLabel={`${model} posterior and decision boundary`}
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={4} deps={[model, ratio, th]} f={(x, y) => { const p = post(x, y); return mix(th.panel, p >= 0.5 ? th.pink : th.blue, 0.1 + 0.32 * Math.abs(p - 0.5) * 2); }} />}>
            {comps.map((g, i) => <Polyline key={i} pts={ellipse(g)} style={{ stroke: i ? C.pink : C.blue, strokeWidth: 1.4, strokeDasharray: "5 3", fill: "none" }} />)}
            {train.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={3.2} color={p.label ? C.pink : C.blue} stroke={C.bg} />)}
          </Plot>
          <Legend items={[{ label: "class 0", color: C.blue }, { label: "class 1", color: C.pink }, { label: `2σ ellipses of the ${model === "lda" ? "shared (pooled)" : "per-class"} covariance`, color: C.muted, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Model" value={model} onChange={setModel} options={[{ value: "lda", label: "LDA (shared Σ)" }, { value: "qda", label: "QDA (Σ per class)" }]} />
          <ParameterSlider label="class-1 elongation" value={ratio} min={1} max={4} step={0.1} onChange={setRatio} />
          <p className="text-[12px] leading-relaxed text-faint">Both model each class as a Gaussian and classify with Bayes&apos; rule. LDA forces one shared covariance, so its boundary is a straight line; QDA gives each class its own covariance and the boundary becomes a curve.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{test accuracy}", tex: true, value: pct(acc), color: C.green },
        { label: "\\text{covariance parameters}", tex: true, value: model === "lda" ? 3 : 6 },
        { label: "\\hat{\\boldsymbol\\mu}_0,\\ \\hat{\\boldsymbol\\mu}_1", tex: true, value: `(${fmt(fit.g[0].mu[0], 2)}, ${fmt(fit.g[0].mu[1], 2)}), (${fmt(fit.g[1].mu[0], 2)}, ${fmt(fit.g[1].mu[1], 2)})` },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\delta_k(\mathbf x)=-\tfrac12(\mathbf x-\boldsymbol\mu_k)^\top\Sigma_k^{-1}(\mathbf x-\boldsymbol\mu_k)-\tfrac12\log|\Sigma_k|+\log\pi_k`, note: model === "lda" ? "with Σ₀ = Σ₁ = Σ the quadratic terms cancel → linear boundary" : "different Σₖ keep the quadratic term → curved boundary" },
        { tex: String.raw`P(y=1\mid\mathbf x)=\frac{1}{1+e^{\delta_0(\mathbf x)-\delta_1(\mathbf x)}}\quad(\pi_0=\pi_1)` },
      ]} />}
    />
  );
}
