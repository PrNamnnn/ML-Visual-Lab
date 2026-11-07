"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, C, Polyline, FnPath, HeatmapCanvas, useThemeRGB } from "@/components/visualization/Plot";
import { Bars } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Select } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { makeDataset, kmeansAssign, kmeansUpdate, type P2, type Point } from "@/lib/ml/models";
import { histogram } from "@/lib/statistics/descriptive";
import { fmt, pct, tex } from "@/lib/utilities/format";
import { Heat, MiniTable, BarList } from "@/visualizers/shared";
import { fitLogistic } from "@/visualizers/machine-learning/MLClassify";

const sd = (v: number[]) => { const m = v.reduce((s, x) => s + x, 0) / v.length; return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length); };

/* ------------------------------ parameter initialisation ------------------------------ */
type Init = "small" | "unit" | "xavier" | "he";
export function InitViz() {
  const [init, setInit] = useState<Init>("xavier");
  const [act, setAct] = useState<"tanh" | "relu">("tanh");
  const [depth, setDepth] = useState(12);
  const width = 64;
  const res = useMemo(() => {
    const r = mulberry32(1);
    const std = init === "small" ? 0.01 : init === "unit" ? 1 : init === "xavier" ? Math.sqrt(1 / width) : Math.sqrt(2 / width);
    const Ws = Array.from({ length: depth }, () => Array.from({ length: width }, () => Array.from({ length: width }, () => std * randn(r))));
    let h = Array.from({ length: 100 }, () => Array.from({ length: width }, () => randn(r)));
    const actStd: number[] = [], pre: number[][] = [];
    const f = (z: number) => (act === "tanh" ? Math.tanh(z) : Math.max(0, z));
    const zs: number[][][] = [];
    for (const W of Ws) {
      const z = h.map((x) => W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0)));
      zs.push(z);
      h = z.map((row) => row.map(f));
      actStd.push(sd(h.flat()));
      pre.push(z.flat());
    }
    // backward: start with unit gradients at the top and push them down through the layers
    let g = h.map((row) => row.map(() => 1));
    const gradStd: number[] = Array(depth).fill(0);
    for (let l = depth - 1; l >= 0; l--) {
      const dz = g.map((row, i) => row.map((v, j) => v * (act === "tanh" ? 1 - Math.tanh(zs[l][i][j]) ** 2 : zs[l][i][j] > 0 ? 1 : 0)));
      g = dz.map((row) => Ws[l][0].map((_, j) => row.reduce((s, v, k) => s + v * Ws[l][k][j], 0)));
      gradStd[l] = sd(g.flat());
    }
    return { actStd, gradStd, last: h.flat(), std };
  }, [init, act, depth]);
  const lo = act === "tanh" ? -1.05 : 0, hi = act === "tanh" ? 1.05 : Math.max(0.5, Math.max(...res.last) * 0.8);
  const bins = histogram(res.last, lo, hi, 40);
  const cap = (v: number) => Math.min(v, 1e6);
  return (
    <LabShell
      viz={
        <div className="grid gap-3 lg:grid-cols-2">
          <div className="space-y-1.5">
            <div className="text-[12px] text-faint">Standard deviation of activations per layer (forward pass)</div>
            <BarList digits={4} items={res.actStd.map((v, i) => ({ label: `layer ${i + 1}`, value: cap(v), color: v < 0.02 || v > 20 ? C.red : C.blue }))} />
          </div>
          <div className="space-y-1.5">
            <div className="text-[12px] text-faint">Standard deviation of gradients per layer (backward pass)</div>
            <BarList digits={4} items={res.gradStd.map((v, i) => ({ label: `layer ${i + 1}`, value: cap(v), color: v < 0.02 || v > 50 ? C.red : C.pink }))} />
          </div>
          <div className="lg:col-span-2">
            <Plot x={[lo, hi]} y={[0, Math.max(...bins.map((b) => b.density)) * 1.1 || 1]} aspect={0.22} minH={120} maxH={170} xLabel={`activations of the last layer (${act})`} ariaLabel="histogram of last-layer activations">
              <Bars bins={bins} />
            </Plot>
          </div>
        </div>
      }
      controls={
        <>
          <Select label="Initialisation" value={init} onChange={setInit} options={[{ value: "small", label: "N(0, 0.01²) — too small" }, { value: "unit", label: "N(0, 1) — too large" }, { value: "xavier", label: "Xavier / Glorot: Var = 1/n" }, { value: "he", label: "He / Kaiming: Var = 2/n" }]} />
          <Segmented label="Activation" value={act} onChange={setAct} options={[{ value: "tanh", label: "tanh" }, { value: "relu", label: "ReLU" }]} />
          <ParameterSlider label="depth (layers of width 64)" value={depth} min={2} max={20} step={1} onChange={setDepth} />
          <p className="text-[12px] leading-relaxed text-faint">With weights too small the signal shrinks to zero layer after layer; too large and tanh saturates at ±1 (ReLU explodes). Xavier keeps tanh stable; ReLU zeroes half its inputs, so it needs twice the variance — He initialisation.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\sigma_w", tex: true, value: fmt(res.std, 4) },
        { label: "\\text{act. std, last layer}", tex: true, value: fmt(res.actStd[depth - 1], 4) },
        { label: "\\text{grad std, first layer}", tex: true, value: fmt(res.gradStd[0], 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\operatorname{Var}(z)=n_{\text{in}}\operatorname{Var}(w)\operatorname{Var}(x)\ \Rightarrow\ \operatorname{Var}(w)=\frac{1}{n_{\text{in}}}\ \text{(Xavier)},\quad\frac{2}{n_{\text{in}}}\ \text{(He, ReLU)}`, note: "keep the variance of each layer's output equal to its input's" },
        { tex: String.raw`n_{\text{in}}=64:\ \sigma_{\text{Xavier}}=${tex(Math.sqrt(1 / 64), 4)},\ \sigma_{\text{He}}=${tex(Math.sqrt(2 / 64), 4)}` },
      ]} />}
    />
  );
}

/* ------------------------------ batch normalisation / layer normalisation ------------------------------ */
export function NormViz({ mode: initial = "batch" }: { mode?: "batch" | "layer" }) {
  const [mode, setMode] = useState<"batch" | "layer">(initial);
  const [gamma, setGamma] = useState(1);
  const [beta, setBeta] = useState(0);
  const [bs, setBs] = useState(8);
  const X = useMemo(() => { const r = mulberry32(3); const scale = [1, 10, 0.1, 3, 0.5, 6], shift = [0, 50, -1, 5, 2, -20]; return Array.from({ length: 8 }, (_, i) => scale.map((s, j) => shift[j] + s * randn(r) + (i === 2 ? 4 * s : 0))); }, []);
  const eps = 1e-5;
  const norm = X.map((row, i) => row.map((v, j) => {
    const vals = mode === "batch" ? X.map((r) => r[j]) : X[i];
    const m = vals.reduce((s, x) => s + x, 0) / vals.length, va = vals.reduce((s, x) => s + (x - m) ** 2, 0) / vals.length;
    return gamma * (v - m) / Math.sqrt(va + eps) + beta;
  }));
  // how noisy are batch statistics? spread of the batch mean of feature 2 across 300 random batches
  const noise = useMemo(() => {
    const r = mulberry32(9);
    const means = Array.from({ length: 300 }, () => { let s = 0; for (let i = 0; i < bs; i++) s += 50 + 10 * randn(r); return s / bs; });
    return sd(means);
  }, [bs]);
  return (
    <LabShell
      viz={
        <div className="grid gap-4 xl:grid-cols-2">
          <div className="space-y-1.5"><div className="text-[12px] text-faint">Pre-activations: 8 examples (rows) × 6 features (columns)</div><Heat M={X} rowLabels={X.map((_, i) => `x${i + 1}`)} colLabels={["f1", "f2", "f3", "f4", "f5", "f6"]} digits={1} signed /></div>
          <div className="space-y-1.5"><div className="text-[12px] text-faint">After {mode === "batch" ? "batch norm (normalise each column)" : "layer norm (normalise each row)"}</div><Heat M={norm} rowLabels={X.map((_, i) => `x${i + 1}`)} colLabels={["f1", "f2", "f3", "f4", "f5", "f6"]} digits={2} signed max={2.5} /></div>
        </div>
      }
      controls={
        <>
          <Segmented label="Normalise over" value={mode} onChange={setMode} options={[{ value: "batch", label: "Batch (per feature)" }, { value: "layer", label: "Layer (per example)" }]} />
          <ParameterSlider label="scale γ" value={gamma} min={0} max={3} step={0.05} onChange={setGamma} />
          <ParameterSlider label="shift β" value={beta} min={-2} max={2} step={0.05} onChange={setBeta} />
          {mode === "batch" && <ParameterSlider label="batch size (for the noise readout)" value={bs} min={2} max={128} step={1} onChange={setBs} />}
          <p className="text-[12px] leading-relaxed text-faint">{mode === "batch" ? "Batch norm standardises every feature using the mean and variance across the batch. Feature f2 (values around 50) and f3 (around −1) end up on the same scale. Statistics depend on the other examples in the batch, so tiny batches give noisy estimates." : "Layer norm standardises every example across its own features, so it does not depend on the batch at all — it behaves the same at batch size 1, during inference, and for sequences of any length. That is why Transformers use it."}</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{statistics computed over}", tex: true, value: mode === "batch" ? "8 examples, per feature" : "6 features, per example" },
        ...(mode === "batch" ? [{ label: `\\operatorname{sd}(\\hat\\mu_{\\text{batch}}),\\ B=${bs}`, tex: true, value: fmt(noise, 3), hint: "for a feature with true mean 50 and std 10; ≈ 10/√B" }] : []),
        { label: "\\text{learnable params per layer}", tex: true, value: "γ and β: 2 × features" },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: mode === "batch" ? String.raw`\hat x_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}},\quad\mu_j=\frac1B\sum_ix_{ij}` : String.raw`\hat x_{ij}=\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}},\quad\mu_i=\frac1d\sum_jx_{ij}` },
        { tex: String.raw`y=\gamma\hat x+\beta=${tex(gamma, 2)}\,\hat x${beta >= 0 ? "+" : ""}${tex(beta, 2)}`, note: mode === "batch" ? "at inference, running averages of μ and σ² replace the batch statistics" : "identical at training and inference" },
      ]} />}
    />
  );
}

/* ------------------------------ dropout ------------------------------ */
export function DropoutViz() {
  const [p, setP] = useState(0.5);
  const [seed, setSeed] = useState(1);
  const layers = [4, 8, 8, 2];
  const r = mulberry32(seed * 7 + Math.round(p * 100));
  const keep = layers.map((n, l) => Array.from({ length: n }, () => (l === 0 || l === layers.length - 1 ? true : r() >= p)));
  // inverted dropout keeps the expected activation unchanged: simulate one unit's input sum over many masks
  const sim = useMemo(() => {
    const rr = mulberry32(5), a = Array.from({ length: 8 }, () => 0.5 + rr()), w = Array.from({ length: 8 }, () => randn(rr));
    const full = a.reduce((s, v, i) => s + v * w[i], 0);
    const samples = Array.from({ length: 2000 }, () => a.reduce((s, v, i) => s + (rr() >= p ? v / (1 - p) : 0) * w[i], 0));
    return { full, mean: samples.reduce((s, x) => s + x, 0) / samples.length, sd: sd(samples), samples };
  }, [p]);
  const X = (l: number) => 0.6 + l * 2.2, Y = (l: number, i: number) => (i - (layers[l] - 1) / 2) * 0.62;
  const lo = Math.min(...sim.samples), hi = Math.max(...sim.samples);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[0, 7.8]} y={[-2.6, 2.6]} aspect={0.45} maxH={300} axes={false} grid={false} ariaLabel="network with dropped units">
            {layers.slice(0, -1).map((n, l) => Array.from({ length: n }, (_, i) => Array.from({ length: layers[l + 1] }, (_, j) => <Polyline key={`${l}-${i}-${j}`} pts={[[X(l), Y(l, i)], [X(l + 1), Y(l + 1, j)]]} style={{ stroke: keep[l][i] && keep[l + 1][j] ? C.a("fg", 0.28) : C.a("fg", 0.04), strokeWidth: 1 }} />)))}
            {layers.map((n, l) => Array.from({ length: n }, (_, i) => <Dot key={`n${l}-${i}`} p={[X(l), Y(l, i)]} r={8} color={keep[l][i] ? (l === 0 ? C.blue : l === layers.length - 1 ? C.green : C.pink) : C.a("fg", 0.12)} stroke={C.bg} />))}
          </Plot>
          <Plot x={[lo, hi]} y={[0, Math.max(...histogram(sim.samples, lo, hi, 40).map((b) => b.density)) * 1.15]} aspect={0.22} minH={120} maxH={160} xLabel="one unit's input over 2000 dropout masks" ariaLabel="distribution of a unit's input under dropout">
            <Bars bins={histogram(sim.samples, lo, hi, 40)} />
            <Polyline pts={[[sim.full, 0], [sim.full, 99]]} style={{ stroke: C.green, strokeWidth: 2 }} />
          </Plot>
          <Legend items={[{ label: "kept hidden unit", color: C.pink }, { label: "dropped (output 0 this step)", color: C.a("fg", 0.25) }, { label: "value without dropout (test time)", color: C.green }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="drop probability p" value={p} min={0} max={0.9} step={0.05} onChange={setP} />
          <button className="btn" onClick={() => setSeed((s) => s + 1)}>Sample a new mask</button>
          <p className="text-[12px] leading-relaxed text-faint">Each training step drops every hidden unit with probability p and scales the survivors by 1/(1 − p). The network cannot rely on any single unit, which acts like training a huge ensemble of thinned networks. At test time nothing is dropped.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{kept hidden units}", tex: true, value: `${keep.slice(1, -1).flat().filter(Boolean).length} / 16` },
        { label: "\\text{no-dropout value}", tex: true, value: fmt(sim.full, 3), color: C.green },
        { label: "\\text{mean with dropout}", tex: true, value: fmt(sim.mean, 3), hint: "≈ the no-dropout value thanks to the 1/(1−p) scaling" },
        { label: "\\text{std with dropout}", tex: true, value: fmt(sim.sd, 3) },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\tilde h_i=\frac{m_i}{1-p}h_i,\quad m_i\sim\text{Bernoulli}(1-${tex(p, 2)})\ \Rightarrow\ \mathbb E[\tilde h_i]=h_i`, note: "inverted dropout" }]} />}
    />
  );
}

/* ------------------------------ early stopping ------------------------------ */
export function EarlyStopViz() {
  const [patience, setPatience] = useState(50);
  const [lr, setLr] = useState(0.3);
  const run = useMemo(() => {
    const r = mulberry32(8), f = (x: number) => Math.sin(2.5 * x);
    const mk = (n: number) => Array.from({ length: n }, () => { const x = 2 * r() - 1; return [x, f(x) + 0.25 * randn(r)] as [number, number]; });
    const tr = mk(18), va = mk(60);
    const D = 14, phi = (x: number) => Array.from({ length: D }, (_, k) => Math.cos(k * Math.acos(Math.max(-1, Math.min(1, x))))); // Chebyshev features
    const w = Array(D).fill(0);
    const hist: { tr: number; va: number; w: number[] }[] = [];
    const mse = (d: [number, number][]) => d.reduce((s, [x, y]) => s + (phi(x).reduce((a, v, k) => a + v * w[k], 0) - y) ** 2, 0) / d.length;
    for (let e = 0; e < 1500; e++) {
      const g = Array(D).fill(0);
      tr.forEach(([x, y]) => { const ph = phi(x), e2 = ph.reduce((a, v, k) => a + v * w[k], 0) - y; ph.forEach((v, k) => (g[k] += (2 * e2 * v) / tr.length)); });
      for (let k = 0; k < D; k++) w[k] -= lr * g[k];
      hist.push({ tr: mse(tr), va: mse(va), w: w.slice() });
    }
    return { hist, tr, phi, f };
  }, [lr]);
  // early stopping rule: stop after `patience` epochs without a new best validation loss
  let best = 0, stop = run.hist.length - 1;
  for (let e = 1; e < run.hist.length; e++) { if (run.hist[e].va < run.hist[best].va) best = e; else if (e - best >= patience) { stop = e; break; } }
  const ymax = Math.max(0.6, run.hist[5].va);
  const pred = (w: number[]) => (x: number) => run.phi(x).reduce((a, v, k) => a + v * w[k], 0);
  return (
    <LabShell
      viz={
        <div className="grid gap-3 lg:grid-cols-2">
          <Plot x={[0, run.hist.length]} y={[0, ymax]} aspect={0.75} maxH={300} xLabel="epoch" yLabel="MSE" ariaLabel="training and validation loss curves">
            <Polyline pts={run.hist.map((h, e) => [e, Math.min(h.tr, ymax)])} style={{ stroke: C.blue, strokeWidth: 1.8 }} />
            <Polyline pts={run.hist.map((h, e) => [e, Math.min(h.va, ymax)])} style={{ stroke: C.amber, strokeWidth: 1.8 }} />
            <Polyline pts={[[best, 0], [best, ymax]]} style={{ stroke: C.green, strokeWidth: 1.5 }} />
            <Polyline pts={[[stop, 0], [stop, ymax]]} style={{ stroke: C.red, strokeWidth: 1.2, strokeDasharray: "4 3" }} />
          </Plot>
          <Plot x={[-1, 1]} y={[-2, 2]} aspect={0.75} maxH={300} ariaLabel="model at the best epoch versus the final epoch">
            <FnPath f={run.f} style={{ stroke: C.a("fg", 0.3), strokeDasharray: "4 3" }} />
            <FnPath f={pred(run.hist[run.hist.length - 1].w)} style={{ stroke: C.red, strokeWidth: 1.6 }} />
            <FnPath f={pred(run.hist[best].w)} style={{ stroke: C.green, strokeWidth: 2.4 }} />
            {run.tr.map((p, i) => <Dot key={i} p={p} r={3.4} color={C.blue} />)}
          </Plot>
          <div className="lg:col-span-2"><Legend items={[{ label: "training loss", color: C.blue }, { label: "validation loss", color: C.amber }, { label: `best epoch ${best} (weights restored)`, color: C.green }, { label: `stopped at ${stop}`, color: C.red, dashed: true }, { label: "fit after all 1500 epochs", color: C.red }]} /></div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="patience (epochs)" value={patience} min={5} max={400} step={5} onChange={setPatience} />
          <ParameterSlider label="learning rate" value={lr} min={0.05} max={0.5} step={0.01} onChange={setLr} />
          <p className="text-[12px] leading-relaxed text-faint">A 14-parameter model on 18 noisy points. Training loss keeps falling, but after a while the model starts fitting noise and validation loss rises. Early stopping keeps the weights from the best validation epoch.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{best val MSE}", tex: true, value: fmt(run.hist[best].va, 4), color: C.green },
        { label: "\\text{final val MSE}", tex: true, value: fmt(run.hist[run.hist.length - 1].va, 4), color: C.red },
        { label: "\\text{epochs saved}", tex: true, value: run.hist.length - 1 - stop },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{stop at the first }e\text{ with }e-\arg\min_{e'\le e}L_{\text{val}}(e')\ge${patience};\ \text{restore }\theta_{e^*}`, note: "early stopping limits how far the weights travel from their initialisation — similar in effect to L2 regularisation" }]} />}
    />
  );
}

/* ------------------------------ CNN architectures ------------------------------ */
type Layer = { kind: "conv" | "pool" | "fc" | "res"; k?: number; c?: number; s?: number; p?: number; out?: number; name: string };
const ARCHS: Record<string, { label: string; input: [number, number]; layers: Layer[] }> = {
  lenet: { label: "LeNet-5 (1998)", input: [32, 1], layers: [
    { kind: "conv", k: 5, c: 6, s: 1, p: 0, name: "conv 5×5, 6" }, { kind: "pool", k: 2, s: 2, name: "avg-pool 2" }, { kind: "conv", k: 5, c: 16, s: 1, p: 0, name: "conv 5×5, 16" }, { kind: "pool", k: 2, s: 2, name: "avg-pool 2" },
    { kind: "fc", out: 120, name: "fc 120" }, { kind: "fc", out: 84, name: "fc 84" }, { kind: "fc", out: 10, name: "fc 10" }] },
  vgg: { label: "VGG-style block stack", input: [224, 3], layers: [
    { kind: "conv", k: 3, c: 64, s: 1, p: 1, name: "conv 3×3, 64" }, { kind: "conv", k: 3, c: 64, s: 1, p: 1, name: "conv 3×3, 64" }, { kind: "pool", k: 2, s: 2, name: "max-pool 2" },
    { kind: "conv", k: 3, c: 128, s: 1, p: 1, name: "conv 3×3, 128" }, { kind: "conv", k: 3, c: 128, s: 1, p: 1, name: "conv 3×3, 128" }, { kind: "pool", k: 2, s: 2, name: "max-pool 2" },
    { kind: "conv", k: 3, c: 256, s: 1, p: 1, name: "conv 3×3, 256" }, { kind: "pool", k: 2, s: 2, name: "max-pool 2" }, { kind: "fc", out: 10, name: "fc 10" }] },
  resnet: { label: "ResNet-18 stem + stages", input: [224, 3], layers: [
    { kind: "conv", k: 7, c: 64, s: 2, p: 3, name: "conv 7×7, 64, /2" }, { kind: "pool", k: 3, s: 2, p: 1, name: "max-pool 3, /2" },
    { kind: "res", c: 64, s: 1, name: "2 × basic block, 64" }, { kind: "res", c: 128, s: 2, name: "2 × basic block, 128, /2" }, { kind: "res", c: 256, s: 2, name: "2 × basic block, 256, /2" }, { kind: "res", c: 512, s: 2, name: "2 × basic block, 512, /2" },
    { kind: "pool", k: 7, s: 7, name: "global avg-pool" }, { kind: "fc", out: 1000, name: "fc 1000" }] },
};
export function ArchViz() {
  const [arch, setArch] = useState<keyof typeof ARCHS>("lenet");
  const A = ARCHS[arch];
  const rows: { name: string; shape: string; params: number; rf: number }[] = [];
  let [hw, ch] = A.input, flat = 0, rf = 1, jump = 1, total = 0;
  rows.push({ name: "input", shape: `${hw}×${hw}×${ch}`, params: 0, rf: 1 });
  for (const L of A.layers) {
    let params = 0;
    if (L.kind === "conv") { params = L.k! * L.k! * ch * L.c! + L.c!; rf += (L.k! - 1) * jump; jump *= L.s!; hw = Math.floor((hw + 2 * L.p! - L.k!) / L.s!) + 1; ch = L.c!; }
    else if (L.kind === "pool") { rf += (L.k! - 1) * jump; jump *= L.s!; hw = Math.floor((hw + 2 * (L.p ?? 0) - L.k!) / L.s!) + 1; }
    else if (L.kind === "res") {
      // two basic blocks (each: two 3×3 convs + BN); a 1×1 projection when the shape changes
      const cin = ch, c = L.c!, s = L.s!;
      params = (9 * cin * c + 9 * c * c) + (9 * c * c + 9 * c * c) + 8 * c + (s !== 1 || cin !== c ? cin * c + 2 * c : 0);
      rf += 2 * jump * 1 + 2 * jump * s * 1 + 2 * jump * s * 2; jump *= s; hw = Math.floor((hw - 1) / s) + 1; ch = c;
    } else { const inp = flat || hw * hw * ch; params = inp * L.out! + L.out!; flat = L.out!; }
    total += params;
    rows.push({ name: L.name, shape: L.kind === "fc" ? `${L.out}` : `${hw}×${hw}×${ch}`, params, rf: L.kind === "fc" ? NaN : rf });
  }
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <MiniTable head={["layer", "output shape", "parameters", "receptive field"]} rows={rows.map((r) => [r.name, r.shape, r.params.toLocaleString("en-US"), Number.isFinite(r.rf) ? `${r.rf}×${r.rf}` : "—"])} />
          <p className="text-[12px] text-faint">Parameter counts include biases (and batch-norm γ, β for ResNet). Receptive field = input pixels that can influence one output value.</p>
        </div>
      }
      controls={
        <>
          <Select label="Architecture" value={arch} onChange={setArch} options={Object.entries(ARCHS).map(([k, v]) => ({ value: k as keyof typeof ARCHS, label: v.label }))} />
          <p className="text-[12px] leading-relaxed text-faint">CNNs alternate convolutions (local patterns, shared weights) with downsampling. Spatial size shrinks while channels grow, and each layer sees a larger patch of the image. ResNets add skip connections so that 18–152 layers still train.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{total parameters}", tex: true, value: total.toLocaleString("en-US") },
        { label: "\\text{layers listed}", tex: true, value: A.layers.length },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`H_{\text{out}}=\Big\lfloor\frac{H_{\text{in}}+2p-k}{s}\Big\rfloor+1,\qquad\#\text{params}_{\text{conv}}=k^2C_{\text{in}}C_{\text{out}}+C_{\text{out}}` },
        { tex: String.raw`\text{RF}_{l}=\text{RF}_{l-1}+(k_l-1)\prod_{i<l}s_i` },
      ]} />}
    />
  );
}

/* ------------------------------ transfer learning ------------------------------ */
export function TransferViz() {
  const [nLab, setNLab] = useState(8);
  const th = useThemeRGB();
  // "pre-training": learn 24 RBF prototype features by k-means on a large unlabelled source set
  const protos = useMemo(() => {
    const src = makeDataset("moons", 1200, 0.12, 77).map((p) => [p.x, p.y] as P2);
    let Cc: P2[] = src.slice(0, 24), a = kmeansAssign(src, Cc);
    for (let t = 0; t < 30; t++) { Cc = kmeansUpdate(src, a, Cc); a = kmeansAssign(src, Cc); }
    return Cc;
  }, []);
  const feat = (x: number, y: number) => protos.map((c) => Math.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / 0.18));
  const test = useMemo(() => makeDataset("moons", 400, 0.12, 91), []);
  const trainSet = (n: number, seed: number): Point[] => { const all = makeDataset("moons", 400, 0.12, seed); return [...all.filter((p) => p.label === 0).slice(0, n / 2), ...all.filter((p) => p.label === 1).slice(0, n / 2)]; };
  const evalBoth = (n: number, seed: number) => {
    const tr = trainSet(n, seed);
    const scratch = fitLogistic(tr.map((p) => [p.x, p.y]), tr.map((p) => p.label), 600, 0.5);
    const transfer = fitLogistic(tr.map((p) => feat(p.x, p.y)), tr.map((p) => p.label), 600, 1, 1e-3);
    const acc = (f: (p: Point) => number) => test.reduce((s, p) => s + ((f(p) >= 0.5 ? 1 : 0) === p.label ? 1 : 0), 0) / test.length;
    return { a: acc((p) => scratch([p.x, p.y])), b: acc((p) => transfer(feat(p.x, p.y))), scratch, transfer };
  };
  const curve = useMemo(() => [2, 4, 8, 16, 32, 64].map((n) => { const rs = [1, 2, 3, 4, 5].map((s) => evalBoth(n, s)); return { n, a: rs.reduce((s, r) => s + r.a, 0) / 5, b: rs.reduce((s, r) => s + r.b, 0) / 5 }; }), []); // eslint-disable-line react-hooks/exhaustive-deps
  const cur = useMemo(() => evalBoth(nLab, 1), [nLab]); // eslint-disable-line react-hooks/exhaustive-deps
  const tr = trainSet(nLab, 1);
  const mixc = (c: number[], k: number): [number, number, number, number] => [th.panel[0] + (c[0] - th.panel[0]) * k, th.panel[1] + (c[1] - th.panel[1]) * k, th.panel[2] + (c[2] - th.panel[2]) * k, 255];
  return (
    <LabShell
      viz={
        <div className="grid gap-3 lg:grid-cols-[1.3fr_1fr]">
          <Plot x={[-1.6, 2.6]} y={[-1.2, 1.6]} equal aspect={0.7} maxH={340} ariaLabel="decision regions of the transfer model"
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={5} deps={[nLab, th]} f={(x, y) => { const p = cur.transfer(feat(x, y)); return mixc(p >= 0.5 ? th.pink : th.blue, 0.1 + 0.3 * Math.abs(p - 0.5) * 2); }} />}>
            {protos.map((c, i) => <Dot key={"c" + i} p={c} r={2.5} color={C.amber} />)}
            {tr.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={6} color={p.label ? C.pink : C.blue} stroke={C.fg} />)}
          </Plot>
          <Plot x={[2, 64]} y={[0.5, 1]} aspect={0.85} maxH={300} xLabel="labelled target examples" yLabel="test accuracy" ariaLabel="accuracy against the number of labels">
            <Polyline pts={curve.map((c) => [c.n, c.a])} style={{ stroke: C.blue, strokeWidth: 2 }} />
            <Polyline pts={curve.map((c) => [c.n, c.b])} style={{ stroke: C.pink, strokeWidth: 2.2 }} />
            {curve.map((c, i) => <Dot key={i} p={[c.n, c.b]} r={3} color={C.pink} />)}
          </Plot>
          <div className="lg:col-span-2"><Legend items={[{ label: "pre-trained features (24 prototypes from 1 200 unlabelled source points)", color: C.amber }, { label: "linear model on raw inputs (from scratch)", color: C.blue }, { label: "linear model on frozen pre-trained features", color: C.pink }]} /></div>
        </div>
      }
      controls={
        <>
          <Select label="Labelled examples" value={String(nLab)} onChange={(v) => setNLab(Number(v))} options={[2, 4, 8, 16, 32, 64].map((n) => ({ value: String(n), label: String(n) }))} />
          <p className="text-[12px] leading-relaxed text-faint">A representation learned once on plentiful source data (here: radial features around k-means prototypes) is frozen and reused. Only a small linear head is trained on the few labelled target examples, so far fewer labels are needed than when learning from raw inputs.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{scratch accuracy}", tex: true, value: pct(cur.a), color: C.blue },
        { label: "\\text{transfer accuracy}", tex: true, value: pct(cur.b), color: C.pink },
        { label: "\\text{trainable params}", tex: true, value: "3 vs 25" },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\hat y=\sigma\big(\mathbf w^\top\boldsymbol\phi(\mathbf x)+b\big),\quad\phi_k(\mathbf x)=e^{-\|\mathbf x-\mathbf c_k\|^2/0.18},\ \mathbf c_k\text{ frozen}`, note: "curve: mean over 5 random labelled sets" }]} />}
    />
  );
}
