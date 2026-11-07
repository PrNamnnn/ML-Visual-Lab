"use client";
import { useMemo, useState } from "react";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented, Toggle } from "@/components/controls";
import { Plot, Polyline, Dot, C, FnPath } from "@/components/visualization/Plot";
import { Bars, VLine } from "@/components/visualization/Bars";
import { softmax } from "@/lib/nn";
import { histogram } from "@/lib/statistics/descriptive";
import { mulberry32, randn } from "@/lib/math/random";
import { fmt, pct, tex, cn } from "@/lib/utilities/format";

/* ------------------------------ decoding / sampling ------------------------------ */
const CANDS = ["mat", "floor", "sofa", "roof", "bed", "table", "chair", "moon", "piano", "cloud"];
const LOGITS = [3.2, 2.6, 2.1, 1.4, 1.2, 0.8, 0.6, -0.5, -1.2, -1.8];

export function Sampling({ mode = "temperature" }: { mode?: "temperature" | "topk" | "topp" }) {
  const [T, setT] = useState(1);
  const [k, setK] = useState(mode === "topk" ? 3 : 10);
  const [p, setP] = useState(mode === "topp" ? 0.8 : 1);
  const base = softmax(LOGITS, T);
  const order = base.map((q, i) => i).sort((a, b) => base[b] - base[a]);
  const keepK = new Set(order.slice(0, k));
  let cum = 0;
  const keepP = new Set<number>();
  for (const i of order) { if (cum < p || keepP.size === 0) { keepP.add(i); cum += base[i]; } }
  const keep = order.filter((i) => keepK.has(i) && keepP.has(i));
  const Z = keep.reduce((s, i) => s + base[i], 0);
  const final = base.map((q, i) => (keep.includes(i) ? q / Z : 0));
  const H = -final.reduce((s, q) => s + (q > 0 ? q * Math.log2(q) : 0), 0);
  return (
    <LabShell
      viz={
        <div className="space-y-1.5 py-1">
          <div className="text-[12.5px] text-muted">Next token after “The cat sat on the …” (illustrative logits)</div>
          {order.map((i) => (
            <div key={i} className="grid grid-cols-[64px_1fr_56px] items-center gap-2 text-[12.5px]">
              <span className={cn(!keep.includes(i) && "text-faint line-through")}>{CANDS[i]}</span>
              <div className="relative h-5 rounded bg-raised">
                <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${base[i] * 100}%`, background: "rgb(var(--blue) / 0.3)" }} />
                <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${final[i] * 100}%`, background: "rgb(var(--pink) / 0.8)", height: "45%", top: "27%" }} />
              </div>
              <span className="num text-right">{fmt(final[i], 3)}</span>
            </div>
          ))}
          <Legend items={[{ label: "softmax(logits / T)", color: C.a("blue", 0.6) }, { label: "after top-k / top-p filtering and renormalising", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="temperature T" value={T} min={0.05} max={3} step={0.05} onChange={setT} />
          <ParameterSlider label="top-k" value={k} min={1} max={10} step={1} onChange={setK} />
          <ParameterSlider label="top-p (nucleus)" value={p} min={0.05} max={1} step={0.01} onChange={setP} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{candidates kept}", tex: true, value: keep.length }, { label: "\\text{entropy (bits)}", tex: true, value: fmt(H, 3) }, { label: "P(\\text{top token})", tex: true, value: pct(final[order[0]]) }]} />}
      calc={<CalcSteps steps={[{ tex: `p_i=\\frac{e^{\\ell_i/T}}{\\sum_je^{\\ell_j/T}},\\quad T=${tex(T, 2)}` }, { tex: "\\text{top-}k:\\ \\text{keep the }k\\text{ most likely};\\quad\\text{top-}p:\\ \\text{keep the smallest set with cumulative probability}\\ge p;\\ \\text{renormalise}" }]} />}
      note="T → 0 approaches greedy decoding (deterministic, repetitive); T > 1 flattens the distribution (diverse, error-prone). Top-p adapts the number of candidates to how confident the model is. Beam search instead keeps the b most probable partial sequences."
    />
  );
}

/* ------------------------------ LoRA ------------------------------ */
export function LoRA() {
  const [d, setD] = useState(4096);
  const [k, setK] = useState(4096);
  const [r, setR] = useState(8);
  const [layers, setLayers] = useState(32);
  const [mats, setMats] = useState(4);
  const full = d * k, lora = r * (d + k);
  const total = full * layers * mats, ltotal = lora * layers * mats;
  return (
    <LabShell
      viz={
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-center gap-4 py-2">
            <div className="flex flex-col items-center gap-1"><div className="flex items-center justify-center rounded border border-line-strong bg-raised text-[12px] text-muted" style={{ width: 150, height: 150 }}>W₀ (frozen)<br />{d}×{k}</div></div>
            <span className="text-[20px] text-faint">+</span>
            <div className="flex items-center gap-2">
              <div className="flex items-center justify-center rounded border border-blue bg-blue/15 text-[11px]" style={{ width: Math.max(8, (r / 64) * 60), height: 150 }}>{r <= 16 ? "" : "B"}</div>
              <span className="text-faint">×</span>
              <div className="flex items-center justify-center rounded border border-pink bg-pink/15 text-[11px]" style={{ width: 150, height: Math.max(8, (r / 64) * 60) }} />
            </div>
          </div>
          <div className="text-center text-[12px] text-faint">B ∈ ℝ^({d}×{r}) (initialised to 0), A ∈ ℝ^({r}×{k}) (random); ΔW = BA has rank ≤ {r}</div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="d (output dim)" value={d} min={256} max={8192} step={256} onChange={setD} />
          <ParameterSlider label="k (input dim)" value={k} min={256} max={8192} step={256} onChange={setK} />
          <ParameterSlider label="rank r" value={r} min={1} max={64} step={1} onChange={setR} />
          <ParameterSlider label="transformer layers" value={layers} min={1} max={80} step={1} onChange={setLayers} />
          <ParameterSlider label="adapted matrices per layer" value={mats} min={1} max={7} step={1} onChange={setMats} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{full fine-tune params}", tex: true, value: total.toLocaleString() }, { label: "\\text{LoRA params}", tex: true, value: ltotal.toLocaleString(), color: C.green }, { label: "\\text{fraction}", tex: true, value: pct(ltotal / total, 3) }, { label: "\\text{Adam state saved (fp32)}", tex: true, value: `${fmt(((total - ltotal) * 8) / 1e9, 2)} GB` }]} />}
      calc={<CalcSteps steps={[{ tex: `h=W_0x+\\frac{\\alpha}{r}BAx,\\quad\\#\\text{params}=r(d+k)=${r}(${d}+${k})=${lora.toLocaleString()}\\ \\text{vs}\\ dk=${full.toLocaleString()}` }]} />}
      note="LoRA freezes the pretrained weights and learns a low-rank update, so optimiser state and gradients exist only for A and B. After training, BA can be merged into W₀ — no extra inference latency. QLoRA additionally stores W₀ in 4-bit precision."
    />
  );
}

/* ------------------------------ quantization ------------------------------ */
export function Quantization() {
  const [bits, setBits] = useState(8);
  const [scheme, setScheme] = useState<"symmetric" | "asymmetric">("symmetric");
  const [outlier, setOutlier] = useState(false);
  const w = useMemo(() => { const r = mulberry32(3); const x = Array.from({ length: 2000 }, () => randn(r) * 0.05); if (outlier) x[0] = 0.9; return x; }, [outlier]);
  const lo = Math.min(...w), hi = Math.max(...w);
  const qmax = 2 ** (bits - 1) - 1, levels = 2 ** bits;
  let scale: number, zp: number;
  if (scheme === "symmetric") { scale = Math.max(Math.abs(lo), Math.abs(hi)) / qmax; zp = 0; }
  else { scale = (hi - lo) / (levels - 1); zp = Math.round(-lo / scale) - 2 ** (bits - 1); }
  const q = (x: number) => (scheme === "symmetric" ? Math.max(-qmax - 1, Math.min(qmax, Math.round(x / scale))) : Math.max(-(2 ** (bits - 1)), Math.min(2 ** (bits - 1) - 1, Math.round(x / scale) + zp)));
  const dq = (v: number) => (v - zp) * scale;
  const wq = w.map((x) => dq(q(x)));
  const mse = w.reduce((s, x, i) => s + (x - wq[i]) ** 2, 0) / w.length;
  const used = new Set(w.map(q)).size;
  const rng: [number, number] = outlier ? [-0.25, 0.25] : [-0.2, 0.2];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={rng} y={[0, 12]} aspect={0.45} maxH={320} ariaLabel="weight distribution and quantization grid">
            <Bars bins={histogram(w, rng[0], rng[1], 80)} />
            {Array.from({ length: Math.min(levels, 256) }, (_, i) => dq(i - 2 ** (bits - 1))).filter((v) => v >= rng[0] && v <= rng[1]).map((v, i) => <VLine key={i} x={v} color={C.a("pink", bits <= 4 ? 0.7 : 0.25)} dashed={false} />)}
          </Plot>
          <div className="mb-1 text-[11.5px] text-faint">x vs dequantised x̂ (staircase)</div>
          <Plot x={rng} y={rng} aspect={0.35} minH={150} maxH={220} ariaLabel="quantization staircase">
            <FnPath f={(x) => x} style={{ stroke: C.a("fg", 0.3), strokeDasharray: "4 3", strokeWidth: 1 }} />
            <FnPath f={(x) => dq(q(x))} n={1200} style={{ stroke: C.pink, strokeWidth: 1.6 }} />
          </Plot>
          <Legend items={[{ label: "fp32 weight histogram", color: C.a("blue", 0.6) }, { label: `int${bits} grid points`, color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="bits" value={bits} min={2} max={8} step={1} onChange={setBits} />
          <Segmented value={scheme} onChange={setScheme} options={[{ value: "symmetric", label: "Symmetric (absmax)" }, { value: "asymmetric", label: "Asymmetric (zero-point)" }]} />
          <Toggle label="Add one outlier weight" checked={outlier} onChange={setOutlier} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{scale }s", tex: true, value: fmt(scale, 6) }, { label: "\\text{zero point}", tex: true, value: zp }, { label: "\\text{levels used}", tex: true, value: `${used} / ${levels}` }, { label: "\\text{MSE}", tex: true, value: mse.toExponential(2) }, { label: "\\text{memory vs fp32}", tex: true, value: `${fmt((bits / 32) * 100, 1)}%` }]} />}
      calc={<CalcSteps steps={[{ tex: scheme === "symmetric" ? `q=\\operatorname{clip}\\big(\\operatorname{round}(x/s),-${qmax + 1},${qmax}\\big),\\ s=\\frac{\\max|x|}{${qmax}}=${tex(scale, 6)}` : `q=\\operatorname{round}(x/s)+z,\\ s=\\frac{x_{\\max}-x_{\\min}}{${levels - 1}},\\ z=${zp}` }, { tex: "\\hat x=s\\,(q-z)" }]} />}
      note="A single outlier stretches the range, wasting most levels on empty space — why LLM quantizers (LLM.int8, GPTQ, AWQ) treat outliers specially or quantize per channel/group. 4-bit weights cut memory 8× versus fp32."
    />
  );
}

/* ------------------------------ KV cache ------------------------------ */
export function KVCache() {
  const [L, setL] = useState(32);
  const [Hh, setHh] = useState(32);
  const [dh, setDh] = useState(128);
  const [ctx, setCtx] = useState(4096);
  const [batch, setBatch] = useState(1);
  const [bytes, setBytes] = useState(2);
  const [kvHeads, setKvHeads] = useState(32);
  const [tok, setTok] = useState(6);
  const mem = 2 * L * kvHeads * dh * ctx * batch * bytes;
  const withCache = tok, without = (tok * (tok + 1)) / 2;
  return (
    <LabShell
      viz={
        <div className="space-y-4">
          <div>
            <div className="mb-1 text-[12px] text-muted">Generating token {tok}: cached K/V rows (blue) are reused; only the new row (pink) is computed</div>
            <div className="flex flex-wrap gap-1">
              {Array.from({ length: tok }, (_, i) => <div key={i} className="flex h-10 w-10 items-center justify-center rounded text-[10.5px]" style={{ background: i === tok - 1 ? "rgb(var(--pink) / 0.6)" : "rgb(var(--blue) / 0.35)" }}>t{i + 1}</div>)}
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-md border border-line p-3 text-[12.5px]"><div className="label mb-1">Without cache</div>K/V projections recomputed for all {tok} tokens each step — {without} token-projections in total so far (O(n²)).</div>
            <div className="rounded-md border border-line p-3 text-[12.5px]"><div className="label mb-1">With cache</div>One new K/V row per step — {withCache} token-projections in total (O(n)), at the cost of storing them.</div>
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="tokens generated" value={tok} min={1} max={24} step={1} onChange={setTok} />
          <ParameterSlider label="layers" value={L} min={1} max={96} step={1} onChange={setL} />
          <ParameterSlider label="query heads" value={Hh} min={1} max={128} step={1} onChange={(v) => { setHh(v); setKvHeads(Math.min(kvHeads, v)); }} />
          <ParameterSlider label="KV heads (GQA/MQA)" value={kvHeads} min={1} max={Hh} step={1} onChange={setKvHeads} />
          <ParameterSlider label="head dim" value={dh} min={32} max={256} step={32} onChange={setDh} />
          <ParameterSlider label="context length" value={ctx} min={512} max={131072} step={512} onChange={setCtx} />
          <ParameterSlider label="batch size" value={batch} min={1} max={64} step={1} onChange={setBatch} />
          <Select label="precision" value={String(bytes)} onChange={(v) => setBytes(Number(v))} options={[{ value: "4", label: "fp32 (4 B)" }, { value: "2", label: "fp16/bf16 (2 B)" }, { value: "1", label: "int8/fp8 (1 B)" }]} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{KV cache size}", tex: true, value: `${fmt(mem / 2 ** 30, 2)} GiB`, color: C.pink }, { label: "\\text{per token}", tex: true, value: `${fmt((2 * L * kvHeads * dh * bytes) / 1024, 1)} KiB` }, { label: "\\text{GQA saving}", tex: true, value: `${fmt(Hh / kvHeads, 1)}×` }]} />}
      calc={<CalcSteps steps={[{ tex: `\\text{bytes}=2\\cdot L\\cdot h_{kv}\\cdot d_h\\cdot n_{ctx}\\cdot B\\cdot b=2\\cdot${L}\\cdot${kvHeads}\\cdot${dh}\\cdot${ctx}\\cdot${batch}\\cdot${bytes}`, note: "factor 2 = keys and values" }]} />}
      note="Defaults approximate a 7B-parameter model (32 layers, 32 heads × 128). Long contexts and large batches make the KV cache — not the weights — the memory bottleneck; grouped-query attention, paged attention and cache quantization attack it."
    />
  );
}

/* ------------------------------ data drift ------------------------------ */
export function Drift() {
  const [shift, setShift] = useState(0.6);
  const [spread, setSpread] = useState(1.2);
  const [bins] = useState(10);
  const ref = useMemo(() => { const r = mulberry32(1); return Array.from({ length: 2000 }, () => randn(r)); }, []);
  const cur = useMemo(() => { const r = mulberry32(2); return Array.from({ length: 2000 }, () => shift + spread * randn(r)); }, [shift, spread]);
  // PSI with deciles of the reference
  const s = ref.slice().sort((a, b) => a - b);
  const edges = Array.from({ length: bins - 1 }, (_, i) => s[Math.floor(((i + 1) * s.length) / bins)]);
  const bucket = (x: number) => edges.findIndex((e) => x < e) === -1 ? bins - 1 : edges.findIndex((e) => x < e);
  const pr = Array(bins).fill(0), pc = Array(bins).fill(0);
  ref.forEach((x) => pr[bucket(x)]++); cur.forEach((x) => pc[bucket(x)]++);
  const psi = pr.reduce((acc, _, i) => { const a = Math.max(1e-4, pr[i] / ref.length), b = Math.max(1e-4, pc[i] / cur.length); return acc + (b - a) * Math.log(b / a); }, 0);
  // KS statistic
  const cs = cur.slice().sort((a, b) => a - b);
  let ks = 0, i = 0, j = 0;
  while (i < s.length && j < cs.length) { if (s[i] <= cs[j]) i++; else j++; ks = Math.max(ks, Math.abs(i / s.length - j / cs.length)); }
  const ksCrit = 1.358 * Math.sqrt((s.length + cs.length) / (s.length * cs.length));
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-5, 7]} y={[0, 0.45]} aspect={0.45} maxH={320} ariaLabel="reference versus production feature distributions">
            <Bars bins={histogram(ref, -5, 7, 60)} color={C.a("blue", 0.3)} stroke={C.a("blue", 0.6)} />
            <Bars bins={histogram(cur, -5, 7, 60)} color={C.a("pink", 0.3)} stroke={C.a("pink", 0.6)} />
          </Plot>
          <Legend items={[{ label: "training (reference) data", color: C.blue }, { label: "production data", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="mean shift" value={shift} min={-2} max={3} step={0.05} onChange={setShift} />
          <ParameterSlider label="spread multiplier" value={spread} min={0.4} max={2.5} step={0.05} onChange={setSpread} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{PSI}", tex: true, value: fmt(psi, 3), color: psi > 0.25 ? C.red : psi > 0.1 ? C.amber : C.green }, { label: "\\text{PSI verdict}", tex: true, value: psi > 0.25 ? "major shift" : psi > 0.1 ? "moderate" : "stable" }, { label: "\\text{KS }D", tex: true, value: fmt(ks, 3) }, { label: "D_{\\text{crit}}\\ (\\alpha=0.05)", tex: true, value: fmt(ksCrit, 3) }]} />}
      calc={<CalcSteps steps={[{ tex: "\\text{PSI}=\\sum_{b}(p_b^{\\text{prod}}-p_b^{\\text{ref}})\\ln\\frac{p_b^{\\text{prod}}}{p_b^{\\text{ref}}}", note: "10 quantile bins of the reference" }, { tex: "D=\\sup_x|F_{\\text{ref}}(x)-F_{\\text{prod}}(x)|" }]} />}
      note="This is data (covariate) drift: P(x) changes. Concept drift — P(y | x) changing — cannot be seen from inputs alone and needs labelled feedback. With 2,000 samples the KS test flags even tiny shifts; PSI rules of thumb (0.1 / 0.25) are industry conventions, not statistical tests."
    />
  );
}

export { Dot, Polyline };
