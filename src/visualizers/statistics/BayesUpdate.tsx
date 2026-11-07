"use client";
import { useState } from "react";
import { Plot, FnPath, C } from "@/components/visualization/Plot";
import { VLine } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Presets } from "@/components/controls";
import { distributionById } from "@/lib/statistics/distributions";
import { mulberry32 } from "@/lib/math/random";
import { fmt, tex } from "@/lib/utilities/format";

/** Beta–Binomial conjugate updating: prior Beta(α, β) + k successes in n trials → Beta(α+k, β+n−k). */
export default function BayesUpdate() {
  const B = distributionById("beta")!;
  const [a0, setA0] = useState(2);
  const [b0, setB0] = useState(2);
  const [trueP, setTrueP] = useState(0.7);
  const [obs, setObs] = useState<number[]>([]);
  const [seed, setSeed] = useState(5);
  const k = obs.filter((x) => x === 1).length;
  const n = obs.length;
  const a = a0 + k, b = b0 + n - k;
  const add = (m: number) => {
    const rng = mulberry32(seed + obs.length * 7919);
    setObs([...obs, ...Array.from({ length: m }, () => (rng() < trueP ? 1 : 0))]);
  };
  // likelihood ∝ p^k (1-p)^(n-k), normalised to integrate to 1 for display (Beta(k+1, n-k+1))
  const lik = (x: number) => (n ? B.density(x, { a: k + 1, b: n - k + 1 }) : NaN);
  const yMax = Math.min(14, Math.max(3, ...[0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98, a / (a + b)].map((x) => Math.max(B.density(x, { a, b }), B.density(x, { a: a0, b: b0 }), n ? lik(x) : 0))) * 1.1);
  const mean = a / (a + b);
  const map = a > 1 && b > 1 ? (a - 1) / (a + b - 2) : NaN;
  const mle = n ? k / n : NaN;
  // 95% credible interval via quantiles (bisection on CDF)
  const q = (p: number) => { let lo = 0, hi = 1; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (B.cdf(m, { a, b }) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
  const ci: [number, number] = [q(0.025), q(0.975)];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, 1]} y={[0, yMax]} aspect={0.55} maxH={420} xLabel="p" ariaLabel="prior, likelihood and posterior distributions">
            <FnPath f={(x) => B.density(x, { a: a0, b: b0 })} style={{ stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "5 3" }} n={400} />
            {n > 0 && <FnPath f={lik} style={{ stroke: C.amber, strokeWidth: 1.4 }} n={400} />}
            <FnPath f={(x) => B.density(x, { a, b })} style={{ stroke: C.pink, strokeWidth: 2.4 }} n={400} />
            <VLine x={trueP} color={C.green} label="true p" />
            <VLine x={ci[0]} color={C.a("pink", 0.5)} />
            <VLine x={ci[1]} color={C.a("pink", 0.5)} />
          </Plot>
          <Legend items={[{ label: `prior Beta(${fmt(a0)}, ${fmt(b0)})`, color: C.muted, dashed: true }, { label: "likelihood (normalised)", color: C.amber }, { label: `posterior Beta(${fmt(a)}, ${fmt(b)})`, color: C.pink }, { label: "95% credible interval", color: C.a("pink", 0.5), dashed: true }]} />
          <div className="flex flex-wrap gap-[3px]" aria-label="observations">
            {obs.slice(-120).map((x, i) => <span key={i} className="h-2.5 w-2.5 rounded-full" style={{ background: x ? "rgb(var(--green))" : "rgb(var(--line-strong))" }} />)}
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="prior α" value={a0} min={0.5} max={20} step={0.5} onChange={setA0} />
          <ParameterSlider label="prior β" value={b0} min={0.5} max={20} step={0.5} onChange={setB0} />
          <ParameterSlider label="true success probability" value={trueP} min={0.01} max={0.99} step={0.01} onChange={setTrueP} />
          <div className="grid grid-cols-3 gap-1.5">
            <button className="btn" onClick={() => add(1)}>+1</button>
            <button className="btn" onClick={() => add(10)}>+10</button>
            <button className="btn" onClick={() => add(100)}>+100</button>
          </div>
          <button className="btn w-full" onClick={() => { setObs([]); setSeed((s) => s + 1); }}>Reset observations</button>
          <Presets presets={[{ label: "Uniform prior", value: [1, 1] }, { label: "Sceptical (≈0.5)", value: [20, 20] }, { label: "Optimistic", value: [8, 2] }]} onPick={([x, y]) => { setA0(x); setB0(y); }} />
        </>
      }
      readouts={<Readouts items={[
        { label: "k/n", tex: true, value: `${k} / ${n}` },
        { label: "\\hat p_{\\text{MLE}}", tex: true, value: fmt(mle) },
        { label: "\\hat p_{\\text{MAP}}", tex: true, value: fmt(map) },
        { label: "E[p\\mid\\text{data}]", tex: true, value: fmt(mean), color: C.pink },
        { label: "95\\%\\ \\text{CrI}", tex: true, value: `[${fmt(ci[0])}, ${fmt(ci[1])}]` },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `\\underbrace{p(\\theta\\mid D)}_{\\text{posterior}}\\propto\\underbrace{\\theta^{k}(1-\\theta)^{n-k}}_{\\text{likelihood}}\\cdot\\underbrace{\\theta^{\\alpha-1}(1-\\theta)^{\\beta-1}}_{\\text{prior}}` },
        { tex: `\\text{Beta}(${tex(a0)},${tex(b0)})+(k=${k},n=${n})\\Rightarrow\\text{Beta}(${tex(a)},${tex(b)})`, note: "conjugacy: posterior stays Beta" },
        { tex: `E[\\theta\\mid D]=\\tfrac{\\alpha+k}{\\alpha+\\beta+n}=${tex(mean, 4)},\\quad\\text{MAP}=\\tfrac{\\alpha+k-1}{\\alpha+\\beta+n-2}=${tex(map, 4)}` },
      ]} />}
      note="With little data the prior dominates; as n grows the posterior concentrates around the MLE regardless of the prior (as long as the prior puts mass there)."
    />
  );
}
