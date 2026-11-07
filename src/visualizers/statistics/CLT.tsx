"use client";
import { useMemo, useState, useEffect } from "react";
import { Plot, FnPath, C, Polyline } from "@/components/visualization/Plot";
import { Bars, VLine } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, AnimationControls } from "@/components/controls";
import { distributionById } from "@/lib/statistics/distributions";
import { histogram, mean, std } from "@/lib/statistics/descriptive";
import { mulberry32 } from "@/lib/math/random";
import { useTicker } from "@/hooks/useSize";
import { fmt, tex } from "@/lib/utilities/format";

const SOURCES = {
  exponential: { label: "Exponential (λ=1) — skewed", id: "exponential", p: { lambda: 1 }, range: [0, 6] as [number, number] },
  uniform: { label: "Uniform(0, 1)", id: "uniform", p: { a: 0, b: 1 }, range: [-0.1, 1.1] as [number, number] },
  bernoulli: { label: "Bernoulli(0.2) — discrete", id: "bernoulli", p: { p: 0.2 }, range: [-0.2, 1.2] as [number, number] },
  beta: { label: "Beta(0.5, 0.5) — U-shaped", id: "beta", p: { a: 0.5, b: 0.5 }, range: [0, 1] as [number, number] },
  lognormal: { label: "Log-normal — heavy right tail", id: "log-normal", p: { mu: 0, sigma: 0.9 }, range: [0, 8] as [number, number] },
};

export default function CLT({ mode = "clt" }: { mode?: "clt" | "lln" }) {
  const [src, setSrc] = useState<keyof typeof SOURCES>("exponential");
  const S = SOURCES[src];
  const D = distributionById(S.id)!;
  const [n, setN] = useState(mode === "lln" ? 1 : 5);
  const [means, setMeans] = useState<number[]>([]);
  const [draws, setDraws] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);
  const [seed, setSeed] = useState(11);
  const rng = useMemo(() => mulberry32(seed), [seed]);
  useEffect(() => { setMeans([]); setDraws([]); }, [src, n]);
  const mu = D.mean(S.p), sigma = Math.sqrt(D.variance(S.p));
  useTicker(playing, 50, () => {
    if (mode === "lln") {
      setDraws((d) => { if (d.length >= 3000) { setPlaying(false); return d; } return [...d, ...Array.from({ length: Math.max(1, Math.round(d.length / 25)) }, () => D.sample(rng, S.p))]; });
    } else {
      setMeans((m) => {
        if (m.length >= 4000) { setPlaying(false); return m; }
        const k = Math.max(1, Math.round(m.length / 15));
        const add = Array.from({ length: k }, () => { let s = 0; for (let i = 0; i < n; i++) s += D.sample(rng, S.p); return s / n; });
        return [...m, ...add];
      });
    }
  });

  if (mode === "lln") {
    let s = 0;
    const running: [number, number][] = draws.map((x, i) => { s += x; return [i + 1, s / (i + 1)]; });
    const N = Math.max(10, draws.length);
    return (
      <LabShell
        viz={
          <div className="space-y-2">
            <Plot x={[1, N]} y={[mu - 3 * sigma, mu + 3 * sigma]} aspect={0.5} maxH={380} xLabel="number of draws n" ariaLabel="running sample mean converging to the true mean">
              <FnPath f={(k) => mu + (2 * sigma) / Math.sqrt(k)} style={{ stroke: C.a("fg", 0.25), strokeWidth: 1, strokeDasharray: "4 3" }} />
              <FnPath f={(k) => mu - (2 * sigma) / Math.sqrt(k)} style={{ stroke: C.a("fg", 0.25), strokeWidth: 1, strokeDasharray: "4 3" }} />
              <Polyline pts={[[1, mu], [N, mu]]} style={{ stroke: C.green, strokeWidth: 1.4 }} />
              <Polyline pts={running} style={{ stroke: C.blue, strokeWidth: 1.8 }} />
            </Plot>
            <Legend items={[{ label: "running mean x̄ₙ", color: C.blue }, { label: "true mean μ", color: C.green }, { label: "μ ± 2σ/√n", color: C.a("fg", 0.4), dashed: true }]} />
          </div>
        }
        controls={
          <>
            <Select label="Population" value={src} onChange={setSrc} options={Object.entries(SOURCES).map(([k, v]) => ({ value: k as keyof typeof SOURCES, label: v.label }))} />
            <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onStep={() => setDraws((d) => [...d, D.sample(rng, S.p)])} onReset={() => { setPlaying(false); setDraws([]); setSeed((x) => x + 1); }} />
          </>
        }
        readouts={<Readouts items={[{ label: "n", tex: true, value: draws.length }, { label: "\\bar x_n", tex: true, value: draws.length ? fmt(running[running.length - 1][1]) : "—" }, { label: "\\mu", tex: true, value: fmt(mu) }, { label: "|\\bar x_n-\\mu|", tex: true, value: draws.length ? fmt(Math.abs(running[running.length - 1][1] - mu)) : "—" }]} />}
        calc={<CalcSteps steps={[{ tex: "\\bar X_n=\\frac1n\\sum_{i=1}^nX_i\\xrightarrow{\\ n\\to\\infty\\ }\\mu\\quad\\text{(almost surely)}" }, { tex: `\\operatorname{SD}(\\bar X_n)=\\sigma/\\sqrt n=${tex(sigma)}/\\sqrt{${Math.max(1, draws.length)}}=${tex(sigma / Math.sqrt(Math.max(1, draws.length)))}` }]} />}
        note="The law of large numbers guarantees convergence, but slowly: to halve the error you need four times the data."
      />
    );
  }

  const se = sigma / Math.sqrt(n);
  const lo = Math.min(S.range[0], mu - 4.5 * se), hi = Math.max(S.range[1] * (n === 1 ? 1 : 0.6), mu + 4.5 * se);
  const bins = histogram(means, lo, hi, 50);
  const yMax = Math.max(1 / (se * Math.sqrt(2 * Math.PI)) * 1.25, ...bins.map((b) => b.density * 1.05));
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div>
            <div className="mb-1 text-[11.5px] text-faint">Population distribution</div>
            <Plot x={S.range} y={[0, D.kind === "discrete" ? 1 : Math.min(3, Math.max(...Array.from({ length: 100 }, (_, i) => D.density(S.range[0] + ((S.range[1] - S.range[0]) * (i + 0.5)) / 100, S.p))) * 1.1)]} aspect={0.18} minH={110} maxH={140} ariaLabel="population distribution">
              {D.kind === "discrete" ? <Polyline pts={[[0, 0], [0, D.density(0, S.p)], [1, 0], [1, D.density(1, S.p)]]} style={{ stroke: C.muted, strokeWidth: 6 }} /> : <FnPath f={(x) => D.density(x, S.p)} style={{ stroke: C.muted, strokeWidth: 1.6 }} />}
            </Plot>
          </div>
          <div>
            <div className="mb-1 text-[11.5px] text-faint">Distribution of the sample mean x̄ over {means.length} repeated samples of size n = {n}</div>
            <Plot x={[lo, hi]} y={[0, yMax]} aspect={0.42} maxH={340} ariaLabel="histogram of sample means with normal approximation">
              <Bars bins={bins} />
              <FnPath f={(x) => Math.exp(-0.5 * ((x - mu) / se) ** 2) / (se * Math.sqrt(2 * Math.PI))} style={{ stroke: C.pink, strokeWidth: 2 }} />
              <VLine x={mu} color={C.green} label="μ" />
            </Plot>
          </div>
          <Legend items={[{ label: "histogram of x̄", color: C.a("blue", 0.6) }, { label: "CLT prediction N(μ, σ²/n)", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <Select label="Population" value={src} onChange={setSrc} options={Object.entries(SOURCES).map(([k, v]) => ({ value: k as keyof typeof SOURCES, label: v.label }))} />
          <ParameterSlider label="sample size n" value={n} min={1} max={100} step={1} onChange={setN} />
          <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onStep={() => setMeans((m) => [...m, ...Array.from({ length: 100 }, () => { let s = 0; for (let i = 0; i < n; i++) s += D.sample(rng, S.p); return s / n; })])} onReset={() => { setPlaying(false); setMeans([]); setSeed((x) => x + 1); }} />
          <p className="text-[12px] leading-relaxed text-faint">Increase n: even for skewed or discrete populations the histogram of means becomes bell-shaped and narrower (width ∝ 1/√n).</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\mu", tex: true, value: fmt(mu) },
        { label: "\\sigma/\\sqrt n", tex: true, value: fmt(se) },
        { label: "\\text{mean of }\\bar x", tex: true, value: means.length ? fmt(mean(means)) : "—" },
        { label: "\\text{SD of }\\bar x", tex: true, value: means.length > 1 ? fmt(std(means)) : "—" },
        { label: "\\#\\text{ samples}", tex: true, value: means.length },
      ]} />}
      calc={<CalcSteps steps={[{ tex: `\\sqrt n\\,\\frac{\\bar X_n-\\mu}{\\sigma}\\xrightarrow{d}\\mathcal N(0,1)\\;\\Rightarrow\\;\\bar X_n\\approx\\mathcal N\\!\\left(${tex(mu)},\\ ${tex(se * se, 4)}\\right)` }]} />}
      note="The CLT needs finite variance. For heavy-tailed populations (log-normal here) convergence is visibly slower — n = 30 is a rule of thumb, not a law."
    />
  );
}
