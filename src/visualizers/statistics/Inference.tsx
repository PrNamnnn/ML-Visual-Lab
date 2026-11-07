"use client";
import { useMemo, useState } from "react";
import { Plot, FnPath, C, Polyline, Dot } from "@/components/visualization/Plot";
import { AreaUnder, VLine, Bars } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, AnimationControls, Select } from "@/components/controls";
import { normalCdf, normalQuantile, normalPdf, studentTCdf, invertCdf } from "@/lib/math/special";
import { mulberry32, randn, randInt } from "@/lib/math/random";
import { mean, std, histogram, quantile } from "@/lib/statistics/descriptive";
import { useTicker } from "@/hooks/useSize";
import { fmt, pct, tex } from "@/lib/utilities/format";

type Mode = "ci" | "hypothesis" | "errors" | "power" | "bootstrap" | "ttest";

export default function Inference({ mode = "hypothesis" }: { mode?: Mode }) {
  if (mode === "ci") return <ConfidenceIntervals />;
  if (mode === "bootstrap") return <Bootstrap />;
  if (mode === "ttest") return <TTest />;
  return <ZTest initialView={mode === "errors" || mode === "power" ? "errors" : "pvalue"} />;
}

/* ------------------------------ z-test: p-value, Type I/II, power ------------------------------ */
function ZTest({ initialView }: { initialView: "pvalue" | "errors" }) {
  const [view, setView] = useState(initialView);
  const [alpha, setAlpha] = useState(0.05);
  const [effect, setEffect] = useState(0.5); // true mean shift in σ units
  const [n, setN] = useState(25);
  const [z, setZ] = useState(2.1);
  const [tail, setTail] = useState<"two" | "right">("two");
  const crit = tail === "two" ? normalQuantile(1 - alpha / 2) : normalQuantile(1 - alpha);
  const shift = effect * Math.sqrt(n); // mean of the z statistic under H1
  const power = tail === "two" ? 1 - normalCdf(crit - shift) + normalCdf(-crit - shift) : 1 - normalCdf(crit - shift);
  const beta = 1 - power;
  const p = tail === "two" ? 2 * (1 - normalCdf(Math.abs(z))) : 1 - normalCdf(z);
  const XR = Math.max(4.5, shift + 3.5);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-4.5, XR]} y={[0, 0.45]} aspect={0.5} maxH={400} xLabel="test statistic z" ariaLabel="null and alternative distributions">
            {view === "pvalue" ? (
              <>
                {tail === "two" ? (<><AreaUnder f={normalPdf} a={Math.abs(z)} b={XR} color={C.a("pink", 0.45)} /><AreaUnder f={normalPdf} a={-4.5} b={-Math.abs(z)} color={C.a("pink", 0.45)} /></>) : <AreaUnder f={normalPdf} a={z} b={XR} color={C.a("pink", 0.45)} />}
                <FnPath f={normalPdf} />
                <VLine x={z} color={C.pink} label={`z = ${fmt(z, 2)}`} />
                <VLine x={crit} color={C.faint} label="critical" />
                {tail === "two" && <VLine x={-crit} color={C.faint} />}
              </>
            ) : (
              <>
                <AreaUnder f={normalPdf} a={crit} b={XR} color={C.a("red", 0.35)} />
                {tail === "two" && <AreaUnder f={normalPdf} a={-4.5} b={-crit} color={C.a("red", 0.35)} />}
                <AreaUnder f={(x) => normalPdf(x - shift)} a={-4.5} b={crit} color={C.a("amber", 0.3)} />
                <AreaUnder f={(x) => normalPdf(x - shift)} a={crit} b={XR} color={C.a("green", 0.22)} />
                <FnPath f={normalPdf} />
                <FnPath f={(x) => normalPdf(x - shift)} style={{ stroke: C.green }} />
                <VLine x={crit} color={C.fg} label="reject H₀ →" />
              </>
            )}
          </Plot>
          <Legend items={view === "pvalue" ? [{ label: "null distribution N(0,1)", color: C.blue }, { label: "p-value area", color: C.pink }] : [{ label: "H₀: N(0,1)", color: C.blue }, { label: `H₁: N(${fmt(shift, 2)}, 1)`, color: C.green }, { label: `α — Type I error`, color: C.a("red", 0.7) }, { label: "β — Type II error", color: C.amber }, { label: "power = 1 − β", color: C.a("green", 0.7) }]} />
        </div>
      }
      controls={
        <>
          <Segmented value={view} onChange={setView} options={[{ value: "pvalue", label: "p-value" }, { value: "errors", label: "Errors & power" }]} />
          <Segmented value={tail} onChange={setTail} options={[{ value: "two", label: "two-sided" }, { value: "right", label: "one-sided (>)" }]} />
          <ParameterSlider label="significance α" value={alpha} min={0.001} max={0.2} step={0.001} onChange={setAlpha} />
          {view === "pvalue" ? (
            <ParameterSlider label="observed z" value={z} min={-4} max={4} step={0.01} onChange={setZ} />
          ) : (
            <>
              <ParameterSlider label="true effect d = (μ₁−μ₀)/σ" value={effect} min={0} max={1.5} step={0.01} onChange={setEffect} />
              <ParameterSlider label="sample size n" value={n} min={2} max={200} step={1} onChange={setN} />
            </>
          )}
        </>
      }
      readouts={<Readouts items={view === "pvalue" ? [
        { label: "p", tex: true, value: fmt(p, 4), color: p < alpha ? C.pink : undefined },
        { label: "z_{\\text{crit}}", tex: true, value: fmt(crit, 3) },
        { label: "\\text{decision}", tex: true, value: p < alpha ? "reject H₀" : "fail to reject H₀" },
      ] : [
        { label: "\\alpha", tex: true, value: fmt(alpha, 3) },
        { label: "\\beta", tex: true, value: fmt(beta, 4), color: C.amber },
        { label: "\\text{power}", tex: true, value: pct(power, 1), color: C.green },
        { label: "\\sqrt n\\,d", tex: true, value: fmt(shift, 3) },
      ]} />}
      calc={<CalcSteps steps={view === "pvalue" ? [
        { tex: "z=\\dfrac{\\bar x-\\mu_0}{\\sigma/\\sqrt n}\\sim\\mathcal N(0,1)\\text{ under }H_0" },
        { tex: tail === "two" ? `p=2\\,(1-\\Phi(|${tex(z, 2)}|))=${tex(p, 4)}` : `p=1-\\Phi(${tex(z, 2)})=${tex(p, 4)}` },
      ] : [
        { tex: `\\text{under }H_1:\\ z\\sim\\mathcal N(\\sqrt n\\,d,1)=\\mathcal N(${tex(shift, 3)},1)` },
        { tex: `\\text{power}=P(|z|>z_{\\alpha/2}\\mid H_1)=${tex(power, 4)}` },
      ]} />}
      note={view === "pvalue" ? "A p-value is P(data at least this extreme | H₀) — not the probability that H₀ is true. A tiny effect can be 'significant' with huge n: statistical significance is not practical significance." : "Power rises with effect size, sample size and α. Lowering α reduces false positives but increases false negatives."}
    />
  );
}

/* ------------------------------ confidence intervals ------------------------------ */
function ConfidenceIntervals() {
  const [conf, setConf] = useState(0.95);
  const [n, setN] = useState(15);
  const [ivs, setIvs] = useState<{ lo: number; hi: number; m: number }[]>([]);
  const [playing, setPlaying] = useState(false);
  const [seed, setSeed] = useState(3);
  const [useT, setUseT] = useState<"t" | "z">("t");
  const mu = 10, sigma = 3;
  const rng = useMemo(() => mulberry32(seed), [seed]);
  const crit = useT === "t" ? invertCdf((t) => studentTCdf(t, n - 1), 1 - (1 - conf) / 2, 0, 50) : normalQuantile(1 - (1 - conf) / 2);
  const draw = () => {
    const xs = Array.from({ length: n }, () => mu + sigma * randn(rng));
    const m = mean(xs);
    const se = (useT === "t" ? std(xs) : sigma) / Math.sqrt(n);
    return { lo: m - crit * se, hi: m + crit * se, m };
  };
  useTicker(playing, 120, () => setIvs((v) => { if (v.length >= 100) { setPlaying(false); return v; } return [...v, draw()]; }));
  const covered = ivs.filter((iv) => iv.lo <= mu && mu <= iv.hi).length;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[mu - 3.5 * sigma / Math.sqrt(Math.min(n, 15)) - 2, mu + 3.5 * sigma / Math.sqrt(Math.min(n, 15)) + 2]} y={[0, 101]} aspect={0.6} maxH={460} grid={false} ariaLabel="100 simulated confidence intervals">
            <VLine x={mu} color={C.green} dashed={false} label="true μ" />
            {ivs.map((iv, i) => {
              const ok = iv.lo <= mu && mu <= iv.hi;
              return (
                <g key={i}>
                  <Polyline pts={[[iv.lo, i + 1], [iv.hi, i + 1]]} style={{ stroke: ok ? C.a("blue", 0.75) : C.red, strokeWidth: ok ? 1.4 : 2 }} />
                  <Dot p={[iv.m, i + 1]} r={1.6} color={ok ? C.blue : C.red} />
                </g>
              );
            })}
          </Plot>
          <Legend items={[{ label: "interval contains μ", color: C.blue }, { label: "interval misses μ", color: C.red }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="confidence level" value={conf} min={0.5} max={0.99} step={0.01} onChange={(v) => { setConf(v); setIvs([]); }} format={(v) => pct(v, 0)} />
          <ParameterSlider label="sample size n" value={n} min={3} max={100} step={1} onChange={(v) => { setN(v); setIvs([]); }} />
          <Segmented value={useT} onChange={(v) => { setUseT(v); setIvs([]); }} options={[{ value: "t", label: "t-interval (σ unknown)" }, { value: "z", label: "z (σ known)" }]} />
          <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onStep={() => setIvs((v) => (v.length < 100 ? [...v, draw()] : v))} onReset={() => { setPlaying(false); setIvs([]); setSeed((s) => s + 1); }} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{intervals}", tex: true, value: ivs.length }, { label: "\\text{covering }\\mu", tex: true, value: ivs.length ? `${covered} (${pct(covered / ivs.length, 0)})` : "—" }, { label: "\\text{nominal}", tex: true, value: pct(conf, 0) }, { label: useT === "t" ? `t_{${n - 1}}` : "z", tex: true, value: fmt(crit, 3) }]} />}
      calc={<CalcSteps steps={[{ tex: useT === "t" ? `\\bar x\\pm t_{${n - 1},\\,${tex(1 - (1 - conf) / 2, 3)}}\\dfrac{s}{\\sqrt n}=\\bar x\\pm${tex(crit, 3)}\\dfrac{s}{\\sqrt{${n}}}` : `\\bar x\\pm z\\dfrac{\\sigma}{\\sqrt n}=\\bar x\\pm${tex(crit, 3)}\\cdot\\dfrac{${sigma}}{\\sqrt{${n}}}` }]} />}
      note="'95% confidence' describes the procedure: about 95% of intervals built this way contain μ. Any single computed interval either contains μ or it doesn't. Population: N(10, 3²)."
    />
  );
}

/* ------------------------------ two-sample t-test ------------------------------ */
function TTest() {
  const [d, setD] = useState(0.6);
  const [n, setN] = useState(20);
  const [seed, setSeed] = useState(2);
  const [paired, setPaired] = useState<"ind" | "paired">("ind");
  const data = useMemo(() => {
    const rng = mulberry32(seed);
    const a = Array.from({ length: n }, () => 50 + 10 * randn(rng));
    const b = paired === "paired" ? a.map((x) => x + d * 10 + 4 * randn(rng)) : Array.from({ length: n }, () => 50 + d * 10 + 10 * randn(rng));
    return { a, b };
  }, [seed, n, d, paired]);
  let t: number, df: number, se: number;
  if (paired === "paired") {
    const diff = data.b.map((x, i) => x - data.a[i]);
    se = std(diff) / Math.sqrt(n); t = mean(diff) / se; df = n - 1;
  } else {
    const va = std(data.a) ** 2, vb = std(data.b) ** 2;
    se = Math.sqrt(va / n + vb / n); t = (mean(data.b) - mean(data.a)) / se;
    df = (va / n + vb / n) ** 2 / ((va / n) ** 2 / (n - 1) + (vb / n) ** 2 / (n - 1)); // Welch
  }
  const p = 2 * (1 - studentTCdf(Math.abs(t), df));
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, 3]} y={[15, 95]} aspect={0.55} maxH={400} grid={false} ariaLabel="two groups of observations">
            {data.a.map((v, i) => <Dot key={`a${i}`} p={[0.9 + ((i * 37) % 17) / 60, v]} r={3} color={C.a("blue", 0.8)} />)}
            {data.b.map((v, i) => <Dot key={`b${i}`} p={[1.9 + ((i * 37) % 17) / 60, v]} r={3} color={C.a("pink", 0.8)} />)}
            {paired === "paired" && data.a.map((v, i) => <Polyline key={`l${i}`} pts={[[0.9 + ((i * 37) % 17) / 60, v], [1.9 + ((i * 37) % 17) / 60, data.b[i]]]} style={{ stroke: C.a("fg", 0.12), strokeWidth: 1 }} />)}
            <Polyline pts={[[0.8, mean(data.a)], [1.4, mean(data.a)]]} style={{ stroke: C.blue, strokeWidth: 2.5 }} />
            <Polyline pts={[[1.8, mean(data.b)], [2.4, mean(data.b)]]} style={{ stroke: C.pink, strokeWidth: 2.5 }} />
          </Plot>
          <Legend items={[{ label: `group A (mean ${fmt(mean(data.a), 1)})`, color: C.blue }, { label: `group B (mean ${fmt(mean(data.b), 1)})`, color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <Segmented value={paired} onChange={setPaired} options={[{ value: "ind", label: "Independent (Welch)" }, { value: "paired", label: "Paired" }]} />
          <ParameterSlider label="true effect d (in σ)" value={d} min={0} max={1.5} step={0.05} onChange={setD} />
          <ParameterSlider label="n per group" value={n} min={3} max={100} step={1} onChange={setN} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>Draw new samples</button>
        </>
      }
      readouts={<Readouts items={[{ label: "t", tex: true, value: fmt(t, 3) }, { label: "\\text{df}", tex: true, value: fmt(df, 1) }, { label: "p", tex: true, value: fmt(p, 4), color: p < 0.05 ? C.pink : undefined }, { label: "\\text{at }\\alpha=0.05", tex: true, value: p < 0.05 ? "significant" : "not significant" }]} />}
      calc={<CalcSteps steps={paired === "paired" ? [{ tex: `t=\\dfrac{\\bar d}{s_d/\\sqrt n}=${tex(t, 3)},\\quad\\text{df}=n-1=${n - 1}` }] : [{ tex: `t=\\dfrac{\\bar x_B-\\bar x_A}{\\sqrt{s_A^2/n_A+s_B^2/n_B}}=\\dfrac{${tex(mean(data.b) - mean(data.a), 2)}}{${tex(se, 3)}}=${tex(t, 3)}` }, { tex: `\\text{Welch–Satterthwaite df}=${tex(df, 2)},\\quad p=2(1-F_t(|t|))=${tex(p, 4)}` }]} />}
      note="Pairing removes between-subject variability, so the same effect is detected with far fewer subjects. Comparing two models on the same test examples is a paired design."
    />
  );
}

/* ------------------------------ bootstrap ------------------------------ */
function Bootstrap() {
  const [stat, setStat] = useState<"mean" | "median">("median");
  const [B, setB] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);
  const [seed] = useState(8);
  const data = useMemo(() => { const r = mulberry32(41); return Array.from({ length: 30 }, () => Math.round(Math.exp(2 + 0.6 * randn(r)) * 10) / 10); }, []);
  const rng = useMemo(() => mulberry32(seed), [seed]);
  const f = (xs: number[]) => (stat === "mean" ? mean(xs) : quantile(xs, 0.5));
  const one = () => f(Array.from({ length: data.length }, () => data[randInt(rng, 0, data.length - 1)]));
  useTicker(playing, 40, () => setB((b) => { if (b.length >= 3000) { setPlaying(false); return b; } return [...b, ...Array.from({ length: Math.max(1, Math.round(b.length / 15)) }, one)]; }));
  const est = f(data);
  const lo = Math.min(...data), hi = Math.max(...data);
  const bins = histogram(B, lo, hi * 0.6, 50);
  const ci = B.length > 20 ? [quantile(B, 0.025), quantile(B, 0.975)] : null;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div>
            <div className="mb-1 text-[11.5px] text-faint">Original sample (n = 30, right-skewed)</div>
            <Plot x={[0, hi * 1.05]} y={[0, 1]} aspect={0.12} minH={70} maxH={90} grid={false} ariaLabel="original sample">
              {data.map((v, i) => <Dot key={i} p={[v, 0.5]} r={3.5} color={C.a("fg", 0.6)} />)}
              <AreaUnder f={() => 0} a={0} b={0} />
            </Plot>
          </div>
          <div>
            <div className="mb-1 text-[11.5px] text-faint">Bootstrap distribution of the {stat} ({B.length} resamples)</div>
            <Plot x={[lo, hi * 0.6]} y={[0, Math.max(0.05, ...bins.map((b) => b.density)) * 1.1]} aspect={0.38} maxH={300} ariaLabel="bootstrap distribution">
              <Bars bins={bins} />
              <VLine x={est} color={C.green} label={`sample ${stat}`} />
              {ci && <><VLine x={ci[0]} color={C.pink} /><VLine x={ci[1]} color={C.pink} /></>}
            </Plot>
          </div>
        </div>
      }
      controls={
        <>
          <Select label="Statistic" value={stat} onChange={(v) => { setStat(v); setB([]); }} options={[{ value: "median", label: "Median" }, { value: "mean", label: "Mean" }]} />
          <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onStep={() => setB((b) => [...b, ...Array.from({ length: 100 }, one)])} onReset={() => { setPlaying(false); setB([]); }} />
        </>
      }
      readouts={<Readouts items={[{ label: `\\text{sample ${stat}}`, tex: true, value: fmt(est) }, { label: "\\text{bootstrap SE}", tex: true, value: B.length > 2 ? fmt(std(B)) : "—" }, { label: "95\\%\\ \\text{percentile CI}", tex: true, value: ci ? `[${fmt(ci[0])}, ${fmt(ci[1])}]` : "—" }]} />}
      calc={<CalcSteps steps={[{ tex: "x^{*(b)}\\sim\\text{resample }n\\text{ points from the data with replacement},\\quad\\hat\\theta^{*(b)}=s(x^{*(b)})" }, { tex: "\\widehat{\\operatorname{SE}}=\\operatorname{sd}\\big(\\hat\\theta^{*(1)},\\dots,\\hat\\theta^{*(B)}\\big)" }]} />}
      note="The bootstrap treats the sample as a stand-in for the population. It works for statistics without a simple standard-error formula (like the median) but cannot fix a small or unrepresentative sample."
    />
  );
}
