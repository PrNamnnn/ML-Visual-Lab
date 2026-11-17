"use client";
import { useMemo, useState } from "react";
import { Plot, FnPath, Dot, Polyline, C, useThemeRGB } from "@/components/visualization/Plot";
import { Bars, VLine } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, AnimationControls } from "@/components/controls";
import { useTicker } from "@/hooks/useSize";
import { mulberry32, randn } from "@/lib/math/random";
import { histogram } from "@/lib/statistics/descriptive";
import { fmt, tex } from "@/lib/utilities/format";
import { clsColor, MiniTable } from "@/visualizers/shared";
import { Trace } from "./common";

const npdf = (x: number, mu: number, s: number) => Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

/* ------------------------------------------------------------------ maximum likelihood: Bernoulli and Gaussian */
function BernoulliMLE() {
  const [trueP, setTrueP] = useState(0.7);
  const [n, setN] = useState(10);
  const [seed, setSeed] = useState(4);
  const [p, setP] = useState(0.5);
  const [view, setView] = useState<"log" | "lik">("log");
  const flips = useMemo(() => { const r = mulberry32(seed * 7 + 1); return Array.from({ length: n }, () => (r() < trueP ? 1 : 0)); }, [trueP, n, seed]);
  const h = flips.reduce((a: number, b) => a + b, 0), t = n - h;
  const ll = (q: number) => (h ? h * Math.log(q) : 0) + (t ? t * Math.log(1 - q) : 0);
  const hat = h / n;
  const llHat = ll(Math.min(1 - 1e-12, Math.max(1e-12, hat)));
  const grad = h / p - t / (1 - p);
  const yLo = Math.max(-60, Math.min(ll(0.05), ll(0.95), llHat - 12));
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <div className="flex flex-wrap gap-[3px]" aria-label={`${h} heads and ${t} tails`}>
            {flips.slice(0, 200).map((x, i) => <span key={i} className="flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-medium" style={{ background: x ? "rgb(var(--green) / 0.25)" : "rgb(var(--line-strong))", color: x ? "rgb(var(--green))" : "rgb(var(--muted))" }}>{x ? "H" : "T"}</span>)}
          </div>
          {view === "log" ? (
            <Plot x={[0, 1]} y={[yLo, Math.min(0, llHat + 1.5)]} aspect={0.5} maxH={380} xLabel="p" yLabel="log-likelihood ℓ(p)" ariaLabel="Bernoulli log-likelihood as a function of p">
              <FnPath f={ll} domain={[0.002, 0.998]} n={400} style={{ stroke: C.blue }} />
              <VLine x={hat} color={C.green} label="p̂" />
              <VLine x={trueP} color={C.a("amber", 0.7)} label="true p" />
              <Polyline pts={[[p - 0.08, ll(p) - 0.08 * grad], [p + 0.08, ll(p) + 0.08 * grad]]} style={{ stroke: C.pink, strokeWidth: 2 }} />
              <Dot p={[p, ll(p)]} r={5} color={C.pink} stroke={C.bg} />
            </Plot>
          ) : (
            <Plot x={[0, 1]} y={[0, 1.08]} aspect={0.5} maxH={380} xLabel="p" yLabel="L(p) / L(p̂)" ariaLabel="Bernoulli likelihood relative to its maximum">
              <FnPath f={(q) => Math.exp(ll(q) - llHat)} domain={[0.002, 0.998]} n={400} style={{ stroke: C.blue }} />
              <VLine x={hat} color={C.green} label="p̂" />
              <VLine x={trueP} color={C.a("amber", 0.7)} label="true p" />
              <Dot p={[p, Math.exp(ll(p) - llHat)]} r={5} color={C.pink} stroke={C.bg} />
            </Plot>
          )}
          <Legend items={[{ label: view === "log" ? "ℓ(p) = n_h log p + n_t log(1 − p)" : "likelihood, scaled so its peak is 1", color: C.blue }, { label: "maximum-likelihood estimate", color: C.green, dashed: true }, { label: "candidate p and slope dℓ/dp", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="candidate p" value={p} min={0.01} max={0.99} step={0.01} onChange={setP} />
          <button className="btn w-full" onClick={() => setP(Math.min(0.99, Math.max(0.01, Number(hat.toFixed(2)))))}>Move candidate to p̂</button>
          <Segmented value={view} onChange={setView} options={[{ value: "log", label: "Log-likelihood" }, { value: "lik", label: "Likelihood" }]} />
          <ParameterSlider label="true p (hidden from the estimator)" value={trueP} min={0.05} max={0.95} step={0.01} onChange={setTrueP} />
          <ParameterSlider label="number of tosses n" value={n} min={1} max={200} step={1} onChange={setN} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>Toss again</button>
        </>
      }
      readouts={<Readouts items={[
        { label: "n_h,\\ n_t", tex: true, value: `${h}, ${t}` },
        { label: String.raw`\hat p_{\text{ML}}=n_h/n`, tex: true, value: fmt(hat, 4), color: C.green },
        { label: String.raw`\ell(\hat p)`, tex: true, value: fmt(llHat, 4) },
        { label: String.raw`\ell(p)`, tex: true, value: fmt(ll(p), 4), color: C.pink },
        { label: String.raw`d\ell/dp`, tex: true, value: fmt(grad, 3), color: C.pink },
        { label: String.raw`|\hat p-p_{\text{true}}|`, tex: true, value: fmt(Math.abs(hat - trueP), 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`L(p)=\prod_{i=1}^{n}p^{x_i}(1-p)^{1-x_i}=p^{${h}}(1-p)^{${t}}` },
        { tex: String.raw`\ell(p)=${h}\log p+${t}\log(1-p)\ \Rightarrow\ \ell(${tex(p, 2)})=${tex(ll(p), 4)}` },
        { tex: String.raw`\frac{d\ell}{dp}=\frac{${h}}{p}-\frac{${t}}{1-p}=0\ \Rightarrow\ \hat p=\frac{${h}}{${n}}=${tex(hat, 4)}`, note: `slope at the candidate: ${fmt(grad, 3)}` },
      ]} />}
      note="The slope is positive to the left of p̂ and negative to the right, so p̂ is the unique maximum. With few tosses the curve is wide and p̂ can sit far from the true p; more data sharpens the peak."
    />
  );
}

function GaussianMLE() {
  const [trueMu, setTrueMu] = useState(1);
  const [trueS, setTrueS] = useState(1.2);
  const [n, setN] = useState(30);
  const [seed, setSeed] = useState(2);
  const [mu, setMu] = useState(-0.5);
  const [s, setS] = useState(2);
  const xs = useMemo(() => { const r = mulberry32(seed * 11 + 3); return Array.from({ length: n }, () => trueMu + trueS * randn(r)); }, [trueMu, trueS, n, seed]);
  const muHat = xs.reduce((a, b) => a + b, 0) / n;
  const varHat = xs.reduce((a, b) => a + (b - muHat) ** 2, 0) / n;
  const sHat = Math.sqrt(Math.max(1e-9, varHat));
  const ll = (m: number, sd: number) => -0.5 * n * Math.log(2 * Math.PI * sd * sd) - xs.reduce((a, b) => a + (b - m) ** 2, 0) / (2 * sd * sd);
  const llHat = ll(muHat, sHat);
  const bins = useMemo(() => histogram(xs, -5, 7, 24), [xs]);
  const top = Math.max(0.6, ...bins.map((b) => b.density), npdf(mu, mu, s)) * 1.12;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-5, 7]} y={[0, top]} aspect={0.42} maxH={300} xLabel="x" ariaLabel="sample histogram with candidate and fitted Gaussian densities">
            <Bars bins={bins} />
            <FnPath f={(x) => npdf(x, muHat, sHat)} n={300} style={{ stroke: C.green, strokeWidth: 1.6, strokeDasharray: "5 4" }} />
            <FnPath f={(x) => npdf(x, mu, s)} n={300} style={{ stroke: C.pink, strokeWidth: 2.2 }} />
            {xs.map((x, i) => <Polyline key={i} pts={[[x, 0], [x, top * 0.04]]} style={{ stroke: C.a("fg", 0.6), strokeWidth: 1 }} />)}
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Log-likelihood as μ varies (σ held at the candidate value)</div>
            <Plot x={[-5, 7]} y={[Math.max(llHat - 60, ll(muHat, s) - 60), ll(muHat, s) + 4]} aspect={0.3} minH={130} maxH={190} xLabel="μ" ariaLabel="Gaussian log-likelihood as a function of the mean">
              <FnPath f={(m) => ll(m, s)} n={200} style={{ stroke: C.blue }} />
              <VLine x={muHat} color={C.green} label="μ̂" />
              <Dot p={[mu, ll(mu, s)]} r={5} color={C.pink} stroke={C.bg} />
            </Plot>
          </div>
          <Legend items={[{ label: "sample", color: C.blue }, { label: "candidate N(μ, σ²)", color: C.pink }, { label: "maximum-likelihood fit", color: C.green, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="candidate μ" value={mu} min={-4} max={6} step={0.05} onChange={setMu} />
          <ParameterSlider label="candidate σ" value={s} min={0.3} max={4} step={0.05} onChange={setS} />
          <button className="btn w-full" onClick={() => { setMu(Number(muHat.toFixed(2))); setS(Number(Math.max(0.3, sHat).toFixed(2))); }}>Move candidate to the MLE</button>
          <ParameterSlider label="true μ" value={trueMu} min={-3} max={5} step={0.1} onChange={setTrueMu} />
          <ParameterSlider label="true σ" value={trueS} min={0.4} max={3} step={0.1} onChange={setTrueS} />
          <ParameterSlider label="sample size n" value={n} min={2} max={300} step={1} onChange={setN} />
          <button className="btn w-full" onClick={() => setSeed((v) => v + 1)}>Draw a new sample</button>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\hat\mu=\bar x`, tex: true, value: fmt(muHat, 4), color: C.green },
        { label: String.raw`\hat\sigma^2=\tfrac1n\sum(x_i-\hat\mu)^2`, tex: true, value: fmt(varHat, 4), color: C.green },
        { label: String.raw`\tfrac{1}{n-1}\sum(x_i-\hat\mu)^2`, tex: true, value: n > 1 ? fmt((varHat * n) / (n - 1), 4) : "—" },
        { label: String.raw`\ell(\hat\mu,\hat\sigma)`, tex: true, value: fmt(llHat, 3) },
        { label: String.raw`\ell(\mu,\sigma)`, tex: true, value: fmt(ll(mu, s), 3), color: C.pink },
        { label: String.raw`\ell(\hat\mu,\hat\sigma)-\ell(\mu,\sigma)`, tex: true, value: fmt(llHat - ll(mu, s), 3) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\ell(\mu,\sigma^2)=-\frac n2\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2=${tex(ll(mu, s), 3)}` },
        { tex: String.raw`\frac{\partial\ell}{\partial\mu}=\frac{1}{\sigma^2}\sum_i(x_i-\mu)=0\ \Rightarrow\ \hat\mu=\frac1n\sum_ix_i=${tex(muHat, 4)}` },
        { tex: String.raw`\frac{\partial\ell}{\partial\sigma^2}=-\frac{n}{2\sigma^2}+\frac{1}{2\sigma^4}\sum_i(x_i-\hat\mu)^2=0\ \Rightarrow\ \hat\sigma^2=${tex(varHat, 4)}`, note: "divides by n, so it is slightly biased downwards" },
      ]} />}
      note="No candidate can beat the fitted curve: the last readout is never negative. The 1/n variance estimate is the maximum-likelihood answer; the unbiased estimate divides by n − 1 instead."
    />
  );
}

function EstimationViz({ model: initial = "bernoulli" }: { model?: "bernoulli" | "gaussian" }) {
  const [model, setModel] = useState(initial);
  return (
    <div className="space-y-3">
      <div className="max-w-sm"><Segmented label="Model" value={model} onChange={setModel} options={[{ value: "bernoulli", label: "Bernoulli (coin)" }, { value: "gaussian", label: "Gaussian" }]} /></div>
      {model === "bernoulli" ? <BernoulliMLE /> : <GaussianMLE />}
    </div>
  );
}

/* ------------------------------------------------------------------ Gaussian mixture: density and likelihood */
interface Comp { pi: number; mu: number; s: number }
const mixPdf = (x: number, cs: Comp[]) => cs.reduce((a, c) => a + c.pi * npdf(x, c.mu, c.s), 0);
const resp = (x: number, cs: Comp[]) => { const v = cs.map((c) => c.pi * npdf(x, c.mu, c.s)); const t = v.reduce((a, b) => a + b, 0) || 1e-300; return v.map((u) => u / t); };
const logLik = (xs: number[], cs: Comp[]) => xs.reduce((a, x) => a + Math.log(Math.max(1e-300, mixPdf(x, cs))), 0);

function sampleMixture(cs: Comp[], n: number, seed: number) {
  const r = mulberry32(seed);
  return Array.from({ length: n }, () => {
    let u = r(), k = 0;
    for (; k < cs.length - 1; k++) { u -= cs[k].pi; if (u <= 0) break; }
    return { x: cs[k].mu + cs[k].s * randn(r), z: k };
  });
}

function MixtureDensity() {
  const [pi1, setPi1] = useState(0.4);
  const [mu1, setMu1] = useState(-1.5);
  const [mu2, setMu2] = useState(2);
  const [s1, setS1] = useState(0.8);
  const [s2, setS2] = useState(1.2);
  const [q, setQ] = useState(0.4);
  const [seed, setSeed] = useState(1);
  const cs: Comp[] = [{ pi: pi1, mu: mu1, s: s1 }, { pi: 1 - pi1, mu: mu2, s: s2 }];
  const data = useMemo(() => sampleMixture(cs, 80, seed * 13 + 5), [pi1, mu1, mu2, s1, s2, seed]); // eslint-disable-line react-hooks/exhaustive-deps
  const lam = resp(q, cs);
  const parts = cs.map((c) => c.pi * npdf(q, c.mu, c.s));
  const ll = logLik(data.map((d) => d.x), cs);
  const top = Math.max(0.2, ...[-6, -4, -2, 0, 2, 4, 6, mu1, mu2].map((x) => mixPdf(x, cs))) * 1.2;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-6, 7]} y={[0, top]} aspect={0.5} maxH={400} xLabel="x" ariaLabel="Gaussian mixture density and its components" onPointerDown={(p) => setQ(Number(p[0].toFixed(2)))}>
            {cs.map((c, k) => <FnPath key={k} f={(x) => c.pi * npdf(x, c.mu, c.s)} n={300} style={{ stroke: clsColor(k), strokeWidth: 1.5, strokeDasharray: "5 4" }} />)}
            <FnPath f={(x) => mixPdf(x, cs)} n={400} style={{ stroke: C.fg, strokeWidth: 2.2 }} />
            {data.map((d, i) => <Polyline key={i} pts={[[d.x, 0], [d.x, top * 0.05]]} style={{ stroke: clsColor(d.z), strokeWidth: 1.2 }} />)}
            <VLine x={q} color={C.a("fg", 0.7)} label="x" />
            {parts.map((v, k) => <Dot key={k} p={[q, v]} r={4.5} color={clsColor(k)} stroke={C.bg} />)}
          </Plot>
          <Legend items={[{ label: "π₁ N(x; μ₁, σ₁²)", color: clsColor(0), dashed: true }, { label: "π₂ N(x; μ₂, σ₂²)", color: clsColor(1), dashed: true }, { label: "mixture density f(x)", color: C.fg }, { label: "sampled points, coloured by the hidden zᵢ", color: C.muted }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="mixing weight π₁ (π₂ = 1 − π₁)" value={pi1} min={0.05} max={0.95} step={0.01} onChange={setPi1} />
          <div className="grid grid-cols-2 gap-x-3 gap-y-3">
            <ParameterSlider label="μ₁" value={mu1} min={-5} max={6} step={0.1} onChange={setMu1} />
            <ParameterSlider label="μ₂" value={mu2} min={-5} max={6} step={0.1} onChange={setMu2} />
            <ParameterSlider label="σ₁" value={s1} min={0.3} max={3} step={0.05} onChange={setS1} />
            <ParameterSlider label="σ₂" value={s2} min={0.3} max={3} step={0.05} onChange={setS2} />
          </div>
          <ParameterSlider label="query point x (or click the plot)" value={q} min={-6} max={7} step={0.05} onChange={setQ} />
          <button className="btn w-full" onClick={() => setSeed((v) => v + 1)}>Sample 80 new points</button>
        </>
      }
      readouts={<Readouts items={[
        { label: "f(x)", tex: true, value: fmt(mixPdf(q, cs), 5) },
        { label: String.raw`\lambda_1(x)=P(z=1\mid x)`, tex: true, value: fmt(lam[0], 4), color: clsColor(0) },
        { label: String.raw`\lambda_2(x)=P(z=2\mid x)`, tex: true, value: fmt(lam[1], 4), color: clsColor(1) },
        { label: String.raw`\ell(\theta)\ \text{of the sample}`, tex: true, value: fmt(ll, 3) },
        { label: String.raw`\text{free parameters}=3K-1`, tex: true, value: 5 },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`f(x)=\sum_{k=1}^{2}\pi_k\,\mathcal N(x;\mu_k,\sigma_k^2)=${tex(parts[0], 5)}+${tex(parts[1], 5)}=${tex(mixPdf(q, cs), 5)}`, note: `at x = ${fmt(q, 2)}` },
        { tex: String.raw`\lambda_1(x)=\frac{\pi_1\mathcal N(x;\mu_1,\sigma_1^2)}{f(x)}=\frac{${tex(parts[0], 5)}}{${tex(mixPdf(q, cs), 5)}}=${tex(lam[0], 4)}`, note: "Bayes' rule for the hidden component" },
        { tex: String.raw`\ell(\theta)=\sum_{i=1}^{n}\log\Big(\sum_{k}\pi_k\mathcal N(x_i;\mu_k,\sigma_k^2)\Big)=${tex(ll, 3)}`, note: "the sum sits inside the logarithm" },
      ]} />}
      note="Each sampled point was produced in two stages: first a component z was drawn with probabilities π, then x was drawn from that component's Gaussian. The estimator sees only x; the colours of the tick marks are the information it is missing."
    />
  );
}

/* ------------------------------------------------------------------ EM for a 1-D Gaussian mixture */
interface EMState { cs: Comp[]; R: number[][] | null; ll: number; phase: "init" | "E" | "M" }

function EMViz() {
  const [K, setK] = useState(2);
  const [seed, setSeed] = useState(3);
  const [init, setInit] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [sel, setSel] = useState(0);
  const th = useThemeRGB();
  const truth: Comp[] = useMemo(() => (K === 2 ? [{ pi: 0.4, mu: -2, s: 0.8 }, { pi: 0.6, mu: 2, s: 1.1 }] : [{ pi: 0.3, mu: -3, s: 0.7 }, { pi: 0.4, mu: 0.3, s: 0.8 }, { pi: 0.3, mu: 3.4, s: 0.9 }]), [K]);
  const xs = useMemo(() => sampleMixture(truth, 150, seed * 17 + K).map((d) => d.x), [truth, seed, K]);
  const start = useMemo((): EMState[] => {
    const r = mulberry32(init * 53 + K);
    const cs = Array.from({ length: K }, () => ({ pi: 1 / K, mu: xs[Math.floor(r() * xs.length)], s: 1.5 }));
    return [{ cs, R: null, ll: logLik(xs, cs), phase: "init" }];
  }, [xs, init, K]);
  const [hist, setHist] = useState<EMState[] | null>(null);
  const H = hist && hist[0] === start[0] ? hist : start;
  const cur = H[H.length - 1];
  const lls = H.filter((h) => h.phase !== "E").map((h) => h.ll);
  const done = lls.length > 3 && Math.abs(lls[lls.length - 1] - lls[lls.length - 2]) < 1e-6;
  const step = () => {
    if (done) return;
    if (cur.phase !== "E") {
      setHist([...H, { cs: cur.cs, R: xs.map((x) => resp(x, cur.cs)), ll: cur.ll, phase: "E" }]);
    } else {
      const R = cur.R!;
      const cs = cur.cs.map((_, k) => {
        const nk = R.reduce((a, r) => a + r[k], 0) || 1e-12;
        const mu = R.reduce((a, r, i) => a + r[k] * xs[i], 0) / nk;
        const v = R.reduce((a, r, i) => a + r[k] * (xs[i] - mu) ** 2, 0) / nk;
        return { pi: nk / xs.length, mu, s: Math.sqrt(Math.max(1e-4, v)) };
      });
      setHist([...H, { cs, R, ll: logLik(xs, cs), phase: "M" }]);
    }
  };
  useTicker(playing && !done, 500, step);
  const bins = useMemo(() => histogram(xs, -6, 7, 40), [xs]);
  const top = Math.max(...bins.map((b) => b.density), 0.3) * 1.2;
  const R = cur.R ?? xs.map((x) => resp(x, cur.cs));
  const iter = H.filter((h) => h.phase === "M").length;
  const i = Math.min(sel, xs.length - 1);
  const parts = cur.cs.map((c) => c.pi * npdf(xs[i], c.mu, c.s));
  const pal = [th.blue, th.pink, th.green];
  const mix = (r: number[]) => { let a = 0, b = 0, c = 0; r.forEach((w, k) => { a += w * pal[k][0]; b += w * pal[k][1]; c += w * pal[k][2]; }); return `rgb(${a | 0} ${b | 0} ${c | 0})`; };
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <Plot x={[-6, 7]} y={[0, top]} aspect={0.44} maxH={340} xLabel="x" ariaLabel="data histogram with the current mixture estimate" onPointerDown={(p) => { let b = 0, bd = Infinity; xs.forEach((x, j) => { const d = Math.abs(x - p[0]); if (d < bd) { bd = d; b = j; } }); setSel(b); }}>
            <Bars bins={bins} color={C.a("fg", 0.1)} stroke={C.a("fg", 0.25)} />
            {cur.cs.map((c, k) => <FnPath key={k} f={(x) => c.pi * npdf(x, c.mu, c.s)} n={300} style={{ stroke: clsColor(k), strokeWidth: 1.8 }} />)}
            <FnPath f={(x) => mixPdf(x, cur.cs)} n={400} style={{ stroke: C.fg, strokeWidth: 1.4, strokeDasharray: "5 4" }} />
            {xs.map((x, j) => <Polyline key={j} pts={[[x, 0], [x, top * (j === i ? 0.12 : 0.055)]]} style={{ stroke: mix(R[j]), strokeWidth: j === i ? 2.4 : 1.2 }} />)}
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Log-likelihood after each M-step — EM never decreases it</div>
            <Trace ariaLabel="log-likelihood per EM iteration" xLabel="iteration" series={[{ pts: lls, color: C.green, dots: true }]} xMax={Math.max(8, lls.length - 1)} yDomain={[Math.min(...lls) - 2, Math.max(...lls) + 2]} />
          </div>
          <Legend items={[...cur.cs.map((_, k) => ({ label: `πₖ N(x; μₖ, σₖ²), k = ${k + 1}`, color: clsColor(k) })), { label: "mixture", color: C.fg, dashed: true }, { label: "ticks: data, coloured by responsibilities λᵢₖ", color: C.muted }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Components K" value={String(K)} onChange={(v) => { setPlaying(false); setK(Number(v)); setSel(0); }} options={[{ value: "2", label: "K = 2" }, { value: "3", label: "K = 3" }]} />
          <AnimationControls playing={playing && !done} onPlay={() => setPlaying(!playing)} onStep={step} canStep={!done} onReset={() => { setPlaying(false); setHist(null); }} />
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={() => { setPlaying(false); setInit((v) => v + 1); }}>New initialisation</button>
            <button className="btn flex-1" onClick={() => { setPlaying(false); setSeed((v) => v + 1); }}>New data</button>
          </div>
          <MiniTable head={["k", "πₖ", "μₖ", "σₖ", "true (π, μ, σ)"]} rows={cur.cs.map((c, k) => [k + 1, fmt(c.pi, 3), fmt(c.mu, 3), fmt(c.s, 3), `${truth[k].pi}, ${truth[k].mu}, ${truth[k].s}`])} />
          <p className="text-[12.5px] leading-relaxed text-muted">
            {cur.phase === "init" ? "Starting guess: equal weights, means at random data points. Step to run the E-step." : done ? `Converged after ${iter} iterations (change in ℓ below 10⁻⁶). Component order is arbitrary, so compare with the true values up to relabelling.` : cur.phase === "E" ? "E-step done: every point now has responsibilities λᵢₖ under the current parameters (tick colours). Next: M-step." : "M-step done: parameters re-estimated as λ-weighted averages. Next: E-step."}
          </p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{iteration}`, tex: true, value: iter },
        { label: String.raw`\ell(\theta^t)`, tex: true, value: fmt(cur.ll, 4), color: C.green },
        { label: String.raw`\Delta\ell`, tex: true, value: lls.length > 1 ? fmt(lls[lls.length - 1] - lls[lls.length - 2], 5) : "—" },
        { label: String.raw`\ell(\theta_{\text{true}})`, tex: true, value: fmt(logLik(xs, truth), 4) },
        { label: "n", tex: true, value: xs.length },
      ]} />}
      calc={<CalcSteps title={`Live calculation for the selected point x = ${fmt(xs[i], 3)} (click a tick to change)`} steps={[
        { tex: String.raw`\text{E-step: }\lambda_{ik}=\frac{\pi_k\mathcal N(x_i;\mu_k,\sigma_k^2)}{\sum_l\pi_l\mathcal N(x_i;\mu_l,\sigma_l^2)}=\frac{(${parts.map((v) => tex(v, 4)).join(",\\ ")})}{${tex(parts.reduce((a, b) => a + b, 0), 4)}}=(${R[i].map((v) => tex(v, 3)).join(",\\ ")})` },
        { tex: String.raw`\text{M-step: }\mu_k=\frac{\sum_i\lambda_{ik}x_i}{\sum_i\lambda_{ik}},\quad\sigma_k^2=\frac{\sum_i\lambda_{ik}(x_i-\mu_k)^2}{\sum_i\lambda_{ik}},\quad\pi_k=\frac1n\sum_i\lambda_{ik}` },
        { tex: String.raw`\textstyle\sum_i\lambda_{ik}=(${cur.cs.map((_, k) => tex(R.reduce((a, r) => a + r[k], 0), 2)).join(",\\ ")})`, note: "effective number of points owned by each component" },
      ]} />}
      note="EM climbs to a local maximum of the likelihood. Some initialisations end at a poorer solution (for example two components sharing one cluster): compare the final ℓ with ℓ at the true parameters."
    />
  );
}

export default function EstimationWeek({ mode = "mle", model }: { mode?: "mle" | "mixture" | "em"; model?: "bernoulli" | "gaussian" }) {
  if (mode === "mixture") return <MixtureDensity />;
  if (mode === "em") return <EMViz />;
  return <EstimationViz model={model} />;
}
