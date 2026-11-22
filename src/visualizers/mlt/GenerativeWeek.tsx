"use client";
import { useMemo, useState } from "react";
import { Plot, FnPath, Dot, Polyline, C } from "@/components/visualization/Plot";
import { AreaUnder, VLine } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend, Warn } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Toggle } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { fmt, tex, cn } from "@/lib/utilities/format";
import { MiniTable, BarList } from "@/visualizers/shared";

const npdf = (x: number, mu: number, s: number) => Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

/* ------------------------------------------------------------------ joint = prior × class-conditional */
export default function Generative() {
  const [p1, setP1] = useState(0.35);
  const [mu0, setMu0] = useState(-1);
  const [mu1, setMu1] = useState(1.6);
  const [s0, setS0] = useState(1);
  const [s1, setS1] = useState(1);
  const [q, setQ] = useState(0.4);
  const [view, setView] = useState<"joint" | "conditional">("joint");
  const [seed, setSeed] = useState(1);
  const j0 = (x: number) => (1 - p1) * npdf(x, mu0, s0), j1 = (x: number) => p1 * npdf(x, mu1, s1);
  const post = (x: number) => { const a = j1(x), b = j0(x); return a + b > 0 ? a / (a + b) : 0.5; };
  const sample = useMemo(() => { const r = mulberry32(seed * 23 + 7); return Array.from({ length: 60 }, () => { const y = r() < p1 ? 1 : 0; return { y, x: (y ? mu1 : mu0) + (y ? s1 : s0) * randn(r) }; }); }, [p1, mu0, mu1, s0, s1, seed]);
  // decision boundaries: sign changes of P(y=1|x) − ½ on a fine grid
  const bounds = useMemo(() => { const out: number[] = []; let prev = post(-7) - 0.5; for (let i = 1; i <= 1400; i++) { const x = -7 + i * 0.01, v = post(x) - 0.5; if (prev * v < 0) out.push(x - 0.005); prev = v; } return out; }, [p1, mu0, mu1, s0, s1]); // eslint-disable-line react-hooks/exhaustive-deps
  const f0 = view === "joint" ? j0 : (x: number) => npdf(x, mu0, s0), f1 = view === "joint" ? j1 : (x: number) => npdf(x, mu1, s1);
  const top = Math.max(...[mu0, mu1].map((m) => Math.max(f0(m), f1(m)))) * 1.18;
  const a = j1(q), b = j0(q);
  const bayesErr = useMemo(() => { let s = 0; for (let i = 0; i < 1400; i++) { const x = -7 + (i + 0.5) * 0.01; s += Math.min(j0(x), j1(x)) * 0.01; } return s; }, [p1, mu0, mu1, s0, s1]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-6, 7]} y={[0, top]} aspect={0.42} maxH={320} xLabel="x" yLabel={view === "joint" ? "P(y) · p(x | y)" : "p(x | y)"} ariaLabel="class-conditional or joint densities of the two classes" onPointerDown={(p) => setQ(Number(p[0].toFixed(2)))}>
            <AreaUnder f={f0} a={-6} b={7} color={C.a("blue", 0.16)} />
            <AreaUnder f={f1} a={-6} b={7} color={C.a("pink", 0.16)} />
            <FnPath f={f0} n={400} style={{ stroke: C.blue, strokeWidth: 2 }} />
            <FnPath f={f1} n={400} style={{ stroke: C.pink, strokeWidth: 2 }} />
            {view === "joint" && <FnPath f={(x) => j0(x) + j1(x)} n={400} style={{ stroke: C.a("fg", 0.55), strokeWidth: 1.3, strokeDasharray: "5 4" }} />}
            {sample.map((d, i) => <Polyline key={i} pts={[[d.x, 0], [d.x, top * 0.05]]} style={{ stroke: d.y ? C.pink : C.blue, strokeWidth: 1.1 }} />)}
            {bounds.map((x) => <VLine key={x} x={x} color={C.a("fg", 0.6)} />)}
            <VLine x={q} color={C.amber} label="x" />
            <Dot p={[q, f0(q)]} r={4.5} color={C.blue} stroke={C.bg} />
            <Dot p={[q, f1(q)]} r={4.5} color={C.pink} stroke={C.bg} />
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Posterior P(y = 1 | x) obtained from the joint by Bayes&apos; rule — predict class 1 where it exceeds ½</div>
            <Plot x={[-6, 7]} y={[0, 1.05]} aspect={0.22} minH={120} maxH={170} xLabel="x" ticks={4} ariaLabel="posterior probability of class one">
              <Polyline pts={[[-9, 0.5], [9, 0.5]]} style={{ stroke: C.a("fg", 0.25), strokeWidth: 1, strokeDasharray: "3 4" }} />
              <FnPath f={post} n={500} style={{ stroke: C.green, strokeWidth: 2 }} />
              {bounds.map((x) => <VLine key={x} x={x} color={C.a("fg", 0.6)} />)}
              <Dot p={[q, post(q)]} r={5} color={C.amber} stroke={C.bg} />
            </Plot>
          </div>
          <Legend items={[{ label: "class 0", color: C.blue }, { label: "class 1", color: C.pink }, ...(view === "joint" ? [{ label: "evidence p(x) = sum of the two joints", color: C.a("fg", 0.6), dashed: true }] : []), { label: "decision boundary", color: C.a("fg", 0.7), dashed: true }, { label: "ticks: 60 points sampled from the model", color: C.muted }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Curves shown" value={view} onChange={setView} options={[{ value: "joint", label: "Joint P(y)p(x|y)" }, { value: "conditional", label: "Class-conditional p(x|y)" }]} />
          <ParameterSlider label="class prior P(y = 1)" value={p1} min={0.02} max={0.98} step={0.01} onChange={setP1} />
          <div className="grid grid-cols-2 gap-x-3 gap-y-3">
            <ParameterSlider label="μ₀" value={mu0} min={-4} max={5} step={0.1} onChange={setMu0} />
            <ParameterSlider label="μ₁" value={mu1} min={-4} max={5} step={0.1} onChange={setMu1} />
            <ParameterSlider label="σ₀" value={s0} min={0.3} max={3} step={0.05} onChange={setS0} />
            <ParameterSlider label="σ₁" value={s1} min={0.3} max={3} step={0.05} onChange={setS1} />
          </div>
          <ParameterSlider label="query x (or click the plot)" value={q} min={-6} max={7} step={0.05} onChange={setQ} />
          <button className="btn w-full" onClick={() => setSeed((v) => v + 1)}>Generate a new sample</button>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`P(y=1)\,p(x\mid1)`, tex: true, value: fmt(a, 5), color: C.pink },
        { label: String.raw`P(y=0)\,p(x\mid0)`, tex: true, value: fmt(b, 5), color: C.blue },
        { label: String.raw`P(y=1\mid x)`, tex: true, value: fmt(post(q), 4), color: C.green },
        { label: String.raw`\text{prediction}`, tex: true, value: post(q) >= 0.5 ? "class 1" : "class 0" },
        { label: String.raw`\text{boundaries}`, tex: true, value: bounds.length ? bounds.map((x) => fmt(x, 2)).join(", ") : "none" },
        { label: String.raw`\text{error of this rule}`, tex: true, value: fmt(bayesErr, 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`P(x,y)=P(y)\,p(x\mid y):\quad P(x,1)=${tex(p1, 2)}\cdot${tex(npdf(q, mu1, s1), 5)}=${tex(a, 5)},\quad P(x,0)=${tex(1 - p1, 2)}\cdot${tex(npdf(q, mu0, s0), 5)}=${tex(b, 5)}`, note: `at x = ${fmt(q, 2)}` },
        { tex: String.raw`P(y=1\mid x)=\frac{P(x,1)}{P(x,0)+P(x,1)}=\frac{${tex(a, 5)}}{${tex(a + b, 5)}}=${tex(post(q), 4)}`, note: "Bayes' rule: normalise the joint over the classes" },
        { tex: String.raw`\hat y=\arg\max_{y}P(y)\,p(x\mid y)=${post(q) >= 0.5 ? 1 : 0}`, note: "the evidence p(x) is the same for both classes, so it can be ignored when predicting" },
      ]} />}
      note={Math.abs(s0 - s1) < 1e-9 ? "With equal variances the log-odds are linear in x and there is a single boundary. Changing the prior slides it: a rarer class needs stronger evidence." : "With unequal variances the log-odds are quadratic in x, so the wider class can win on both far sides: two boundaries. A generative model can also produce data — the tick marks were sampled from it."}
    />
  );
}

/* ------------------------------------------------------------------ Naive Bayes on binary features */
const WORDS = ["offer", "free", "meeting", "report"];
const TRUE1 = [0.7, 0.8, 0.15, 0.1], TRUE0 = [0.1, 0.25, 0.6, 0.55];

export function NaiveBayes() {
  const [n, setN] = useState(16);
  const [seed, setSeed] = useState(2);
  const [smooth, setSmooth] = useState(false);
  const [x, setX] = useState([1, 1, 0, 0]);
  const d = WORDS.length;
  const data = useMemo(() => { const r = mulberry32(seed * 41 + 9); return Array.from({ length: n }, () => { const y = r() < 0.4 ? 1 : 0; return { y, f: (y ? TRUE1 : TRUE0).map((p) => (r() < p ? 1 : 0)) }; }); }, [n, seed]);
  const n1 = data.filter((e) => e.y === 1).length, n0 = n - n1;
  const cnt = (y: number, j: number) => data.filter((e) => e.y === y && e.f[j] === 1).length;
  const est = (y: number, j: number) => { const c = cnt(y, j), m = y ? n1 : n0; return smooth ? (c + 1) / (m + 2) : m ? c / m : NaN; };
  const prior = smooth ? (n1 + 1) / (n + 2) : n1 / n;
  const lik = (y: number) => WORDS.reduce((s, _, j) => { const p = est(y, j); return s * (x[j] ? p : 1 - p); }, 1);
  const s1 = prior * lik(1), s0 = (1 - prior) * lik(0);
  const post = s1 + s0 > 0 ? s1 / (s1 + s0) : NaN;
  const degenerate = !Number.isFinite(post) || n1 === 0 || n0 === 0;
  const wj = WORDS.map((_, j) => { const a = est(1, j), b = est(0, j); return Math.log((a * (1 - b)) / (b * (1 - a))); });
  const bias = WORDS.reduce((s, _, j) => s + Math.log((1 - est(1, j)) / (1 - est(0, j))), 0) + Math.log(prior / (1 - prior));
  const lin = wj.reduce((s, w, j) => s + w * x[j], bias);
  const zeroes = WORDS.flatMap((w, j) => [0, 1].flatMap((y) => { const p = est(y, j); return p === 0 || p === 1 ? [`P(${w} = 1 | ${y ? "spam" : "ham"}) = ${p}`] : []; }));
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <div className="label mb-2">Training emails ({n1} spam, {n0} ham)</div>
              <div className="max-h-[250px] overflow-auto rounded-lg border border-line">
                <table className="w-full text-left text-[12px]">
                  <thead className="sticky top-0 bg-raised text-faint"><tr><th className="px-2 py-1.5 font-medium">#</th>{WORDS.map((w) => <th key={w} className="px-2 py-1.5 font-medium">{w}</th>)}<th className="px-2 py-1.5 font-medium">label</th></tr></thead>
                  <tbody>
                    {data.map((e, i) => (
                      <tr key={i} className="border-t border-line">
                        <td className="num px-2 py-1 text-faint">{i + 1}</td>
                        {e.f.map((v, j) => <td key={j} className={cn("num px-2 py-1", v ? "text-fg" : "text-faint")}>{v}</td>)}
                        <td className="px-2 py-1" style={{ color: e.y ? C.pink : C.blue }}>{e.y ? "spam" : "ham"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="space-y-3">
              <div className="label">Estimated parameters p̂ⱼʸ = P(word j present | class y)</div>
              <MiniTable head={["word", "count in spam", "P( · | spam)", "count in ham", "P( · | ham)", "weight wⱼ"]} rows={WORDS.map((w, j) => [w, `${cnt(1, j)} / ${n1}`, fmt(est(1, j), 3), `${cnt(0, j)} / ${n0}`, fmt(est(0, j), 3), Number.isFinite(wj[j]) ? fmt(wj[j], 3) : wj[j] > 0 ? "+∞" : "−∞"])} />
              <div className="label">Test email — click to toggle each word</div>
              <div className="flex flex-wrap gap-1.5">
                {WORDS.map((w, j) => <button key={w} onClick={() => setX(x.map((v, t) => (t === j ? 1 - v : v)))} aria-pressed={!!x[j]} className={cn("rounded-md border px-2.5 py-1 text-[12.5px]", x[j] ? "border-line-strong bg-hover text-fg" : "border-line text-faint")}>{w} = {x[j]}</button>)}
              </div>
              <BarList digits={4} max={1} items={[{ label: "P(spam | x)", value: Number.isFinite(post) ? post : 0, color: C.pink }, { label: "P(ham | x)", value: Number.isFinite(post) ? 1 - post : 0, color: C.blue }]} />
            </div>
          </div>
          {degenerate && <Warn>{n1 === 0 || n0 === 0 ? "One class has no training examples, so its parameters cannot be estimated." : "Both class scores are exactly zero: the test email contains a combination that each class has ruled out with a zero estimate. Turn on Laplace smoothing."}</Warn>}
          {!degenerate && zeroes.length > 0 && <Warn>Zero or one estimates: {zeroes.slice(0, 3).join("; ")}{zeroes.length > 3 ? "; …" : ""}. A single such factor forces the whole product for that class to 0 whatever the other words say.</Warn>}
        </div>
      }
      controls={
        <>
          <ParameterSlider label="training emails n" value={n} min={6} max={60} step={1} onChange={setN} />
          <Toggle label="Laplace smoothing (add one pseudo-count each way)" checked={smooth} onChange={setSmooth} />
          <button className="btn w-full" onClick={() => setSeed((v) => v + 1)}>New training set</button>
          <div className="space-y-1 text-[12.5px] leading-relaxed text-muted">
            <p>Parameters to estimate with d = {d} binary features:</p>
            <p>without assumptions: 1 + 2(2ᵈ − 1) = <span className="num text-fg">{1 + 2 * (2 ** d - 1)}</span></p>
            <p>with Naive Bayes: 1 + 2d = <span className="num text-fg">{1 + 2 * d}</span></p>
          </div>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\hat p=P(y=\text{spam})`, tex: true, value: fmt(prior, 4) },
        { label: String.raw`P(\text{spam})\prod_jP(x_j\mid\text{spam})`, tex: true, value: fmt(s1, 6), color: C.pink },
        { label: String.raw`P(\text{ham})\prod_jP(x_j\mid\text{ham})`, tex: true, value: fmt(s0, 6), color: C.blue },
        { label: String.raw`P(\text{spam}\mid\mathbf x)`, tex: true, value: fmt(post, 4), color: C.pink },
        { label: String.raw`\text{prediction}`, tex: true, value: degenerate ? "undefined" : post >= 0.5 ? "spam" : "ham" },
        { label: String.raw`\mathbf w^\top\mathbf x+b`, tex: true, value: Number.isFinite(lin) ? fmt(lin, 4) : "±∞" },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`P(\mathbf x\mid\text{spam})=\prod_{j=1}^{${d}}(\hat p_j^{1})^{x_j}(1-\hat p_j^{1})^{1-x_j}=${WORDS.map((_, j) => tex(x[j] ? est(1, j) : 1 - est(1, j), 3)).join(String.raw`\times`)}=${tex(lik(1), 5)}`, note: "independence given the class: multiply one factor per word" },
        { tex: String.raw`P(\mathbf x\mid\text{ham})=${WORDS.map((_, j) => tex(x[j] ? est(0, j) : 1 - est(0, j), 3)).join(String.raw`\times`)}=${tex(lik(0), 5)}` },
        { tex: String.raw`P(\text{spam}\mid\mathbf x)=\frac{${tex(prior, 3)}\times${tex(lik(1), 5)}}{${tex(prior, 3)}\times${tex(lik(1), 5)}+${tex(1 - prior, 3)}\times${tex(lik(0), 5)}}=${tex(post, 4)}` },
        ...(Number.isFinite(lin) && Number.isFinite(post) && post > 0 && post < 1 ? [{ tex: String.raw`\log\frac{P(\text{spam}\mid\mathbf x)}{P(\text{ham}\mid\mathbf x)}=${tex(Math.log(post / (1 - post)), 4)}=\mathbf w^\top\mathbf x+b=${tex(lin, 4)}`, note: "the same decision written as a linear classifier" }] : []),
      ]} />}
      note="Each parameter is just a fraction of training emails, which is the maximum-likelihood estimate. The training data are generated from a model in which the words really are independent given the class, so here the assumption is true; on real text it is not, yet the classifier often still ranks the classes correctly."
    />
  );
}
