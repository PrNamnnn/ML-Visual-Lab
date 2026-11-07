"use client";
import { useMemo, useState } from "react";
import { Plot, FnPath, Dot, C, Polyline } from "@/components/visualization/Plot";
import { Bars, AreaUnder, VLine } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented, AnimationControls } from "@/components/controls";
import { distributionById } from "@/lib/statistics/distributions";
import { mean, std, variance, histogram, pearson } from "@/lib/statistics/descriptive";
import { mulberry32, randn, shuffle, randBivariate } from "@/lib/math/random";
import { solve } from "@/lib/math/linalg";
import { useTicker } from "@/hooks/useSize";
import { fmt, pct, tex, cn } from "@/lib/utilities/format";

/* ------------------------------ sample spaces & axioms (two dice) ------------------------------ */
type Ev = "sum7" | "doubles" | "first6" | "sumGt8" | "even";
const EVENTS: Record<Ev, { label: string; test: (a: number, b: number) => boolean }> = {
  sum7: { label: "sum = 7", test: (a, b) => a + b === 7 },
  doubles: { label: "doubles", test: (a, b) => a === b },
  first6: { label: "first die = 6", test: (a) => a === 6 },
  sumGt8: { label: "sum > 8", test: (a, b) => a + b > 8 },
  even: { label: "sum is even", test: (a, b) => (a + b) % 2 === 0 },
};

export function DiceViz({ mode = "space" }: { mode?: "space" | "axioms" }) {
  const [A, setA] = useState<Ev>("sumGt8");
  const [B, setB] = useState<Ev>("doubles");
  const cells = Array.from({ length: 36 }, (_, i) => ({ a: Math.floor(i / 6) + 1, b: (i % 6) + 1 }));
  const inA = (c: { a: number; b: number }) => EVENTS[A].test(c.a, c.b);
  const inB = (c: { a: number; b: number }) => EVENTS[B].test(c.a, c.b);
  const nA = cells.filter(inA).length, nB = cells.filter(inB).length, nAB = cells.filter((c) => inA(c) && inB(c)).length;
  const nU = nA + nB - nAB;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div className="inline-grid gap-1" style={{ gridTemplateColumns: "28px repeat(6, 40px)" }}>
            <span />
            {[1, 2, 3, 4, 5, 6].map((b) => <span key={b} className="text-center text-[11px] text-faint">{b}</span>)}
            {[1, 2, 3, 4, 5, 6].map((a) => (
              <div key={a} className="contents">
                <span className="flex items-center justify-end pr-1 text-[11px] text-faint">{a}</span>
                {[1, 2, 3, 4, 5, 6].map((b) => {
                  const c = { a, b };
                  const ia = inA(c), ib = mode === "axioms" && inB(c);
                  return (
                    <div key={b} className={cn("flex h-9 items-center justify-center rounded text-[11px]", ia && ib && "ring-2 ring-fg")} style={{ background: ia && ib ? "rgb(var(--amber) / 0.55)" : ia ? "rgb(var(--blue) / 0.45)" : ib ? "rgb(var(--pink) / 0.45)" : "rgb(var(--raised))" }}>
                      {a + b}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          <Legend items={mode === "axioms" ? [{ label: `A: ${EVENTS[A].label}`, color: C.blue }, { label: `B: ${EVENTS[B].label}`, color: C.pink }, { label: "A ∩ B", color: C.amber }] : [{ label: `event A: ${EVENTS[A].label}`, color: C.blue }]} />
          <p className="text-[12px] text-faint">Rows: first die, columns: second die; each of the 36 outcomes is equally likely. Cells show the sum.</p>
        </div>
      }
      controls={
        <>
          <Select label="Event A" value={A} onChange={setA} options={Object.entries(EVENTS).map(([k, v]) => ({ value: k as Ev, label: v.label }))} />
          {mode === "axioms" && <Select label="Event B" value={B} onChange={setB} options={Object.entries(EVENTS).map(([k, v]) => ({ value: k as Ev, label: v.label }))} />}
        </>
      }
      readouts={<Readouts items={mode === "axioms" ? [
        { label: "P(A)", tex: true, value: `${nA}/36 = ${fmt(nA / 36, 4)}` },
        { label: "P(B)", tex: true, value: `${nB}/36 = ${fmt(nB / 36, 4)}` },
        { label: "P(A\\cap B)", tex: true, value: `${nAB}/36` },
        { label: "P(A\\cup B)", tex: true, value: `${nU}/36 = ${fmt(nU / 36, 4)}` },
        { label: "P(A^c)", tex: true, value: `${36 - nA}/36` },
      ] : [
        { label: "|\\Omega|", tex: true, value: 36 },
        { label: "|A|", tex: true, value: nA },
        { label: "P(A)", tex: true, value: `${nA}/36 = ${fmt(nA / 36, 4)}` },
      ]} />}
      calc={<CalcSteps steps={mode === "axioms" ? [
        { tex: String.raw`P(A\cup B)=P(A)+P(B)-P(A\cap B)=\tfrac{${nA}}{36}+\tfrac{${nB}}{36}-\tfrac{${nAB}}{36}=\tfrac{${nU}}{36}`, note: "inclusion–exclusion (follows from the additivity axiom)" },
        { tex: String.raw`P(A^c)=1-P(A)=\tfrac{${36 - nA}}{36}`, note: "complement rule (from P(Ω) = 1)" },
      ] : [{ tex: String.raw`P(A)=\frac{|A|}{|\Omega|}=\frac{${nA}}{36}=${tex(nA / 36, 4)}`, note: "valid because all outcomes are equally likely" }]} />}
    />
  );
}

/* ------------------------------ conditional independence ------------------------------ */
/** Rain R causes both Umbrella U and Wet ground W. U and W are dependent, but independent given R. */
export function CondIndepViz() {
  const [pr, setPr] = useState(0.3);
  const [pu1, setPu1] = useState(0.9);
  const [pu0, setPu0] = useState(0.1);
  const [pw1, setPw1] = useState(0.95);
  const [pw0, setPw0] = useState(0.2);
  const pU = pr * pu1 + (1 - pr) * pu0;
  const pW = pr * pw1 + (1 - pr) * pw0;
  const pUW = pr * pu1 * pw1 + (1 - pr) * pu0 * pw0;
  const pW_U = pUW / pU;
  const rows = [
    { r: 1, u: 1, w: 1 }, { r: 1, u: 1, w: 0 }, { r: 1, u: 0, w: 1 }, { r: 1, u: 0, w: 0 },
    { r: 0, u: 1, w: 1 }, { r: 0, u: 1, w: 0 }, { r: 0, u: 0, w: 1 }, { r: 0, u: 0, w: 0 },
  ].map((x) => ({ ...x, p: (x.r ? pr : 1 - pr) * (x.u ? (x.r ? pu1 : pu0) : 1 - (x.r ? pu1 : pu0)) * (x.w ? (x.r ? pw1 : pw0) : 1 - (x.r ? pw1 : pw0)) }));
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <svg viewBox="0 0 320 120" className="w-full max-w-md" role="img" aria-label="rain causes umbrella and wet ground">
            {[["Rain", 160, 25], ["Umbrella", 70, 95], ["Wet ground", 250, 95]].map(([l, x, y]) => (
              <g key={l as string}><rect x={(x as number) - 48} y={(y as number) - 15} width={96} height={30} rx={6} style={{ fill: C.panel, stroke: C.lineStrong }} /><text x={x as number} y={(y as number) + 4} textAnchor="middle" style={{ fill: C.fg, fontSize: 12 }}>{l}</text></g>
            ))}
            <line x1={140} y1={40} x2={90} y2={80} style={{ stroke: C.muted }} markerEnd="url(#ci)" />
            <line x1={180} y1={40} x2={230} y2={80} style={{ stroke: C.muted }} markerEnd="url(#ci)" />
            <defs><marker id="ci" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" style={{ fill: C.muted }} /></marker></defs>
          </svg>
          <table className="w-full max-w-md text-[12px]">
            <thead><tr className="text-faint"><th className="text-left">Rain</th><th className="text-left">Umbrella</th><th className="text-left">Wet</th><th className="text-right">P(r, u, w)</th></tr></thead>
            <tbody>{rows.map((x, i) => <tr key={i} className="border-t border-line/60"><td>{x.r}</td><td>{x.u}</td><td>{x.w}</td><td className="num text-right">{fmt(x.p, 4)}</td></tr>)}</tbody>
          </table>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="P(rain)" value={pr} min={0.05} max={0.95} step={0.01} onChange={setPr} />
          <ParameterSlider label="P(umbrella | rain)" value={pu1} min={0} max={1} step={0.01} onChange={setPu1} />
          <ParameterSlider label="P(umbrella | no rain)" value={pu0} min={0} max={1} step={0.01} onChange={setPu0} />
          <ParameterSlider label="P(wet | rain)" value={pw1} min={0} max={1} step={0.01} onChange={setPw1} />
          <ParameterSlider label="P(wet | no rain)" value={pw0} min={0} max={1} step={0.01} onChange={setPw0} />
        </>
      }
      readouts={<Readouts items={[
        { label: "P(W)", tex: true, value: fmt(pW, 4) },
        { label: "P(W\\mid U)", tex: true, value: fmt(pW_U, 4), color: Math.abs(pW_U - pW) > 1e-6 ? C.amber : C.green },
        { label: "P(W\\mid U,R)", tex: true, value: fmt(pw1, 4) },
        { label: "P(W\\mid R)", tex: true, value: fmt(pw1, 4), color: C.green },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`P(U,W)=${tex(pUW, 4)}\ ${Math.abs(pUW - pU * pW) < 1e-9 ? "=" : "\\neq"}\ P(U)P(W)=${tex(pU, 4)}\cdot${tex(pW, 4)}=${tex(pU * pW, 4)}`, note: "marginally dependent: seeing an umbrella makes wet ground more likely" },
        { tex: String.raw`P(U,W\mid R)=P(U\mid R)P(W\mid R)`, note: "given rain, the umbrella tells you nothing more about the ground (by construction)" },
      ]} />}
      note="Naive Bayes assumes features are conditionally independent given the class — like umbrella and wet ground given rain. Dependencies through the class are allowed; dependencies left over after conditioning are not."
    />
  );
}

/* ------------------------------ population vs sample ------------------------------ */
export function PopulationSampleViz() {
  const pop = useMemo(() => { const r = mulberry32(12); return Array.from({ length: 2000 }, () => 170 + 9 * randn(r)); }, []);
  const mu = mean(pop), sigma2 = variance(pop, false);
  const [n, setN] = useState(5);
  const [seed, setSeed] = useState(1);
  const [reps, setReps] = useState<{ m: number; vB: number; vU: number }[]>([]);
  const [playing, setPlaying] = useState(false);
  const draw = (s: number) => { const r = mulberry32(s); const x = Array.from({ length: n }, () => pop[Math.floor(r() * pop.length)]); return { x, m: mean(x), vB: variance(x, false), vU: variance(x, true) }; };
  const cur = useMemo(() => draw(seed), [seed, n]); // eslint-disable-line
  useTicker(playing, 40, () => setReps((v) => { if (v.length >= 2000) { setPlaying(false); return v; } const add = Array.from({ length: 20 }, (_, k) => { const d = draw(seed * 7919 + v.length + k); return { m: d.m, vB: d.vB, vU: d.vU }; }); return [...v, ...add]; }));
  const avgB = reps.length ? mean(reps.map((r) => r.vB)) : NaN, avgU = reps.length ? mean(reps.map((r) => r.vU)) : NaN;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[135, 205]} y={[0, 0.06]} aspect={0.45} maxH={320} ariaLabel="population histogram with a sample">
            <Bars bins={histogram(pop, 135, 205, 40)} color={C.a("fg", 0.12)} stroke={C.a("fg", 0.25)} />
            {cur.x.map((v, i) => <Dot key={i} p={[v, 0.004 + (i % 6) * 0.003]} r={4} color={C.pink} />)}
            <VLine x={mu} color={C.green} label="μ" />
            <VLine x={cur.m} color={C.pink} label="x̄" />
          </Plot>
          <Legend items={[{ label: "population of 2,000 heights (cm)", color: C.a("fg", 0.4) }, { label: `sample of n = ${n}`, color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="sample size n" value={n} min={2} max={100} step={1} onChange={(v) => { setN(v); setReps([]); }} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>Draw another sample</button>
          <div className="label pt-2">Repeat many times</div>
          <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onReset={() => { setPlaying(false); setReps([]); }} />
        </>
      }
      readouts={<Readouts items={[
        { label: "\\mu\\ \\text{(population)}", tex: true, value: fmt(mu, 2) },
        { label: "\\bar x\\ \\text{(this sample)}", tex: true, value: fmt(cur.m, 2) },
        { label: "\\sigma^2\\ \\text{(population)}", tex: true, value: fmt(sigma2, 2) },
        { label: "\\text{avg of }\\tfrac1n\\sum(x-\\bar x)^2", tex: true, value: reps.length ? fmt(avgB, 2) : "—", color: C.amber },
        { label: "\\text{avg of }\\tfrac1{n-1}\\sum(x-\\bar x)^2", tex: true, value: reps.length ? fmt(avgU, 2) : "—", color: C.green },
        { label: "\\text{repetitions}", tex: true, value: reps.length },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`E\Big[\tfrac1n\sum(x_i-\bar x)^2\Big]=\tfrac{n-1}{n}\sigma^2=${tex(((n - 1) / n) * sigma2, 2)}`, note: "biased low: the sample mean is fitted to the sample" },
        { tex: String.raw`E\Big[\tfrac1{n-1}\sum(x_i-\bar x)^2\Big]=\sigma^2=${tex(sigma2, 2)}`, note: "Bessel's correction makes it unbiased" },
      ]} />}
      note="A sample is a small random subset we can measure; the population is what we want to know about. Statistics computed on samples vary from sample to sample — run the repetition to see the n−1 correction land on σ² on average."
    />
  );
}

/* ------------------------------ chi-square test of independence ------------------------------ */
export function ChiSquareViz() {
  const [T, setT] = useState([[30, 20, 10], [20, 25, 25]]);
  const rowS = T.map((r) => r.reduce((a, b) => a + b, 0));
  const colS = T[0].map((_, j) => T[0][j] + T[1][j]);
  const N = rowS[0] + rowS[1];
  const E = T.map((r, i) => r.map((_, j) => (rowS[i] * colS[j]) / N));
  const chi = T.reduce((s, r, i) => s + r.reduce((t, o, j) => t + (o - E[i][j]) ** 2 / E[i][j], 0), 0);
  const df = (2 - 1) * (3 - 1);
  const D = distributionById("chi-square")!;
  const p = 1 - D.cdf(chi, { k: df });
  const cols = ["Email", "Social", "Search"];
  const rowsL = ["Bought", "Did not buy"];
  return (
    <LabShell
      viz={
        <div className="grid gap-4 md:grid-cols-2">
          <table className="text-[12.5px]">
            <thead><tr><th /><th colSpan={3} className="pb-1 text-faint">observed (expected)</th></tr><tr className="text-faint"><th /><>{cols.map((c) => <th key={c} className="px-2 text-right font-medium">{c}</th>)}</></tr></thead>
            <tbody>
              {T.map((r, i) => (
                <tr key={i} className="border-t border-line/60">
                  <td className="pr-2 text-faint">{rowsL[i]}</td>
                  {r.map((o, j) => (
                    <td key={j} className="px-1 py-1 text-right">
                      <input type="number" aria-label={`${rowsL[i]} via ${cols[j]}`} value={o} min={0} onChange={(e) => { const v = Math.max(0, Math.round(Number(e.target.value) || 0)); setT(T.map((rr, a) => rr.map((x, b) => (a === i && b === j ? v : x)))); }} className="num h-7 w-14 rounded border border-line bg-raised px-1 text-right" />
                      <div className="num text-[10.5px] text-faint">({fmt(E[i][j], 1)})</div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <Plot x={[0, Math.max(12, chi * 1.2)]} y={[0, 0.5]} aspect={0.6} maxH={240} ariaLabel="chi-square distribution with observed statistic">
            <AreaUnder f={(x) => D.density(x, { k: df })} a={chi} b={Math.max(12, chi * 1.2)} color={C.a("pink", 0.45)} />
            <FnPath f={(x) => D.density(x, { k: df })} />
            <VLine x={chi} color={C.pink} label={`χ² = ${fmt(chi, 2)}`} />
          </Plot>
        </div>
      }
      controls={<p className="text-[12.5px] leading-relaxed text-muted">Edit the counts of customers who bought, by marketing channel. If channel and buying were independent, each cell would be close to its expected count (row total × column total / N).</p>}
      readouts={<Readouts items={[{ label: "\\chi^2", tex: true, value: fmt(chi, 3) }, { label: "\\text{df}", tex: true, value: df }, { label: "p", tex: true, value: fmt(p, 4), color: p < 0.05 ? C.pink : undefined }, { label: "N", tex: true, value: N }]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`E_{ij}=\frac{R_iC_j}{N},\quad E_{11}=\frac{${rowS[0]}\cdot${colS[0]}}{${N}}=${tex(E[0][0], 2)}` },
        { tex: String.raw`\chi^2=\sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}=${tex(chi, 3)},\quad\text{df}=(r-1)(c-1)=${df},\quad p=${tex(p, 4)}` },
      ]} />}
      note="Rule of thumb: expected counts should be at least 5 for the chi-square approximation. A small p says the variables are associated, not which cells drive it or why."
    />
  );
}

/* ------------------------------ one-way ANOVA ------------------------------ */
export function AnovaViz() {
  const [m, setM] = useState([70, 75, 72]);
  const [sd, setSd] = useState(8);
  const [n, setN] = useState(15);
  const groups = useMemo(() => m.map((mu, g) => { const r = mulberry32(100 + g); return Array.from({ length: n }, () => mu + sd * randn(r)); }), [m, sd, n]);
  const all = groups.flat();
  const gm = mean(all);
  const ssb = groups.reduce((s, x) => s + x.length * (mean(x) - gm) ** 2, 0);
  const ssw = groups.reduce((s, x) => { const mx = mean(x); return s + x.reduce((t, v) => t + (v - mx) ** 2, 0); }, 0);
  const dfb = 2, dfw = all.length - 3;
  const F = ssb / dfb / (ssw / dfw);
  const p = 1 - distributionById("f")!.cdf(F, { d1: dfb, d2: dfw });
  const names = ["Method A", "Method B", "Method C"];
  const cols = [C.blue, C.pink, C.green];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, 4]} y={[30, 110]} aspect={0.55} maxH={360} grid={false} ariaLabel="three groups of exam scores">
            {groups.map((x, g) => x.map((v, i) => <Dot key={`${g}-${i}`} p={[g + 0.75 + ((i * 37) % 11) / 22, v]} r={3} color={cols[g]} opacity={0.75} />))}
            {groups.map((x, g) => <Polyline key={g} pts={[[g + 0.65, mean(x)], [g + 1.35, mean(x)]]} style={{ stroke: cols[g], strokeWidth: 3 }} />)}
            <Polyline pts={[[0.4, gm], [3.6, gm]]} style={{ stroke: C.a("fg", 0.5), strokeDasharray: "5 3" }} />
          </Plot>
          <Legend items={[...names.map((nm, g) => ({ label: `${nm} (mean ${fmt(mean(groups[g]), 1)})`, color: cols[g] })), { label: "grand mean", color: C.a("fg", 0.5), dashed: true }]} />
        </div>
      }
      controls={
        <>
          {m.map((v, g) => <ParameterSlider key={g} label={`${names[g]} true mean`} value={v} min={50} max={90} step={0.5} onChange={(x) => setM(m.map((y, k) => (k === g ? x : y)))} />)}
          <ParameterSlider label="within-group SD" value={sd} min={2} max={20} step={0.5} onChange={setSd} />
          <ParameterSlider label="students per group" value={n} min={3} max={60} step={1} onChange={setN} />
        </>
      }
      readouts={<Readouts items={[{ label: "SS_B", tex: true, value: fmt(ssb, 1) }, { label: "SS_W", tex: true, value: fmt(ssw, 1) }, { label: "F", tex: true, value: fmt(F, 3) }, { label: "p", tex: true, value: fmt(p, 4), color: p < 0.05 ? C.pink : undefined }]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`F=\frac{SS_B/(k-1)}{SS_W/(N-k)}=\frac{${tex(ssb, 1)}/${dfb}}{${tex(ssw, 1)}/${dfw}}=${tex(F, 3)}` },
        { tex: String.raw`p=P(F_{${dfb},${dfw}}>${tex(F, 3)})=${tex(p, 4)}`, note: "H₀: all group means are equal" },
      ]} />}
      note="ANOVA compares the spread between group means with the spread inside groups. A significant F says some means differ — follow up with pairwise tests (with a multiple-testing correction) to find which."
    />
  );
}

/* ------------------------------ multiple testing ------------------------------ */
export function MultipleTestingViz() {
  const [m, setM] = useState(100);
  const [trueFrac, setTrueFrac] = useState(0.1);
  const [effect, setEffect] = useState(3);
  const [alpha, setAlpha] = useState(0.05);
  const [method, setMethod] = useState<"none" | "bonf" | "bh">("bh");
  const [seed, setSeed] = useState(4);
  const tests = useMemo(() => {
    const r = mulberry32(seed);
    const N = distributionById("normal")!;
    return Array.from({ length: m }, (_, i) => {
      const real = i < Math.round(m * trueFrac);
      const z = (real ? effect : 0) + randn(r);
      return { real, p: 1 - N.cdf(z, { mu: 0, sigma: 1 }) };
    }).sort((a, b) => a.p - b.p);
  }, [m, trueFrac, effect, seed]);
  let k = 0;
  if (method === "none") k = tests.filter((t) => t.p < alpha).length;
  else if (method === "bonf") k = tests.filter((t) => t.p < alpha / m).length;
  else tests.forEach((t, i) => { if (t.p <= ((i + 1) / m) * alpha) k = i + 1; });
  const rejected = tests.slice(0, k);
  const fd = rejected.filter((t) => !t.real).length, td = rejected.length - fd;
  const nReal = tests.filter((t) => t.real).length;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, m + 1]} y={[0, 1]} aspect={0.5} maxH={340} xLabel="rank i" yLabel="p-value" ariaLabel="sorted p-values with correction thresholds">
            {tests.map((t, i) => <Dot key={i} p={[i + 1, t.p]} r={2.6} color={t.real ? C.green : C.muted} stroke={i < k ? C.pink : undefined} />)}
            <Polyline pts={[[0, alpha], [m + 1, alpha]]} style={{ stroke: C.amber, strokeDasharray: "4 3" }} />
            <Polyline pts={[[0, alpha / m], [m + 1, alpha / m]]} style={{ stroke: C.red, strokeDasharray: "2 3" }} />
            <Polyline pts={[[0, 0], [m + 1, ((m + 1) / m) * alpha]]} style={{ stroke: C.blue }} />
          </Plot>
          <Legend items={[{ label: "real effect", color: C.green }, { label: "null (no effect)", color: C.muted }, { label: "rejected (ringed)", color: C.pink }, { label: "α (no correction)", color: C.amber, dashed: true }, { label: "Bonferroni α/m", color: C.red, dashed: true }, { label: "Benjamini–Hochberg line iα/m", color: C.blue }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Correction" value={method} onChange={setMethod} options={[{ value: "none", label: "None" }, { value: "bonf", label: "Bonferroni" }, { value: "bh", label: "BH (FDR)" }]} />
          <ParameterSlider label="number of tests m" value={m} min={10} max={300} step={10} onChange={setM} />
          <ParameterSlider label="fraction with a real effect" value={trueFrac} min={0} max={0.5} step={0.01} onChange={setTrueFrac} />
          <ParameterSlider label="effect size (z units)" value={effect} min={0} max={5} step={0.1} onChange={setEffect} />
          <ParameterSlider label="α" tex value={alpha} min={0.01} max={0.2} step={0.01} onChange={setAlpha} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New experiment</button>
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{rejected}", tex: true, value: k }, { label: "\\text{true discoveries}", tex: true, value: `${td} / ${nReal}`, color: C.green }, { label: "\\text{false discoveries}", tex: true, value: fd, color: fd ? C.red : undefined }, { label: "\\text{FDP}", tex: true, value: k ? pct(fd / k) : "—" }]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`P(\text{at least one false positive})=1-(1-\alpha)^{m_0}`, note: `with ${m - nReal} null tests at α = ${alpha}: ${pct(1 - (1 - alpha) ** (m - nReal))}` },
        { tex: method === "bonf" ? String.raw`\text{reject if }p_i<\alpha/m=${tex(alpha / m, 5)}` : method === "bh" ? String.raw`k=\max\{i:p_{(i)}\le\tfrac{i}{m}\alpha\},\ \text{reject }p_{(1)},\dots,p_{(k)}` : String.raw`\text{reject if }p_i<\alpha`, note: method === "bonf" ? "controls the family-wise error rate" : method === "bh" ? "controls the expected false discovery rate" : "no control" },
      ]} />}
      note="Testing many hypotheses (features, A/B variants, metrics) guarantees some 'significant' results by chance. Correct for it, or treat findings as exploratory."
    />
  );
}

/* ------------------------------ permutation test ------------------------------ */
export function PermutationViz() {
  const [shift, setShift] = useState(4);
  const data = useMemo(() => { const r = mulberry32(31); return { a: Array.from({ length: 12 }, () => 30 + 6 * randn(r)), b: Array.from({ length: 12 }, () => 30 + shift + 6 * randn(r)) }; }, [shift]);
  const obs = mean(data.b) - mean(data.a);
  const [perms, setPerms] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);
  const pool = [...data.a, ...data.b];
  const rng = useMemo(() => mulberry32(77), [shift]); // eslint-disable-line
  const one = () => { const s = shuffle(pool, rng); return mean(s.slice(12)) - mean(s.slice(0, 12)); };
  useTicker(playing, 30, () => setPerms((p) => { if (p.length >= 5000) { setPlaying(false); return p; } return [...p, ...Array.from({ length: 50 }, one)]; }));
  const ext = perms.filter((d) => Math.abs(d) >= Math.abs(obs)).length;
  const p = perms.length ? (ext + 1) / (perms.length + 1) : NaN;
  const bins = histogram(perms, -12, 12, 48);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-12, 12]} y={[0, Math.max(0.1, ...bins.map((b) => b.density)) * 1.15]} aspect={0.45} maxH={320} xLabel="difference in means under shuffled labels" ariaLabel="permutation null distribution">
            <Bars bins={bins} />
            <VLine x={obs} color={C.pink} label="observed" />
            <VLine x={-obs} color={C.a("pink", 0.5)} />
          </Plot>
          <Legend items={[{ label: "differences after randomly reassigning group labels", color: C.a("blue", 0.6) }, { label: `observed difference ${fmt(obs, 2)}`, color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="true difference between groups" value={shift} min={0} max={10} step={0.5} onChange={(v) => { setShift(v); setPerms([]); }} />
          <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onStep={() => setPerms((x) => [...x, ...Array.from({ length: 200 }, one)])} onReset={() => { setPlaying(false); setPerms([]); }} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\bar x_B-\\bar x_A", tex: true, value: fmt(obs, 3) }, { label: "\\text{permutations}", tex: true, value: perms.length }, { label: "p\\ \\text{(two-sided)}", tex: true, value: perms.length ? fmt(p, 4) : "—", color: p < 0.05 ? C.pink : undefined }]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`p=\frac{1+\#\{|d^*|\ge|d_{\text{obs}}|\}}{1+B}=\frac{1+${ext}}{1+${perms.length}}`, note: "+1 counts the observed labelling itself" }]} />}
      note="If the group labels don't matter (H₀), every reshuffle is as plausible as the real one. The test needs no normality assumption — only exchangeability."
    />
  );
}

/* ------------------------------ partial correlation ------------------------------ */
export function PartialCorrViz() {
  const [conf, setConf] = useState(0.8);
  const [direct, setDirect] = useState(0);
  const d = useMemo(() => {
    const r = mulberry32(8);
    const z = Array.from({ length: 200 }, () => randn(r)); // e.g. temperature
    const x = z.map((v) => conf * v + Math.sqrt(1 - conf * conf) * randn(r)); // ice-cream sales
    const y = z.map((v, i) => conf * v + direct * x[i] + Math.sqrt(Math.max(0.05, 1 - conf * conf)) * randn(r)); // sunburns
    const resid = (a: number[]) => { const ma = mean(a), mz = mean(z); const b = a.reduce((s, v, i) => s + (v - ma) * (z[i] - mz), 0) / z.reduce((s, v) => s + (v - mz) ** 2, 0); return a.map((v, i) => v - ma - b * (z[i] - mz)); };
    return { x, y, z, rx: resid(x), ry: resid(y) };
  }, [conf, direct]);
  const rxy = pearson(d.x, d.y), rp = pearson(d.rx, d.ry);
  return (
    <LabShell
      viz={
        <div className="grid gap-3 md:grid-cols-2">
          <div><div className="mb-1 text-[11.5px] text-faint">Raw: ice-cream sales vs sunburns</div>
            <Plot x={[-3.5, 3.5]} y={[-3.5, 3.5]} equal aspect={1} maxH={280} ariaLabel="raw scatter">{d.x.map((v, i) => <Dot key={i} p={[v, d.y[i]]} r={2.4} color={C.a("blue", 0.7)} />)}</Plot></div>
          <div><div className="mb-1 text-[11.5px] text-faint">After removing temperature from both (residuals)</div>
            <Plot x={[-3.5, 3.5]} y={[-3.5, 3.5]} equal aspect={1} maxH={280} ariaLabel="residual scatter">{d.rx.map((v, i) => <Dot key={i} p={[v, d.ry[i]]} r={2.4} color={C.a("pink", 0.7)} />)}</Plot></div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="strength of the confounder (temperature)" value={conf} min={0} max={0.95} step={0.01} onChange={setConf} />
          <ParameterSlider label="direct effect of X on Y" value={direct} min={-0.8} max={0.8} step={0.01} onChange={setDirect} />
        </>
      }
      readouts={<Readouts items={[{ label: "r_{XY}", tex: true, value: fmt(rxy, 3), color: C.blue }, { label: "r_{XY\\cdot Z}", tex: true, value: fmt(rp, 3), color: C.pink }]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`r_{XY\cdot Z}=\frac{r_{XY}-r_{XZ}r_{YZ}}{\sqrt{(1-r_{XZ}^2)(1-r_{YZ}^2)}}=${tex(rp, 3)}`, note: "equivalently: correlate the residuals of X and Y after regressing each on Z" }]} />}
      note="Ice-cream sales and sunburns correlate only because hot weather drives both. Partial correlation removes the confounder; with no direct effect it drops to about 0. (Synthetic data.)"
    />
  );
}

/* ------------------------------ multicollinearity ------------------------------ */
export function MulticollinearityViz() {
  const [rho, setRho] = useState(0.95);
  const [n, setN] = useState(40);
  const fits = useMemo(() => Array.from({ length: 60 }, (_, s) => {
    const r = mulberry32(500 + s);
    const X: number[][] = [], y: number[] = [];
    for (let i = 0; i < n; i++) { const [a, b] = randBivariate(r, 0, 0, 1, 1, rho); X.push([1, a, b]); y.push(1 + 2 * a + 1 * b + randn(r)); }
    const Xt = [0, 1, 2].map((j) => X.map((row) => row[j]));
    const XtX = Xt.map((rj) => Xt.map((rk) => rj.reduce((s2, v, i) => s2 + v * rk[i], 0)));
    const Xty = Xt.map((rj) => rj.reduce((s2, v, i) => s2 + v * y[i], 0));
    const w = solve(XtX, Xty) ?? [0, 0, 0];
    return { b1: w[1], b2: w[2] };
  }), [rho, n]);
  const vif = 1 / (1 - rho * rho);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 7]} y={[-4, 6]} equal aspect={0.8} maxH={380} xLabel="β₁ estimate" yLabel="β₂ estimate" ariaLabel="coefficient estimates across resamples">
            {fits.map((f, i) => <Dot key={i} p={[f.b1, f.b2]} r={3} color={C.a("blue", 0.75)} />)}
            <Dot p={[2, 1]} r={6} color={C.green} stroke={C.bg} />
            <Polyline pts={[[-3, 6], [7, -4]]} style={{ stroke: C.a("fg", 0.2), strokeDasharray: "4 3" }} />
          </Plot>
          <Legend items={[{ label: "OLS estimates from 60 independent samples", color: C.blue }, { label: "true (β₁, β₂) = (2, 1)", color: C.green }, { label: "β₁ + β₂ = 3 (what the data pins down)", color: C.a("fg", 0.3), dashed: true }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="correlation between x₁ and x₂" value={rho} min={0} max={0.995} step={0.005} onChange={setRho} />
          <ParameterSlider label="sample size n" value={n} min={10} max={300} step={5} onChange={setN} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{VIF}", tex: true, value: fmt(vif, 2), color: vif > 10 ? C.red : vif > 5 ? C.amber : C.green }, { label: "\\operatorname{sd}(\\hat\\beta_1)", tex: true, value: fmt(std(fits.map((f) => f.b1)), 3) }, { label: "\\operatorname{sd}(\\hat\\beta_1+\\hat\\beta_2)", tex: true, value: fmt(std(fits.map((f) => f.b1 + f.b2)), 3) }]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{VIF}_j=\frac{1}{1-R_j^2}=\frac{1}{1-${tex(rho * rho, 4)}}=${tex(vif, 2)}`, note: "how much the variance of β̂ⱼ is inflated by correlation with other features" }]} />}
      note="With highly correlated features the individual coefficients swing wildly from sample to sample (the cloud stretches along β₁ + β₂ = 3), yet their sum — and the predictions — stay stable. Ridge regression or dropping a feature fixes it."
    />
  );
}

