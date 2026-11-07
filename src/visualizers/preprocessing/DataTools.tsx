"use client";
import { useMemo, useState } from "react";
import { Plot, C, Dot, Polyline, Arrow, usePlot } from "@/components/visualization/Plot";
import { VLine } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented, Toggle } from "@/components/controls";
import { iqrBounds, mean, std, median, mad, quantile } from "@/lib/statistics/descriptive";
import { sample1D } from "@/lib/statistics/datasets";
import { mulberry32, randBivariate } from "@/lib/math/random";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, tex, cn } from "@/lib/utilities/format";

type Mode = "outliers" | "imputation" | "encoding" | "cyclical" | "winsorize";

export default function DataTools({ mode = "outliers", method }: { mode?: Mode; method?: string }) {
  if (mode === "imputation") return <Imputation initial={method} />;
  if (mode === "encoding") return <Encoding initial={method} />;
  if (mode === "cyclical") return <Cyclical />;
  return <Outliers initialWinsor={mode === "winsorize"} />;
}

/* ------------------------------ outliers ------------------------------ */
function BoxPlot({ b, y }: { b: ReturnType<typeof iqrBounds>; y: number }) {
  const { sx, sy } = usePlot();
  const h = 14;
  const wl = Math.max(b.min, b.lower), wu = Math.min(b.max, b.upper);
  return (
    <g style={{ stroke: C.fg }}>
      <line x1={sx(wl)} x2={sx(b.q1)} y1={sy(y)} y2={sy(y)} />
      <line x1={sx(b.q3)} x2={sx(wu)} y1={sy(y)} y2={sy(y)} />
      <line x1={sx(wl)} x2={sx(wl)} y1={sy(y) - h / 2} y2={sy(y) + h / 2} />
      <line x1={sx(wu)} x2={sx(wu)} y1={sy(y) - h / 2} y2={sy(y) + h / 2} />
      <rect x={sx(b.q1)} width={sx(b.q3) - sx(b.q1)} y={sy(y) - h} height={2 * h} style={{ fill: C.a("blue", 0.15), stroke: C.blue }} />
      <line x1={sx(b.median)} x2={sx(b.median)} y1={sy(y) - h} y2={sy(y) + h} style={{ stroke: C.pink, strokeWidth: 2 }} />
    </g>
  );
}

function Outliers({ initialWinsor }: { initialWinsor: boolean }) {
  const [method, setMethod] = useState<"iqr" | "z" | "modz">("iqr");
  const [k, setK] = useState(1.5);
  const [zt, setZt] = useState(3);
  const [treat, setTreat] = useState<"none" | "clip" | "trim" | "winsor">(initialWinsor ? "winsor" : "none");
  const [pctl, setPctl] = useState(0.05);
  const x = useMemo(() => sample1D("outliers", 160, 12), []);
  const b = iqrBounds(x, k);
  const m = mean(x), s = std(x), med = median(x), md = mad(x);
  const flag = x.map((v) => (method === "iqr" ? v < b.lower || v > b.upper : method === "z" ? Math.abs((v - m) / s) > zt : Math.abs((0.6745 * (v - med)) / (md || 1e-12)) > zt));
  const lo = method === "iqr" ? b.lower : method === "z" ? m - zt * s : med - (zt * md) / 0.6745;
  const hi = method === "iqr" ? b.upper : method === "z" ? m + zt * s : med + (zt * md) / 0.6745;
  const wlo = quantile(x, pctl), whi = quantile(x, 1 - pctl);
  const treated = treat === "clip" ? x.map((v) => Math.min(hi, Math.max(lo, v))) : treat === "trim" ? x.filter((_, i) => !flag[i]) : treat === "winsor" ? x.map((v) => Math.min(whi, Math.max(wlo, v))) : x;
  const jitter = (i: number) => ((i * 7919) % 100) / 100;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[5, 85]} y={[0, 3]} aspect={0.42} maxH={300} grid={false} ariaLabel="strip plot with outlier thresholds">
            <BoxPlot b={b} y={2.5} />
            {x.map((v, i) => <Dot key={i} p={[v, 1.3 + jitter(i) * 0.6]} r={flag[i] ? 4.5 : 2.6} color={flag[i] ? C.red : C.a("blue", 0.7)} />)}
            {treat !== "none" && treated.map((v, i) => <Dot key={`t${i}`} p={[v, 0.35 + jitter(i) * 0.4]} r={2.4} color={C.a("green", 0.75)} />)}
            <VLine x={lo} color={C.amber} label="lower" />
            <VLine x={hi} color={C.amber} label="upper" />
          </Plot>
          <Legend items={[{ label: "box plot (Q1–Q3, median)", color: C.blue }, { label: `flagged (${flag.filter(Boolean).length})`, color: C.red }, ...(treat !== "none" ? [{ label: `after ${treat === "winsor" ? "winsorising" : treat === "clip" ? "clipping to thresholds" : "trimming"}`, color: C.green }] : [])]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Detection" value={method} onChange={setMethod} options={[{ value: "iqr", label: "IQR" }, { value: "z", label: "Z-score" }, { value: "modz", label: "Modified Z" }]} />
          {method === "iqr" ? <ParameterSlider label="k (fences at Q1 − k·IQR, Q3 + k·IQR)" value={k} min={0.5} max={3} step={0.1} onChange={setK} /> : <ParameterSlider label="threshold" value={zt} min={1.5} max={5} step={0.1} onChange={setZt} />}
          <Select label="Treatment" value={treat} onChange={setTreat} options={[{ value: "none", label: "None (detect only)" }, { value: "clip", label: "Clip to thresholds" }, { value: "trim", label: "Trim (remove)" }, { value: "winsor", label: "Winsorise (percentiles)" }]} />
          {treat === "winsor" && <ParameterSlider label="winsorise tails (each side)" value={pctl} min={0.01} max={0.2} step={0.01} onChange={setPctl} />}
        </>
      }
      readouts={<Readouts items={[
        { label: "Q_1,\\ Q_3", tex: true, value: `${fmt(b.q1)}, ${fmt(b.q3)}` },
        { label: "\\text{IQR}", tex: true, value: fmt(b.iqr) },
        { label: "\\bar x,\\ s", tex: true, value: `${fmt(m)}, ${fmt(s)}` },
        { label: "\\text{median},\\ \\text{MAD}", tex: true, value: `${fmt(med)}, ${fmt(md)}` },
        { label: "\\text{mean after}", tex: true, value: fmt(mean(treated)) },
      ]} />}
      calc={<CalcSteps steps={method === "iqr" ? [{ tex: `[Q_1-${tex(k, 1)}\\,\\text{IQR},\\ Q_3+${tex(k, 1)}\\,\\text{IQR}]=[${tex(b.lower)},\\ ${tex(b.upper)}]` }] : method === "z" ? [{ tex: `|z_i|=\\left|\\tfrac{x_i-\\bar x}{s}\\right|>${tex(zt, 1)}`, note: "mean and s are themselves inflated by the outliers (masking)" }] : [{ tex: `M_i=\\dfrac{0.6745\\,(x_i-\\tilde x)}{\\text{MAD}}>${tex(zt, 1)}`, note: "robust: median and MAD barely move" }]} />}
      note="Outliers can be errors or the most interesting points. Decide with domain knowledge; tree models are robust to them, linear models and distance-based methods are not."
    />
  );
}

/* ------------------------------ missing data imputation ------------------------------ */
function Imputation({ initial }: { initial?: string }) {
  const [mech, setMech] = useState<"mcar" | "mar" | "mnar">("mcar");
  const [method, setMethod] = useState<"mean" | "median" | "constant" | "knn" | "regression">((initial as "mean") ?? "mean");
  const [rate, setRate] = useState(0.25);
  const [k, setK] = useState(5);
  const [indicator, setIndicator] = useState(false);
  const data = useMemo(() => {
    const r = mulberry32(21);
    return Array.from({ length: 120 }, () => { const [a, b2] = randBivariate(r, 0, 0, 1, 1, 0.8); return [a * 10 + 50, Math.exp(0.25 * b2) * 30] as [number, number]; });
  }, []);
  const missing = useMemo(() => {
    const r = mulberry32(5);
    return data.map(([x, y]) => {
      const u = r();
      if (mech === "mcar") return u < rate;
      if (mech === "mar") return u < rate * (x > 50 ? 1.6 : 0.4); // depends on observed x
      return u < rate * (y > 32 ? 1.7 : 0.3); // depends on the missing value itself
    });
  }, [data, mech, rate]);
  const obs = data.filter((_, i) => !missing[i]);
  const ys = obs.map((p) => p[1]);
  const imputeVal = (x: number): number => {
    if (method === "mean") return mean(ys);
    if (method === "median") return median(ys);
    if (method === "constant") return 0;
    if (method === "knn") { const nb = obs.map((p) => ({ d: Math.abs(p[0] - x), y: p[1] })).sort((a, b) => a.d - b.d).slice(0, k); return mean(nb.map((n) => n.y)); }
    const mx = mean(obs.map((p) => p[0])), my = mean(ys);
    let sxy = 0, sxx = 0; for (const [a, b2] of obs) { sxy += (a - mx) * (b2 - my); sxx += (a - mx) ** 2; }
    return my + (sxy / sxx) * (x - mx);
  };
  const imputed = data.map(([x, y], i) => (missing[i] ? [x, imputeVal(x)] as [number, number] : [x, y] as [number, number]));
  const trueMissingMean = mean(data.filter((_, i) => missing[i]).map((p) => p[1]));
  const rmse = Math.sqrt(mean(data.map((p, i) => (missing[i] ? (p[1] - imputed[i][1]) ** 2 : NaN)).filter(Number.isFinite)));
  const yAll = data.map((p) => p[1]);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[20, 80]} y={[method === "constant" ? -5 : 10, 70]} aspect={0.6} maxH={420} xLabel="x (always observed)" yLabel="y (may be missing)" ariaLabel="scatter plot with imputed values">
            {data.map((p, i) => missing[i] ? <Polyline key={`l${i}`} pts={[p, imputed[i]]} style={{ stroke: C.a("fg", 0.15), strokeWidth: 1 }} /> : null)}
            {data.map((p, i) => missing[i] ? <Dot key={`h${i}`} p={p} r={3} color="transparent" stroke={C.a("fg", 0.35)} /> : <Dot key={i} p={p} r={3} color={C.a("blue", 0.8)} />)}
            {imputed.map((p, i) => missing[i] ? <Dot key={`i${i}`} p={p} r={3.6} color={C.pink} /> : null)}
          </Plot>
          <Legend items={[{ label: "observed", color: C.blue }, { label: "imputed", color: C.pink }, { label: "hidden true value (unknown in practice)", color: C.a("fg", 0.4) }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Missingness mechanism" value={mech} onChange={setMech} options={[{ value: "mcar", label: "MCAR" }, { value: "mar", label: "MAR" }, { value: "mnar", label: "MNAR" }]} />
          <ParameterSlider label="missing rate" value={rate} min={0.05} max={0.6} step={0.01} onChange={setRate} />
          <Select label="Imputation" value={method} onChange={setMethod} options={[{ value: "mean", label: "Mean" }, { value: "median", label: "Median" }, { value: "constant", label: "Constant (0)" }, { value: "knn", label: "KNN (on x)" }, { value: "regression", label: "Regression on x (iterative, 1 step)" }]} />
          {method === "knn" && <ParameterSlider label="k neighbours" value={k} min={1} max={25} step={1} onChange={setK} />}
          <Toggle label="Add missing-indicator feature" checked={indicator} onChange={setIndicator} />
          <p className="text-[12px] leading-relaxed text-faint">{mech === "mcar" ? "MCAR: missingness is pure chance." : mech === "mar" ? "MAR: missingness depends on observed x (higher x → more missing y)." : "MNAR: missingness depends on the unobserved y itself (large y more often missing) — no method can fully correct this from the data alone."}</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{missing}", tex: true, value: `${missing.filter(Boolean).length} / ${data.length}` },
        { label: "\\text{imputation RMSE}", tex: true, value: fmt(rmse) },
        { label: "\\text{true mean of missing }y", tex: true, value: fmt(trueMissingMean) },
        { label: "\\text{SD of }y:\\ \\text{true}\\to\\text{imputed}", tex: true, value: `${fmt(std(yAll))} → ${fmt(std(imputed.map((p) => p[1])))}` },
      ]} />}
      calc={<CalcSteps steps={[{ tex: method === "mean" ? `\\hat y=\\bar y_{\\text{obs}}=${tex(mean(ys))}` : method === "median" ? `\\hat y=\\operatorname{median}(y_{\\text{obs}})=${tex(median(ys))}` : method === "knn" ? `\\hat y_i=\\tfrac1k\\sum_{j\\in N_k(x_i)}y_j` : method === "regression" ? "\\hat y_i=\\hat\\beta_0+\\hat\\beta_1x_i\\ \\text{(fit on complete rows)}" : "\\hat y=0", note: method === "mean" || method === "median" || method === "constant" ? "shrinks variance and breaks the x–y relationship" : "uses the correlation with x" }, ...(indicator ? [{ tex: "m_i=\\mathbb 1[y_i\\text{ missing}]", note: "lets the model learn whether missingness itself is informative" }] : [])]} />}
    />
  );
}

/* ------------------------------ categorical encoding ------------------------------ */
const CATS = ["red", "green", "blue", "green", "red", "red", "yellow", "blue", "green", "red"];
const TARGET = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1];
const ORDER = ["small", "medium", "large"];

function Encoding({ initial }: { initial?: string }) {
  const [method, setMethod] = useState<"label" | "ordinal" | "onehot" | "binary" | "frequency" | "count" | "target" | "loo" | "hash">((initial as "onehot") ?? "onehot");
  const [smooth, setSmooth] = useState(2);
  const [buckets, setBuckets] = useState(4);
  const levels = [...new Set(CATS)].sort();
  const counts = Object.fromEntries(levels.map((l) => [l, CATS.filter((c) => c === l).length]));
  const prior = mean(TARGET);
  const hash = (s: string) => { let h = 2166136261; for (const ch of s) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); } return (h >>> 0) % buckets; };
  let cols: string[] = [];
  let rows: (number | string)[][] = [];
  const nb = Math.max(1, Math.ceil(Math.log2(levels.length + 1)));
  switch (method) {
    case "label": cols = ["code"]; rows = CATS.map((c) => [levels.indexOf(c)]); break;
    case "ordinal": cols = ["size", "code"]; rows = ["small", "large", "medium", "small", "large", "medium", "medium", "small", "large", "small"].map((s) => [s, ORDER.indexOf(s)]); break;
    case "onehot": cols = levels.map((l) => `is_${l}`); rows = CATS.map((c) => levels.map((l) => (l === c ? 1 : 0))); break;
    case "binary": cols = Array.from({ length: nb }, (_, i) => `bit${nb - 1 - i}`); rows = CATS.map((c) => (levels.indexOf(c) + 1).toString(2).padStart(nb, "0").split("").map(Number)); break;
    case "frequency": cols = ["freq"]; rows = CATS.map((c) => [counts[c] / CATS.length]); break;
    case "count": cols = ["count"]; rows = CATS.map((c) => [counts[c]]); break;
    case "target": cols = ["target_enc"]; rows = CATS.map((c) => { const ys = TARGET.filter((_, i) => CATS[i] === c); return [(ys.reduce((a, b) => a + b, 0) + smooth * prior) / (ys.length + smooth)]; }); break;
    case "loo": cols = ["loo_enc"]; rows = CATS.map((c, i) => { const ys = TARGET.filter((_, j) => CATS[j] === c && j !== i); return [ys.length ? mean(ys) : prior]; }); break;
    case "hash": cols = Array.from({ length: buckets }, (_, i) => `h${i}`); rows = CATS.map((c) => Array.from({ length: buckets }, (_, i) => (i === hash(c) ? 1 : 0))); break;
  }
  const info: Record<typeof method, { tex: string; note: string }> = {
    label: { tex: "\\text{category}\\mapsto\\{0,\\dots,K-1\\}", note: "Implies a fake order (blue < green < red). Fine for trees, misleading for linear models." },
    ordinal: { tex: "\\text{small}\\mapsto0,\\ \\text{medium}\\mapsto1,\\ \\text{large}\\mapsto2", note: "Use only when categories have a genuine order." },
    onehot: { tex: "x_k=\\mathbb 1[\\text{category}=k]", note: "No false order; K columns (drop one with an intercept to avoid perfect collinearity). Explodes for high-cardinality features." },
    binary: { tex: "\\text{code}+1\\ \\text{in base 2}", note: `⌈log₂(K+1)⌉ = ${nb} columns instead of K = ${levels.length}.` },
    frequency: { tex: "x=\\text{count}(k)/n", note: "Categories with equal frequency collide." },
    count: { tex: "x=\\text{count}(k)", note: "Same as frequency but unnormalised." },
    target: { tex: `x_k=\\dfrac{\\sum_{i:c_i=k}y_i+m\\,\\bar y}{n_k+m},\\ m=${smooth}`, note: "Smoothing m shrinks rare categories toward the global mean. Must be fit out-of-fold to avoid target leakage." },
    loo: { tex: "x_i=\\text{mean of }y_j\\text{ over }j\\neq i\\text{ with }c_j=c_i", note: "Excludes the row's own target to reduce leakage." },
    hash: { tex: `\\text{bucket}=h(\\text{category})\\bmod ${buckets}`, note: "Fixed width, handles unseen categories, but distinct categories can collide." },
  };
  return (
    <LabShell
      wideControls
      viz={
        <div className="overflow-x-auto">
          <table className="w-full min-w-[420px] border-collapse text-[12.5px]">
            <thead>
              <tr className="border-b border-line text-left text-faint">
                <th className="py-1.5 pr-3 font-medium">colour</th>
                {(method === "target" || method === "loo") && <th className="py-1.5 pr-3 font-medium">y</th>}
                {cols.map((c) => <th key={c} className="num py-1.5 pr-3 font-medium text-pink">{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {CATS.map((c, i) => (
                <tr key={i} className="border-b border-line/60">
                  <td className="py-1.5 pr-3"><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: c === "yellow" ? "#e6c35c" : c === "red" ? "#f06868" : c === "green" ? "#6ecf97" : "#5aa9ff" }} />{method === "ordinal" ? "—" : c}</td>
                  {(method === "target" || method === "loo") && <td className="num py-1.5 pr-3">{TARGET[i]}</td>}
                  {rows[i].map((v, j) => <td key={j} className={cn("num py-1.5 pr-3", typeof v === "number" && v === 0 && cols.length > 1 ? "text-faint" : "")}>{typeof v === "number" ? fmt(v, 3) : v}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      }
      controls={
        <>
          <Select label="Encoding" value={method} onChange={setMethod} options={[{ value: "label", label: "Label" }, { value: "ordinal", label: "Ordinal" }, { value: "onehot", label: "One-hot" }, { value: "binary", label: "Binary" }, { value: "frequency", label: "Frequency" }, { value: "count", label: "Count" }, { value: "target", label: "Target (smoothed mean)" }, { value: "loo", label: "Leave-one-out" }, { value: "hash", label: "Hashing" }]} />
          {method === "target" && <ParameterSlider label="smoothing m" value={smooth} min={0} max={10} step={0.5} onChange={setSmooth} />}
          {method === "hash" && <ParameterSlider label="buckets" value={buckets} min={2} max={8} step={1} onChange={setBuckets} />}
          <div className="text-[13px]"><Tex tex={info[method].tex} /></div>
          <p className="text-[12.5px] leading-relaxed text-muted">{info[method].note}</p>
        </>
      }
      readouts={<Readouts items={[{ label: "K\\ \\text{(levels)}", tex: true, value: levels.length }, { label: "\\text{output columns}", tex: true, value: cols.filter((c) => c !== "size").length }, { label: "\\bar y\\ \\text{(global)}", tex: true, value: fmt(prior) }]} />}
    />
  );
}

/* ------------------------------ cyclical encoding ------------------------------ */
function Cyclical() {
  const [hour, setHour] = useState(23);
  const [period, setPeriod] = useState(24);
  const ang = (h: number) => (2 * Math.PI * h) / period;
  const pt = (h: number): [number, number] => [Math.cos(ang(h)), Math.sin(ang(h))];
  const a = pt(hour), b = pt(0);
  const dCyc = Math.hypot(a[0] - b[0], a[1] - b[1]);
  return (
    <LabShell
      viz={
        <div className="grid items-center gap-4 md:grid-cols-2">
          <Plot x={[-1.4, 1.4]} y={[-1.4, 1.4]} equal aspect={1} maxH={340} grid={false} ariaLabel="hours mapped onto a circle">
            <Polyline pts={Array.from({ length: 121 }, (_, i) => pt((i / 120) * period))} style={{ stroke: C.lineStrong }} />
            {Array.from({ length: period }, (_, h) => <Dot key={h} p={pt(h)} r={h === hour ? 6 : h === 0 ? 5 : 2.5} color={h === hour ? C.pink : h === 0 ? C.blue : C.muted} />)}
            <Polyline pts={[a, b]} style={{ stroke: C.green, strokeWidth: 2 }} />
            <Arrow from={[0, 0]} to={a} color="pink" width={1.4} />
          </Plot>
          <div className="space-y-3 text-[13px] text-muted">
            <p>Raw encoding puts hour {hour} and hour 0 <span className="text-fg">{Math.abs(hour - 0)}</span> units apart, although they are {Math.min(hour, period - hour)} hour(s) apart on the clock.</p>
            <p>On the circle their Euclidean distance is <span className="num text-green">{fmt(dCyc)}</span> — neighbouring times stay neighbours.</p>
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="value t" value={hour} min={0} max={period - 1} step={1} onChange={setHour} />
          <Select label="Period P" value={String(period)} onChange={(v) => { setPeriod(Number(v)); setHour(Math.min(hour, Number(v) - 1)); }} options={[{ value: "24", label: "24 (hour of day)" }, { value: "7", label: "7 (day of week)" }, { value: "12", label: "12 (month)" }]} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\sin(2\\pi t/P)", tex: true, value: fmt(a[1]) }, { label: "\\cos(2\\pi t/P)", tex: true, value: fmt(a[0]) }, { label: "\\text{raw distance to }0", tex: true, value: hour }, { label: "\\text{circular distance to }0", tex: true, value: fmt(dCyc) }]} />}
      calc={<CalcSteps steps={[{ tex: `(x_{\\sin},x_{\\cos})=\\left(\\sin\\tfrac{2\\pi\\cdot${hour}}{${period}},\\ \\cos\\tfrac{2\\pi\\cdot${hour}}{${period}}\\right)=(${tex(a[1])},${tex(a[0])})` }]} />}
      note="Both sine and cosine are needed: either one alone maps two different times to the same value."
    />
  );
}

