"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, C, Polyline } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { solve } from "@/lib/math/linalg";
import { pearson, variance } from "@/lib/statistics/descriptive";
import { fmt, tex } from "@/lib/utilities/format";
import { MiniTable, BarList } from "@/visualizers/shared";

/** Ordinary least squares with an intercept; returns weights [b, w1..wp]. */
export function ols(X: number[][], y: number[], ridge = 1e-8): number[] {
  const A = X.map((r) => [1, ...r]);
  const p = A[0].length;
  const AtA = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => A.reduce((s, r) => s + r[i] * r[j], 0) + (i === j && i > 0 ? ridge : 0)));
  const Aty = Array.from({ length: p }, (_, i) => A.reduce((s, r, k) => s + r[i] * y[k], 0));
  return solve(AtA, Aty) ?? Array(p).fill(0);
}
const predict = (w: number[], r: number[]) => w[0] + r.reduce((s, x, j) => s + x * w[j + 1], 0);
const r2 = (y: number[], yh: number[]) => { const m = y.reduce((a, b) => a + b, 0) / y.length; return 1 - y.reduce((s, v, i) => s + (v - yh[i]) ** 2, 0) / y.reduce((s, v) => s + (v - m) ** 2, 0); };

/* ------------------------------ aggregation features ------------------------------ */
const TX = [
  ["ana", 12, 1], ["ben", 40, 2], ["ana", 8, 3], ["cara", 95, 3], ["ben", 22, 5], ["ana", 30, 6], ["cara", 15, 8], ["dev", 5, 9],
  ["ana", 18, 10], ["ben", 60, 11], ["dev", 7, 12], ["cara", 110, 13], ["ana", 9, 14],
] as const;
type Agg = "count" | "sum" | "mean" | "max" | "recency";
export function AggregationViz() {
  const [window_, setWindow] = useState(14);
  const [aggs, setAggs] = useState<Record<Agg, boolean>>({ count: true, sum: true, mean: true, max: false, recency: true });
  const today = 14;
  const rows = TX.filter((t) => t[2] > today - window_);
  const customers = ["ana", "ben", "cara", "dev"];
  const feats = customers.map((c) => {
    const a = rows.filter((t) => t[0] === c).map((t) => t[1] as number);
    const days = rows.filter((t) => t[0] === c).map((t) => t[2] as number);
    return { c, count: a.length, sum: a.reduce((s, x) => s + x, 0), mean: a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN, max: a.length ? Math.max(...a) : NaN, recency: days.length ? today - Math.max(...days) : NaN };
  });
  const on = (Object.keys(aggs) as Agg[]).filter((k) => aggs[k]);
  return (
    <LabShell
      viz={
        <div className="grid gap-4 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
          <div className="space-y-1.5">
            <div className="text-[12px] text-faint">Raw transactions (one row per purchase)</div>
            <MiniTable head={["customer", "amount", "day"]} rows={TX.map((t) => [t[0], `$${t[1]}`, t[2]])} rowClass={(i) => (TX[i][2] > today - window_ ? undefined : "opacity-35")} />
          </div>
          <div className="space-y-1.5">
            <div className="text-[12px] text-faint">Customer-level features (one row per customer, last {window_} days)</div>
            <MiniTable head={["customer", ...on.map((k) => k === "recency" ? "days since last" : `${k}(amount)`)]} rows={feats.map((f) => [f.c, ...on.map((k) => (Number.isFinite(f[k]) ? fmt(f[k], 1) : "—"))])} />
            <p className="pt-1 text-[12px] text-faint">A model predicting churn per customer cannot use the 13 raw rows directly; group-by aggregation turns them into a fixed-length vector per customer.</p>
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="time window (days)" value={window_} min={1} max={14} step={1} onChange={setWindow} />
          {(Object.keys(aggs) as Agg[]).map((k) => <Toggle key={k} label={k === "recency" ? "recency (days since last)" : k} checked={aggs[k]} onChange={(v) => setAggs((a) => ({ ...a, [k]: v }))} />)}
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{rows in window}", tex: true, value: `${rows.length} / ${TX.length}` },
        { label: "\\text{features per customer}", tex: true, value: on.length },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{mean}_c=\frac{1}{|T_c|}\sum_{t\in T_c}\text{amount}_t,\qquad T_c=\{t:\text{customer}_t=c,\ \text{day}_t>${today - window_}\}`, note: "only transactions inside the window and before the prediction time may be used" }]} />}
    />
  );
}

/* ------------------------------ lag & rolling features ------------------------------ */
export function LagViz() {
  const [lag, setLag] = useState(7);
  const [win, setWin] = useState(7);
  const y = useMemo(() => { const r = mulberry32(4); return Array.from({ length: 70 }, (_, t) => 50 + 0.3 * t + 12 * Math.sin((2 * Math.PI * t) / 7) + 4 * randn(r)); }, []);
  const lagged = y.map((_, t) => (t - lag >= 0 ? y[t - lag] : NaN));
  // rolling mean of the PREVIOUS `win` values (shifted by one so it never sees y_t)
  const roll = y.map((_, t) => (t - win >= 0 ? y.slice(t - win, t).reduce((s, v) => s + v, 0) / win : NaN));
  const valid = y.map((_, t) => t).filter((t) => Number.isFinite(lagged[t]));
  const acf = pearson(valid.map((t) => y[t]), valid.map((t) => lagged[t]));
  const seg = (v: number[]) => v.map((x, t) => [t, x] as [number, number]).filter((p) => Number.isFinite(p[1]));
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[0, 69]} y={[25, 95]} aspect={0.42} maxH={320} xLabel="day t" yLabel="daily sales" ariaLabel="sales series with lag and rolling mean features">
            <Polyline pts={seg(y)} style={{ stroke: C.fg, strokeWidth: 1.6 }} />
            <Polyline pts={seg(lagged)} style={{ stroke: C.blue, strokeWidth: 1.4, strokeDasharray: "5 3" }} />
            <Polyline pts={seg(roll)} style={{ stroke: C.pink, strokeWidth: 2 }} />
          </Plot>
          <Legend items={[{ label: "y_t (target)", color: C.fg }, { label: `lag_${lag} = y(t−${lag})`, color: C.blue, dashed: true }, { label: `rolling mean of the previous ${win} days`, color: C.pink }]} />
          <MiniTable head={["t", "y_t", `lag_${lag}`, `roll_${win}`]} rows={[0, 1, lag, lag + 1, 20].filter((t, i, a) => a.indexOf(t) === i && t < 70).map((t) => [t, fmt(y[t], 1), Number.isFinite(lagged[t]) ? fmt(lagged[t], 1) : "NaN", Number.isFinite(roll[t]) ? fmt(roll[t], 1) : "NaN"])} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="lag k (days)" value={lag} min={1} max={21} step={1} onChange={setLag} />
          <ParameterSlider label="rolling window w" value={win} min={2} max={21} step={1} onChange={setWin} />
          <p className="text-[12px] leading-relaxed text-faint">The series has a weekly cycle, so lag 7 (and 14) lines up with the target. The first rows have no history and become NaN — drop or impute them.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: `\\operatorname{corr}(y_t,y_{t-${lag}})`, tex: true, value: fmt(acf, 3), color: Math.abs(acf) > 0.7 ? C.green : undefined },
        { label: "\\text{rows lost}", tex: true, value: Math.max(lag, win) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{lag}_k(t)=y_{t-k},\qquad \text{roll}_w(t)=\frac1w\sum_{i=1}^{w}y_{t-i}`, note: "the rolling window stops at t − 1: including y_t would leak the target" },
      ]} />}
    />
  );
}

/* ------------------------------ domain-specific features ------------------------------ */
export function DomainFeatureViz() {
  const [useArea, setUseArea] = useState(true);
  const [useLog, setUseLog] = useState(false);
  const data = useMemo(() => {
    const r = mulberry32(8);
    return Array.from({ length: 80 }, () => {
      const L = 5 + 15 * r(), W = 4 + 10 * r();
      const price = 2.2 * L * W * Math.exp(0.08 * randn(r)); // price ∝ floor area
      return { L, W, area: L * W, price };
    });
  }, []);
  const ys = data.map((d) => (useLog ? Math.log(d.price) : d.price));
  const raw = data.map((d) => [d.L, d.W]);
  const withA = data.map((d) => [d.L, d.W, useLog ? Math.log(d.area) : d.area]);
  const wr = ols(raw, ys), wa = ols(withA, ys);
  const r2raw = r2(ys, raw.map((r) => predict(wr, r))), r2a = r2(ys, withA.map((r) => predict(wa, r)));
  const xf = data.map((d) => (useArea ? (useLog ? Math.log(d.area) : d.area) : d.L));
  const xr: [number, number] = [Math.min(...xf) * 0.95, Math.max(...xf) * 1.03], yr: [number, number] = [Math.min(...ys) * 0.95, Math.max(...ys) * 1.03];
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={xr} y={yr} aspect={0.55} maxH={330} xLabel={useArea ? (useLog ? "log(area)" : "area = length × width (m²)") : "length (m)"} yLabel={useLog ? "log(price)" : "price (k)"} ariaLabel="price against a raw or engineered feature">
            {data.map((d, i) => <Dot key={i} p={[xf[i], ys[i]]} r={3} color={useArea ? C.pink : C.blue} opacity={0.8} />)}
          </Plot>
          <BarList max={1} items={[{ label: "R² length + width", value: r2raw, color: C.blue }, { label: "R² + area feature", value: r2a, color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <Toggle label="Plot against the engineered area" checked={useArea} onChange={setUseArea} />
          <Toggle label="Log-transform price and area" checked={useLog} onChange={setUseLog} />
          <p className="text-[12px] leading-relaxed text-faint">An estate agent knows price depends on floor area, i.e. the product of length and width. A linear model cannot form that product itself; adding it as a feature captures the domain knowledge. In log space the relation becomes exactly additive: log area = log L + log W.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "R^2_{\\text{raw}}", tex: true, value: fmt(r2raw, 4), color: C.blue },
        { label: "R^2_{\\text{+area}}", tex: true, value: fmt(r2a, 4), color: C.pink },
      ]} />}
      calc={<CalcSteps steps={[{ tex: useLog ? String.raw`\log\text{price}=\beta_0+\beta_1L+\beta_2W+\beta_3\log(LW)` : String.raw`\text{price}=\beta_0+\beta_1L+\beta_2W+\beta_3\,(L\cdot W)`, note: "fitted by least squares on 80 synthetic houses" }]} />}
    />
  );
}

/* ------------------------------ filter methods ------------------------------ */
const FEATS = ["age", "income", "visits", "constant_id", "noise_a", "tenure", "near_dup_income", "noise_b"];
export function FilterViz() {
  const [vt, setVt] = useState(0.05);
  const [ct, setCt] = useState(0.2);
  const [dedupe, setDedupe] = useState(true);
  const { X, y } = useMemo(() => {
    const r = mulberry32(13);
    const X: number[][] = [], y: number[] = [];
    for (let i = 0; i < 200; i++) {
      const age = randn(r), inc = randn(r), vis = randn(r), ten = randn(r);
      const row = [age, inc, vis, 1 + 0.01 * randn(r), randn(r), ten, inc + 0.05 * randn(r), randn(r)];
      X.push(row);
      y.push(0.8 * inc + 0.5 * vis + 0.3 * ten + 0.1 * age + 0.5 * randn(r));
    }
    return { X, y };
  }, []);
  const cols = FEATS.map((_, j) => X.map((r) => r[j]));
  const vars = cols.map((c) => variance(c));
  const cors = cols.map((c) => Math.abs(pearson(c, y)));
  const keep = FEATS.map((_, j) => vars[j] >= vt && cors[j] >= ct);
  if (dedupe) FEATS.forEach((_, j) => { for (let i = 0; i < j; i++) if (keep[i] && keep[j] && Math.abs(pearson(cols[i], cols[j])) > 0.95) keep[j] = false; });
  return (
    <LabShell
      viz={
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-1.5"><div className="text-[12px] text-faint">Variance of each feature (threshold {fmt(vt, 2)})</div><BarList items={FEATS.map((f, j) => ({ label: f, value: vars[j], color: vars[j] >= vt ? C.blue : C.a("fg", 0.25) }))} /></div>
          <div className="space-y-1.5"><div className="text-[12px] text-faint">|correlation| with the target (threshold {fmt(ct, 2)})</div><BarList max={1} items={FEATS.map((f, j) => ({ label: f, value: cors[j], color: keep[j] ? C.green : C.a("fg", 0.25) }))} /></div>
          <div className="md:col-span-2 text-[12.5px] text-muted">Kept: <span className="text-fg">{FEATS.filter((_, j) => keep[j]).join(", ") || "none"}</span></div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="variance threshold" value={vt} min={0} max={1} step={0.01} onChange={setVt} />
          <ParameterSlider label="|corr| threshold" value={ct} min={0} max={0.8} step={0.01} onChange={setCt} />
          <Toggle label="Drop near-duplicates (|r| > 0.95)" checked={dedupe} onChange={setDedupe} />
          <p className="text-[12px] leading-relaxed text-faint">Filter methods score each feature on its own, before any model is trained: fast, but blind to interactions (a feature useless alone can matter together with another).</p>
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{kept}", tex: true, value: `${keep.filter(Boolean).length} / ${FEATS.length}` }]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{keep }j\iff \operatorname{Var}(x_j)\ge${tex(vt, 2)}\ \wedge\ |r(x_j,y)|\ge${tex(ct, 2)}` }]} />}
    />
  );
}

/* ------------------------------ mutual information ------------------------------ */
type Rel = "linear" | "quadratic" | "sine" | "independent";
export function mutualInfo(x: number[], y: number[], bins: number) {
  const xlo = Math.min(...x), xhi = Math.max(...x), ylo = Math.min(...y), yhi = Math.max(...y);
  const bx = (v: number) => Math.min(bins - 1, Math.floor(((v - xlo) / (xhi - xlo || 1)) * bins));
  const by = (v: number) => Math.min(bins - 1, Math.floor(((v - ylo) / (yhi - ylo || 1)) * bins));
  const J = Array.from({ length: bins }, () => Array(bins).fill(0));
  x.forEach((v, i) => J[bx(v)][by(y[i])]++);
  const n = x.length, px = J.map((r) => r.reduce((s, c) => s + c, 0) / n), py = J[0].map((_, j) => J.reduce((s, r) => s + r[j], 0) / n);
  let mi = 0;
  J.forEach((r, i) => r.forEach((c, j) => { if (c > 0) { const p = c / n; mi += p * Math.log(p / (px[i] * py[j])); } }));
  return mi;
}
export function MutualInfoViz() {
  const [rel, setRel] = useState<Rel>("quadratic");
  const [noise, setNoise] = useState(0.15);
  const [bins, setBins] = useState(8);
  const { x, y } = useMemo(() => {
    const r = mulberry32(17);
    const x = Array.from({ length: 400 }, () => 2 * r() - 1);
    const f: Record<Rel, (v: number) => number> = { linear: (v) => v, quadratic: (v) => v * v * 2 - 0.66, sine: (v) => Math.sin(3 * Math.PI * v), independent: () => 0 };
    const y = x.map((v) => f[rel](v) + (rel === "independent" ? randn(r) * 0.6 : noise * randn(r)));
    return { x, y };
  }, [rel, noise]);
  const r = pearson(x, y), mi = mutualInfo(x, y, bins);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-1.05, 1.05]} y={[-2, 2]} aspect={0.5} maxH={320} xLabel="feature x" yLabel="target y" ariaLabel="scatter plot of a feature against the target">
            {x.map((v, i) => <Dot key={i} p={[v, y[i]]} r={2.2} color={C.blue} opacity={0.6} />)}
          </Plot>
          <BarList max={1.5} items={[{ label: "|Pearson r|", value: Math.abs(r), color: C.amber }, { label: "mutual information (nats)", value: mi, color: C.green }]} />
        </div>
      }
      controls={
        <>
          <Select label="Relationship" value={rel} onChange={setRel} options={[{ value: "linear", label: "linear" }, { value: "quadratic", label: "quadratic (U-shape)" }, { value: "sine", label: "periodic" }, { value: "independent", label: "independent" }]} />
          <ParameterSlider label="noise" value={noise} min={0} max={1} step={0.01} onChange={setNoise} />
          <ParameterSlider label="histogram bins" value={bins} min={3} max={20} step={1} onChange={setBins} />
          <p className="text-[12px] leading-relaxed text-faint">For the U-shape and the periodic pattern, correlation is near 0 although y is almost a function of x. Mutual information detects any dependence. The histogram estimate is biased upward with many bins and few points, so even independent data shows a small positive MI.</p>
        </>
      }
      readouts={<Readouts items={[{ label: "r", tex: true, value: fmt(r, 3), color: C.amber }, { label: "\\hat I(X;Y)", tex: true, value: fmt(mi, 3), color: C.green }]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)\,p(y)}\ \ge0,\qquad =0\iff X\perp Y`, note: `estimated from a ${bins} × ${bins} joint histogram of 400 points` }]} />}
    />
  );
}

/* ------------------------------ wrapper methods / RFE ------------------------------ */
const RFE_NAMES = ["x1*", "x2*", "x3*", "n1", "n2", "n3", "n4", "n5"];
export function RFEViz() {
  const [k, setK] = useState(3);
  const res = useMemo(() => {
    const r = mulberry32(31);
    const gen = (n: number) => Array.from({ length: n }, () => { const row = Array.from({ length: 8 }, () => randn(r)); return { row, y: 3 * row[0] - 2 * row[1] + 1 * row[2] + randn(r) }; });
    const tr = gen(20), va = gen(200);
    let active = [0, 1, 2, 3, 4, 5, 6, 7];
    const path: { feats: number[]; mse: number; removed?: number }[] = [];
    while (active.length >= 1) {
      const w = ols(tr.map((d) => active.map((j) => d.row[j])), tr.map((d) => d.y));
      const mse = va.reduce((s, d) => s + (d.y - predict(w, active.map((j) => d.row[j]))) ** 2, 0) / va.length;
      // all features have unit variance, so |coefficient| is a fair importance score
      const weakest = active[w.slice(1).reduce((bi, v, i, a) => (Math.abs(v) < Math.abs(a[bi]) ? i : bi), 0)];
      path.push({ feats: active.slice(), mse, removed: active.length > 1 ? weakest : undefined });
      active = active.filter((j) => j !== weakest);
    }
    return path;
  }, []);
  const cur = res.find((p) => p.feats.length === k)!;
  const best = res.reduce((b, p) => (p.mse < b.mse ? p : b));
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[0.5, 8.5]} y={[0, Math.max(...res.map((p) => p.mse)) * 1.1]} aspect={0.42} maxH={280} xLabel="number of features kept" yLabel="validation MSE" ariaLabel="validation error along the recursive feature elimination path">
            <Polyline pts={res.map((p) => [p.feats.length, p.mse])} style={{ stroke: C.blue, strokeWidth: 2 }} />
            {res.map((p, i) => <Dot key={i} p={[p.feats.length, p.mse]} r={p.feats.length === k ? 6 : 3.5} color={p === best ? C.green : p.feats.length === k ? C.pink : C.blue} />)}
          </Plot>
          <div className="flex flex-wrap gap-1.5">{RFE_NAMES.map((n, j) => <span key={n} className="rounded-md border border-line px-2 py-0.5 text-[12px]" style={{ background: cur.feats.includes(j) ? "rgb(var(--green) / 0.18)" : undefined, color: cur.feats.includes(j) ? undefined : "rgb(var(--faint))" }}>{n}</span>)}</div>
          <p className="text-[12px] text-faint">Elimination order: {res.filter((p) => p.removed !== undefined).map((p) => RFE_NAMES[p.removed!]).join(" → ")}. Starred features are the true signal.</p>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="features to keep" value={k} min={1} max={8} step={1} onChange={setK} />
          <p className="text-[12px] leading-relaxed text-faint">Recursive feature elimination fits the model, drops the feature with the smallest |coefficient|, and repeats. Wrapper methods judge features by the model&apos;s actual performance, at the cost of many fits.</p>
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{val MSE}", tex: true, value: fmt(cur.mse, 3), color: C.pink }, { label: "\\text{best size}", tex: true, value: best.feats.length, color: C.green }]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`y=3x_1-2x_2+x_3+\varepsilon,\quad \text{20 training rows, 200 validation rows}` }, { tex: String.raw`\text{repeat: fit OLS on }S;\ S\leftarrow S\setminus\{\arg\min_{j\in S}|\hat\beta_j|\}` }]} />}
    />
  );
}
