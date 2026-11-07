"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, C, Polyline } from "@/components/visualization/Plot";
import { Bars } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Toggle, Select } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { makeDataset } from "@/lib/ml/models";
import { histogram } from "@/lib/statistics/descriptive";
import { fmt, pct, tex, cn } from "@/lib/utilities/format";
import { MiniTable, BarList } from "@/visualizers/shared";
import { fitLogistic } from "@/visualizers/machine-learning/MLClassify";

/* ------------------------------ data validation ------------------------------ */
type Rec = { id: string; age: unknown; country: unknown; income: unknown; signup: unknown };
const COUNTRIES = ["IN", "US", "DE", "BR", "JP"];
export function DataValidationViz() {
  const [faults, setFaults] = useState(0.25);
  const [rules, setRules] = useState({ types: true, range: true, nulls: true, enums: true, unique: true });
  const [policy, setPolicy] = useState<"quarantine" | "fail">("quarantine");
  const rows: Rec[] = useMemo(() => {
    const r = mulberry32(4);
    const out: Rec[] = Array.from({ length: 12 }, (_, i) => ({ id: `u${100 + i}`, age: 18 + Math.floor(r() * 60), country: COUNTRIES[Math.floor(r() * 5)], income: Math.round(20000 + r() * 80000), signup: `2025-0${1 + Math.floor(r() * 9)}-1${Math.floor(r() * 9)}` }));
    const bad = [(x: Rec) => { x.age = -4; }, (x: Rec) => { x.country = "Germany"; }, (x: Rec) => { x.income = "N/A"; }, (x: Rec) => { x.age = null; }, (x: Rec) => { x.id = "u101"; }, (x: Rec) => { x.age = 214; }, (x: Rec) => { x.signup = "not a date"; }];
    const r2 = mulberry32(9);
    out.forEach((x, i) => { if (i > 1 && r2() < faults) bad[i % bad.length](x); });
    return out;
  }, [faults]);
  const check = (x: Rec, all: Rec[]) => {
    const errs: Partial<Record<keyof Rec, string>> = {};
    if (rules.nulls) (["age", "country", "income"] as const).forEach((k) => { if (x[k] === null || x[k] === undefined) errs[k] = "null"; });
    if (rules.types) { if (!errs.age && typeof x.age !== "number") errs.age = "type"; if (typeof x.income !== "number") errs.income = "type"; if (typeof x.signup !== "string" || Number.isNaN(Date.parse(x.signup as string))) errs.signup = "not a date"; }
    if (rules.range && typeof x.age === "number" && (x.age < 0 || x.age > 120)) errs.age = "range 0–120";
    if (rules.enums && !COUNTRIES.includes(x.country as string)) errs.country = "not in enum";
    if (rules.unique && all.filter((y) => y.id === x.id).length > 1) errs.id = "duplicate";
    return errs;
  };
  const results = rows.map((x) => check(x, rows));
  const failing = results.filter((e) => Object.keys(e).length > 0).length;
  const counts: Record<string, number> = {};
  results.forEach((e) => Object.values(e).forEach((v) => (counts[v!] = (counts[v!] ?? 0) + 1)));
  const outcome = failing === 0 ? "accept batch" : policy === "fail" ? "reject whole batch" : `load ${rows.length - failing} rows, quarantine ${failing}`;
  const cell = (x: Rec, k: keyof Rec, e: Partial<Record<keyof Rec, string>>) => <span className={cn(e[k] && "rounded bg-red/25 px-1 text-fg")} title={e[k]}>{x[k] === null ? "null" : String(x[k])}</span>;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <MiniTable head={["id", "age", "country", "income", "signup", "issues"]} rows={rows.map((x, i) => [cell(x, "id", results[i]), cell(x, "age", results[i]), cell(x, "country", results[i]), cell(x, "income", results[i]), cell(x, "signup", results[i]), Object.values(results[i]).join(", ") || "✓"])} />
          <div className={cn("rounded-lg border px-3 py-2 text-[13px]", failing ? "border-red/50 text-fg" : "border-green/50 text-fg")}>Decision: <b>{outcome}</b></div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="fault rate in the incoming batch" value={faults} min={0} max={0.8} step={0.05} onChange={setFaults} format={(v) => pct(v)} />
          {(Object.keys(rules) as (keyof typeof rules)[]).map((k) => <Toggle key={k} label={{ types: "type checks", range: "age range 0–120", nulls: "no nulls", enums: "country ∈ {IN, US, DE, BR, JP}", unique: "unique id" }[k]} checked={rules[k]} onChange={(v) => setRules((r) => ({ ...r, [k]: v }))} />)}
          <Segmented label="On failure" value={policy} onChange={setPolicy} options={[{ value: "quarantine", label: "Quarantine rows" }, { value: "fail", label: "Fail batch" }]} />
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{rows failing}", tex: true, value: `${failing} / ${rows.length}`, color: failing ? C.red : C.green },
        ...Object.entries(counts).slice(0, 4).map(([k, v]) => ({ label: `\\text{${k.replace(/[–]/g, "-")}}`, tex: true, value: v })),
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{valid}(r)=\bigwedge_{\text{rules }\rho}\rho(r),\qquad\text{batch pass rate}=1-\frac{${failing}}{${rows.length}}=${tex(1 - failing / rows.length, 3)}`, note: "turn a rule off to see which problems it was catching" }]} />}
    />
  );
}

/* ------------------------------ feature store: point-in-time correctness ------------------------------ */
export function FeatureStoreViz() {
  const [labelDay, setLabelDay] = useState(14);
  const [ttl, setTtl] = useState(10);
  const feat = useMemo(() => { const r = mulberry32(3); let v = 40; return [1, 4, 6, 9, 13, 18, 21, 25, 28].map((d) => { v = Math.max(5, v + 15 * randn(r)); return { day: d, v }; }); }, []);
  const labels = [5, 10, 14, 19, 26];
  const pit = (t: number) => { const c = feat.filter((f) => f.day <= t); const last = c[c.length - 1]; return last && t - last.day <= ttl ? last : null; };
  const naive = feat[feat.length - 1];
  const cur = pit(labelDay);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[0, 30]} y={[0, Math.max(...feat.map((f) => f.v)) * 1.15]} aspect={0.38} maxH={280} xLabel="day" yLabel="avg_spend_7d" ariaLabel="feature values over time with the label timestamp">
            <Polyline pts={feat.flatMap((f, i) => [[f.day, f.v], [feat[i + 1]?.day ?? 30, f.v]] as [number, number][])} style={{ stroke: C.blue, strokeWidth: 2 }} />
            {feat.map((f, i) => <Dot key={i} p={[f.day, f.v]} r={4} color={C.blue} />)}
            <Polyline pts={[[labelDay, 0], [labelDay, 999]]} style={{ stroke: C.pink, strokeWidth: 2 }} />
            {cur && <Dot p={[cur.day, cur.v]} r={7} color={C.green} stroke={C.fg} />}
            <Dot p={[naive.day, naive.v]} r={7} color={C.red} stroke={C.fg} />
          </Plot>
          <Legend items={[{ label: "feature value written to the store", color: C.blue }, { label: "label (prediction) time", color: C.pink }, { label: "point-in-time value", color: C.green }, { label: "naive 'latest' value (future leak)", color: C.red }]} />
          <MiniTable head={["label day", "point-in-time feature", "naive latest feature", "leaks future?"]} rows={labels.map((t) => { const p = pit(t); return [t, p ? `${fmt(p.v, 1)} (day ${p.day})` : "missing (TTL)", `${fmt(naive.v, 1)} (day ${naive.day})`, naive.day > t ? "yes" : "no"]; })} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="label timestamp (day)" value={labelDay} min={1} max={30} step={1} onChange={setLabelDay} />
          <ParameterSlider label="feature TTL (days)" value={ttl} min={1} max={30} step={1} onChange={setTtl} />
          <p className="text-[12px] leading-relaxed text-faint">A feature store keeps every feature value with its timestamp. To build training data it performs a point-in-time join: for each label, take the latest feature value written <i>before</i> that moment. Joining on the latest value instead lets information from the future leak into training — offline metrics look great, production does not. The same store serves the latest values online, so training and serving compute features identically.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{point-in-time value}", tex: true, value: cur ? fmt(cur.v, 2) : "missing", color: C.green },
        { label: "\\text{naive value}", tex: true, value: fmt(naive.v, 2), color: C.red },
        { label: "\\text{training rows that leak}", tex: true, value: `${labels.filter((t) => naive.day > t).length} / ${labels.length}` },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`x(t_{\text{label}})=v_{k^*},\quad k^*=\arg\max_k\{t_k: t_k\le t_{\text{label}}\},\ \text{provided }t_{\text{label}}-t_{k^*}\le\text{TTL}` }]} />}
    />
  );
}

/* ------------------------------ batch vs real-time inference ------------------------------ */
export function PipelineViz() {
  const [interval, setInterval_] = useState(24);
  const [users, setUsers] = useState(1_000_000);
  const [reqPerDay, setReqPerDay] = useState(200_000);
  const [modelMs, setModelMs] = useState(30);
  const batchComputesPerDay = users * (24 / interval);
  const onlineComputesPerDay = reqPerDay;
  const stalenessBatch = interval / 2; // average age of a precomputed score (hours)
  const latBatch = 3, latOnline = 8 + modelMs; // ms: key-value lookup vs feature fetch + model
  const curve = [1, 2, 4, 6, 12, 24, 48].map((h) => ({ h, cost: users * (24 / h) }));
  return (
    <LabShell
      viz={
        <div className="grid gap-3 lg:grid-cols-2">
          <Plot x={[0, 48]} y={[0, Math.max(users * 24, reqPerDay) * 1.05]} aspect={0.75} maxH={280} xLabel="batch interval (hours)" yLabel="predictions computed / day" ariaLabel="compute cost against batch interval">
            <Polyline pts={curve.map((c) => [c.h, c.cost])} style={{ stroke: C.blue, strokeWidth: 2 }} />
            <Polyline pts={[[0, onlineComputesPerDay], [48, onlineComputesPerDay]]} style={{ stroke: C.pink, strokeWidth: 2, strokeDasharray: "5 3" }} />
            <Dot p={[interval, batchComputesPerDay]} r={6} color={C.blue} stroke={C.fg} />
          </Plot>
          <div className="space-y-2">
            <BarList items={[{ label: "batch: predictions/day", value: batchComputesPerDay, color: C.blue, note: "" }, { label: "online: predictions/day", value: onlineComputesPerDay, color: C.pink }]} digits={0} />
            <BarList items={[{ label: "batch: avg staleness (h)", value: stalenessBatch, color: C.blue }, { label: "online: staleness (h)", value: 0, color: C.pink }]} digits={1} max={24} />
            <BarList items={[{ label: "batch: request latency (ms)", value: latBatch, color: C.blue }, { label: "online: request latency (ms)", value: latOnline, color: C.pink }]} digits={0} max={100} />
          </div>
          <div className="lg:col-span-2"><Legend items={[{ label: "batch scoring (precompute for every user)", color: C.blue }, { label: "real-time scoring (only when requested)", color: C.pink, dashed: true }]} /></div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="batch interval (hours)" value={interval} min={1} max={48} step={1} onChange={setInterval_} />
          <ParameterSlider label="users" value={users} min={10_000} max={5_000_000} step={10_000} onChange={setUsers} format={(v) => v.toLocaleString("en-US")} />
          <ParameterSlider label="requests per day" value={reqPerDay} min={1_000} max={5_000_000} step={1_000} onChange={setReqPerDay} format={(v) => v.toLocaleString("en-US")} />
          <ParameterSlider label="model latency (ms)" value={modelMs} min={1} max={200} step={1} onChange={setModelMs} />
          <p className="text-[12px] leading-relaxed text-faint">Batch inference precomputes predictions for everyone on a schedule and serves them from a key-value store: cheap per request, but predictions are stale and wasted on users who never come back. Real-time inference computes on demand with fresh features, at the cost of latency and always-on serving infrastructure. Training pipelines run separately on a schedule or when data drifts.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{batch / online compute}", tex: true, value: `${fmt(batchComputesPerDay / onlineComputesPerDay, 2)}×` },
        { label: "\\text{batch staleness}", tex: true, value: `${fmt(stalenessBatch, 1)} h (max ${interval} h)` },
        { label: "\\text{online latency}", tex: true, value: `${latOnline} ms` },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{batch work}=N_{\text{users}}\cdot\frac{24}{\Delta}=${Math.round(batchComputesPerDay).toLocaleString("en-US")},\qquad\overline{\text{staleness}}=\frac{\Delta}{2}`, note: "Δ = batch interval in hours; assumes requests arrive uniformly in time" }]} />}
    />
  );
}

/* ------------------------------ model versioning & reproducibility ------------------------------ */
export function fnv1a(s: string) { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, "0"); }
export function VersioningViz() {
  const [data, setData] = useState("v3");
  const [commit, setCommit] = useState("a1c9e2f");
  const [lr, setLr] = useState(0.01);
  const [seed, setSeed] = useState(42);
  const [pinned, setPinned] = useState(true);
  const env = pinned ? "python==3.12.4 numpy==2.1.0 torch==2.5.1" : `python==3.12 numpy>=2 torch (resolved ${new Date().getUTCMinutes() % 2 ? "2.5.1" : "2.6.0"})`;
  const parts = { data: fnv1a(`dataset:${data}`), code: fnv1a(`commit:${commit}`), params: fnv1a(JSON.stringify({ lr, seed })), env: fnv1a(env) };
  const id = fnv1a(Object.values(parts).join("|"));
  const registry = [
    { v: "1.0.0", d: "v1", c: "77b02aa", lr: 0.01, s: 42, stage: "archived" },
    { v: "1.1.0", d: "v2", c: "77b02aa", lr: 0.01, s: 42, stage: "archived" },
    { v: "1.2.0", d: "v3", c: "a1c9e2f", lr: 0.01, s: 42, stage: "production" },
    { v: "1.3.0-rc", d: "v3", c: "b4d17c0", lr: 0.003, s: 7, stage: "staging" },
  ].map((m) => ({ ...m, id: fnv1a([fnv1a(`dataset:${m.d}`), fnv1a(`commit:${m.c}`), fnv1a(JSON.stringify({ lr: m.lr, seed: m.s })), fnv1a("python==3.12.4 numpy==2.1.0 torch==2.5.1")].join("|")) }));
  const match = registry.find((m) => m.id === id);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <div className="grid gap-2 sm:grid-cols-2">
            {Object.entries(parts).map(([k, h]) => <div key={k} className="rounded-lg border border-line bg-raised px-3 py-2 text-[12.5px]"><div className="text-faint">{k} hash</div><div className="num text-fg">{h}</div></div>)}
          </div>
          <div className="rounded-lg border px-3 py-2 text-[13px]" style={{ borderColor: match ? "rgb(var(--green))" : "rgb(var(--amber))" }}>Model id = hash(data ‖ code ‖ params ‖ env) = <b className="num">{id}</b> — {match ? `identical to registered ${match.v} (${match.stage}): this run is reproducible` : "new, unregistered combination"}</div>
          <MiniTable head={["version", "dataset", "commit", "lr", "seed", "stage", "id"]} rows={registry.map((m) => [m.v, m.d, m.c, m.lr, m.s, m.stage, m.id])} rowClass={(i) => (registry[i].id === id ? "bg-green/10" : undefined)} />
        </div>
      }
      controls={
        <>
          <Select label="Dataset version" value={data} onChange={setData} options={["v1", "v2", "v3", "v4"].map((v) => ({ value: v, label: v }))} />
          <Select label="Code commit" value={commit} onChange={setCommit} options={["77b02aa", "a1c9e2f", "b4d17c0"].map((v) => ({ value: v, label: v }))} />
          <ParameterSlider label="learning rate" value={lr} min={0.001} max={0.05} step={0.001} onChange={setLr} />
          <ParameterSlider label="random seed" value={seed} min={0} max={100} step={1} onChange={setSeed} />
          <Toggle label="Pin exact dependency versions" checked={pinned} onChange={setPinned} />
          <p className="text-[12px] leading-relaxed text-faint">A model is a function of its data, code, hyperparameters, random seed and environment. Hashing all of them gives a content address: equal hashes mean the same recipe. Set dataset v3, commit a1c9e2f, lr 0.01, seed 42 to reproduce the production model. Unpinned dependencies can resolve differently over time, silently changing the recipe.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{model id}", tex: true, value: id },
        { label: "\\text{in registry}", tex: true, value: match ? match.v : "no", color: match ? C.green : C.amber },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{id}=H\big(H(\text{data})\,\|\,H(\text{code})\,\|\,H(\text{params})\,\|\,H(\text{env})\big),\quad H=\text{FNV-1a (32-bit)}`, note: "real systems use SHA-256 content hashes (DVC, MLflow, git)" }]} />}
    />
  );
}

/* ------------------------------ experiment tracking ------------------------------ */
export function ExperimentViz() {
  const [sortBy, setSortBy] = useState<"val" | "train" | "time">("val");
  const [sel, setSel] = useState<number[]>([0, 4, 8]);
  const runs = useMemo(() => {
    const tr = makeDataset("moons", 120, 0.25, 3), va = makeDataset("moons", 300, 0.25, 33);
    const feats = (x: number, y: number) => [x, y, x * x, x * y, y * y, x ** 3 / 2, y ** 3 / 2];
    const out: { id: number; lr: number; l2: number; curve: number[]; train: number; val: number; ms: number }[] = [];
    let id = 0;
    for (const lr of [0.05, 0.3, 1.5]) for (const l2 of [0, 0.01, 0.1]) {
      const t0 = performance.now();
      const curve: number[] = [];
      for (const ep of [25, 50, 100, 200, 400]) { const f = fitLogistic(tr.map((p) => feats(p.x, p.y)), tr.map((p) => p.label), ep, lr, l2); curve.push(va.reduce((s, p) => s + ((f(feats(p.x, p.y)) >= 0.5 ? 1 : 0) === p.label ? 1 : 0), 0) / va.length); }
      const f = fitLogistic(tr.map((p) => feats(p.x, p.y)), tr.map((p) => p.label), 400, lr, l2);
      const acc = (d: typeof tr) => d.reduce((s, p) => s + ((f(feats(p.x, p.y)) >= 0.5 ? 1 : 0) === p.label ? 1 : 0), 0) / d.length;
      out.push({ id: id++, lr, l2, curve, train: acc(tr), val: acc(va), ms: performance.now() - t0 });
    }
    return out;
  }, []);
  const sorted = runs.slice().sort((a, b) => (sortBy === "time" ? a.ms - b.ms : sortBy === "train" ? b.train - a.train : b.val - a.val));
  const best = runs.reduce((a, b) => (b.val > a.val ? b : a));
  const cols = [C.blue, C.pink, C.green, C.amber, C.red];
  return (
    <LabShell
      wideControls
      viz={
        <div className="grid gap-3 xl:grid-cols-[1.2fr_1fr]">
          <MiniTable head={["track", "run", "lr", "L2", "train acc", "val acc", "time (ms)"]} rows={sorted.map((r) => [
            <input key="c" type="checkbox" aria-label={`compare run ${r.id}`} checked={sel.includes(r.id)} onChange={(e) => setSel((s) => (e.target.checked ? [...s, r.id].slice(-5) : s.filter((x) => x !== r.id)))} />,
            `run-${r.id}${r.id === best.id ? " ★" : ""}`, r.lr, r.l2, pct(r.train), pct(r.val), fmt(r.ms, 0)])} />
          <Plot x={[0, 4]} y={[0.5, 1]} aspect={0.8} maxH={300} xLabel="checkpoint (25, 50, 100, 200, 400 epochs)" yLabel="validation accuracy" ariaLabel="validation curves of the selected runs">
            {sel.map((id, k) => <Polyline key={id} pts={runs[id].curve.map((v, i) => [i, v])} style={{ stroke: cols[k % 5], strokeWidth: 2 }} />)}
          </Plot>
        </div>
      }
      controls={
        <>
          <Segmented label="Sort runs by" value={sortBy} onChange={setSortBy} options={[{ value: "val", label: "Val acc" }, { value: "train", label: "Train acc" }, { value: "time", label: "Time" }]} />
          <Legend items={sel.map((id, k) => ({ label: `run-${id} (lr ${runs[id].lr}, L2 ${runs[id].l2})`, color: cols[k % 5] }))} />
          <p className="text-[12px] leading-relaxed text-faint">Nine real training runs of a logistic model on cubic features, executed in your browser. An experiment tracker records every run&apos;s parameters, metrics over time and artefacts so results can be compared, reproduced and audited instead of living in notebooks and memory.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{runs logged}", tex: true, value: runs.length },
        { label: "\\text{best run}", tex: true, value: `run-${best.id} (${pct(best.val)})`, color: C.green },
        { label: "\\text{train–val gap (best)}", tex: true, value: fmt(best.train - best.val, 3) },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{run}=\{\text{params},\ \text{metrics}(t),\ \text{artefacts},\ \text{code version},\ \text{data version}\}` }]} />}
    />
  );
}

/* ------------------------------ evaluation pipelines & release gates ------------------------------ */
export function EvalPipelineViz() {
  const [gain, setGain] = useState(0.03);
  const [regress, setRegress] = useState(0.08);
  const [tol, setTol] = useState(0.02);
  const slices = ["mobile", "desktop", "new users", "returning", "non-English"];
  const sizes = [400, 350, 150, 500, 100];
  const base = [0.86, 0.9, 0.78, 0.91, 0.74];
  const cand = base.map((b, i) => Math.min(0.99, b + gain - (i === 4 ? regress : 0)));
  const overall = (accs: number[]) => accs.reduce((s, a, i) => s + a * sizes[i], 0) / sizes.reduce((s, n) => s + n, 0);
  const oB = overall(base), oC = overall(cand);
  const gates = [
    { name: "overall accuracy ≥ baseline", pass: oC >= oB },
    ...slices.map((s, i) => ({ name: `slice '${s}' drops ≤ ${pct(tol)}`, pass: base[i] - cand[i] <= tol + 1e-12 })),
    { name: "latency p95 ≤ 120 ms", pass: true },
  ];
  const ok = gates.every((g) => g.pass);
  return (
    <LabShell
      viz={
        <div className="grid gap-3 lg:grid-cols-2">
          <div className="space-y-1.5">
            <div className="text-[12px] text-faint">Accuracy per slice — baseline vs candidate</div>
            {slices.map((s, i) => <BarList key={s} max={1} items={[{ label: `${s} (base)`, value: base[i], color: C.a("fg", 0.3) }, { label: `${s} (cand.)`, value: cand[i], color: cand[i] < base[i] - tol ? C.red : C.green }]} />)}
          </div>
          <div className="space-y-2">
            <MiniTable head={["gate", "result"]} rows={gates.map((g) => [g.name, g.pass ? "pass" : "FAIL"])} rowClass={(i) => (gates[i].pass ? undefined : "bg-red/15")} />
            <div className="rounded-lg border px-3 py-2 text-[13px]" style={{ borderColor: ok ? "rgb(var(--green))" : "rgb(var(--red))" }}>Release decision: <b>{ok ? "promote candidate" : "block release"}</b></div>
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="candidate improvement (all slices)" value={gain} min={-0.05} max={0.1} step={0.005} onChange={setGain} format={(v) => pct(v)} />
          <ParameterSlider label="regression on 'non-English'" value={regress} min={0} max={0.2} step={0.005} onChange={setRegress} format={(v) => pct(v)} />
          <ParameterSlider label="allowed slice drop" value={tol} min={0} max={0.1} step={0.005} onChange={setTol} format={(v) => pct(v)} />
          <p className="text-[12px] leading-relaxed text-faint">An evaluation pipeline runs the same fixed test suites on every candidate and compares against the current model, overall and per slice. Here the overall number improves, yet a small slice gets much worse — invisible in the average, caught by the per-slice gate.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{overall: base → cand.}", tex: true, value: `${pct(oB)} → ${pct(oC)}` },
        { label: "\\text{gates passed}", tex: true, value: `${gates.filter((g) => g.pass).length} / ${gates.length}`, color: ok ? C.green : C.red },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{acc}_{\text{overall}}=\frac{\sum_sn_s\,\text{acc}_s}{\sum_sn_s},\qquad\text{gate}_s:\ \text{acc}^{\text{base}}_s-\text{acc}^{\text{cand}}_s\le${tex(tol, 3)}`, note: "the 100-example slice barely moves the weighted average" }]} />}
    />
  );
}

/* ------------------------------ model serving: batching, latency, throughput ------------------------------ */
export function ServingViz() {
  const [rate, setRate] = useState(150);
  const [maxBatch, setMaxBatch] = useState(8);
  const [maxWait, setMaxWait] = useState(10);
  const [base, setBase] = useState(20);
  const perItem = 2; // ms per extra item in a batch
  const sim = useMemo(() => {
    const r = mulberry32(7), n = 3000;
    let t = 0; const arr: number[] = [];
    for (let i = 0; i < n; i++) { t += (-Math.log(1 - r()) / rate) * 1000; arr.push(t); } // Poisson arrivals (ms)
    const lat: number[] = []; let free = 0, i = 0, busy = 0;
    while (i < n) {
      const start0 = Math.max(free, arr[i]);
      // dynamic batching: wait until the batch is full or the oldest request has waited maxWait
      const deadline = Math.max(start0, arr[i] + maxWait);
      let j = i; while (j < n && j - i < maxBatch && arr[j] <= deadline) j++;
      const start = j - i === maxBatch ? Math.max(free, arr[j - 1]) : deadline;
      const svc = base + perItem * (j - i);
      for (let k = i; k < j; k++) lat.push(start + svc - arr[k]);
      busy += svc; free = start + svc; i = j;
    }
    const s = lat.slice().sort((a, b) => a - b);
    const span = free - arr[0];
    return { p50: s[Math.floor(0.5 * n)], p95: s[Math.floor(0.95 * n)], p99: s[Math.floor(0.99 * n)], thr: (n / span) * 1000, util: busy / span, lat, cap: (maxBatch / (base + perItem * maxBatch)) * 1000 };
  }, [rate, maxBatch, maxWait, base]);
  const hi = Math.min(Math.max(...sim.lat), sim.p99 * 1.5);
  const bins = histogram(sim.lat.filter((v) => v <= hi), 0, hi, 40);
  const overloaded = rate > sim.cap;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, hi]} y={[0, Math.max(...bins.map((b) => b.density)) * 1.15]} aspect={0.42} maxH={300} xLabel="request latency (ms)" ariaLabel="latency distribution of simulated requests">
            <Bars bins={bins} />
            {[sim.p50, sim.p95].map((p, k) => <Polyline key={k} pts={[[p, 0], [p, 999]]} style={{ stroke: k ? C.red : C.green, strokeWidth: 1.6, strokeDasharray: "4 3" }} />)}
          </Plot>
          <Legend items={[{ label: "latency of 3 000 simulated requests", color: C.blue }, { label: "p50", color: C.green, dashed: true }, { label: "p95", color: C.red, dashed: true }]} />
          {overloaded && <div className="rounded-lg border border-red/50 px-3 py-2 text-[12.5px] text-fg">Arrival rate exceeds capacity ({fmt(sim.cap, 0)} req/s at full batches): the queue grows without bound and latency explodes.</div>}
        </div>
      }
      controls={
        <>
          <ParameterSlider label="arrival rate (requests / s)" value={rate} min={10} max={600} step={10} onChange={setRate} />
          <ParameterSlider label="max batch size" value={maxBatch} min={1} max={32} step={1} onChange={setMaxBatch} />
          <ParameterSlider label="max batching wait (ms)" value={maxWait} min={0} max={50} step={1} onChange={setMaxWait} />
          <ParameterSlider label="fixed cost per forward pass (ms)" value={base} min={2} max={60} step={1} onChange={setBase} />
          <p className="text-[12px] leading-relaxed text-faint">A GPU model server pays a large fixed cost per forward pass and a small cost per extra item, so batching requests raises throughput. But waiting to fill a batch adds latency. Discrete-event simulation with Poisson arrivals and one server; service time = fixed + 2 ms × batch size.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "p_{50},\\ p_{95},\\ p_{99}", tex: true, value: `${fmt(sim.p50, 0)}, ${fmt(sim.p95, 0)}, ${fmt(sim.p99, 0)} ms` },
        { label: "\\text{throughput}", tex: true, value: `${fmt(sim.thr, 0)} req/s` },
        { label: "\\text{utilisation}", tex: true, value: pct(Math.min(1, sim.util)), color: sim.util > 0.9 ? C.red : undefined },
        { label: "\\text{capacity (full batches)}", tex: true, value: `${fmt(sim.cap, 0)} req/s` },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{capacity}=\frac{B}{t_0+B\,t_1}=\frac{${maxBatch}}{${base}+${perItem}\cdot${maxBatch}}\ \text{per ms},\qquad \text{latency}=\text{wait}+\text{queue}+\text{service}` }]} />}
    />
  );
}
