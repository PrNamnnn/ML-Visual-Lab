"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, C, HeatmapCanvas, useThemeRGB } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Toggle, Select, AnimationControls } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { kmeansAssign, kmeansUpdate, type P2 } from "@/lib/ml/models";
import { fmt, pct, tex, cn } from "@/lib/utilities/format";
import { MiniTable } from "@/visualizers/shared";

/* ------------------------------ vector database: IVF approximate nearest neighbours ------------------------------ */
export function VectorDBViz() {
  const [cells, setCells] = useState(12);
  const [nprobe, setNprobe] = useState(2);
  const [q, setQ] = useState<P2>([0.3, -0.4]);
  const th = useThemeRGB();
  const K = 10;
  const pts = useMemo(() => { const r = mulberry32(3); const cs = Array.from({ length: 9 }, () => [4 * r() - 2, 4 * r() - 2]); return Array.from({ length: 900 }, (_, i) => { const c = cs[i % 9]; return [c[0] + 0.35 * randn(r), c[1] + 0.35 * randn(r)] as P2; }); }, []);
  const index = useMemo(() => {
    let Cc: P2[] = pts.filter((_, i) => i % Math.floor(pts.length / cells) === 0).slice(0, cells), a = kmeansAssign(pts, Cc);
    for (let t = 0; t < 15; t++) { Cc = kmeansUpdate(pts, a, Cc); a = kmeansAssign(pts, Cc); }
    return { Cc, a };
  }, [pts, cells]);
  const d2 = (a: P2, b: P2) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const exact = pts.map((p, i) => ({ i, d: d2(p, q) })).sort((a, b) => a.d - b.d).slice(0, K).map((o) => o.i);
  const probeCells = index.Cc.map((c, k) => ({ k, d: d2(c, q) })).sort((a, b) => a.d - b.d).slice(0, nprobe).map((o) => o.k);
  const cand = pts.map((_, i) => i).filter((i) => probeCells.includes(index.a[i]));
  const approx = cand.map((i) => ({ i, d: d2(pts[i], q) })).sort((a, b) => a.d - b.d).slice(0, K).map((o) => o.i);
  const recall = approx.filter((i) => exact.includes(i)).length / K;
  const palette = [th.blue, th.pink, th.green, th.amber, th.red, th.muted];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-3, 3]} equal aspect={0.75} maxH={460} ariaLabel="IVF index cells and query results" onPointerDown={(p) => setQ(p)}
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={6} deps={[index, nprobe, q, th]} f={(x, y) => { let b = 0, bd = Infinity; index.Cc.forEach((c, k) => { const dd = (c[0] - x) ** 2 + (c[1] - y) ** 2; if (dd < bd) { bd = dd; b = k; } }); const c = palette[b % palette.length], on = probeCells.includes(b); const kk = on ? 0.22 : 0.06; return [th.panel[0] + (c[0] - th.panel[0]) * kk, th.panel[1] + (c[1] - th.panel[1]) * kk, th.panel[2] + (c[2] - th.panel[2]) * kk, 255]; }} />}>
            {pts.map((p, i) => <Dot key={i} p={p} r={approx.includes(i) ? 4.5 : exact.includes(i) ? 4.5 : 1.6} color={approx.includes(i) && exact.includes(i) ? C.green : exact.includes(i) ? C.red : approx.includes(i) ? C.amber : C.a("fg", 0.35)} />)}
            {index.Cc.map((c, k) => <Dot key={"c" + k} p={c} r={3} color={C.fg} stroke={C.bg} />)}
            <Dot p={q} r={7} color={C.pink} stroke={C.fg} />
          </Plot>
          <Legend items={[{ label: "query (click to move)", color: C.pink }, { label: "found and truly nearest", color: C.green }, { label: "true neighbour missed", color: C.red }, { label: "returned but not top-10", color: C.amber }, { label: "probed cells (brighter)", color: C.muted }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="IVF cells (k-means centroids)" value={cells} min={2} max={40} step={1} onChange={setCells} />
          <ParameterSlider label="cells probed (nprobe)" value={nprobe} min={1} max={cells} step={1} onChange={setNprobe} />
          <p className="text-[12px] leading-relaxed text-faint">An inverted-file index clusters the vectors into cells. A query is compared with the cell centroids, and only vectors in the nprobe closest cells are scanned. Fewer comparisons, but true neighbours just across a cell boundary can be missed. Raising nprobe trades speed for recall. (HNSW graphs and product quantisation are other ways to make the same trade.)</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{recall@10}", tex: true, value: pct(recall), color: recall === 1 ? C.green : recall < 0.8 ? C.red : C.amber },
        { label: "\\text{distances computed}", tex: true, value: `${cells + cand.length} vs ${pts.length}` },
        { label: "\\text{speed-up}", tex: true, value: `${fmt(pts.length / (cells + cand.length), 1)}×` },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{cost}\approx C+\frac{n_{\text{probe}}}{C}N=${cells}+\frac{${nprobe}}{${cells}}\cdot${pts.length},\qquad\text{recall@}k=\frac{|\text{ANN}_k\cap\text{exact}_k|}{k}` }]} />}
    />
  );
}

/* ------------------------------ hybrid search: BM25 + dense + RRF ------------------------------ */
const DOCS = [
  "How to reset a forgotten account password",
  "Changing your login credentials in settings",
  "Password policy: length and special characters",
  "Troubleshooting sign in problems on mobile",
  "Refund policy for annual subscriptions",
  "Cancel a subscription and request a refund",
  "Error code E1042 when syncing files",
  "Two-factor authentication setup guide",
  "Recover access when locked out of your account",
  "Billing address and invoice download",
];
const CONCEPTS: Record<string, string[]> = { auth: ["password", "login", "credentials", "sign", "locked", "access", "authentication", "account", "recover"], money: ["refund", "billing", "invoice", "subscription", "cancel", "payment"], error: ["error", "e1042", "troubleshooting", "problems", "syncing"] };
const QUERIES = [
  { q: "can't log in to my account", rel: [0, 1, 3, 8] },
  { q: "error E1042", rel: [6] },
  { q: "get my money back", rel: [4, 5] },
];
const tok = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter((w) => w.length > 1 && !["a", "to", "and", "of", "my", "your", "in", "on", "for", "the", "can", "when", "how", "get", "back"].includes(w)).map((w) => (w === "log" ? "login" : w));
export function HybridSearchViz() {
  const [qi, setQi] = useState(0);
  const [kRrf, setKRrf] = useState(60);
  const [mode, setMode] = useState<"bm25" | "dense" | "hybrid">("hybrid");
  const docs = DOCS.map(tok), N = DOCS.length, avgdl = docs.reduce((s, d) => s + d.length, 0) / N;
  const qt = tok(QUERIES[qi].q);
  // BM25 (k1 = 1.2, b = 0.75)
  const bm25 = docs.map((d) => qt.reduce((s, t) => { const n = docs.filter((x) => x.includes(t)).length; if (!n) return s; const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5)), f = d.filter((x) => x === t).length; return s + (idf * f * 2.2) / (f + 1.2 * (1 - 0.75 + (0.75 * d.length) / avgdl)); }, 0));
  // dense stand-in: concept vectors (synonym groups) + a little lexical signal, cosine similarity
  const vec = (ws: string[]) => [...Object.values(CONCEPTS).map((c) => ws.filter((w) => c.includes(w)).length + (ws.join(" ").includes("money") && c.includes("refund") ? 1.5 : 0)), 0.15 * ws.length];
  const cosv = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0) / ((Math.hypot(...a) * Math.hypot(...b)) || 1);
  const qv = vec(qt);
  const dense = docs.map((d) => cosv(vec(d), qv));
  const rank = (s: number[]) => s.map((v, i) => ({ i, v })).sort((a, b) => b.v - a.v).map((o) => o.i);
  const rb = rank(bm25), rd = rank(dense);
  const rrf = DOCS.map((_, i) => 1 / (kRrf + rb.indexOf(i) + 1) + 1 / (kRrf + rd.indexOf(i) + 1));
  const final = mode === "bm25" ? rb : mode === "dense" ? rd : rank(rrf);
  const rel = QUERIES[qi].rel;
  const ndcg = (() => { const dcg = final.slice(0, 5).reduce((s, d, k) => s + (rel.includes(d) ? 1 / Math.log2(k + 2) : 0), 0); const idcg = rel.slice(0, 5).reduce((s, _, k) => s + 1 / Math.log2(k + 2), 0); return dcg / idcg; })();
  const list = (r: number[], score: (i: number) => number, label: string, active: boolean) => (
    <div className={cn("space-y-1 rounded-lg border p-2", active ? "border-line-strong" : "border-line opacity-70")}>
      <div className="text-[12px] text-faint">{label}</div>
      {r.slice(0, 5).map((d, k) => <div key={d} className="flex items-start gap-1.5 text-[12px]"><span className="num w-4 text-faint">{k + 1}</span><span className={cn("flex-1", rel.includes(d) ? "text-fg" : "text-muted")}>{rel.includes(d) ? "● " : ""}{DOCS[d]}</span><span className="num text-faint">{fmt(score(d), 3)}</span></div>)}
    </div>
  );
  return (
    <LabShell
      wideControls
      viz={
        <div className="grid gap-2 lg:grid-cols-3">
          {list(rb, (i) => bm25[i], "Lexical (BM25)", mode === "bm25")}
          {list(rd, (i) => dense[i], "Dense (embedding similarity)", mode === "dense")}
          {list(rank(rrf), (i) => rrf[i], `Hybrid (RRF, k = ${kRrf})`, mode === "hybrid")}
          <p className="text-[12px] text-faint lg:col-span-3">● = relevant document for this query. Query tokens after normalisation: {qt.join(", ")}.</p>
        </div>
      }
      controls={
        <>
          <Select label="Query" value={String(qi)} onChange={(v) => setQi(Number(v))} options={QUERIES.map((q, i) => ({ value: String(i), label: q.q }))} />
          <Segmented label="Ranking used" value={mode} onChange={setMode} options={[{ value: "bm25", label: "BM25" }, { value: "dense", label: "Dense" }, { value: "hybrid", label: "Hybrid" }]} />
          <ParameterSlider label="RRF constant k" value={kRrf} min={1} max={100} step={1} onChange={setKRrf} />
          <p className="text-[12px] leading-relaxed text-faint">BM25 matches exact words — perfect for codes like “E1042”, blind to synonyms. Dense retrieval matches meaning (“money back” ≈ refund) but can miss rare exact terms. Reciprocal-rank fusion combines the two rankings without needing comparable scores; a reranker can then re-score the fused top results. The dense scores come from hand-made concept vectors, not a trained embedding model.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{nDCG@5}", tex: true, value: fmt(ndcg, 3), color: ndcg > 0.9 ? C.green : undefined },
        { label: "\\text{relevant in top 5}", tex: true, value: `${final.slice(0, 5).filter((d) => rel.includes(d)).length} / ${Math.min(5, rel.length)}` },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{BM25}(q,d)=\sum_{t\in q}\text{IDF}(t)\frac{f_{t,d}(k_1+1)}{f_{t,d}+k_1\big(1-b+b\frac{|d|}{\overline{|d|}}\big)}` },
        { tex: String.raw`\text{RRF}(d)=\sum_{r\in\{\text{bm25},\text{dense}\}}\frac{1}{k+\text{rank}_r(d)}` },
      ]} />}
    />
  );
}

/* ------------------------------ prompt engineering & structured output validation ------------------------------ */
const SCHEMA = { name: "string", sentiment: ["positive", "neutral", "negative"], rating: "integer 1-5" } as const;
const CANDIDATES = [
  { label: "valid JSON", text: '{"name": "Aster Café", "sentiment": "positive", "rating": 4}' },
  { label: "JSON wrapped in prose", text: 'Sure! Here is the result:\n{"name": "Aster Café", "sentiment": "positive", "rating": 4}' },
  { label: "wrong type", text: '{"name": "Aster Café", "sentiment": "positive", "rating": "four"}' },
  { label: "invalid enum + missing field", text: '{"name": "Aster Café", "sentiment": "great"}' },
];
export function validateOutput(text: string) {
  const errs: string[] = [];
  let obj: Record<string, unknown> | null = null;
  try { obj = JSON.parse(text); } catch { errs.push("not valid JSON (extra text or syntax error)"); }
  if (obj && typeof obj === "object") {
    if (typeof obj.name !== "string") errs.push("name: expected string");
    if (!(SCHEMA.sentiment as readonly string[]).includes(obj.sentiment as string)) errs.push(`sentiment: expected one of ${SCHEMA.sentiment.join(" | ")}`);
    if (!Number.isInteger(obj.rating) || (obj.rating as number) < 1 || (obj.rating as number) > 5) errs.push("rating: expected integer 1–5");
    const extra = Object.keys(obj).filter((k) => !(k in SCHEMA)); if (extra.length) errs.push(`unexpected keys: ${extra.join(", ")}`);
  }
  return errs;
}
export function PromptViz() {
  const [shots, setShots] = useState(1);
  const [schema, setSchema] = useState(true);
  const [role, setRole] = useState(true);
  const [cand, setCand] = useState(0);
  const [custom, setCustom] = useState(CANDIDATES[0].text);
  const examples = [
    ["The pasta was cold and the waiter ignored us.", '{"name": "Luigi\'s", "sentiment": "negative", "rating": 1}'],
    ["Decent coffee, nothing special.", '{"name": "Bean Bar", "sentiment": "neutral", "rating": 3}'],
    ["Best ramen I have had in years!", '{"name": "Kumo", "sentiment": "positive", "rating": 5}'],
  ];
  const prompt = [
    role ? "System: You extract structured data from restaurant reviews. Reply with JSON only — no prose." : "",
    schema ? 'Schema: {"name": string, "sentiment": "positive" | "neutral" | "negative", "rating": integer 1-5}' : "",
    ...examples.slice(0, shots).flatMap(([r, o]) => [`Review: ${r}`, `Output: ${o}`]),
    "Review: Aster Café has lovely pastries, a little pricey but worth it.",
    "Output:",
  ].filter(Boolean).join("\n");
  const tokens = Math.ceil(prompt.length / 4);
  const errs = validateOutput(custom);
  return (
    <LabShell
      wideControls
      viz={
        <div className="grid gap-3 lg:grid-cols-2">
          <div className="space-y-1"><div className="text-[12px] text-faint">Assembled prompt (≈ {tokens} tokens)</div><pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-raised p-2.5 text-[12px] text-muted">{prompt}</pre></div>
          <div className="space-y-2">
            <div className="text-[12px] text-faint">Model output (edit it) → validated against the schema</div>
            <textarea value={custom} onChange={(e) => setCustom(e.target.value)} rows={4} aria-label="model output to validate" className="num w-full rounded-lg border border-line bg-raised p-2.5 text-[12px] text-fg outline-none focus:border-line-strong" />
            <div className={cn("rounded-lg border px-3 py-2 text-[12.5px]", errs.length ? "border-red/50" : "border-green/50")}>{errs.length ? <ul className="list-disc space-y-0.5 pl-4">{errs.map((e) => <li key={e}>{e}</li>)}</ul> : "✓ valid: safe to pass to downstream code"}</div>
            {errs.length > 0 && <p className="text-[12px] text-faint">Typical fix: retry with the validation error appended to the prompt, or use the API&apos;s constrained / JSON-schema decoding so invalid tokens can never be produced.</p>}
          </div>
        </div>
      }
      controls={
        <>
          <Toggle label="System instruction (role + format rule)" checked={role} onChange={setRole} />
          <Toggle label="Explicit schema" checked={schema} onChange={setSchema} />
          <ParameterSlider label="few-shot examples" value={shots} min={0} max={3} step={1} onChange={setShots} />
          <Select label="Load an example output" value={String(cand)} onChange={(v) => { setCand(Number(v)); setCustom(CANDIDATES[Number(v)].text); }} options={CANDIDATES.map((c, i) => ({ value: String(i), label: c.label }))} />
          <p className="text-[12px] leading-relaxed text-faint">Clear instructions, an explicit schema and a few worked examples make the desired format far more likely — but never guaranteed. Production systems therefore always parse and validate model output before using it.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{prompt tokens (≈ chars/4)}", tex: true, value: tokens },
        { label: "\\text{validation}", tex: true, value: errs.length ? `${errs.length} error(s)` : "pass", color: errs.length ? C.red : C.green },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{prompt}=\text{instruction}\ \|\ \text{schema}\ \|\ \underbrace{(x_1,y_1),\dots,(x_k,y_k)}_{k=${shots}\text{ shots}}\ \|\ x_{\text{new}}` }]} />}
    />
  );
}

/* ------------------------------ tool calling / agent loop ------------------------------ */
const CITY: Record<string, number> = { lisbon: 545_000, porto: 232_000, oslo: 709_000, kyoto: 1_464_000 };
function calc(expr: string): number {
  // tiny safe arithmetic parser: + − × ÷, parentheses, decimals (no eval)
  let i = 0; const s = expr.replace(/\s+/g, "");
  const num = (): number => { if (s[i] === "(") { i++; const v = add(); i++; return v; } const j = i; while (/[0-9.]/.test(s[i] ?? "")) i++; return parseFloat(s.slice(j, i)); };
  const mul = (): number => { let v = num(); while (s[i] === "*" || s[i] === "/") { const op = s[i++]; const r = num(); v = op === "*" ? v * r : v / r; } return v; };
  const add = (): number => { let v = mul(); while (s[i] === "+" || s[i] === "-") { const op = s[i++]; const r = mul(); v = op === "+" ? v + r : v - r; } return v; };
  return add();
}
type ToolCall = { tool: string; args: Record<string, string> };
const TASKS: { q: string; plan: ToolCall[] }[] = [
  { q: "What is 17% of the population of Lisbon?", plan: [{ tool: "lookup_population", args: { city: "lisbon" } }, { tool: "calculator", args: { expr: "545000*0.17" } }] },
  { q: "How many more people live in Oslo than in Porto?", plan: [{ tool: "lookup_population", args: { city: "oslo" } }, { tool: "lookup_population", args: { city: "porto" } }, { tool: "calculator", args: { expr: "709000-232000" } }] },
  { q: "What is the population of Atlantis?", plan: [{ tool: "lookup_population", args: { city: "atlantis" } }] },
];
export function ToolCallingViz() {
  const [ti, setTi] = useState(0);
  const [step, setStep] = useState(0);
  const [maxSteps, setMaxSteps] = useState(5);
  const task = TASKS[ti];
  const exec = (c: ToolCall) => {
    if (c.tool === "calculator") return { ok: true, result: calc(c.args.expr) };
    const v = CITY[c.args.city]; return v === undefined ? { ok: false, result: `no data for '${c.args.city}'` } : { ok: true, result: v };
  };
  const calls = task.plan.slice(0, maxSteps).map((c) => ({ c, r: exec(c) }));
  const failed = calls.find((x) => !x.r.ok);
  const last = calls[calls.length - 1];
  const answer = failed ? `I could not find that: the lookup tool returned ${failed.r.result}. I won't guess a number.` : task.plan.length > maxSteps ? "Stopped: step limit reached before finishing." : `The answer is ${Number(last.r.result).toLocaleString("en-US")}.`;
  const msgs: { who: string; text: string }[] = [{ who: "user", text: task.q }];
  calls.forEach((x) => { msgs.push({ who: "assistant → tool call", text: JSON.stringify({ name: x.c.tool, arguments: x.c.args }) }); msgs.push({ who: `tool (${x.c.tool})`, text: JSON.stringify(x.r.ok ? { result: x.r.result } : { error: x.r.result }) }); });
  msgs.push({ who: "assistant", text: answer });
  const shown = msgs.slice(0, step + 1);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-1.5">
          {shown.map((m, k) => <div key={k} className={cn("rounded-lg border border-line px-3 py-2 text-[12.5px]", m.who.startsWith("tool") ? "bg-raised" : m.who.startsWith("assistant") ? "bg-blue/5" : "")}><div className="text-[11px] text-faint">{m.who}</div><div className={cn(m.who.includes("tool") ? "num text-muted" : "text-fg")}>{m.text}</div></div>)}
          <div className="text-[12px] text-faint">Tools available: lookup_population(city) — reads a small local table; calculator(expr) — a safe arithmetic parser. The tools really run; the model&apos;s decisions are scripted for this demo.</div>
        </div>
      }
      controls={
        <>
          <Select label="Task" value={String(ti)} onChange={(v) => { setTi(Number(v)); setStep(0); }} options={TASKS.map((t, i) => ({ value: String(i), label: t.q }))} />
          <ParameterSlider label="max tool calls (loop guard)" value={maxSteps} min={1} max={5} step={1} onChange={(v) => { setMaxSteps(v); setStep(0); }} />
          <AnimationControls playing={false} onPlay={() => setStep(msgs.length - 1)} onStep={() => setStep((s) => Math.min(msgs.length - 1, s + 1))} onReset={() => setStep(0)} canStep={step < msgs.length - 1} />
          <p className="text-[12px] leading-relaxed text-faint">The model is given tool descriptions (name, purpose, JSON parameter schema). Instead of answering directly it may emit a structured tool call; the application executes it and returns the result as a new message; the loop repeats until the model answers. A step limit prevents runaway loops, and tool errors must be surfaced, not papered over.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{tool calls}", tex: true, value: calls.length },
        { label: "\\text{messages}", tex: true, value: `${shown.length} / ${msgs.length}` },
        { label: "\\text{outcome}", tex: true, value: failed ? "tool error handled" : task.plan.length > maxSteps ? "step limit" : "answered", color: failed ? C.amber : C.green },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\textbf{loop: } m\leftarrow\text{LLM}(\text{history});\ \text{if }m\text{ is a tool call: history}\mathrel{+}=[m,\ \text{run}(m)];\ \text{else return }m` }]} />}
    />
  );
}

/* ------------------------------ evaluating LLM outputs ------------------------------ */
const QA = [
  { q: "When did the bridge open?", ctx: "The Vasco da Gama bridge opened in 1998 and is 12.3 km long.", ref: "1998", ans: "It opened in 1998." },
  { q: "How long is the bridge?", ctx: "The Vasco da Gama bridge opened in 1998 and is 12.3 km long.", ref: "12.3 km", ans: "The bridge is about 17 km long." },
  { q: "Who designed the tower?", ctx: "The tower was designed by Gustave Eiffel's company and finished in 1889.", ref: "Gustave Eiffel's company", ans: "Gustave Eiffel's company designed it." },
  { q: "What colour is the logo?", ctx: "The company logo has been blue since 2010.", ref: "blue", ans: "The logo is green and white." },
  { q: "What year was the tower finished?", ctx: "The tower was designed by Gustave Eiffel's company and finished in 1889.", ref: "1889", ans: "1889" },
  { q: "What is the refund window?", ctx: "Customers can request a refund within 30 days of purchase.", ref: "30 days", ans: "You have 30 days, and an extra 14 days if you ask nicely." },
];
const words = (s: string) => s.toLowerCase().replace(/[^a-z0-9.' ]/g, " ").split(/\s+/).filter((w) => w && !["the", "a", "is", "it", "in", "of", "and", "you", "have", "if", "an", "about"].includes(w)).map((w) => w.replace(/\.$/, ""));
export function LLMEvalViz() {
  const [thr, setThr] = useState(0.7);
  const [strict, setStrict] = useState(false);
  const rows = QA.map((x) => {
    const a = words(x.ans), r = words(x.ref), c = new Set(words(x.ctx));
    const em = words(x.ans).join(" ") === r.join(" ") ? 1 : 0;
    const common = a.filter((w) => r.includes(w)).length;
    const p = a.length ? common / a.length : 0, rc = r.length ? r.filter((w) => a.includes(w)).length / r.length : 0;
    const f1 = p + rc ? (2 * p * rc) / (p + rc) : 0;
    const grounded = a.length ? a.filter((w) => c.has(w)).length / a.length : 1;
    const nums = (x.ans.match(/\d+(\.\d+)?/g) ?? []).filter((n) => !x.ctx.includes(n));
    const flag = grounded < thr || nums.length > 0;
    return { ...x, em, f1, grounded, nums, flag };
  });
  const agg = (k: "em" | "f1" | "grounded") => rows.reduce((s, r) => s + r[k], 0) / rows.length;
  return (
    <LabShell
      wideControls
      viz={
        <MiniTable head={["question", "answer", "EM", "token F1", "grounded", "guardrail"]} rows={rows.map((r) => [r.q, r.ans, r.em, fmt(r.f1, 2), pct(r.grounded), r.flag ? `⚠ ${r.nums.length ? `unsupported number ${r.nums.join(", ")}` : "low support"}` : "ok"])} rowClass={(i) => (rows[i].flag ? "bg-red/10" : undefined)} />
      }
      controls={
        <>
          <ParameterSlider label="groundedness threshold" value={thr} min={0.3} max={1} step={0.05} onChange={setThr} format={(v) => pct(v)} />
          <Toggle label="Block flagged answers (strict guardrail)" checked={strict} onChange={setStrict} />
          <p className="text-[12px] leading-relaxed text-faint">Exact match punishes correct paraphrases; token F1 gives partial credit. Neither checks faithfulness to the source, so a groundedness score measures how much of the answer is supported by the retrieved context, and a simple guardrail flags numbers that never appear in it — a common form of hallucination. Production systems add LLM-as-judge and human review on samples.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{exact match}", tex: true, value: pct(agg("em")) },
        { label: "\\text{mean token F1}", tex: true, value: fmt(agg("f1"), 3) },
        { label: "\\text{mean groundedness}", tex: true, value: pct(agg("grounded")) },
        { label: strict ? "\\text{answers blocked}" : "\\text{answers flagged}", tex: true, value: `${rows.filter((r) => r.flag).length} / ${rows.length}`, color: C.red },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`F_1=\frac{2PR}{P+R},\quad P=\frac{|\text{ans}\cap\text{ref}|}{|\text{ans}|},\ R=\frac{|\text{ans}\cap\text{ref}|}{|\text{ref}|}` },
        { tex: String.raw`\text{grounded}=\frac{|\{w\in\text{ans}: w\in\text{context}\}|}{|\text{ans}|}\ \ge${tex(thr, 2)}\ \text{to pass}` },
      ]} />}
    />
  );
}

/* ------------------------------ context windows ------------------------------ */
export function ContextWindowViz() {
  const [win, setWin] = useState(8192);
  const [turns, setTurns] = useState(20);
  const [docs, setDocs] = useState(5);
  const [strategy, setStrategy] = useState<"oldest" | "summarize" | "fewer-docs">("oldest");
  const sys = 600, perTurn = 180, chunk = 500, reserve = 1000;
  let hist = turns * perTurn, rag = docs * chunk;
  const need = sys + hist + rag + reserve;
  let dropped = 0;
  if (need > win) {
    const over = need - win;
    if (strategy === "oldest") { const cut = Math.min(hist, Math.ceil(over / perTurn) * perTurn); hist -= cut; dropped = cut / perTurn; }
    else if (strategy === "summarize") { const keep = Math.max(0, hist - over); const summary = Math.min(hist, Math.round((hist - keep) * 0.15)); hist = keep + summary; dropped = 0; }
    else { const cut = Math.min(rag, Math.ceil(over / chunk) * chunk); rag -= cut; dropped = cut / chunk; }
  }
  const used = Math.min(win, sys + hist + rag + reserve);
  const overflow = sys + hist + rag + reserve > win;
  const parts = [{ label: "system prompt", v: sys, color: "rgb(var(--amber))" }, { label: "conversation", v: hist, color: "rgb(var(--blue))" }, { label: "retrieved documents", v: rag, color: "rgb(var(--green))" }, { label: "reserved for the answer", v: reserve, color: "rgb(var(--pink))" }];
  // a 7B-style config: 32 layers, d = 4096, fp16 → KV cache bytes = 2 · L · n · d · 2
  const kv = 2 * 32 * used * 4096 * 2;
  const attn = 32 * used * used * 4096 * 2; // QKᵀ and AV FLOPs per layer ≈ 2·n²·d
  const pricePerM = 3;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div className="flex h-9 w-full overflow-hidden rounded-lg border border-line">
            {parts.map((p) => <div key={p.label} title={`${p.label}: ${p.v} tokens`} style={{ width: `${(p.v / win) * 100}%`, background: p.color, opacity: 0.75 }} />)}
          </div>
          <div className="flex justify-between text-[11px] text-faint"><span>0</span><span>{win.toLocaleString("en-US")} tokens</span></div>
          <Legend items={parts.map((p) => ({ label: `${p.label} (${p.v.toLocaleString("en-US")})`, color: p.color }))} />
          {(dropped > 0 || (strategy === "summarize" && need > win)) && <div className="rounded-lg border border-amber/50 px-3 py-2 text-[12.5px]">Needed {need.toLocaleString("en-US")} tokens for a {win.toLocaleString("en-US")}-token window: {strategy === "oldest" ? `dropped the ${dropped} oldest turns` : strategy === "summarize" ? "older turns compressed into a summary (~15 % of their length)" : `kept ${docs - dropped} of ${docs} retrieved documents`}.</div>}
          {overflow && <div className="rounded-lg border border-red/50 px-3 py-2 text-[12.5px]">Still over the limit — the request would be rejected or truncated by the API.</div>}
        </div>
      }
      controls={
        <>
          <Select label="Context window" value={String(win)} onChange={(v) => setWin(Number(v))} options={[4096, 8192, 32768, 131072].map((w) => ({ value: String(w), label: `${w / 1024}k tokens` }))} />
          <ParameterSlider label="conversation turns (≈180 tokens each)" value={turns} min={0} max={200} step={1} onChange={setTurns} />
          <ParameterSlider label="retrieved chunks (500 tokens each)" value={docs} min={0} max={40} step={1} onChange={setDocs} />
          <Segmented label="When it does not fit" value={strategy} onChange={setStrategy} options={[{ value: "oldest", label: "Drop oldest" }, { value: "summarize", label: "Summarise" }, { value: "fewer-docs", label: "Fewer docs" }]} />
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{tokens used}", tex: true, value: `${used.toLocaleString("en-US")} (${pct(used / win)})` },
        { label: "\\text{KV cache (32 layers, d=4096, fp16)}", tex: true, value: `${fmt(kv / 2 ** 30, 2)} GiB` },
        { label: "\\text{attention FLOPs / forward}", tex: true, value: `${fmt(attn / 1e12, 2)} TFLOP` },
        { label: "\\text{input cost at \\$3 / 1M tokens}", tex: true, value: `$${fmt((used - reserve) * pricePerM / 1e6, 4)}` },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{KV cache}=2\cdot L\cdot n\cdot d\cdot\text{bytes}=2\cdot32\cdot${used}\cdot4096\cdot2`, note: "grows linearly with context length n" },
        { tex: String.raw`\text{attention}\approx L\cdot2n^2d\ \Rightarrow\ \text{doubling }n\text{ quadruples attention compute}` },
      ]} />}
    />
  );
}
