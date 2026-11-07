"use client";
import { useMemo, useState } from "react";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Select } from "@/components/controls";
import { cosineSimilarity } from "@/lib/math/linalg";
import { fmt, cn } from "@/lib/utilities/format";

const DOCS = [
  "The Transformer architecture relies entirely on self-attention, dispensing with recurrence. Multi-head attention lets the model attend to different representation subspaces. Positional encodings inject order information.",
  "Gradient descent updates parameters in the direction of the negative gradient. The learning rate controls the step size. Adam combines momentum with per-parameter adaptive learning rates.",
  "Principal component analysis finds orthogonal directions of maximum variance. The principal components are eigenvectors of the covariance matrix. Explained variance ratios guide how many components to keep.",
  "Retrieval-augmented generation retrieves relevant passages from a document store and adds them to the prompt. This grounds the language model's answer in external knowledge and reduces hallucination.",
  "Overfitting happens when a model memorises training noise. Regularisation such as L2 weight decay, dropout and early stopping improves generalisation to unseen data.",
];

const STOP = new Set("the a an of and to in on is are with by for from this that it its as be or".split(" "));
const tokenize = (s: string) => s.toLowerCase().replace(/[^a-z0-9\s-]/g, " ").split(/\s+/).filter((w) => w && !STOP.has(w));
const stem = (w: string) => w.replace(/(ations|ation|ing|ies|es|s)$/, "");

/** Deterministic bag-of-words "embedding": hashed term counts in 64 dims (a stand-in for a neural embedding model). */
const DIM = 64;
function embed(text: string): number[] {
  const v = Array(DIM).fill(0);
  for (const w of tokenize(text).map(stem)) {
    let h = 2166136261;
    for (const ch of w) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); }
    v[(h >>> 0) % DIM] += 1;
  }
  return v;
}

function chunk(doc: string, mode: "sentence" | "fixed", size: number, overlap: number): string[] {
  if (mode === "sentence") return doc.split(/(?<=\.)\s+/).filter(Boolean);
  const words = doc.split(/\s+/);
  const out: string[] = [];
  for (let i = 0; i < words.length; i += Math.max(1, size - overlap)) { out.push(words.slice(i, i + size).join(" ")); if (i + size >= words.length) break; }
  return out;
}

export default function RAG({ focus = "pipeline" }: { focus?: "pipeline" | "chunking" }) {
  const [query, setQuery] = useState("How does attention handle word order?");
  const [mode, setMode] = useState<"sentence" | "fixed">(focus === "chunking" ? "fixed" : "sentence");
  const [size, setSize] = useState(14);
  const [overlap, setOverlap] = useState(4);
  const [k, setK] = useState(3);
  const [retr, setRetr] = useState<"dense" | "keyword" | "hybrid">("dense");
  const [stage, setStage] = useState(4);
  const chunks = useMemo(() => DOCS.flatMap((d, di) => chunk(d, mode, size, overlap).map((t) => ({ t, di }))), [mode, size, overlap]);
  const qv = embed(query);
  const qTerms = new Set(tokenize(query).map(stem));
  const scored = chunks.map((c, i) => {
    const dense = cosineSimilarity(qv, embed(c.t)) ?? 0;
    const terms = tokenize(c.t).map(stem);
    const kw = terms.filter((w) => qTerms.has(w)).length / Math.sqrt(terms.length || 1);
    return { ...c, i, dense, kw, score: retr === "dense" ? dense : retr === "keyword" ? kw : 0.5 * dense + 0.5 * Math.min(1, kw) };
  }).sort((a, b) => b.score - a.score);
  const top = scored.slice(0, k);
  const prompt = `Answer using only the context.\n\nContext:\n${top.map((c, i) => `[${i + 1}] ${c.t}`).join("\n")}\n\nQuestion: ${query}`;
  const STAGES = ["Documents", "Chunking", "Embedding", "Retrieval", "Augmented prompt"];
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-4">
          <ol className="flex flex-wrap items-center gap-1 text-[12px]">
            {STAGES.map((s, i) => (
              <li key={s} className="flex items-center gap-1">
                <button className={cn("rounded-md border px-2 py-1", i === stage ? "border-blue bg-blue/10" : "border-line text-muted hover:text-fg", i < stage && "text-fg")} onClick={() => setStage(i)}>{i + 1}. {s}</button>
                {i < STAGES.length - 1 && <span className="text-faint">→</span>}
              </li>
            ))}
          </ol>
          <div className="max-h-[420px] overflow-y-auto rounded-md border border-line p-3">
            {stage === 0 && <ol className="space-y-2 text-[12.5px] text-muted">{DOCS.map((d, i) => <li key={i}><span className="num mr-2 text-faint">doc {i + 1}</span>{d}</li>)}</ol>}
            {stage === 1 && <div className="flex flex-wrap gap-1.5">{chunks.map((c, i) => <span key={i} className="rounded border border-line px-2 py-1 text-[11.5px]" style={{ background: `rgb(var(--${["blue", "pink", "green", "amber", "blue"][c.di]}) / 0.12)` }}>{c.t}</span>)}</div>}
            {stage === 2 && (
              <div className="space-y-1">
                {[{ t: query, q: true }, ...chunks.slice(0, 8).map((c) => ({ t: c.t, q: false }))].map((c, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className={cn("w-40 shrink-0 truncate text-[11px]", c.q ? "text-pink" : "text-muted")}>{c.q ? "query" : c.t}</span>
                    <div className="grid flex-1 gap-[1px]" style={{ gridTemplateColumns: `repeat(${DIM}, minmax(0,1fr))` }}>
                      {embed(c.t).map((v, j) => <span key={j} className="h-3" style={{ background: v ? `rgb(var(--${c.q ? "pink" : "blue"}) / ${Math.min(1, v * 0.5)})` : "rgb(var(--line) / 0.4)" }} />)}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {stage === 3 && <ol className="space-y-1.5">{scored.slice(0, 10).map((c, r) => <li key={c.i} className={cn("flex gap-3 rounded px-2 py-1 text-[12px]", r < k ? "bg-green/10" : "opacity-60")}><span className="num w-12 shrink-0 text-faint">{fmt(c.score, 3)}</span><span>{c.t}</span></li>)}</ol>}
            {stage === 4 && <pre className="whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-muted">{prompt}</pre>}
          </div>
        </div>
      }
      controls={
        <>
          <label className="block space-y-1">
            <span className="text-[12.5px] text-muted">Query</span>
            <input value={query} onChange={(e) => setQuery(e.target.value)} className="h-8 w-full rounded-md border border-line bg-raised px-2 text-[13px] outline-none focus:border-blue" />
          </label>
          <Select label="Example queries" value="" onChange={(v) => v && setQuery(v)} options={[{ value: "", label: "Choose…" }, { value: "Why does Adam adapt the learning rate?", label: "Adam" }, { value: "How do I pick the number of principal components?", label: "PCA" }, { value: "How can I reduce hallucination?", label: "Hallucination" }, { value: "What prevents overfitting?", label: "Overfitting" }]} />
          <Segmented label="Chunking" value={mode} onChange={setMode} options={[{ value: "sentence", label: "By sentence" }, { value: "fixed", label: "Fixed words" }]} />
          {mode === "fixed" && <><ParameterSlider label="chunk size (words)" value={size} min={5} max={40} step={1} onChange={setSize} /><ParameterSlider label="overlap (words)" value={overlap} min={0} max={Math.max(0, size - 1)} step={1} onChange={setOverlap} /></>}
          <Segmented label="Retrieval" value={retr} onChange={setRetr} options={[{ value: "dense", label: "Dense" }, { value: "keyword", label: "Keyword" }, { value: "hybrid", label: "Hybrid" }]} />
          <ParameterSlider label="top-k chunks" value={k} min={1} max={6} step={1} onChange={setK} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{chunks}", tex: true, value: chunks.length }, { label: "\\text{best match from}", tex: true, value: `doc ${top[0].di + 1}` }, { label: "\\text{top score}", tex: true, value: fmt(top[0].score, 3) }, { label: "\\text{prompt length}", tex: true, value: `${prompt.split(/\s+/).length} words` }]} />}
      calc={<CalcSteps steps={[{ tex: "\\text{score}(q,c)=\\cos(\\mathbf e_q,\\mathbf e_c)=\\frac{\\mathbf e_q\\cdot\\mathbf e_c}{\\|\\mathbf e_q\\|\\|\\mathbf e_c\\|}", note: retr === "hybrid" ? "hybrid = 0.5 · dense + 0.5 · keyword overlap" : undefined }, { tex: "\\text{answer}=\\operatorname{LLM}(\\text{instructions}\\,\\|\\,\\text{top-}k\\text{ chunks}\\,\\|\\,\\text{question})" }]} />}
      note="Runs fully offline: the 'embedding' is a hashed bag-of-words vector, a deliberately simple stand-in that matches words, not meaning. Production RAG uses neural embedding models (so 'word order' matches 'positional encodings'), approximate nearest-neighbour indexes, rerankers and an actual LLM call for the final step, which is not performed here."
    />
  );
}
