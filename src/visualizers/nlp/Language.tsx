"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented, AnimationControls } from "@/components/controls";
import { Plot, Arrow, Dot, Label, C, Handle, Polyline } from "@/components/visualization/Plot";
import { useTicker } from "@/hooks/useSize";
import { cosineSimilarity } from "@/lib/math/linalg";
import { fmt, tex, cn } from "@/lib/utilities/format";

/* ------------------------------ BPE tokenizer ------------------------------ */
const CORPUS = "low lower lowest newer newest wider widest slow slower slowest";

export function trainBPE(corpus: string, merges: number) {
  const words = corpus.split(/\s+/).filter(Boolean);
  const freq = new Map<string, number>();
  for (const w of words) freq.set(w, (freq.get(w) ?? 0) + 1);
  let vocabWords = [...freq.entries()].map(([w, f]) => ({ toks: [...w, "▁"], f }));
  const history: { pair: [string, string]; count: number }[] = [];
  for (let m = 0; m < merges; m++) {
    const counts = new Map<string, number>();
    for (const { toks, f } of vocabWords) for (let i = 0; i < toks.length - 1; i++) { const k = toks[i] + "\u0000" + toks[i + 1]; counts.set(k, (counts.get(k) ?? 0) + f); }
    let best: string | null = null, bc = 0;
    for (const [k, c] of counts) if (c > bc || (c === bc && best !== null && k < best)) { best = k; bc = c; }
    if (!best || bc < 2) break;
    const [a, b] = best.split("\u0000") as [string, string];
    history.push({ pair: [a, b], count: bc });
    vocabWords = vocabWords.map(({ toks, f }) => {
      const out: string[] = [];
      for (let i = 0; i < toks.length; i++) {
        if (i < toks.length - 1 && toks[i] === a && toks[i + 1] === b) { out.push(a + b); i++; } else out.push(toks[i]);
      }
      return { toks: out, f };
    });
  }
  return { history, vocabWords };
}

export function applyBPE(word: string, history: { pair: [string, string] }[]) {
  let toks = [...word, "▁"];
  for (const { pair: [a, b] } of history) {
    const out: string[] = [];
    for (let i = 0; i < toks.length; i++) { if (i < toks.length - 1 && toks[i] === a && toks[i + 1] === b) { out.push(a + b); i++; } else out.push(toks[i]); }
    toks = out;
  }
  return toks;
}

export function Tokenizer() {
  const [m, setM] = useState(6);
  const [text, setText] = useState("the slowest newest widest lowly");
  const [scheme, setScheme] = useState<"bpe" | "word" | "char">("bpe");
  const { history } = useMemo(() => trainBPE(CORPUS, 30), []);
  const used = history.slice(0, m);
  const base = [...new Set(CORPUS.replace(/\s+/g, "").split(""))].concat("▁");
  const vocab = [...base, ...used.map((h) => h.pair[0] + h.pair[1])];
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const tokens = scheme === "word" ? words.map((w) => [w]) : scheme === "char" ? words.map((w) => [...w]) : words.map((w) => applyBPE(w, used));
  const id = (t: string) => { const i = vocab.indexOf(t); return i >= 0 ? i : -1; };
  const flat = tokens.flat();
  const PAL = ["blue", "pink", "green", "amber"];
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-4">
          <div className="flex flex-wrap gap-1.5">
            {tokens.map((ws, wi) => (
              <div key={wi} className="flex gap-[2px] rounded border border-line p-1">
                {ws.map((t, ti) => (
                  <span key={ti} className="num rounded px-1.5 py-0.5 text-[13px]" style={{ background: `rgb(var(--${PAL[(wi + ti) % 4]}) / 0.22)` }} title={scheme === "bpe" ? `id ${id(t)}` : undefined}>
                    {t}
                  </span>
                ))}
              </div>
            ))}
          </div>
          {scheme === "bpe" && (
            <div className="space-y-1">
              <div className="label">Learned merges (most frequent adjacent pair first)</div>
              <ol className="grid gap-1 sm:grid-cols-2">
                {history.slice(0, Math.max(m, 1) + 2).map((h, i) => (
                  <li key={i} className={cn("num flex items-center gap-2 rounded px-2 py-1 text-[12.5px]", i < m ? "bg-raised" : "opacity-40")}>
                    <span className="w-5 text-faint">{i + 1}</span>
                    <span>{h.pair[0]} + {h.pair[1]} → <span className="text-blue">{h.pair[0] + h.pair[1]}</span></span>
                    <span className="ml-auto text-faint">×{h.count}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
          <div className="text-[12px] text-faint">Training corpus: “{CORPUS}”. ▁ marks the end of a word.</div>
        </div>
      }
      controls={
        <>
          <Segmented value={scheme} onChange={setScheme} options={[{ value: "word", label: "Word" }, { value: "char", label: "Character" }, { value: "bpe", label: "BPE subword" }]} />
          {scheme === "bpe" && <ParameterSlider label="number of merges" value={m} min={0} max={history.length} step={1} onChange={setM} />}
          <label className="block space-y-1">
            <span className="text-[12.5px] text-muted">Text to tokenize</span>
            <input value={text} onChange={(e) => setText(e.target.value)} className="h-8 w-full rounded-md border border-line bg-raised px-2 text-[13px] outline-none focus:border-blue" />
          </label>
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{tokens}", tex: true, value: flat.length }, { label: "\\text{vocabulary size}", tex: true, value: scheme === "bpe" ? vocab.length : scheme === "char" ? "all characters" : "every word (unbounded)" }, { label: "\\text{tokens / word}", tex: true, value: fmt(flat.length / Math.max(1, words.length), 2) }, { label: "\\text{unknown}", tex: true, value: scheme === "bpe" ? flat.filter((t) => id(t) < 0).length : "—" }]} />}
      calc={<CalcSteps steps={[{ tex: "\\text{repeat: count adjacent symbol pairs (weighted by word frequency); merge the most frequent pair into a new symbol}" }, ...(used.length ? [{ tex: `\\text{merge }${used.length}:\\ (\\texttt{${used[used.length - 1].pair[0]}},\\texttt{${used[used.length - 1].pair[1]}})\\to\\texttt{${used[used.length - 1].pair[0] + used[used.length - 1].pair[1]}}` }] : [])]} />}
      note="BPE trades vocabulary size against sequence length: frequent words become single tokens, rare words split into reusable pieces (so nothing is ever out-of-vocabulary). WordPiece chooses merges by likelihood gain instead of raw frequency; SentencePiece runs BPE or unigram LM directly on raw text including spaces."
    />
  );
}

/* ------------------------------ embedding space ------------------------------ */
// Hand-crafted 2-D illustrative embeddings (NOT from a trained model) with an exact gender/royalty structure.
const EMB: Record<string, [number, number]> = {
  king: [2.2, 2.0], queen: [2.2, -0.4], man: [0.4, 1.9], woman: [0.4, -0.5], prince: [1.6, 1.7], princess: [1.6, -0.7],
  boy: [-0.4, 1.6], girl: [-0.4, -0.8], apple: [-2.2, 0.3], banana: [-2.5, 0.8], orange: [-2.0, 0.9], car: [0.8, 3.2], truck: [1.2, 3.5],
};

export function Embeddings({ mode = "similarity" }: { mode?: "similarity" | "analogy" }) {
  const [view, setView] = useState(mode);
  const [q, setQ] = useState<[number, number]>([1.8, 1.2]);
  const [a, setA] = useState("king"); const [b, setB] = useState("man"); const [c, setC] = useState("woman");
  const words = Object.keys(EMB);
  const sims = words.map((w) => ({ w, s: cosineSimilarity(q, EMB[w]) ?? 0, d: Math.hypot(q[0] - EMB[w][0], q[1] - EMB[w][1]) })).sort((x, y) => y.s - x.s);
  const target: [number, number] = [EMB[a][0] - EMB[b][0] + EMB[c][0], EMB[a][1] - EMB[b][1] + EMB[c][1]];
  const best = words.filter((w) => ![a, b, c].includes(w)).map((w) => ({ w, s: cosineSimilarity(target, EMB[w]) ?? 0 })).sort((x, y) => y.s - x.s);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3.2, 3.2]} y={[-1.6, 4]} equal aspect={0.72} maxH={460} originAxes ariaLabel="2D word embedding space">
            {words.map((w) => (
              <g key={w}>
                <Dot p={EMB[w]} r={4} color={C.a("fg", 0.7)} />
                <Label p={EMB[w]} dx={6} dy={-5} color={C.muted}>{w}</Label>
              </g>
            ))}
            {view === "similarity" ? (
              <>
                {sims.slice(0, 3).map((s) => <Arrow key={s.w} from={[0, 0]} to={EMB[s.w]} color="green" width={1.4} dashed />)}
                <Arrow from={[0, 0]} to={q} color="pink" width={2.6} label="query" />
                <Handle p={q} onDrag={(p) => setQ([Math.max(-3, Math.min(3, p[0])), Math.max(-1.5, Math.min(3.9, p[1]))])} color="pink" label="query vector" />
              </>
            ) : (
              <>
                <Arrow from={EMB[b]} to={EMB[a]} color="blue" width={2} label={`${a} − ${b}`} />
                <Arrow from={EMB[c]} to={target} color="blue" width={2} dashed />
                <Dot p={target} r={6} color={C.pink} />
                <Label p={target} dx={8} dy={14} color={C.pink}>{`${a} − ${b} + ${c}`}</Label>
              </>
            )}
          </Plot>
          <Legend items={view === "similarity" ? [{ label: "query (drag)", color: C.pink }, { label: "3 most cosine-similar words", color: C.green, dashed: true }] : [{ label: "offset vector reused", color: C.blue }, { label: "result vector", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <Segmented value={view} onChange={setView} options={[{ value: "similarity", label: "Cosine similarity" }, { value: "analogy", label: "Vector analogy" }]} />
          {view === "analogy" && (
            <>
              <Select label="A" value={a} onChange={setA} options={words.map((w) => ({ value: w, label: w }))} />
              <Select label="minus B" value={b} onChange={setB} options={words.map((w) => ({ value: w, label: w }))} />
              <Select label="plus C" value={c} onChange={setC} options={words.map((w) => ({ value: w, label: w }))} />
            </>
          )}
          {view === "similarity" && (
            <div className="space-y-1">
              <div className="label">Nearest by cosine</div>
              {sims.slice(0, 6).map((s) => <div key={s.w} className="flex justify-between text-[12.5px]"><span>{s.w}</span><span className="num text-muted">{fmt(s.s, 3)}</span></div>)}
            </div>
          )}
        </>
      }
      readouts={<Readouts items={view === "similarity" ? [{ label: "\\text{top match}", tex: true, value: sims[0].w }, { label: "\\cos\\theta", tex: true, value: fmt(sims[0].s, 4) }, { label: "\\text{Euclidean nearest}", tex: true, value: [...sims].sort((x, y) => x.d - y.d)[0].w }] : [{ label: "\\text{result vector}", tex: true, value: `(${fmt(target[0], 2)}, ${fmt(target[1], 2)})` }, { label: "\\text{nearest word}", tex: true, value: best[0].w, color: C.pink }, { label: "\\cos", tex: true, value: fmt(best[0].s, 4) }]} />}
      calc={<CalcSteps steps={view === "similarity" ? [{ tex: `\\cos(\\mathbf q,\\mathbf e_{\\text{${sims[0].w}}})=\\frac{\\mathbf q\\cdot\\mathbf e}{\\|\\mathbf q\\|\\|\\mathbf e\\|}=${tex(sims[0].s, 4)}` }] : [{ tex: `\\mathbf e_{\\text{${a}}}-\\mathbf e_{\\text{${b}}}+\\mathbf e_{\\text{${c}}}=(${tex(target[0], 2)},${tex(target[1], 2)})\\approx\\mathbf e_{\\text{${best[0].w}}}` }]} />}
      note="These 2-D vectors are hand-placed for illustration — real embeddings have hundreds of dimensions and are learned (Word2Vec, GloVe, or a Transformer's embedding layer). Cosine similarity ignores length and is the standard metric for retrieval."
    />
  );
}

/* ------------------------------ positional encoding ------------------------------ */
export function PositionalEncoding() {
  const [d, setD] = useState(32);
  const [L, setL] = useState(48);
  const [p1, setP1] = useState(10);
  const ref = useRef<HTMLCanvasElement>(null);
  const PE = useMemo(() => Array.from({ length: L }, (_, pos) => Array.from({ length: d }, (_, i) => { const k = Math.floor(i / 2); const w = 1 / Math.pow(10000, (2 * k) / d); return i % 2 === 0 ? Math.sin(pos * w) : Math.cos(pos * w); })), [d, L]);
  useEffect(() => {
    const g = ref.current?.getContext("2d");
    if (!g) return;
    const im = g.createImageData(d, L);
    for (let p = 0; p < L; p++) for (let i = 0; i < d; i++) {
      const v = PE[p][i], k = (p * d + i) * 4;
      im.data[k] = v < 0 ? 255 * -v : 20; im.data[k + 1] = 40 + 40 * Math.abs(v); im.data[k + 2] = v > 0 ? 255 * v : 20; im.data[k + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }, [PE, d, L]);
  const dotWith = PE.map((row) => row.reduce((s, v, i) => s + v * PE[p1][i], 0));
  return (
    <LabShell
      viz={
        <div className="grid gap-4 md:grid-cols-2">
          <figure className="space-y-1">
            <canvas ref={ref} width={d} height={L} className="h-[300px] w-full rounded border border-line" style={{ imageRendering: "pixelated" }} role="img" aria-label="positional encoding matrix heatmap" />
            <figcaption className="text-[12px] text-faint">rows = positions 0…{L - 1}, columns = dimensions 0…{d - 1} (blue +1, red −1)</figcaption>
          </figure>
          <div>
            <div className="mb-1 text-[11.5px] text-faint">PE(p)·PE({p1}) — similarity decays with distance |p − {p1}|</div>
            <Plot x={[0, L - 1]} y={[Math.min(...dotWith) - 1, d / 2 + 1]} aspect={0.8} maxH={300} ariaLabel="dot product of positional encodings">
              <Polyline pts={dotWith.map((v, p) => [p, v])} style={{ stroke: C.blue, strokeWidth: 1.8 }} />
              <Dot p={[p1, dotWith[p1]]} r={4} color={C.pink} />
            </Plot>
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="model dimension d" value={d} min={8} max={128} step={8} onChange={setD} />
          <ParameterSlider label="sequence length" value={L} min={16} max={128} step={8} onChange={(v) => { setL(v); setP1(Math.min(p1, v - 1)); }} />
          <ParameterSlider label="reference position" value={p1} min={0} max={L - 1} step={1} onChange={setP1} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\|PE(p)\\|^2", tex: true, value: fmt(d / 2) }, { label: "\\text{fastest wavelength}", tex: true, value: fmt(2 * Math.PI, 2) }, { label: "\\text{slowest wavelength}", tex: true, value: fmt(2 * Math.PI * Math.pow(10000, (d - 2) / d), 0) }]} />}
      calc={<CalcSteps steps={[{ tex: "PE_{(p,2k)}=\\sin\\!\\Big(\\frac{p}{10000^{2k/d}}\\Big),\\quad PE_{(p,2k+1)}=\\cos\\!\\Big(\\frac{p}{10000^{2k/d}}\\Big)" }, { tex: "PE(p+\\Delta)=R_\\Delta\\,PE(p)", note: "a fixed rotation per frequency pair: relative offsets are linear functions" }]} />}
      note="Self-attention is permutation-invariant, so order must be injected. Sinusoidal encodings (original Transformer) are added to token embeddings; modern LLMs often use rotary embeddings (RoPE) that rotate queries and keys instead."
    />
  );
}

export { AnimationControls, useTicker };
