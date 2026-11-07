"use client";
import { useMemo, useState } from "react";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented } from "@/components/controls";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { forwardTrace, makeWeights, tokenize, VOCAB, D, DH, FF } from "@/lib/nn/transformer";
import { fmt, tex, cn } from "@/lib/utilities/format";

const STAGES = [
  "Input text", "Tokenization", "Token IDs", "Embeddings", "Positional information", "Q, K, V projections", "Attention scores", "Softmax weights",
  "Weighted values", "Multi-head concat", "Residual connection", "Feed-forward", "Final logits", "Next-token probabilities", "Autoregressive generation",
];

export default function TransformerExplorer() {
  const W = useMemo(() => makeWeights(42), []);
  const [text, setText] = useState("the cat sat on");
  const [stage, setStage] = useState(0);
  const [head, setHead] = useState(0);
  const [temp, setTemp] = useState(1);
  const [genSteps, setGenSteps] = useState(4);
  const [strategy, setStrategy] = useState<"greedy" | "sample">("greedy");
  const ids = useMemo(() => tokenize(text).slice(0, 8), [text]);
  const T = useMemo(() => forwardTrace(ids, W, temp), [ids, W, temp]);
  const gen = useMemo(() => {
    let cur = ids.slice();
    const out: { tok: string; p: number }[] = [];
    let seed = 7;
    for (let s = 0; s < genSteps && cur.length < 14; s++) {
      const tr = forwardTrace(cur, W, temp);
      let next: number;
      if (strategy === "greedy") next = tr.probs.indexOf(Math.max(...tr.probs));
      else { seed = (seed * 16807) % 2147483647; let u = seed / 2147483647; next = tr.probs.findIndex((p) => (u -= p) <= 0); if (next < 0) next = tr.probs.length - 1; }
      out.push({ tok: VOCAB[next], p: tr.probs[next] });
      cur = [...cur, next];
    }
    return out;
  }, [ids, W, temp, genSteps, strategy]);
  const H = T.heads[head];
  const n = ids.length;
  const top = T.probs.map((p, i) => ({ w: VOCAB[i], p, l: T.logits[i] })).sort((a, b) => b.p - a.p);
  const unknown = text.toLowerCase().replace(/\./g, " .").split(/\s+/).filter((w) => w && VOCAB.indexOf(w) <= 0);
  let body: React.ReactNode = null;
  let steps: { tex: string; note?: string }[] = [];
  const rows = T.tokens.map((t, i) => `${i}:${t}`);
  const withRows = (M: number[][], label: string, digits = 2) => <div className="space-y-1"><MatrixView M={M} label={label} digits={digits} cell={46} /><div className="text-[10.5px] text-faint">rows: {rows.join(" · ")}</div></div>;
  switch (stage) {
    case 0: body = <p className="text-[15px]">“{text}”</p>; steps = [{ tex: "\\text{raw string}\\to\\text{model}", note: `vocabulary has ${VOCAB.length} entries: ${VOCAB.join(", ")}` }]; break;
    case 1: body = <div className="flex flex-wrap gap-1.5">{T.tokens.map((t, i) => <span key={i} className="num rounded bg-blue/20 px-2 py-1 text-[13px]">{t}</span>)}</div>; steps = [{ tex: "\\text{word-level tokens; a }\\texttt{<bos>}\\text{ token marks the start}" }]; break;
    case 2: body = <div className="flex flex-wrap gap-1.5">{T.tokens.map((t, i) => <span key={i} className="num rounded border border-line px-2 py-1 text-[13px]">{t} → <span className="text-blue">{ids[i]}</span></span>)}</div>; steps = [{ tex: `\\text{ids}=(${ids.join(",")})` }]; break;
    case 3: body = withRows(T.emb, `embedding rows E[id] (${n}×${D})`); steps = [{ tex: `\\mathbf e_i=E[\\text{id}_i],\\quad E\\in\\mathbb R^{${VOCAB.length}\\times${D}}` }]; break;
    case 4: body = <div className="flex flex-wrap gap-3">{withRows(T.pos, "positional encoding P")}<MatrixView M={T.x0} label="x₀ = E + P" digits={2} cell={46} /></div>; steps = [{ tex: `\\mathbf x^{(0)}_i=\\mathbf e_i+PE(i),\\quad PE_{(i,2k)}=\\sin(i/10000^{2k/${D}})` }]; break;
    case 5: body = <div className="flex flex-wrap gap-3"><MatrixView M={T.ln1} label="LayerNorm(x₀)" digits={2} cell={44} /><MatrixView M={H.Q} label={`Q (head ${head + 1})`} digits={2} cell={44} /><MatrixView M={H.K} label="K" digits={2} cell={44} /><MatrixView M={H.V} label="V" digits={2} cell={44} /></div>; steps = [{ tex: `Q=\\operatorname{LN}(X)W_Q^{(${head + 1})},\\ K=\\operatorname{LN}(X)W_K,\\ V=\\operatorname{LN}(X)W_V,\\quad W\\in\\mathbb R^{${D}\\times${DH}}`, note: "pre-LayerNorm block (as in GPT-2 and later)" }]; break;
    case 6: body = withRows(H.masked, `scores QKᵀ/√${DH} with causal mask`); steps = [{ tex: `s_{ij}=\\frac{\\mathbf q_i\\cdot\\mathbf k_j}{\\sqrt{${DH}}},\\quad s_{ij}=-\\infty\\text{ for }j>i` }]; break;
    case 7: body = withRows(H.weights, "attention weights (rows sum to 1)", 3); steps = [{ tex: `\\alpha_{${n - 1},\\cdot}=(${H.weights[n - 1].map((v) => tex(v, 3)).join(",")})`, note: "last token's attention" }]; break;
    case 8: body = withRows(H.out, `head ${head + 1} output = AV (${n}×${DH})`, 3); steps = [{ tex: "\\mathbf z_i=\\sum_j\\alpha_{ij}\\mathbf v_j" }]; break;
    case 9: body = <div className="flex flex-wrap gap-3">{withRows(T.concat, "concat(head₁, head₂)", 3)}<MatrixView M={T.attnOut} label="× W_O" digits={3} cell={46} /></div>; steps = [{ tex: `\\operatorname{MHA}(X)=[\\mathbf z^{(1)}\\,\\|\\,\\mathbf z^{(2)}]W_O,\\ W_O\\in\\mathbb R^{${D}\\times${D}}` }]; break;
    case 10: body = <div className="flex flex-wrap gap-3"><MatrixView M={T.x0} label="x₀" digits={2} cell={44} /><span className="self-center text-faint">+</span><MatrixView M={T.attnOut} label="attention out" digits={2} cell={44} /><span className="self-center text-faint">=</span><MatrixView M={T.res1} label="x₁" digits={2} cell={44} /></div>; steps = [{ tex: "\\mathbf x^{(1)}=\\mathbf x^{(0)}+\\operatorname{MHA}(\\operatorname{LN}(\\mathbf x^{(0)}))", note: "residual path lets gradients and information bypass the sub-layer" }]; break;
    case 11: body = <div className="flex flex-wrap gap-3"><MatrixView M={T.ffHidden} label={`GELU(LN(x₁)W₁) (${n}×${FF})`} digits={2} cell={36} /><MatrixView M={T.res2} label="x₂ = x₁ + FFN" digits={2} cell={44} /></div>; steps = [{ tex: `\\operatorname{FFN}(\\mathbf x)=\\operatorname{GELU}(\\mathbf xW_1+\\mathbf b_1)W_2+\\mathbf b_2,\\ W_1\\in\\mathbb R^{${D}\\times${FF}}`, note: "applied to each position independently" }]; break;
    case 12: body = <div className="grid gap-1 sm:grid-cols-2">{top.map((t) => <div key={t.w} className="flex items-center gap-2 text-[12.5px]"><span className="w-12">{t.w}</span><div className="h-3 flex-1 rounded bg-raised"><div className="h-3 rounded" style={{ width: `${Math.max(0, (t.l - Math.min(...T.logits)) / (Math.max(...T.logits) - Math.min(...T.logits) || 1)) * 100}%`, background: "rgb(var(--amber))" }} /></div><span className="num w-12 text-right text-muted">{fmt(t.l, 2)}</span></div>)}</div>; steps = [{ tex: "\\boldsymbol\\ell=\\operatorname{LN}(\\mathbf x^{(2)}_{\\text{last}})\\,E^\\top", note: "unembedding tied to the embedding matrix" }]; break;
    case 13: body = <div className="grid gap-1 sm:grid-cols-2">{top.map((t) => <div key={t.w} className="flex items-center gap-2 text-[12.5px]"><span className="w-12">{t.w}</span><div className="h-3 flex-1 rounded bg-raised"><div className="h-3 rounded" style={{ width: `${t.p * 100}%`, background: "rgb(var(--blue))" }} /></div><span className="num w-12 text-right text-muted">{fmt(t.p, 3)}</span></div>)}</div>; steps = [{ tex: `p(w\\mid\\text{context})=\\operatorname{softmax}(\\boldsymbol\\ell/T)_w,\\ T=${tex(temp, 2)}` }]; break;
    case 14: body = <p className="text-[15px]">{T.tokens.slice(1).join(" ")} {gen.map((g, i) => <span key={i} className="rounded bg-pink/20 px-1" title={`p = ${fmt(g.p, 3)}`}>{g.tok}</span>)}</p>; steps = [{ tex: "w_{t+1}\\sim p(\\cdot\\mid w_{\\le t}),\\ \\text{append, run the model again on the longer sequence}", note: strategy === "greedy" ? "greedy: always the arg-max token" : "sampling from the softmax distribution" }]; break;
  }
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-4">
          <ol className="flex flex-wrap gap-1" aria-label="Transformer stages">
            {STAGES.map((s, i) => (
              <li key={s}>
                <button onClick={() => setStage(i)} className={cn("rounded-md border px-2 py-1 text-[11.5px]", i === stage ? "border-blue bg-blue/10 text-fg" : "border-line text-muted hover:text-fg")}>
                  <span className="num mr-1 text-faint">{i + 1}</span>{s}
                </button>
              </li>
            ))}
          </ol>
          <div className="min-h-[220px] overflow-x-auto rounded-md border border-line p-4">
            <div className="label mb-3">{stage + 1}. {STAGES[stage]}</div>
            {body}
          </div>
          <div className="flex gap-2">
            <button className="btn" disabled={stage === 0} onClick={() => setStage(stage - 1)}>← Previous</button>
            <button className="btn" disabled={stage === STAGES.length - 1} onClick={() => setStage(stage + 1)}>Next →</button>
          </div>
        </div>
      }
      controls={
        <>
          <label className="block space-y-1">
            <span className="text-[12.5px] text-muted">Input (words from the vocabulary)</span>
            <input value={text} onChange={(e) => setText(e.target.value)} className="h-8 w-full rounded-md border border-line bg-raised px-2 text-[13px] outline-none focus:border-blue" />
          </label>
          {unknown.length > 0 && <p className="text-[12px] text-amber">Not in the toy vocabulary (dropped): {unknown.join(", ")}</p>}
          <Segmented label="Attention head" value={String(head)} onChange={(v) => setHead(Number(v))} options={[{ value: "0", label: "Head 1" }, { value: "1", label: "Head 2" }]} />
          <ParameterSlider label="temperature" value={temp} min={0.1} max={3} step={0.05} onChange={setTemp} />
          <ParameterSlider label="tokens to generate" value={genSteps} min={1} max={6} step={1} onChange={setGenSteps} />
          <Select label="Decoding" value={strategy} onChange={setStrategy} options={[{ value: "greedy", label: "Greedy (arg-max)" }, { value: "sample", label: "Sampling (fixed seed)" }]} />
        </>
      }
      readouts={<Readouts items={[{ label: "n\\ \\text{tokens}", tex: true, value: n }, { label: "d_{\\text{model}},\\ h,\\ d_{\\text{head}}", tex: true, value: `${D}, 2, ${DH}` }, { label: "\\text{top next token}", tex: true, value: `${top[0].w} (${fmt(top[0].p, 3)})` }, { label: "\\text{parameters}", tex: true, value: VOCAB.length * D + 3 * 2 * D * DH + D * D + D * FF + FF + FF * D + D }]} />}
      calc={<CalcSteps steps={steps} />}
      note="A one-block, two-head decoder with fixed random weights (seed 42). It is not trained, so its predictions are arbitrary — but every number shown is the exact output of the Transformer equations, consistent from stage to stage. Real LLMs stack dozens of these blocks with d_model in the thousands."
    />
  );
}
