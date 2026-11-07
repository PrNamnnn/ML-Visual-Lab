"use client";
import { useMemo, useRef, useState } from "react";
import { Plot, Dot, C, Polyline, Label, HeatmapCanvas, useThemeRGB } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Toggle, AnimationControls, Select } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { eigSym } from "@/lib/math/linalg";
import { makeDataset, knnPredict, sigmoid, type Point } from "@/lib/ml/models";
import { useTicker } from "@/hooks/useSize";
import { fmt, pct, tex, cn } from "@/lib/utilities/format";
import { BarList, Heat, MiniTable } from "@/visualizers/shared";
import { fitLogistic } from "@/visualizers/machine-learning/MLClassify";

/* ------------------------------ text preprocessing ------------------------------ */
const STOP = new Set(["the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of", "in", "on", "for", "it", "its", "this", "that", "with", "as", "at", "by", "be", "i", "we", "you", "they"]);
/** A deliberately small suffix-stripping stemmer (a few Porter-style rules). */
export function stem(w: string) {
  if (w.length <= 3) return w;
  for (const [suf, rep] of [["ational", "ate"], ["ization", "ize"], ["fulness", "ful"], ["iveness", "ive"], ["ies", "y"], ["ing", ""], ["edly", ""], ["ed", ""], ["ly", ""], ["es", ""], ["s", ""]] as const) {
    if (w.endsWith(suf) && w.length - suf.length >= 3) { let b = w.slice(0, -suf.length) + rep; if ((suf === "ing" || suf === "ed") && /([b-df-hj-np-tv-z])\1$/.test(b) && !/(ll|ss|zz)$/.test(b)) b = b.slice(0, -1); return b; }
  }
  return w;
}
export function TextPrepViz() {
  const [text, setText] = useState("The runners were Running quickly; the RUNNER runs, and runs again!! Visit https://example.com — it's the 3rd race of 2024.");
  const [lower, setLower] = useState(true);
  const [punct, setPunct] = useState(true);
  const [stop, setStop] = useState(true);
  const [doStem, setDoStem] = useState(true);
  const [urls, setUrls] = useState(true);
  let t = text;
  if (urls) t = t.replace(/https?:\/\/\S+/g, " <url> ");
  if (lower) t = t.toLowerCase();
  if (punct) t = t.replace(/[^\p{L}\p{N}<>'\s]/gu, " ").replace(/'s\b/g, "");
  const raw = text.split(/\s+/).filter(Boolean);
  let toks = t.split(/\s+/).filter(Boolean);
  const removed = stop ? toks.filter((w) => STOP.has(w.toLowerCase())) : [];
  if (stop) toks = toks.filter((w) => !STOP.has(w.toLowerCase()));
  const final = doStem ? toks.map((w) => (w.startsWith("<") ? w : stem(w))) : toks;
  const counts = new Map<string, number>(); final.forEach((w) => counts.set(w, (counts.get(w) ?? 0) + 1));
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <textarea value={text} onChange={(e) => setText(e.target.value.slice(0, 600))} rows={3} aria-label="text to preprocess" className="w-full resize-y rounded-lg border border-line bg-raised p-2.5 text-[13px] text-fg outline-none focus:border-line-strong" />
          <div className="space-y-1"><div className="text-[12px] text-faint">Tokens after the pipeline ({final.length})</div><div className="flex flex-wrap gap-1">{final.map((w, i) => <span key={i} className={cn("rounded-md border border-line px-1.5 py-0.5 text-[12.5px]", w.startsWith("<") ? "text-amber" : "text-fg")} style={{ background: "rgb(var(--blue) / 0.12)" }}>{w}</span>)}</div></div>
          {stop && removed.length > 0 && <div className="text-[12px] text-faint">Stop words removed: {removed.join(", ")}</div>}
          <MiniTable head={["token", "count"]} rows={[...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([w, c]) => [w, c])} />
        </div>
      }
      controls={
        <>
          <Toggle label="Replace URLs with <url>" checked={urls} onChange={setUrls} />
          <Toggle label="Lowercase" checked={lower} onChange={setLower} />
          <Toggle label="Strip punctuation" checked={punct} onChange={setPunct} />
          <Toggle label="Remove stop words" checked={stop} onChange={setStop} />
          <Toggle label="Stem (suffix stripping)" checked={doStem} onChange={setDoStem} />
          <p className="text-[12px] leading-relaxed text-faint">Each step merges surface variants into fewer, more frequent tokens: “Running”, “runs” and “runners” collapse towards “run”. That helps count-based models (bag of words, TF-IDF) but loses information — modern subword tokenizers for Transformers keep casing and punctuation instead.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{raw whitespace tokens}", tex: true, value: raw.length },
        { label: "\\text{final tokens}", tex: true, value: final.length },
        { label: "\\text{distinct types}", tex: true, value: `${new Set(raw).size} → ${counts.size}`, color: C.green },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{running}\to\text{runn}\to\text{run},\quad\text{runners}\to\text{runner},\quad\text{studies}\to\text{study}`, note: "rule-based stemming can over- or under-merge; lemmatisation uses a dictionary instead" }]} />}
    />
  );
}

/* ------------------------------ vocabulary design ------------------------------ */
const CORPUS = `the cat sat on the mat . the dog sat on the log . a cat and a dog played in the garden . the garden was green and the sun was warm .
the cat chased a mouse . the mouse ran under the table . a small dog barked at the cat . the children fed the dog and the cat .
the old man walked his dog in the park every morning . the park had tall trees and a quiet pond . ducks swam in the pond .
the cat slept in the warm sun . the dog dug a hole in the garden . the mouse found cheese under the table . the children laughed .
a bird sang in the tall tree . the old cat watched the bird . the dog chased the ducks to the pond . the sun set behind the trees .`;
const HELD = "the young fox jumped over the lazy dog near the quiet pond while a bird sang .";
export function VocabViz() {
  const [K, setK] = useState(30);
  const words = CORPUS.split(/\s+/).filter(Boolean);
  const freq = useMemo(() => { const m = new Map<string, number>(); words.forEach((w) => m.set(w, (m.get(w) ?? 0) + 1)); return [...m.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const vocab = new Set(freq.slice(0, K).map((f) => f[0]));
  const held = HELD.split(/\s+/);
  const mapped = held.map((w) => (vocab.has(w) ? w : "[UNK]"));
  const covTrain = words.filter((w) => vocab.has(w)).length / words.length;
  const oov = mapped.filter((w) => w === "[UNK]").length / held.length;
  return (
    <LabShell
      viz={
        <div className="grid gap-4 lg:grid-cols-2">
          <Plot x={[0, Math.log10(freq.length) + 0.1]} y={[0, Math.log10(freq[0][1]) + 0.15]} aspect={0.75} maxH={300} xLabel="log₁₀ rank" yLabel="log₁₀ frequency" ariaLabel="Zipf plot of word frequencies">
            {freq.map(([, c], r) => <Dot key={r} p={[Math.log10(r + 1), Math.log10(c)]} r={2.8} color={r < K ? C.blue : C.a("fg", 0.25)} />)}
            <Polyline pts={[[0, Math.log10(freq[0][1])], [Math.log10(freq.length), Math.log10(freq[0][1]) - Math.log10(freq.length)]]} style={{ stroke: C.pink, strokeDasharray: "4 3" }} />
          </Plot>
          <div className="space-y-2">
            <div className="text-[12px] text-faint">Held-out sentence mapped to the vocabulary</div>
            <div className="flex flex-wrap gap-1">{["[BOS]", ...mapped, "[EOS]"].map((w, i) => <span key={i} className="rounded-md border border-line px-1.5 py-0.5 text-[12.5px]" style={{ background: w === "[UNK]" ? "rgb(var(--red) / 0.25)" : w.startsWith("[") ? "rgb(var(--amber) / 0.2)" : "rgb(var(--blue) / 0.12)" }}>{w}</span>)}</div>
            <div className="text-[12px] text-faint">Special tokens: [PAD] (batch padding), [UNK] (unknown), [BOS]/[EOS] (sequence boundaries).</div>
            <MiniTable head={["rank", "word", "count"]} rows={freq.slice(0, 6).map(([w, c], r) => [r + 1, w, c])} />
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="vocabulary size K" value={K} min={5} max={freq.length} step={1} onChange={setK} />
          <p className="text-[12px] leading-relaxed text-faint">Word frequencies follow Zipf&apos;s law (a straight line on the log-log plot): a few words are extremely common and most are rare. A small vocabulary covers most running text but maps every rare or new word to [UNK]; a large one costs memory in the embedding and output layers. Subword tokenizers avoid [UNK] altogether.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{distinct words in corpus}", tex: true, value: freq.length },
        { label: "\\text{corpus tokens covered}", tex: true, value: pct(covTrain), color: C.green },
        { label: "\\text{held-out OOV rate}", tex: true, value: pct(oov), color: C.red },
        { label: "\\text{embedding table (d = 512)}", tex: true, value: `${(K + 4) * 512} weights` },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`f(r)\propto\frac1{r^{s}},\ s\approx1\qquad\text{OOV rate}=\frac{\#\{w\notin V\}}{\#\text{tokens}}=${tex(oov, 3)}` }]} />}
    />
  );
}

/* ------------------------------ word2vec (skip-gram / CBOW, trained live) ------------------------------ */
const W2V_SENT = (() => {
  const r = mulberry32(11), out: string[][] = [];
  const pets = ["cat", "dog"], royals = ["king", "queen"], food = ["bread", "cheese"], drinks = ["milk", "water"];
  for (let i = 0; i < 120; i++) {
    const t = Math.floor(r() * 4), pick = (a: string[]) => a[Math.floor(r() * a.length)];
    if (t === 0) out.push(["the", pick(pets), pick(["drinks", "likes"]), pick(drinks)]);
    else if (t === 1) out.push(["the", pick(royals), pick(["rules", "visits"]), "the", "castle"]);
    else if (t === 2) out.push(["people", pick(["eat", "buy"]), pick(food), "daily"]);
    else out.push(["the", pick(pets), "sleeps", "near", "the", pick(royals)]);
  }
  return out;
})();
const W2V_VOCAB = [...new Set(W2V_SENT.flat())];
const W2V_IDX = new Map(W2V_VOCAB.map((w, i) => [w, i]));
function pca2(E: number[][]) {
  const d = E[0].length, n = E.length, mu = Array.from({ length: d }, (_, k) => E.reduce((s, e) => s + e[k], 0) / n);
  const Cv = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => E.reduce((s, e) => s + (e[a] - mu[a]) * (e[b] - mu[b]), 0) / n));
  const { vectors } = eigSym(Cv, 300);
  return E.map((e) => [0, 1].map((c) => e.reduce((s, v, k) => s + (v - mu[k]) * vectors[k][c], 0)) as [number, number]);
}
const cos = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0) / (Math.hypot(...a) * Math.hypot(...b) || 1);
export function Word2VecViz() {
  const [arch, setArch] = useState<"skipgram" | "cbow">("skipgram");
  const [win, setWin] = useState(2);
  const [playing, setPlaying] = useState(true);
  const [epoch, setEpoch] = useState(0);
  const [probe, setProbe] = useState("cat");
  const D = 6, V = W2V_VOCAB.length;
  const init = () => { const r = mulberry32(2); return { Win: Array.from({ length: V }, () => Array.from({ length: D }, () => 0.3 * randn(r))), Wout: Array.from({ length: V }, () => Array.from({ length: D }, () => 0)) }; };
  const W = useRef(init());
  const pairs = useMemo(() => W2V_SENT.flatMap((s) => s.map((w, i) => ({ c: W2V_IDX.get(w)!, ctx: s.slice(Math.max(0, i - win), i).concat(s.slice(i + 1, i + 1 + win)).map((x) => W2V_IDX.get(x)!) }))), [win]);
  const step = () => {
    const r = mulberry32(epoch + 7), lr = 0.05, neg = 3, { Win, Wout } = W.current;
    const upd = (vin: number[], o: number, label: number) => { const s = sigmoid(vin.reduce((a, x, k) => a + x * Wout[o][k], 0)); const g = lr * (label - s); const dv = Wout[o].map((w) => g * w); for (let k = 0; k < D; k++) Wout[o][k] += g * vin[k]; return dv; };
    for (const p of pairs) {
      if (arch === "skipgram") {
        for (const o of p.ctx) { const vin = Win[p.c]; const grad = Array(D).fill(0); [[o, 1], ...Array.from({ length: neg }, () => [Math.floor(r() * V), 0])].forEach(([t, l]) => { if (l === 0 && t === o) return; upd(vin, t, l).forEach((v, k) => (grad[k] += v)); }); for (let k = 0; k < D; k++) Win[p.c][k] += grad[k]; }
      } else if (p.ctx.length) {
        const h = Array.from({ length: D }, (_, k) => p.ctx.reduce((s, c) => s + Win[c][k], 0) / p.ctx.length);
        const grad = Array(D).fill(0);
        [[p.c, 1], ...Array.from({ length: neg }, () => [Math.floor(r() * V), 0])].forEach(([t, l]) => { if (l === 0 && t === p.c) return; upd(h, t, l).forEach((v, k) => (grad[k] += v)); });
        p.ctx.forEach((c) => { for (let k = 0; k < D; k++) Win[c][k] += grad[k] / p.ctx.length; });
      }
    }
    setEpoch((e) => e + 1);
  };
  useTicker(playing && epoch < 60, 60, step);
  const E = W.current.Win;
  const xy = pca2(E);
  const pi = W2V_IDX.get(probe) ?? 0;
  const near = W2V_VOCAB.map((w, i) => ({ w, s: cos(E[pi], E[i]) })).filter((o) => o.w !== probe).sort((a, b) => b.s - a.s).slice(0, 5);
  const ex = W2V_SENT[3], exPairs = arch === "skipgram" ? ex.flatMap((w, i) => ex.slice(Math.max(0, i - win), i).concat(ex.slice(i + 1, i + 1 + win)).map((c) => `${w} → ${c}`)).slice(0, 8) : ex.map((w, i) => `[${ex.slice(Math.max(0, i - win), i).concat(ex.slice(i + 1, i + 1 + win)).join(", ")}] → ${w}`);
  const group = (w: string) => (["cat", "dog"].includes(w) ? C.blue : ["king", "queen"].includes(w) ? C.pink : ["bread", "cheese", "milk", "water"].includes(w) ? C.green : C.a("fg", 0.45));
  const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]);
  const pad = (a: number[]) => { const lo = Math.min(...a), hi = Math.max(...a), d = hi - lo || 1; return [lo - 0.15 * d, hi + 0.15 * d] as [number, number]; };
  return (
    <LabShell
      viz={
        <div className="grid gap-3 lg:grid-cols-[1.4fr_1fr]">
          <Plot x={pad(xs)} y={pad(ys)} aspect={0.8} maxH={380} axes={false} ariaLabel="word embeddings projected to 2-D">
            {xy.map((p, i) => <g key={i}><Dot p={p} r={4} color={group(W2V_VOCAB[i])} /><Label p={p} dx={6} dy={-4} color={group(W2V_VOCAB[i])}>{W2V_VOCAB[i]}</Label></g>)}
          </Plot>
          <div className="space-y-2">
            <div className="text-[12px] text-faint">Training examples from “{ex.join(" ")}”</div>
            <div className="flex flex-wrap gap-1">{exPairs.map((p, i) => <span key={i} className="rounded-md border border-line bg-raised px-1.5 py-0.5 text-[11.5px] text-muted">{p}</span>)}</div>
            <div className="text-[12px] text-faint">Nearest words to “{probe}” (cosine)</div>
            <BarList max={1} items={near.map((n) => ({ label: n.w, value: Math.max(0, n.s), color: group(n.w) }))} />
          </div>
        </div>
      }
      controls={
        <>
          <Segmented label="Architecture" value={arch} onChange={(v) => { setArch(v); W.current = init(); setEpoch(0); }} options={[{ value: "skipgram", label: "Skip-gram" }, { value: "cbow", label: "CBOW" }]} />
          <ParameterSlider label="context window" value={win} min={1} max={4} step={1} onChange={(v) => { setWin(v); W.current = init(); setEpoch(0); }} />
          <Select label="Probe word" value={probe} onChange={setProbe} options={W2V_VOCAB.map((w) => ({ value: w, label: w }))} />
          <AnimationControls playing={playing} onPlay={() => setPlaying((p) => !p)} onStep={step} onReset={() => { W.current = init(); setEpoch(0); }} />
          <p className="text-[12px] leading-relaxed text-faint">Trained live with negative sampling on 120 template sentences. Words used in the same contexts (cat/dog, king/queen, milk/water) end up close together, although nobody told the model they are related. Plot = first two principal components of the 6-D vectors.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{epoch}", tex: true, value: epoch },
        { label: "\\cos(\\text{cat},\\text{dog})", tex: true, value: fmt(cos(E[W2V_IDX.get("cat")!], E[W2V_IDX.get("dog")!]), 3), color: C.blue },
        { label: "\\cos(\\text{cat},\\text{bread})", tex: true, value: fmt(cos(E[W2V_IDX.get("cat")!], E[W2V_IDX.get("bread")!]), 3) },
        { label: "\\text{vocab / dim}", tex: true, value: `${V} / ${D}` },
      ]} />}
      calc={<CalcSteps steps={[arch === "skipgram"
        ? { tex: String.raw`\max\ \log\sigma(\mathbf u_o^\top\mathbf v_c)+\sum_{k=1}^{3}\log\sigma(-\mathbf u_{n_k}^\top\mathbf v_c)`, note: "predict each context word o from the centre word c; 3 random negative words n_k" }
        : { tex: String.raw`\mathbf h=\frac1{|\mathcal C|}\sum_{o\in\mathcal C}\mathbf v_o,\qquad\max\ \log\sigma(\mathbf u_c^\top\mathbf h)+\sum_k\log\sigma(-\mathbf u_{n_k}^\top\mathbf h)`, note: "predict the centre word from the averaged context" }]} />}
    />
  );
}

/* ------------------------------ GloVe ------------------------------ */
export function GloVeViz() {
  const [win, setWin] = useState(2);
  const [playing, setPlaying] = useState(true);
  const [epoch, setEpoch] = useState(0);
  const V = W2V_VOCAB.length, D = 6;
  const X = useMemo(() => {
    const M = Array.from({ length: V }, () => Array(V).fill(0));
    W2V_SENT.forEach((s) => s.forEach((w, i) => { for (let j = Math.max(0, i - win); j <= Math.min(s.length - 1, i + win); j++) if (j !== i) M[W2V_IDX.get(w)!][W2V_IDX.get(s[j])!] += 1 / Math.abs(i - j); }));
    return M;
  }, [win, V]);
  const init = () => { const r = mulberry32(5); const g = () => 0.1 * randn(r); return { W: Array.from({ length: V }, () => Array.from({ length: D }, g)), Wt: Array.from({ length: V }, () => Array.from({ length: D }, g)), b: Array(V).fill(0), bt: Array(V).fill(0) }; };
  const P = useRef(init());
  const [loss, setLoss] = useState(NaN);
  const xmax = 10, alpha = 0.75;
  const step = () => {
    const p = P.current, lr = 0.05; let L = 0;
    for (let rep = 0; rep < 5; rep++) for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) {
      if (X[i][j] <= 0) continue;
      const f = Math.min(1, (X[i][j] / xmax) ** alpha);
      const e = p.W[i].reduce((s, v, k) => s + v * p.Wt[j][k], 0) + p.b[i] + p.bt[j] - Math.log(X[i][j]);
      L += f * e * e;
      for (let k = 0; k < D; k++) { const wi = p.W[i][k]; p.W[i][k] -= lr * f * e * p.Wt[j][k]; p.Wt[j][k] -= lr * f * e * wi; }
      p.b[i] -= lr * f * e; p.bt[j] -= lr * f * e;
    }
    setLoss(L / 5); setEpoch((e) => e + 5);
  };
  useTicker(playing && epoch < 300, 60, step);
  const E = P.current.W.map((w, i) => w.map((v, k) => v + P.current.Wt[i][k]));
  const xy = pca2(E);
  const rowsum = X.map((r) => r.reduce((s, v) => s + v, 0));
  const pr = (a: string, b: string) => X[W2V_IDX.get(a)!][W2V_IDX.get(b)!] / rowsum[W2V_IDX.get(a)!];
  const probes = ["milk", "castle", "drinks", "rules", "the"];
  const show = ["cat", "dog", "king", "queen", "milk", "castle", "drinks", "rules"];
  const group = (w: string) => (["cat", "dog"].includes(w) ? C.blue : ["king", "queen"].includes(w) ? C.pink : ["bread", "cheese", "milk", "water"].includes(w) ? C.green : C.a("fg", 0.45));
  const pad = (a: number[]) => { const lo = Math.min(...a), hi = Math.max(...a), d = hi - lo || 1; return [lo - 0.15 * d, hi + 0.15 * d] as [number, number]; };
  return (
    <LabShell
      wideControls
      viz={
        <div className="grid gap-3 xl:grid-cols-2">
          <div className="space-y-1.5"><div className="text-[12px] text-faint">Co-occurrence counts X (distance-weighted, window {win})</div><Heat M={show.map((a) => show.map((b) => X[W2V_IDX.get(a)!][W2V_IDX.get(b)!]))} rowLabels={show} colLabels={show} digits={0} cell={34} /></div>
          <Plot x={pad(xy.map((p) => p[0]))} y={pad(xy.map((p) => p[1]))} aspect={0.8} maxH={340} axes={false} ariaLabel="GloVe vectors projected to 2-D">
            {xy.map((p, i) => <g key={i}><Dot p={p} r={4} color={group(W2V_VOCAB[i])} /><Label p={p} dx={6} dy={-4} color={group(W2V_VOCAB[i])}>{W2V_VOCAB[i]}</Label></g>)}
          </Plot>
          <div className="xl:col-span-2"><MiniTable head={["probe k", "P(k | cat)", "P(k | king)", "ratio"]} rows={probes.map((k) => { const a = pr("cat", k), b = pr("king", k); return [k, fmt(a, 3), fmt(b, 3), b > 0 ? fmt(a / b, 2) : a > 0 ? "∞" : "—"]; })} /></div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="context window" value={win} min={1} max={4} step={1} onChange={(v) => { setWin(v); P.current = init(); setEpoch(0); }} />
          <AnimationControls playing={playing} onPlay={() => setPlaying((p) => !p)} onStep={step} onReset={() => { P.current = init(); setEpoch(0); }} />
          <p className="text-[12px] leading-relaxed text-faint">GloVe first counts how often words appear near each other over the whole corpus, then fits vectors whose dot products match the log counts. Ratios of co-occurrence probabilities carry meaning: “milk” is far more likely near “cat” than near “king” (large ratio), “castle” the reverse, while “the” is uninformative (ratio ≈ 1).</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{epoch}", tex: true, value: epoch },
        { label: "J", tex: true, value: fmt(loss, 3), hint: "weighted least-squares loss" },
        { label: "\\cos(\\text{king},\\text{queen})", tex: true, value: fmt(cos(E[W2V_IDX.get("king")!], E[W2V_IDX.get("queen")!]), 3), color: C.pink },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`J=\sum_{i,j:X_{ij}>0}f(X_{ij})\big(\mathbf w_i^\top\tilde{\mathbf w}_j+b_i+\tilde b_j-\log X_{ij}\big)^2,\quad f(x)=\min\!\big(1,(x/${xmax})^{0.75}\big)`, note: "f down-weights rare pairs and caps very frequent ones" }]} />}
    />
  );
}

/* ------------------------------ beam search ------------------------------ */
// A toy language model: next-token probabilities depend only on the previous token
const LM: Record<string, [string, number][]> = {
  "<s>": [["the", 0.5], ["a", 0.4], ["one", 0.1]],
  the: [["dog", 0.4], ["cat", 0.35], ["end", 0.25]],
  a: [["very", 0.9], ["dog", 0.1]],
  one: [["cat", 0.9], ["dog", 0.1]],
  very: [["good", 0.95], ["bad", 0.05]],
  dog: [["barks", 0.6], ["</s>", 0.4]],
  cat: [["sleeps", 0.7], ["</s>", 0.3]],
  good: [["dog", 0.8], ["</s>", 0.2]],
  bad: [["</s>", 1]], end: [["</s>", 1]], barks: [["</s>", 1]], sleeps: [["</s>", 1]],
};
export function BeamViz() {
  const [k, setK] = useState(2);
  const [norm, setNorm] = useState(false);
  const maxLen = 6;
  const res = useMemo(() => {
    let beams = [{ seq: ["<s>"], lp: 0, done: false }];
    const levels: { seq: string[]; lp: number; kept: boolean }[][] = [];
    for (let t = 0; t < maxLen; t++) {
      const cand = beams.flatMap((b) => (b.done ? [b] : LM[b.seq[b.seq.length - 1]].map(([w, p]) => ({ seq: [...b.seq, w], lp: b.lp + Math.log(p), done: w === "</s>" }))));
      const score = (c: { seq: string[]; lp: number }) => (norm ? c.lp / (c.seq.length - 1) : c.lp);
      cand.sort((a, b) => score(b) - score(a));
      const keep = cand.slice(0, k);
      levels.push(cand.map((c) => ({ seq: c.seq, lp: c.lp, kept: keep.includes(c) })));
      beams = keep;
      if (beams.every((b) => b.done)) break;
    }
    const best = beams.reduce((a, b) => ((norm ? b.lp / (b.seq.length - 1) : b.lp) > (norm ? a.lp / (a.seq.length - 1) : a.lp) ? b : a));
    // exhaustive search for reference
    const all: { seq: string[]; lp: number }[] = [];
    const rec = (seq: string[], lp: number) => { const last = seq[seq.length - 1]; if (last === "</s>" || seq.length > maxLen) { if (last === "</s>") all.push({ seq, lp }); return; } LM[last].forEach(([w, p]) => rec([...seq, w], lp + Math.log(p))); };
    rec(["<s>"], 0);
    const opt = all.reduce((a, b) => ((norm ? b.lp / (b.seq.length - 1) : b.lp) > (norm ? a.lp / (a.seq.length - 1) : a.lp) ? b : a));
    return { levels, best, opt, nAll: all.length };
  }, [k, norm]);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          {res.levels.map((L, t) => (
            <div key={t} className="flex flex-wrap items-center gap-1.5">
              <span className="w-10 text-[11px] text-faint">t = {t + 1}</span>
              {L.slice(0, 9).map((c, i) => <span key={i} className="rounded-md border px-1.5 py-0.5 text-[11.5px]" style={{ borderColor: c.kept ? "rgb(var(--green))" : "rgb(var(--line))", background: c.kept ? "rgb(var(--green) / 0.14)" : undefined, color: c.kept ? undefined : "rgb(var(--faint))" }}>{c.seq.slice(1).join(" ")} <span className="num opacity-70">{fmt(Math.exp(c.lp), 3)}</span></span>)}
            </div>
          ))}
          <MiniTable head={["", "sequence", "probability"]} rows={[["beam result", res.best.seq.slice(1).join(" "), fmt(Math.exp(res.best.lp), 4)], ["exact best", res.opt.seq.slice(1).join(" "), fmt(Math.exp(res.opt.lp), 4)]]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="beam width k" value={k} min={1} max={5} step={1} onChange={setK} />
          <Toggle label="Length-normalise scores" checked={norm} onChange={setNorm} />
          <p className="text-[12px] leading-relaxed text-faint">Greedy decoding (k = 1) takes the most likely word each step: “the” (0.5) beats “a” (0.4). But “a very good …” is much more likely overall, because its later words are near-certain. Keeping the k best partial sequences finds it once k ≥ 2. Green = kept at that step; numbers are sequence probabilities.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{beam found the optimum?}", tex: true, value: res.best.seq.join() === res.opt.seq.join() ? "yes ✓" : "no", color: res.best.seq.join() === res.opt.seq.join() ? C.green : C.red },
        { label: "\\text{sequences scored}", tex: true, value: res.levels.reduce((s, L) => s + L.length, 0) },
        { label: "\\text{all complete sequences}", tex: true, value: res.nAll },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\text{score}(y_{1:t})=\sum_{i=1}^{t}\log p(y_i\mid y_{<i})${norm ? String.raw`\ \big/\ t` : ""},\qquad\text{keep the top }k=${k}\text{ at every step}`, note: "length normalisation stops the search from favouring short outputs" }]} />}
    />
  );
}

/* ------------------------------ fine-tuning and forgetting ------------------------------ */
export function FineTuneViz() {
  const [steps, setSteps] = useState(150);
  const [lr, setLr] = useState(0.3);
  const [l2sp, setL2sp] = useState(0);
  const th = useThemeRGB();
  const rot = (pts: Point[], deg: number) => { const a = (deg * Math.PI) / 180; return pts.map((p) => ({ x: p.x, y: p.y, label: p.x * Math.sin(a) + p.y * Math.cos(a) > 0 ? 1 : 0 })); };
  const base = (n: number, seed: number) => { const r = mulberry32(seed); return Array.from({ length: n }, () => ({ x: 4 * r() - 2, y: 4 * r() - 2, label: 0 })); };
  const src = useMemo(() => rot(base(400, 1), 0), []);  
  const tgt = useMemo(() => rot(base(12, 2), 50), []);  
  const srcTest = useMemo(() => rot(base(400, 3), 0), []);  
  const tgtTest = useMemo(() => rot(base(400, 4), 50), []);  
  const pre = useMemo(() => { let w = [0, 0, 0]; for (let it = 0; it < 400; it++) { const g = [0, 0, 0]; src.forEach((p) => { const e = sigmoid(w[0] + w[1] * p.x + w[2] * p.y) - p.label; g[0] += e; g[1] += e * p.x; g[2] += e * p.y; }); w = w.map((v, k) => v - 0.5 * g[k] / src.length); } return w; }, [src]);
  const run = useMemo(() => {
    let w = pre.slice(); const hist: { s: number; t: number }[] = [];
    const acc = (w: number[], d: Point[]) => d.reduce((s, p) => s + ((w[0] + w[1] * p.x + w[2] * p.y > 0 ? 1 : 0) === p.label ? 1 : 0), 0) / d.length;
    for (let it = 0; it <= 300; it++) {
      if (it % 5 === 0) hist.push({ s: acc(w, srcTest), t: acc(w, tgtTest) });
      const g = [0, 0, 0]; tgt.forEach((p) => { const e = sigmoid(w[0] + w[1] * p.x + w[2] * p.y) - p.label; g[0] += e; g[1] += e * p.x; g[2] += e * p.y; });
      w = w.map((v, k) => v - lr * (g[k] / tgt.length + l2sp * (v - pre[k]))); // L2-SP: pull towards the pretrained weights
      if (it === steps) hist.push({ s: -1, t: -1 });
    }
    // weights after `steps`
    let ws = pre.slice(); for (let it = 0; it < steps; it++) { const g = [0, 0, 0]; tgt.forEach((p) => { const e = sigmoid(ws[0] + ws[1] * p.x + ws[2] * p.y) - p.label; g[0] += e; g[1] += e * p.x; g[2] += e * p.y; }); ws = ws.map((v, k) => v - lr * (g[k] / tgt.length + l2sp * (v - pre[k]))); }
    return { hist: hist.filter((h) => h.s >= 0), ws, sAcc: acc(ws, srcTest), tAcc: acc(ws, tgtTest), tAcc0: acc(pre, tgtTest), sAcc0: acc(pre, srcTest) };
  }, [pre, steps, lr, l2sp, tgt, srcTest, tgtTest]);
  const line = (w: number[]) => [[-2.2, (-w[0] + 2.2 * w[1]) / w[2]], [2.2, (-w[0] - 2.2 * w[1]) / w[2]]] as [number, number][];
  return (
    <LabShell
      viz={
        <div className="grid gap-3 lg:grid-cols-[1.2fr_1fr]">
          <Plot x={[-2.2, 2.2]} y={[-2.2, 2.2]} equal aspect={0.9} maxH={340} ariaLabel="pretrained and fine-tuned decision boundaries"
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={6} deps={[run.ws, th]} f={(x, y) => { const v = run.ws[0] + run.ws[1] * x + run.ws[2] * y > 0; const c = v ? th.pink : th.blue; return [th.panel[0] + (c[0] - th.panel[0]) * 0.18, th.panel[1] + (c[1] - th.panel[1]) * 0.18, th.panel[2] + (c[2] - th.panel[2]) * 0.18, 255]; }} />}>
            <Polyline pts={line(pre)} style={{ stroke: C.amber, strokeWidth: 2, strokeDasharray: "6 4" }} />
            <Polyline pts={line(run.ws)} style={{ stroke: C.green, strokeWidth: 2.4 }} />
            {tgt.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={5.5} color={p.label ? C.pink : C.blue} stroke={C.fg} />)}
          </Plot>
          <Plot x={[0, 300]} y={[0.4, 1.01]} aspect={0.9} maxH={300} xLabel="fine-tuning steps" yLabel="test accuracy" ariaLabel="source and target accuracy during fine-tuning">
            <Polyline pts={run.hist.map((h, i) => [i * 5, h.s])} style={{ stroke: C.amber, strokeWidth: 2 }} />
            <Polyline pts={run.hist.map((h, i) => [i * 5, h.t])} style={{ stroke: C.green, strokeWidth: 2 }} />
            <Polyline pts={[[steps, 0.4], [steps, 1.01]]} style={{ stroke: C.a("fg", 0.4), strokeDasharray: "3 3" }} />
          </Plot>
          <div className="lg:col-span-2"><Legend items={[{ label: "pretrained boundary (source task)", color: C.amber, dashed: true }, { label: "after fine-tuning on 12 target examples", color: C.green }, { label: "source accuracy", color: C.amber }, { label: "target accuracy", color: C.green }]} /></div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="fine-tuning steps" value={steps} min={0} max={300} step={5} onChange={setSteps} />
          <ParameterSlider label="learning rate" value={lr} min={0.02} max={1} step={0.02} onChange={setLr} />
          <ParameterSlider label="pull towards pretrained weights (L2-SP)" value={l2sp} min={0} max={0.5} step={0.01} onChange={setL2sp} />
          <p className="text-[12px] leading-relaxed text-faint">A model pretrained on a large source task is adapted to a new task (boundary rotated by 50°) with only 12 labelled examples. Target accuracy rises while source accuracy falls — catastrophic forgetting. A penalty pulling the weights towards their pretrained values trades some target accuracy for keeping the old skill; small learning rates and few steps act similarly.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{target acc: before → after}", tex: true, value: `${pct(run.tAcc0)} → ${pct(run.tAcc)}`, color: C.green },
        { label: "\\text{source acc: before → after}", tex: true, value: `${pct(run.sAcc0)} → ${pct(run.sAcc)}`, color: C.amber },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\theta\leftarrow\theta-\eta\Big(\nabla L_{\text{target}}(\theta)+${tex(l2sp, 2)}\,(\theta-\theta_{\text{pre}})\Big),\qquad\theta_0=\theta_{\text{pre}}` }]} />}
    />
  );
}

/* ------------------------------ knowledge distillation ------------------------------ */
const KD_CLASSES = ["cat", "dog", "fox", "car", "truck"];
export function DistillViz() {
  const [T, setT] = useState(4);
  const [alpha, setAlpha] = useState(0.5);
  const [nLab, setNLab] = useState(20);
  const th = useThemeRGB();
  const teacher = [6.2, 4.8, 3.9, -1.5, -2.2]; // logits for an image of a cat
  const soft = (z: number[], t: number) => { const m = Math.max(...z), e = z.map((v) => Math.exp((v - m) / t)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
  const pT = soft(teacher, T), p1 = soft(teacher, 1);
  // training demo: a teacher (15-NN on 600 labelled points) labels a transfer set; a small student learns from it
  const demo = useMemo(() => {
    const big = makeDataset("moons", 600, 0.2, 5), test = makeDataset("moons", 400, 0.2, 55);
    const transfer = makeDataset("moons", 300, 0.2, 9);
    const lab = [...transfer.filter((p) => p.label === 0).slice(0, nLab / 2), ...transfer.filter((p) => p.label === 1).slice(0, nLab / 2)];
    const feats = (x: number, y: number) => [x, y, x * x, x * y, y * y, x ** 3 / 2, y ** 3 / 2];
    const tProb = (x: number, y: number) => knnPredict(big, x, y, 15).prob;
    const hard = fitLogistic(lab.map((p) => feats(p.x, p.y)), lab.map((p) => p.label), 1200, 0.5);
    const distil = fitLogistic(transfer.map((p) => feats(p.x, p.y)), transfer.map((p) => tProb(p.x, p.y)), 1200, 0.5); // soft targets on unlabelled transfer data
    const acc = (f: (x: number, y: number) => number) => test.reduce((s, p) => s + ((f(p.x, p.y) >= 0.5 ? 1 : 0) === p.label ? 1 : 0), 0) / test.length;
    return { tAcc: acc(tProb), hAcc: acc((x, y) => hard(feats(x, y))), dAcc: acc((x, y) => distil(feats(x, y))), distil, feats };
  }, [nLab]);
  const kl = pT.reduce((s, p, i) => s + p * Math.log(p / soft([5, 3, 2, 0, 0], T)[i]), 0);
  return (
    <LabShell
      viz={
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-2">
            <div className="text-[12px] text-faint">Teacher&apos;s output for one cat image</div>
            <BarList max={1} items={KD_CLASSES.map((c, i) => ({ label: `${c} (T = 1)`, value: p1[i], color: C.a("fg", 0.35) }))} />
            <BarList max={1} items={KD_CLASSES.map((c, i) => ({ label: `${c} (T = ${T})`, value: pT[i], color: C.pink }))} />
            <p className="text-[12px] text-faint">At T = 1 the teacher says “cat” and little else. Raising T reveals “dark knowledge”: a cat looks more like a dog or fox than like a car.</p>
          </div>
          <div className="space-y-2">
            <Plot x={[-1.6, 2.6]} y={[-1.2, 1.6]} equal aspect={0.7} maxH={260} ariaLabel="distilled student decision regions"
              underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={6} deps={[nLab, th]} f={(x, y) => { const p = demo.distil(demo.feats(x, y)); const c = p >= 0.5 ? th.pink : th.blue; const k = 0.1 + 0.3 * Math.abs(p - 0.5) * 2; return [th.panel[0] + (c[0] - th.panel[0]) * k, th.panel[1] + (c[1] - th.panel[1]) * k, th.panel[2] + (c[2] - th.panel[2]) * k, 255]; }} />} />
            <BarList max={1} items={[{ label: "teacher (15-NN, 600 labels)", value: demo.tAcc, color: C.amber }, { label: `student, ${nLab} hard labels`, value: demo.hAcc, color: C.blue }, { label: "student, distilled", value: demo.dAcc, color: C.green }]} />
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="temperature T" value={T} min={1} max={10} step={0.5} onChange={setT} />
          <ParameterSlider label="weight on the hard-label loss α" value={alpha} min={0} max={1} step={0.05} onChange={setAlpha} />
          <ParameterSlider label="labelled examples for the student" value={nLab} min={4} max={60} step={2} onChange={setNLab} />
          <p className="text-[12px] leading-relaxed text-faint">The student is trained to match the teacher&apos;s softened probabilities, which carry much more information per example than a single hard label. Right: a small model trained on teacher soft labels for 300 unlabelled points versus the same model trained on only the hard labels available.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{teacher }p(\\text{dog})\\ \\text{at }T", tex: true, value: fmt(pT[1], 3) },
        { label: "\\mathrm{KL}(p_T^{\\text{teacher}}\\|p_T^{\\text{student}})", tex: true, value: fmt(kl, 4), hint: "for an example student with logits (5, 3, 2, 0, 0)" },
        { label: "\\text{distilled vs hard}", tex: true, value: `${pct(demo.dAcc)} vs ${pct(demo.hAcc)}`, color: C.green },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`p_i^{(T)}=\frac{e^{z_i/T}}{\sum_je^{z_j/T}},\qquad L=${tex(alpha, 2)}\,\mathrm{CE}(y,p^{(1)}_s)+${tex(1 - alpha, 2)}\,T^2\,\mathrm{KL}\big(p^{(T)}_t\,\|\,p^{(T)}_s\big)`, note: "the T² factor keeps gradient sizes comparable as T changes" },
      ]} />}
    />
  );
}
