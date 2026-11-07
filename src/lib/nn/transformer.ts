/**
 * A tiny decoder-only Transformer (1 block, 2 heads, d_model = 4) with fixed seeded weights.
 * It is NOT trained — its predictions are arbitrary — but every intermediate number is the
 * exact result of the real Transformer equations, so the visualizer can inspect each stage.
 */
import { mulberry32, randn } from "@/lib/math/random";
import { softmax } from "@/lib/nn";

export const VOCAB = ["<bos>", "the", "cat", "sat", "on", "mat", "dog", "ran", "a", "big", "red", "."];
export const D = 4;
export const H = 2;
export const DH = D / H;
export const FF = 8;

type M = number[][];
const mat = (r: number, c: number, rng: () => number, s: number): M => Array.from({ length: r }, () => Array.from({ length: c }, () => Math.round(randn(rng) * s * 100) / 100));
export const mm = (A: M, B: M): M => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
export const tr = (A: M): M => A[0].map((_, j) => A.map((r) => r[j]));

export interface Weights {
  E: M; // vocab × D (tied with output projection)
  Wq: M[]; Wk: M[]; Wv: M[]; // per head: D × DH
  Wo: M; // D × D
  W1: M; b1: number[]; W2: M; b2: number[];
  g1: number[]; g2: number[]; gf: number[];
}

export function makeWeights(seed = 42): Weights {
  const r = mulberry32(seed);
  return {
    E: mat(VOCAB.length, D, r, 1),
    Wq: [mat(D, DH, r, 0.7), mat(D, DH, r, 0.7)],
    Wk: [mat(D, DH, r, 0.7), mat(D, DH, r, 0.7)],
    Wv: [mat(D, DH, r, 0.7), mat(D, DH, r, 0.7)],
    Wo: mat(D, D, r, 0.5),
    W1: mat(D, FF, r, 0.5), b1: Array(FF).fill(0),
    W2: mat(FF, D, r, 0.5), b2: Array(D).fill(0),
    g1: Array(D).fill(1), g2: Array(D).fill(1), gf: Array(D).fill(1),
  };
}

export const positional = (n: number): M => Array.from({ length: n }, (_, p) => Array.from({ length: D }, (_, i) => { const k = Math.floor(i / 2); const w = 1 / Math.pow(10000, (2 * k) / D); return i % 2 === 0 ? Math.sin(p * w) : Math.cos(p * w); }));

export function layerNorm(X: M, g: number[], eps = 1e-5): M {
  return X.map((x) => {
    const mu = x.reduce((a, b) => a + b, 0) / x.length;
    const v = x.reduce((a, b) => a + (b - mu) ** 2, 0) / x.length;
    return x.map((xi, i) => (g[i] * (xi - mu)) / Math.sqrt(v + eps));
  });
}

const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

export interface Trace {
  tokens: string[];
  ids: number[];
  emb: M;
  pos: M;
  x0: M;
  ln1: M;
  heads: { Q: M; K: M; V: M; scores: M; masked: M; weights: M; out: M }[];
  concat: M;
  attnOut: M;
  res1: M;
  ln2: M;
  ffHidden: M;
  ffOut: M;
  res2: M;
  lnf: M;
  logits: number[];
  probs: number[];
}

/** Pre-LN decoder block forward pass with causal masking. */
export function forwardTrace(ids: number[], W: Weights, temperature = 1): Trace {
  const n = ids.length;
  const emb = ids.map((i) => W.E[i].slice());
  const pos = positional(n);
  const x0 = emb.map((e, p) => e.map((v, i) => v + pos[p][i]));
  const ln1 = layerNorm(x0, W.g1);
  const heads = [0, 1].map((h) => {
    const Q = mm(ln1, W.Wq[h]), K = mm(ln1, W.Wk[h]), V = mm(ln1, W.Wv[h]);
    const scores = mm(Q, tr(K)).map((r) => r.map((v) => v / Math.sqrt(DH)));
    const masked = scores.map((r, i) => r.map((v, j) => (j > i ? -Infinity : v)));
    const weights = masked.map((r) => softmax(r.map((v) => (Number.isFinite(v) ? v : -1e9))));
    const out = mm(weights, V);
    return { Q, K, V, scores, masked, weights, out };
  });
  const concat = heads[0].out.map((r, i) => [...r, ...heads[1].out[i]]);
  const attnOut = mm(concat, W.Wo);
  const res1 = x0.map((r, i) => r.map((v, j) => v + attnOut[i][j]));
  const ln2 = layerNorm(res1, W.g2);
  const ffHidden = mm(ln2, W.W1).map((r) => r.map((v, j) => gelu(v + W.b1[j])));
  const ffOut = mm(ffHidden, W.W2).map((r) => r.map((v, j) => v + W.b2[j]));
  const res2 = res1.map((r, i) => r.map((v, j) => v + ffOut[i][j]));
  const lnf = layerNorm(res2, W.gf);
  const last = lnf[n - 1];
  const logits = W.E.map((e) => e.reduce((s, v, k) => s + v * last[k], 0)); // tied unembedding
  const probs = softmax(logits, temperature);
  return { tokens: ids.map((i) => VOCAB[i]), ids, emb, pos, x0, ln1, heads, concat, attnOut, res1, ln2, ffHidden, ffOut, res2, lnf, logits, probs };
}

export function tokenize(text: string): number[] {
  const words = text.toLowerCase().replace(/\./g, " .").split(/\s+/).filter(Boolean);
  return [0, ...words.map((w) => VOCAB.indexOf(w)).filter((i) => i > 0)];
}
