/** Neural-network building blocks: activations, losses, a small MLP with exact backprop. */
import { mulberry32, randn } from "@/lib/math/random";
import { erf } from "@/lib/math/special";

export interface Activation {
  id: string;
  name: string;
  tex: string;
  dtex: string;
  range: string;
  f: (x: number, a?: number) => number;
  df: (x: number, a?: number) => number;
  param?: { label: string; min: number; max: number; step: number; default: number };
  notes: string;
}

const SELU_L = 1.0507009873554805;
const SELU_A = 1.6732632423543772;
const sig = (x: number) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
const phi = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
const Phi = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2));

export const activations: Activation[] = [
  { id: "linear", name: "Linear", tex: "f(x)=x", dtex: "f'(x)=1", range: "(-\\infty,\\infty)", f: (x) => x, df: () => 1, notes: "No non-linearity: stacking linear layers collapses into one linear map." },
  { id: "sigmoid", name: "Sigmoid", tex: "\\sigma(x)=\\dfrac{1}{1+e^{-x}}", dtex: "\\sigma'(x)=\\sigma(x)(1-\\sigma(x))", range: "(0,1)", f: sig, df: (x) => sig(x) * (1 - sig(x)), notes: "Max gradient 0.25 at x=0; saturates for |x| ≳ 5 → vanishing gradients in deep nets." },
  { id: "tanh", name: "Tanh", tex: "\\tanh(x)=\\dfrac{e^x-e^{-x}}{e^x+e^{-x}}", dtex: "1-\\tanh^2(x)", range: "(-1,1)", f: Math.tanh, df: (x) => 1 - Math.tanh(x) ** 2, notes: "Zero-centred sigmoid (tanh(x) = 2σ(2x) − 1); still saturates." },
  { id: "relu", name: "ReLU", tex: "\\max(0,x)", dtex: "\\mathbb{1}[x>0]", range: "[0,\\infty)", f: (x) => Math.max(0, x), df: (x) => (x > 0 ? 1 : 0), notes: "Gradient 1 for positive inputs; units with always-negative input 'die' (zero gradient)." },
  { id: "leaky-relu", name: "Leaky ReLU", tex: "\\max(\\alpha x, x)", dtex: "\\begin{cases}1&x>0\\\\\\alpha&x\\le0\\end{cases}", range: "(-\\infty,\\infty)", f: (x, a = 0.1) => (x > 0 ? x : a * x), df: (x, a = 0.1) => (x > 0 ? 1 : a), param: { label: "\\alpha", min: 0, max: 0.5, step: 0.01, default: 0.1 }, notes: "Small negative slope keeps gradient flowing for negative inputs." },
  { id: "prelu", name: "PReLU", tex: "\\begin{cases}x&x>0\\\\a x&x\\le0\\end{cases}", dtex: "\\partial f/\\partial a = x\\,\\mathbb 1[x\\le0]", range: "(-\\infty,\\infty)", f: (x, a = 0.25) => (x > 0 ? x : a * x), df: (x, a = 0.25) => (x > 0 ? 1 : a), param: { label: "a", min: 0, max: 1, step: 0.01, default: 0.25 }, notes: "Leaky ReLU where the slope a is a learned parameter." },
  { id: "elu", name: "ELU", tex: "\\begin{cases}x&x>0\\\\\\alpha(e^x-1)&x\\le0\\end{cases}", dtex: "\\begin{cases}1&x>0\\\\\\alpha e^x&x\\le0\\end{cases}", range: "(-\\alpha,\\infty)", f: (x, a = 1) => (x > 0 ? x : a * (Math.exp(x) - 1)), df: (x, a = 1) => (x > 0 ? 1 : a * Math.exp(x)), param: { label: "\\alpha", min: 0.1, max: 2, step: 0.05, default: 1 }, notes: "Smooth negative saturation pushes mean activations toward zero." },
  { id: "selu", name: "SELU", tex: "\\lambda\\begin{cases}x&x>0\\\\\\alpha(e^x-1)&x\\le0\\end{cases}", dtex: "\\lambda\\begin{cases}1&x>0\\\\\\alpha e^x&x\\le0\\end{cases}", range: "(-\\lambda\\alpha,\\infty)", f: (x) => SELU_L * (x > 0 ? x : SELU_A * (Math.exp(x) - 1)), df: (x) => SELU_L * (x > 0 ? 1 : SELU_A * Math.exp(x)), notes: "λ≈1.0507, α≈1.6733 chosen so activations self-normalise to mean 0 / variance 1 (with LeCun-normal init)." },
  { id: "gelu", name: "GELU", tex: "x\\,\\Phi(x)", dtex: "\\Phi(x)+x\\,\\varphi(x)", range: "[\\approx-0.17,\\infty)", f: (x) => x * Phi(x), df: (x) => Phi(x) + x * phi(x), notes: "Weights the input by the probability a standard normal is below it. Default in BERT and GPT." },
  { id: "swish", name: "Swish / SiLU", tex: "x\\,\\sigma(\\beta x)", dtex: "\\sigma(\\beta x)+\\beta x\\,\\sigma(\\beta x)(1-\\sigma(\\beta x))", range: "[\\approx-0.28,\\infty)", f: (x, b = 1) => x * sig(b * x), df: (x, b = 1) => sig(b * x) + b * x * sig(b * x) * (1 - sig(b * x)), param: { label: "\\beta", min: 0.1, max: 5, step: 0.1, default: 1 }, notes: "Smooth, non-monotonic. β=1 is SiLU (used in LLaMA's SwiGLU); β→∞ approaches ReLU." },
];

export function softmax(z: number[], temperature = 1): number[] {
  const t = Math.max(1e-6, temperature);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp((v - m) / t));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/* ------------------------------ losses ------------------------------ */

export interface LossSpec {
  id: string;
  name: string;
  tex: string;
  kind: "regression" | "binary" | "multiclass" | "distribution" | "metric";
  notes: string;
}

export const losses: LossSpec[] = [
  { id: "mse", name: "Mean squared error", tex: "L=(y-\\hat y)^2", kind: "regression", notes: "Penalises large errors quadratically; MLE under Gaussian noise. Sensitive to outliers." },
  { id: "mae", name: "Mean absolute error", tex: "L=|y-\\hat y|", kind: "regression", notes: "Linear penalty; MLE under Laplace noise; gradient has constant magnitude (non-smooth at 0)." },
  { id: "huber", name: "Huber loss", tex: "L_\\delta=\\begin{cases}\\tfrac12 r^2 & |r|\\le\\delta\\\\ \\delta(|r|-\\tfrac12\\delta)&|r|>\\delta\\end{cases},\\; r=y-\\hat y", kind: "regression", notes: "Quadratic near zero, linear in the tails: robust to outliers yet smooth." },
  { id: "bce", name: "Binary cross-entropy", tex: "L=-[y\\log p+(1-y)\\log(1-p)]", kind: "binary", notes: "Negative log-likelihood of a Bernoulli. Goes to ∞ as a confident prediction is wrong." },
  { id: "hinge", name: "Hinge loss", tex: "L=\\max(0,\\,1-t\\cdot s),\\; t\\in\\{-1,1\\}", kind: "binary", notes: "Used by SVMs. Zero once the margin t·s exceeds 1 — correctly classified points far from the boundary contribute nothing." },
  { id: "focal", name: "Focal loss", tex: "L=-(1-p_t)^{\\gamma}\\log p_t", kind: "binary", notes: "Down-weights easy examples (p_t near 1). γ=0 recovers cross-entropy. Used for class imbalance in detection (RetinaNet)." },
  { id: "cce", name: "Categorical cross-entropy", tex: "L=-\\sum_k y_k\\log p_k,\\; p=\\operatorname{softmax}(z)", kind: "multiclass", notes: "With a one-hot y this is −log p of the true class. 'Sparse' CCE is identical but takes the class index instead of a one-hot vector." },
  { id: "kl", name: "KL divergence", tex: "D_{KL}(P\\|Q)=\\sum_k P_k\\log\\dfrac{P_k}{Q_k}", kind: "distribution", notes: "Asymmetric, ≥ 0, zero iff P=Q. Used in VAEs, distillation and RLHF penalties." },
  { id: "contrastive", name: "Contrastive loss", tex: "L=y\\,d^2+(1-y)\\max(0,m-d)^2", kind: "metric", notes: "Pulls similar pairs (y=1) together; pushes dissimilar pairs apart until they are at least margin m away." },
  { id: "triplet", name: "Triplet loss", tex: "L=\\max(0,\\,d(a,p)-d(a,n)+m)", kind: "metric", notes: "Anchor must be closer to the positive than to the negative by margin m. Used in face recognition / embeddings." },
];

export const lossFns = {
  mse: (y: number, yh: number) => (y - yh) ** 2,
  mae: (y: number, yh: number) => Math.abs(y - yh),
  huber: (y: number, yh: number, d = 1) => { const r = Math.abs(y - yh); return r <= d ? 0.5 * r * r : d * (r - 0.5 * d); },
  bce: (y: number, p: number) => { const q = Math.min(1 - 1e-12, Math.max(1e-12, p)); return -(y * Math.log(q) + (1 - y) * Math.log(1 - q)); },
  hinge: (t: number, s: number) => Math.max(0, 1 - t * s),
  focal: (y: number, p: number, g = 2) => { const q = Math.min(1 - 1e-12, Math.max(1e-12, p)); const pt = y ? q : 1 - q; return -((1 - pt) ** g) * Math.log(pt); },
  contrastive: (y: number, d: number, m = 1) => y * d * d + (1 - y) * Math.max(0, m - d) ** 2,
  triplet: (dap: number, dan: number, m = 0.5) => Math.max(0, dap - dan + m),
};

export const crossEntropy = (pTrue: number[], q: number[]) => -pTrue.reduce((s, p, i) => s + (p > 0 ? p * Math.log(Math.max(1e-12, q[i])) : 0), 0);
export const klDivergence = (p: number[], q: number[]) => p.reduce((s, pi, i) => s + (pi > 0 ? pi * Math.log(pi / Math.max(1e-12, q[i])) : 0), 0);

/* ------------------------------ MLP ------------------------------ */

export type ActId = "tanh" | "relu" | "sigmoid" | "linear";
const actF: Record<ActId, (x: number) => number> = { tanh: Math.tanh, relu: (x) => Math.max(0, x), sigmoid: sig, linear: (x) => x };
const actD: Record<ActId, (y: number, x: number) => number> = {
  tanh: (y) => 1 - y * y,
  relu: (_, x) => (x > 0 ? 1 : 0),
  sigmoid: (y) => y * (1 - y),
  linear: () => 1,
};

export interface MLP {
  sizes: number[];
  W: number[][][]; // W[l][j][i]: weight from unit i of layer l to unit j of layer l+1
  b: number[][];
  act: ActId;
}

export function createMLP(sizes: number[], act: ActId, seed = 1, init: "xavier" | "he" | "small" = "xavier"): MLP {
  const rng = mulberry32(seed);
  const W: number[][][] = [];
  const b: number[][] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const fanIn = sizes[l];
    const fanOut = sizes[l + 1];
    const s = init === "he" ? Math.sqrt(2 / fanIn) : init === "xavier" ? Math.sqrt(2 / (fanIn + fanOut)) : 0.1;
    W.push(Array.from({ length: fanOut }, () => Array.from({ length: fanIn }, () => randn(rng) * s)));
    b.push(Array(fanOut).fill(0));
  }
  return { sizes, W, b, act };
}

/** Forward pass. Hidden layers use `act`; output layer is sigmoid (binary classification). */
export function forward(net: MLP, x: number[]) {
  const zs: number[][] = [];
  const as: number[][] = [x];
  let a = x;
  for (let l = 0; l < net.W.length; l++) {
    const z = net.W[l].map((row, j) => row.reduce((s, w, i) => s + w * a[i], net.b[l][j]));
    const last = l === net.W.length - 1;
    a = z.map((v) => (last ? sig(v) : actF[net.act](v)));
    zs.push(z);
    as.push(a);
  }
  return { zs, as, out: a[0] };
}

/** One full-batch gradient step on binary cross-entropy (+ optional L2). Returns mean loss and per-layer gradient norms. */
export function trainStep(net: MLP, X: number[][], Y: number[], lr: number, l2 = 0) {
  const gW = net.W.map((L) => L.map((r) => r.map(() => 0)));
  const gb = net.b.map((L) => L.map(() => 0));
  let loss = 0;
  for (let n = 0; n < X.length; n++) {
    const { zs, as, out } = forward(net, X[n]);
    loss += lossFns.bce(Y[n], out);
    // δ at output for sigmoid + BCE is (a − y)
    let delta = [out - Y[n]];
    for (let l = net.W.length - 1; l >= 0; l--) {
      const aPrev = as[l];
      for (let j = 0; j < delta.length; j++) {
        gb[l][j] += delta[j];
        for (let i = 0; i < aPrev.length; i++) gW[l][j][i] += delta[j] * aPrev[i];
      }
      if (l > 0) {
        const nd = aPrev.map((_, i) => {
          let s = 0;
          for (let j = 0; j < delta.length; j++) s += net.W[l][j][i] * delta[j];
          return s * actD[net.act](aPrev[i], zs[l - 1][i]);
        });
        delta = nd;
      }
    }
  }
  const N = X.length;
  const gradNorms: number[] = [];
  for (let l = 0; l < net.W.length; l++) {
    let gn = 0;
    for (let j = 0; j < net.W[l].length; j++) {
      for (let i = 0; i < net.W[l][j].length; i++) {
        const g = gW[l][j][i] / N + l2 * net.W[l][j][i];
        gn += g * g;
        net.W[l][j][i] -= lr * g;
      }
      net.b[l][j] -= (lr * gb[l][j]) / N;
    }
    gradNorms.push(Math.sqrt(gn));
  }
  const reg = l2 ? (l2 / 2) * net.W.flat(2).reduce((s, w) => s + w * w, 0) : 0;
  return { loss: loss / N + reg, gradNorms };
}

export function cloneMLP(n: MLP): MLP {
  return { sizes: n.sizes.slice(), act: n.act, W: n.W.map((L) => L.map((r) => r.slice())), b: n.b.map((r) => r.slice()) };
}
