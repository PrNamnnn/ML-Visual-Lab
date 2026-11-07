import { mean, std, ranks, quantile, skewness, extent } from "@/lib/statistics/descriptive";
import { normalQuantile } from "@/lib/math/special";

export interface TransformResult {
  values: number[];
  /** indices of inputs that were invalid for this transform (e.g. log of ≤ 0) */
  invalid: number[];
  note?: string;
}

export interface TransformSpec {
  id: string;
  name: string;
  formula: string;
  intuition: string;
  zeroNegative: string;
  useCases: string;
  limitations: string;
  param?: { key: string; label: string; min: number; max: number; step: number; default: number };
  /** whether a λ can be estimated by maximum likelihood */
  fitLambda?: boolean;
  apply: (x: number[], p: number) => TransformResult;
}

const mapValid = (x: number[], f: (v: number) => number, ok: (v: number) => boolean): TransformResult => {
  const invalid: number[] = [];
  const values = x.map((v, i) => {
    if (!ok(v)) {
      invalid.push(i);
      return NaN;
    }
    return f(v);
  });
  return { values, invalid };
};

/** Box-Cox: (x^λ − 1)/λ for λ≠0, ln x for λ=0. Requires x > 0. */
export function boxCox(v: number, lambda: number): number {
  return Math.abs(lambda) < 1e-8 ? Math.log(v) : (Math.pow(v, lambda) - 1) / lambda;
}

/** Yeo–Johnson: defined for all real x. */
export function yeoJohnson(v: number, lambda: number): number {
  if (v >= 0) return Math.abs(lambda) < 1e-8 ? Math.log1p(v) : (Math.pow(v + 1, lambda) - 1) / lambda;
  return Math.abs(lambda - 2) < 1e-8 ? -Math.log1p(-v) : -(Math.pow(1 - v, 2 - lambda) - 1) / (2 - lambda);
}

/** Profile log-likelihood for Box-Cox λ (up to a constant). */
export function boxCoxLogLik(x: number[], lambda: number): number {
  const n = x.length;
  const y = x.map((v) => boxCox(v, lambda));
  const m = mean(y);
  const s2 = y.reduce((a, v) => a + (v - m) ** 2, 0) / n;
  const sumLog = x.reduce((a, v) => a + Math.log(v), 0);
  return (-n / 2) * Math.log(s2) + (lambda - 1) * sumLog;
}

export function yeoJohnsonLogLik(x: number[], lambda: number): number {
  const n = x.length;
  const y = x.map((v) => yeoJohnson(v, lambda));
  const m = mean(y);
  const s2 = y.reduce((a, v) => a + (v - m) ** 2, 0) / n;
  const jac = x.reduce((a, v) => a + Math.sign(v) * Math.log1p(Math.abs(v)), 0);
  return (-n / 2) * Math.log(s2) + (lambda - 1) * jac;
}

/** Golden-section search for the MLE of λ on [-3, 3]. */
export function fitLambda(ll: (l: number) => number, lo = -3, hi = 3): number {
  const g = (Math.sqrt(5) - 1) / 2;
  let a = lo;
  let b = hi;
  let c = b - g * (b - a);
  let d = a + g * (b - a);
  for (let i = 0; i < 80; i++) {
    if (ll(c) > ll(d)) b = d;
    else a = c;
    c = b - g * (b - a);
    d = a + g * (b - a);
  }
  return (a + b) / 2;
}

export const transforms: TransformSpec[] = [
  {
    id: "log",
    name: "Log transform",
    formula: "y = \\ln(x)",
    intuition: "Turns multiplicative differences into additive ones: equal ratios become equal distances, compressing the long right tail.",
    zeroNegative: "Undefined for x ≤ 0. Those points are flagged and dropped here — shift the data or use log1p / Yeo-Johnson instead.",
    useCases: "Right-skewed positive data: incomes, prices, populations, latencies, counts spanning orders of magnitude.",
    limitations: "Fails on zero/negative values; can over-correct mild skew into left skew; interpretation moves to the log scale.",
    apply: (x) => mapValid(x, Math.log, (v) => v > 0),
  },
  {
    id: "log1p",
    name: "Log1p transform",
    formula: "y = \\ln(1 + x)",
    intuition: "Like the log, but shifted by one so that 0 maps to 0. Near zero it is approximately the identity (ln(1+x) ≈ x).",
    zeroNegative: "Defined for x > −1. Zeros are safe — ideal for count data with many zeros. Values ≤ −1 are invalid.",
    useCases: "Count features (views, purchases), sparse non-negative features, word counts.",
    limitations: "The +1 is arbitrary relative to the data's scale: for values ≪ 1 it barely changes anything.",
    apply: (x) => mapValid(x, Math.log1p, (v) => v > -1),
  },
  {
    id: "sqrt",
    name: "Square-root transform",
    formula: "y = \\sqrt{x}",
    intuition: "A milder compression than the log. Variance-stabilising for Poisson counts (variance ∝ mean).",
    zeroNegative: "Defined for x ≥ 0; negative values are invalid.",
    useCases: "Moderately right-skewed counts; Poisson-like data.",
    limitations: "Weaker than log for heavy skew; still undefined for negatives.",
    apply: (x) => mapValid(x, Math.sqrt, (v) => v >= 0),
  },
  {
    id: "cbrt",
    name: "Cube-root transform",
    formula: "y = \\sqrt[3]{x}",
    intuition: "Compresses large magnitudes symmetrically on both sides of zero, preserving the sign.",
    zeroNegative: "Defined for every real number — negatives stay negative.",
    useCases: "Skewed data containing negative values (e.g. net balances, profit/loss).",
    limitations: "Weaker effect than log; distorts values near zero (steep slope at 0).",
    apply: (x) => mapValid(x, Math.cbrt, () => true),
  },
  {
    id: "reciprocal",
    name: "Reciprocal transform",
    formula: "y = 1/x",
    intuition: "Very strong compression that reverses order: large values become small. Rates ↔ times (km/h ↔ h/km).",
    zeroNegative: "Undefined at x = 0 (flagged). Sign changes break monotonicity across zero.",
    useCases: "Extremely right-skewed positive data; converting between rate and duration.",
    limitations: "Reverses ordering; explodes near zero; rarely appropriate for data crossing zero.",
    apply: (x) => mapValid(x, (v) => 1 / v, (v) => v !== 0),
  },
  {
    id: "power",
    name: "Power transform",
    formula: "y = x^{p}",
    intuition: "p < 1 compresses the upper tail (reduces right skew); p > 1 stretches it (reduces left skew).",
    zeroNegative: "For non-integer p, negative x is undefined (flagged). x = 0 is invalid when p < 0.",
    useCases: "Hand-tuned skew correction; making relationships more linear.",
    limitations: "Choice of p is manual; Box-Cox automates this choice.",
    param: { key: "p", label: "p", min: -1, max: 3, step: 0.05, default: 0.5 },
    apply: (x, p) => mapValid(x, (v) => Math.pow(v, p), (v) => (Number.isInteger(p) ? !(p < 0 && v === 0) : v > 0 || (v === 0 && p > 0))),
  },
  {
    id: "box-cox",
    name: "Box-Cox transform",
    formula: "y^{(\\lambda)} = \\begin{cases} \\dfrac{x^{\\lambda}-1}{\\lambda} & \\lambda \\ne 0 \\\\ \\ln x & \\lambda = 0 \\end{cases}",
    intuition: "A family of power transforms indexed by λ that includes log (λ=0), sqrt (λ=0.5, up to scale) and identity (λ=1). λ is chosen by maximum likelihood to make data as Gaussian as possible.",
    zeroNegative: "Requires strictly positive x. Non-positive points are flagged and excluded.",
    useCases: "Normalising positive skewed features for linear models; stabilising variance.",
    limitations: "Positive data only; λ is estimated on training data (must be re-used, not refit, at inference).",
    param: { key: "lambda", label: "\\lambda", min: -2, max: 2, step: 0.01, default: 0 },
    fitLambda: true,
    apply: (x, l) => mapValid(x, (v) => boxCox(v, l), (v) => v > 0),
  },
  {
    id: "yeo-johnson",
    name: "Yeo-Johnson transform",
    formula:
      "y = \\begin{cases} \\frac{(x+1)^{\\lambda}-1}{\\lambda} & x\\ge0,\\lambda\\ne0 \\\\ \\ln(x+1) & x\\ge0,\\lambda=0 \\\\ -\\frac{(1-x)^{2-\\lambda}-1}{2-\\lambda} & x<0,\\lambda\\ne2 \\\\ -\\ln(1-x) & x<0,\\lambda=2 \\end{cases}",
    intuition: "Extends Box-Cox to the whole real line by treating positive and negative values with mirrored power functions.",
    zeroNegative: "Defined for all real x — no shifting needed.",
    useCases: "Default choice in scikit-learn's PowerTransformer; skewed features containing zeros or negatives.",
    limitations: "Less interpretable than log; λ must be fitted on training data only.",
    param: { key: "lambda", label: "\\lambda", min: -2, max: 3, step: 0.01, default: 0.5 },
    fitLambda: true,
    apply: (x, l) => mapValid(x, (v) => yeoJohnson(v, l), () => true),
  },
  {
    id: "exp",
    name: "Exponential transform",
    formula: "y = e^{c\\,x}",
    intuition: "The inverse of the log: stretches the upper tail. Useful to undo a log transform or to correct left skew.",
    zeroNegative: "Defined everywhere, but overflows for large c·x (values above ~709 become ∞, flagged).",
    useCases: "Correcting left-skewed data; back-transforming log-scale predictions.",
    limitations: "Amplifies outliers dramatically; numerically unstable for large inputs.",
    param: { key: "c", label: "c", min: 0.05, max: 1.5, step: 0.05, default: 0.3 },
    apply: (x, c) => mapValid(x, (v) => Math.exp(c * v), (v) => c * v < 709),
  },
  {
    id: "quantile",
    name: "Quantile transform",
    formula: "y = \\Phi^{-1}\\big(\\hat F_n(x)\\big)",
    intuition: "Maps each value to its empirical percentile, then through the inverse normal CDF. Any continuous distribution becomes (approximately) standard normal.",
    zeroNegative: "Defined for all values; only the order matters.",
    useCases: "Forcing Gaussian-shaped inputs; robust to outliers since only ranks are used.",
    limitations: "Non-linear and non-parametric: distorts distances between values, and saturates on unseen extremes.",
    apply: (x) => {
      const r = ranks(x);
      const n = x.length;
      return { values: r.map((ri) => normalQuantile((ri - 0.5) / n)), invalid: [] };
    },
  },
  {
    id: "rank",
    name: "Rank transform",
    formula: "y_i = \\operatorname{rank}(x_i)/n",
    intuition: "Replaces values by their (average) rank, scaled to (0, 1]. The output is uniform regardless of the input shape.",
    zeroNegative: "Defined for all values. Ties receive the average rank.",
    useCases: "Spearman correlation; robust non-parametric features; tree-model-friendly normalisation.",
    limitations: "Discards magnitudes entirely; new data needs the training ranks to be mapped consistently.",
    apply: (x) => {
      const r = ranks(x);
      return { values: r.map((v) => v / x.length), invalid: [] };
    },
  },
  {
    id: "binning",
    name: "Binning / discretisation",
    formula: "y = \\left\\lfloor \\dfrac{x - x_{\\min}}{(x_{\\max}-x_{\\min})/B} \\right\\rfloor",
    intuition: "Groups a continuous variable into B equal-width bins (ordinal codes 0…B−1). Removes fine-grained noise and outlier magnitude.",
    zeroNegative: "Defined for all values.",
    useCases: "Age groups, credit scoring bands, histogram features, making non-linear effects learnable by linear models.",
    limitations: "Loses information inside each bin; bin edges are arbitrary and can split similar values.",
    param: { key: "bins", label: "B", min: 2, max: 20, step: 1, default: 6 },
    apply: (x, B) => {
      const [lo, hi] = extent(x);
      const w = (hi - lo) / B || 1;
      return { values: x.map((v) => Math.min(B - 1, Math.floor((v - lo) / w))), invalid: [] };
    },
  },
];

export const transformById = (id: string) => transforms.find((t) => t.id === id);

/* ------------------------------ feature scaling ------------------------------ */

export interface ScalerSpec {
  id: string;
  name: string;
  formula: string;
  description: string;
  apply: (x: number[]) => { values: number[]; note?: string };
}

export const scalers: ScalerSpec[] = [
  {
    id: "minmax",
    name: "Min-Max normalisation",
    formula: "x' = \\dfrac{x - x_{\\min}}{x_{\\max} - x_{\\min}}",
    description: "Linearly maps the range to [0, 1]. Sensitive to outliers: one extreme value squashes everything else.",
    apply: (x) => {
      const [lo, hi] = extent(x);
      if (hi - lo < 1e-12) return { values: x.map(() => 0), note: "Zero range — all values map to 0." };
      return { values: x.map((v) => (v - lo) / (hi - lo)) };
    },
  },
  {
    id: "standard",
    name: "Standardisation (z-score)",
    formula: "x' = \\dfrac{x - \\mu}{\\sigma}",
    description: "Centres to mean 0 and unit variance. Shape (skewness) is unchanged. Preferred for gradient-based models, PCA, SVMs.",
    apply: (x) => {
      const m = mean(x);
      const s = std(x);
      if (!(s > 1e-12)) return { values: x.map(() => 0), note: "Zero variance — feature carries no information." };
      return { values: x.map((v) => (v - m) / s) };
    },
  },
  {
    id: "robust",
    name: "Robust scaling",
    formula: "x' = \\dfrac{x - \\operatorname{median}(x)}{Q_3 - Q_1}",
    description: "Uses median and IQR, which outliers barely affect. The bulk of the data keeps a sensible scale.",
    apply: (x) => {
      const med = quantile(x, 0.5);
      const iqr = quantile(x, 0.75) - quantile(x, 0.25);
      if (iqr < 1e-12) return { values: x.map((v) => v - med), note: "IQR is zero — only centring applied." };
      return { values: x.map((v) => (v - med) / iqr) };
    },
  },
  {
    id: "maxabs",
    name: "MaxAbs scaling",
    formula: "x' = \\dfrac{x}{\\max_i |x_i|}",
    description: "Scales to [−1, 1] without shifting, so zeros stay zero — preserves sparsity.",
    apply: (x) => {
      const m = Math.max(...x.map(Math.abs));
      if (m < 1e-12) return { values: x.slice(), note: "All values are zero." };
      return { values: x.map((v) => v / m) };
    },
  },
  {
    id: "l2",
    name: "Unit-vector (L2) normalisation",
    formula: "x' = \\dfrac{x}{\\lVert x \\rVert_2}",
    description: "Divides the whole vector by its Euclidean length (applied per sample in practice). Here the feature column is treated as one vector.",
    apply: (x) => {
      const n = Math.hypot(...x);
      if (n < 1e-12) return { values: x.slice(), note: "Zero vector cannot be normalised." };
      return { values: x.map((v) => v / n) };
    },
  },
  {
    id: "decimal",
    name: "Decimal scaling",
    formula: "x' = \\dfrac{x}{10^{j}},\\; j = \\min\\{j : \\max|x'| < 1\\}",
    description: "Moves the decimal point so all magnitudes are below 1. Simple and invertible, but coarse.",
    apply: (x) => {
      const m = Math.max(...x.map(Math.abs));
      const j = m < 1 ? 0 : Math.floor(Math.log10(m)) + 1;
      return { values: x.map((v) => v / 10 ** j), note: `j = ${j}` };
    },
  },
];

export { skewness };
