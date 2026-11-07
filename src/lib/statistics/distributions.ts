import { logChoose, logGamma, logBeta, regularizedBeta, regularizedGammaP, normalCdf, invertCdf } from "@/lib/math/special";
import { randn, randGamma, type RNG } from "@/lib/math/random";

export interface ParamSpec {
  key: string;
  label: string; // KaTeX-able symbol
  min: number;
  max: number;
  step: number;
  default: number;
  integer?: boolean;
}

export interface Distribution {
  id: string;
  name: string;
  kind: "discrete" | "continuous";
  params: ParamSpec[];
  /** KaTeX string for the PMF/PDF */
  formula: string;
  support: string;
  /** density (continuous) or mass (discrete) */
  density: (x: number, p: Record<string, number>) => number;
  cdf: (x: number, p: Record<string, number>) => number;
  mean: (p: Record<string, number>) => number;
  variance: (p: Record<string, number>) => number;
  /** plotting range */
  range: (p: Record<string, number>) => [number, number];
  sample: (rng: RNG, p: Record<string, number>) => number;
  /** validation: returns error message or null */
  validate?: (p: Record<string, number>) => string | null;
  notes: string;
  mlUse: string;
}

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

const sumPmf = (pmf: (k: number) => number, upTo: number) => {
  let s = 0;
  for (let k = 0; k <= Math.floor(upTo); k++) s += pmf(k);
  return clamp01(s);
};

/** Sample a discrete distribution by inversion of its CDF over integers. */
const invertDiscrete = (rng: RNG, cdf: (k: number) => number, lo = 0, hi = 10000) => {
  const u = rng();
  for (let k = lo; k <= hi; k++) if (cdf(k) >= u) return k;
  return hi;
};

export const distributions: Distribution[] = [
  {
    id: "bernoulli",
    name: "Bernoulli",
    kind: "discrete",
    params: [{ key: "p", label: "p", min: 0, max: 1, step: 0.01, default: 0.3 }],
    formula: "P(X=k)=p^k(1-p)^{1-k},\\quad k\\in\\{0,1\\}",
    support: "k \\in \\{0,1\\}",
    density: (x, { p }) => (x === 1 ? p : x === 0 ? 1 - p : 0),
    cdf: (x, { p }) => (x < 0 ? 0 : x < 1 ? 1 - p : 1),
    mean: ({ p }) => p,
    variance: ({ p }) => p * (1 - p),
    range: () => [0, 1],
    sample: (rng, { p }) => (rng() < p ? 1 : 0),
    notes: "A single yes/no trial with success probability p.",
    mlUse: "The likelihood behind binary cross-entropy and logistic regression outputs.",
  },
  {
    id: "binomial",
    name: "Binomial",
    kind: "discrete",
    params: [
      { key: "n", label: "n", min: 1, max: 60, step: 1, default: 20, integer: true },
      { key: "p", label: "p", min: 0, max: 1, step: 0.01, default: 0.4 },
    ],
    formula: "P(X=k)=\\binom{n}{k}p^k(1-p)^{n-k}",
    support: "k \\in \\{0,\\dots,n\\}",
    density: (k, { n, p }) => {
      if (k < 0 || k > n || !Number.isInteger(k)) return 0;
      if (p === 0) return k === 0 ? 1 : 0;
      if (p === 1) return k === n ? 1 : 0;
      return Math.exp(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
    },
    cdf(x, prm) {
      if (x < 0) return 0;
      return sumPmf((k) => this.density(k, prm), Math.min(x, prm.n));
    },
    mean: ({ n, p }) => n * p,
    variance: ({ n, p }) => n * p * (1 - p),
    range: ({ n }) => [0, n],
    sample: (rng, { n, p }) => {
      let s = 0;
      for (let i = 0; i < n; i++) if (rng() < p) s++;
      return s;
    },
    notes: "Number of successes in n independent Bernoulli(p) trials.",
    mlUse: "Counting correct predictions on a test set; the basis of accuracy confidence intervals.",
  },
  {
    id: "geometric",
    name: "Geometric",
    kind: "discrete",
    params: [{ key: "p", label: "p", min: 0.02, max: 1, step: 0.01, default: 0.25 }],
    formula: "P(X=k)=(1-p)^{k-1}p,\\quad k=1,2,\\dots",
    support: "k \\in \\{1,2,\\dots\\}",
    density: (k, { p }) => (k >= 1 && Number.isInteger(k) ? Math.pow(1 - p, k - 1) * p : 0),
    cdf: (x, { p }) => (x < 1 ? 0 : 1 - Math.pow(1 - p, Math.floor(x))),
    mean: ({ p }) => 1 / p,
    variance: ({ p }) => (1 - p) / (p * p),
    range: ({ p }) => [1, Math.max(5, Math.ceil(Math.log(0.002) / Math.log(Math.max(1e-9, 1 - p))))],
    sample: (rng, { p }) => (p >= 1 ? 1 : Math.max(1, Math.ceil(Math.log(1 - rng()) / Math.log(1 - p)))),
    notes: "Number of trials until (and including) the first success. Memoryless.",
    mlUse: "Modelling waiting times in discrete steps, e.g. number of retries until success.",
  },
  {
    id: "poisson",
    name: "Poisson",
    kind: "discrete",
    params: [{ key: "lambda", label: "\\lambda", min: 0.1, max: 30, step: 0.1, default: 4 }],
    formula: "P(X=k)=\\dfrac{\\lambda^k e^{-\\lambda}}{k!}",
    support: "k \\in \\{0,1,2,\\dots\\}",
    density: (k, { lambda }) => (k >= 0 && Number.isInteger(k) ? Math.exp(k * Math.log(lambda) - lambda - logGamma(k + 1)) : 0),
    cdf: (x, { lambda }) => (x < 0 ? 0 : 1 - regularizedGammaP(Math.floor(x) + 1, lambda)),
    mean: ({ lambda }) => lambda,
    variance: ({ lambda }) => lambda,
    range: ({ lambda }) => [0, Math.ceil(lambda + 4.5 * Math.sqrt(lambda) + 3)],
    sample(rng, { lambda }) {
      return invertDiscrete(rng, (k) => this.cdf(k, { lambda }), 0, 2000);
    },
    notes: "Counts of events in a fixed interval when events occur independently at constant rate λ.",
    mlUse: "Poisson regression for count targets (clicks, arrivals); variance equals the mean.",
  },
  {
    id: "discrete-uniform",
    name: "Discrete uniform",
    kind: "discrete",
    params: [
      { key: "a", label: "a", min: 0, max: 10, step: 1, default: 1, integer: true },
      { key: "b", label: "b", min: 1, max: 20, step: 1, default: 6, integer: true },
    ],
    formula: "P(X=k)=\\dfrac{1}{b-a+1},\\quad k\\in\\{a,\\dots,b\\}",
    support: "k \\in \\{a,\\dots,b\\}",
    density: (k, { a, b }) => (k >= a && k <= b && Number.isInteger(k) ? 1 / (b - a + 1) : 0),
    cdf: (x, { a, b }) => (x < a ? 0 : x >= b ? 1 : (Math.floor(x) - a + 1) / (b - a + 1)),
    mean: ({ a, b }) => (a + b) / 2,
    variance: ({ a, b }) => ((b - a + 1) ** 2 - 1) / 12,
    range: ({ a, b }) => [a, b],
    sample: (rng, { a, b }) => a + Math.floor(rng() * (b - a + 1)),
    validate: ({ a, b }) => (b < a ? "Requires b ≥ a" : null),
    notes: "Every integer in [a, b] is equally likely — e.g. a fair die (a=1, b=6).",
    mlUse: "Random search over integer hyperparameters; uniform random sampling of indices.",
  },
  {
    id: "categorical",
    name: "Categorical",
    kind: "discrete",
    params: [
      { key: "w1", label: "w_1", min: 0, max: 5, step: 0.1, default: 1 },
      { key: "w2", label: "w_2", min: 0, max: 5, step: 0.1, default: 2.5 },
      { key: "w3", label: "w_3", min: 0, max: 5, step: 0.1, default: 0.8 },
      { key: "w4", label: "w_4", min: 0, max: 5, step: 0.1, default: 1.7 },
    ],
    formula: "P(X=k)=p_k=\\dfrac{w_k}{\\sum_j w_j},\\quad \\sum_k p_k = 1",
    support: "k \\in \\{1,\\dots,K\\}",
    density: (k, w) => {
      const ws = [w.w1, w.w2, w.w3, w.w4];
      const s = ws.reduce((a, b) => a + b, 0);
      return k >= 1 && k <= 4 && Number.isInteger(k) && s > 0 ? ws[k - 1] / s : 0;
    },
    cdf(x, w) {
      let s = 0;
      for (let k = 1; k <= Math.min(4, Math.floor(x)); k++) s += this.density(k, w);
      return clamp01(s);
    },
    mean(w) {
      let m = 0;
      for (let k = 1; k <= 4; k++) m += k * this.density(k, w);
      return m;
    },
    variance(w) {
      const mu = this.mean(w);
      let v = 0;
      for (let k = 1; k <= 4; k++) v += (k - mu) ** 2 * this.density(k, w);
      return v;
    },
    range: () => [1, 4],
    sample(rng, w) {
      return invertDiscrete(rng, (k) => this.cdf(k, w), 1, 4);
    },
    validate: (w) => (w.w1 + w.w2 + w.w3 + w.w4 <= 0 ? "At least one weight must be positive" : null),
    notes: "One draw from K categories; weights are normalised into probabilities (as softmax does).",
    mlUse: "The output distribution of a softmax classifier and of next-token prediction in LLMs.",
  },
  {
    id: "multinomial",
    name: "Multinomial (marginal)",
    kind: "discrete",
    params: [
      { key: "n", label: "n", min: 1, max: 60, step: 1, default: 30, integer: true },
      { key: "p1", label: "p_1", min: 0, max: 1, step: 0.01, default: 0.2 },
    ],
    formula: "P(\\mathbf{X}=\\mathbf{x})=\\dfrac{n!}{x_1!\\cdots x_K!}\\prod_k p_k^{x_k};\\quad X_1\\sim\\text{Bin}(n,p_1)",
    support: "x_1 \\in \\{0,\\dots,n\\}",
    density: (k, { n, p1 }) => distributions[1].density(k, { n, p: p1 }),
    cdf: (x, { n, p1 }) => distributions[1].cdf(x, { n, p: p1 }),
    mean: ({ n, p1 }) => n * p1,
    variance: ({ n, p1 }) => n * p1 * (1 - p1),
    range: ({ n }) => [0, n],
    sample: (rng, { n, p1 }) => distributions[1].sample(rng, { n, p: p1 }),
    notes: "Counts of each of K categories over n draws. Each single count's marginal is Binomial(n, p_k), which is what is plotted.",
    mlUse: "Bag-of-words document models and multinomial Naive Bayes for text classification.",
  },
  {
    id: "uniform",
    name: "Uniform",
    kind: "continuous",
    params: [
      { key: "a", label: "a", min: -5, max: 5, step: 0.1, default: -1 },
      { key: "b", label: "b", min: -4, max: 8, step: 0.1, default: 2 },
    ],
    formula: "f(x)=\\dfrac{1}{b-a},\\quad a\\le x\\le b",
    support: "x\\in[a,b]",
    density: (x, { a, b }) => (x >= a && x <= b && b > a ? 1 / (b - a) : 0),
    cdf: (x, { a, b }) => clamp01((x - a) / (b - a)),
    mean: ({ a, b }) => (a + b) / 2,
    variance: ({ a, b }) => (b - a) ** 2 / 12,
    range: ({ a, b }) => [a - 0.25 * (b - a), b + 0.25 * (b - a)],
    sample: (rng, { a, b }) => a + (b - a) * rng(),
    validate: ({ a, b }) => (b <= a ? "Requires b > a" : null),
    notes: "Constant density on [a, b]: maximum entropy with bounded support.",
    mlUse: "Uniform weight initialisation (Xavier/Glorot uniform) and random search.",
  },
  {
    id: "normal",
    name: "Normal (Gaussian)",
    kind: "continuous",
    params: [
      { key: "mu", label: "\\mu", min: -5, max: 5, step: 0.1, default: 0 },
      { key: "sigma", label: "\\sigma", min: 0.1, max: 4, step: 0.05, default: 1 },
    ],
    formula: "f(x)=\\dfrac{1}{\\sigma\\sqrt{2\\pi}}\\exp\\!\\left(-\\dfrac{(x-\\mu)^2}{2\\sigma^2}\\right)",
    support: "x\\in\\mathbb{R}",
    density: (x, { mu, sigma }) => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)),
    cdf: (x, { mu, sigma }) => normalCdf((x - mu) / sigma),
    mean: ({ mu }) => mu,
    variance: ({ sigma }) => sigma * sigma,
    range: ({ mu, sigma }) => [mu - 4 * sigma, mu + 4 * sigma],
    sample: (rng, { mu, sigma }) => mu + sigma * randn(rng),
    notes: "Symmetric bell curve; ~68% / 95% / 99.7% of mass within 1 / 2 / 3 σ.",
    mlUse: "Gaussian noise assumption behind MSE, weight initialisation, VAEs and diffusion models.",
  },
  {
    id: "standard-normal",
    name: "Standard normal",
    kind: "continuous",
    params: [],
    formula: "\\varphi(z)=\\dfrac{1}{\\sqrt{2\\pi}}e^{-z^2/2},\\quad Z=\\dfrac{X-\\mu}{\\sigma}",
    support: "z\\in\\mathbb{R}",
    density: (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI),
    cdf: (z) => normalCdf(z),
    mean: () => 0,
    variance: () => 1,
    range: () => [-4, 4],
    sample: (rng) => randn(rng),
    notes: "N(0, 1). Any normal variable becomes standard normal after z-scoring.",
    mlUse: "Standardisation of features and the reference distribution for z-tests.",
  },
  {
    id: "exponential",
    name: "Exponential",
    kind: "continuous",
    params: [{ key: "lambda", label: "\\lambda", min: 0.1, max: 5, step: 0.05, default: 1 }],
    formula: "f(x)=\\lambda e^{-\\lambda x},\\quad x\\ge 0",
    support: "x\\ge0",
    density: (x, { lambda }) => (x < 0 ? 0 : lambda * Math.exp(-lambda * x)),
    cdf: (x, { lambda }) => (x < 0 ? 0 : 1 - Math.exp(-lambda * x)),
    mean: ({ lambda }) => 1 / lambda,
    variance: ({ lambda }) => 1 / (lambda * lambda),
    range: ({ lambda }) => [0, 6 / lambda],
    sample: (rng, { lambda }) => -Math.log(1 - rng()) / lambda,
    notes: "Continuous waiting time between Poisson events. Memoryless.",
    mlUse: "Survival / time-to-event modelling; inter-arrival times in queueing for inference servers.",
  },
  {
    id: "gamma",
    name: "Gamma",
    kind: "continuous",
    params: [
      { key: "k", label: "k", min: 0.3, max: 10, step: 0.1, default: 2 },
      { key: "theta", label: "\\theta", min: 0.2, max: 4, step: 0.1, default: 1.5 },
    ],
    formula: "f(x)=\\dfrac{x^{k-1}e^{-x/\\theta}}{\\Gamma(k)\\,\\theta^k},\\quad x>0",
    support: "x>0",
    density: (x, { k, theta }) => (x <= 0 ? 0 : Math.exp((k - 1) * Math.log(x) - x / theta - logGamma(k) - k * Math.log(theta))),
    cdf: (x, { k, theta }) => (x <= 0 ? 0 : regularizedGammaP(k, x / theta)),
    mean: ({ k, theta }) => k * theta,
    variance: ({ k, theta }) => k * theta * theta,
    range: ({ k, theta }) => [0, k * theta + 5 * Math.sqrt(k) * theta],
    sample: (rng, { k, theta }) => randGamma(rng, k) * theta,
    notes: "Sum of k exponential waiting times (for integer k). Shape k, scale θ.",
    mlUse: "Conjugate prior for Poisson rates and precision of Gaussians in Bayesian models.",
  },
  {
    id: "beta",
    name: "Beta",
    kind: "continuous",
    params: [
      { key: "a", label: "\\alpha", min: 0.2, max: 12, step: 0.1, default: 2 },
      { key: "b", label: "\\beta", min: 0.2, max: 12, step: 0.1, default: 5 },
    ],
    formula: "f(x)=\\dfrac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha,\\beta)},\\quad 0<x<1",
    support: "x\\in(0,1)",
    density: (x, { a, b }) => (x <= 0 || x >= 1 ? 0 : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logBeta(a, b))),
    cdf: (x, { a, b }) => regularizedBeta(x, a, b),
    mean: ({ a, b }) => a / (a + b),
    variance: ({ a, b }) => (a * b) / ((a + b) ** 2 * (a + b + 1)),
    range: () => [0, 1],
    sample: (rng, { a, b }) => {
      const x = randGamma(rng, a);
      const y = randGamma(rng, b);
      return x / (x + y);
    },
    notes: "A distribution over probabilities. α−1 and β−1 act like pseudo-counts of successes and failures.",
    mlUse: "Conjugate prior for Bernoulli/Binomial (A/B testing, Thompson sampling).",
  },
  {
    id: "chi-square",
    name: "Chi-square",
    kind: "continuous",
    params: [{ key: "k", label: "k", min: 1, max: 30, step: 1, default: 4, integer: true }],
    formula: "f(x)=\\dfrac{x^{k/2-1}e^{-x/2}}{2^{k/2}\\Gamma(k/2)},\\quad x>0",
    support: "x>0",
    density: (x, { k }) => (x <= 0 ? 0 : Math.exp((k / 2 - 1) * Math.log(x) - x / 2 - (k / 2) * Math.log(2) - logGamma(k / 2))),
    cdf: (x, { k }) => (x <= 0 ? 0 : regularizedGammaP(k / 2, x / 2)),
    mean: ({ k }) => k,
    variance: ({ k }) => 2 * k,
    range: ({ k }) => [0, k + 6 * Math.sqrt(2 * k) + 2],
    sample: (rng, { k }) => {
      let s = 0;
      for (let i = 0; i < k; i++) s += randn(rng) ** 2;
      return s;
    },
    notes: "Sum of k squared independent standard normals.",
    mlUse: "Chi-square tests of independence for categorical feature selection.",
  },
  {
    id: "student-t",
    name: "Student's t",
    kind: "continuous",
    params: [{ key: "nu", label: "\\nu", min: 1, max: 60, step: 1, default: 5, integer: true }],
    formula: "f(t)=\\dfrac{\\Gamma(\\frac{\\nu+1}{2})}{\\sqrt{\\nu\\pi}\\,\\Gamma(\\frac{\\nu}{2})}\\left(1+\\dfrac{t^2}{\\nu}\\right)^{-\\frac{\\nu+1}{2}}",
    support: "t\\in\\mathbb{R}",
    density: (t, { nu }) => Math.exp(logGamma((nu + 1) / 2) - logGamma(nu / 2) - 0.5 * Math.log(nu * Math.PI) - ((nu + 1) / 2) * Math.log(1 + (t * t) / nu)),
    cdf: (t, { nu }) => {
      const x = nu / (nu + t * t);
      const ib = regularizedBeta(x, nu / 2, 0.5);
      return t >= 0 ? 1 - 0.5 * ib : 0.5 * ib;
    },
    mean: ({ nu }) => (nu > 1 ? 0 : NaN),
    variance: ({ nu }) => (nu > 2 ? nu / (nu - 2) : nu > 1 ? Infinity : NaN),
    range: () => [-6, 6],
    sample: (rng, { nu }) => {
      let s = 0;
      for (let i = 0; i < nu; i++) s += randn(rng) ** 2;
      return randn(rng) / Math.sqrt(s / nu);
    },
    notes: "Heavier tails than the normal; approaches N(0,1) as ν → ∞. Mean undefined for ν ≤ 1.",
    mlUse: "t-tests comparing model metrics with small samples; robust heavy-tailed noise models.",
  },
  {
    id: "f",
    name: "F-distribution",
    kind: "continuous",
    params: [
      { key: "d1", label: "d_1", min: 1, max: 40, step: 1, default: 5, integer: true },
      { key: "d2", label: "d_2", min: 1, max: 60, step: 1, default: 12, integer: true },
    ],
    formula: "f(x)=\\dfrac{\\sqrt{\\frac{(d_1x)^{d_1}d_2^{d_2}}{(d_1x+d_2)^{d_1+d_2}}}}{x\\,B(\\frac{d_1}{2},\\frac{d_2}{2})}",
    support: "x>0",
    density: (x, { d1, d2 }) =>
      x <= 0 ? 0 : Math.exp(0.5 * (d1 * Math.log(d1 * x) + d2 * Math.log(d2) - (d1 + d2) * Math.log(d1 * x + d2)) - Math.log(x) - logBeta(d1 / 2, d2 / 2)),
    cdf: (x, { d1, d2 }) => (x <= 0 ? 0 : regularizedBeta((d1 * x) / (d1 * x + d2), d1 / 2, d2 / 2)),
    mean: ({ d2 }) => (d2 > 2 ? d2 / (d2 - 2) : NaN),
    variance: ({ d1, d2 }) => (d2 > 4 ? (2 * d2 * d2 * (d1 + d2 - 2)) / (d1 * (d2 - 2) ** 2 * (d2 - 4)) : NaN),
    range: () => [0, 5],
    sample: (rng, { d1, d2 }) => {
      let a = 0;
      let b = 0;
      for (let i = 0; i < d1; i++) a += randn(rng) ** 2;
      for (let i = 0; i < d2; i++) b += randn(rng) ** 2;
      return a / d1 / (b / d2);
    },
    notes: "Ratio of two scaled chi-square variables. Mean defined only for d₂ > 2.",
    mlUse: "ANOVA F-tests and the F-statistic for regression feature significance.",
  },
  {
    id: "log-normal",
    name: "Log-normal",
    kind: "continuous",
    params: [
      { key: "mu", label: "\\mu", min: -1, max: 2, step: 0.05, default: 0 },
      { key: "sigma", label: "\\sigma", min: 0.1, max: 1.5, step: 0.05, default: 0.6 },
    ],
    formula: "f(x)=\\dfrac{1}{x\\sigma\\sqrt{2\\pi}}\\exp\\!\\left(-\\dfrac{(\\ln x-\\mu)^2}{2\\sigma^2}\\right)",
    support: "x>0",
    density: (x, { mu, sigma }) => (x <= 0 ? 0 : Math.exp(-0.5 * ((Math.log(x) - mu) / sigma) ** 2) / (x * sigma * Math.sqrt(2 * Math.PI))),
    cdf: (x, { mu, sigma }) => (x <= 0 ? 0 : normalCdf((Math.log(x) - mu) / sigma)),
    mean: ({ mu, sigma }) => Math.exp(mu + (sigma * sigma) / 2),
    variance: ({ mu, sigma }) => (Math.exp(sigma * sigma) - 1) * Math.exp(2 * mu + sigma * sigma),
    range: ({ mu, sigma }) => [0, Math.exp(mu + 3.2 * sigma)],
    sample: (rng, { mu, sigma }) => Math.exp(mu + sigma * randn(rng)),
    notes: "If ln X is normal, X is log-normal. Right-skewed, strictly positive.",
    mlUse: "Incomes, prices, latencies — the reason a log transform often makes features look normal.",
  },
  {
    id: "weibull",
    name: "Weibull",
    kind: "continuous",
    params: [
      { key: "k", label: "k", min: 0.3, max: 5, step: 0.05, default: 1.5 },
      { key: "lambda", label: "\\lambda", min: 0.3, max: 4, step: 0.05, default: 1 },
    ],
    formula: "f(x)=\\dfrac{k}{\\lambda}\\left(\\dfrac{x}{\\lambda}\\right)^{k-1}e^{-(x/\\lambda)^k},\\quad x\\ge0",
    support: "x\\ge0",
    density: (x, { k, lambda }) => (x < 0 ? 0 : x === 0 && k < 1 ? Infinity : (k / lambda) * Math.pow(x / lambda, k - 1) * Math.exp(-Math.pow(x / lambda, k))),
    cdf: (x, { k, lambda }) => (x < 0 ? 0 : 1 - Math.exp(-Math.pow(x / lambda, k))),
    mean: ({ k, lambda }) => lambda * Math.exp(logGamma(1 + 1 / k)),
    variance: ({ k, lambda }) => lambda * lambda * (Math.exp(logGamma(1 + 2 / k)) - Math.exp(2 * logGamma(1 + 1 / k))),
    range: ({ k, lambda }) => [0, lambda * Math.pow(-Math.log(0.002), 1 / k)],
    sample: (rng, { k, lambda }) => lambda * Math.pow(-Math.log(1 - rng()), 1 / k),
    notes: "k < 1: failure rate decreases over time; k = 1: exponential; k > 1: wear-out.",
    mlUse: "Reliability and survival analysis; time-to-failure prediction for predictive maintenance.",
  },
];

export const distributionById = (id: string) => distributions.find((d) => d.id === id);

export function defaultParams(d: Distribution): Record<string, number> {
  return Object.fromEntries(d.params.map((p) => [p.key, p.default]));
}

/** Numerical quantile for any distribution (bisection over its plotting range). */
export function quantile(d: Distribution, p: Record<string, number>, q: number): number {
  const [lo, hi] = d.range(p);
  return invertCdf((x) => d.cdf(x, p), q, lo - 50, hi + 50);
}
