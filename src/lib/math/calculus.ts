/** Single-variable function catalogue with exact (analytic) derivatives and integrals. */

export interface Fn1 {
  id: string;
  name: string;
  tex: string;
  dtex: string;
  d2tex: string;
  f: (x: number) => number;
  df: (x: number) => number;
  d2f: (x: number) => number;
  /** antiderivative F with F' = f (when it exists in closed form) */
  F?: (x: number) => number;
  domain: [number, number];
  yRange: [number, number];
  /** valid input check */
  defined?: (x: number) => boolean;
}

export const functions1D: Fn1[] = [
  {
    id: "cubic", name: "Cubic", tex: "f(x)=0.25x^3-x", dtex: "f'(x)=0.75x^2-1", d2tex: "f''(x)=1.5x",
    f: (x) => 0.25 * x ** 3 - x, df: (x) => 0.75 * x * x - 1, d2f: (x) => 1.5 * x, F: (x) => x ** 4 / 16 - x * x / 2,
    domain: [-3.5, 3.5], yRange: [-4, 4],
  },
  {
    id: "quadratic", name: "Quadratic", tex: "f(x)=x^2-2x", dtex: "f'(x)=2x-2", d2tex: "f''(x)=2",
    f: (x) => x * x - 2 * x, df: (x) => 2 * x - 2, d2f: () => 2, F: (x) => x ** 3 / 3 - x * x,
    domain: [-2, 4], yRange: [-2, 8],
  },
  {
    id: "sin", name: "Sine", tex: "f(x)=\\sin x", dtex: "f'(x)=\\cos x", d2tex: "f''(x)=-\\sin x",
    f: Math.sin, df: Math.cos, d2f: (x) => -Math.sin(x), F: (x) => -Math.cos(x),
    domain: [-2 * Math.PI, 2 * Math.PI], yRange: [-1.6, 1.6],
  },
  {
    id: "exp", name: "Exponential", tex: "f(x)=e^{x}", dtex: "f'(x)=e^{x}", d2tex: "f''(x)=e^{x}",
    f: Math.exp, df: Math.exp, d2f: Math.exp, F: Math.exp,
    domain: [-3, 2.2], yRange: [-0.5, 9],
  },
  {
    id: "log", name: "Natural log", tex: "f(x)=\\ln x", dtex: "f'(x)=\\dfrac1x", d2tex: "f''(x)=-\\dfrac1{x^2}",
    f: Math.log, df: (x) => 1 / x, d2f: (x) => -1 / (x * x), F: (x) => x * Math.log(x) - x,
    domain: [0.05, 6], yRange: [-3, 2.5], defined: (x) => x > 0,
  },
  {
    id: "sigmoid", name: "Sigmoid", tex: "f(x)=\\dfrac{1}{1+e^{-x}}", dtex: "f'(x)=\\sigma(x)(1-\\sigma(x))", d2tex: "f''(x)=\\sigma'(x)(1-2\\sigma(x))",
    f: (x) => 1 / (1 + Math.exp(-x)),
    df: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
    d2f: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s) * (1 - 2 * s); },
    F: (x) => Math.log1p(Math.exp(x)),
    domain: [-7, 7], yRange: [-0.3, 1.3],
  },
  {
    id: "gauss", name: "Gaussian bump", tex: "f(x)=e^{-x^2}", dtex: "f'(x)=-2xe^{-x^2}", d2tex: "f''(x)=(4x^2-2)e^{-x^2}",
    f: (x) => Math.exp(-x * x), df: (x) => -2 * x * Math.exp(-x * x), d2f: (x) => (4 * x * x - 2) * Math.exp(-x * x),
    domain: [-3, 3], yRange: [-1, 1.4],
  },
  {
    id: "abs", name: "Absolute value", tex: "f(x)=|x|", dtex: "f'(x)=\\operatorname{sign}(x),\\ x\\neq0", d2tex: "f''(x)=0,\\ x\\neq0",
    f: Math.abs, df: (x) => (x === 0 ? NaN : Math.sign(x)), d2f: (x) => (x === 0 ? NaN : 0), F: (x) => (x * Math.abs(x)) / 2,
    domain: [-3, 3], yRange: [-1, 3.5],
  },
];

export const fnById = (id: string) => functions1D.find((f) => f.id === id) ?? functions1D[0];

/** Central difference with step h (O(h²) error). */
export const centralDiff = (f: (x: number) => number, x: number, h = 1e-5) => (f(x + h) - f(x - h)) / (2 * h);

export type RiemannRule = "left" | "right" | "midpoint" | "trapezoid";

export function riemann(f: (x: number) => number, a: number, b: number, n: number, rule: RiemannRule) {
  const w = (b - a) / n;
  let s = 0;
  const rects: { x0: number; x1: number; h: number; h1?: number }[] = [];
  for (let i = 0; i < n; i++) {
    const x0 = a + i * w;
    const x1 = x0 + w;
    if (rule === "trapezoid") {
      const h0 = f(x0), h1 = f(x1);
      s += ((h0 + h1) / 2) * w;
      rects.push({ x0, x1, h: h0, h1 });
    } else {
      const xs = rule === "left" ? x0 : rule === "right" ? x1 : (x0 + x1) / 2;
      const h = f(xs);
      s += h * w;
      rects.push({ x0, x1, h });
    }
  }
  return { sum: s, rects, width: w };
}

/** Composite Simpson's rule (n even) — high-accuracy reference for functions without closed-form F. */
export function simpson(f: (x: number) => number, a: number, b: number, n = 1000) {
  if (n % 2) n++;
  const h = (b - a) / n;
  let s = f(a) + f(b);
  for (let i = 1; i < n; i++) s += f(a + i * h) * (i % 2 ? 4 : 2);
  return (s * h) / 3;
}

/** Taylor-series catalogue: nth derivative at 0 known in closed form. */
export const taylorFns = [
  { id: "exp", name: "e^x", tex: "e^x=\\sum_{k=0}^{\\infty}\\frac{x^k}{k!}", f: Math.exp, dk: () => 1, range: [-4, 3] as [number, number], yRange: [-2, 12] as [number, number] },
  { id: "sin", name: "sin x", tex: "\\sin x=\\sum_{k=0}^{\\infty}\\frac{(-1)^k x^{2k+1}}{(2k+1)!}", f: Math.sin, dk: (k: number) => [0, 1, 0, -1][k % 4], range: [-8, 8] as [number, number], yRange: [-3, 3] as [number, number] },
  { id: "cos", name: "cos x", tex: "\\cos x=\\sum_{k=0}^{\\infty}\\frac{(-1)^k x^{2k}}{(2k)!}", f: Math.cos, dk: (k: number) => [1, 0, -1, 0][k % 4], range: [-8, 8] as [number, number], yRange: [-3, 3] as [number, number] },
  {
    id: "log1p", name: "ln(1+x)", tex: "\\ln(1+x)=\\sum_{k=1}^{\\infty}\\frac{(-1)^{k+1}x^k}{k},\\ |x|<1", f: (x: number) => (x > -1 ? Math.log1p(x) : NaN),
    dk: (k: number) => (k === 0 ? 0 : ((k % 2 ? 1 : -1) * factorial(k - 1))), range: [-0.99, 2.5] as [number, number], yRange: [-3, 2] as [number, number],
  },
];

export function factorial(n: number): number {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

export function taylorEval(dk: (k: number) => number, order: number, x: number, a = 0) {
  let s = 0;
  for (let k = 0; k <= order; k++) s += (dk(k) / factorial(k)) * (x - a) ** k;
  return s;
}

/* ------------------------------ multivariable ------------------------------ */

export interface Fn2 {
  id: string;
  name: string;
  tex: string;
  gradTex: string;
  f: (x: number, y: number) => number;
  grad: (x: number, y: number) => [number, number];
  hess: (x: number, y: number) => [[number, number], [number, number]];
  domain: [number, number];
}

export const functions2D: Fn2[] = [
  {
    id: "bowl", name: "Paraboloid", tex: "f(x,y)=x^2+2y^2", gradTex: "\\nabla f=(2x,\\;4y)",
    f: (x, y) => x * x + 2 * y * y, grad: (x, y) => [2 * x, 4 * y], hess: () => [[2, 0], [0, 4]], domain: [-3, 3],
  },
  {
    id: "saddle", name: "Saddle", tex: "f(x,y)=x^2-y^2", gradTex: "\\nabla f=(2x,\\;-2y)",
    f: (x, y) => x * x - y * y, grad: (x, y) => [2 * x, -2 * y], hess: () => [[2, 0], [0, -2]], domain: [-3, 3],
  },
  {
    id: "wave", name: "Wave", tex: "f(x,y)=\\sin x\\cos y", gradTex: "\\nabla f=(\\cos x\\cos y,\\;-\\sin x\\sin y)",
    f: (x, y) => Math.sin(x) * Math.cos(y), grad: (x, y) => [Math.cos(x) * Math.cos(y), -Math.sin(x) * Math.sin(y)],
    hess: (x, y) => [[-Math.sin(x) * Math.cos(y), -Math.cos(x) * Math.sin(y)], [-Math.cos(x) * Math.sin(y), -Math.sin(x) * Math.cos(y)]], domain: [-3.2, 3.2],
  },
  {
    id: "gauss", name: "Gaussian hill", tex: "f(x,y)=e^{-(x^2+y^2)/2}", gradTex: "\\nabla f=-(x,\\;y)\\,f(x,y)",
    f: (x, y) => Math.exp(-(x * x + y * y) / 2),
    grad: (x, y) => { const v = Math.exp(-(x * x + y * y) / 2); return [-x * v, -y * v]; },
    hess: (x, y) => { const v = Math.exp(-(x * x + y * y) / 2); return [[(x * x - 1) * v, x * y * v], [x * y * v, (y * y - 1) * v]]; }, domain: [-3, 3],
  },
  {
    id: "monkey", name: "Monkey saddle", tex: "f(x,y)=x^3-3xy^2", gradTex: "\\nabla f=(3x^2-3y^2,\\;-6xy)",
    f: (x, y) => x ** 3 - 3 * x * y * y, grad: (x, y) => [3 * x * x - 3 * y * y, -6 * x * y], hess: (x, y) => [[6 * x, -6 * y], [-6 * y, -6 * x]], domain: [-2, 2],
  },
];
