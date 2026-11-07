/** 2-D test objectives and optimisers with exact analytic gradients. */

export interface Objective {
  id: string;
  name: string;
  tex: string;
  f: (x: number, y: number) => number;
  grad: (x: number, y: number) => [number, number];
  hess?: (x: number, y: number) => [[number, number], [number, number]];
  domain: [[number, number], [number, number]];
  start: [number, number];
  minimum?: [number, number];
  notes: string;
}

export const objectives: Objective[] = [
  {
    id: "bowl",
    name: "Convex bowl",
    tex: "f(x,y)=x^2+y^2",
    f: (x, y) => x * x + y * y,
    grad: (x, y) => [2 * x, 2 * y],
    hess: () => [[2, 0], [0, 2]],
    domain: [[-3, 3], [-3, 3]],
    start: [-2.5, 2.2],
    minimum: [0, 0],
    notes: "Perfectly conditioned (Hessian ∝ I): gradient points straight at the minimum.",
  },
  {
    id: "ellipse",
    name: "Ill-conditioned valley",
    tex: "f(x,y)=0.5x^2+5y^2",
    f: (x, y) => 0.5 * x * x + 5 * y * y,
    grad: (x, y) => [x, 10 * y],
    hess: () => [[1, 0], [0, 10]],
    domain: [[-3.5, 3.5], [-2, 2]],
    start: [-3.2, 1.2],
    minimum: [0, 0],
    notes: "Condition number κ = 10. Plain GD zig-zags across the steep direction; momentum / Adam help.",
  },
  {
    id: "rosenbrock",
    name: "Rosenbrock",
    tex: "f(x,y)=(1-x)^2+100(y-x^2)^2",
    f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    grad: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
    hess: (x, y) => [[2 - 400 * (y - x * x) + 800 * x * x, -400 * x], [-400 * x, 200]],
    domain: [[-2, 2], [-1, 3]],
    start: [-1.5, 2.2],
    minimum: [1, 1],
    notes: "Non-convex banana-shaped valley; finding the valley is easy, travelling along it is hard.",
  },
  {
    id: "saddle",
    name: "Saddle point",
    tex: "f(x,y)=x^2-y^2+0.1y^4",
    f: (x, y) => x * x - y * y + 0.1 * y ** 4,
    grad: (x, y) => [2 * x, -2 * y + 0.4 * y ** 3],
    hess: (_x, y) => [[2, 0], [0, -2 + 1.2 * y * y]],
    domain: [[-3, 3], [-3.5, 3.5]],
    start: [-2.5, 0.02],
    minimum: [0, Math.sqrt(5)],
    notes: "Saddle at the origin (Hessian eigenvalues 2 and −2). Starting near y=0, GD stalls before escaping; minima at y = ±√5.",
  },
  {
    id: "multimodal",
    name: "Multiple minima",
    tex: "f(x,y)=\\tfrac{x^2+y^2}{4}-\\cos(2x)\\cos(2y)",
    f: (x, y) => (x * x + y * y) / 4 - Math.cos(2 * x) * Math.cos(2 * y),
    grad: (x, y) => [x / 2 + 2 * Math.sin(2 * x) * Math.cos(2 * y), y / 2 + 2 * Math.cos(2 * x) * Math.sin(2 * y)],
    domain: [[-3.5, 3.5], [-3.5, 3.5]],
    start: [2.6, 2.4],
    minimum: [0, 0],
    notes: "Many local minima around a global one at the origin — initialisation and learning rate decide where you land.",
  },
];

export type OptimizerId = "gd" | "momentum" | "nesterov" | "adagrad" | "rmsprop" | "adam" | "adamw" | "newton" | "bfgs" | "coord";

export interface OptimizerConfig {
  id: OptimizerId;
  lr: number;
  beta1?: number; // momentum / Adam β1
  beta2?: number; // RMSProp / Adam β2
  eps?: number;
  weightDecay?: number;
  clip?: number; // gradient norm clip (0 = off)
}

export const optimizerInfo: Record<OptimizerId, { name: string; tex: string; color: string }> = {
  gd: { name: "Gradient descent", tex: "\\theta_{t+1}=\\theta_t-\\eta\\nabla f(\\theta_t)", color: "#8b9bb4" },
  momentum: { name: "Momentum", tex: "v_{t+1}=\\beta v_t+\\nabla f(\\theta_t),\\;\\theta_{t+1}=\\theta_t-\\eta v_{t+1}", color: "#5aa9ff" },
  nesterov: { name: "Nesterov", tex: "v_{t+1}=\\beta v_t+\\nabla f(\\theta_t-\\eta\\beta v_t),\\;\\theta_{t+1}=\\theta_t-\\eta v_{t+1}", color: "#38d4c4" },
  adagrad: { name: "AdaGrad", tex: "G_t=G_{t-1}+g_t^2,\\;\\theta_{t+1}=\\theta_t-\\dfrac{\\eta}{\\sqrt{G_t}+\\epsilon}g_t", color: "#e6c35c" },
  rmsprop: { name: "RMSProp", tex: "s_t=\\beta s_{t-1}+(1-\\beta)g_t^2,\\;\\theta_{t+1}=\\theta_t-\\dfrac{\\eta}{\\sqrt{s_t}+\\epsilon}g_t", color: "#f59e5b" },
  adam: { name: "Adam", tex: "\\hat m_t=\\dfrac{m_t}{1-\\beta_1^t},\\;\\hat v_t=\\dfrac{v_t}{1-\\beta_2^t},\\;\\theta_{t+1}=\\theta_t-\\eta\\dfrac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}", color: "#ff6fae" },
  adamw: { name: "AdamW", tex: "\\theta_{t+1}=\\theta_t-\\eta\\left(\\dfrac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}+\\lambda\\theta_t\\right)", color: "#c58bff" },
  newton: { name: "Newton", tex: "\\theta_{t+1}=\\theta_t-H^{-1}\\nabla f(\\theta_t)", color: "#7ddc7a" },
  bfgs: { name: "BFGS (quasi-Newton)", tex: String.raw`\theta_{t+1}=\theta_t-\alpha_tB_t\nabla f,\ B_{t+1}=(I-\rho\mathbf s\mathbf y^\top)B_t(I-\rho\mathbf y\mathbf s^\top)+\rho\mathbf s\mathbf s^\top`, color: "#f472b6" },
  coord: { name: "Coordinate descent", tex: String.raw`\theta_{t+1,j}=\theta_{t,j}-\eta\,\partial_jf(\theta_t),\ j\text{ alternates }x,y`, color: "#a3a3a3" },
};

export interface OptState { p: [number, number]; m: [number, number]; v: [number, number]; t: number; B?: number[][] }

export const initState = (p: [number, number]): OptState => ({ p: [p[0], p[1]], m: [0, 0], v: [0, 0], t: 0 });

/** Advance one optimiser step. Pure function. */
export function optimStep(obj: Objective, cfg: OptimizerConfig, s: OptState): OptState {
  const { lr } = cfg;
  const b1 = cfg.beta1 ?? 0.9;
  const b2 = cfg.beta2 ?? 0.999;
  const eps = cfg.eps ?? 1e-8;
  const clip = (g: [number, number]): [number, number] => {
    if (!cfg.clip) return g;
    const n = Math.hypot(g[0], g[1]);
    return n > cfg.clip ? [(g[0] * cfg.clip) / n, (g[1] * cfg.clip) / n] : g;
  };
  const [x, y] = s.p;
  const t = s.t + 1;
  switch (cfg.id) {
    case "gd": {
      const g = clip(obj.grad(x, y));
      return { ...s, p: [x - lr * g[0], y - lr * g[1]], t };
    }
    case "momentum": {
      const g = clip(obj.grad(x, y));
      const m: [number, number] = [b1 * s.m[0] + g[0], b1 * s.m[1] + g[1]];
      return { ...s, m, p: [x - lr * m[0], y - lr * m[1]], t };
    }
    case "nesterov": {
      const g = clip(obj.grad(x - lr * b1 * s.m[0], y - lr * b1 * s.m[1]));
      const m: [number, number] = [b1 * s.m[0] + g[0], b1 * s.m[1] + g[1]];
      return { ...s, m, p: [x - lr * m[0], y - lr * m[1]], t };
    }
    case "adagrad": {
      const g = clip(obj.grad(x, y));
      const v: [number, number] = [s.v[0] + g[0] ** 2, s.v[1] + g[1] ** 2];
      return { ...s, v, p: [x - (lr * g[0]) / (Math.sqrt(v[0]) + eps), y - (lr * g[1]) / (Math.sqrt(v[1]) + eps)], t };
    }
    case "rmsprop": {
      const g = clip(obj.grad(x, y));
      const beta = cfg.beta2 ?? 0.9;
      const v: [number, number] = [beta * s.v[0] + (1 - beta) * g[0] ** 2, beta * s.v[1] + (1 - beta) * g[1] ** 2];
      return { ...s, v, p: [x - (lr * g[0]) / (Math.sqrt(v[0]) + eps), y - (lr * g[1]) / (Math.sqrt(v[1]) + eps)], t };
    }
    case "adam":
    case "adamw": {
      const g = clip(obj.grad(x, y));
      const m: [number, number] = [b1 * s.m[0] + (1 - b1) * g[0], b1 * s.m[1] + (1 - b1) * g[1]];
      const v: [number, number] = [b2 * s.v[0] + (1 - b2) * g[0] ** 2, b2 * s.v[1] + (1 - b2) * g[1] ** 2];
      const mh = [m[0] / (1 - b1 ** t), m[1] / (1 - b1 ** t)];
      const vh = [v[0] / (1 - b2 ** t), v[1] / (1 - b2 ** t)];
      const wd = cfg.id === "adamw" ? cfg.weightDecay ?? 0.01 : 0;
      return {
        m, v, t,
        p: [x - lr * (mh[0] / (Math.sqrt(vh[0]) + eps) + wd * x), y - lr * (mh[1] / (Math.sqrt(vh[1]) + eps) + wd * y)],
      };
    }
    case "bfgs": {
      // quasi-Newton: inverse-Hessian estimate B built from gradient differences, Armijo backtracking
      const B = s.B ?? [[1, 0], [0, 1]];
      const g = obj.grad(x, y);
      const d: [number, number] = [-(B[0][0] * g[0] + B[0][1] * g[1]), -(B[1][0] * g[0] + B[1][1] * g[1])];
      const slope = g[0] * d[0] + g[1] * d[1];
      if (slope >= 0) return { ...s, B: [[1, 0], [0, 1]], t };
      let a = lr;
      const f0 = obj.f(x, y);
      for (let i = 0; i < 40 && obj.f(x + a * d[0], y + a * d[1]) > f0 + 1e-4 * a * slope; i++) a *= 0.5;
      const pn: [number, number] = [x + a * d[0], y + a * d[1]];
      const gn = obj.grad(pn[0], pn[1]);
      const sv = [pn[0] - x, pn[1] - y], yv = [gn[0] - g[0], gn[1] - g[1]];
      const sy = sv[0] * yv[0] + sv[1] * yv[1];
      let Bn = B;
      if (sy > 1e-12) {
        const rho = 1 / sy;
        const L = [[1 - rho * sv[0] * yv[0], -rho * sv[0] * yv[1]], [-rho * sv[1] * yv[0], 1 - rho * sv[1] * yv[1]]];
        const LB = [[L[0][0] * B[0][0] + L[0][1] * B[1][0], L[0][0] * B[0][1] + L[0][1] * B[1][1]], [L[1][0] * B[0][0] + L[1][1] * B[1][0], L[1][0] * B[0][1] + L[1][1] * B[1][1]]];
        Bn = [[LB[0][0] * L[0][0] + LB[0][1] * L[0][1] + rho * sv[0] * sv[0], LB[0][0] * L[1][0] + LB[0][1] * L[1][1] + rho * sv[0] * sv[1]], [LB[1][0] * L[0][0] + LB[1][1] * L[0][1] + rho * sv[1] * sv[0], LB[1][0] * L[1][0] + LB[1][1] * L[1][1] + rho * sv[1] * sv[1]]];
      }
      return { ...s, p: pn, B: Bn, t };
    }
    case "coord": {
      const g = clip(obj.grad(x, y));
      return t % 2 === 1 ? { ...s, p: [x - lr * g[0], y], t } : { ...s, p: [x, y - lr * g[1]], t };
    }
    case "newton": {
      const g = obj.grad(x, y);
      const H = obj.hess?.(x, y);
      if (!H) return { ...s, t };
      const det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
      if (Math.abs(det) < 1e-12) return { ...s, t };
      const dx = (H[1][1] * g[0] - H[0][1] * g[1]) / det;
      const dy = (-H[1][0] * g[0] + H[0][0] * g[1]) / det;
      // damped step with step size lr (lr=1 is the pure Newton step)
      return { ...s, p: [x - lr * dx, y - lr * dy], t };
    }
  }
}

/** Run an optimiser for n steps; stops early on divergence (non-finite or far outside domain). */
export function runOptimizer(obj: Objective, cfg: OptimizerConfig, start: [number, number], n: number) {
  let s = initState(start);
  const path: [number, number][] = [s.p];
  let diverged = false;
  for (let i = 0; i < n; i++) {
    s = optimStep(obj, cfg, s);
    if (!Number.isFinite(s.p[0]) || !Number.isFinite(s.p[1]) || Math.abs(s.p[0]) > 1e3 || Math.abs(s.p[1]) > 1e3) {
      diverged = true;
      break;
    }
    path.push(s.p);
  }
  return { path, diverged };
}

/* ------------------------------ learning-rate schedules ------------------------------ */

export type ScheduleId = "constant" | "step" | "exponential" | "cosine" | "warmup-cosine" | "linear-warmup" | "one-cycle";

export function lrSchedule(id: ScheduleId, t: number, T: number, base: number, opts: { warmup?: number; stepEvery?: number; gamma?: number } = {}) {
  const warm = opts.warmup ?? Math.round(T * 0.1);
  switch (id) {
    case "constant": return base;
    case "step": return base * Math.pow(opts.gamma ?? 0.5, Math.floor(t / (opts.stepEvery ?? Math.max(1, Math.round(T / 4)))));
    case "exponential": return base * Math.pow(opts.gamma ?? 0.97, t);
    case "cosine": return 0.5 * base * (1 + Math.cos((Math.PI * t) / T));
    case "warmup-cosine": return t < warm ? (base * (t + 1)) / warm : 0.5 * base * (1 + Math.cos((Math.PI * (t - warm)) / Math.max(1, T - warm)));
    case "linear-warmup": return t < warm ? (base * (t + 1)) / warm : base * Math.max(0, (T - t) / Math.max(1, T - warm));
    case "one-cycle": {
      const half = T * 0.45;
      if (t < half) return base * 0.1 + (base - base * 0.1) * (t / half);
      if (t < 2 * half) return base - (base - base * 0.1) * ((t - half) / half);
      return base * 0.1 * (1 - (t - 2 * half) / Math.max(1, T - 2 * half)) + 1e-6;
    }
  }
}
