import { describe, it, expect } from "vitest";
import * as M from "@/lib/ml/metrics";
import * as ML from "@/lib/ml/models";
import { activations, softmax, lossFns, klDivergence, createMLP, trainStep, forward, cloneMLP } from "@/lib/nn";
import { objectives, runOptimizer } from "@/lib/math/optim";
import { functions1D, centralDiff, riemann, taylorEval, taylorFns } from "@/lib/math/calculus";

const close = (a: number, b: number, tol = 1e-6) => expect(Math.abs(a - b)).toBeLessThan(tol);

describe("metrics", () => {
  it("confusion + derived metrics", () => {
    const y = [1, 1, 1, 0, 0, 0, 0, 1];
    const s = [0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.4, 0.7];
    const c = M.confusion(y, s, 0.5);
    expect(c).toEqual({ tp: 3, fp: 1, tn: 3, fn: 1 });
    const m = M.classificationMetrics(c);
    close(m.precision, 0.75);
    close(m.recall, 0.75);
    close(m.f1, 0.75);
    close(m.accuracy, 0.75);
  });
  it("ROC AUC equals pairwise ranking probability", () => {
    const y = [1, 0, 1, 0, 1, 0];
    const s = [0.9, 0.8, 0.7, 0.4, 0.35, 0.1];
    let good = 0, tot = 0;
    for (let i = 0; i < y.length; i++) for (let j = 0; j < y.length; j++) if (y[i] === 1 && y[j] === 0) { tot++; good += s[i] > s[j] ? 1 : 0; }
    close(M.rocCurve(y, s).auc, good / tot);
  });
  it("regression metrics", () => {
    const r = M.regressionMetrics([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]);
    close(r.mae, 0.5);
    close(r.mse, 0.375);
    close(r.r2, 0.9486081370449679);
  });
  it("log loss", () => {
    close(M.logLoss([1, 0], [0.8, 0.2]), -Math.log(0.8));
  });
});

describe("models", () => {
  it("OLS recovers an exact line", () => {
    const xs = [0, 1, 2, 3, 4];
    const w = ML.ridgeFit(ML.polyDesign(xs, 1), xs.map((x) => 2 + 3 * x))!;
    close(w[0], 2);
    close(w[1], 3);
  });
  it("lasso shrinks to zero for large lambda, matches OLS for 0", () => {
    const xs = [0, 1, 2, 3, 4];
    const X = ML.polyDesign(xs, 1);
    const y = xs.map((x) => 2 + 3 * x);
    const w0 = ML.elasticNetFit(X, y, 0, 1, 5000);
    close(w0[1], 3, 1e-4);
    const wBig = ML.elasticNetFit(X, y, 100, 1);
    expect(wBig[1]).toBe(0);
    close(wBig[0], 8, 1e-6); // intercept → mean(y)
  });
  it("decision tree separates a simple split perfectly", () => {
    const pts = [{ x: 0, y: 0, label: 0 }, { x: 1, y: 0, label: 0 }, { x: 3, y: 0, label: 1 }, { x: 4, y: 1, label: 1 }];
    const t = ML.buildTree(pts, 3);
    expect(ML.treePredict(t, 0.5, 0)).toBe(0);
    expect(ML.treePredict(t, 3.5, 0)).toBe(1);
  });
  it("SVM separates linearly separable data", () => {
    const pts = ML.makeDataset("blobs", 60, 0, 3);
    const m = ML.svmFit(pts, 10, { type: "linear" });
    const acc = pts.filter((p) => (ML.svmDecision(m, p.x, p.y) > 0 ? 1 : 0) === p.label).length / pts.length;
    expect(acc).toBeGreaterThan(0.95);
  });
  it("k-means decreases inertia", () => {
    const pts = ML.makeClusters(3, 30, 0.25, 5);
    let C = ML.kmeansInit(pts, 3, "random", 1);
    let a = ML.kmeansAssign(pts, C);
    const i0 = ML.inertia(pts, a, C);
    for (let k = 0; k < 10; k++) { C = ML.kmeansUpdate(pts, a, C); a = ML.kmeansAssign(pts, C); }
    expect(ML.inertia(pts, a, C)).toBeLessThanOrEqual(i0);
  });
  it("EM never decreases log-likelihood", () => {
    const pts = ML.makeClusters(2, 40, 0.3, 9);
    let comps: ML.GMMComp[] = [
      { pi: 0.5, mu: [-0.5, 0], cov: [[1, 0], [0, 1]] },
      { pi: 0.5, mu: [0.5, 0], cov: [[1, 0], [0, 1]] },
    ];
    let prev = -Infinity;
    for (let k = 0; k < 15; k++) {
      const { R, logLik } = ML.gmmEStep(pts, comps);
      expect(logLik).toBeGreaterThanOrEqual(prev - 1e-8);
      prev = logLik;
      comps = ML.gmmMStep(pts, R);
    }
  });
  it("DBSCAN labels isolated point as noise", () => {
    const pts: ML.P2[] = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [5, 5]];
    const r = ML.dbscan(pts, 0.2, 3);
    expect(r.role[4]).toBe("noise");
    expect(r.clusters).toBe(1);
  });
});

describe("neural nets", () => {
  it("activation derivatives match finite differences", () => {
    for (const a of activations) {
      for (const x of [-2.3, -0.7, 0.4, 1.9]) close(a.df(x), centralDiff((t) => a.f(t), x), 1e-5);
    }
  });
  it("softmax sums to 1 and is stable", () => {
    const p = softmax([1000, 1001, 1002]);
    close(p.reduce((a, b) => a + b, 0), 1);
    expect(p[2]).toBeGreaterThan(p[1]);
  });
  it("losses", () => {
    close(lossFns.huber(0, 3, 1), 2.5);
    close(lossFns.bce(1, 0.5), Math.log(2));
    close(lossFns.focal(1, 0.5, 0), Math.log(2));
    close(klDivergence([0.5, 0.5], [0.5, 0.5]), 0);
  });
  it("backprop gradient matches numerical gradient", () => {
    const net = createMLP([2, 3, 1], "tanh", 4);
    const X = [[0.5, -0.3], [-0.2, 0.8]];
    const Y = [1, 0];
    const loss = (n: typeof net) => X.reduce((s, x, i) => s + lossFns.bce(Y[i], forward(n, x).out), 0) / X.length;
    const lr = 1e-3;
    const a = cloneMLP(net);
    const w0 = a.W[0][1][0];
    const h = 1e-5;
    const p = cloneMLP(net); p.W[0][1][0] += h;
    const m = cloneMLP(net); m.W[0][1][0] -= h;
    const numGrad = (loss(p) - loss(m)) / (2 * h);
    trainStep(a, X, Y, lr);
    const analytic = (w0 - a.W[0][1][0]) / lr;
    close(analytic, numGrad, 1e-6);
  });
});

describe("optimisation & calculus", () => {
  it("GD converges on the bowl", () => {
    const { path } = runOptimizer(objectives[0], { id: "gd", lr: 0.1 }, [2, 2], 200);
    const [x, y] = path[path.length - 1];
    close(x, 0, 1e-6);
    close(y, 0, 1e-6);
  });
  it("Newton converges in one step on a quadratic", () => {
    const { path } = runOptimizer(objectives[1], { id: "newton", lr: 1 }, [3, 1], 1);
    close(path[1][0], 0);
    close(path[1][1], 0);
  });
  it("GD diverges when lr exceeds 2/L", () => {
    const { diverged } = runOptimizer(objectives[0], { id: "gd", lr: 1.1 }, [2, 2], 400);
    expect(diverged).toBe(true);
  });
  it("analytic derivatives match finite differences", () => {
    for (const f of functions1D) {
      for (const x of [0.3, 1.1, 1.7]) {
        if (f.defined && !f.defined(x)) continue;
        close(f.df(x), centralDiff(f.f, x), 1e-5);
        close(f.d2f(x), centralDiff(f.df, x), 1e-5);
      }
    }
  });
  it("Riemann sums converge to the exact integral", () => {
    const f = functions1D.find((g) => g.id === "quadratic")!;
    const exact = f.F!(3) - f.F!(0);
    close(riemann(f.f, 0, 3, 2000, "midpoint").sum, exact, 1e-5);
    close(riemann(f.f, 0, 3, 7, "trapezoid").sum, exact, 0.1);
  });
  it("Taylor series of e^x", () => {
    close(taylorEval(taylorFns[0].dk, 12, 1), Math.E, 1e-8);
    close(taylorEval(taylorFns[1].dk, 15, 1), Math.sin(1), 1e-10);
  });
});
