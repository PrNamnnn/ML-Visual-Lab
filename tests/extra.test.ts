import { describe, it, expect } from "vitest";
import { jacobiEig, topEig } from "@/visualizers/shared";
import { cholesky, cholSolve } from "@/visualizers/machine-learning/MLUnsupervised";
import { mutualInfo, ols } from "@/visualizers/preprocessing/FeatureExtra";
import { poissonSample } from "@/visualizers/machine-learning/MLRegression";
import { stem } from "@/visualizers/nlp/NLPExtra";
import { fnv1a } from "@/visualizers/ai-engineering/AIOps";
import { validateOutput } from "@/visualizers/ai-engineering/AIRetrieval";
import { mulberry32, randn } from "@/lib/math/random";

describe("eigen solvers", () => {
  const S = [[4, 1, 0], [1, 3, 1], [0, 1, 2]];
  it("Jacobi reproduces A v = λ v and sorts ascending", () => {
    const { values, vectors } = jacobiEig(S);
    expect(values[0]).toBeLessThan(values[1]);
    vectors.forEach((v, k) => {
      const Av = S.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
      Av.forEach((x, i) => expect(x).toBeCloseTo(values[k] * v[i], 8));
    });
    expect(values.reduce((a, b) => a + b, 0)).toBeCloseTo(9, 10); // trace
  });
  it("power iteration finds the largest eigenvalue of a PSD matrix", () => {
    const { values } = topEig(S, 1, 500);
    expect(values[0]).toBeCloseTo(jacobiEig(S).values[2], 6);
  });
});

describe("Cholesky", () => {
  it("solves a symmetric positive-definite system", () => {
    const A = [[4, 2, 0.4], [2, 5, 1], [0.4, 1, 3]], b = [1, 2, 3];
    const L = cholesky(A);
    const x = cholSolve(L, b);
    A.forEach((row, i) => expect(row.reduce((s, v, j) => s + v * x[j], 0)).toBeCloseTo(b[i], 10));
    // L Lᵀ = A
    A.forEach((row, i) => row.forEach((v, j) => expect(L[i].reduce((s, l, k) => s + l * L[j][k], 0)).toBeCloseTo(v, 10)));
  });
});

describe("statistics helpers", () => {
  it("mutual information is ~0 for independent and large for dependent data", () => {
    const r = mulberry32(1);
    const x = Array.from({ length: 4000 }, () => r());
    const yInd = Array.from({ length: 4000 }, () => r());
    const yDep = x.map((v) => (v - 0.5) ** 2);
    expect(mutualInfo(x, yInd, 8)).toBeLessThan(0.03);
    expect(mutualInfo(x, yDep, 8)).toBeGreaterThan(0.8);
  });
  it("OLS recovers known coefficients", () => {
    const r = mulberry32(2);
    const X = Array.from({ length: 200 }, () => [randn(r), randn(r)]);
    const y = X.map(([a, b]) => 1 + 2 * a - 3 * b);
    const w = ols(X, y);
    expect(w[0]).toBeCloseTo(1, 6); expect(w[1]).toBeCloseTo(2, 6); expect(w[2]).toBeCloseTo(-3, 6);
  });
  it("Poisson sampler has mean ≈ variance ≈ λ", () => {
    const r = mulberry32(3), lam = 4;
    const s = Array.from({ length: 20000 }, () => poissonSample(r, lam));
    const m = s.reduce((a, b) => a + b, 0) / s.length, v = s.reduce((a, b) => a + (b - m) ** 2, 0) / s.length;
    expect(m).toBeGreaterThan(3.9); expect(m).toBeLessThan(4.1);
    expect(v).toBeGreaterThan(3.8); expect(v).toBeLessThan(4.2);
  });
});

describe("text and engineering helpers", () => {
  it("stemmer merges simple inflections", () => {
    expect(stem("running")).toBe("run");
    expect(stem("runs")).toBe("run");
    expect(stem("studies")).toBe("study");
    expect(stem("cat")).toBe("cat");
  });
  it("FNV-1a matches the reference vectors", () => {
    expect(fnv1a("")).toBe("811c9dc5");
    expect(fnv1a("a")).toBe("e40c292c");
    expect(fnv1a("foobar")).toBe("bf9cf968");
  });
  it("structured-output validator accepts valid and rejects invalid JSON", () => {
    expect(validateOutput('{"name": "A", "sentiment": "positive", "rating": 4}')).toEqual([]);
    expect(validateOutput('Sure! {"name": "A"}')[0]).toMatch(/not valid JSON/);
    expect(validateOutput('{"name": "A", "sentiment": "great", "rating": 4}').join()).toMatch(/sentiment/);
    expect(validateOutput('{"name": "A", "sentiment": "neutral", "rating": "four"}').join()).toMatch(/rating/);
  });
});
