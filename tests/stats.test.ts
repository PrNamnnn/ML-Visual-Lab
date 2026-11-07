import { describe, it, expect } from "vitest";
import * as D from "@/lib/statistics/descriptive";
import { distributions, distributionById } from "@/lib/statistics/distributions";
import { normalCdf, normalQuantile, erf, gamma, regularizedBeta, studentTCdf } from "@/lib/math/special";
import { simpson } from "@/lib/math/calculus";
import { transformById, boxCox, yeoJohnson, scalers } from "@/lib/transformations";
import { mulberry32 } from "@/lib/math/random";

const close = (a: number, b: number, tol = 1e-6) => expect(Math.abs(a - b)).toBeLessThan(tol);

describe("special functions", () => {
  it("gamma/erf/normal", () => {
    close(gamma(5), 24, 1e-9);
    close(gamma(0.5), Math.sqrt(Math.PI), 1e-9);
    close(erf(1), 0.8427007929497149, 1e-9);
    close(normalCdf(1.96), 0.9750021048517795, 1e-9);
    close(normalQuantile(0.975), 1.959963984540054, 1e-8);
    close(regularizedBeta(0.5, 2, 3), 0.6875, 1e-9);
    close(studentTCdf(2.0, 10), 0.963306, 1e-5);
  });
});

describe("descriptive", () => {
  const x = [2, 4, 4, 4, 5, 5, 7, 9];
  it("mean / variance / std", () => {
    expect(D.mean(x)).toBe(5);
    close(D.variance(x, false), 4);
    close(D.std(x, false), 2);
    close(D.variance(x), 32 / 7);
    expect(Number.isNaN(D.variance([1]))).toBe(true);
  });
  it("median / quantile / mode", () => {
    expect(D.median([3, 1, 2])).toBe(2);
    expect(D.median([4, 1, 2, 3])).toBe(2.5);
    expect(D.mode(x)).toBe(4);
    const b = D.iqrBounds([1, 2, 3, 4, 5, 6, 7, 8]);
    close(b.q1, 2.75);
    close(b.q3, 6.25);
  });
  it("skewness / kurtosis", () => {
    close(D.skewness([1, 2, 3, 4, 5]), 0);
    close(D.skewness([1, 2, 3, 10]), 1.7636, 1e-3); // G1 = sqrt(n(n-1))/(n-2) · m3/m2^1.5
    close(D.kurtosis([1, 2, 3, 4, 5]), -1.3, 1e-9);
  });
  it("correlation", () => {
    close(D.pearson([1, 2, 3], [2, 4, 6]), 1);
    close(D.pearson([1, 2, 3], [3, 2, 1]), -1);
    close(D.spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1);
    expect(D.ranks([10, 20, 20, 30])).toEqual([1, 2.5, 2.5, 4]);
    expect(Number.isNaN(D.pearson([1, 1, 1], [1, 2, 3]))).toBe(true);
  });
});

describe("distributions", () => {
  it("every continuous pdf integrates to ~1 and matches its CDF", () => {
    for (const d of distributions.filter((d) => d.kind === "continuous")) {
      const p = Object.fromEntries(d.params.map((q) => [q.key, q.default]));
      const [lo, hi] = d.range(p);
      const a = Math.max(lo, d.id === "student-t" ? -60 : lo);
      const mass = simpson((x) => d.density(x, p), a + 1e-9, hi, 4000);
      const cdfMass = d.cdf(hi, p) - d.cdf(a + 1e-9, p);
      close(mass, cdfMass, 2e-3);
    }
  });
  it("every discrete pmf sums to its CDF", () => {
    for (const d of distributions.filter((d) => d.kind === "discrete")) {
      const p = Object.fromEntries(d.params.map((q) => [q.key, q.default]));
      const [lo, hi] = d.range(p);
      let s = 0;
      for (let k = lo; k <= hi; k++) s += d.density(k, p);
      close(s, d.cdf(hi, p), 1e-9);
    }
  });
  it("known values", () => {
    const bin = distributionById("binomial")!;
    close(bin.density(3, { n: 10, p: 0.5 }), 120 / 1024, 1e-12);
    const poi = distributionById("poisson")!;
    close(poi.density(2, { lambda: 3 }), (9 * Math.exp(-3)) / 2, 1e-12);
    close(poi.cdf(2, { lambda: 3 }), Math.exp(-3) * (1 + 3 + 4.5), 1e-9);
    const chi = distributionById("chi-square")!;
    close(chi.cdf(3.841458820694124, { k: 1 }), 0.95, 1e-7);
    const beta = distributionById("beta")!;
    close(beta.mean({ a: 2, b: 5 }), 2 / 7);
  });
  it("sample means approach theoretical means", () => {
    const rng = mulberry32(42);
    for (const id of ["normal", "exponential", "gamma", "beta", "poisson", "binomial"]) {
      const d = distributionById(id)!;
      const p = Object.fromEntries(d.params.map((q) => [q.key, q.default]));
      let s = 0;
      const N = 20000;
      for (let i = 0; i < N; i++) s += d.sample(rng, p);
      const sd = Math.sqrt(d.variance(p) / N);
      expect(Math.abs(s / N - d.mean(p))).toBeLessThan(5 * sd + 1e-9);
    }
  });
});

describe("transformations", () => {
  it("log flags non-positive values", () => {
    const r = transformById("log")!.apply([1, Math.E, 0, -1], 0);
    expect(r.invalid).toEqual([2, 3]);
    close(r.values[1], 1);
  });
  it("box-cox / yeo-johnson formulas", () => {
    close(boxCox(4, 0.5), 2);
    close(boxCox(Math.E, 0), 1);
    close(yeoJohnson(3, 0), Math.log(4));
    close(yeoJohnson(-1, 2), -Math.log(2));
    close(yeoJohnson(-3, 1), -3); // λ=1 is the identity
    close(yeoJohnson(3, 1), 3);
  });
  it("standardisation gives mean 0 sd 1, minmax in [0,1]", () => {
    const x = [3, 7, 1, 9, 4];
    const z = scalers.find((s) => s.id === "standard")!.apply(x).values;
    close(D.mean(z), 0);
    close(D.std(z), 1);
    const m = scalers.find((s) => s.id === "minmax")!.apply(x).values;
    expect(Math.min(...m)).toBe(0);
    expect(Math.max(...m)).toBe(1);
  });
});
