import { mulberry32, randn, randGamma } from "@/lib/math/random";

export type SampleId = "lognormal" | "exponential" | "counts" | "negatives" | "left-skewed" | "bimodal" | "outliers" | "uniform";

export const sampleInfo: Record<SampleId, string> = {
  lognormal: "Log-normal (incomes, prices)",
  exponential: "Exponential (waiting times)",
  counts: "Counts with many zeros",
  negatives: "Skewed with negatives (profit/loss)",
  "left-skewed": "Left-skewed (exam scores)",
  bimodal: "Bimodal mixture",
  outliers: "Normal with outliers",
  uniform: "Uniform",
};

/** Deterministic 1-D samples for preprocessing demos. */
export function sample1D(id: SampleId, n = 400, seed = 1): number[] {
  const r = mulberry32(seed);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    switch (id) {
      case "lognormal": out.push(Math.exp(3 + 0.8 * randn(r))); break;
      case "exponential": out.push(-Math.log(1 - r()) * 2); break;
      case "counts": { const lam = randGamma(r, 0.6) * 3; let k = 0, p = Math.exp(-lam), s = p; const u = r(); while (u > s && k < 200) { k++; p *= lam / k; s += p; } out.push(k); break; }
      case "negatives": out.push(randGamma(r, 1.5) * 20 - 15); break;
      case "left-skewed": out.push(100 - randGamma(r, 2) * 8); break;
      case "bimodal": out.push(r() < 0.6 ? 20 + 4 * randn(r) : 45 + 6 * randn(r)); break;
      case "outliers": out.push(i % 40 === 0 ? 50 + 30 * r() : 20 + 3 * randn(r)); break;
      case "uniform": out.push(10 + 30 * r()); break;
    }
  }
  return out;
}
