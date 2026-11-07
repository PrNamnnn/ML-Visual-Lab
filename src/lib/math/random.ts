/** Seeded PRNG utilities so every visualization is reproducible. */
export type RNG = () => number;

/** mulberry32 — small, fast, good-enough 32-bit generator for visual demos. */
export function mulberry32(seed: number): RNG {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample via Box–Muller. */
export function randn(rng: RNG): number {
  let u = 0;
  while (u <= Number.EPSILON) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function randInt(rng: RNG, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1));
}

export function shuffle<T>(arr: T[], rng: RNG): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Gamma(k, 1) sample via Marsaglia–Tsang (k > 0). */
export function randGamma(rng: RNG, k: number): number {
  if (k < 1) return randGamma(rng, k + 1) * Math.pow(rng(), 1 / k);
  const d = k - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = randn(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** Bivariate normal sample with correlation rho. */
export function randBivariate(rng: RNG, mx: number, my: number, sx: number, sy: number, rho: number): [number, number] {
  const z1 = randn(rng);
  const z2 = randn(rng);
  const x = mx + sx * z1;
  const y = my + sy * (rho * z1 + Math.sqrt(Math.max(0, 1 - rho * rho)) * z2);
  return [x, y];
}
