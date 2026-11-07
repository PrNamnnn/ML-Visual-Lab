/** Number formatting helpers for readouts and KaTeX. All TeX built with String.raw to avoid JS escape sequences. */
export function fmt(x: number, digits = 3): string {
  if (x === Infinity) return "∞";
  if (x === -Infinity) return "−∞";
  if (Number.isNaN(x)) return "undefined";
  if (x !== 0 && (Math.abs(x) >= 1e5 || Math.abs(x) < 1e-3)) return x.toExponential(2).replace("-", "−");
  const r = Number(x.toFixed(digits));
  const s = (Object.is(r, -0) ? 0 : r).toString();
  return s.replace("-", "−");
}

/**
 * Number → TeX. `snap` (optional) rounds |x| < snap to exactly 0 — used only for displaying
 * verification results such as A⁻¹A, where the caller also reports the true residual.
 */
export function tex(x: number, digits = 3, snap = 0): string {
  if (x === Infinity) return String.raw`\infty`;
  if (x === -Infinity) return String.raw`-\infty`;
  if (Number.isNaN(x)) return String.raw`\text{undef}`;
  if (snap > 0 && Math.abs(x) < snap) return "0";
  if (x !== 0 && (Math.abs(x) >= 1e5 || Math.abs(x) < 1e-3)) {
    const [m, e] = x.toExponential(2).split("e");
    return String.raw`${m}\times10^{${Number(e)}}`;
  }
  const r = Number(x.toFixed(digits));
  return (Object.is(r, -0) ? 0 : r).toString();
}

export const pct = (x: number, d = 1) => (Number.isNaN(x) ? "undefined" : `${(x * 100).toFixed(d)}%`);

const ROW = String.raw` \\ `;

export const texMatrix = (M: number[][], digits = 2, kind: "bmatrix" | "pmatrix" = "bmatrix", snap = 0) =>
  String.raw`\begin{${kind}}` + M.map((r) => r.map((v) => tex(v, digits, snap)).join(" & ")).join(ROW) + String.raw`\end{${kind}}`;

export const texVec = (v: number[], digits = 2) => texMatrix(v.map((x) => [x]), digits);

/** Largest absolute entry-wise difference from the identity. */
export const identityResidual = (M: number[][]) => Math.max(...M.flatMap((r, i) => r.map((v, j) => Math.abs(v - (i === j ? 1 : 0)))));

export const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

export const range = (n: number) => Array.from({ length: n }, (_, i) => i);

export const linspace = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

export function cn(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(" ");
}
