"use client";
import { useMemo, useState } from "react";
import { Plot, Arrow, Dot, Handle, Polyline, C } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented } from "@/components/controls";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { mulberry32, randn, randInt } from "@/lib/math/random";
import { fmt, pct, tex, texMatrix, clamp } from "@/lib/utilities/format";
import { jacobiEig, BarList } from "@/visualizers/shared";
import { eigSym2, meanOf, secondMoment, type V2 } from "./common";

type Shape = "circle" | "ellipse" | "parabola" | "line";

function makeShape(shape: Shape, n: number, noise: number, seed: number): V2[] {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, (_, i) => {
    const t = (i + rng() * 0.6) / n;
    const e: V2 = [randn(rng) * noise, randn(rng) * noise];
    if (shape === "circle") return [1.6 * Math.cos(2 * Math.PI * t) + e[0], 1.6 * Math.sin(2 * Math.PI * t) + e[1]];
    if (shape === "ellipse") return [2 * Math.cos(2 * Math.PI * t) + e[0], 1.1 * Math.sin(2 * Math.PI * t) + e[1]];
    if (shape === "parabola") { const x = -1.8 + 3.6 * t; return [x + e[0], 0.8 * x * x - 1.2 + e[1]]; }
    const x = -2 + 4 * t; return [x + e[0], 0.6 * x + e[1]];
  });
}

/** Fraction of the total variance left over after projecting centred 2-D points onto their best line. */
function lineResidual(P: V2[]) {
  const mu = meanOf(P);
  const Cm = secondMoment(P.map((p) => [p[0] - mu[0], p[1] - mu[1]]));
  const e = eigSym2(Cm);
  return { mu, e, frac: e.values[1] / Math.max(1e-12, e.values[0] + e.values[1]) };
}

/* ------------------------------------------------------------------ two limitations of linear PCA */
function LimitsViz() {
  const [shape, setShape] = useState<Shape>("circle");
  const [noise, setNoise] = useState(0.06);
  const [logD, setLogD] = useState(4);
  const [logN, setLogN] = useState(2.3);
  const P = useMemo(() => makeShape(shape, 70, noise, 3), [shape, noise]);
  const { mu, e, frac } = useMemo(() => lineResidual(P), [P]);
  const w = e.vectors[0];
  const d = Math.round(10 ** logD), n = Math.round(10 ** logN);
  const costC = d ** 3, costK = n ** 3;
  const useK = n < d;
  return (
    <LabShell
      viz={
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <div className="text-[12px] text-faint">Issue 1 — structure that is not a line. Best PCA line and what it misses</div>
            <Plot x={[-3, 3]} y={[-2.6, 2.6]} equal originAxes aspect={0.86} maxH={380} ariaLabel="best PCA line through non-linear data">
              <Polyline pts={[[mu[0] - 9 * w[0], mu[1] - 9 * w[1]], [mu[0] + 9 * w[0], mu[1] + 9 * w[1]]]} style={{ stroke: C.green, strokeWidth: 1.6 }} />
              {P.map((p, i) => { const z = (p[0] - mu[0]) * w[0] + (p[1] - mu[1]) * w[1]; return <Polyline key={`e${i}`} pts={[p, [mu[0] + z * w[0], mu[1] + z * w[1]]]} style={{ stroke: C.a("pink", 0.5), strokeWidth: 1 }} />; })}
              {P.map((p, i) => <Dot key={i} p={p} r={3} color={C.blue} />)}
            </Plot>
            <Legend items={[{ label: "first principal component", color: C.green }, { label: "reconstruction errors", color: C.pink }]} />
          </div>
          <div className="space-y-3">
            <div className="text-[12px] text-faint">Issue 2 — cost of the eigendecomposition for n points in d dimensions</div>
            <BarList digits={2} max={Math.max(logD, logN) * 3} items={[
              { label: "C = (1/n)XXᵀ is d × d", value: 3 * logD, color: useK ? C.a("fg", 0.3) : C.green, note: "= log₁₀ d³" },
              { label: "K = XᵀX is n × n", value: 3 * logN, color: useK ? C.green : C.a("fg", 0.3), note: "= log₁₀ n³" },
            ]} />
            <p className="text-[13px] leading-relaxed text-muted">
              Decompose <span className="text-fg">{useK ? "K (n × n)" : "C (d × d)"}</span>: about {fmt(Math.min(costC, costK))} operations instead of {fmt(Math.max(costC, costK))}
              {costC !== costK && <> — a factor of {fmt(Math.max(costC, costK) / Math.min(costC, costK))}</>}. Both matrices share the same non-zero eigenvalues, at most min(d, n) = {Math.min(d, n)} of them.
            </p>
          </div>
        </div>
      }
      controls={
        <>
          <Segmented label="Shape of the data" value={shape} onChange={setShape} options={[{ value: "circle", label: "Circle" }, { value: "ellipse", label: "Ellipse" }, { value: "parabola", label: "Parabola" }, { value: "line", label: "Line" }]} />
          <ParameterSlider label="noise" value={noise} min={0} max={0.4} step={0.01} onChange={setNoise} />
          <ParameterSlider label="features d" value={logD} min={0.5} max={6} step={0.1} onChange={setLogD} format={() => String(d)} />
          <ParameterSlider label="data points n" value={logN} min={0.5} max={6} step={0.1} onChange={setLogN} format={() => String(n)} />
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\lambda_1`, tex: true, value: fmt(e.values[0], 4), color: C.green },
        { label: String.raw`\lambda_2`, tex: true, value: fmt(e.values[1], 4), color: C.pink },
        { label: String.raw`\text{variance lost with }k=1`, tex: true, value: pct(frac, 1), color: C.pink },
        { label: String.raw`d^3`, tex: true, value: fmt(costC) },
        { label: String.raw`n^3`, tex: true, value: fmt(costK) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\frac{\lambda_2}{\lambda_1+\lambda_2}=\frac{${tex(e.values[1], 3)}}{${tex(e.values[0] + e.values[1], 3)}}=${tex(frac, 3)}`, note: shape === "line" ? "a line is exactly what PCA can represent" : "one linear component cannot summarise a curve" },
        { tex: String.raw`O(d^3)\ \text{vs}\ O(n^3):\quad ${d}^3=${tex(costC, 3)},\qquad ${n}^3=${tex(costK, 3)}` },
      ]} />}
      note="Points on a circle have one degree of freedom (the angle), yet the best line still discards about half of the variance. The cure for the cost issue is the n × n matrix XᵀX; the cure for the non-linearity is to replace its inner products by a kernel."
    />
  );
}

/* ------------------------------------------------------------------ XXᵀ and XᵀX share their non-zero spectrum */
function DualityViz() {
  const [d, setD] = useState(3);
  const [n, setN] = useState(2);
  const [seed, setSeed] = useState(1);
  const [idx, setIdx] = useState(0);
  const res = useMemo(() => {
    const rng = mulberry32(seed * 31 + d * 5 + n);
    const X = seed === 1 && d === 3 && n === 2 ? [[2, 1], [1, 2], [1, 1]] : Array.from({ length: d }, () => Array.from({ length: n }, () => randInt(rng, -2, 3)));
    const XXt = X.map((r) => X.map((q) => r.reduce((s, v, j) => s + v * q[j], 0)));
    const XtX = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => X.reduce((s, r) => s + r[a] * r[b], 0)));
    const eD = jacobiEig(XXt, 30), eN = jacobiEig(XtX, 30);
    return { X, XXt, XtX, valD: eD.values.slice().reverse(), valN: eN.values.slice().reverse(), vecN: eN.vectors.slice().reverse() };
  }, [d, n, seed]);
  const r = Math.min(d, n);
  const k = Math.min(idx, r - 1);
  const mu = Math.max(0, res.valN[k]);
  let beta = res.vecN[k];
  if ((beta.find((v) => Math.abs(v) > 1e-9) ?? 1) < 0) beta = beta.map((v) => -v);
  const Xb = res.X.map((row) => row.reduce((s, v, j) => s + v * beta[j], 0));
  const w = mu > 1e-9 ? Xb.map((v) => v / Math.sqrt(mu)) : Xb.map(() => 0);
  const XXtw = res.XXt.map((row) => row.reduce((s, v, j) => s + v * w[j], 0));
  const resid = Math.hypot(...XXtw.map((v, i) => v - mu * w[i]));
  const cell = Math.max(34, 54 - 3 * Math.max(d, n));
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-4">
          <div className="flex flex-wrap items-start gap-x-6 gap-y-4">
            <MatrixView M={res.X} label={<span>X ({d} × {n}): columns are data points</span>} digits={0} cell={cell} />
            <MatrixView M={res.XXt} label={<span>XXᵀ ({d} × {d})</span>} digits={0} cell={cell} />
            <MatrixView M={res.XtX} label={<span>K = XᵀX ({n} × {n})</span>} digits={0} cell={cell} />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <div className="label mb-2">Eigenvalues of XXᵀ</div>
              <BarList digits={4} max={Math.max(1e-9, res.valD[0])} items={res.valD.map((v, i) => ({ label: `μ${i + 1}`, value: Math.abs(v) < 1e-9 ? 0 : v, color: i === k && i < r ? C.green : i < r ? C.blue : C.a("fg", 0.3) }))} />
            </div>
            <div>
              <div className="label mb-2">Eigenvalues of XᵀX</div>
              <BarList digits={4} max={Math.max(1e-9, res.valD[0])} items={res.valN.map((v, i) => ({ label: `μ${i + 1}`, value: Math.abs(v) < 1e-9 ? 0 : v, color: i === k && i < r ? C.green : i < r ? C.blue : C.a("fg", 0.3) }))} />
            </div>
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="features d (rows of X)" value={d} min={1} max={6} step={1} onChange={(v) => { setD(v); setIdx(0); }} />
          <ParameterSlider label="data points n (columns of X)" value={n} min={1} max={6} step={1} onChange={(v) => { setN(v); setIdx(0); }} />
          <ParameterSlider label="eigenpair to convert" value={k + 1} min={1} max={r} step={1} onChange={(v) => setIdx(v - 1)} disabled={r < 2} />
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={() => { setSeed((s) => s + 1); setIdx(0); }}>New matrix</button>
            <button className="btn flex-1" onClick={() => { setD(3); setN(2); setSeed(1); setIdx(0); }}>Worked example</button>
          </div>
          <p className="text-[12.5px] leading-relaxed text-muted">The first min(d, n) = {r} eigenvalues agree to rounding error. The larger matrix only adds {Math.abs(d - n)} extra zero{Math.abs(d - n) === 1 ? "" : "s"}.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\mu_k\ \text{of}\ X^\top X`, tex: true, value: fmt(res.valN[k], 5), color: C.green },
        { label: String.raw`\mu_k\ \text{of}\ XX^\top`, tex: true, value: fmt(res.valD[k], 5), color: C.green },
        { label: String.raw`\|\mathbf w\|`, tex: true, value: fmt(Math.hypot(...w), 5) },
        { label: String.raw`\|XX^\top\mathbf w-\mu\mathbf w\|`, tex: true, value: fmt(resid, 6) },
        { label: String.raw`\operatorname{tr}(XX^\top)=\operatorname{tr}(X^\top X)`, tex: true, value: fmt(res.XXt.reduce((s, row, i) => s + row[i], 0)) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`X^\top X\,\boldsymbol\beta=\mu\boldsymbol\beta:\quad\mu=${tex(mu, 4)},\ \boldsymbol\beta=${texMatrix(beta.map((v) => [v]), 3)}`, note: "eigenpair of the n × n matrix" },
        { tex: String.raw`\mathbf w=\frac{X\boldsymbol\beta}{\sqrt{\mu}}=\frac{1}{${tex(Math.sqrt(mu), 4)}}${texMatrix(Xb.map((v) => [v]), 3)}=${texMatrix(w.map((v) => [v]), 3)}`, note: "unit eigenvector of the d × d matrix" },
        { tex: String.raw`XX^\top\mathbf w=${texMatrix(XXtw.map((v) => [v]), 3)}=${tex(mu, 4)}\,\mathbf w`, note: `residual ${fmt(resid, 6)}` },
      ]} />}
      note="Eigenvalues are computed independently for the two matrices with the Jacobi method, then compared. With the 1/n scaling of the covariance, λ = μ/n; the eigenvectors are unaffected."
    />
  );
}

/* ------------------------------------------------------------------ feature maps and kernel values */
const featNames = ["1", "√2·x₁", "√2·x₂", "x₁²", "x₂²", "√2·x₁x₂"];
const phi2 = (p: V2) => [1, Math.SQRT2 * p[0], Math.SQRT2 * p[1], p[0] * p[0], p[1] * p[1], Math.SQRT2 * p[0] * p[1]];
const binom = (a: number, b: number) => { let r = 1; for (let i = 1; i <= b; i++) r = (r * (a - b + i)) / i; return Math.round(r); };

function LiftViz() {
  const [shape, setShape] = useState<Shape>("circle");
  const [noise, setNoise] = useState(0.05);
  const [x, setX] = useState<V2>([1, 2]);
  const [y, setY] = useState<V2>([3, -1]);
  const [p, setP] = useState(2);
  const [sigma, setSigma] = useState(1.5);
  const P = useMemo(() => makeShape(shape, 70, noise, 5), [shape, noise]);
  const lift = (q: V2): V2 => (shape === "parabola" ? [q[0] * q[0], q[1]] : [q[0] * q[0], q[1] * q[1]]);
  const Q = useMemo(() => P.map(lift), [P]); // eslint-disable-line react-hooks/exhaustive-deps
  const orig = useMemo(() => lineResidual(P), [P]);
  const lifted = useMemo(() => lineResidual(Q), [Q]);
  const dot = x[0] * y[0] + x[1] * y[1];
  const kPoly = (1 + dot) ** p;
  const fx = phi2(x), fy = phi2(y);
  const explicit = fx.reduce((s, v, i) => s + v * fy[i], 0);
  const d2 = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2;
  const kRbf = Math.exp(-d2 / (2 * sigma * sigma));
  const qx = Q.map((q) => q[0]), qy = Q.map((q) => q[1]);
  const ext = (v: number[]): [number, number] => { const lo = Math.min(...v), hi = Math.max(...v), s = (hi - lo || 1) * 0.12; return [lo - s, hi + s]; };
  const lw = lifted.e.vectors[0], lm = lifted.mu;
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <div className="mb-1 text-[12px] text-faint">Input space ℝ² — drag x and y</div>
              <Plot x={[-3.4, 3.4]} y={[-3.4, 3.4]} equal originAxes aspect={1} maxH={380} ariaLabel="data in the original input space">
                {P.map((q, i) => <Dot key={i} p={q} r={2.8} color={C.a("blue", 0.8)} />)}
                <Arrow from={[0, 0]} to={x} color="green" label="x" />
                <Arrow from={[0, 0]} to={y} color="amber" label="y" />
                <Handle p={x} color="green" label="vector x" onDrag={(q) => setX([clamp(q[0], -3.2, 3.2), clamp(q[1], -3.2, 3.2)])} />
                <Handle p={y} color="amber" label="vector y" onDrag={(q) => setY([clamp(q[0], -3.2, 3.2), clamp(q[1], -3.2, 3.2)])} />
              </Plot>
            </div>
            <div>
              <div className="mb-1 text-[12px] text-faint">Two coordinates of the feature space: ({shape === "parabola" ? "x₁², x₂" : "x₁², x₂²"})</div>
              <Plot x={ext(qx)} y={ext(qy)} aspect={1} maxH={380} xLabel="x₁²" yLabel={shape === "parabola" ? "x₂" : "x₂²"} ariaLabel="data after the non-linear feature map">
                <Polyline pts={[[lm[0] - 50 * lw[0], lm[1] - 50 * lw[1]], [lm[0] + 50 * lw[0], lm[1] + 50 * lw[1]]]} style={{ stroke: C.green, strokeWidth: 1.4 }} />
                {Q.map((q, i) => <Dot key={i} p={q} r={2.8} color={C.a("pink", 0.85)} />)}
              </Plot>
            </div>
          </div>
          <Legend items={[{ label: "original points", color: C.blue }, { label: "transformed points φ(x)", color: C.pink }, { label: "best line in feature space", color: C.green }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Shape of the data" value={shape} onChange={setShape} options={[{ value: "circle", label: "Circle" }, { value: "ellipse", label: "Ellipse" }, { value: "parabola", label: "Parabola" }]} />
          <ParameterSlider label="noise" value={noise} min={0} max={0.3} step={0.01} onChange={setNoise} />
          <ParameterSlider label="polynomial degree p" value={p} min={1} max={5} step={1} onChange={setP} />
          <ParameterSlider label="Gaussian width σ" value={sigma} min={0.3} max={4} step={0.1} onChange={setSigma} />
          {p === 2 && (
            <div>
              <div className="label mb-1.5">φ(x) and φ(y) for p = 2</div>
              <div className="grid grid-cols-[auto_1fr_1fr] gap-x-3 gap-y-0.5 text-[12px]">
                <span className="text-faint">feature</span><span className="num text-right" style={{ color: C.green }}>φ(x)</span><span className="num text-right" style={{ color: C.amber }}>φ(y)</span>
                {featNames.map((f, i) => <div key={f} className="contents"><span className="text-muted">{f}</span><span className="num text-right">{fmt(fx[i])}</span><span className="num text-right">{fmt(fy[i])}</span></div>)}
              </div>
            </div>
          )}
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\mathbf x^\top\mathbf y`, tex: true, value: fmt(dot, 4) },
        { label: String.raw`k_{\text{poly}}=(1+\mathbf x^\top\mathbf y)^{p}`, tex: true, value: fmt(kPoly, 4), color: C.pink },
        { label: String.raw`\dim\phi=\tbinom{2+p}{p}`, tex: true, value: binom(2 + p, p) },
        { label: String.raw`k_{\text{Gauss}}`, tex: true, value: fmt(kRbf, 4) },
        { label: String.raw`\text{variance off the line, input}`, tex: true, value: pct(orig.frac, 1) },
        { label: String.raw`\text{variance off the line, features}`, tex: true, value: pct(lifted.frac, 1), color: C.green },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`k(\mathbf x,\mathbf y)=(1+\mathbf x^\top\mathbf y)^{${p}}=(1+${tex(dot, 3)})^{${p}}=${tex(kPoly, 4)}`, note: "one inner product in ℝ², then a power" },
        ...(p === 2 ? [{ tex: String.raw`\phi(\mathbf x)^\top\phi(\mathbf y)=${fx.map((v, i) => `(${tex(v * fy[i], 2)})`).join("+")}=${tex(explicit, 4)}`, note: `six products in feature space give the same number (difference ${fmt(Math.abs(explicit - kPoly), 6)})` }] : [{ tex: String.raw`\phi:\mathbb R^2\to\mathbb R^{${binom(2 + p, p)}}`, note: "the explicit map is never formed; set p = 2 to see it written out" }]),
        { tex: String.raw`k(\mathbf x,\mathbf y)=\exp\!\Big(-\frac{\|\mathbf x-\mathbf y\|^2}{2\sigma^2}\Big)=\exp\!\Big(-\frac{${tex(d2, 3)}}{${tex(2 * sigma * sigma, 3)}}\Big)=${tex(kRbf, 4)}`, note: "Gaussian kernel: infinite-dimensional φ" },
      ]} />}
      note="Squaring the coordinates turns x₁² + x₂² = r² into the straight line u + v = r²: a relationship that is non-linear in the inputs becomes linear in the features, so ordinary PCA works there. The kernel returns the feature-space inner product without ever building φ."
    />
  );
}

export default function KernelWeek({ mode = "lift" }: { mode?: "limits" | "duality" | "lift" }) {
  if (mode === "limits") return <LimitsViz />;
  if (mode === "duality") return <DualityViz />;
  return <LiftViz />;
}
