"use client";
import { useMemo, useState } from "react";
import { Plot, Arrow, Dot, Handle, Polyline, Label, C } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Toggle, AnimationControls, Presets } from "@/components/controls";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { useTicker } from "@/hooks/useSize";
import { mulberry32, randn } from "@/lib/math/random";
import { fmt, pct, tex, texMatrix, clamp } from "@/lib/utilities/format";
import { jacobiEig } from "@/visualizers/shared";
import { Trace, eigSym2, gaussCloud, meanOf, secondMoment, type V2 } from "./common";

type Mode = "center" | "direction" | "basis" | "choose-k";

/** The four-point dataset used in the week's worked examples (mean (1, 2)). */
const WORKED: V2[] = [[3, 3], [2, 4], [-1, 1], [0, 0]];
const cloud = (seed: number, n = 40): V2[] => gaussCloud(mulberry32(seed), n, 0.62, 1.7, 0.55, [1, 0.6]);
const centre = (X: V2[]): { mu: V2; Xc: V2[] } => { const mu = meanOf(X); return { mu, Xc: X.map((p) => [p[0] - mu[0], p[1] - mu[1]] as V2) }; };

/* ------------------------------------------------------------------ centring and covariance */
function CenterViz() {
  const [X, setX] = useState<V2[]>(WORKED);
  const [centred, setCentred] = useState(false);
  const { mu, Xc } = centre(X);
  const Cm = secondMoment(Xc);
  const raw2 = secondMoment(X);
  const show = centred ? Xc : X;
  const n = X.length;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-4.5, 5.5]} y={[-3.5, 5]} equal originAxes aspect={0.7} maxH={480} ariaLabel="dataset before and after centring">
            {X.map((p, i) => <Polyline key={`t${i}`} pts={[p, Xc[i]]} style={{ stroke: C.a("fg", 0.16), strokeWidth: 1, strokeDasharray: "3 3" }} />)}
            {(centred ? X : Xc).map((p, i) => <Dot key={`g${i}`} p={p} r={3.2} color={C.a("fg", 0.22)} />)}
            <Arrow from={[0, 0]} to={mu} color="amber" label="μ" width={centred ? 1.2 : 2.2} dashed={centred} />
            {centred
              ? show.map((p, i) => <Dot key={i} p={p} r={4.6} color={C.blue} stroke={C.bg} />)
              : X.map((p, i) => <Handle key={i} p={p} color="blue" label={`point x${i + 1}`} onDrag={(q) => setX(X.map((o, j) => (j === i ? [clamp(q[0], -4, 5), clamp(q[1], -3, 4.6)] : o)))} />)}
            {show.map((p, i) => <Label key={`l${i}`} p={p} dx={8} dy={-8} color={C.muted}>{`x${i + 1}`}</Label>)}
          </Plot>
          <Legend items={[{ label: centred ? "centred points xᵢ − μ" : "data points (drag them)", color: C.blue }, { label: "mean μ", color: C.amber }, { label: centred ? "original positions" : "centred positions", color: C.a("fg", 0.3) }]} />
        </div>
      }
      controls={
        <>
          <Toggle label="Show centred dataset" checked={centred} onChange={setCentred} />
          <Presets presets={[{ label: "Worked example", value: "w" }, { label: "Six points", value: "s" }, { label: "Nearly collinear", value: "c" }]} onPick={(v) => { setCentred(false); setX(v === "w" ? WORKED : v === "s" ? [[3, 3], [2, 4], [-1, 1], [0, 0], [4, 2], [-2, -1]] : [[-2, -1.1], [-1, -0.4], [0, 0.1], [1, 0.4], [2, 1.1], [3, 1.4]]); }} />
          <div className="flex flex-wrap items-start gap-4">
            <MatrixView M={Cm} label={<span>C (centred)</span>} digits={3} cell={58} />
            <MatrixView M={raw2} label={<span>(1/n)XXᵀ uncentred</span>} digits={3} cell={58} />
          </div>
          <p className="text-[12.5px] leading-relaxed text-muted">Without centring, (1/n)XXᵀ measures spread about the origin, so the mean leaks into every entry: it equals C + μμᵀ.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "n", tex: true, value: n },
        { label: String.raw`\boldsymbol\mu`, tex: true, value: `(${fmt(mu[0])}, ${fmt(mu[1])})`, color: C.amber },
        { label: String.raw`C_{11}=\text{var}(x_1)`, tex: true, value: fmt(Cm[0][0]) },
        { label: String.raw`C_{22}=\text{var}(x_2)`, tex: true, value: fmt(Cm[1][1]) },
        { label: String.raw`C_{12}=\text{cov}(x_1,x_2)`, tex: true, value: fmt(Cm[0][1]) },
        { label: String.raw`\operatorname{tr}C`, tex: true, value: fmt(Cm[0][0] + Cm[1][1]) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\boldsymbol\mu=\frac1n\sum_{i=1}^{n}\mathbf x_i=\frac1{${n}}${texMatrix([[X.reduce((s, p) => s + p[0], 0)], [X.reduce((s, p) => s + p[1], 0)]], 2)}=${texMatrix([[mu[0]], [mu[1]]], 3)}` },
        { tex: String.raw`\mathbf x_1-\boldsymbol\mu=${texMatrix([[X[0][0]], [X[0][1]]], 2)}-${texMatrix([[mu[0]], [mu[1]]], 3)}=${texMatrix([[Xc[0][0]], [Xc[0][1]]], 3)}`, note: "every point is shifted by the same vector" },
        { tex: String.raw`C=\frac1n\sum_{i}(\mathbf x_i-\boldsymbol\mu)(\mathbf x_i-\boldsymbol\mu)^\top=${texMatrix(Cm, 3)}` },
        { tex: String.raw`\tfrac1nXX^\top-\boldsymbol\mu\boldsymbol\mu^\top=${texMatrix(raw2, 3)}-${texMatrix([[mu[0] * mu[0], mu[0] * mu[1]], [mu[0] * mu[1], mu[1] * mu[1]]], 3)}=C`, note: "check" },
      ]} />}
      note="Centring only translates the cloud: distances between points, and therefore the covariance, are unchanged. It is needed because PCA fits lines and subspaces that pass through the origin."
    />
  );
}

/* ------------------------------------------------------------------ one direction: error versus variance */
function DirectionViz() {
  const [set, setSet] = useState<"worked" | "cloud">("worked");
  const [seed, setSeed] = useState(4);
  const [deg, setDeg] = useState(0);
  const [playing, setPlaying] = useState(false);
  const Xc = useMemo(() => centre(set === "worked" ? WORKED : cloud(seed)).Xc, [set, seed]);
  const Cm = useMemo(() => secondMoment(Xc), [Xc]);
  const eig = useMemo(() => eigSym2(Cm), [Cm]);
  const bestDeg = ((Math.atan2(eig.vectors[0][1], eig.vectors[0][0]) * 180) / Math.PI + 180) % 180;
  useTicker(playing, 30, () => setDeg((d) => { const nd = d + 1; if (nd >= 180) { setPlaying(false); return 180; } return nd; }));
  const th = (deg * Math.PI) / 180;
  const w: V2 = [Math.cos(th), Math.sin(th)];
  const n = Xc.length;
  const proj = Xc.map((p) => { const z = p[0] * w[0] + p[1] * w[1]; return { z, q: [z * w[0], z * w[1]] as V2 }; });
  const variance = proj.reduce((s, p) => s + p.z * p.z, 0) / n;
  const err = Xc.reduce((s, p, i) => s + (p[0] - proj[i].q[0]) ** 2 + (p[1] - proj[i].q[1]) ** 2, 0) / n;
  const total = Cm[0][0] + Cm[1][1];
  const quad = w[0] * (Cm[0][0] * w[0] + Cm[0][1] * w[1]) + w[1] * (Cm[1][0] * w[0] + Cm[1][1] * w[1]);
  const curve = useMemo(() => Array.from({ length: 181 }, (_, d) => { const a = (d * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a); return c * (Cm[0][0] * c + Cm[0][1] * s) + s * (Cm[1][0] * c + Cm[1][1] * s); }), [Cm]);
  const R = Math.max(3.2, ...Xc.map((p) => Math.hypot(p[0], p[1]))) * 1.15;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-R, R]} y={[-R * 0.82, R * 0.82]} equal originAxes aspect={0.66} maxH={440} ariaLabel="projection of centred data onto the direction w">
            <Polyline pts={[[-9 * w[0], -9 * w[1]], [9 * w[0], 9 * w[1]]]} style={{ stroke: C.a("green", 0.45), strokeWidth: 1.2 }} />
            {Xc.map((p, i) => <Polyline key={`e${i}`} pts={[p, proj[i].q]} style={{ stroke: C.a("pink", 0.75), strokeWidth: 1.2 }} />)}
            {proj.map((p, i) => <Dot key={`q${i}`} p={p.q} r={3} color={C.green} />)}
            {Xc.map((p, i) => <Dot key={i} p={p} r={3.6} color={C.blue} stroke={C.bg} />)}
            <Arrow from={[0, 0]} to={w} color="green" width={2.6} label="w" />
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Variance wᵀCw (green) and reconstruction error (pink) for every direction θ — they always add up to tr C</div>
            <Trace ariaLabel="variance and reconstruction error versus direction angle" xLabel="θ (degrees)" xMax={180} marker={deg} yDomain={[0, total * 1.08]}
              series={[{ pts: curve, color: C.green }, { pts: curve.map((v) => total - v), color: C.pink }, { pts: curve.map(() => total), color: C.a("fg", 0.35), dashed: true }]}>
              <Dot p={[bestDeg, eig.values[0]]} r={4} color={C.green} stroke={C.bg} />
              <Dot p={[deg, variance]} r={3.4} color={C.fg} />
            </Trace>
          </div>
          <Legend items={[{ label: "projections (xᵢᵀw)w", color: C.green }, { label: "errors xᵢ − (xᵢᵀw)w", color: C.pink }, { label: "tr C (constant)", color: C.a("fg", 0.45), dashed: true }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Dataset (centred)" value={set} onChange={setSet} options={[{ value: "worked", label: "Worked example" }, { value: "cloud", label: "Random cloud" }]} />
          <ParameterSlider label="direction θ of w" value={deg} min={0} max={180} step={1} onChange={(v) => { setPlaying(false); setDeg(v); }} format={(v) => `${v}°`} />
          <AnimationControls playing={playing} onPlay={() => { if (!playing && deg >= 180) setDeg(0); setPlaying(!playing); }} onReset={() => { setPlaying(false); setDeg(0); }} extra={<button className="btn" onClick={() => { setPlaying(false); setDeg(Math.round(bestDeg)); }}>Best w</button>} />
          {set === "cloud" && <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New sample</button>}
          <MatrixView M={Cm} label={<span>C = (1/n)XXᵀ</span>} digits={3} cell={58} />
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\mathbf w`, tex: true, value: `(${fmt(w[0])}, ${fmt(w[1])})`, color: C.green },
        { label: String.raw`\text{variance}=\mathbf w^\top C\mathbf w`, tex: true, value: fmt(variance, 4), color: C.green },
        { label: String.raw`\text{mean error}`, tex: true, value: fmt(err, 4), color: C.pink },
        { label: String.raw`\text{sum}=\operatorname{tr}C`, tex: true, value: fmt(variance + err, 4) },
        { label: String.raw`\lambda_1\ (\text{best variance})`, tex: true, value: fmt(eig.values[0], 4) },
        { label: String.raw`\lambda_2\ (\text{least error})`, tex: true, value: fmt(eig.values[1], 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\frac1n\sum_i\|\mathbf x_i-(\mathbf x_i^\top\mathbf w)\mathbf w\|^2=\underbrace{\frac1n\sum_i\|\mathbf x_i\|^2}_{${tex(total, 4)}}-\underbrace{\mathbf w^\top C\mathbf w}_{${tex(quad, 4)}}=${tex(total - quad, 4)}`, note: `measured directly: ${fmt(err, 4)}` },
        { tex: String.raw`C\mathbf w_1=\lambda_1\mathbf w_1:\quad\mathbf w_1=(${tex(eig.vectors[0][0])},\ ${tex(eig.vectors[0][1])}),\ \lambda_1=${tex(eig.values[0], 4)}`, note: `best direction θ = ${fmt(bestDeg, 1)}°` },
      ]} />}
      note="Sweep θ: the error is smallest exactly where the variance is largest, at the top eigenvector of C. Both curves are computed from the points, not from the formula; the dashed line shows that their sum never changes."
    />
  );
}

/* ------------------------------------------------------------------ principal components as a new basis */
function BasisViz() {
  const [set, setSet] = useState<"worked" | "cloud">("cloud");
  const [seed, setSeed] = useState(7);
  const [k, setK] = useState<"1" | "2">("1");
  const [sel, setSel] = useState(0);
  const Xc = useMemo(() => centre(set === "worked" ? WORKED : cloud(seed, 36)).Xc, [set, seed]);
  const Cm = useMemo(() => secondMoment(Xc), [Xc]);
  const { values, vectors } = useMemo(() => eigSym2(Cm), [Cm]);
  const [w1, w2] = vectors;
  const coords = Xc.map((p) => [p[0] * w1[0] + p[1] * w1[1], p[0] * w2[0] + p[1] * w2[1]] as V2);
  const rec = coords.map((c) => (k === "2" ? [c[0] * w1[0] + c[1] * w2[0], c[0] * w1[1] + c[1] * w2[1]] : [c[0] * w1[0], c[0] * w1[1]]) as V2);
  const n = Xc.length;
  const err = Xc.reduce((s, p, i) => s + (p[0] - rec[i][0]) ** 2 + (p[1] - rec[i][1]) ** 2, 0) / n;
  const Lam = secondMoment(coords);
  const R = Math.max(3, ...Xc.map((p) => Math.hypot(p[0], p[1]))) * 1.12;
  const i = Math.min(sel, n - 1);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <div className="mb-1 text-[12px] text-faint">Standard basis (e₁, e₂) — click a point</div>
              <Plot x={[-R, R]} y={[-R, R]} equal originAxes aspect={1} maxH={380} ariaLabel="data in the standard basis with principal directions" onPointerDown={(q) => { let b = 0, bd = Infinity; Xc.forEach((p, j) => { const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2; if (d < bd) { bd = d; b = j; } }); setSel(b); }}>
                {k === "1" && Xc.map((p, j) => <Polyline key={`e${j}`} pts={[p, rec[j]]} style={{ stroke: C.a("pink", 0.55), strokeWidth: 1 }} />)}
                {Xc.map((p, j) => <Dot key={j} p={p} r={j === i ? 5 : 3.2} color={C.blue} stroke={j === i ? C.fg : C.bg} />)}
                {k === "1" && rec.map((p, j) => <Dot key={`r${j}`} p={p} r={2.4} color={C.pink} />)}
                <Arrow from={[0, 0]} to={[w1[0] * 1.6, w1[1] * 1.6]} color="green" width={2.6} label="w₁" />
                <Arrow from={[0, 0]} to={[w2[0] * 1.6, w2[1] * 1.6]} color="amber" width={2.4} label="w₂" />
              </Plot>
            </div>
            <div>
              <div className="mb-1 text-[12px] text-faint">Principal basis: coordinates c = Wᵀx</div>
              <Plot x={[-R, R]} y={[-R, R]} equal originAxes aspect={1} maxH={380} xLabel="c₁ = xᵀw₁" yLabel="c₂ = xᵀw₂" ariaLabel="data expressed in principal component coordinates">
                {coords.map((c, j) => <Dot key={j} p={k === "2" ? c : [c[0], 0]} r={j === i ? 5 : 3.2} color={k === "2" ? C.blue : C.pink} stroke={j === i ? C.fg : C.bg} />)}
                {k === "1" && coords.map((c, j) => <Dot key={`o${j}`} p={c} r={2.4} color={C.a("blue", 0.3)} />)}
                <Arrow from={[0, 0]} to={[1.6, 0]} color="green" width={2.6} />
                <Arrow from={[0, 0]} to={[0, 1.6]} color="amber" width={2.4} />
              </Plot>
            </div>
          </div>
          <Legend items={[{ label: "data", color: C.blue }, { label: "w₁ (largest variance)", color: C.green }, { label: "w₂", color: C.amber }, ...(k === "1" ? [{ label: "rank-1 reconstruction and its error", color: C.pink }] : [])]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Dataset (centred)" value={set} onChange={(v) => { setSet(v); setSel(0); }} options={[{ value: "cloud", label: "Random cloud" }, { value: "worked", label: "Worked example" }]} />
          <Segmented label="Components kept" value={k} onChange={setK} options={[{ value: "1", label: "k = 1" }, { value: "2", label: "k = 2 (all)" }]} />
          {set === "cloud" && <button className="btn w-full" onClick={() => { setSeed((s) => s + 1); setSel(0); }}>New sample</button>}
          <div className="flex flex-wrap items-start gap-4">
            <MatrixView M={Cm} label={<span>C in the standard basis</span>} digits={3} cell={58} />
            <MatrixView M={Lam} label={<span>WᵀCW in the principal basis</span>} digits={3} cell={58} />
          </div>
          <p className="text-[12.5px] leading-relaxed text-muted">The off-diagonal entries of WᵀCW vanish: in the principal basis the coordinates are uncorrelated and the diagonal holds λ₁, λ₂.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\lambda_1`, tex: true, value: fmt(values[0], 4), color: C.green },
        { label: String.raw`\lambda_2`, tex: true, value: fmt(values[1], 4), color: C.amber },
        { label: String.raw`\text{reconstruction error}`, tex: true, value: fmt(err, 4), color: C.pink },
        { label: String.raw`\textstyle\sum_{j>k}\lambda_j`, tex: true, value: fmt(k === "1" ? values[1] : 0, 4) },
        { label: String.raw`\mathbf w_1^\top\mathbf w_2`, tex: true, value: fmt(w1[0] * w2[0] + w1[1] * w2[1], 4) },
      ]} />}
      calc={<CalcSteps title={`Live calculation for the selected point x${i + 1}`} steps={[
        { tex: String.raw`\mathbf x=${texMatrix([[Xc[i][0]], [Xc[i][1]]], 3)},\quad W=\begin{bmatrix}\mathbf w_1&\mathbf w_2\end{bmatrix}=${texMatrix([[w1[0], w2[0]], [w1[1], w2[1]]], 3)}` },
        { tex: String.raw`\mathbf c=W^\top\mathbf x=${texMatrix([[coords[i][0]], [coords[i][1]]], 3)}`, note: "coordinates in the principal basis" },
        { tex: k === "2" ? String.raw`\tilde{\mathbf x}=c_1\mathbf w_1+c_2\mathbf w_2=${texMatrix([[rec[i][0]], [rec[i][1]]], 3)}=\mathbf x` : String.raw`\tilde{\mathbf x}=c_1\mathbf w_1=${tex(coords[i][0])}${texMatrix([[w1[0]], [w1[1]]], 3)}=${texMatrix([[rec[i][0]], [rec[i][1]]], 3)},\quad\|\mathbf x-\tilde{\mathbf x}\|^2=c_2^2=${tex(coords[i][1] ** 2, 4)}`, note: k === "2" ? "nothing is lost with all components" : "the dropped coordinate is the error" },
      ]} />}
      note="A change of basis is a rotation here because W is orthogonal (W⁻¹ = Wᵀ). Keeping k coordinates and rotating back gives the rank-k reconstruction; the mean squared error is exactly the sum of the discarded eigenvalues."
    />
  );
}

/* ------------------------------------------------------------------ higher dimensions: how many components? */
function ChooseKViz() {
  const [d, setD] = useState(8);
  const [m, setM] = useState(3);
  const [noise, setNoise] = useState(0.35);
  const [thr, setThr] = useState(0.95);
  const [kManual, setKManual] = useState<number | null>(null);
  const [seed, setSeed] = useState(2);
  const n = 200;
  const res = useMemo(() => {
    const rng = mulberry32(seed * 101 + d * 7 + m);
    // x = A z + noise, with m latent directions of decreasing strength
    const A = Array.from({ length: d }, () => Array.from({ length: m }, () => randn(rng)));
    const X = Array.from({ length: n }, () => {
      const z = Array.from({ length: m }, (_, j) => randn(rng) * (2.2 / (1 + 0.7 * j)));
      return A.map((row) => row.reduce((s, a, j) => s + a * z[j], 0) + randn(rng) * noise);
    });
    const mu = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
    const Xc = X.map((r) => r.map((v, j) => v - mu[j]));
    const Cm = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => Xc.reduce((s, r) => s + r[a] * r[b], 0) / n));
    const e = jacobiEig(Cm, 30);
    const values = e.values.slice().reverse().map((v) => Math.max(0, v));
    const vectors = e.vectors.slice().reverse();
    return { Xc, values, vectors };
  }, [d, m, noise, seed]);
  const total = res.values.reduce((a, b) => a + b, 0);
  const cum = res.values.map((_, j) => res.values.slice(0, j + 1).reduce((a, b) => a + b, 0) / total);
  const kAuto = cum.findIndex((c) => c >= thr - 1e-12) + 1;
  const k = Math.min(d, kManual ?? kAuto);
  // measured reconstruction error with k components
  const measured = useMemo(() => {
    const W = res.vectors.slice(0, k);
    let s = 0;
    for (const x of res.Xc) {
      const c = W.map((w) => w.reduce((a, wi, j) => a + wi * x[j], 0));
      for (let j = 0; j < x.length; j++) { const r = x[j] - W.reduce((a, w, t) => a + c[t] * w[j], 0); s += r * r; }
    }
    return s / res.Xc.length;
  }, [res, k]);
  const tail = res.values.slice(k).reduce((a, b) => a + b, 0);
  const maxV = Math.max(...res.values);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div>
            <div className="mb-1 text-[12px] text-faint">Eigenvalues of C (variance along each principal component), largest first</div>
            <Plot x={[0.4, d + 0.6]} y={[0, maxV * 1.1]} aspect={0.36} minH={150} maxH={220} ticks={d} xLabel="component j" yLabel="λⱼ" ariaLabel="eigenvalue spectrum">
              {res.values.map((v, j) => <Polyline key={j} pts={[[j + 1, 0], [j + 1, v]]} style={{ stroke: j < k ? C.green : C.a("fg", 0.28), strokeWidth: 14 }} />)}
            </Plot>
          </div>
          <div>
            <div className="mb-1 text-[12px] text-faint">Cumulative fraction of variance retained by the first k components</div>
            <Plot x={[0.4, d + 0.6]} y={[0, 1.05]} aspect={0.36} minH={150} maxH={220} ticks={d} xLabel="k" ariaLabel="cumulative explained variance">
              <Polyline pts={[[0, thr], [d + 1, thr]]} style={{ stroke: C.amber, strokeWidth: 1.2, strokeDasharray: "5 4" }} />
              <Polyline pts={cum.map((c, j) => [j + 1, c] as V2)} style={{ stroke: C.green, strokeWidth: 2 }} />
              {cum.map((c, j) => <Dot key={j} p={[j + 1, c]} r={j + 1 === k ? 5 : 3} color={j + 1 === k ? C.fg : C.green} stroke={C.bg} />)}
            </Plot>
          </div>
          <Legend items={[{ label: `kept components (k = ${k})`, color: C.green }, { label: "discarded", color: C.a("fg", 0.35) }, { label: `threshold ${pct(thr, 0)}`, color: C.amber, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="dimension d" value={d} min={3} max={12} step={1} onChange={(v) => { setD(v); setM(Math.min(m, v)); setKManual(null); }} />
          <ParameterSlider label="latent directions in the data" value={m} min={1} max={d} step={1} onChange={(v) => { setM(v); setKManual(null); }} />
          <ParameterSlider label="noise σ" value={noise} min={0} max={1.5} step={0.05} onChange={setNoise} />
          <ParameterSlider label="variance to retain" value={thr} min={0.5} max={0.99} step={0.01} onChange={(v) => { setThr(v); setKManual(null); }} format={(v) => pct(v, 0)} />
          <ParameterSlider label={kManual === null ? "k (set by the threshold)" : "k (manual)"} value={k} min={1} max={d} step={1} onChange={setKManual} />
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={() => setKManual(null)} disabled={kManual === null}>Use threshold</button>
            <button className="btn flex-1" onClick={() => setSeed((s) => s + 1)}>New data</button>
          </div>
        </>
      }
      readouts={<Readouts items={[
        { label: "k", tex: true, value: `${k} of ${d}`, color: C.green },
        { label: String.raw`\text{variance retained}`, tex: true, value: pct(cum[k - 1], 2) },
        { label: String.raw`\textstyle\sum_{j>k}\lambda_j`, tex: true, value: fmt(tail, 4) },
        { label: String.raw`\text{measured error}`, tex: true, value: fmt(measured, 4), color: C.pink },
        { label: String.raw`\text{numbers stored: }dn`, tex: true, value: d * n },
        { label: String.raw`k(d+n)`, tex: true, value: k * (d + n) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{d}\lambda_j}=\frac{${tex(total - tail, 3)}}{${tex(total, 3)}}=${tex(cum[k - 1], 4)}${cum[k - 1] >= thr - 1e-12 ? String.raw`\ \ge\ ` : String.raw`\ <\ `}${tex(thr, 2)}`, note: kManual === null ? `smallest k that reaches the threshold` : cum[k - 1] >= thr ? "meets the threshold" : "below the threshold" },
        { tex: String.raw`\frac1n\sum_i\|\mathbf x_i-\tilde{\mathbf x}_i\|^2=${tex(measured, 4)}\quad\text{and}\quad\sum_{j>k}\lambda_j=${tex(tail, 4)}`, note: "reconstruction error measured from the 200 points equals the discarded eigenvalues" },
        { tex: String.raw`\text{storage: }dn=${d * n}\ \longrightarrow\ k(d+n)=${k * (d + n)}`, note: "k coefficients per point plus k directions of length d" },
      ]} />}
      note={`The data are generated as ${m} latent direction${m > 1 ? "s" : ""} plus isotropic noise in ${d} dimensions, so about ${m} eigenvalue${m > 1 ? "s" : ""} stand out and the rest sit near σ² = ${fmt(noise * noise, 3)}. More noise flattens the spectrum and pushes k up.`}
    />
  );
}

export default function PCAWeek({ mode = "direction" }: { mode?: Mode }) {
  if (mode === "center") return <CenterViz />;
  if (mode === "basis") return <BasisViz />;
  if (mode === "choose-k") return <ChooseKViz />;
  return <DirectionViz />;
}
