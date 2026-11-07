"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Plot, Dot, C, Polyline } from "@/components/visualization/Plot";
import { Bars } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, AnimationControls } from "@/components/controls";
import { makeDataset } from "@/lib/ml/models";
import { mulberry32, randn } from "@/lib/math/random";
import { svd, inverse, matVec, covarianceMatrix, eigSym } from "@/lib/math/linalg";
import { histogram } from "@/lib/statistics/descriptive";
import { useTicker } from "@/hooks/useSize";
import { fmt, pct, tex } from "@/lib/utilities/format";
import { topEig, Heat, clsColor, sqd } from "@/visualizers/shared";

const extentOf = (v: number[], pad = 0.1): [number, number] => {
  const lo = Math.min(...v), hi = Math.max(...v), d = hi - lo || 1;
  return [lo - pad * d, hi + pad * d];
};

/* ------------------------------ kernel PCA ------------------------------ */
export function KernelPCAViz() {
  const [kernel, setKernel] = useState<"linear" | "rbf" | "poly">("rbf");
  const [gamma, setGamma] = useState(0.5);
  const [pair, setPair] = useState<"12" | "13" | "23">("13");
  const pts = useMemo(() => makeDataset("circles", 120, 0.06, 5), []);
  const res = useMemo(() => {
    const n = pts.length;
    const k = (a: (typeof pts)[0], b: (typeof pts)[0]) => {
      const d = a.x * b.x + a.y * b.y;
      if (kernel === "linear") return d;
      if (kernel === "poly") return (d + 1) ** 2;
      return Math.exp(-gamma * ((a.x - b.x) ** 2 + (a.y - b.y) ** 2));
    };
    const K = pts.map((a) => pts.map((b) => k(a, b)));
    // double centring: Kc = K − 1K − K1 + 1K1 (centres the data in feature space)
    const rm = K.map((r) => r.reduce((s, x) => s + x, 0) / n);
    const all = rm.reduce((s, x) => s + x, 0) / n;
    const Kc = K.map((r, i) => r.map((x, j) => x - rm[i] - rm[j] + all));
    const trace = Kc.reduce((s, r, i) => s + r[i], 0);
    const { values, vectors } = topEig(Kc, 3, 400);
    const proj = pts.map((_, i) => values.map((v, c) => Math.sqrt(Math.max(0, v)) * vectors[c][i]));
    // separability of the classes along each component alone: best single-threshold accuracy
    const sep = values.map((_, c) => {
      const order = proj.map((p, i) => ({ v: p[c], l: pts[i].label })).sort((a, b) => a.v - b.v);
      let best = 0;
      for (let t = 0; t <= order.length; t++) {
        let ok = 0;
        order.forEach((o, i) => { if ((i < t ? 0 : 1) === o.l) ok++; });
        best = Math.max(best, ok, order.length - ok);
      }
      return best / order.length;
    });
    return { values, proj, trace, sep };
  }, [pts, kernel, gamma]);
  const [ci, cj] = [Number(pair[0]) - 1, Number(pair[1]) - 1];
  const px = res.proj.map((p) => p[ci]), py = res.proj.map((p) => p[cj]);
  return (
    <LabShell
      viz={
        <div className="grid gap-3 sm:grid-cols-2">
          <Plot x={[-2, 2]} y={[-2, 2]} equal aspect={0.9} maxH={300} ariaLabel="original data: two concentric circles">
            {pts.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={3} color={clsColor(p.label)} />)}
          </Plot>
          <Plot x={extentOf(px)} y={extentOf(py)} aspect={0.9} maxH={300} xLabel={`kernel PC ${ci + 1}`} yLabel={`kernel PC ${cj + 1}`} ariaLabel="data projected on two kernel principal components">
            {res.proj.map((p, i) => <Dot key={i} p={[p[ci], p[cj]]} r={3} color={clsColor(pts[i].label)} />)}
          </Plot>
        </div>
      }
      controls={
        <>
          <Segmented label="Kernel" value={kernel} onChange={setKernel} options={[{ value: "linear", label: "Linear (= PCA)" }, { value: "poly", label: "Poly deg 2" }, { value: "rbf", label: "RBF" }]} />
          {kernel === "rbf" && <ParameterSlider label="RBF width γ" value={gamma} min={0.1} max={10} step={0.1} onChange={setGamma} />}
          <Segmented label="Components plotted" value={pair} onChange={setPair} options={[{ value: "12", label: "1 & 2" }, { value: "13", label: "1 & 3" }, { value: "23", label: "2 & 3" }]} />
          <p className="text-[12px] leading-relaxed text-faint">Left: the raw points. Right: their coordinates on two of the top three principal components of the centred kernel matrix. A linear kernel reproduces ordinary PCA (only a rotation). With an RBF kernel and small γ, components 1–2 still look like x and y, but component 3 measures distance from the centre and splits the rings.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\lambda_1,\\lambda_2,\\lambda_3", tex: true, value: res.values.map((v) => fmt(v, 2)).join(", ") },
        { label: "\\text{variance share (top 3)}", tex: true, value: pct(res.values.reduce((s, v) => s + v, 0) / res.trace) },
        ...res.sep.map((a, c) => ({ label: `\\text{rings split by PC}${c + 1}`, tex: true, value: pct(a), color: a > 0.95 ? C.green : undefined, hint: `best accuracy of a single threshold on kernel PC ${c + 1}` })),
      ]} />}
      calc={<CalcSteps steps={[
        { tex: kernel === "rbf" ? String.raw`K_{ij}=\exp(-${tex(gamma, 1)}\,\|\mathbf x_i-\mathbf x_j\|^2)` : kernel === "poly" ? String.raw`K_{ij}=(\mathbf x_i^\top\mathbf x_j+1)^2` : String.raw`K_{ij}=\mathbf x_i^\top\mathbf x_j` },
        { tex: String.raw`\tilde K=K-\mathbf 1_nK-K\mathbf 1_n+\mathbf 1_nK\mathbf 1_n,\qquad \tilde K\mathbf v_k=\lambda_k\mathbf v_k`, note: "centre in feature space, then eigendecompose (power iteration here)" },
        { tex: String.raw`z_{ik}=\sqrt{\lambda_k}\,v_{k,i}`, note: "projection of training point i on component k" },
      ]} />}
    />
  );
}

/* ------------------------------ truncated SVD / LSA ------------------------------ */
const TERMS = ["cat", "dog", "pet", "vet", "stock", "market", "price", "trade"];
const DOCS = ["d1 pets", "d2 vet", "d3 dogs", "d4 stocks", "d5 trading", "d6 mixed"];
const TD = [
  [2, 0, 1, 0, 0, 1],
  [1, 2, 2, 0, 0, 0],
  [2, 1, 1, 0, 0, 1],
  [0, 2, 1, 0, 0, 0],
  [0, 0, 0, 2, 1, 1],
  [0, 0, 0, 2, 2, 0],
  [0, 0, 0, 1, 1, 1],
  [0, 0, 0, 1, 2, 0],
];

export function TruncatedSVDViz() {
  const [k, setK] = useState(2);
  const { U, S, V } = useMemo(() => svd(TD), []);
  const Ak = TD.map((_, i) => TD[0].map((__, j) => S.slice(0, k).reduce((s, sv, c) => s + U[i][c] * sv * V[j][c], 0)));
  const err = Math.sqrt(TD.reduce((s, r, i) => s + r.reduce((t, x, j) => t + (x - Ak[i][j]) ** 2, 0), 0));
  const tot = S.reduce((s, x) => s + x * x, 0), kept = S.slice(0, k).reduce((s, x) => s + x * x, 0);
  const docXY = V.map((r) => [S[0] * r[0], S[1] * r[1]] as [number, number]);
  const ex = extentOf(docXY.map((p) => p[0]), 0.25), ey = extentOf(docXY.map((p) => p[1]), 0.25);
  return (
    <LabShell
      viz={
        <div className="grid gap-4 xl:grid-cols-2">
          <div className="space-y-2">
            <div className="text-[12px] text-faint">Term–document counts A (8 × 6)</div>
            <Heat M={TD} rowLabels={TERMS} colLabels={DOCS.map((d) => d.split(" ")[0])} digits={0} cell={30} />
            <div className="pt-1 text-[12px] text-faint">Rank-{k} reconstruction A<sub>{k}</sub></div>
            <Heat M={Ak} rowLabels={TERMS} colLabels={DOCS.map((d) => d.split(" ")[0])} digits={1} cell={30} signed max={2} />
          </div>
          <div className="space-y-2">
            <div className="text-[12px] text-faint">Documents in the 2-D topic space (σ₁v₁, σ₂v₂)</div>
            <Plot x={ex} y={ey} aspect={0.8} maxH={300} xLabel="topic 1" yLabel="topic 2" ariaLabel="documents plotted in latent topic space">
              {docXY.map((p, j) => <Dot key={j} p={p} r={5} color={j < 3 ? C.blue : j < 5 ? C.pink : C.amber} />)}
            </Plot>
            <Legend items={[{ label: "pet documents", color: C.blue }, { label: "finance documents", color: C.pink }, { label: "mixed", color: C.amber }]} />
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="rank k (topics kept)" value={k} min={1} max={6} step={1} onChange={setK} />
          <div className="text-[12px] text-faint">Singular values: {S.map((s) => fmt(s, 2)).join(", ")}</div>
          <p className="text-[12px] leading-relaxed text-faint">Truncated SVD keeps only the k largest singular values. Unlike PCA it does not centre the matrix, so it works directly on sparse count matrices — this is Latent Semantic Analysis.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\|A-A_k\\|_F", tex: true, value: fmt(err, 3) },
        { label: "\\text{energy kept}", tex: true, value: pct(kept / tot) },
        { label: "\\text{storage}", tex: true, value: `${k * (8 + 6 + 1)} vs 48 numbers` },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`A\approx A_k=U_k\Sigma_kV_k^\top=\sum_{c=1}^{${k}}\sigma_c\,\mathbf u_c\mathbf v_c^\top` },
        { tex: String.raw`\|A-A_k\|_F=\sqrt{\textstyle\sum_{c>${k}}\sigma_c^2}=${tex(err, 3)}`, note: "Eckart–Young: the best possible rank-k approximation" },
      ]} />}
    />
  );
}

/* ------------------------------ LDA as a projection ------------------------------ */
export function LDAProjViz() {
  const [sep, setSep] = useState(1.6);
  const [rot, setRot] = useState(35);
  const [method, setMethod] = useState<"lda" | "pca">("lda");
  const data = useMemo(() => {
    const rng = mulberry32(11);
    const th = (rot * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th);
    const out: { p: [number, number]; l: number }[] = [];
    for (let l = 0; l < 2; l++)
      for (let i = 0; i < 70; i++) {
        const u = randn(rng) * 2.2, v = randn(rng) * 0.45; // long axis u, short axis v
        const off = (l === 0 ? -1 : 1) * sep * 0.5;
        out.push({ p: [c * u - s * v - s * off, s * u + c * v + c * off], l });
      }
    return out;
  }, [sep, rot]);
  const stats = useMemo(() => {
    const m = [0, 1].map((l) => {
      const P = data.filter((d) => d.l === l).map((d) => d.p);
      return [P.reduce((a, p) => a + p[0], 0) / P.length, P.reduce((a, p) => a + p[1], 0) / P.length];
    });
    const Sw = [[0, 0], [0, 0]];
    data.forEach((d) => {
      const dx = d.p[0] - m[d.l][0], dy = d.p[1] - m[d.l][1];
      Sw[0][0] += dx * dx; Sw[0][1] += dx * dy; Sw[1][0] += dx * dy; Sw[1][1] += dy * dy;
    });
    const wl = matVec(inverse(Sw) ?? [[1, 0], [0, 1]], [m[1][0] - m[0][0], m[1][1] - m[0][1]]);
    const cov = covarianceMatrix(data.map((d) => d.p));
    const pc = eigSym(cov).vectors.map((r) => r[0]);
    return { m, lda: norm([wl[0], wl[1]]), pca: norm([pc[0], pc[1]]) };
  }, [data]);
  const w = method === "lda" ? stats.lda : stats.pca;
  const z = data.map((d) => d.p[0] * w[0] + d.p[1] * w[1]);
  const fisher = (dir: number[]) => {
    const zz = data.map((d) => d.p[0] * dir[0] + d.p[1] * dir[1]);
    const g = [0, 1].map((l) => zz.filter((_, i) => data[i].l === l));
    const mu = g.map((a) => a.reduce((s, x) => s + x, 0) / a.length);
    const v = g.map((a, l) => a.reduce((s, x) => s + (x - mu[l]) ** 2, 0) / a.length);
    return (mu[1] - mu[0]) ** 2 / (v[0] + v[1]);
  };
  const [zlo, zhi] = extentOf(z, 0.05);
  const L = 5;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-5, 5]} y={[-4, 4]} equal aspect={0.6} maxH={340} ariaLabel="two classes with LDA and PCA directions">
            {data.map((d, i) => <Dot key={i} p={d.p} r={2.8} color={clsColor(d.l)} opacity={0.8} />)}
            <Polyline pts={[[-L * stats.lda[0], -L * stats.lda[1]], [L * stats.lda[0], L * stats.lda[1]]]} style={{ stroke: C.green, strokeWidth: method === "lda" ? 2.6 : 1.2 }} />
            <Polyline pts={[[-L * stats.pca[0], -L * stats.pca[1]], [L * stats.pca[0], L * stats.pca[1]]]} style={{ stroke: C.amber, strokeWidth: method === "pca" ? 2.6 : 1.2, strokeDasharray: "6 4" }} />
          </Plot>
          <Plot x={[zlo, zhi]} y={[0, 0.9]} aspect={0.25} minH={120} maxH={170} xLabel={`projection onto the ${method.toUpperCase()} direction`} ariaLabel="histogram of the 1-D projections by class">
            {[0, 1].map((l) => <Bars key={l} bins={histogram(z.filter((_, i) => data[i].l === l), zlo, zhi, 28)} color={l ? C.a("pink", 0.35) : C.a("blue", 0.35)} stroke={l ? C.pink : C.blue} />)}
          </Plot>
          <Legend items={[{ label: "LDA direction (supervised)", color: C.green }, { label: "PCA direction (unsupervised)", color: C.amber, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Project onto" value={method} onChange={setMethod} options={[{ value: "lda", label: "LDA" }, { value: "pca", label: "PCA" }]} />
          <ParameterSlider label="class separation" value={sep} min={0} max={4} step={0.1} onChange={setSep} />
          <ParameterSlider label="orientation of the spread (°)" value={rot} min={0} max={90} step={1} onChange={setRot} />
          <p className="text-[12px] leading-relaxed text-faint">PCA picks the direction of largest total spread — here the long axis, where the classes overlap. LDA uses the labels and picks the direction that separates the class means relative to their within-class spread.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "J(\\mathbf w_{LDA})", tex: true, value: fmt(fisher(stats.lda), 3), color: C.green },
        { label: "J(\\mathbf w_{PCA})", tex: true, value: fmt(fisher(stats.pca), 3), color: C.amber },
        { label: "\\mathbf w_{LDA}", tex: true, value: `(${fmt(stats.lda[0], 3)}, ${fmt(stats.lda[1], 3)})` },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`J(\mathbf w)=\frac{(\mathbf w^\top(\boldsymbol\mu_1-\boldsymbol\mu_0))^2}{\mathbf w^\top S_W\mathbf w}`, note: "Fisher's criterion: between-class separation over within-class spread" },
        { tex: String.raw`\mathbf w^*\propto S_W^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)` },
      ]} />}
    />
  );
}
const norm = (v: [number, number]): [number, number] => { const n = Math.hypot(v[0], v[1]) || 1; const s = v[0] < 0 ? -1 : 1; return [(s * v[0]) / n, (s * v[1]) / n]; };

/* ------------------------------ t-SNE and (simplified) UMAP ------------------------------ */
// 4 clusters in 8 dimensions; two of them are close to each other in the original space
function makeHighDim() {
  const rng = mulberry32(21);
  const e = (d: number, s: number) => Array.from({ length: 8 }, (_, i) => (i === d ? s : 0));
  const centres = [e(0, 6), e(0, 6).map((x, i) => x + (i === 1 ? 3 : 0)), e(2, 6), e(3, 6)]; // clusters 1 and 2 are only 3 apart
  const X: number[][] = [], lab: number[] = [];
  centres.forEach((m, c) => { for (let i = 0; i < 20; i++) { X.push(m.map((x) => x + randn(rng))); lab.push(c); } });
  return { X, lab };
}

function tsneP(X: number[][], perp: number) {
  const n = X.length, D = X.map((a) => X.map((b) => sqd(a, b)));
  const P = Array.from({ length: n }, () => Array(n).fill(0));
  const target = Math.log(perp);
  for (let i = 0; i < n; i++) {
    let lo = 1e-6, hi = 1e6, beta = 1;
    for (let it = 0; it < 60; it++) {
      let s = 0, hs = 0;
      for (let j = 0; j < n; j++) if (j !== i) { const p = Math.exp(-D[i][j] * beta); s += p; hs += D[i][j] * beta * p; }
      const H = Math.log(s) + hs / s; // entropy (nats)
      if (H > target) { lo = beta; beta = hi > 1e5 ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; }
    }
    let s = 0;
    for (let j = 0; j < n; j++) if (j !== i) { P[i][j] = Math.exp(-D[i][j] * beta); s += P[i][j]; }
    for (let j = 0; j < n; j++) P[i][j] /= s;
  }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = (P[i][j] + P[j][i]) / (2 * n); P[i][j] = P[j][i] = v; }
  return P;
}

function umapGraph(X: number[][], k: number) {
  const n = X.length, D = X.map((a) => X.map((b) => Math.sqrt(sqd(a, b))));
  const W = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    const nb = D[i].map((d, j) => ({ d, j })).filter((o) => o.j !== i).sort((a, b) => a.d - b.d).slice(0, k);
    const rho = nb[0].d;
    let lo = 1e-4, hi = 100, sig = 1;
    for (let it = 0; it < 50; it++) { // σ so that Σ exp(−(d−ρ)/σ) = log2 k
      sig = (lo + hi) / 2;
      const s = nb.reduce((a, o) => a + Math.exp(-(o.d - rho) / sig), 0);
      if (s > Math.log2(k)) hi = sig; else lo = sig;
    }
    nb.forEach((o) => { W[i][o.j] = Math.exp(-(o.d - rho) / sig); });
  }
  const edges: { i: number; j: number; w: number }[] = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const w = W[i][j] + W[j][i] - W[i][j] * W[j][i]; // fuzzy union
    if (w > 1e-3) edges.push({ i, j, w });
  }
  return edges;
}

export function EmbeddingViz({ method: initial = "tsne" }: { method?: "tsne" | "umap" }) {
  const [method, setMethod] = useState<"tsne" | "umap">(initial);
  const [perp, setPerp] = useState(10);
  const [k, setK] = useState(10);
  const [playing, setPlaying] = useState(true);
  const { X, lab } = useMemo(makeHighDim, []);
  const P = useMemo(() => (method === "tsne" ? tsneP(X, perp) : null), [X, perp, method]);
  const edges = useMemo(() => (method === "umap" ? umapGraph(X, k) : []), [X, k, method]);
  const init = () => { const r = mulberry32(3); return X.map(() => [randn(r) * 1e-2 * (method === "umap" ? 500 : 1), randn(r) * 1e-2 * (method === "umap" ? 500 : 1)]); };
  const Y = useRef<number[][]>(init());
  const gains = useRef<number[][]>(X.map(() => [1, 1]));
  const vel = useRef<number[][]>(X.map(() => [0, 0]));
  const [iter, setIter] = useState(0);
  const [kl, setKl] = useState(NaN);
  const tRef = useRef(0); // iteration counter that is always current (state may lag between rapid ticks)
  const reset = () => { Y.current = init(); gains.current = X.map(() => [1, 1]); vel.current = X.map(() => [0, 0]); tRef.current = 0; setIter(0); setPlaying(true); };
  useEffect(reset, [method, perp, k]); // eslint-disable-line react-hooks/exhaustive-deps
  const MAX = method === "tsne" ? 600 : 300;
  const step = () => {
    const n = X.length, y = Y.current;
    if (tRef.current >= MAX) return;
    if (method === "tsne" && P) {
      for (let rep = 0; rep < 4; rep++) {
        const t = tRef.current++, exag = t < 100 ? 4 : 1;
        const num = y.map((a) => y.map((b) => 1 / (1 + sqd(a, b))));
        let Z = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) Z += num[i][j];
        // compute every gradient from the same positions, then move all points
        const G = y.map((yi, i) => {
          let gx = 0, gy = 0;
          for (let j = 0; j < n; j++) if (i !== j) {
            const m = (exag * P[i][j] - num[i][j] / Z) * num[i][j];
            gx += 4 * m * (yi[0] - y[j][0]); gy += 4 * m * (yi[1] - y[j][1]);
          }
          return [gx, gy];
        });
        for (let i = 0; i < n; i++) {
          for (let d = 0; d < 2; d++) {
            const g = G[i][d];
            gains.current[i][d] = Math.sign(g) !== Math.sign(vel.current[i][d]) ? gains.current[i][d] + 0.2 : Math.max(0.01, gains.current[i][d] * 0.8);
            vel.current[i][d] = (t < 250 ? 0.5 : 0.8) * vel.current[i][d] - 100 * gains.current[i][d] * g;
            y[i][d] += vel.current[i][d];
          }
        }
        if (rep === 3) {
          let s = 0;
          for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j && P[i][j] > 1e-12) s += P[i][j] * Math.log(P[i][j] / (num[i][j] / Z));
          setKl(s);
        }
      }
    } else {
      const t = tRef.current++;
      const rng = mulberry32(t + 1), a = 1.58, b = 0.9, lr = 1 - t / MAX;
      for (const e of edges) {
        const d2 = sqd(y[e.i], y[e.j]) + 1e-6;
        const c = (-2 * a * b * d2 ** (b - 1)) / (1 + a * d2 ** b);
        for (let d = 0; d < 2; d++) {
          const g = Math.max(-4, Math.min(4, c * (y[e.i][d] - y[e.j][d]))) * e.w * lr;
          y[e.i][d] += g; y[e.j][d] -= g;
        }
        for (let s = 0; s < 3; s++) { // negative samples push random pairs apart
          const j = Math.floor(rng() * n);
          if (j === e.i) continue;
          const dd = sqd(y[e.i], y[j]) + 1e-6;
          const cr = (2 * b) / ((0.001 + dd) * (1 + a * dd ** b));
          for (let d = 0; d < 2; d++) y[e.i][d] += Math.max(-4, Math.min(4, cr * (y[e.i][d] - y[j][d]))) * lr;
        }
      }
    }
    setIter(tRef.current);
  };
  useTicker(playing && iter < MAX, 30, step);
  const y = Y.current;
  const ex = extentOf(y.map((p) => p[0]), 0.08), ey = extentOf(y.map((p) => p[1]), 0.08);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={ex} y={ey} equal aspect={0.7} maxH={420} axes={false} ariaLabel={`${method} embedding of 80 points from 8 dimensions`}>
            {method === "umap" && edges.filter((e) => e.w > 0.5).map((e, i) => <Polyline key={i} pts={[y[e.i], y[e.j]]} style={{ stroke: C.a("fg", 0.07), strokeWidth: 1 }} />)}
            {y.map((p, i) => <Dot key={i} p={p} r={3.5} color={clsColor(lab[i])} />)}
          </Plot>
          <Legend items={[0, 1, 2, 3].map((c) => ({ label: `cluster ${c + 1}${c < 2 ? " (1 and 2 are close in 8-D)" : ""}`, color: clsColor(c) }))} />
        </div>
      }
      controls={
        <>
          <Segmented label="Method" value={method} onChange={setMethod} options={[{ value: "tsne", label: "t-SNE" }, { value: "umap", label: "UMAP (simplified)" }]} />
          {method === "tsne" ? <ParameterSlider label="perplexity (effective neighbours)" value={perp} min={2} max={40} step={1} onChange={setPerp} /> : <ParameterSlider label="n_neighbors" value={k} min={3} max={30} step={1} onChange={setK} />}
          <AnimationControls playing={playing} onPlay={() => setPlaying((p) => !p)} onStep={step} onReset={reset} />
          <p className="text-[12px] leading-relaxed text-faint">Both methods only try to keep each point&apos;s neighbours close. Distances between far-apart clusters and cluster sizes in the picture are not meaningful.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{iteration}", tex: true, value: iter },
        ...(method === "tsne" ? [{ label: "\\mathrm{KL}(P\\,\\|\\,Q)", tex: true, value: fmt(kl, 4) }] : [{ label: "\\text{graph edges}", tex: true, value: edges.length }]),
        { label: "\\text{input}", tex: true, value: "80 points × 8 dims" },
      ]} />}
      calc={<CalcSteps steps={method === "tsne" ? [
        { tex: String.raw`p_{j|i}=\frac{e^{-\beta_i\|\mathbf x_i-\mathbf x_j\|^2}}{\sum_{k\ne i}e^{-\beta_i\|\mathbf x_i-\mathbf x_k\|^2}},\quad 2^{H(P_i)}=${perp}`, note: "β_i found by binary search so each point has the chosen perplexity" },
        { tex: String.raw`q_{ij}\propto(1+\|\mathbf y_i-\mathbf y_j\|^2)^{-1},\qquad \min_Y\ \mathrm{KL}(P\|Q)=\sum p_{ij}\log\frac{p_{ij}}{q_{ij}}`, note: "heavy-tailed Student-t in 2-D avoids crowding" },
      ] : [
        { tex: String.raw`w_{ij}=\exp\!\Big(-\frac{d_{ij}-\rho_i}{\sigma_i}\Big)\ \text{for the }${k}\text{ nearest},\quad w^{sym}=w_{ij}+w_{ji}-w_{ij}w_{ji}`, note: "fuzzy k-NN graph" },
        { tex: String.raw`q_{ij}=\big(1+a\,\|\mathbf y_i-\mathbf y_j\|^{2b}\big)^{-1},\ \text{SGD on cross-entropy with negative sampling}`, note: "simplified: fixed a, b and random initialisation (real UMAP uses a spectral initialisation)" },
      ]} />}
    />
  );
}
