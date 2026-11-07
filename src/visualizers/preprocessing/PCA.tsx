"use client";
import { useMemo, useState } from "react";
import { Plot, Arrow, Dot, C, Polyline } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Toggle, AnimationControls, Presets } from "@/components/controls";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { covarianceMatrix, eigSym } from "@/lib/math/linalg";
import { mulberry32, randBivariate } from "@/lib/math/random";
import { useTicker } from "@/hooks/useSize";
import { fmt, tex, texMatrix } from "@/lib/utilities/format";

export default function PCA({ focus = "pca", lab = false }: { focus?: "pca" | "covariance"; lab?: boolean }) {
  const [n, setN] = useState(80);
  const [rho, setRho] = useState(0.75);
  const [sx, setSx] = useState(1.4);
  const [sy, setSy] = useState(0.8);
  const [mx, setMx] = useState(1);
  const [my, setMy] = useState(0.5);
  const [seed, setSeed] = useState(3);
  const [centered, setCentered] = useState(focus === "pca");
  const [k, setK] = useState<1 | 2>(1);
  const [t, setT] = useState(focus === "pca" ? 1 : 0);
  const [playing, setPlaying] = useState(false);
  useTicker(playing, 16, () => setT((v) => { const nv = v + 0.015; if (nv >= 1) { setPlaying(false); return 1; } return nv; }));

  const X = useMemo(() => {
    const rng = mulberry32(seed);
    return Array.from({ length: n }, () => randBivariate(rng, mx, my, sx, sy, rho));
  }, [n, rho, sx, sy, mx, my, seed]);
  const mu = useMemo(() => [X.reduce((s, p) => s + p[0], 0) / n, X.reduce((s, p) => s + p[1], 0) / n], [X, n]);
  const S = useMemo(() => covarianceMatrix(X), [X]);
  const { values, vectors } = useMemo(() => eigSym(S), [S]);
  const v1: [number, number] = [vectors[0][0], vectors[1][0]];
  const v2: [number, number] = [vectors[0][1], vectors[1][1]];
  const total = values[0] + values[1];
  const off: [number, number] = centered ? [0, 0] : [mu[0], mu[1]];

  // projections onto PC1 (k=1) — reconstruction in original coordinates
  const pts = X.map((p) => {
    const c: [number, number] = [p[0] - mu[0], p[1] - mu[1]];
    const z1 = c[0] * v1[0] + c[1] * v1[1];
    const z2 = c[0] * v2[0] + c[1] * v2[1];
    const rec: [number, number] = k === 2 ? c : [z1 * v1[0], z1 * v1[1]];
    const show: [number, number] = [c[0] + (rec[0] - c[0]) * t + off[0], c[1] + (rec[1] - c[1]) * t + off[1]];
    return { orig: [c[0] + off[0], c[1] + off[1]] as [number, number], show, z1, z2, err: k === 2 ? 0 : z2 * z2 };
  });
  const recErr = pts.reduce((s, p) => s + p.err, 0) / n;

  // 2σ ellipse
  const ell: [number, number][] = Array.from({ length: 101 }, (_, i) => {
    const a = (i / 100) * 2 * Math.PI;
    const r1 = 2 * Math.sqrt(Math.max(0, values[0])), r2 = 2 * Math.sqrt(Math.max(0, values[1]));
    return [off[0] + r1 * Math.cos(a) * v1[0] + r2 * Math.sin(a) * v2[0], off[1] + r1 * Math.cos(a) * v1[1] + r2 * Math.sin(a) * v2[1]];
  });
  const L = 2.2 * Math.sqrt(Math.max(1e-9, values[0]));
  const L2 = 2.2 * Math.sqrt(Math.max(1e-9, values[1]));

  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <Plot x={[-5, 6]} y={[-4.5, 5]} equal aspect={0.72} maxH={520} originAxes ariaLabel="PCA scatter plot with principal directions">
            <Polyline pts={ell} closed style={{ stroke: C.a("fg", 0.35), strokeWidth: 1, strokeDasharray: "4 3" }} />
            {focus === "pca" && k === 1 && <Polyline pts={[[off[0] - 8 * v1[0], off[1] - 8 * v1[1]], [off[0] + 8 * v1[0], off[1] + 8 * v1[1]]]} style={{ stroke: C.a("green", 0.4), strokeWidth: 1 }} />}
            {focus === "pca" && t > 0 && k === 1 && pts.map((p, i) => <Polyline key={`r${i}`} pts={[p.orig, p.show]} style={{ stroke: C.a("pink", 0.35), strokeWidth: 0.8 }} />)}
            {pts.map((p, i) => <Dot key={i} p={focus === "pca" ? p.show : p.orig} r={3} color={C.a("blue", 0.85)} />)}
            <Arrow from={off} to={[off[0] + L * v1[0], off[1] + L * v1[1]]} color="green" width={2.8} label="PC1" />
            <Arrow from={off} to={[off[0] + L2 * v2[0], off[1] + L2 * v2[1]]} color="amber" width={2.4} label="PC2" />
            {!centered && <Dot p={[mu[0], mu[1]]} r={4.5} color={C.fg} />}
          </Plot>
          <Legend items={[{ label: `PC1 — λ₁ = ${fmt(values[0])}`, color: C.green }, { label: `PC2 — λ₂ = ${fmt(values[1])}`, color: C.amber }, { label: "2σ covariance ellipse", color: C.a("fg", 0.5), dashed: true }, ...(focus === "pca" && k === 1 ? [{ label: "reconstruction error (⊥ PC1)", color: C.pink }] : [])]} />
        </div>
      }
      controls={
        <>
          <div className="grid grid-cols-2 gap-x-3 gap-y-3">
            <ParameterSlider label="correlation ρ" value={rho} min={-0.99} max={0.99} step={0.01} onChange={setRho} />
            <ParameterSlider label="samples n" value={n} min={10} max={300} step={5} onChange={setN} />
            <ParameterSlider label="σₓ" value={sx} min={0.2} max={2.5} step={0.05} onChange={setSx} />
            <ParameterSlider label="σᵧ" value={sy} min={0.2} max={2.5} step={0.05} onChange={setSy} />
            <ParameterSlider label="mean x" value={mx} min={-2} max={2} step={0.1} onChange={setMx} />
            <ParameterSlider label="mean y" value={my} min={-2} max={2} step={0.1} onChange={setMy} />
          </div>
          <Toggle label="Mean-centre the data" checked={centered} onChange={setCentered} />
          {focus === "pca" && (
            <>
              <Segmented label="Components kept k" value={String(k)} onChange={(v) => setK(Number(v) as 1 | 2)} options={[{ value: "1", label: "k = 1" }, { value: "2", label: "k = 2" }]} />
              <ParameterSlider label="projection progress" value={t} min={0} max={1} step={0.01} onChange={(v) => { setPlaying(false); setT(v); }} />
              <AnimationControls playing={playing} onPlay={() => { if (!playing && t >= 1) setT(0); setPlaying(!playing); }} onReset={() => { setPlaying(false); setT(0); }} />
            </>
          )}
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={() => setSeed((s) => s + 1)}>New sample</button>
          </div>
          <Presets presets={[{ label: "Strong correlation", value: [0.92, 1.6, 1.2] }, { label: "Uncorrelated", value: [0, 1.5, 0.6] }, { label: "Isotropic", value: [0, 1, 1] }, { label: "Negative", value: [-0.8, 1.2, 1.2] }]} onPick={([r, a, b]) => { setRho(r); setSx(a); setSy(b); }} />
          <div className="space-y-1">
            <div className="label">Explained variance</div>
            {[values[0], values[1]].map((v, i) => (
              <div key={i} className="flex items-center gap-2 text-[12px]">
                <span className="w-8 text-muted">PC{i + 1}</span>
                <div className="h-2 flex-1 rounded bg-raised"><div className="h-2 rounded" style={{ width: `${(100 * v) / total}%`, background: i ? "rgb(var(--amber))" : "rgb(var(--green))" }} /></div>
                <span className="num w-12 text-right">{fmt((100 * v) / total, 1)}%</span>
              </div>
            ))}
          </div>
        </>
      }
      readouts={
        <div className="flex flex-wrap items-center gap-6">
          <MatrixView M={S} label={<span>Sample covariance Σ</span>} digits={3} cell={56} />
          <div className="min-w-[220px] flex-1">
            <Readouts items={[
              { label: "\\bar{\\mathbf x}", tex: true, value: `(${fmt(mu[0])}, ${fmt(mu[1])})` },
              { label: "\\operatorname{tr}\\Sigma=\\lambda_1+\\lambda_2", tex: true, value: fmt(total) },
              { label: "\\text{sample }r", tex: true, value: fmt(S[0][1] / Math.sqrt(S[0][0] * S[1][1])) },
              ...(focus === "pca" ? [{ label: "\\text{recon. MSE}", tex: true, value: k === 2 ? "0 (lossless)" : fmt(recErr) }, { label: "\\lambda_2(n-1)/n", tex: true, value: fmt((values[1] * (n - 1)) / n) }] : []),
            ]} />
          </div>
        </div>
      }
      calc={
        <CalcSteps steps={[
          { tex: `\\Sigma=\\frac{1}{n-1}\\sum_i(\\mathbf x_i-\\bar{\\mathbf x})(\\mathbf x_i-\\bar{\\mathbf x})^\\top=${texMatrix(S, 3)}` },
          { tex: `\\Sigma\\mathbf v_1=\\lambda_1\\mathbf v_1:\\ \\lambda_1=${tex(values[0])},\\ \\mathbf v_1=(${tex(v1[0])},${tex(v1[1])})` },
          ...(focus === "pca" ? [{ tex: `z_i=\\mathbf v_1^\\top(\\mathbf x_i-\\bar{\\mathbf x}),\\quad\\hat{\\mathbf x}_i=\\bar{\\mathbf x}+z_i\\mathbf v_1`, note: "project then reconstruct" }, { tex: `\\text{explained variance ratio}=\\lambda_1/(\\lambda_1+\\lambda_2)=${tex(values[0] / total)}` }] : []),
        ]} />
      }
      note={focus === "pca" ? "The reconstruction MSE (per sample, squared distance to PC1) equals λ₂·(n−1)/n exactly — the variance PCA discards. PCA is sensitive to feature scale: standardise first when features have different units." : lab ? undefined : "Covariance is symmetric and positive semidefinite; its eigenvectors are the principal axes of the ellipse."}
    />
  );
}
