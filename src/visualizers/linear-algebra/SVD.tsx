"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Plot, Arrow, C, Polyline } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { MatrixInput, ParameterSlider, Segmented, Presets } from "@/components/controls";
import { svd2, svd, matMul, matVec, type Matrix } from "@/lib/math/linalg";
import { fmt, tex, texMatrix } from "@/lib/utilities/format";

const rot = (t: number): Matrix => [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]];
const circle = (M: Matrix) => Array.from({ length: 121 }, (_, i) => { const t = (i / 120) * 2 * Math.PI; return matVec(M, [Math.cos(t), Math.sin(t)]) as [number, number]; });

function Decompose() {
  const [A, setA] = useState<Matrix>([[1.6, 0.9], [0.4, 1.2]]);
  const [stage, setStage] = useState(3); // 0: I, 1: Vᵀ, 2: ΣVᵀ, 3: UΣVᵀ
  const s = svd2(A);
  const Vt: Matrix = [[s.V[0][0], s.V[1][0]], [s.V[0][1], s.V[1][1]]];
  const S: Matrix = [[s.S[0], 0], [0, s.S[1]]];
  const stages: Matrix[] = [[[1, 0], [0, 1]], Vt, matMul(S, Vt), matMul(s.U, matMul(S, Vt))];
  const M = stages[stage];
  const e1 = matVec(M, [1, 0]) as [number, number];
  const e2 = matVec(M, [0, 1]) as [number, number];
  const v1 = matVec(M, [s.V[0][0], s.V[1][0]]) as [number, number];
  const v2 = matVec(M, [s.V[0][1], s.V[1][1]]) as [number, number];
  const recon = stages[3];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-3, 3]} equal originAxes aspect={0.75} maxH={480} ariaLabel="SVD as rotate, scale, rotate">
            <Polyline pts={circle([[1, 0], [0, 1]])} style={{ stroke: C.lineStrong, strokeDasharray: "3 3", strokeWidth: 1 }} />
            <Polyline pts={circle(M)} style={{ stroke: C.fg, strokeWidth: 1.8 }} />
            <Arrow from={[0, 0]} to={e1} color="muted" width={1.2} label="e₁" />
            <Arrow from={[0, 0]} to={e2} color="muted" width={1.2} label="e₂" />
            <Arrow from={[0, 0]} to={v1} color="green" width={2.6} label={stage === 3 ? "σ₁u₁" : "v₁"} />
            <Arrow from={[0, 0]} to={v2} color="amber" width={2.6} label={stage === 3 ? "σ₂u₂" : "v₂"} />
          </Plot>
          <Legend items={[{ label: "image of v₁ (right singular vector 1)", color: C.green }, { label: "image of v₂", color: C.amber }, { label: "image of unit circle", color: C.fg }]} />
        </div>
      }
      controls={
        <>
          <MatrixInput label="A" value={A} onChange={setA} step={0.1} />
          <Segmented label="Stage" value={String(stage)} onChange={(v) => setStage(Number(v))} options={[{ value: "0", label: "x" }, { value: "1", label: "Vᵀx" }, { value: "2", label: "ΣVᵀx" }, { value: "3", label: "UΣVᵀx" }]} />
          <Presets presets={[{ label: "Shear", value: [[1, 1], [0, 1]] }, { label: "Symmetric", value: [[2, 1], [1, 2]] }, { label: "Near-singular", value: [[1, 1], [1, 1.05]] }, { label: "Reflection mix", value: [[0.5, 1.2], [1.4, -0.3]] }]} onPick={(m) => setA(m)} />
          <p className="text-[12px] leading-relaxed text-faint">Vᵀ rotates v₁, v₂ onto the axes; Σ stretches by σ₁, σ₂; U rotates the axes onto u₁, u₂.{s.S[1] < 0 ? " det A < 0: the reflection is absorbed as a negative second scale so U and V can be drawn as pure rotations." : ""}</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\sigma_1", tex: true, value: fmt(Math.abs(s.S[0])), color: C.green },
        { label: "\\sigma_2", tex: true, value: fmt(Math.abs(s.S[1])), color: C.amber },
        { label: "\\text{rotation }V", tex: true, value: `${fmt((s.thetaV * 180) / Math.PI, 1)}°` },
        { label: "\\text{rotation }U", tex: true, value: `${fmt((s.thetaU * 180) / Math.PI, 1)}°` },
        { label: "|\\det A|=\\sigma_1\\sigma_2", tex: true, value: fmt(Math.abs(s.S[0] * s.S[1])) },
      ]} />}
      calc={<CalcSteps steps={[{ tex: `A=${texMatrix(s.U, 3)}${texMatrix([[Math.abs(s.S[0]), 0], [0, Math.abs(s.S[1])]], 3)}${texMatrix(Vt.map((r, i) => (s.S[1] < 0 && i === 1 ? r.map((x) => -x) : r)), 3)}` }, { tex: `U\\Sigma V^\\top=${texMatrix(recon, 3)}` }]} />}
    />
  );
}

/** A 32×32 test pattern with a few smooth components plus edges. */
function pattern(n = 32): Matrix {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const x = j / (n - 1) - 0.5, y = i / (n - 1) - 0.5;
      const ring = Math.exp(-((Math.hypot(x, y) - 0.28) ** 2) / 0.004);
      const bar = Math.abs(x + 0.15) < 0.06 && y > -0.35 && y < 0.35 ? 0.8 : 0;
      const grad = 0.3 * (y + 0.5);
      const checker = i < 9 && j > 22 ? ((Math.floor(i / 3) + Math.floor(j / 3)) % 2) * 0.7 : 0;
      return ring + bar + grad + checker;
    }),
  );
}

function HeatCanvas({ M, label }: { M: Matrix; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const g = c?.getContext("2d");
    if (!c || !g) return;
    const n = M.length;
    const img = g.createImageData(n, n);
    const max = Math.max(...M.flat()), min = Math.min(...M.flat());
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = (M[i][j] - min) / (max - min || 1);
      const k = (i * n + j) * 4;
      img.data[k] = 20 + 70 * v; img.data[k + 1] = 22 + 150 * v; img.data[k + 2] = 30 + 225 * v; img.data[k + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }, [M]);
  return (
    <figure className="flex flex-col items-center gap-1">
      <canvas ref={ref} width={M.length} height={M.length} className="h-[180px] w-[180px] rounded border border-line" style={{ imageRendering: "pixelated" }} aria-label={label} role="img" />
      <figcaption className="text-[12px] text-muted">{label}</figcaption>
    </figure>
  );
}

function LowRank() {
  const A = useMemo(() => pattern(32), []);
  const dec = useMemo(() => svd(A), [A]);
  const [k, setK] = useState(3);
  const Ak = useMemo(() => {
    const n = A.length;
    const R = Array.from({ length: n }, () => Array(n).fill(0));
    for (let r = 0; r < k; r++) {
      const s = dec.S[r];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) R[i][j] += s * dec.U[i][r] * dec.V[j][r];
    }
    return R;
  }, [A, dec, k]);
  const total = dec.S.reduce((s, x) => s + x * x, 0);
  const kept = dec.S.slice(0, k).reduce((s, x) => s + x * x, 0);
  const err = Math.sqrt(Math.max(0, total - kept));
  const errDirect = Math.sqrt(A.flat().reduce((s, v, i) => s + (v - Ak.flat()[i]) ** 2, 0));
  const maxS = dec.S[0];
  return (
    <LabShell
      viz={
        <div className="space-y-4">
          <div className="flex flex-wrap justify-center gap-6">
            <HeatCanvas M={A} label="original (rank 32)" />
            <HeatCanvas M={Ak} label={`rank-${k} approximation`} />
          </div>
          <div>
            <div className="label mb-1">Singular values σᵢ (log scale)</div>
            <div className="flex h-20 items-end gap-[2px]">
              {dec.S.map((s, i) => (
                <div key={i} className="flex-1 rounded-t-sm" title={`σ${i + 1} = ${fmt(s)}`} style={{ height: `${Math.max(2, (Math.log10(s + 1e-6) + 6) / (Math.log10(maxS) + 6)) * 100}%`, background: i < k ? "rgb(var(--blue))" : "rgb(var(--line-strong))" }} />
              ))}
            </div>
          </div>
        </div>
      }
      controls={<ParameterSlider label="rank k" value={k} min={1} max={32} step={1} onChange={setK} />}
      readouts={<Readouts items={[
        { label: "\\text{energy kept}", tex: true, value: `${fmt((100 * kept) / total, 2)}%` },
        { label: "\\|A-A_k\\|_F", tex: true, value: fmt(errDirect) },
        { label: "\\sqrt{\\sum_{i>k}\\sigma_i^2}", tex: true, value: fmt(err) },
        { label: "\\text{numbers stored}", tex: true, value: `${k * (32 + 32 + 1)} vs 1024` },
      ]} />}
      calc={<CalcSteps steps={[{ tex: `A_{${k}}=\\sum_{i=1}^{${k}}\\sigma_i\\mathbf u_i\\mathbf v_i^\\top,\\quad\\sigma_1=${tex(dec.S[0])},\\;\\sigma_{${k}}=${tex(dec.S[k - 1])}` }]} />}
      note="Eckart–Young: the two error values agree (up to rounding) — the Frobenius error is exactly the energy of the discarded singular values."
    />
  );
}

export default function SVD({ mode = "decompose" }: { mode?: "decompose" | "lowrank" }) {
  return mode === "lowrank" ? <LowRank /> : <Decompose />;
}

export { rot };
