"use client";
import { useEffect, useMemo, useState } from "react";
import { Plot, Arrow, Handle, C, Polyline, usePlot } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend, Warn } from "@/components/visualization/LabShell";
import { MatrixInput, ParameterSlider, Presets, Select, Toggle, AnimationControls } from "@/components/controls";
import { det2, inverse, eigen2, matVec, matMul, svd2, type Matrix } from "@/lib/math/linalg";
import { useTicker } from "@/hooks/useSize";
import { fmt, tex, texMatrix, identityResidual } from "@/lib/utilities/format";

type Mode = "transform" | "determinant" | "singular" | "inverse" | "eigen" | "condition";

const PRESETS: Record<string, Matrix> = {
  identity: [[1, 0], [0, 1]],
  rotation: [[Math.cos(Math.PI / 6), -Math.sin(Math.PI / 6)], [Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)]],
  shear: [[1, 1], [0, 1]],
  scale: [[2, 0], [0, 0.5]],
  reflection: [[1, 0], [0, -1]],
  symmetric: [[2, 1], [1, 2]],
  projection: [[1, 0], [0, 0]],
  general: [[1.5, 0.5], [0.3, 1.1]],
};

const lerpM = (A: Matrix, t: number): Matrix => [[1 + (A[0][0] - 1) * t, A[0][1] * t], [A[1][0] * t, 1 + (A[1][1] - 1) * t]];

function Grid({ M, color = C.a("blue", 0.32), extent = 6 }: { M: Matrix; color?: string; extent?: number }) {
  const lines: [number, number][][] = [];
  for (let k = -extent; k <= extent; k++) {
    lines.push([matVec(M, [k, -extent]) as [number, number], matVec(M, [k, extent]) as [number, number]]);
    lines.push([matVec(M, [-extent, k]) as [number, number], matVec(M, [extent, k]) as [number, number]]);
  }
  return <g>{lines.map((l, i) => <Polyline key={i} pts={l} style={{ stroke: color, strokeWidth: 1 }} />)}</g>;
}

function Parallelogram({ M, fill }: { M: Matrix; fill: string }) {
  const { sx, sy } = usePlot();
  const pts = [[0, 0], [1, 0], [1, 1], [0, 1]].map((p) => matVec(M, p));
  return <path d={pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0])},${sy(p[1])}`).join("") + "Z"} style={{ fill, stroke: "none" }} />;
}

function Circle({ M, style }: { M: Matrix; style?: React.CSSProperties }) {
  const pts: [number, number][] = Array.from({ length: 121 }, (_, i) => {
    const t = (i / 120) * 2 * Math.PI;
    return matVec(M, [Math.cos(t), Math.sin(t)]) as [number, number];
  });
  return <Polyline pts={pts} style={style} />;
}

export default function MatrixTransform({ mode: initial = "transform", preset, lab = false }: { mode?: Mode; preset?: string; lab?: boolean }) {
  const [mode, setMode] = useState<Mode>(initial);
  const start = PRESETS[preset ?? (initial === "eigen" ? "symmetric" : initial === "singular" ? "general" : "general")] ?? PRESETS.general;
  const [A, setA] = useState<Matrix>(start);
  const [t, setT] = useState(1); // animation progress identity→A
  const [playing, setPlaying] = useState(false);
  const [probe, setProbe] = useState<[number, number]>([Math.cos(0.4), Math.sin(0.4)]);
  const [applyInverse, setApplyInverse] = useState(false);
  const [s, setS] = useState(0.4); // singular slider

  useTicker(playing, 16, () => setT((x) => { const n = x + 0.02; if (n >= 1) { setPlaying(false); return 1; } return n; }));

  // singular mode builds A from slider: columns a and (1-s)·perp + s·a
  const Aeff: Matrix = useMemo(() => {
    if (mode !== "singular") return A;
    const a = [1.5, 0.5];
    const b0 = [-0.4, 1.2];
    const b = [b0[0] * (1 - s) + a[0] * 0.8 * s, b0[1] * (1 - s) + a[1] * 0.8 * s];
    return [[a[0], b[0]], [a[1], b[1]]];
  }, [mode, A, s]);

  const D = det2(Aeff);
  const inv = inverse(Aeff);
  const eig = eigen2(Aeff);
  const sv = svd2(Aeff);
  const shown = lerpM(Aeff, t);
  const finalM = mode === "inverse" && applyInverse && inv ? matMul(inv, Aeff) : shown;

  useEffect(() => setT(1), [mode]);

  const steps: { tex: string; note?: string }[] = [];
  const readouts: { label: string; value: React.ReactNode; tex?: boolean; color?: string }[] = [
    { label: "\\det A", tex: true, value: fmt(D), color: Math.abs(D) < 1e-9 ? C.red : D < 0 ? C.amber : C.fg },
    { label: "\\operatorname{tr} A", tex: true, value: fmt(Aeff[0][0] + Aeff[1][1]) },
  ];
  steps.push({ tex: String.raw`\det A=ad-bc=(${tex(Aeff[0][0], 3)})(${tex(Aeff[1][1], 3)})-(${tex(Aeff[0][1], 3)})(${tex(Aeff[1][0], 3)})=${tex(Aeff[0][0] * Aeff[1][1], 4)}-(${tex(Aeff[0][1] * Aeff[1][0], 4)})=${tex(D, 4)}` });

  const Ae1 = matVec(finalM, [1, 0]) as [number, number];
  const Ae2 = matVec(finalM, [0, 1]) as [number, number];
  const legend: { label: string; color: string; dashed?: boolean }[] = [{ label: "A e₁ (column 1)", color: C.blue }, { label: "A e₂ (column 2)", color: C.pink }];

  let extra: React.ReactNode = null;
  if (mode === "determinant" || mode === "transform" || mode === "singular") {
    readouts.push({ label: "\\text{area scale}", tex: true, value: `×${fmt(Math.abs(D))}` }, { label: "\\text{orientation}", tex: true, value: Math.abs(D) < 1e-9 ? "collapsed" : D > 0 ? "preserved" : "flipped" });
    if (Math.abs(D) < 1e-9) steps.push({ tex: "\\det A=0\\Rightarrow\\text{the plane collapses onto a line (or point): } A \\text{ is singular}" });
    else steps.push({ tex: `\\text{unit square (area 1)}\\mapsto\\text{parallelogram of area }|${tex(D)}|=${tex(Math.abs(D))}` });
  }
  if (mode === "inverse") {
    if (inv) {
      const P = matMul(inv, Aeff);
      const res = identityResidual(P);
      const [[a, b], [c, d]] = Aeff;
      readouts.push({ label: "A^{-1}", tex: true, value: `[[${fmt(inv[0][0], 4)}, ${fmt(inv[0][1], 4)}], [${fmt(inv[1][0], 4)}, ${fmt(inv[1][1], 4)}]]` }, { label: "\\det A^{-1}=1/\\det A", tex: true, value: fmt(det2(inv), 4) }, { label: "\\max|A^{-1}A-I|", tex: true, value: res === 0 ? "0 (exact)" : res.toExponential(1) });
      steps.push({ tex: String.raw`\text{adj}(A)=\begin{bmatrix}d&-b\\-c&a\end{bmatrix}=` + texMatrix([[d, -b], [-c, a]], 3), note: "swap the diagonal, negate the off-diagonal" });
      steps.push({ tex: String.raw`A^{-1}=\frac{1}{\det A}\,\text{adj}(A)=\frac{1}{${tex(D, 4)}}` + texMatrix([[d, -b], [-c, a]], 3) + "=" + texMatrix(inv, 4) });
      steps.push({
        tex: String.raw`A^{-1}A=` + texMatrix(inv, 4) + texMatrix(Aeff, 3) + (res === 0 ? "=" : String.raw`\approx`) + texMatrix(P, 4, "bmatrix", 1e-9) + "=I",
        note: res === 0 ? "exact in floating point" : `entries differ from I by at most ${res.toExponential(1)} — floating-point rounding, not a mathematical error`,
      });
    } else {
      readouts.push({ label: "A^{-1}", tex: true, value: "does not exist", color: C.red });
      steps.push({ tex: "\\det A=0\\Rightarrow A^{-1}\\text{ does not exist: points that collapsed together cannot be separated again}" });
    }
  }
  let eigLines: React.ReactNode = null;
  if (mode === "eigen") {
    const Ap = matVec(Aeff, probe) as [number, number];
    const cross = probe[0] * Ap[1] - probe[1] * Ap[0];
    const parallel = Math.abs(cross) < 0.04 * Math.hypot(...Ap);
    if (eig.real && eig.vectors) {
      readouts.push(
        { label: "\\lambda_1", tex: true, value: fmt(eig.values[0]), color: C.green },
        { label: "\\lambda_2", tex: true, value: fmt(eig.values[1]), color: C.amber },
        { label: "\\mathbf v_1", tex: true, value: `(${fmt(eig.vectors[0][0])}, ${fmt(eig.vectors[0][1])})` },
        { label: "\\mathbf v_2", tex: true, value: `(${fmt(eig.vectors[1][0])}, ${fmt(eig.vectors[1][1])})` },
      );
      eigLines = (
        <>
          {[0, 1].map((i) => {
            const v = eig.vectors![i];
            return <Polyline key={i} pts={[[-8 * v[0], -8 * v[1]], [8 * v[0], 8 * v[1]]]} style={{ stroke: i ? C.amber : C.green, strokeWidth: 1.2, strokeDasharray: "6 4" }} />;
          })}
          {[0, 1].map((i) => {
            const v = eig.vectors![i];
            const l = eig.values[i];
            return <Arrow key={`e${i}`} from={[0, 0]} to={[l * v[0], l * v[1]]} color={i ? "amber" : "green"} width={3} label={`λ${i + 1}v${i + 1}`} />;
          })}
        </>
      );
      const tr = Aeff[0][0] + Aeff[1][1];
      steps.push({ tex: `\\lambda^2-${tex(tr)}\\lambda+${tex(D)}=0\\;\\Rightarrow\\;\\lambda=\\dfrac{${tex(tr)}\\pm\\sqrt{${tex(tr * tr - 4 * D)}}}{2}=${tex(eig.values[0])},\\;${tex(eig.values[1])}` });
      if (Math.abs(eig.values[0] - eig.values[1]) < 1e-9 && eig.vectors[0] === eig.vectors[1]) steps.push({ tex: "\\text{repeated eigenvalue with one eigen-direction: defective (not diagonalisable)}" });
    } else {
      readouts.push({ label: "\\lambda", tex: true, value: `${fmt(eig.values[0])} ± ${fmt(eig.imag)}i`, color: C.amber });
      steps.push({ tex: `\\operatorname{tr}^2-4\\det=${tex((Aeff[0][0] + Aeff[1][1]) ** 2 - 4 * D)}<0\\Rightarrow\\text{complex eigenvalues: no real direction is preserved (rotation component)}` });
    }
    steps.push({ tex: `A\\mathbf x=(${tex(Ap[0])},${tex(Ap[1])})${parallel ? "\\parallel\\mathbf x\\;\\checkmark" : "\\not\\parallel\\mathbf x"}`, note: parallel ? "x is (close to) an eigenvector" : "drag x around the circle" });
    extra = (
      <>
        <Circle M={[[1, 0], [0, 1]]} style={{ stroke: C.lineStrong, strokeDasharray: "3 3", strokeWidth: 1 }} />
        <Circle M={Aeff} style={{ stroke: C.a("fg", 0.5), strokeWidth: 1.2 }} />
        {eigLines}
        <Arrow from={[0, 0]} to={Ap} color={parallel ? "green" : "fg"} width={2} label="Ax" />
        <Arrow from={[0, 0]} to={probe} color="blue" width={2} label="x" />
        <Handle p={probe} onDrag={(q) => { const n = Math.hypot(q[0], q[1]) || 1; setProbe([q[0] / n, q[1] / n]); }} color="blue" label="probe vector x on unit circle" />
      </>
    );
  }
  if (mode === "condition") {
    const k = Math.abs(sv.S[1]) < 1e-12 ? Infinity : Math.abs(sv.S[0] / sv.S[1]);
    readouts.push({ label: "\\sigma_1", tex: true, value: fmt(Math.abs(sv.S[0])) }, { label: "\\sigma_2", tex: true, value: fmt(Math.abs(sv.S[1])) }, { label: "\\kappa(A)", tex: true, value: fmt(k), color: k > 100 ? C.red : k > 10 ? C.amber : C.green }, { label: "\\text{digits lost}", tex: true, value: Number.isFinite(k) ? `≈ ${fmt(Math.log10(k), 1)}` : "all" });
    steps.push({ tex: `\\kappa(A)=\\dfrac{\\sigma_{\\max}}{\\sigma_{\\min}}=\\dfrac{${tex(Math.abs(sv.S[0]))}}{${tex(Math.abs(sv.S[1]))}}=${tex(k)}` });
    const u1 = [sv.U[0][0], sv.U[1][0]];
    const u2 = [sv.U[0][1], sv.U[1][1]];
    extra = (
      <>
        <Circle M={[[1, 0], [0, 1]]} style={{ stroke: C.lineStrong, strokeDasharray: "3 3", strokeWidth: 1 }} />
        <Circle M={Aeff} style={{ stroke: C.fg, strokeWidth: 1.6 }} />
        <Arrow from={[0, 0]} to={[sv.S[0] * u1[0], sv.S[0] * u1[1]]} color="green" label="σ₁u₁" />
        <Arrow from={[0, 0]} to={[sv.S[1] * u2[0], sv.S[1] * u2[1]]} color="amber" label="σ₂u₂" />
      </>
    );
  }

  const showGrid = mode !== "condition";
  const showSquare = mode === "determinant" || mode === "transform" || mode === "singular" || mode === "inverse";

  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-4, 4]} y={[-4, 4]} equal originAxes aspect={0.75} maxH={520} ariaLabel={`Matrix transformation: ${mode}`}>
            <Grid M={[[1, 0], [0, 1]]} color={C.grid} />
            {showGrid && <Grid M={finalM} />}
            {showSquare && <Parallelogram M={finalM} fill={Math.abs(det2(finalM)) < 1e-9 ? C.a("red", 0.35) : det2(finalM) < 0 ? C.a("amber", 0.28) : C.a("pink", 0.22)} />}
            {extra}
            {mode !== "eigen" && mode !== "condition" && (
              <>
                <Arrow from={[0, 0]} to={Ae1} color="blue" label="Ae₁" />
                <Arrow from={[0, 0]} to={Ae2} color="pink" label="Ae₂" />
              </>
            )}
          </Plot>
          {mode !== "eigen" && mode !== "condition" && <Legend items={legend} />}
          {mode === "eigen" && <Legend items={[{ label: "eigen-direction 1", color: C.green, dashed: true }, { label: "eigen-direction 2", color: C.amber, dashed: true }, { label: "image of the unit circle", color: C.a("fg", 0.5) }]} />}
        </div>
      }
      controls={
        <>
          {lab && (
            <Select
              label="View"
              value={mode}
              onChange={setMode}
              options={[
                { value: "transform", label: "Transformation" },
                { value: "determinant", label: "Determinant (area)" },
                { value: "inverse", label: "Inverse" },
                { value: "singular", label: "Singular collapse" },
                { value: "eigen", label: "Eigenvectors" },
                { value: "condition", label: "Conditioning (SVD ellipse)" },
              ]}
            />
          )}
          {mode === "singular" ? (
            <ParameterSlider label="t (move column 2 toward column 1)" value={s} min={0} max={1} step={0.01} onChange={setS} />
          ) : (
            <MatrixInput label="Matrix A" value={A} onChange={setA} step={0.1} />
          )}
          {mode !== "singular" && (
            <Presets
              presets={Object.entries(PRESETS).map(([k, v]) => ({ label: k, value: v }))}
              onPick={(m) => { setA(m.map((r) => r.map((x) => Math.round(x * 1000) / 1000))); setT(1); }}
            />
          )}
          {mode === "inverse" && <Toggle label="Apply A⁻¹ after A" checked={applyInverse} onChange={setApplyInverse} />}
          {(mode === "transform" || mode === "determinant") && (
            <div className="space-y-2">
              <ParameterSlider label="Interpolate I → A" value={t} min={0} max={1} step={0.01} onChange={(v) => { setPlaying(false); setT(v); }} />
              <AnimationControls playing={playing} onPlay={() => { if (!playing && t >= 1) setT(0); setPlaying(!playing); }} onReset={() => { setPlaying(false); setT(0); }} />
            </div>
          )}
          {mode === "inverse" && !inv && <Warn>A is singular — no inverse exists.</Warn>}
        </>
      }
      readouts={<Readouts items={readouts} />}
      calc={<CalcSteps steps={steps} />}
      note={mode === "transform" || mode === "determinant" ? "The interpolation slider blends I and A entry-wise ((1−t)I + tA) to show motion; only t = 1 is the actual transformation A." : mode === "inverse" ? "Numerical note: real solvers avoid forming A⁻¹ and solve Ax = b with LU/QR instead — the inverse is shown here because it makes 'undoing' visible." : undefined}
    />
  );
}
