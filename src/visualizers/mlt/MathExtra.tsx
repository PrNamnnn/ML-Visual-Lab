"use client";
import { useMemo, useState } from "react";
import { Plot, Arrow, Dot, Handle, Polyline, HeatmapCanvas, C, useThemeRGB, type PlotCtx } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend, Warn } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Presets, MatrixInput, Select, NumberInput } from "@/components/controls";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { useSize } from "@/hooks/useSize";
import { rank, nullSpace, transpose, matMul, matVec, inverse, pinv, svd } from "@/lib/math/linalg";
import { mulberry32 } from "@/lib/math/random";
import { fmt, tex, texMatrix, clamp } from "@/lib/utilities/format";
import { Heat, BarList } from "@/visualizers/shared";
import { LineWB, ContourField, eigSym2, type V2 } from "./common";

const dot2 = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1];
const unit2 = (v: readonly number[]): V2 => { const n = Math.hypot(v[0], v[1]) || 1; return [v[0] / n, v[1] / n]; };
const span = (v: V2, L = 9): V2[] => [[-L * v[0], -L * v[1]], [L * v[0], L * v[1]]];

/* ------------------------------------------------------------------ the four fundamental subspaces of a 2 × 2 matrix */
export function SubspacesViz() {
  const [A, setA] = useState([[1, 2], [2, 4]]);
  const [x, setX] = useState<V2>([2, 0.5]);
  const r = rank(A);
  const N = nullSpace(A), Nt = nullSpace(transpose(A));
  const nullDir = N.length === 1 ? unit2(N[0]) : null;
  const leftNull = Nt.length === 1 ? unit2(Nt[0]) : null;
  const rowDir: V2 | null = r === 1 ? (nullDir ? [-nullDir[1], nullDir[0]] : null) : null;
  const colDir: V2 | null = r === 1 ? (leftNull ? [-leftNull[1], leftNull[0]] : null) : null;
  // x = x_row + x_null (orthogonal decomposition)
  const xn: V2 = r === 2 ? [0, 0] : r === 0 ? x : [dot2(x, nullDir!) * nullDir![0], dot2(x, nullDir!) * nullDir![1]];
  const xr: V2 = [x[0] - xn[0], x[1] - xn[1]];
  const Ax = matVec(A, x) as V2, Axr = matVec(A, xr) as V2;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <div className="mb-1 text-[12px] text-faint">Input space ℝ² — drag x</div>
              <Plot x={[-4, 4]} y={[-4, 4]} equal originAxes aspect={1} maxH={370} ariaLabel="row space and null space in the input plane">
                {rowDir && <Polyline pts={span(rowDir)} style={{ stroke: C.a("blue", 0.55), strokeWidth: 2 }} />}
                {nullDir && <Polyline pts={span(nullDir)} style={{ stroke: C.a("pink", 0.6), strokeWidth: 2 }} />}
                {r < 2 && <Polyline pts={[xr, x]} style={{ stroke: C.a("fg", 0.3), strokeWidth: 1, strokeDasharray: "3 3" }} />}
                {r < 2 && r > 0 && <Arrow from={[0, 0]} to={xr} color="blue" label="x_row" />}
                {r < 2 && <Arrow from={[0, 0]} to={xn} color="pink" label="x_null" />}
                <Arrow from={[0, 0]} to={x} color="fg" label="x" />
                <Handle p={x} color="fg" label="input vector x" onDrag={(q) => setX([clamp(q[0], -3.8, 3.8), clamp(q[1], -3.8, 3.8)])} />
              </Plot>
            </div>
            <div>
              <div className="mb-1 text-[12px] text-faint">Output space ℝ² — where A sends x</div>
              <Plot x={[-8, 8]} y={[-8, 8]} equal originAxes aspect={1} maxH={370} ariaLabel="column space and left null space in the output plane">
                {colDir && <Polyline pts={span(colDir, 20)} style={{ stroke: C.a("green", 0.6), strokeWidth: 2 }} />}
                {leftNull && <Polyline pts={span(leftNull, 20)} style={{ stroke: C.a("amber", 0.6), strokeWidth: 2 }} />}
                <Arrow from={[0, 0]} to={[A[0][0], A[1][0]]} color="muted" width={1.4} label="a₁" />
                <Arrow from={[0, 0]} to={[A[0][1], A[1][1]]} color="muted" width={1.4} label="a₂" />
                <Arrow from={[0, 0]} to={Ax} color="green" label="Ax" width={2.6} />
              </Plot>
            </div>
          </div>
          <Legend items={[{ label: r === 2 ? "row space: the whole plane" : "row space C(Aᵀ)", color: C.blue }, { label: r === 2 ? "null space: only 0" : "null space N(A)", color: C.pink }, { label: r === 2 ? "column space: the whole plane" : "column space C(A)", color: C.green }, { label: "left null space N(Aᵀ)", color: C.amber }]} />
        </div>
      }
      controls={
        <>
          <MatrixInput label="A" value={A} onChange={setA} step={0.5} />
          <Presets presets={[{ label: "Rank 1", value: [[1, 2], [2, 4]] }, { label: "Rank 2", value: [[2, 1], [0, 1.5]] }, { label: "Projection onto x-axis", value: [[1, 0], [0, 0]] }, { label: "Zero matrix", value: [[0, 0], [0, 0]] }]} onPick={setA} />
          <p className="text-[12.5px] leading-relaxed text-muted">Slide x along the pink line: only its null-space part changes, and Ax does not move. Everything A can “see” of x is its row-space part.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\operatorname{rank}(A)`, tex: true, value: r },
        { label: String.raw`\dim N(A)=n-\operatorname{rank}`, tex: true, value: 2 - r, color: C.pink },
        { label: String.raw`A\mathbf x`, tex: true, value: `(${fmt(Ax[0])}, ${fmt(Ax[1])})`, color: C.green },
        { label: String.raw`A\mathbf x_{\text{row}}`, tex: true, value: `(${fmt(Axr[0])}, ${fmt(Axr[1])})` },
        { label: String.raw`\|A\mathbf x_{\text{null}}\|`, tex: true, value: fmt(Math.hypot(...matVec(A, xn)), 6) },
        { label: String.raw`\mathbf x_{\text{row}}\cdot\mathbf x_{\text{null}}`, tex: true, value: fmt(dot2(xr, xn), 6) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`A=${texMatrix(A, 2)},\qquad\operatorname{rank}(A)=${r},\qquad\dim N(A)=2-${r}=${2 - r}`, note: "rank–nullity theorem" },
        ...(nullDir ? [{ tex: String.raw`N(A)=\operatorname{span}\Big\{${texMatrix([[N[0][0]], [N[0][1]]], 3)}\Big\}:\quad A${texMatrix([[N[0][0]], [N[0][1]]], 3)}=${texMatrix(matVec(A, N[0]).map((v) => [v]), 3)}`, note: "from the free variable of the reduced row echelon form" }] : []),
        { tex: String.raw`\mathbf x=\mathbf x_{\text{row}}+\mathbf x_{\text{null}}=${texMatrix([[xr[0]], [xr[1]]], 3)}+${texMatrix([[xn[0]], [xn[1]]], 3)},\qquad A\mathbf x=A\mathbf x_{\text{row}}=${texMatrix([[Ax[0]], [Ax[1]]], 3)}` },
      ]} />}
      note="Row space and null space are orthogonal complements in the input space; column space and left null space are orthogonal complements in the output space. A maps the row space one-to-one onto the column space and flattens the null space to zero."
    />
  );
}

/* ------------------------------------------------------------------ orthogonal projection onto a plane in ℝ³ */
type V3 = [number, number, number];
export function SubspaceProjectionViz() {
  const [A, setA] = useState([[1, 0], [0, 1], [0.5, 0.5]]);
  const [b, setB] = useState<V3>([1, 2, 3]);
  const [yaw, setYaw] = useState(35);
  const [pitch, setPitch] = useState(22);
  const { ref, width } = useSize<HTMLDivElement>(0.7);
  const H = Math.max(280, Math.min(460, Math.round(width * 0.68)));
  const At = transpose(A), G = matMul(At, A), Gi = inverse(G);
  const P = Gi ? matMul(matMul(A, Gi), At) : null;
  const p = (P ? matVec(P, b) : [0, 0, 0]) as V3;
  const e: V3 = [b[0] - p[0], b[1] - p[1], b[2] - p[2]];
  const xhat = Gi ? matVec(matMul(Gi, At), b) : [NaN, NaN];
  const Ate = matVec(At, e);
  const P2 = P ? matMul(P, P) : null;
  const idem = P && P2 ? Math.max(...P.flatMap((row, i) => row.map((v, j) => Math.abs(v - P2[i][j])))) : NaN;
  // orthographic camera
  const cy = Math.cos((yaw * Math.PI) / 180), sy = Math.sin((yaw * Math.PI) / 180), cp = Math.cos((pitch * Math.PI) / 180), sp = Math.sin((pitch * Math.PI) / 180);
  const S = Math.min(width, H * 1.4) / 9;
  const pr = (v: readonly number[]): [number, number] => { const x1 = cy * v[0] + sy * v[1], y1 = -sy * v[0] + cy * v[1]; return [width / 2 + S * x1, H / 2 - S * (cp * v[2] + sp * y1) + 10]; };
  const a1 = A.map((r) => r[0]) as V3, a2 = A.map((r) => r[1]) as V3;
  const quad = [[-1.6, -1.6], [1.6, -1.6], [1.6, 1.6], [-1.6, 1.6]].map(([s, t]) => pr([s * a1[0] + t * a2[0], s * a1[1] + t * a2[1], s * a1[2] + t * a2[2]]));
  const seg = (u: readonly number[], v: readonly number[], color: string, w = 2, dash?: string) => { const [x1, y1] = pr(u), [x2, y2] = pr(v); return <line x1={x1} y1={y1} x2={x2} y2={y2} style={{ stroke: color, strokeWidth: w, strokeDasharray: dash }} />; };
  const tip = (v: readonly number[], label: string, color: string) => { const [x, y] = pr(v); return <g><circle cx={x} cy={y} r={4} style={{ fill: color }} /><text x={x + 7} y={y - 6} style={{ fill: color, fontSize: 12, fontWeight: 500 }}>{label}</text></g>; };
  return (
    <LabShell
      wideControls
      viz={
        <div ref={ref} className="space-y-2">
          <svg width={width} height={H} role="img" aria-label="projection of the vector b onto the plane spanned by the columns of A">
            {([[3.2, 0, 0], [0, 3.2, 0], [0, 0, 3.2]] as V3[]).map((ax, i) => <g key={i}>{seg([0, 0, 0], ax, C.axis, 1)}<text x={pr(ax)[0] + 4} y={pr(ax)[1] + 4} style={{ fill: C.faint, fontSize: 11 }}>{["x", "y", "z"][i]}</text></g>)}
            {P && <polygon points={quad.map((q) => q.join(",")).join(" ")} style={{ fill: C.a("blue", 0.13), stroke: C.a("blue", 0.5), strokeWidth: 1 }} />}
            {seg([0, 0, 0], a1, C.blue, 2)}{tip(a1, "a₁", C.blue)}
            {seg([0, 0, 0], a2, C.blue, 2)}{tip(a2, "a₂", C.blue)}
            {P && seg(p, b, C.pink, 2, "5 4")}
            {P && <>{seg([0, 0, 0], p, C.green, 2.6)}{tip(p, "p = Pb", C.green)}</>}
            {seg([0, 0, 0], b, C.amber, 2.6)}{tip(b, "b", C.amber)}
          </svg>
          <Legend items={[{ label: "column space of A (a plane through 0)", color: C.blue }, { label: "b", color: C.amber }, { label: "projection p", color: C.green }, { label: "error e = b − p, perpendicular to the plane", color: C.pink, dashed: true }]} />
          {!P && <Warn>The two columns are parallel, so AᵀA is singular and the column space is only a line. The formula below needs independent columns.</Warn>}
        </div>
      }
      controls={
        <>
          <MatrixInput label="A (3 × 2): the columns span the plane" value={A} onChange={setA} step={0.5} />
          <div>
            <div className="mb-1 text-[12.5px] text-muted">b</div>
            <div className="grid grid-cols-3 gap-1">{b.map((v, i) => <NumberInput key={i} ariaLabel={`component ${i + 1} of b`} value={v} step={0.5} onChange={(nv) => setB(b.map((q, j) => (j === i ? nv : q)) as V3)} />)}</div>
          </div>
          <div className="grid grid-cols-2 gap-x-3">
            <ParameterSlider label="rotate view" value={yaw} min={-180} max={180} step={1} onChange={setYaw} format={(v) => `${v}°`} />
            <ParameterSlider label="tilt view" value={pitch} min={0} max={85} step={1} onChange={setPitch} format={(v) => `${v}°`} />
          </div>
          {P && <MatrixView M={P} label={<span>P = A(AᵀA)⁻¹Aᵀ</span>} digits={3} cell={54} />}
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\mathbf p=P\mathbf b`, tex: true, value: P ? `(${p.map((v) => fmt(v, 3)).join(", ")})` : "—", color: C.green },
        { label: String.raw`\|\mathbf e\|=\|\mathbf b-\mathbf p\|`, tex: true, value: P ? fmt(Math.hypot(...e), 4) : "—", color: C.pink },
        { label: String.raw`A^\top\mathbf e`, tex: true, value: P ? `(${Ate.map((v) => fmt(v, 6)).join(", ")})` : "—" },
        { label: String.raw`\max|P^2-P|`, tex: true, value: fmt(idem, 6) },
        { label: String.raw`\hat{\mathbf x}`, tex: true, value: P ? `(${xhat.map((v) => fmt(v, 3)).join(", ")})` : "—" },
      ]} />}
      calc={P ? <CalcSteps steps={[
        { tex: String.raw`A^\top A=${texMatrix(G, 2)},\qquad A^\top\mathbf b=${texMatrix(matVec(At, b).map((v) => [v]), 2)}` },
        { tex: String.raw`\hat{\mathbf x}=(A^\top A)^{-1}A^\top\mathbf b=${texMatrix(xhat.map((v) => [v]), 3)}`, note: "coefficients of the projection in the basis a₁, a₂" },
        { tex: String.raw`\mathbf p=A\hat{\mathbf x}=${texMatrix(p.map((v) => [v]), 3)},\qquad\mathbf e=\mathbf b-\mathbf p=${texMatrix(e.map((v) => [v]), 3)}` },
        { tex: String.raw`A^\top\mathbf e=${texMatrix(Ate.map((v) => [v]), 5)}`, note: "the error is orthogonal to both columns — these are the normal equations" },
      ]} /> : undefined}
      note="Projecting twice changes nothing (P² = P) and P is symmetric: those two properties characterise orthogonal projection matrices. Least-squares regression is this picture with b = y and the columns of A as the features."
    />
  );
}

/* ------------------------------------------------------------------ spectral theorem for a symmetric 2 × 2 matrix */
export function SpectralDecompViz() {
  const [a, setA] = useState(2);
  const [b, setB] = useState(1);
  const [d, setD] = useState(1);
  const [x, setX] = useState<V2>([1, 1.4]);
  const M = [[a, b], [b, d]];
  const { values, vectors } = eigSym2(M);
  const [q1, q2] = vectors;
  const c1 = dot2(x, q1), c2 = dot2(x, q2);
  const Ax = matVec(M, x) as V2;
  const part1: V2 = [values[0] * c1 * q1[0], values[0] * c1 * q1[1]];
  const R1 = [[q1[0] * q1[0], q1[0] * q1[1]], [q1[1] * q1[0], q1[1] * q1[1]]], R2 = [[q2[0] * q2[0], q2[0] * q2[1]], [q2[1] * q2[0], q2[1] * q2[1]]];
  const ell: V2[] = Array.from({ length: 121 }, (_, i) => { const t = (i * Math.PI) / 60; return matVec(M, [Math.cos(t), Math.sin(t)]) as V2; });
  const circ: V2[] = Array.from({ length: 121 }, (_, i) => [Math.cos((i * Math.PI) / 60), Math.sin((i * Math.PI) / 60)]);
  const kind = values[1] > 1e-9 ? "positive definite" : values[1] > -1e-9 ? (values[0] > 1e-9 ? "positive semidefinite" : "zero") : values[0] < -1e-9 ? "negative definite" : "indefinite";
  const R = Math.max(3.5, Math.abs(values[0]) * 1.25, Math.abs(values[1]) * 1.25, Math.hypot(...Ax) * 1.1);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <Plot x={[-R, R]} y={[-R * 0.8, R * 0.8]} equal originAxes aspect={0.72} maxH={450} ariaLabel="symmetric matrix acting along its orthogonal eigenvectors">
            <Polyline pts={circ} closed style={{ stroke: C.a("fg", 0.3), strokeWidth: 1, strokeDasharray: "4 3" }} />
            <Polyline pts={ell} closed style={{ stroke: C.a("fg", 0.7), strokeWidth: 1.4 }} />
            <Polyline pts={span(q1, 30)} style={{ stroke: C.a("green", 0.3), strokeWidth: 1 }} />
            <Polyline pts={span(q2, 30)} style={{ stroke: C.a("amber", 0.3), strokeWidth: 1 }} />
            <Arrow from={[0, 0]} to={[values[0] * q1[0], values[0] * q1[1]]} color="green" label="λ₁q₁" />
            <Arrow from={[0, 0]} to={[values[1] * q2[0], values[1] * q2[1]]} color="amber" label="λ₂q₂" />
            <Arrow from={[0, 0]} to={part1} color="green" width={1.4} dashed />
            <Arrow from={part1} to={Ax} color="amber" width={1.4} dashed />
            <Arrow from={[0, 0]} to={Ax} color="pink" label="Ax" />
            <Arrow from={[0, 0]} to={x} color="fg" label="x" />
            <Handle p={x} color="fg" label="vector x" onDrag={(p) => setX([clamp(p[0], -R, R), clamp(p[1], -R * 0.8, R * 0.8)])} />
          </Plot>
          <Legend items={[{ label: "unit circle", color: C.a("fg", 0.4), dashed: true }, { label: "its image under A", color: C.a("fg", 0.8) }, { label: "eigen-direction q₁", color: C.green }, { label: "eigen-direction q₂ (perpendicular)", color: C.amber }, { label: "Ax built from the two pieces", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="a₁₁" value={a} min={-3} max={3} step={0.1} onChange={setA} />
          <ParameterSlider label="a₁₂ = a₂₁" value={b} min={-3} max={3} step={0.1} onChange={setB} />
          <ParameterSlider label="a₂₂" value={d} min={-3} max={3} step={0.1} onChange={setD} />
          <Presets presets={[{ label: "Covariance-like", value: [2, 1, 1] }, { label: "Indefinite", value: [1, 2, -1] }, { label: "Singular", value: [1, 1, 1] }, { label: "Diagonal", value: [2.5, 0, 0.8] }]} onPick={([p, q, r]) => { setA(p); setB(q); setD(r); }} />
          <div className="flex flex-wrap items-start gap-4">
            <MatrixView M={R1} label={<span>q₁q₁ᵀ</span>} digits={3} cell={50} />
            <MatrixView M={R2} label={<span>q₂q₂ᵀ</span>} digits={3} cell={50} />
          </div>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\lambda_1`, tex: true, value: fmt(values[0], 4), color: C.green },
        { label: String.raw`\lambda_2`, tex: true, value: fmt(values[1], 4), color: C.amber },
        { label: String.raw`\mathbf q_1^\top\mathbf q_2`, tex: true, value: fmt(dot2(q1, q2), 6) },
        { label: String.raw`\lambda_1+\lambda_2=\operatorname{tr}A`, tex: true, value: `${fmt(values[0] + values[1], 4)} = ${fmt(a + d, 4)}` },
        { label: String.raw`\lambda_1\lambda_2=\det A`, tex: true, value: `${fmt(values[0] * values[1], 4)} = ${fmt(a * d - b * b, 4)}` },
        { label: String.raw`\text{type}`, tex: true, value: kind },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`A=${texMatrix(M, 2)}=Q\Lambda Q^\top,\qquad Q=${texMatrix([[q1[0], q2[0]], [q1[1], q2[1]]], 3)},\quad\Lambda=${texMatrix([[values[0], 0], [0, values[1]]], 3)}` },
        { tex: String.raw`A=\lambda_1\mathbf q_1\mathbf q_1^\top+\lambda_2\mathbf q_2\mathbf q_2^\top=${tex(values[0], 3)}${texMatrix(R1, 3)}+${tex(values[1], 3)}${texMatrix(R2, 3)}`, note: "a weighted sum of projections onto perpendicular lines" },
        { tex: String.raw`A\mathbf x=\lambda_1(\mathbf q_1^\top\mathbf x)\mathbf q_1+\lambda_2(\mathbf q_2^\top\mathbf x)\mathbf q_2=${tex(values[0], 3)}(${tex(c1, 3)})\mathbf q_1+${tex(values[1], 3)}(${tex(c2, 3)})\mathbf q_2=${texMatrix([[Ax[0]], [Ax[1]]], 3)}`, note: "split x along the eigenvectors, scale each part, add" },
      ]} />}
      note="Because the matrix is symmetric its eigenvalues are real and its eigenvectors are perpendicular for every setting of the sliders. That is the spectral theorem, and it is what makes covariance matrices, kernel matrices and Hessians so easy to analyse."
    />
  );
}

/* ------------------------------------------------------------------ pseudoinverse: minimum norm and least squares */
export function PinvViz() {
  const [kind, setKind] = useState<"under" | "over">("under");
  const [a, setA] = useState<V2>([1, 2]);
  const [bb, setBb] = useState(4);
  const [t, setT] = useState(1.2);
  const [A3, setA3] = useState([[1, 1], [1, -1], [0.5, 2]]);
  const [b3, setB3] = useState([3, 0.5, 3.5]);
  if (kind === "under") {
    const A = [[a[0], a[1]]];
    const Ap = pinv(A);
    const xs: V2 = [Ap[0][0] * bb, Ap[1][0] * bb];
    const n2 = a[0] * a[0] + a[1] * a[1];
    const dir: V2 = unit2([-a[1], a[0]]);
    const other: V2 = [xs[0] + t * dir[0], xs[1] + t * dir[1]];
    return (
      <LabShell
        viz={
          <div className="space-y-2">
            <Plot x={[-4, 5]} y={[-3.5, 4.5]} equal originAxes aspect={0.78} maxH={440} ariaLabel="solution line of one equation and its minimum norm point">
              {n2 > 1e-9 && <LineWB w={a} b={-bb} style={{ stroke: C.blue, strokeWidth: 2 }} />}
              <Polyline pts={Array.from({ length: 73 }, (_, i) => [Math.hypot(...xs) * Math.cos((i * Math.PI) / 36), Math.hypot(...xs) * Math.sin((i * Math.PI) / 36)] as V2)} style={{ stroke: C.a("green", 0.45), strokeWidth: 1, strokeDasharray: "4 3" }} />
              <Arrow from={[0, 0]} to={other} color="muted" label="another solution" width={1.6} />
              <Arrow from={[0, 0]} to={xs} color="green" label="x⁺ = A⁺b" />
              <Dot p={other} r={4.5} color={C.fg} stroke={C.bg} />
            </Plot>
            <Legend items={[{ label: "all solutions of a₁x₁ + a₂x₂ = b", color: C.blue }, { label: "minimum-norm solution", color: C.green }, { label: "circle of radius ‖x⁺‖ touching the line", color: C.a("green", 0.6), dashed: true }]} />
          </div>
        }
        controls={
          <>
            <Segmented label="System" value={kind} onChange={setKind} options={[{ value: "under", label: "Fewer equations" }, { value: "over", label: "More equations" }]} />
            <ParameterSlider label="a₁" value={a[0]} min={-3} max={3} step={0.1} onChange={(v) => setA([v, a[1]])} />
            <ParameterSlider label="a₂" value={a[1]} min={-3} max={3} step={0.1} onChange={(v) => setA([a[0], v])} />
            <ParameterSlider label="b" value={bb} min={-6} max={6} step={0.1} onChange={setBb} />
            <ParameterSlider label="slide the other solution along the line" value={t} min={-4} max={4} step={0.05} onChange={setT} />
          </>
        }
        readouts={<Readouts items={[
          { label: String.raw`\mathbf x^+`, tex: true, value: `(${fmt(xs[0], 4)}, ${fmt(xs[1], 4)})`, color: C.green },
          { label: String.raw`\|\mathbf x^+\|`, tex: true, value: fmt(Math.hypot(...xs), 4), color: C.green },
          { label: String.raw`\|\text{other solution}\|`, tex: true, value: fmt(Math.hypot(...other), 4) },
          { label: String.raw`A\,(\text{other})-b`, tex: true, value: fmt(dot2(a, other) - bb, 6) },
          { label: String.raw`\operatorname{rank}(A)`, tex: true, value: rank(A) },
        ]} />}
        calc={<CalcSteps steps={[
          { tex: String.raw`A=\begin{bmatrix}${tex(a[0], 2)}&${tex(a[1], 2)}\end{bmatrix},\qquad A^+=A^\top(AA^\top)^{-1}=\frac{1}{${tex(n2, 3)}}${texMatrix([[a[0]], [a[1]]], 2)}=${texMatrix(Ap, 4)}`, note: "full row rank; the numbers come from the SVD" },
          { tex: String.raw`\mathbf x^+=A^+b=${texMatrix([[xs[0]], [xs[1]]], 4)},\qquad A\mathbf x^+=${tex(dot2(a, xs), 4)}=b` },
          { tex: String.raw`\|\mathbf x^++t\,\mathbf n\|^2=\|\mathbf x^+\|^2+t^2=${tex(xs[0] ** 2 + xs[1] ** 2, 4)}+${tex(t * t, 4)}`, note: "n spans the null space and is perpendicular to x⁺, so every other solution is longer" },
        ]} />}
        note="With more unknowns than equations there are infinitely many exact solutions. The pseudoinverse returns the one with no null-space component — the shortest. Gradient descent started from zero on an over-parameterised linear model converges to this same solution."
      />
    );
  }
  const Ap = pinv(A3);
  const xs = matVec(Ap, b3) as V2;
  const resid = A3.map((r, i) => dot2(r, xs) - b3[i]);
  const cols = [C.blue, C.pink, C.amber];
  const { S } = svd(A3);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 6]} y={[-3.5, 5]} equal originAxes aspect={0.78} maxH={440} ariaLabel="three inconsistent equations and their least squares point">
            {A3.map((r, i) => <LineWB key={i} w={r as V2} b={-b3[i]} style={{ stroke: cols[i], strokeWidth: 1.8 }} />)}
            {A3.map((r, i) => { const n2 = dot2(r, r) || 1; const foot: V2 = [xs[0] - (resid[i] / n2) * r[0], xs[1] - (resid[i] / n2) * r[1]]; return <Polyline key={`r${i}`} pts={[xs, foot]} style={{ stroke: C.a("fg", 0.6), strokeWidth: 1.2, strokeDasharray: "3 3" }} />; })}
            <Dot p={xs} r={6} color={C.green} stroke={C.bg} />
          </Plot>
          <Legend items={[...A3.map((_, i) => ({ label: `equation ${i + 1}`, color: cols[i] })), { label: "x⁺ = A⁺b (least squares)", color: C.green }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="System" value={kind} onChange={setKind} options={[{ value: "under", label: "Fewer equations" }, { value: "over", label: "More equations" }]} />
          <MatrixInput label="A (3 equations, 2 unknowns)" value={A3} onChange={setA3} step={0.5} />
          <div>
            <div className="mb-1 text-[12.5px] text-muted">b</div>
            <div className="grid grid-cols-3 gap-1">{b3.map((v, i) => <NumberInput key={i} ariaLabel={`right-hand side ${i + 1}`} value={v} step={0.5} onChange={(nv) => setB3(b3.map((q, j) => (j === i ? nv : q)))} />)}</div>
          </div>
          <MatrixView M={Ap} label={<span>A⁺ (2 × 3)</span>} digits={3} cell={54} />
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\mathbf x^+`, tex: true, value: `(${fmt(xs[0], 4)}, ${fmt(xs[1], 4)})`, color: C.green },
        { label: String.raw`\|A\mathbf x^+-\mathbf b\|^2`, tex: true, value: fmt(resid.reduce((s, r) => s + r * r, 0), 5) },
        { label: String.raw`A^\top(A\mathbf x^+-\mathbf b)`, tex: true, value: `(${matVec(transpose(A3), resid).map((v) => fmt(v, 6)).join(", ")})` },
        { label: String.raw`\sigma_1,\ \sigma_2`, tex: true, value: S.map((s) => fmt(s, 4)).join(", ") },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`A^+=V\Sigma^+U^\top=(A^\top A)^{-1}A^\top=${texMatrix(Ap, 3)}`, note: "full column rank" },
        { tex: String.raw`\mathbf x^+=A^+\mathbf b=${texMatrix([[xs[0]], [xs[1]]], 4)},\qquad A\mathbf x^+-\mathbf b=${texMatrix(resid.map((v) => [v]), 4)}` },
        { tex: String.raw`\Sigma^+=\operatorname{diag}(1/\sigma_1,\,1/\sigma_2)=\operatorname{diag}(${tex(1 / S[0], 4)},\ ${tex(S[1] > 1e-10 ? 1 / S[1] : 0, 4)})`, note: "a tiny singular value makes A⁺ huge: the ill-conditioned case" },
      ]} />}
      note="Three lines in the plane usually do not meet in one point. The pseudoinverse picks the point whose equation residuals have the smallest sum of squares; the dashed segments show how far it is from each line."
    />
  );
}

/* ------------------------------------------------------------------ gradients of vector functions, checked numerically */
type FKind = "linear" | "quadratic" | "lsq";
function Contours({ ctx, f, lo, hi, deps }: { ctx: PlotCtx; f: (a: number, b: number) => number; lo: number; hi: number; deps: unknown[] }) {
  return <ContourField ctx={ctx} f={f} signed level={(v) => (12 * v) / Math.max(1e-9, Math.abs(lo), Math.abs(hi))} deps={[...deps, lo, hi]} />;
}

export function MatrixCalculusViz() {
  const [kind, setKind] = useState<FKind>("quadratic");
  const [A, setA] = useState([[2, 1], [0, 1]]);
  const [bv, setBv] = useState<V2>([1, -1]);
  const [x, setX] = useState<V2>([1, 0.5]);
  const f = (p: readonly number[]) => {
    if (kind === "linear") return dot2(bv, p);
    if (kind === "quadratic") return dot2(p, matVec(A, p as number[]));
    const r = matVec(A, p as number[]); return (r[0] - bv[0]) ** 2 + (r[1] - bv[1]) ** 2;
  };
  const At = transpose(A);
  const grad: V2 = kind === "linear" ? bv : kind === "quadratic" ? (matVec([[2 * A[0][0], A[0][1] + A[1][0]], [A[0][1] + A[1][0], 2 * A[1][1]]], x) as V2) : (matVec(At, matVec(A, x).map((v, i) => 2 * (v - bv[i]))) as V2);
  const h = 1e-5;
  const num: V2 = [(f([x[0] + h, x[1]]) - f([x[0] - h, x[1]])) / (2 * h), (f([x[0], x[1] + h]) - f([x[0], x[1] - h])) / (2 * h)];
  const Hs = kind === "linear" ? [[0, 0], [0, 0]] : kind === "quadratic" ? [[2 * A[0][0], A[0][1] + A[1][0]], [A[0][1] + A[1][0], 2 * A[1][1]]] : matMul(At, A).map((r) => r.map((v) => 2 * v));
  const samples = useMemo(() => { const v: number[] = []; for (let i = -3; i <= 3; i += 0.5) for (let j = -3; j <= 3; j += 0.5) v.push(f([i, j])); return [Math.min(...v), Math.max(...v)]; }, [kind, A, bv]); // eslint-disable-line react-hooks/exhaustive-deps
  const gn = Math.hypot(...grad) || 1;
  const formula = kind === "linear" ? String.raw`f(\mathbf x)=\mathbf a^\top\mathbf x\ \Rightarrow\ \nabla f=\mathbf a` : kind === "quadratic" ? String.raw`f(\mathbf x)=\mathbf x^\top A\mathbf x\ \Rightarrow\ \nabla f=(A+A^\top)\mathbf x` : String.raw`f(\mathbf x)=\|A\mathbf x-\mathbf b\|^2\ \Rightarrow\ \nabla f=2A^\top(A\mathbf x-\mathbf b)`;
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-3, 3]} equal originAxes aspect={0.86} maxH={450} xLabel="x₁" yLabel="x₂" ariaLabel="contours of the function with its gradient at x" underlay={(ctx) => <Contours ctx={ctx} f={(p, q) => f([p, q])} lo={samples[0]} hi={samples[1]} deps={[kind, A, bv]} />}>
            <Arrow from={x} to={[x[0] + (0.9 * grad[0]) / gn, x[1] + (0.9 * grad[1]) / gn]} color="amber" label="∇f" />
            <Handle p={x} color="fg" label="point x" onDrag={(q) => setX([clamp(q[0], -2.9, 2.9), clamp(q[1], -2.9, 2.9)])} />
          </Plot>
          <Legend items={[{ label: "level sets of f (blue: positive, pink: negative)", color: C.blue }, { label: "gradient direction at x: perpendicular to the level set, pointing uphill", color: C.amber }]} />
        </div>
      }
      controls={
        <>
          <Select label="Function" value={kind} onChange={setKind} options={[{ value: "linear", label: "Linear: aᵀx" }, { value: "quadratic", label: "Quadratic form: xᵀAx" }, { value: "lsq", label: "Least squares: ‖Ax − b‖²" }]} />
          {kind !== "linear" && <MatrixInput label="A" value={A} onChange={setA} step={0.5} />}
          <div>
            <div className="mb-1 text-[12.5px] text-muted">{kind === "linear" ? "a" : kind === "lsq" ? "b" : "(not used)"}</div>
            <div className="grid grid-cols-2 gap-1">{bv.map((v, i) => <NumberInput key={i} ariaLabel={`vector component ${i + 1}`} value={v} step={0.5} onChange={(nv) => setBv(bv.map((q, j) => (j === i ? nv : q)) as V2)} />)}</div>
          </div>
          {kind !== "linear" && <Presets presets={[{ label: "Symmetric", value: [[2, 1], [1, 1]] }, { label: "Not symmetric", value: [[2, 1], [0, 1]] }, { label: "Saddle", value: [[1, 0], [0, -1]] }]} onPick={setA} />}
          <MatrixView M={Hs} label={<span>Hessian ∇²f (constant)</span>} digits={2} cell={54} />
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`f(\mathbf x)`, tex: true, value: fmt(f(x), 4) },
        { label: String.raw`\nabla f\ \text{(formula)}`, tex: true, value: `(${fmt(grad[0], 4)}, ${fmt(grad[1], 4)})`, color: C.amber },
        { label: String.raw`\nabla f\ \text{(finite differences)}`, tex: true, value: `(${fmt(num[0], 4)}, ${fmt(num[1], 4)})` },
        { label: String.raw`\text{difference}`, tex: true, value: fmt(Math.hypot(grad[0] - num[0], grad[1] - num[1]), 8) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: formula },
        { tex: kind === "linear" ? String.raw`\nabla f=${texMatrix([[bv[0]], [bv[1]]], 2)}` : kind === "quadratic" ? String.raw`(A+A^\top)\mathbf x=${texMatrix(Hs, 2)}${texMatrix([[x[0]], [x[1]]], 2)}=${texMatrix([[grad[0]], [grad[1]]], 3)}` : String.raw`2A^\top(A\mathbf x-\mathbf b)=2${texMatrix(At, 2)}${texMatrix(matVec(A, x).map((v, i) => [v - bv[i]]), 3)}=${texMatrix([[grad[0]], [grad[1]]], 3)}` },
        { tex: String.raw`\frac{\partial f}{\partial x_1}\approx\frac{f(x_1+h,x_2)-f(x_1-h,x_2)}{2h}=${tex(num[0], 4)},\qquad\frac{\partial f}{\partial x_2}\approx${tex(num[1], 4)}`, note: "independent numerical check with h = 10⁻⁵" },
      ]} />}
      note={kind === "quadratic" ? "For a symmetric A the rule reduces to the familiar ∇(xᵀAx) = 2Ax. Try the non-symmetric preset: the gradient uses A + Aᵀ, because only the symmetric part of A affects xᵀAx." : kind === "lsq" ? "Setting this gradient to zero gives the normal equations AᵀAx = Aᵀb. The Hessian 2AᵀA is positive semidefinite, so the function is convex." : "A linear function has the same gradient everywhere, and its level sets are parallel lines perpendicular to a."}
    />
  );
}

/* ------------------------------------------------------------------ joint, marginal and conditional distributions */
export function JointMarginalViz() {
  const [dep, setDep] = useState(0.6);
  const [sel, setSel] = useState<[number, number]>([1, 2]);
  const px = [0.1, 0.2, 0.3, 0.4], py = [0.2, 0.5, 0.3];
  const J = useMemo(() => {
    // blend of the independent table pₓ(x)p_y(y) with a table that concentrates mass where x/3 ≈ y/2
    const dependent = py.map((_, j) => px.map((__, i) => Math.exp(-3 * (i / 3 - j / 2) ** 2)));
    const s = dependent.flat().reduce((a, b) => a + b, 0);
    return py.map((q, j) => px.map((p, i) => (1 - dep) * p * q + (dep * dependent[j][i]) / s));
  }, [dep]); // eslint-disable-line react-hooks/exhaustive-deps
  const mX = px.map((_, i) => J.reduce((s, r) => s + r[i], 0));
  const mY = J.map((r) => r.reduce((a, b) => a + b, 0));
  const [j, i] = sel;
  const condX = J[j].map((v) => v / mY[j]);
  const maxGap = Math.max(...J.flatMap((r, b) => r.map((v, a) => Math.abs(v - mX[a] * mY[b]))));
  const mi = J.reduce((s, r, b) => s + r.reduce((t, v, a) => t + (v > 0 ? v * Math.log2(v / (mX[a] * mY[b])) : 0), 0), 0);
  const ex = mX.reduce((s, p, a) => s + p * (a + 1), 0), ey = mY.reduce((s, p, b) => s + p * (b + 1), 0);
  const cov = J.reduce((s, r, b) => s + r.reduce((t, v, a) => t + v * (a + 1 - ex) * (b + 1 - ey), 0), 0);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-4">
          <div className="grid gap-5 md:grid-cols-[auto_1fr]">
            <div>
              <div className="label mb-2">Joint table P(X = x, Y = y) — click a cell</div>
              <div className="inline-grid gap-[3px]" style={{ gridTemplateColumns: "auto repeat(4, 58px) 64px" }}>
                <span />
                {px.map((_, a) => <span key={a} className="text-center text-[11px] text-faint">x = {a + 1}</span>)}
                <span className="text-center text-[11px] text-faint">P(Y = y)</span>
                {J.map((row, b) => (
                  <div key={b} className="contents">
                    <span className="flex items-center pr-2 text-[11px] text-faint">y = {b + 1}</span>
                    {row.map((v, a) => (
                      <button key={a} onClick={() => setSel([b, a])} aria-label={`select x ${a + 1}, y ${b + 1}`} className={`num h-9 rounded-[3px] text-[12px] text-fg ${b === j && a === i ? "ring-2 ring-fg" : b === j ? "ring-1 ring-amber" : ""}`} style={{ background: `rgb(var(--blue) / ${0.08 + 2.4 * v})` }}>{fmt(v, 3)}</button>
                    ))}
                    <span className="num flex items-center justify-center rounded-[3px] text-[12px]" style={{ background: `rgb(var(--green) / ${0.1 + 0.7 * mY[b]})` }}>{fmt(mY[b], 3)}</span>
                  </div>
                ))}
                <span className="flex items-center pr-2 text-[11px] text-faint">P(X = x)</span>
                {mX.map((v, a) => <span key={a} className="num flex h-9 items-center justify-center rounded-[3px] text-[12px]" style={{ background: `rgb(var(--pink) / ${0.1 + 0.9 * v})` }}>{fmt(v, 3)}</span>)}
                <span className="num flex items-center justify-center text-[12px] text-muted">{fmt(mX.reduce((a, b) => a + b, 0), 3)}</span>
              </div>
            </div>
            <div className="space-y-4">
              <div>
                <div className="label mb-2">Marginal P(X = x): sum each column</div>
                <BarList max={1} digits={3} items={mX.map((v, a) => ({ label: `x = ${a + 1}`, value: v, color: C.pink }))} />
              </div>
              <div>
                <div className="label mb-2">Conditional P(X = x | Y = {j + 1}): the selected row, renormalised</div>
                <BarList max={1} digits={3} items={condX.map((v, a) => ({ label: `x = ${a + 1}`, value: v, color: C.amber }))} />
              </div>
            </div>
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="dependence between X and Y" value={dep} min={0} max={1} step={0.01} onChange={setDep} />
          <p className="text-[12.5px] leading-relaxed text-muted">At 0 the table is the outer product of two fixed marginals, so every row has the same shape and conditioning on Y tells you nothing about X. Raising the slider moves mass towards the diagonal: the conditional (amber) pulls away from the marginal (pink).</p>
          <div>
            <div className="label mb-1.5">Product of the marginals P(x)P(y)</div>
            <Heat M={mY.map((q) => mX.map((p) => p * q))} digits={3} cell={50} />
          </div>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`P(X=${i + 1},Y=${j + 1})`, tex: true, value: fmt(J[j][i], 4) },
        { label: String.raw`P(X=${i + 1})\,P(Y=${j + 1})`, tex: true, value: fmt(mX[i] * mY[j], 4) },
        { label: String.raw`P(X=${i + 1}\mid Y=${j + 1})`, tex: true, value: fmt(condX[i], 4), color: C.amber },
        { label: String.raw`\max|P(x,y)-P(x)P(y)|`, tex: true, value: fmt(maxGap, 5) },
        { label: String.raw`I(X;Y)\ \text{bits}`, tex: true, value: fmt(mi, 5) },
        { label: String.raw`\operatorname{Cov}(X,Y)`, tex: true, value: fmt(cov, 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`P(Y=${j + 1})=\sum_xP(x,${j + 1})=${J[j].map((v) => tex(v, 3)).join("+")}=${tex(mY[j], 4)}`, note: "marginalise: sum the joint over the variable you do not care about" },
        { tex: String.raw`P(X=${i + 1}\mid Y=${j + 1})=\frac{P(X=${i + 1},Y=${j + 1})}{P(Y=${j + 1})}=\frac{${tex(J[j][i], 4)}}{${tex(mY[j], 4)}}=${tex(condX[i], 4)}` },
        { tex: String.raw`\text{independent}\iff P(x,y)=P(x)P(y)\ \forall x,y:\quad\max\text{ gap}=${tex(maxGap, 5)}`, note: maxGap < 1e-9 ? "independent" : "dependent" },
      ]} />}
      note="The joint table contains everything: both marginals and every conditional can be computed from it, but the marginals alone cannot rebuild the joint. Generative classifiers (MLT Week 8) model the joint P(x, y) and obtain P(y | x) by exactly this row normalisation."
    />
  );
}

/* ------------------------------------------------------------------ convex sets */
type Shape = "disk" | "polygon" | "halfplanes" | "crescent" | "annulus" | "union";
const INSIDE: Record<Shape, (x: number, y: number) => boolean> = {
  disk: (x, y) => x * x + y * y <= 4,
  polygon: (x, y) => Math.abs(x) + Math.abs(y) <= 2.4 && y <= 1.4,
  halfplanes: (x, y) => x + y <= 2 && x - 2 * y <= 2.5 && -x <= 2.2 && y <= 1.8 && y >= -1.9,
  crescent: (x, y) => x * x + y * y <= 4.4 && (x - 1.1) ** 2 + y * y > 1.9,
  annulus: (x, y) => x * x + y * y <= 4.4 && x * x + y * y >= 1.2,
  union: (x, y) => (x + 1.1) ** 2 + y * y <= 1.7 || (x - 1.1) ** 2 + y * y <= 1.7,
};
export function ConvexSetsViz() {
  const [shape, setShape] = useState<Shape>("crescent");
  const [a, setA] = useState<V2>([0.2, 1.7]);
  const [b, setB] = useState<V2>([0.2, -1.7]);
  const [th, setTh] = useState(0.5);
  const rgb = useThemeRGB();
  const inside = INSIDE[shape];
  const pt = (t: number): V2 => [t * a[0] + (1 - t) * b[0], t * a[1] + (1 - t) * b[1]];
  const N = 200;
  const flags = Array.from({ length: N + 1 }, (_, k) => inside(...pt(k / N)));
  const endsIn = inside(...a) && inside(...b);
  const leaves = flags.filter((v) => !v).length / (N + 1);
  const mc = useMemo(() => {
    const r = mulberry32(12);
    const P: V2[] = [];
    while (P.length < 400) { const p: V2 = [r() * 6 - 3, r() * 6 - 3]; if (inside(...p)) P.push(p); }
    let bad = 0;
    for (let k = 0; k < 2000; k++) {
      const p = P[Math.floor(r() * P.length)], q = P[Math.floor(r() * P.length)];
      for (let s = 1; s < 20; s++) if (!inside((s / 20) * p[0] + (1 - s / 20) * q[0], (s / 20) * p[1] + (1 - s / 20) * q[1])) { bad++; break; }
    }
    return bad / 2000;
  }, [shape]); // eslint-disable-line react-hooks/exhaustive-deps
  const segs: { pts: V2[]; ok: boolean }[] = [];
  flags.forEach((ok, k) => { const p = pt(k / N); const last = segs[segs.length - 1]; if (last && last.ok === ok) last.pts.push(p); else segs.push({ pts: last ? [last.pts[last.pts.length - 1], p] : [p], ok }); });
  const z = pt(th);
  const convex = mc === 0;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-2.6, 2.6]} equal aspect={0.8} maxH={440} ariaLabel="a set in the plane with a segment between two of its points"
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={2} deps={[shape, rgb.blue[0], rgb.panel[0]]} f={(x, y) => { const p = rgb.panel, c = rgb.blue, k = inside(x, y) ? 0.28 : 0; return [p[0] + (c[0] - p[0]) * k, p[1] + (c[1] - p[1]) * k, p[2] + (c[2] - p[2]) * k, 255]; }} />}>
            {segs.map((s, k) => <Polyline key={k} pts={s.pts} style={{ stroke: s.ok ? C.green : C.red, strokeWidth: 2.6 }} />)}
            <Dot p={z} r={5.5} color={inside(...z) ? C.green : C.red} stroke={C.bg} />
            <Handle p={a} color="amber" label="point a" onDrag={(q) => setA([clamp(q[0], -2.9, 2.9), clamp(q[1], -2.5, 2.5)])} />
            <Handle p={b} color="amber" label="point b" onDrag={(q) => setB([clamp(q[0], -2.9, 2.9), clamp(q[1], -2.5, 2.5)])} />
          </Plot>
          <Legend items={[{ label: "the set S", color: C.a("blue", 0.6) }, { label: "points a and b (drag)", color: C.amber }, { label: "part of the segment inside S", color: C.green }, { label: "part outside S", color: C.red }]} />
          {!endsIn && <Warn>Drag both endpoints into the shaded set: the definition only concerns segments between two points of S.</Warn>}
        </div>
      }
      controls={
        <>
          <Select label="Set S" value={shape} onChange={setShape} options={[{ value: "disk", label: "Disk (L2 ball)" }, { value: "polygon", label: "Polygon" }, { value: "halfplanes", label: "Intersection of five half-planes" }, { value: "crescent", label: "Crescent" }, { value: "annulus", label: "Ring" }, { value: "union", label: "Union of two disks" }]} />
          <ParameterSlider label="θ: position on the segment" value={th} min={0} max={1} step={0.01} onChange={setTh} />
          <p className="text-[12.5px] leading-relaxed text-muted">A set is convex when this test passes for every pair of its points. The Monte-Carlo readout tries 2000 random pairs; one failing pair is enough to prove a set is not convex.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\theta\mathbf a+(1-\theta)\mathbf b`, tex: true, value: `(${fmt(z[0], 3)}, ${fmt(z[1], 3)})`, color: inside(...z) ? C.green : C.red },
        { label: String.raw`\text{in }S?`, tex: true, value: inside(...z) ? "yes" : "no" },
        { label: String.raw`\text{fraction of this segment outside }S`, tex: true, value: fmt(leaves, 3) },
        { label: String.raw`\text{random pairs whose segment leaves }S`, tex: true, value: `${fmt(mc * 100, 1)}%` },
        { label: String.raw`\text{verdict}`, tex: true, value: convex ? "no counter-example found" : "not convex", color: convex ? C.green : C.red },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`S\text{ is convex}\iff\forall\,\mathbf a,\mathbf b\in S,\ \forall\,\theta\in[0,1]:\quad\theta\mathbf a+(1-\theta)\mathbf b\in S` },
        { tex: String.raw`${tex(th, 2)}\begin{bmatrix}${tex(a[0], 2)}\\${tex(a[1], 2)}\end{bmatrix}+${tex(1 - th, 2)}\begin{bmatrix}${tex(b[0], 2)}\\${tex(b[1], 2)}\end{bmatrix}=\begin{bmatrix}${tex(z[0], 3)}\\${tex(z[1], 3)}\end{bmatrix}${inside(...z) ? String.raw`\in S` : String.raw`\notin S`}` },
        { tex: String.raw`S_1,S_2\text{ convex}\ \Rightarrow\ S_1\cap S_2\text{ convex},\qquad\text{but }S_1\cup S_2\text{ need not be}`, note: "a half-plane is convex, so any polygon cut out by linear constraints is convex" },
      ]} />}
      note="Constraint sets in machine learning are usually convex: norm balls (ridge, LASSO), half-spaces (SVM margin constraints) and their intersections. Minimising a convex function over a convex set has no spurious local minima."
    />
  );
}
