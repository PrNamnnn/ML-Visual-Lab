"use client";
import { useMemo, useState } from "react";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { MatrixInput, ParameterSlider, Segmented, Select, AnimationControls } from "@/components/controls";
import { matMul, matVec, transpose, hadamard, matAdd, matScale, trace, matPow, isSymmetric, isOrthogonal, det, type Matrix } from "@/lib/math/linalg";
import { useTicker } from "@/hooks/useSize";
import { Plot, Arrow, Polyline, C } from "@/components/visualization/Plot";
import { fmt, tex, texMatrix, range } from "@/lib/utilities/format";
import { mulberry32 } from "@/lib/math/random";
import { Tex } from "@/components/mathematics/Tex";

type Op = "sparse" | "dims" | "add" | "multiply" | "matvec" | "transpose" | "special" | "power" | "hadamard" | "broadcast";

const resize = (M: Matrix, r: number, c: number, fill = (i: number, j: number) => ((i * 3 + j * 5) % 7) - 3): Matrix =>
  range(r).map((i) => range(c).map((j) => M[i]?.[j] ?? fill(i, j)));

function OpSym({ children }: { children: React.ReactNode }) {
  return <div className="self-center px-1 text-[18px] text-faint">{children}</div>;
}

export default function MatrixOps({ op: initial = "multiply", lab = false }: { op?: Op; lab?: boolean }) {
  const [op, setOp] = useState<Op>(initial);
  const [A, setA] = useState<Matrix>([[1, 2, 0], [3, -1, 2]]);
  const [B, setB] = useState<Matrix>([[2, 1], [0, 3], [-1, 4]]);
  const [x, setX] = useState<number[]>([2, -1, 1]);
  const [k, setK] = useState(0); // step index for multiplication
  const [playing, setPlaying] = useState(false);
  const [scalar, setScalar] = useState(2);
  const [rows, setRows] = useState(4);
  const [cols, setCols] = useState(6);
  const [hover, setHover] = useState<[number, number]>([-1, -1]);
  const [colView, setColView] = useState(false);
  const [power, setPower] = useState(3);
  const [P, setP] = useState<Matrix>([[0.9, 0.4], [-0.3, 0.8]]);
  const [special, setSpecial] = useState<"identity" | "diagonal" | "upper" | "lower" | "symmetric" | "orthogonal" | "sparse">("diagonal");
  const [tStep, setTStep] = useState(1);
  const [density, setDensity] = useState(0.25);

  const C2 = useMemo(() => {
    try { return matMul(A, B); } catch { return null; }
  }, [A, B]);
  const m = A.length, p = B[0]?.length ?? 0;
  const total = m * p;
  useTicker(playing && op === "multiply", 900, () => setK((s) => { if (s + 1 >= total) { setPlaying(false); return total - 1; } return s + 1; }));
  useTicker(playing && op === "transpose", 30, () => setTStep((s) => { if (s >= 1) { setPlaying(false); return 1; } return Math.min(1, s + 0.03); }));

  let viz: React.ReactNode = null;
  let steps: { tex: string; note?: string }[] = [];
  let readouts: { label: string; value: React.ReactNode; tex?: boolean }[] = [];
  let controls: React.ReactNode = null;

  const dimControls = (
    <div className="grid grid-cols-2 gap-3">
      <ParameterSlider label="rows of A" value={A.length} min={1} max={4} step={1} onChange={(r) => { setA(resize(A, r, A[0].length)); setK(0); }} />
      <ParameterSlider label="cols of A = rows of B" value={A[0].length} min={1} max={4} step={1} onChange={(c) => { setA(resize(A, A.length, c)); setB(resize(B, c, B[0].length)); setX(range(c).map((i) => x[i] ?? 1)); setK(0); }} />
    </div>
  );

  switch (op) {
    case "dims": {
      const M = range(rows).map((i) => range(cols).map((j) => Math.round(10 * Math.sin(i * 1.3 + j * 0.7) * Math.cos(j * 0.4 - i)) / 10));
      viz = (
        <div className="flex flex-col items-center gap-4 py-4">
          <MatrixView M={M} digits={1} cell={44} onHover={(i, j) => setHover([i, j])} hl={(i, j) => (i === hover[0] && j === hover[1] ? "out" : i === hover[0] ? "row" : j === hover[1] ? "col" : null)} label={<Tex tex={`A\\in\\mathbb R^{${rows}\\times${cols}}`} />} />
          <div className="text-[13px] text-muted">
            {hover[0] >= 0 ? <Tex tex={`a_{${hover[0] + 1}${hover[1] + 1}}=${tex(M[hover[0]][hover[1]], 1)}\\quad\\text{(row }${hover[0] + 1},\\text{ column }${hover[1] + 1})`} /> : "Hover a cell"}
          </div>
        </div>
      );
      readouts = [{ label: "shape", value: `${rows} × ${cols}` }, { label: "entries", value: rows * cols }, { label: "row vectors", value: `${rows} in ℝ^${cols}` }, { label: "column vectors", value: `${cols} in ℝ^${rows}` }];
      controls = (
        <>
          <ParameterSlider label="rows m" value={rows} min={1} max={8} step={1} onChange={setRows} />
          <ParameterSlider label="columns n" value={cols} min={1} max={8} step={1} onChange={setCols} />
        </>
      );
      steps = [{ tex: `A\\mathbf x\\text{ needs }\\mathbf x\\in\\mathbb R^{${cols}}\\text{ and returns }\\mathbb R^{${rows}}` }];
      break;
    }
    case "add": {
      const B2 = resize(B, A.length, A[0].length);
      const S = matAdd(A, matScale(B2, scalar));
      viz = (
        <div className="flex flex-wrap items-center justify-center gap-2 py-4">
          <MatrixView M={A} label="A" hl={(i, j) => (i === hover[0] && j === hover[1] ? "row" : null)} onHover={(i, j) => setHover([i, j])} />
          <OpSym>+</OpSym>
          <div className="self-center num text-[14px]">{fmt(scalar)}</div>
          <MatrixView M={B2} label="B" hl={(i, j) => (i === hover[0] && j === hover[1] ? "col" : null)} onHover={(i, j) => setHover([i, j])} />
          <OpSym>=</OpSym>
          <MatrixView M={S} label="A + cB" hl={(i, j) => (i === hover[0] && j === hover[1] ? "out" : null)} onHover={(i, j) => setHover([i, j])} />
        </div>
      );
      const [hi, hj] = hover[0] >= 0 ? hover : [0, 0];
      steps = [{ tex: `(A+cB)_{${hi + 1}${hj + 1}}=${tex(A[hi][hj], 2)}+${tex(scalar, 2)}\\cdot(${tex(B2[hi][hj], 2)})=${tex(S[hi][hj], 3)}`, note: "hover cells to inspect" }];
      controls = (
        <>
          <MatrixInput label="A" value={A} onChange={setA} step={1} />
          <MatrixInput label="B" value={B2} onChange={setB} step={1} />
          <ParameterSlider label="scalar c" value={scalar} min={-3} max={3} step={0.5} onChange={setScalar} />
        </>
      );
      readouts = [{ label: "shape", value: `${A.length} × ${A[0].length}` }];
      break;
    }
    case "multiply": {
      const i = Math.floor(k / p), j = k % p;
      const n = A[0].length;
      const partial = C2 ? C2.map((r, a) => r.map((v, b) => (a * p + b <= k ? v : NaN))) : null;
      viz = C2 && partial ? (
        <div className="flex flex-wrap items-center justify-center gap-2 py-4">
          <MatrixView M={A} label="A" hl={(a) => (a === i ? "row" : null)} />
          <OpSym>×</OpSym>
          <MatrixView M={B} label="B" hl={(_, b) => (b === j ? "col" : null)} />
          <OpSym>=</OpSym>
          <MatrixView M={partial} label="C = AB" hl={(a, b) => (a === i && b === j ? "out" : null)} heat />
        </div>
      ) : null;
      if (C2) {
        const terms = range(n).map((t) => `(${tex(A[i][t], 2)})(${tex(B[t][j], 2)})`).join("+");
        steps = [
          { tex: `c_{${i + 1}${j + 1}}=\\sum_{k=1}^{${n}}a_{${i + 1}k}b_{k${j + 1}}=${terms}=${tex(C2[i][j], 3)}`, note: `row ${i + 1} · column ${j + 1}` },
          { tex: `(${m}\\times${n})\\cdot(${n}\\times${p})\\to(${m}\\times${p})` },
        ];
        readouts = [{ label: "output cell", value: `(${i + 1}, ${j + 1}) of ${m}×${p}` }, { label: "multiply-adds", value: m * n * p }, { label: "FLOPs ≈ 2mnp", value: 2 * m * n * p }];
      }
      controls = (
        <>
          {dimControls}
          <ParameterSlider label="cols of B" value={B[0].length} min={1} max={4} step={1} onChange={(c) => { setB(resize(B, B.length, c)); setK(0); }} />
          <MatrixInput label="A" value={A} onChange={setA} step={1} />
          <MatrixInput label="B" value={B} onChange={setB} step={1} />
          <AnimationControls
            playing={playing}
            onPlay={() => { if (!playing && k >= total - 1) setK(0); setPlaying(!playing); }}
            onStep={() => setK((s) => Math.min(total - 1, s + 1))}
            canStep={k < total - 1}
            onReset={() => { setPlaying(false); setK(0); }}
          />
        </>
      );
      break;
    }
    case "matvec": {
      const y = matVec(A, x);
      const A2 = resize(A, 2, A[0].length);
      const cols2 = transpose(A2);
      let acc: [number, number] = [0, 0];
      const chain = cols2.map((c, idx) => {
        const from = acc;
        acc = [acc[0] + x[idx] * c[0], acc[1] + x[idx] * c[1]];
        return { from, to: acc, idx };
      });
      viz = (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-center gap-2 py-2">
            <MatrixView M={A} label="A" hl={(a, b) => (colView ? (b === hover[1] ? "col" : null) : a === hover[0] ? "row" : null)} onHover={(a, b) => setHover([a, b])} />
            <OpSym>×</OpSym>
            <MatrixView M={x.map((v) => [v])} label="x" />
            <OpSym>=</OpSym>
            <MatrixView M={y.map((v) => [v])} label="Ax" hl={(a) => (!colView && a === hover[0] ? "out" : null)} />
          </div>
          {colView && A.length === 2 && (
            <Plot x={[-6, 6]} y={[-5, 5]} equal originAxes aspect={0.5} maxH={300} ariaLabel="column view of matrix-vector product">
              {chain.map((c) => <Arrow key={c.idx} from={c.from} to={c.to} color={(["blue", "pink", "amber", "green"] as const)[c.idx % 4]} label={`x${c.idx + 1}a${c.idx + 1}`} />)}
              <Arrow from={[0, 0]} to={[y[0], y[1]]} color="fg" width={2.5} dashed label="Ax" />
            </Plot>
          )}
        </div>
      );
      steps = colView
        ? [{ tex: `A\\mathbf x=${x.map((v, idx) => `${tex(v, 2)}\\,\\mathbf a_{${idx + 1}}`).join("+")}=${texMatrix(y.map((v) => [v]))}` }]
        : A.map((r, a) => ({ tex: `(A\\mathbf x)_{${a + 1}}=${r.map((v, b) => `(${tex(v, 2)})(${tex(x[b], 2)})`).join("+")}=${tex(y[a], 3)}` }));
      controls = (
        <>
          <Segmented value={colView ? "col" : "row"} onChange={(v) => setColView(v === "col")} options={[{ value: "row", label: "Row view" }, { value: "col", label: "Column view" }]} />
          {dimControls}
          <MatrixInput label="A" value={A} onChange={setA} step={1} />
          <MatrixInput label="x" value={x.map((v) => [v])} onChange={(m2) => setX(m2.map((r) => r[0]))} step={1} />
          {colView && A.length !== 2 && <p className="text-[12px] text-faint">The 2D column-view picture needs A to have 2 rows.</p>}
        </>
      );
      break;
    }
    case "transpose": {
      const T = transpose(A);
      viz = (
        <div className="flex flex-wrap items-center justify-center gap-6 py-4">
          <MatrixView M={A} label={`A (${A.length}×${A[0].length})`} hl={(i, j) => (i === hover[0] && j === hover[1] ? "row" : i === j ? "pivot" : null)} onHover={(i, j) => setHover([i, j])} />
          <div className="relative" style={{ opacity: 0.25 + 0.75 * tStep, transform: `rotate(${(1 - tStep) * -90}deg) scale(${0.8 + 0.2 * tStep})`, transition: "none" }}>
            <MatrixView M={T} label={`Aᵀ (${T.length}×${T[0].length})`} hl={(i, j) => (j === hover[0] && i === hover[1] ? "out" : i === j ? "pivot" : null)} />
          </div>
        </div>
      );
      const [hi, hj] = hover[0] >= 0 ? hover : [0, 1 % A[0].length];
      steps = [{ tex: `(A^\\top)_{${hj + 1}${hi + 1}}=a_{${hi + 1}${hj + 1}}=${tex(A[hi]?.[hj] ?? 0, 2)}`, note: "amber = diagonal (unchanged)" }];
      controls = (
        <>
          {dimControls}
          <MatrixInput label="A" value={A} onChange={setA} step={1} />
          <AnimationControls playing={playing} onPlay={() => { if (!playing) setTStep(0); setPlaying(!playing); }} onReset={() => { setPlaying(false); setTStep(1); }} />
        </>
      );
      readouts = [{ label: "shape A", value: `${A.length}×${A[0].length}` }, { label: "shape Aᵀ", value: `${A[0].length}×${A.length}` }, { label: "symmetric?", value: isSymmetric(A) ? "yes (A = Aᵀ)" : "no" }];
      break;
    }
    case "hadamard": {
      const B2 = resize(B, A.length, A[0].length);
      const H = hadamard(A, B2);
      let MM: Matrix | null = null;
      try { MM = matMul(A, transpose(B2)); } catch { MM = null; }
      viz = (
        <div className="flex flex-wrap items-center justify-center gap-2 py-4">
          <MatrixView M={A} label="A" hl={(i, j) => (i === hover[0] && j === hover[1] ? "row" : null)} onHover={(i, j) => setHover([i, j])} />
          <OpSym>⊙</OpSym>
          <MatrixView M={B2} label="B" hl={(i, j) => (i === hover[0] && j === hover[1] ? "col" : null)} onHover={(i, j) => setHover([i, j])} />
          <OpSym>=</OpSym>
          <MatrixView M={H} label="A ⊙ B" hl={(i, j) => (i === hover[0] && j === hover[1] ? "out" : null)} onHover={(i, j) => setHover([i, j])} />
        </div>
      );
      const [hi, hj] = hover[0] >= 0 ? hover : [0, 0];
      steps = [{ tex: `(A\\odot B)_{${hi + 1}${hj + 1}}=a_{${hi + 1}${hj + 1}}b_{${hi + 1}${hj + 1}}=(${tex(A[hi][hj], 2)})(${tex(B2[hi][hj], 2)})=${tex(H[hi][hj], 3)}` }];
      if (MM) steps.push({ tex: `\\text{contrast: }AB^\\top=${texMatrix(MM, 1)}\\text{ mixes whole rows}` });
      controls = (
        <>
          <MatrixInput label="A" value={A} onChange={setA} step={1} />
          <MatrixInput label="B" value={B2} onChange={setB} step={1} />
        </>
      );
      break;
    }
    case "broadcast": {
      const b = A[0].map((_, j) => B[0]?.[j] ?? 1);
      const tiled = A.map(() => b);
      const R = A.map((r) => r.map((v, j) => v + b[j]));
      viz = (
        <div className="flex flex-wrap items-center justify-center gap-2 py-4">
          <MatrixView M={A} label={`X (${A.length}×${A[0].length})`} />
          <OpSym>+</OpSym>
          <MatrixView M={tiled} label={`b (1×${b.length}) → broadcast`} faded={(i) => i > 0} />
          <OpSym>=</OpSym>
          <MatrixView M={R} label="X + b" />
        </div>
      );
      steps = [{ tex: `(X+\\mathbf b)_{ij}=x_{ij}+b_j,\\quad\\text{shapes }(${A.length},${A[0].length})+(1,${b.length})\\to(${A.length},${A[0].length})` }];
      controls = (
        <>
          {dimControls}
          <MatrixInput label="X" value={A} onChange={setA} step={1} />
          <MatrixInput label="b (row vector)" value={[b]} onChange={(m2) => setB([m2[0], ...B.slice(1)])} step={1} />
        </>
      );
      break;
    }
    case "power": {
      const Pk = matPow(P, power);
      const traj: [number, number][] = [];
      let v: number[] = [1, 0.5];
      for (let t = 0; t <= 20; t++) { traj.push([v[0], v[1]]); v = matVec(P, v); }
      viz = (
        <div className="grid items-center gap-4 md:grid-cols-[auto_1fr]">
          <div className="flex flex-col items-center gap-3">
            <MatrixView M={P} label="A" digits={3} />
            <MatrixView M={Pk} label={<Tex tex={`A^{${power}}`} />} digits={3} />
          </div>
          <Plot x={[-2, 2]} y={[-2, 2]} equal originAxes aspect={0.7} maxH={340} ariaLabel="trajectory of repeated multiplication">
            <Polyline pts={traj} style={{ stroke: C.blue, strokeWidth: 1.4 }} />
            <Arrow from={[0, 0]} to={traj[Math.min(power, traj.length - 1)]} color="pink" label={`A^${power}x₀`} />
          </Plot>
        </div>
      );
      const tr = trace(P);
      const dd = det(P);
      const disc = tr * tr - 4 * dd;
      const rho = disc >= 0 ? Math.max(Math.abs((tr + Math.sqrt(disc)) / 2), Math.abs((tr - Math.sqrt(disc)) / 2)) : Math.sqrt(dd);
      readouts = [{ label: "\\operatorname{tr}A", tex: true, value: fmt(tr) }, { label: "\\rho(A)=\\max|\\lambda|", tex: true, value: fmt(rho) }, { label: "\\text{behaviour}", tex: true, value: rho > 1.0001 ? "grows" : rho < 0.9999 ? "decays → 0" : "neutral" }];
      steps = [{ tex: `A^{${power}}=${texMatrix(Pk, 3)}` }, { tex: `\\rho(A)=${tex(rho)}${rho > 1 ? ">1\\Rightarrow\\|A^k\\mathbf x\\|\\to\\infty" : "<1\\Rightarrow A^k\\mathbf x\\to\\mathbf 0"}` }];
      controls = (
        <>
          <MatrixInput label="A (2×2)" value={P} onChange={setP} step={0.05} />
          <ParameterSlider label="power k" value={power} min={0} max={20} step={1} onChange={setPower} />
          <p className="text-[12px] text-faint">The blue path is x₀, Ax₀, A²x₀, … starting from x₀ = (1, 0.5).</p>
        </>
      );
      break;
    }
    case "sparse": {
      const n = 6;
      const r0 = mulberry32(7);
      const S = range(n).map(() => range(n).map(() => (r0() < density ? Math.round((r0() * 8 - 4) * 2) / 2 || 1 : 0)));
      const vals: number[] = [], colIdx: number[] = [], rowPtr: number[] = [0];
      S.forEach((r) => { r.forEach((v, j) => { if (v !== 0) { vals.push(v); colIdx.push(j); } }); rowPtr.push(vals.length); });
      const xv = range(n).map((i) => i + 1);
      const y = range(n).map((i) => { let s = 0; for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) s += vals[k] * xv[colIdx[k]]; return s; });
      const nnz = vals.length;
      const arr = (a: number[]) => <span className="num">[{a.map((v) => fmt(v, 1)).join(", ")}]</span>;
      viz = (
        <div className="grid items-start gap-4 md:grid-cols-[auto_1fr]">
          <MatrixView M={S} label={`A (${n}×${n}), ${nnz} non-zeros`} digits={1} cell={38} faded={(i, j) => S[i][j] === 0} />
          <div className="space-y-2 text-[12.5px]">
            <div className="label">CSR (compressed sparse row) storage</div>
            <div><span className="text-faint">values  </span>{arr(vals)}</div>
            <div><span className="text-faint">col_idx </span>{arr(colIdx)}</div>
            <div><span className="text-faint">row_ptr </span>{arr(rowPtr)}</div>
            <p className="text-faint">Row i&apos;s entries are values[row_ptr[i] … row_ptr[i+1]−1], in the columns given by col_idx.</p>
            <div className="pt-2"><span className="text-faint">y = A·x with x = (1,…,6): </span>{arr(y)}</div>
          </div>
        </div>
      );
      readouts = [{ label: "density", value: `${fmt((100 * nnz) / (n * n), 1)}%` }, { label: "dense storage", value: `${n * n} numbers` }, { label: "CSR storage", value: `${2 * nnz + n + 1} numbers` }, { label: "multiplications for A·x", value: `${nnz} (dense: ${n * n})` }];
      steps = [{ tex: String.raw`y_i=\sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1}\text{values}[k]\cdot x_{\text{col\_idx}[k]}`, note: "only stored non-zeros are touched" }, { tex: String.raw`\text{memory: } 2\cdot\text{nnz}+n+1=${2 * nnz + n + 1}\ \text{vs}\ n^2=${n * n}` }];
      controls = <ParameterSlider label="density (fraction non-zero)" value={density} min={0.05} max={1} step={0.05} onChange={setDensity} />;
      break;
    }
    case "special": {
      const S: Record<typeof special, Matrix> = {
        identity: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        diagonal: [[2, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 3]],
        upper: [[2, 1, -1, 3], [0, 1, 2, 1], [0, 0, 3, -2], [0, 0, 0, 1]],
        lower: [[2, 0, 0, 0], [1, 1, 0, 0], [-1, 2, 3, 0], [3, 1, -2, 1]],
        symmetric: [[4, 1, 2, 0], [1, 3, 0, 1], [2, 0, 5, -1], [0, 1, -1, 2]],
        orthogonal: [[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, -0.5, 0.5]],
        sparse: [[0, 0, 3, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2]],
      };
      const M = S[special];
      const two = M.slice(0, 2).map((r) => r.slice(0, 2)) as Matrix;
      const zerosFrac = M.flat().filter((v) => v === 0).length / 16;
      viz = (
        <div className="grid items-center gap-4 md:grid-cols-2">
          <div className="flex justify-center py-2"><MatrixView M={M} label={special} digits={1} /></div>
          <Plot x={[-3, 3]} y={[-3, 3]} equal originAxes aspect={0.8} maxH={300} ariaLabel="effect of the leading 2x2 block">
            <Arrow from={[0, 0]} to={[two[0][0], two[1][0]]} color="blue" label="Ae₁" />
            <Arrow from={[0, 0]} to={[two[0][1], two[1][1]]} color="pink" label="Ae₂" />
          </Plot>
        </div>
      );
      readouts = [
        { label: "symmetric", value: isSymmetric(M) ? "yes" : "no" },
        { label: "orthogonal (QᵀQ = I)", value: isOrthogonal(M) ? "yes" : "no" },
        { label: "det", value: fmt(det(M)) },
        { label: "sparsity", value: `${Math.round(zerosFrac * 100)}% zeros` },
      ];
      steps = [
        special === "upper" || special === "lower" ? { tex: `\\det=\\prod_i a_{ii}=${M.map((r, i) => tex(r[i], 1)).join("\\cdot")}=${tex(det(M))}`, note: "triangular: determinant is the diagonal product" } :
        special === "orthogonal" ? { tex: "Q^\\top Q=I\\Rightarrow Q^{-1}=Q^\\top,\\ |\\det Q|=1" } :
        special === "diagonal" ? { tex: `D\\mathbf x=(${M.map((r, i) => `${tex(r[i], 1)}x_${i + 1}`).join(",\\;")})`, note: "each coordinate scaled independently" } :
        special === "symmetric" ? { tex: "A=A^\\top\\Rightarrow\\text{real eigenvalues, orthogonal eigenvectors}" } :
        special === "sparse" ? { tex: `\\text{store only }${M.flat().filter((v) => v !== 0).length}\\text{ non-zeros (e.g. CSR format)}` } :
        { tex: "I\\mathbf x=\\mathbf x" },
      ];
      controls = (
        <Select
          label="Structure"
          value={special}
          onChange={setSpecial}
          options={[
            { value: "identity", label: "Identity" }, { value: "diagonal", label: "Diagonal" }, { value: "upper", label: "Upper triangular" },
            { value: "lower", label: "Lower triangular" }, { value: "symmetric", label: "Symmetric" }, { value: "orthogonal", label: "Orthogonal" }, { value: "sparse", label: "Sparse" },
          ]}
        />
      );
      break;
    }
  }

  return (
    <LabShell
      viz={<div className="min-h-[260px] overflow-x-auto">{viz ?? <div className="p-6 text-[13px] text-red">Shapes are incompatible.</div>}</div>}
      controls={
        <>
          {lab && (
            <Select label="Operation" value={op} onChange={(v) => { setOp(v); setK(0); }} options={[
              { value: "dims", label: "Dimensions" }, { value: "add", label: "Addition & scaling" }, { value: "multiply", label: "Multiplication" },
              { value: "matvec", label: "Matrix × vector" }, { value: "transpose", label: "Transpose" }, { value: "hadamard", label: "Hadamard product" },
              { value: "broadcast", label: "Broadcasting" }, { value: "power", label: "Powers" }, { value: "special", label: "Special matrices" }, { value: "sparse", label: "Sparse (CSR)" },
            ]} />
          )}
          {controls}
        </>
      }
      readouts={readouts.length ? <Readouts items={readouts} /> : undefined}
      calc={steps.length ? <CalcSteps steps={steps} /> : undefined}
    />
  );
}

