"use client";
import { useMemo, useState, useEffect } from "react";
import { LabShell, Readouts, CalcSteps, Warn } from "@/components/visualization/LabShell";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { MatrixInput, AnimationControls, Presets } from "@/components/controls";
import { Plot, Polyline, Dot, C } from "@/components/visualization/Plot";
import { rrefSteps, nullSpace, lu, matMul, det, type Matrix } from "@/lib/math/linalg";
import { useTicker } from "@/hooks/useSize";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, tex, texMatrix } from "@/lib/utilities/format";

type Mode = "rref" | "system" | "rank" | "lu";

const DEFAULTS: Record<Mode, Matrix> = {
  rref: [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]],
  system: [[1, 1, 3], [2, -1, 0]],
  rank: [[1, 2, 3], [2, 4, 6], [1, 0, 1]],
  lu: [[2, 1, 1], [4, -6, 0], [-2, 7, 2]],
};

export default function Gauss({ mode = "rref" }: { mode?: Mode }) {
  const [M, setM] = useState<Matrix>(DEFAULTS[mode]);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const aug = mode === "rref" || mode === "system" ? 1 : 0;
  const res = useMemo(() => rrefSteps(M, aug), [M, aug]);
  const n = res.steps.length;
  useEffect(() => setK(0), [M]);
  useTicker(playing, 1100, () => setK((s) => { if (s + 1 >= n) { setPlaying(false); return n - 1; } return s + 1; }));
  const cur = res.steps[Math.min(k, n - 1)];
  const nCoef = M[0].length - aug;
  const rankA = res.pivots.length;
  // rank of augmented: count non-zero rows in result
  const rankAug = res.result.filter((r) => r.some((v) => Math.abs(v) > 1e-9)).length;
  const consistent = aug ? rankA === rankAug : true;

  const readouts: { label: string; value: React.ReactNode; tex?: boolean; color?: string }[] = [
    { label: "\\operatorname{rank}A", tex: true, value: rankA },
    { label: "\\text{pivot columns}", tex: true, value: res.pivots.map((p) => p + 1).join(", ") || "none" },
  ];
  const steps: { tex: string; note?: string }[] = [];
  let extra: React.ReactNode = null;

  if (aug) {
    readouts.push({ label: "\\operatorname{rank}[A|\\mathbf b]", tex: true, value: rankAug }, {
      label: "\\text{solutions}", tex: true,
      value: !consistent ? "none (inconsistent)" : rankA === nCoef ? "unique" : `infinitely many (${nCoef - rankA} free)`,
      color: !consistent ? C.red : rankA === nCoef ? C.green : C.amber,
    });
    if (consistent && rankA === nCoef) {
      const sol = res.result.slice(0, nCoef).map((r) => r[nCoef]);
      steps.push({ tex: `\\mathbf x=(${sol.map((v) => tex(v)).join(",\\;")})` });
    }
  }

  if (mode === "rank") {
    const ns = nullSpace(M);
    readouts.push({ label: "\\dim\\mathcal N(A)", tex: true, value: ns.length }, { label: "\\text{rank + nullity}", tex: true, value: `${rankA} + ${ns.length} = ${M[0].length}` });
    steps.push({ tex: `\\mathcal C(A)=\\operatorname{span}\\{${res.pivots.map((p) => `\\mathbf a_{${p + 1}}`).join(",") || "\\varnothing"}\\}` });
    steps.push({ tex: ns.length ? `\\mathcal N(A)=\\operatorname{span}\\{${ns.map((v) => texMatrix(v.map((x) => [x]), 2)).join(",")}\\}` : "\\mathcal N(A)=\\{\\mathbf 0\\}", note: ns.length ? "A times each basis vector is 0" : "only the zero vector" });
  }

  if (mode === "lu") {
    const f = lu(M);
    if (f) {
      extra = (
        <div className="flex flex-wrap items-center justify-center gap-3 border-t border-line pt-4">
          <MatrixView M={f.P} label="P" digits={0} />
          <MatrixView M={M} label="A" />
          <span className="text-faint">=</span>
          <MatrixView M={f.L} label="L (multipliers)" digits={3} />
          <MatrixView M={f.U} label="U (echelon)" digits={3} />
        </div>
      );
      const PA = matMul(f.P, M);
      const LU = matMul(f.L, f.U);
      const err = Math.max(...PA.flat().map((v, i) => Math.abs(v - LU.flat()[i])));
      readouts.push({ label: "\\max|PA-LU|", tex: true, value: fmt(err, 12) }, { label: "\\det A", tex: true, value: fmt(det(M)) });
      steps.push({ tex: `\\det A=\\pm\\prod u_{ii}=\\pm(${f.U.map((r, i) => tex(r[i], 3)).join(")(")})=${tex(det(M))}` });
    } else extra = <Warn>A is singular — LU with partial pivoting breaks down (zero pivot).</Warn>;
  }

  if (mode === "system" && M.length === 2 && nCoef === 2) {
    const lines = M.map((r) => {
      const [a, b, c] = r;
      if (Math.abs(b) > 1e-9) return [[-6, (c - a * -6) / b], [6, (c - a * 6) / b]] as [number, number][];
      if (Math.abs(a) > 1e-9) return [[c / a, -6], [c / a, 6]] as [number, number][];
      return null;
    });
    const sol = consistent && rankA === 2 ? [res.result[0][2], res.result[1][2]] as [number, number] : null;
    extra = (
      <Plot x={[-5, 5]} y={[-5, 5]} equal originAxes aspect={0.55} maxH={300} ariaLabel="lines of the linear system">
        {lines.map((l, i) => l && <Polyline key={i} pts={l} style={{ stroke: i ? C.pink : C.blue, strokeWidth: 2 }} />)}
        {sol && <Dot p={sol} r={5} color={C.green} />}
      </Plot>
    );
  }

  return (
    <LabShell
      viz={
        <div className="space-y-4">
          <div className="flex flex-col items-center gap-2 py-2">
            <div className="text-[12.5px] text-muted">Step {Math.min(k, n - 1) + 1} / {n}</div>
            <div className="min-h-[24px] text-[13.5px]"><Tex tex={cur.tex} /></div>
            <MatrixView M={cur.matrix} cell={52} digits={3} hl={(i, j) => (cur.pivot && cur.pivot[0] === i && cur.pivot[1] === j ? "pivot" : null)} faded={() => false} />
            <div className="text-[12px] text-faint">{cur.description}{aug ? " · last column is the right-hand side b" : ""}</div>
          </div>
          {extra}
        </div>
      }
      controls={
        <>
          <MatrixInput label={aug ? "Augmented matrix [A | b]" : "Matrix A"} value={M} onChange={setM} step={1} highlight={(_, j) => aug === 1 && j === M[0].length - 1} />
          <AnimationControls playing={playing} onPlay={() => { if (!playing && k >= n - 1) setK(0); setPlaying(!playing); }} onStep={() => setK((s) => Math.min(n - 1, s + 1))} canStep={k < n - 1} onReset={() => { setPlaying(false); setK(0); }} />
          <Presets
            presets={
              mode === "system"
                ? [{ label: "Unique", value: [[1, 1, 3], [2, -1, 0]] }, { label: "Parallel (none)", value: [[1, 2, 2], [2, 4, 1]] }, { label: "Same line (∞)", value: [[1, 2, 2], [2, 4, 4]] }]
                : mode === "rank"
                  ? [{ label: "Rank 2", value: [[1, 2, 3], [2, 4, 6], [1, 0, 1]] }, { label: "Rank 3", value: [[2, 0, 1], [1, 3, 0], [0, 1, 4]] }, { label: "Rank 1", value: [[1, 2, 3], [2, 4, 6], [3, 6, 9]] }]
                  : [{ label: "Default", value: DEFAULTS[mode] }, { label: "Needs pivoting", value: mode === "lu" ? [[0, 2, 1], [1, 1, 1], [2, 1, 0]] : [[0, 2, 1, 3], [1, 1, 1, 6], [2, 1, 0, 4]] }]
            }
            onPick={(v) => setM(v as Matrix)}
          />
        </>
      }
      readouts={<Readouts items={readouts} />}
      calc={steps.length ? <CalcSteps title="Result" steps={steps} /> : undefined}
      note="Partial pivoting (swapping the largest pivot into place) is used at every step, as production solvers do."
    />
  );
}
