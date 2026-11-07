"use client";
import { useMemo, useState } from "react";
import { PlotlyChart } from "@/components/visualization/PlotlyChart";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { MatrixInput, Presets, Select, ParameterSlider } from "@/components/controls";
import { eigSym, type Matrix } from "@/lib/math/linalg";
import { functions2D } from "@/lib/math/calculus";
import { objectives, runOptimizer } from "@/lib/math/optim";
import { fmt, tex, linspace } from "@/lib/utilities/format";

function grid(f: (x: number, y: number) => number, lo: number, hi: number, n = 50, ylo = lo, yhi = hi) {
  const xs = linspace(lo, hi, n);
  const ys = linspace(ylo, yhi, n);
  const z = ys.map((y) => xs.map((x) => f(x, y)));
  return { xs, ys, z };
}

const colorscale = [[0, "#16324f"], [0.35, "#2f6fb0"], [0.65, "#5aa9ff"], [0.85, "#e88fc0"], [1, "#ff6fae"]];

export default function Surface3D({ fn = "quadform" }: { fn?: "quadform" | "function" | "loss" }) {
  const [A, setA] = useState<Matrix>([[2, 0.5], [0.5, 1]]);
  const [fid, setFid] = useState(functions2D[0].id);
  const [oid, setOid] = useState("rosenbrock");
  const [lr, setLr] = useState(0.002);

  if (fn === "quadform") {
    const S: Matrix = [[A[0][0], (A[0][1] + A[1][0]) / 2], [(A[0][1] + A[1][0]) / 2, A[1][1]]];
    const { values } = eigSym(S);
    const kind = values.every((v) => v > 1e-9) ? "positive definite (bowl)" : values.every((v) => v < -1e-9) ? "negative definite (dome)" : values.some((v) => v > 1e-9) && values.some((v) => v < -1e-9) ? "indefinite (saddle)" : "semidefinite (flat direction)";
    return <QuadForm A={A} setA={setA} S={S} values={values} kind={kind} />;
  }
  if (fn === "loss") {
    const obj = objectives.find((o) => o.id === oid)!;
    return <LossSurface obj={obj} oid={oid} setOid={setOid} lr={lr} setLr={setLr} />;
  }
  const f = functions2D.find((g) => g.id === fid)!;
  return <FunctionSurface fid={fid} setFid={setFid} f={f} />;
}

function QuadForm({ A, setA, S, values, kind }: { A: Matrix; setA: (m: Matrix) => void; S: Matrix; values: number[]; kind: string }) {
  const data = useMemo(() => {
    const g = grid((x, y) => S[0][0] * x * x + 2 * S[0][1] * x * y + S[1][1] * y * y, -2, 2, 46);
    return [{ type: "surface", x: g.xs, y: g.ys, z: g.z, colorscale, showscale: false, contours: { z: { show: true, usecolormap: true, project: { z: true } } } }];
  }, [S]);
  return (
    <LabShell
      viz={<PlotlyChart data={data} ariaLabel={`surface of quadratic form, ${kind}`} />}
      controls={
        <>
          <MatrixInput label="A (symmetrised)" value={A} onChange={setA} step={0.1} />
          <Presets presets={[{ label: "PD bowl", value: [[2, 0.5], [0.5, 1]] }, { label: "Saddle", value: [[1, 0], [0, -1]] }, { label: "PSD valley", value: [[1, 1], [1, 1]] }, { label: "ND dome", value: [[-1, 0.2], [0.2, -1.5]] }]} onPick={(m) => setA(m)} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\lambda_1", tex: true, value: fmt(values[0]) }, { label: "\\lambda_2", tex: true, value: fmt(values[1]) }, { label: "\\text{type}", tex: true, value: kind }]} />}
      calc={<CalcSteps steps={[{ tex: `q(x,y)=${tex(S[0][0])}x^2+${tex(2 * S[0][1])}xy+${tex(S[1][1])}y^2` }, { tex: `\\text{eigenvalues of }A:\\ ${tex(values[0])},\\ ${tex(values[1])}` }]} />}
      note="Only the symmetric part (A + Aᵀ)/2 affects xᵀAx, so the matrix is symmetrised before analysis."
    />
  );
}

function FunctionSurface({ fid, setFid, f }: { fid: string; setFid: (s: string) => void; f: (typeof functions2D)[number] }) {
  const data = useMemo(() => {
    const g = grid(f.f, -f.domain[1], f.domain[1], 50);
    return [{ type: "surface", x: g.xs, y: g.ys, z: g.z, colorscale, showscale: false, contours: { z: { show: true, usecolormap: true, project: { z: true } } } }];
  }, [f]);
  return (
    <LabShell
      viz={<PlotlyChart data={data} ariaLabel={`surface of ${f.name}`} />}
      controls={<Select label="Function" value={fid} onChange={setFid} options={functions2D.map((g) => ({ value: g.id, label: g.name }))} />}
      calc={<CalcSteps steps={[{ tex: f.tex }, { tex: f.gradTex }]} />}
    />
  );
}

function LossSurface({ obj, oid, setOid, lr, setLr }: { obj: (typeof objectives)[number]; oid: string; setOid: (s: string) => void; lr: number; setLr: (v: number) => void }) {
  const { path, diverged } = useMemo(() => runOptimizer(obj, { id: "gd", lr }, obj.start, 400), [obj, lr]);
  const data = useMemo(() => {
    const [[x0, x1], [y0, y1]] = obj.domain;
    const zf = obj.id === "rosenbrock" ? (x: number, y: number) => Math.log1p(obj.f(x, y)) : obj.f;
    const g = grid(zf, x0, x1, 56, y0, y1);
    const p = path.filter((q) => q[0] >= x0 && q[0] <= x1 && q[1] >= y0 && q[1] <= y1);
    return [
      { type: "surface", x: g.xs, y: g.ys, z: g.z, colorscale, showscale: false, opacity: 0.92 },
      { type: "scatter3d", mode: "lines+markers", x: p.map((q) => q[0]), y: p.map((q) => q[1]), z: p.map((q) => zf(q[0], q[1]) + 0.02), line: { color: "#ffffff", width: 4 }, marker: { size: 2, color: "#ffffff" } },
    ];
  }, [obj, path]);
  const last = path[path.length - 1];
  return (
    <LabShell
      viz={<PlotlyChart data={data} ariaLabel={`loss surface ${obj.name} with gradient descent path`} />}
      controls={
        <>
          <Select label="Loss landscape" value={oid} onChange={setOid} options={objectives.map((o) => ({ value: o.id, label: o.name }))} />
          <ParameterSlider label="learning rate η" value={lr} min={0.0005} max={0.3} step={0.0005} onChange={setLr} format={(v) => v.toFixed(4)} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{steps}", tex: true, value: path.length - 1 }, { label: "\\text{final }f", tex: true, value: diverged ? "diverged" : fmt(obj.f(last[0], last[1])) }, { label: "\\theta_{\\text{final}}", tex: true, value: `(${fmt(last[0])}, ${fmt(last[1])})` }]} />}
      calc={<CalcSteps steps={[{ tex: obj.tex }, { tex: "\\theta_{t+1}=\\theta_t-\\eta\\nabla f(\\theta_t)" }]} />}
      note={obj.id === "rosenbrock" ? "Height shows log(1 + f) so the narrow valley stays visible." : obj.notes}
    />
  );
}
