"use client";
import { useState } from "react";
import { Plot, Arrow, Handle, C, Polyline, Dot, usePlot } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Presets } from "@/components/controls";
import { det2, solve } from "@/lib/math/linalg";
import { fmt, tex } from "@/lib/utilities/format";

type Mode = "combination" | "independence" | "basis";
type V = [number, number];
const snap = (p: number[]): V => [Math.round(Math.max(-4.5, Math.min(4.5, p[0])) * 10) / 10, Math.round(Math.max(-4.5, Math.min(4.5, p[1])) * 10) / 10];

function SpanRegion({ a, b, dependent }: { a: V; b: V; dependent: boolean }) {
  const { inner } = usePlot();
  if (!dependent) return <rect x={inner.left} y={inner.top} width={inner.right - inner.left} height={inner.bottom - inner.top} style={{ fill: C.a("green", 0.06) }} />;
  const d = Math.hypot(...a) > 1e-9 ? a : b;
  if (Math.hypot(...d) < 1e-9) return null;
  return <Polyline pts={[[-20 * d[0], -20 * d[1]], [20 * d[0], 20 * d[1]]]} style={{ stroke: C.a("green", 0.6), strokeWidth: 6 }} />;
}

function SpanCore({ mode = "combination" }: { mode?: Mode }) {
  const [a, setA] = useState<V>([2, 0.5]);
  const [b, setB] = useState<V>([0.5, 1.5]);
  const [c1, setC1] = useState(1.2);
  const [c2, setC2] = useState(0.8);
  const [x, setX] = useState<V>([2.5, 2.5]);
  const D = det2([[a[0], b[0]], [a[1], b[1]]]);
  const dependent = Math.abs(D) < 1e-6;
  const target: V = [c1 * a[0] + c2 * b[0], c1 * a[1] + c2 * b[1]];
  const coords = !dependent ? solve([[a[0], b[0]], [a[1], b[1]]], x) : null;

  const gridLines: [number, number][][] = [];
  if (mode === "basis" && !dependent) {
    for (let k = -8; k <= 8; k++) {
      gridLines.push([[k * a[0] - 10 * b[0], k * a[1] - 10 * b[1]], [k * a[0] + 10 * b[0], k * a[1] + 10 * b[1]]]);
      gridLines.push([[k * b[0] - 10 * a[0], k * b[1] - 10 * a[1]], [k * b[0] + 10 * a[0], k * b[1] + 10 * a[1]]]);
    }
  }

  const readouts = [
    { label: "\\det[\\mathbf a\\;\\mathbf b]", tex: true, value: fmt(D), color: dependent ? C.red : C.green },
    { label: "\\text{independent?}", tex: true, value: dependent ? "no — dependent" : "yes" },
    { label: "\\dim\\operatorname{span}", tex: true, value: Math.hypot(...a) < 1e-9 && Math.hypot(...b) < 1e-9 ? 0 : dependent ? 1 : 2 },
  ];
  const steps: { tex: string; note?: string }[] = [];
  if (mode === "combination") {
    readouts.push({ label: "c_1\\mathbf a+c_2\\mathbf b", tex: true, value: `(${fmt(target[0])}, ${fmt(target[1])})`, color: C.fg });
    steps.push({ tex: `${tex(c1, 2)}\\,(${tex(a[0], 1)},${tex(a[1], 1)})+${tex(c2, 2)}\\,(${tex(b[0], 1)},${tex(b[1], 1)})=(${tex(target[0])},${tex(target[1])})` });
  } else if (mode === "independence") {
    steps.push({ tex: `\\det\\begin{bmatrix}${tex(a[0], 1)}&${tex(b[0], 1)}\\\\${tex(a[1], 1)}&${tex(b[1], 1)}\\end{bmatrix}=${tex(D)}` });
    steps.push({ tex: dependent ? "c_1\\mathbf a+c_2\\mathbf b=\\mathbf 0\\text{ has a non-trivial solution: span is a line}" : "\\text{only }c_1=c_2=0\\text{ gives }\\mathbf 0:\\text{ span is the whole plane}" });
  } else {
    if (coords) {
      readouts.push({ label: "[\\mathbf x]_B", tex: true, value: `(${fmt(coords[0])}, ${fmt(coords[1])})`, color: C.fg });
      steps.push({ tex: `\\text{solve }\\begin{bmatrix}${tex(a[0], 1)}&${tex(b[0], 1)}\\\\${tex(a[1], 1)}&${tex(b[1], 1)}\\end{bmatrix}\\mathbf c=(${tex(x[0], 1)},${tex(x[1], 1)})\\Rightarrow\\mathbf c=(${tex(coords[0])},${tex(coords[1])})` });
      steps.push({ tex: `\\mathbf x=${tex(coords[0])}\\,\\mathbf a+${tex(coords[1])}\\,\\mathbf b` });
    } else steps.push({ tex: "\\{\\mathbf a,\\mathbf b\\}\\text{ is not a basis: coordinates are not unique (or don't exist)}" });
  }

  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-5, 5]} y={[-5, 5]} equal originAxes aspect={0.75} maxH={520} ariaLabel={`span visualization: ${mode}`}>
            {mode !== "basis" && <SpanRegion a={a} b={b} dependent={dependent} />}
            {gridLines.map((l, i) => <Polyline key={i} pts={l} style={{ stroke: C.a("blue", 0.22), strokeWidth: 1 }} />)}
            {mode === "combination" && (
              <>
                <Arrow from={[0, 0]} to={[c1 * a[0], c1 * a[1]]} color="blue" dashed width={1.6} />
                <Arrow from={[c1 * a[0], c1 * a[1]]} to={target} color="pink" dashed width={1.6} />
                <Dot p={target} r={5} color={C.fg} />
              </>
            )}
            {mode === "basis" && coords && (
              <>
                <Arrow from={[0, 0]} to={[coords[0] * a[0], coords[0] * a[1]]} color="blue" dashed width={1.4} />
                <Arrow from={[coords[0] * a[0], coords[0] * a[1]]} to={x} color="pink" dashed width={1.4} />
                <Handle p={x} onDrag={(q) => setX(snap(q))} color="fg" label="point x" />
              </>
            )}
            <Arrow from={[0, 0]} to={a} color="blue" label="a" width={2.6} />
            <Arrow from={[0, 0]} to={b} color="pink" label="b" width={2.6} />
            <Handle p={a} onDrag={(q) => setA(snap(q))} color="blue" label="vector a" />
            <Handle p={b} onDrag={(q) => setB(snap(q))} color="pink" label="vector b" />
          </Plot>
          <Legend items={mode === "basis" ? [{ label: "grid of the basis {a, b}", color: C.a("blue", 0.5) }] : [{ label: dependent ? "span = a line" : "span = the whole plane (shaded)", color: C.green }]} />
        </div>
      }
      controls={
        <>
          {mode === "combination" && (
            <>
              <ParameterSlider label="c_1" tex value={c1} min={-3} max={3} step={0.05} onChange={setC1} />
              <ParameterSlider label="c_2" tex value={c2} min={-3} max={3} step={0.05} onChange={setC2} />
            </>
          )}
          <Presets
            presets={[
              { label: "Standard basis", value: [[1, 0], [0, 1]] as [V, V] },
              { label: "Skewed basis", value: [[2, 0.5], [0.5, 1.5]] as [V, V] },
              { label: "Dependent", value: [[2, 1], [-1, -0.5]] as [V, V] },
              { label: "Nearly dependent", value: [[2, 1], [2, 1.1]] as [V, V] },
            ]}
            onPick={([p, q]) => { setA(p); setB(q); }}
          />
          <p className="text-[12px] leading-relaxed text-faint">Drag a and b{mode === "basis" ? " and the point x" : ""}.</p>
        </>
      }
      readouts={<Readouts items={readouts} />}
      calc={<CalcSteps steps={steps} />}
    />
  );
}

/* ------------------------------ subspace test ------------------------------ */
/**
 * The set S = { t·d + c·n : t ∈ ℝ } is a line with direction d, shifted by c along the normal n.
 * It is a subspace exactly when it contains 0 (c = 0). We pick two members p, q and a scalar k
 * and test whether p + q and k·p stay inside S.
 */
function SubspaceViz() {
  const [angle, setAngle] = useState(30);
  const [c, setC] = useState(0);
  const [s, setS] = useState(1.2);
  const [t, setT] = useState(-0.6);
  const [k, setK] = useState(1.8);
  const th = (angle * Math.PI) / 180;
  const d: V = [Math.cos(th), Math.sin(th)];
  const nrm: V = [-d[1], d[0]];
  const pt = (u: number): V => [u * d[0] + c * nrm[0], u * d[1] + c * nrm[1]];
  const p = pt(s), q = pt(t);
  const sum: V = [p[0] + q[0], p[1] + q[1]];
  const kp: V = [k * p[0], k * p[1]];
  const dist = (x: V) => x[0] * nrm[0] + x[1] * nrm[1] - c; // signed distance to the line
  const inS = (x: V) => Math.abs(dist(x)) < 1e-9;
  const subspace = Math.abs(c) < 1e-9;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-4, 4]} y={[-4, 4]} equal originAxes aspect={0.75} maxH={480} ariaLabel="subspace closure test">
            <Polyline pts={[pt(-12), pt(12)]} style={{ stroke: subspace ? C.green : C.amber, strokeWidth: 3, opacity: 0.8 }} />
            <Arrow from={[0, 0]} to={p} color="blue" label="p" />
            <Arrow from={[0, 0]} to={q} color="pink" label="q" />
            <Arrow from={[0, 0]} to={sum} color={inS(sum) ? "green" : "red"} label="p+q" dashed />
            <Arrow from={[0, 0]} to={kp} color={inS(kp) ? "green" : "red"} label="k·p" dashed />
            <Dot p={[0, 0]} r={5} color={subspace ? C.green : C.red} />
          </Plot>
          <Legend items={[{ label: subspace ? "S passes through 0 — a subspace" : "S misses 0 — not a subspace", color: subspace ? C.green : C.amber }, { label: "result inside S", color: C.green, dashed: true }, { label: "result leaves S", color: C.red, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="line direction (degrees)" value={angle} min={0} max={179} step={1} onChange={setAngle} />
          <ParameterSlider label="offset c from the origin" value={c} min={-2} max={2} step={0.05} onChange={setC} />
          <ParameterSlider label="p = point at parameter s" value={s} min={-3} max={3} step={0.05} onChange={setS} />
          <ParameterSlider label="q = point at parameter t" value={t} min={-3} max={3} step={0.05} onChange={setT} />
          <ParameterSlider label="scalar k" value={k} min={-2} max={2.5} step={0.05} onChange={setK} />
          <button className="btn w-full" onClick={() => setC(0)}>Shift line through the origin</button>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\mathbf 0\\in S", tex: true, value: subspace ? "yes" : "no", color: subspace ? C.green : C.red },
        { label: "\\mathbf p+\\mathbf q\\in S", tex: true, value: inS(sum) ? "yes" : `no (off by ${fmt(Math.abs(dist(sum)))})`, color: inS(sum) ? C.green : C.red },
        { label: "k\\mathbf p\\in S", tex: true, value: inS(kp) ? "yes" : `no (off by ${fmt(Math.abs(dist(kp)))})`, color: inS(kp) ? C.green : C.red },
        { label: "\dim S", tex: true, value: subspace ? 1 : "— (affine line)" },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`S=\{t\,\mathbf d+${tex(c, 2)}\,\mathbf n\},\quad \mathbf d=(${tex(d[0])},${tex(d[1])})` },
        { tex: String.raw`\mathbf n\cdot(\mathbf p+\mathbf q)=2c=${tex(2 * c, 2)}\ ${subspace ? "=" : "\neq"}\ c=${tex(c, 2)}`, note: "points of S satisfy n·x = c" },
        { tex: String.raw`\mathbf n\cdot(k\mathbf p)=kc=${tex(k * c, 2)}\ ${subspace || Math.abs(k - 1) < 1e-9 ? "=" : "\neq"}\ c`, note: "closure fails unless c = 0" },
      ]} />}
      note="A subspace must contain the zero vector and be closed under addition and scalar multiplication. Lines and planes through the origin qualify; shifted (affine) lines do not."
    />
  );
}

export default function Span(props: { mode?: Mode | "subspace" }) {
  if (props.mode === "subspace") return <SubspaceViz />;
  return <SpanCore mode={props.mode as Mode | undefined} />;
}
