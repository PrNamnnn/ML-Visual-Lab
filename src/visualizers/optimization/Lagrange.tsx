"use client";
import { useMemo, useState } from "react";
import { Plot, Polyline, Arrow, Dot, Handle, C } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented } from "@/components/controls";
import { Contours } from "@/visualizers/calculus/Multivariable";
import { fmt, tex } from "@/lib/utilities/format";

/**
 * Minimise f(x,y) = (x−a)² + (y−b)² subject to g(x,y) = x + y − c = 0  (line),
 * or g = x² + y² − r² = 0 (circle). Closed-form optimum shown alongside a draggable point
 * constrained to the curve, so the user can see ∇f ∥ ∇g at the optimum.
 */
export default function Lagrange({ kind: initialKind = "circle" }: { kind?: "line" | "circle" | "disk" }) {
  const [kind, setKind] = useState<"line" | "circle" | "disk">(initialKind);
  const [a, setA] = useState(2);
  const [b, setB] = useState(1);
  const [c, setC] = useState(1);
  const [r, setR] = useState(1.5);
  const [s, setS] = useState(0.6); // parameter along the constraint
  const f = (x: number, y: number) => (x - a) ** 2 + (y - b) ** 2;
  const levels = useMemo(() => Array.from({ length: 14 }, (_, i) => (i + 0.5) ** 2 * 0.18), []);

  const point: [number, number] = kind === "line" ? [s * 3, c - s * 3] : [r * Math.cos(s * Math.PI), r * Math.sin(s * Math.PI)];
  const gf: [number, number] = [2 * (point[0] - a), 2 * (point[1] - b)];
  const gg: [number, number] = kind === "line" ? [1, 1] : [2 * point[0], 2 * point[1]];
  // optimum
  let opt: [number, number];
  let lambda: number;
  if (kind === "line") {
    // ∇f = λ∇g → 2(x−a) = λ, 2(y−b) = λ → x − a = y − b, x + y = c
    const x = (c + a - b) / 2, y = c - x;
    opt = [x, y];
    lambda = 2 * (x - a);
  } else if (kind === "disk") {
    // inequality x² + y² ≤ r²: KKT — inactive (μ = 0) if the target is inside, else on the boundary
    const n = Math.hypot(a, b);
    if (n <= r) { opt = [a, b]; lambda = 0; } else { opt = [(r * a) / n, (r * b) / n]; lambda = n / r - 1; }
  } else {
    const n = Math.hypot(a, b);
    opt = n > 1e-9 ? [(r * a) / n, (r * b) / n] : [r, 0];
    // 2(x−a) = λ·2x → λ = 1 − a/x
    lambda = n > 1e-9 ? 1 - n / r : 1;
  }
  const cross = gf[0] * gg[1] - gf[1] * gg[0];
  const parallel = Math.abs(cross) / (Math.hypot(...gf) * Math.hypot(...gg) + 1e-12) < 0.03;
  const curve: [number, number][] = kind === "line" ? [[-6, c + 6], [6, c - 6]] : Array.from({ length: 121 }, (_, i) => [r * Math.cos((i / 120) * 2 * Math.PI), r * Math.sin((i / 120) * 2 * Math.PI)]);
  const sc = 0.35;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3.5, 3.5]} y={[-3.5, 3.5]} equal aspect={0.75} maxH={480} originAxes ariaLabel="Lagrange multiplier geometry">
            <Contours f={f} levels={levels} />
            {kind === "disk" && <Polyline pts={curve} closed style={{ stroke: "none", fill: C.a("amber", 0.12) }} />}
            <Polyline pts={curve} style={{ stroke: C.amber, strokeWidth: 2.4 }} />
            <Dot p={[a, b]} r={4} color={C.fg} />
            <Dot p={opt} r={6} color={C.green} />
            <Arrow from={point} to={[point[0] + gf[0] * sc, point[1] + gf[1] * sc]} color="blue" label="∇f" />
            <Arrow from={point} to={[point[0] + gg[0] * sc, point[1] + gg[1] * sc]} color="pink" label="∇g" />
            <Handle p={point} color={parallel ? "green" : "fg"} label="point on the constraint" onDrag={(q) => {
              if (kind === "line") setS(Math.max(-2, Math.min(2, (q[0] - (q[1] - c)) / 6)));
              else { const t = Math.atan2(q[1], q[0]) / Math.PI; setS(t); }
            }} />
          </Plot>
          <Legend items={[{ label: "constraint g = 0", color: C.amber }, { label: "level curves of f", color: C.a("fg", 0.4) }, { label: "constrained optimum", color: C.green }, { label: "∇f", color: C.blue }, { label: "∇g", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <Segmented value={kind} onChange={setKind} options={[{ value: "circle", label: "On circle" }, { value: "disk", label: "Inside disk (≤)" }, { value: "line", label: "On line" }]} />
          <ParameterSlider label="target a (x)" value={a} min={-3} max={3} step={0.05} onChange={setA} />
          <ParameterSlider label="target b (y)" value={b} min={-3} max={3} step={0.05} onChange={setB} />
          {kind === "line" ? <ParameterSlider label="c in x + y = c" value={c} min={-3} max={3} step={0.05} onChange={setC} /> : <ParameterSlider label="radius r" value={r} min={0.3} max={3} step={0.05} onChange={setR} />}
          <p className="text-[12px] leading-relaxed text-faint">Slide the point along the amber constraint. At the optimum the level curve of f is tangent to the constraint, so ∇f and ∇g line up.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "f(\\text{point})", tex: true, value: fmt(f(point[0], point[1])) },
        { label: "\\nabla f\\parallel\\nabla g?", tex: true, value: parallel ? "yes ✓" : "no", color: parallel ? C.green : undefined },
        { label: "\\text{optimum}", tex: true, value: `(${fmt(opt[0])}, ${fmt(opt[1])})` },
        { label: "f^*", tex: true, value: fmt(f(opt[0], opt[1])) },
        { label: kind === "disk" ? "\\mu^*\\ \\text{(KKT)}" : "\\lambda^*", tex: true, value: fmt(lambda) },
        ...(kind === "disk" ? [{ label: "\\text{constraint}", tex: true, value: lambda === 0 ? "inactive (optimum inside)" : "active (on boundary)" }] : []),
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `\\min_{x,y}\\ (x-${tex(a)})^2+(y-${tex(b)})^2\\quad\\text{s.t.}\\quad ${kind === "line" ? `x+y=${tex(c)}` : `x^2+y^2=${tex(r * r)}`}` },
        { tex: `\\mathcal L(x,y,\\lambda)=f(x,y)-\\lambda\\,g(x,y),\\qquad\\nabla_{x,y}\\mathcal L=0\\iff\\nabla f=\\lambda\\nabla g` },
        { tex: `(x^*,y^*)=(${tex(opt[0])},${tex(opt[1])}),\\quad\\lambda^*=${tex(lambda)}`, note: "closed form for this problem" },
      ]} />}
      note="With the sign convention 𝓛 = f − λg, λ* measures how fast the optimal value f* changes as the constraint level is relaxed: ∂f*/∂c = λ*."
    />
  );
}
