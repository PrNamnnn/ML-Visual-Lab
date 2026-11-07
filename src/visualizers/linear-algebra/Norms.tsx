"use client";
import { useState } from "react";
import { Plot, Handle, C, Polyline, Arrow } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented } from "@/components/controls";
import { normP, norm1, norm2, normInf, manhattan, euclidean, chebyshev, minkowski } from "@/lib/math/linalg";
import { fmt, tex } from "@/lib/utilities/format";

type V = [number, number];
const snap = (p: number[]): V => [Math.round(Math.max(-2.8, Math.min(2.8, p[0])) * 20) / 20, Math.round(Math.max(-2.8, Math.min(2.8, p[1])) * 20) / 20];

/** Points on the Lp unit sphere ‖x‖_p = r in 2D. */
function lpBall(p: number, r = 1): V[] {
  const pts: V[] = [];
  for (let i = 0; i <= 360; i++) {
    const t = (i / 360) * 2 * Math.PI;
    const c = Math.cos(t), s = Math.sin(t);
    const n = p === Infinity ? Math.max(Math.abs(c), Math.abs(s)) : Math.pow(Math.pow(Math.abs(c), p) + Math.pow(Math.abs(s), p), 1 / p);
    pts.push([(r * c) / n, (r * s) / n]);
  }
  return pts;
}

export default function Norms({ mode = "unitball" }: { mode?: "unitball" | "distance" }) {
  const [p, setP] = useState(2);
  const [pInf, setPInf] = useState(false);
  const [x, setX] = useState<V>([1.2, 0.7]);
  const [A, setA] = useState<V>([-1.5, -1]);
  const [B, setB] = useState<V>([1.5, 1.2]);
  const pe = pInf ? Infinity : p;

  if (mode === "unitball") {
    const r = normP(x, pe);
    return (
      <LabShell
        viz={
          <div className="space-y-2">
            <Plot x={[-2.5, 2.5]} y={[-2.5, 2.5]} equal originAxes aspect={0.75} maxH={500} ariaLabel="Lp unit balls">
              <Polyline pts={lpBall(1)} closed style={{ stroke: C.a("pink", 0.45), strokeWidth: 1 }} />
              <Polyline pts={lpBall(2)} closed style={{ stroke: C.a("blue", 0.45), strokeWidth: 1 }} />
              <Polyline pts={lpBall(Infinity)} closed style={{ stroke: C.a("amber", 0.45), strokeWidth: 1 }} />
              <Polyline pts={lpBall(pe)} closed style={{ stroke: C.fg, strokeWidth: 2.2 }} />
              <Polyline pts={lpBall(pe, r)} closed style={{ stroke: C.green, strokeWidth: 1.2, strokeDasharray: "4 3" }} />
              <Arrow from={[0, 0]} to={x} color="green" label="x" />
              <Handle p={x} onDrag={(q) => setX(snap(q))} color="green" label="point x" />
            </Plot>
            <Legend items={[{ label: "L1 (diamond)", color: C.pink }, { label: "L2 (circle)", color: C.blue }, { label: "L∞ (square)", color: C.amber }, { label: `selected p = ${pInf ? "∞" : fmt(p, 2)}`, color: C.fg }, { label: "level set through x", color: C.green, dashed: true }]} />
          </div>
        }
        controls={
          <>
            <ParameterSlider label="p" tex value={p} min={1} max={8} step={0.05} onChange={(v) => { setP(v); setPInf(false); }} disabled={pInf} />
            <Segmented value={pInf ? "inf" : "p"} onChange={(v) => setPInf(v === "inf")} options={[{ value: "p", label: "finite p" }, { value: "inf", label: "p = ∞" }]} />
            <p className="text-[12px] leading-relaxed text-faint">As p grows, the unit ball fattens from a diamond (p=1) through a circle (p=2) toward a square (p→∞). For p &lt; 1 the shape would be non-convex, which is why it is not a norm.</p>
          </>
        }
        readouts={<Readouts items={[
          { label: "\\|\\mathbf x\\|_1", tex: true, value: fmt(norm1(x)), color: C.pink },
          { label: "\\|\\mathbf x\\|_2", tex: true, value: fmt(norm2(x)), color: C.blue },
          { label: "\\|\\mathbf x\\|_\\infty", tex: true, value: fmt(normInf(x)), color: C.amber },
          { label: `\\|\\mathbf x\\|_{${pInf ? "\\infty" : tex(p, 2)}}`, tex: true, value: fmt(r) },
        ]} />}
        calc={<CalcSteps steps={[
          { tex: `\\|\\mathbf x\\|_1=|${tex(x[0], 2)}|+|${tex(x[1], 2)}|=${tex(norm1(x))}` },
          { tex: `\\|\\mathbf x\\|_2=\\sqrt{${tex(x[0], 2)}^2+${tex(x[1], 2)}^2}=${tex(norm2(x))}` },
          { tex: `\\|\\mathbf x\\|_\\infty=\\max(|${tex(x[0], 2)}|,|${tex(x[1], 2)}|)=${tex(normInf(x))}` },
        ]} />}
      />
    );
  }

  const d1 = manhattan(A, B), d2 = euclidean(A, B), di = chebyshev(A, B), dp = minkowski(A, B, pe);
  const corner: V = [B[0], A[1]];
  const dx = Math.abs(B[0] - A[0]), dy = Math.abs(B[1] - A[1]);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-3, 3]} equal aspect={0.75} maxH={500} ariaLabel="distance metrics between two points">
            <Polyline pts={[A, corner, B]} style={{ stroke: C.pink, strokeWidth: 2.2 }} />
            <Polyline pts={[A, B]} style={{ stroke: C.blue, strokeWidth: 2.2 }} />
            {dx >= dy ? <Polyline pts={[A, [B[0], A[1]]]} style={{ stroke: C.amber, strokeWidth: 4, opacity: 0.6 }} /> : <Polyline pts={[corner, B]} style={{ stroke: C.amber, strokeWidth: 4, opacity: 0.6 }} />}
            <Handle p={A} onDrag={(q) => setA(snap(q))} color="fg" label="point A" />
            <Handle p={B} onDrag={(q) => setB(snap(q))} color="fg" label="point B" />
          </Plot>
          <Legend items={[{ label: `Manhattan ${fmt(d1)}`, color: C.pink }, { label: `Euclidean ${fmt(d2)}`, color: C.blue }, { label: `Chebyshev ${fmt(di)} (largest gap)`, color: C.amber }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="Minkowski p" value={p} min={1} max={8} step={0.05} onChange={(v) => { setP(v); setPInf(false); }} />
          <p className="text-[12px] leading-relaxed text-faint">Drag A and B. Note d∞ ≤ d₂ ≤ d₁ always holds.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "d_1", tex: true, value: fmt(d1), color: C.pink },
        { label: "d_2", tex: true, value: fmt(d2), color: C.blue },
        { label: "d_\\infty", tex: true, value: fmt(di), color: C.amber },
        { label: `d_{${tex(p, 2)}}`, tex: true, value: fmt(dp) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `|\\Delta x|=${tex(dx, 2)},\\quad|\\Delta y|=${tex(dy, 2)}` },
        { tex: `d_1=${tex(dx, 2)}+${tex(dy, 2)}=${tex(d1)},\\quad d_2=\\sqrt{${tex(dx, 2)}^2+${tex(dy, 2)}^2}=${tex(d2)},\\quad d_\\infty=${tex(di)}` },
      ]} />}
    />
  );
}
