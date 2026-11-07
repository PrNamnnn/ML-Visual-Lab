"use client";
import { useState } from "react";
import { Plot, FnPath, C } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select } from "@/components/controls";
import { taylorFns, taylorEval, factorial } from "@/lib/math/calculus";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, tex } from "@/lib/utilities/format";

export default function Taylor() {
  const [fid, setFid] = useState("sin");
  const [order, setOrder] = useState(3);
  const [x, setX] = useState(1);
  const T = taylorFns.find((t) => t.id === fid)!;
  const approx = taylorEval(T.dk, order, x);
  const exact = T.f(x);
  const terms = Array.from({ length: order + 1 }, (_, k) => ({ k, c: T.dk(k) / factorial(k) })).filter((t) => Math.abs(t.c) > 1e-15);
  const poly = terms.map(({ k, c }) => `${tex(c, 4)}${k === 0 ? "" : k === 1 ? "x" : `x^{${k}}`}`).join("+").replace(/\+-/g, "-") || "0";
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={T.range} y={T.yRange} aspect={0.58} maxH={460} originAxes ariaLabel={`Taylor approximation of ${T.name}`}>
            <FnPath f={T.f} style={{ stroke: C.blue, strokeWidth: 2.5 }} />
            {order > 0 && <FnPath f={(t) => taylorEval(T.dk, order - 1, t)} style={{ stroke: C.a("pink", 0.3), strokeWidth: 1.2 }} />}
            <FnPath f={(t) => taylorEval(T.dk, order, t)} style={{ stroke: C.pink, strokeWidth: 2 }} />
          </Plot>
          <Legend items={[{ label: T.name, color: C.blue }, { label: `degree-${order} Taylor polynomial at 0`, color: C.pink }, { label: `degree ${order - 1}`, color: C.a("pink", 0.4) }]} />
        </div>
      }
      controls={
        <>
          <Select label="Function" value={fid} onChange={setFid} options={taylorFns.map((t) => ({ value: t.id, label: t.name }))} />
          <ParameterSlider label="order n" value={order} min={0} max={20} step={1} onChange={setOrder} />
          <ParameterSlider label="evaluate at x" value={x} min={T.range[0]} max={T.range[1]} step={0.01} onChange={setX} />
          <div className="text-[12.5px]"><Tex tex={T.tex} /></div>
        </>
      }
      readouts={<Readouts items={[{ label: "f(x)", tex: true, value: fmt(exact, 6) }, { label: "T_n(x)", tex: true, value: fmt(approx, 6) }, { label: "|f-T_n|", tex: true, value: fmt(Math.abs(exact - approx), 6), color: Math.abs(exact - approx) < 1e-3 ? C.green : C.amber }]} />}
      calc={<CalcSteps steps={[{ tex: `T_{${order}}(x)=\\sum_{k=0}^{${order}}\\frac{f^{(k)}(0)}{k!}x^k=${poly}` }]} />}
      note={fid === "log1p" ? "ln(1+x) has radius of convergence 1: beyond x = 1 adding terms makes the approximation worse, not better." : "Near the expansion point the approximation is excellent; far away you need more terms. Gradient descent uses the first-order (linear) Taylor model; Newton's method uses the second-order (quadratic) one."}
    />
  );
}
