"use client";
import { useState } from "react";
import { Plot, FnPath, C } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle } from "@/components/controls";
import { fmt, tex } from "@/lib/utilities/format";

type Fam = "linear" | "polynomial" | "exponential" | "logarithmic" | "trigonometric" | "composite" | "inverse";

export default function Functions({ family: initial = "linear" }: { family?: Fam }) {
  const [fam, setFam] = useState<Fam>(initial);
  const [a, setA] = useState(1);
  const [b, setB] = useState(0.5);
  const [c, setC] = useState(0);
  const [showInv, setShowInv] = useState(true);
  let f: (x: number) => number = (x) => x;
  let g: ((x: number) => number) | null = null;
  let texF = "";
  let domain = "\\mathbb R", rng = "\\mathbb R";
  const notes: string[] = [];
  switch (fam) {
    case "linear": f = (x) => a * x + b; texF = `f(x)=${tex(a, 2)}x+${tex(b, 2)}`; rng = a === 0 ? `\\{${tex(b, 2)}\\}` : "\\mathbb R"; notes.push("slope a = rate of change, constant everywhere"); break;
    case "polynomial": f = (x) => a * x ** 3 + b * x ** 2 + c * x; texF = `f(x)=${tex(a, 2)}x^3+${tex(b, 2)}x^2+${tex(c, 2)}x`; rng = a !== 0 ? "\\mathbb R" : b > 0 ? "[\\min,\\infty)" : "(-\\infty,\\max]"; break;
    case "exponential": f = (x) => a * Math.exp(b * x); texF = `f(x)=${tex(a, 2)}e^{${tex(b, 2)}x}`; rng = a > 0 ? "(0,\\infty)" : a < 0 ? "(-\\infty,0)" : "\\{0\\}"; notes.push("growth rate proportional to current value: f′ = b·f"); break;
    case "logarithmic": f = (x) => (x > 0 ? a * Math.log(x) + b : NaN); texF = `f(x)=${tex(a, 2)}\\ln x+${tex(b, 2)}`; domain = "(0,\\infty)"; notes.push("undefined for x ≤ 0 — the curve is not drawn there"); break;
    case "trigonometric": f = (x) => a * Math.sin(b * x + c); texF = `f(x)=${tex(a, 2)}\\sin(${tex(b, 2)}x+${tex(c, 2)})`; rng = `[-${tex(Math.abs(a), 2)},${tex(Math.abs(a), 2)}]`; notes.push(`period 2π/|b| = ${b === 0 ? "∞" : fmt((2 * Math.PI) / Math.abs(b))}`); break;
    case "composite": f = (x) => Math.exp(-((a * x + b) ** 2)); g = (x) => a * x + b; texF = `f(g(x))=e^{-(${tex(a, 2)}x+${tex(b, 2)})^2},\\ g(x)=${tex(a, 2)}x+${tex(b, 2)}`; rng = "(0,1]"; notes.push("inner g shifts/stretches the input before the outer exp(−u²) is applied"); break;
    case "inverse": f = (x) => Math.exp(a * x); g = (x) => (x > 0 && a !== 0 ? Math.log(x) / a : NaN); texF = `f(x)=e^{${tex(a, 2)}x},\\quad f^{-1}(x)=\\tfrac{\\ln x}{${tex(a, 2)}}`; notes.push("the inverse is the mirror image across y = x"); break;
  }
  const params: { label: string; v: number; set: (v: number) => void; min: number; max: number }[] = [
    { label: "a", v: a, set: setA, min: -3, max: 3 },
    ...(fam !== "inverse" ? [{ label: "b", v: b, set: setB, min: -3, max: 3 }] : []),
    ...(fam === "polynomial" || fam === "trigonometric" ? [{ label: "c", v: c, set: setC, min: -3, max: 3 }] : []),
  ];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-5, 5]} y={[-4, 4]} equal aspect={0.62} maxH={460} originAxes ariaLabel={`${fam} function plot`}>
            {fam === "inverse" && <FnPath f={(x) => x} style={{ stroke: C.a("fg", 0.25), strokeWidth: 1, strokeDasharray: "4 3" }} />}
            {g && (fam !== "inverse" || showInv) && <FnPath f={g} style={{ stroke: C.pink, strokeWidth: 1.6, strokeDasharray: fam === "composite" ? "5 3" : undefined }} />}
            <FnPath f={f} n={500} />
          </Plot>
          <Legend items={[{ label: fam === "composite" ? "f(g(x))" : "f(x)", color: C.blue }, ...(g ? [{ label: fam === "composite" ? "inner g(x)" : "f⁻¹(x)", color: C.pink, dashed: fam === "composite" }] : [])]} />
        </div>
      }
      controls={
        <>
          <Select label="Family" value={fam} onChange={setFam} options={[{ value: "linear", label: "Linear" }, { value: "polynomial", label: "Polynomial (cubic)" }, { value: "exponential", label: "Exponential" }, { value: "logarithmic", label: "Logarithmic" }, { value: "trigonometric", label: "Trigonometric" }, { value: "composite", label: "Composite f(g(x))" }, { value: "inverse", label: "Inverse function" }]} />
          {params.map((p) => <ParameterSlider key={p.label} label={p.label} tex value={p.v} min={p.min} max={p.max} step={0.05} onChange={p.set} />)}
          {fam === "inverse" && <Toggle label="Show inverse" checked={showInv} onChange={setShowInv} />}
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{domain}", tex: true, value: domain.replace(/\\mathbb R/, "ℝ").replace(/\\infty/g, "∞") }, { label: "\\text{range}", tex: true, value: rng.replace(/\\mathbb R/, "ℝ").replace(/\\infty/g, "∞").replace(/\\[a-z]+/g, "") }, { label: "f(1)", tex: true, value: fmt(f(1)) }]} />}
      calc={<CalcSteps title="Definition" steps={[{ tex: texF, note: notes.join("; ") }]} />}
    />
  );
}
