"use client";
import { useMemo, useState } from "react";
import { LabShell, CalcSteps, Readouts } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, AnimationControls } from "@/components/controls";
import { useTicker, useSize } from "@/hooks/useSize";
import { C } from "@/components/visualization/Plot";
import { fmt, tex, cn } from "@/lib/utilities/format";

type Op = "input" | "mul" | "add" | "sub" | "square" | "sin" | "sigmoid" | "exp";
interface N { id: string; label: string; op: Op; in: string[]; pos: [number, number] }

const sig = (z: number) => 1 / (1 + Math.exp(-z));
const fwd = (op: Op, a: number[]): number => {
  switch (op) {
    case "mul": return a[0] * a[1];
    case "add": return a[0] + a[1];
    case "sub": return a[0] - a[1];
    case "square": return a[0] * a[0];
    case "sin": return Math.sin(a[0]);
    case "sigmoid": return sig(a[0]);
    case "exp": return Math.exp(a[0]);
    default: return 0;
  }
};
/** ∂out/∂input_i */
const local = (op: Op, a: number[], out: number): number[] => {
  switch (op) {
    case "mul": return [a[1], a[0]];
    case "add": return [1, 1];
    case "sub": return [1, -1];
    case "square": return [2 * a[0]];
    case "sin": return [Math.cos(a[0])];
    case "sigmoid": return [out * (1 - out)];
    case "exp": return [out];
    default: return [];
  }
};
const localTex: Record<Op, (a: string[]) => string[]> = {
  mul: (a) => [a[1], a[0]], add: () => ["1", "1"], sub: () => ["1", "-1"], square: (a) => [`2${a[0]}`], sin: (a) => [`\\cos ${a[0]}`],
  sigmoid: () => ["\\sigma(1-\\sigma)"], exp: () => ["e^{(\\cdot)}"], input: () => [],
};

const PRESETS: Record<string, { title: string; tex: string; nodes: N[]; inputs: Record<string, { v: number; min: number; max: number }> }> = {
  neuron: {
    title: "Neuron + squared loss",
    tex: "L=(\\sigma(wx+b)-t)^2",
    inputs: { w: { v: 0.8, min: -3, max: 3 }, x: { v: 1.5, min: -3, max: 3 }, b: { v: -0.4, min: -3, max: 3 }, t: { v: 1, min: 0, max: 1 } },
    nodes: [
      { id: "w", label: "w", op: "input", in: [], pos: [0.06, 0.15] },
      { id: "x", label: "x", op: "input", in: [], pos: [0.06, 0.42] },
      { id: "b", label: "b", op: "input", in: [], pos: [0.06, 0.7] },
      { id: "t", label: "t", op: "input", in: [], pos: [0.55, 0.9] },
      { id: "m", label: "×", op: "mul", in: ["w", "x"], pos: [0.27, 0.28] },
      { id: "z", label: "+", op: "add", in: ["m", "b"], pos: [0.45, 0.45] },
      { id: "a", label: "σ", op: "sigmoid", in: ["z"], pos: [0.62, 0.45] },
      { id: "e", label: "−", op: "sub", in: ["a", "t"], pos: [0.78, 0.6] },
      { id: "L", label: "²", op: "square", in: ["e"], pos: [0.94, 0.6] },
    ],
  },
  chain: {
    title: "Composition sin(x²)",
    tex: "f(x)=\\sin(x^2)",
    inputs: { x: { v: 1.2, min: -2.5, max: 2.5 } },
    nodes: [
      { id: "x", label: "x", op: "input", in: [], pos: [0.1, 0.5] },
      { id: "u", label: "²", op: "square", in: ["x"], pos: [0.5, 0.5] },
      { id: "f", label: "sin", op: "sin", in: ["u"], pos: [0.9, 0.5] },
    ],
  },
  multichain: {
    title: "z = x·y with x = t², y = sin t",
    tex: String.raw`z=x\,y,\ x=t^2,\ y=\sin t`,
    inputs: { t: { v: 1.2, min: -3, max: 3 } },
    nodes: [
      { id: "t", label: "t", op: "input", in: [], pos: [0.08, 0.5] },
      { id: "x", label: "²", op: "square", in: ["t"], pos: [0.45, 0.22] },
      { id: "y", label: "sin", op: "sin", in: ["t"], pos: [0.45, 0.78] },
      { id: "z", label: "×", op: "mul", in: ["x", "y"], pos: [0.9, 0.5] },
    ],
  },
  product: {
    title: "f = (x·y + z)²",
    tex: "f=(xy+z)^2",
    inputs: { x: { v: 2, min: -3, max: 3 }, y: { v: -1, min: -3, max: 3 }, z: { v: 3, min: -3, max: 3 } },
    nodes: [
      { id: "x", label: "x", op: "input", in: [], pos: [0.08, 0.2] },
      { id: "y", label: "y", op: "input", in: [], pos: [0.08, 0.5] },
      { id: "z", label: "z", op: "input", in: [], pos: [0.08, 0.82] },
      { id: "q", label: "×", op: "mul", in: ["x", "y"], pos: [0.38, 0.35] },
      { id: "s", label: "+", op: "add", in: ["q", "z"], pos: [0.64, 0.55] },
      { id: "f", label: "²", op: "square", in: ["s"], pos: [0.92, 0.55] },
    ],
  },
};

export default function CompGraph({ preset: initial = "neuron" }: { preset?: string }) {
  const [preset, setPreset] = useState(initial);
  const P = PRESETS[preset];
  const [vals, setVals] = useState<Record<string, number>>(Object.fromEntries(Object.entries(P.inputs).map(([k, v]) => [k, v.v])));
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const { ref, width } = useSize<HTMLDivElement>(0.45);
  const H = 300;

  const comp = useMemo(() => {
    const val: Record<string, number> = {};
    for (const n of P.nodes) val[n.id] = n.op === "input" ? vals[n.id] ?? 0 : fwd(n.op, n.in.map((i) => val[i]));
    const out = P.nodes[P.nodes.length - 1].id;
    const grad: Record<string, number> = Object.fromEntries(P.nodes.map((n) => [n.id, 0]));
    grad[out] = 1;
    const edgeGrad: Record<string, number> = {};
    for (const n of [...P.nodes].reverse()) {
      if (n.op === "input") continue;
      const lg = local(n.op, n.in.map((i) => val[i]), val[n.id]);
      n.in.forEach((i, k) => {
        edgeGrad[`${i}>${n.id}`] = lg[k];
        grad[i] += grad[n.id] * lg[k];
      });
    }
    return { val, grad, edgeGrad, out };
  }, [P, vals]);

  const order = P.nodes.filter((n) => n.op !== "input");
  const total = 1 + order.length * 2; // 0 = inputs only, then forward, then backward
  useTicker(playing, 900, () => setStep((s) => { if (s + 1 >= total) { setPlaying(false); return total - 1; } return s + 1; }));
  const fwdDone = Math.min(order.length, step);
  const bwdDone = Math.max(0, step - order.length);
  const visibleVal = (id: string) => P.nodes.find((n) => n.id === id)!.op === "input" || order.findIndex((n) => n.id === id) < fwdDone;
  const revOrder = [...order].reverse();
  const gradShown = (id: string) => {
    if (bwdDone === 0) return false;
    if (id === comp.out) return true;
    // a node's gradient is final once all its consumers have been processed
    const consumers = P.nodes.filter((n) => n.in.includes(id));
    return consumers.every((c) => revOrder.findIndex((n) => n.id === c.id) < bwdDone);
  };

  const W = width;
  const px = (n: N) => [24 + n.pos[0] * (W - 48), 20 + n.pos[1] * (H - 40)] as const;
  const r = 22;
  const steps: { tex: string; note?: string }[] = [];
  const current = step === 0 ? null : step <= order.length ? order[step - 1] : revOrder[step - order.length - 1];
  if (current) {
    const ins = current.in.map((i) => P.nodes.find((n) => n.id === i)!.label);
    if (step <= order.length) steps.push({ tex: `\\text{forward: }${current.label === "²" ? `(${tex(comp.val[current.in[0]])})^2` : current.op === "sigmoid" ? `\\sigma(${tex(comp.val[current.in[0]])})` : current.op === "sin" ? `\\sin(${tex(comp.val[current.in[0]])})` : `${tex(comp.val[current.in[0]])}\\ ${current.label === "×" ? "\\times" : current.label}\\ ${tex(comp.val[current.in[1]])}`}=${tex(comp.val[current.id])}` });
    else {
      const lt = localTex[current.op](ins.map((s) => (s.length > 1 ? `(${s})` : s)));
      current.in.forEach((i, k) => {
        const lab = P.nodes.find((n) => n.id === i)!.label;
        steps.push({ tex: `\\frac{\\partial ${P.nodes[P.nodes.length - 1].label === "²" ? "L" : "f"}}{\\partial(${lab})}\\mathrel{+}=\\underbrace{${tex(comp.grad[current.id])}}_{\\text{upstream}}\\cdot\\underbrace{${tex(comp.edgeGrad[`${i}>${current.id}`])}}_{${lt[k] ?? ""}}`, note: "chain rule: upstream × local" });
      });
    }
  }

  return (
    <LabShell
      viz={
        <div ref={ref} className="w-full">
          <svg width={W} height={H} role="img" aria-label={`computational graph for ${P.title}`}>
            {P.nodes.flatMap((n) =>
              n.in.map((i) => {
                const a = P.nodes.find((m) => m.id === i)!;
                const [x1, y1] = px(a), [x2, y2] = px(n);
                const done = gradShown(i) && bwdDone > 0;
                const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
                return (
                  <g key={`${i}-${n.id}`}>
                    <line x1={x1} y1={y1} x2={x2} y2={y2} style={{ stroke: done ? C.pink : visibleVal(n.id) ? C.blue : C.lineStrong, strokeWidth: done ? 2 : 1.4 }} />
                    {done && <text x={mx} y={my - 6} textAnchor="middle" className="num" style={{ fill: C.pink, fontSize: 10.5 }}>∂ {fmt(comp.grad[n.id] * comp.edgeGrad[`${i}>${n.id}`])}</text>}
                  </g>
                );
              }),
            )}
            {P.nodes.map((n) => {
              const [x, y] = px(n);
              const isCur = current?.id === n.id;
              return (
                <g key={n.id}>
                  <circle cx={x} cy={y} r={r} style={{ fill: C.panel, stroke: isCur ? C.amber : n.op === "input" ? C.blue : C.lineStrong, strokeWidth: isCur ? 2.5 : 1.5 }} />
                  <text x={x} y={y + 4} textAnchor="middle" style={{ fill: C.fg, fontSize: 13, fontWeight: 600 }}>{n.label}</text>
                  {visibleVal(n.id) && <text x={x} y={y - r - 6} textAnchor="middle" className="num" style={{ fill: C.blue, fontSize: 11 }}>{fmt(comp.val[n.id])}</text>}
                  {gradShown(n.id) && <text x={x} y={y + r + 14} textAnchor="middle" className="num" style={{ fill: C.pink, fontSize: 11 }}>g={fmt(comp.grad[n.id])}</text>}
                </g>
              );
            })}
          </svg>
          <div className={cn("mt-1 flex gap-4 text-[11.5px]")}>
            <span style={{ color: C.blue }}>blue = forward value</span>
            <span style={{ color: C.pink }}>pink = gradient of output w.r.t. node (g)</span>
          </div>
        </div>
      }
      controls={
        <>
          <Select label="Graph" value={preset} onChange={(v) => { setPreset(v); setVals(Object.fromEntries(Object.entries(PRESETS[v].inputs).map(([k, q]) => [k, q.v]))); setStep(0); }} options={Object.entries(PRESETS).map(([k, v]) => ({ value: k, label: v.title }))} />
          {Object.entries(P.inputs).map(([k, q]) => <ParameterSlider key={k} label={k} tex value={vals[k] ?? q.v} min={q.min} max={q.max} step={0.05} onChange={(v) => setVals({ ...vals, [k]: v })} />)}
          <div className="text-[12px] text-muted">Step {step} / {total - 1} — {step === 0 ? "inputs" : step <= order.length ? "forward pass" : "backward pass"}</div>
          <AnimationControls playing={playing} onPlay={() => { if (!playing && step >= total - 1) setStep(0); setPlaying(!playing); }} onStep={() => setStep((s) => Math.min(total - 1, s + 1))} canStep={step < total - 1} onReset={() => { setPlaying(false); setStep(0); }} />
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{output}", tex: true, value: fmt(comp.val[comp.out]) },
        ...Object.keys(P.inputs).map((k) => ({ label: `\\partial/\\partial ${k}`, tex: true, value: step >= total - 1 ? fmt(comp.grad[k], 5) : "…" })),
      ]} />}
      calc={<CalcSteps title={`Current step · ${P.tex}`} steps={steps.length ? steps : [{ tex: P.tex, note: "press Step to start the forward pass" }]} />}
      note="Reverse-mode automatic differentiation (what PyTorch/JAX do): one forward pass stores values, one backward pass multiplies upstream gradients by local derivatives — every input's gradient in a single sweep."
    />
  );
}
