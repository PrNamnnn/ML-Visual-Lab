"use client";
import { useState } from "react";
import { Plot, FnPath, Dot, C, Polyline, Handle, usePlot } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle, AnimationControls } from "@/components/controls";
import { functions1D, fnById } from "@/lib/math/calculus";
import { useTicker } from "@/hooks/useSize";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, tex } from "@/lib/utilities/format";

type Mode = "tangent" | "secant" | "second" | "monotonic";

function Shade({ f, pred, color }: { f: (x: number) => number; pred: (x: number) => boolean; color: string }) {
  const { sx, sy, inner } = usePlot();
  const [a, b] = sx.domain();
  const n = 240;
  const rects: React.ReactNode[] = [];
  let start: number | null = null;
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    const ok = i < n && pred(x) && Number.isFinite(f(x));
    if (ok && start === null) start = x;
    if (!ok && start !== null) {
      rects.push(<rect key={i} x={sx(start)} width={sx(x) - sx(start)} y={inner.top} height={inner.bottom - inner.top} style={{ fill: color }} />);
      start = null;
    }
  }
  void sy;
  return <g>{rects}</g>;
}

export default function Derivative({ mode: initial = "tangent", fn = "cubic" }: { mode?: Mode; fn?: string }) {
  const [mode, setMode] = useState<Mode>(initial);
  const [fid, setFid] = useState(fn);
  const F = fnById(fid);
  const [x0, setX0] = useState(F.id === "log" ? 1.5 : 1.2);
  const [h, setH] = useState(1.5);
  const [playing, setPlaying] = useState(false);
  const [showD, setShowD] = useState(initial !== "secant");
  const [dir, setDir] = useState(1);
  const [lo, hi] = F.domain;
  const x = Math.min(hi - 0.05, Math.max(lo + 0.05, x0));

  useTicker(playing, 16, () => {
    if (mode === "secant") setH((v) => { const nv = v * 0.97; if (nv < 0.002) { setPlaying(false); return 0.002; } return nv; });
    else setX0((v) => { let nv = v + dir * (hi - lo) * 0.004; if (nv > hi - 0.1 || nv < lo + 0.1) { setDir(-dir); nv = Math.max(lo + 0.1, Math.min(hi - 0.1, nv)); } return nv; });
  });

  const y = F.f(x);
  const m = F.df(x);
  const m2 = F.d2f(x);
  const xh = x + h;
  const secSlope = (F.f(xh) - y) / h;
  const span = hi - lo;

  const legend = [{ label: F.tex.replace(/\\[a-z]+/g, "").slice(0, 0) || "f(x)", color: C.blue }];
  if (mode === "secant") legend.push({ label: "secant through x and x+h", color: C.pink }, { label: "tangent (limit h → 0)", color: C.green });
  else legend.push({ label: "tangent line", color: C.pink });
  if (showD) legend.push({ label: "f′(x)", color: C.amber });
  if (mode === "second") legend.push({ label: "f″(x)", color: C.green });

  const steps: { tex: string; note?: string }[] = [];
  if (mode === "secant") {
    steps.push({ tex: `\\dfrac{f(x+h)-f(x)}{h}=\\dfrac{${tex(F.f(xh))}-(${tex(y)})}{${tex(h, 4)}}=${tex(secSlope, 5)}` });
    steps.push({ tex: `f'(x)=\\lim_{h\\to0}\\dfrac{f(x+h)-f(x)}{h}=${tex(m, 5)}`, note: `error ${fmt(Math.abs(secSlope - m), 5)}` });
  } else {
    steps.push({ tex: F.dtex });
    steps.push({ tex: `f'(${tex(x)})=${tex(m)}\\;\\Rightarrow\\;\\text{tangent: }y=${tex(y)}+${tex(m)}(x-${tex(x)})` });
    if (mode === "second") steps.push({ tex: `${F.d2tex}\\;\\Rightarrow\\;f''(${tex(x)})=${tex(m2)}`, note: m2 > 1e-9 ? "concave up (curving upward)" : m2 < -1e-9 ? "concave down" : "inflection candidate" });
  }

  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={F.domain} y={F.yRange} aspect={0.62} maxH={480} originAxes ariaLabel={`derivative visualization of ${F.name}`}>
            {mode === "monotonic" && (
              <>
                <Shade f={F.f} pred={(t) => F.df(t) > 0} color={C.a("green", 0.08)} />
                <Shade f={F.f} pred={(t) => F.df(t) < 0} color={C.a("red", 0.08)} />
              </>
            )}
            {showD && <FnPath f={F.df} style={{ stroke: C.amber, strokeWidth: 1.5, strokeDasharray: "5 3" }} />}
            {mode === "second" && <FnPath f={F.d2f} style={{ stroke: C.green, strokeWidth: 1.4, strokeDasharray: "2 3" }} />}
            <FnPath f={F.f} />
            {mode === "secant" ? (
              <>
                <Polyline pts={[[x - span, y - secSlope * span], [x + span, y + secSlope * span]]} style={{ stroke: C.pink, strokeWidth: 1.8 }} />
                <Polyline pts={[[x - span, y - m * span], [x + span, y + m * span]]} style={{ stroke: C.green, strokeWidth: 1, strokeDasharray: "4 3" }} />
                <Polyline pts={[[x, y], [xh, y], [xh, F.f(xh)]]} style={{ stroke: C.muted, strokeWidth: 1, strokeDasharray: "2 2" }} />
                <Dot p={[xh, F.f(xh)]} r={4.5} color={C.pink} />
              </>
            ) : (
              Number.isFinite(m) && <Polyline pts={[[x - span, y - m * span], [x + span, y + m * span]]} style={{ stroke: C.pink, strokeWidth: 1.8 }} />
            )}
            {showD && Number.isFinite(m) && <Dot p={[x, m]} r={3.5} color={C.amber} />}
            <Handle p={[x, y]} onDrag={(q) => setX0(Math.min(hi - 0.05, Math.max(lo + 0.05, q[0])))} color="blue" label="point on curve" />
          </Plot>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
            <span className="text-[12.5px]"><Tex tex={F.tex} /></span>
            <Legend items={legend.slice(1)} />
          </div>
        </div>
      }
      controls={
        <>
          <Select label="Function" value={fid} onChange={(v) => { setFid(v); const nf = fnById(v); setX0(nf.id === "log" ? 1.5 : (nf.domain[0] + nf.domain[1]) / 2 + 0.7); }} options={functions1D.map((g) => ({ value: g.id, label: g.name }))} />
          <Select label="View" value={mode} onChange={setMode} options={[{ value: "tangent", label: "Tangent line" }, { value: "secant", label: "Secant → tangent (limit)" }, { value: "second", label: "First & second derivative" }, { value: "monotonic", label: "Increasing / decreasing" }]} />
          <ParameterSlider label="x" value={x} min={lo + 0.05} max={hi - 0.05} step={0.01} onChange={setX0} />
          {mode === "secant" && <ParameterSlider label="h" value={h} min={0.002} max={3} step={0.001} onChange={setH} format={(v) => v.toFixed(3)} />}
          <Toggle label="Show f′(x) curve" checked={showD} onChange={setShowD} />
          <AnimationControls playing={playing} onPlay={() => { if (mode === "secant" && !playing && h < 0.01) setH(2); setPlaying(!playing); }} onReset={() => { setPlaying(false); setH(1.5); setX0(1.2); }} />
          {F.id === "abs" && <p className="text-[12px] text-amber">|x| has no derivative at x = 0: left slope −1, right slope +1.</p>}
        </>
      }
      readouts={<Readouts items={[
        { label: "f(x)", tex: true, value: fmt(y) },
        { label: "f'(x)", tex: true, value: fmt(m), color: C.amber },
        ...(mode === "secant" ? [{ label: "\\text{secant slope}", tex: true, value: fmt(secSlope, 5), color: C.pink }] : []),
        { label: "f''(x)", tex: true, value: fmt(m2) },
        { label: "\\text{trend}", tex: true, value: !Number.isFinite(m) ? "undefined" : m > 1e-9 ? "increasing" : m < -1e-9 ? "decreasing" : "stationary" },
      ]} />}
      calc={<CalcSteps steps={steps} />}
      note={mode === "secant" ? "Finite-difference slopes approach f′(x) as h → 0; in floating point, extremely small h eventually loses accuracy to round-off, which is why numerical gradients use h ≈ 1e-5 and central differences." : undefined}
    />
  );
}
