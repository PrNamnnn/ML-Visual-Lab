"use client";
import { useMemo, useState } from "react";
import { Plot, FnPath, Dot, C, Polyline, Arrow } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend, Warn } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, AnimationControls } from "@/components/controls";
import { useTicker } from "@/hooks/useSize";
import { fmt, tex } from "@/lib/utilities/format";

const FNS = {
  quadratic: { name: "Convex quadratic", tex: "f(x)=x^2", f: (x: number) => x * x, df: (x: number) => 2 * x, dom: [-3, 3] as [number, number], yr: [-0.5, 9] as [number, number], x0: 2.6, L: 2 },
  quartic: { name: "Two minima", tex: "f(x)=x^4-3x^2+x", f: (x: number) => x ** 4 - 3 * x * x + x, df: (x: number) => 4 * x ** 3 - 6 * x + 1, dom: [-2.2, 2.2] as [number, number], yr: [-4, 4] as [number, number], x0: 2, L: NaN },
  flat: { name: "Flat region", tex: "f(x)=\\tfrac{1}{4}(x-1)^4", f: (x: number) => 0.25 * (x - 1) ** 4, df: (x: number) => (x - 1) ** 3, dom: [-1.5, 3] as [number, number], yr: [-0.3, 5] as [number, number], x0: -1, L: NaN },
};

export default function GradientDescent1D() {
  const [fid, setFid] = useState<keyof typeof FNS>("quadratic");
  const F = FNS[fid];
  const [lr, setLr] = useState(0.1);
  const [x0, setX0] = useState(F.x0);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const path = useMemo(() => {
    const p = [x0];
    for (let i = 0; i < 80; i++) {
      const x = p[p.length - 1];
      const nx = x - lr * F.df(x);
      if (!Number.isFinite(nx) || Math.abs(nx) > 1e6) break;
      p.push(nx);
    }
    return p;
  }, [x0, lr, F]);
  const diverged = path.length < 81 || Math.abs(path[path.length - 1]) > 50;
  const kk = Math.min(k, path.length - 1);
  useTicker(playing, 380, () => setK((s) => { if (s + 1 >= path.length) { setPlaying(false); return s; } return s + 1; }));
  const x = path[kk];
  const g = F.df(x);
  const visible = path.slice(0, kk + 1);
  const regime = fid === "quadratic" ? (lr < 0.4975 ? "monotone convergence (η < 0.5)" : lr < 0.5025 ? "one-step convergence (η = 0.5)" : lr < 0.9975 ? "oscillating convergence" : lr < 1.0025 ? "oscillates forever (η = 1)" : "divergence (η > 1)") : "";
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={F.dom} y={F.yr} aspect={0.58} maxH={440} originAxes ariaLabel="gradient descent on a one-dimensional function">
            <FnPath f={F.f} />
            {visible.slice(1).map((xi, i) => (
              <Polyline key={i} pts={[[visible[i], F.f(visible[i])], [xi, F.f(xi)]]} style={{ stroke: C.a("pink", 0.75), strokeWidth: 1.4 }} />
            ))}
            {visible.map((xi, i) => <Dot key={i} p={[xi, F.f(xi)]} r={i === kk ? 6 : 3} color={i === kk ? C.pink : C.a("pink", 0.6)} />)}
            {Number.isFinite(g) && Math.abs(x) < 10 && <Arrow from={[x, F.f(x)]} to={[x - Math.sign(g) * Math.min(1, Math.abs(lr * g)), F.f(x)]} color="green" width={2} />}
          </Plot>
          <Legend items={[{ label: "θ path", color: C.pink }, { label: "step −η f′(θ) (clipped for display)", color: C.green }]} />
        </div>
      }
      controls={
        <>
          <Select label="Function" value={fid} onChange={(v) => { setFid(v); setX0(FNS[v].x0); setK(0); }} options={Object.entries(FNS).map(([k2, v]) => ({ value: k2 as keyof typeof FNS, label: v.name }))} />
          <ParameterSlider label="learning rate η" value={lr} min={0.005} max={1.2} step={0.005} onChange={(v) => { setLr(v); setK(0); }} format={(v) => v.toFixed(3)} />
          <ParameterSlider label="start θ₀" value={x0} min={F.dom[0]} max={F.dom[1]} step={0.05} onChange={(v) => { setX0(v); setK(0); }} />
          <AnimationControls playing={playing} onPlay={() => { if (!playing && kk >= path.length - 1) setK(0); setPlaying(!playing); }} onStep={() => setK((s) => Math.min(path.length - 1, s + 1))} canStep={kk < path.length - 1} onReset={() => { setPlaying(false); setK(0); }} />
          {diverged && <Warn>The iterates are blowing up: η is too large for this curvature.</Warn>}
        </>
      }
      readouts={<Readouts items={[
        { label: "t", tex: true, value: kk },
        { label: "\\theta_t", tex: true, value: fmt(x) },
        { label: "f(\\theta_t)", tex: true, value: fmt(F.f(x)) },
        { label: "f'(\\theta_t)", tex: true, value: fmt(g) },
        ...(fid === "quadratic" ? [{ label: "\\text{regime}", tex: true, value: regime }] : []),
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `${F.tex}` },
        { tex: `\\theta_{${kk + 1}}=\\theta_{${kk}}-\\eta f'(\\theta_{${kk}})=${tex(x)}-${tex(lr)}\\cdot(${tex(g)})=${tex(x - lr * g)}` },
        ...(fid === "quadratic" ? [{ tex: `\\theta_{t+1}=(1-2\\eta)\\theta_t\\Rightarrow\\text{converges iff }|1-2\\eta|<1\\iff0<\\eta<1`, note: "exact analysis for f = x²" }] : []),
      ]} />}
    />
  );
}
