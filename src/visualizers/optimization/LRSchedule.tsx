"use client";
import { useState } from "react";
import { Plot, Polyline, Dot } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider } from "@/components/controls";
import { lrSchedule, type ScheduleId } from "@/lib/math/optim";
import { fmt } from "@/lib/utilities/format";

const S: { id: ScheduleId; name: string; color: string; tex: string }[] = [
  { id: "constant", name: "Constant", color: "#8b9bb4", tex: "\\eta_t=\\eta_0" },
  { id: "step", name: "Step decay", color: "#e6c35c", tex: "\\eta_t=\\eta_0\\,\\gamma^{\\lfloor t/s\\rfloor}" },
  { id: "exponential", name: "Exponential", color: "#f59e5b", tex: "\\eta_t=\\eta_0\\,\\gamma^{t}" },
  { id: "cosine", name: "Cosine annealing", color: "#5aa9ff", tex: "\\eta_t=\\tfrac{\\eta_0}{2}\\left(1+\\cos\\tfrac{\\pi t}{T}\\right)" },
  { id: "warmup-cosine", name: "Warmup + cosine", color: "#ff6fae", tex: "\\eta_t=\\begin{cases}\\eta_0\\frac{t+1}{w}&t<w\\\\\\frac{\\eta_0}{2}\\left(1+\\cos\\frac{\\pi(t-w)}{T-w}\\right)&t\\ge w\\end{cases}" },
  { id: "linear-warmup", name: "Warmup + linear decay", color: "#7ddc7a", tex: "\\eta_t=\\begin{cases}\\eta_0\\frac{t+1}{w}&t<w\\\\\\eta_0\\frac{T-t}{T-w}&t\\ge w\\end{cases}" },
];

export default function LRSchedule() {
  const [T, setT] = useState(100);
  const [base, setBase] = useState(1e-3);
  const [warm, setWarm] = useState(10);
  const [gamma, setGamma] = useState(0.5);
  const [on, setOn] = useState<ScheduleId[]>(["constant", "step", "cosine", "warmup-cosine"]);
  const [probe, setProbe] = useState(30);
  const opts = { warmup: warm, gamma, stepEvery: Math.max(1, Math.round(T / 4)) };
  const optsExp = { warmup: warm, gamma: Math.pow(0.05, 1 / T) };
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, T]} y={[0, base * 1.1]} aspect={0.5} maxH={380} xLabel="step t" ariaLabel="learning rate schedules">
            {S.filter((s) => on.includes(s.id)).map((s) => (
              <Polyline key={s.id} pts={Array.from({ length: T + 1 }, (_, t) => [t, lrSchedule(s.id, t, T, base, s.id === "exponential" ? optsExp : opts)])} style={{ stroke: s.color, strokeWidth: 2 }} />
            ))}
            {S.filter((s) => on.includes(s.id)).map((s) => <Dot key={s.id} p={[probe, lrSchedule(s.id, probe, T, base, s.id === "exponential" ? optsExp : opts)]} r={3.5} color={s.color} />)}
          </Plot>
          <Legend items={S.filter((s) => on.includes(s.id)).map((s) => ({ label: s.name, color: s.color }))} />
        </div>
      }
      controls={
        <>
          <div className="space-y-1.5">
            {S.map((s) => (
              <label key={s.id} className="flex cursor-pointer items-center gap-2 text-[12.5px]">
                <input type="checkbox" checked={on.includes(s.id)} onChange={() => setOn(on.includes(s.id) ? on.filter((x) => x !== s.id) : [...on, s.id])} />
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                {s.name}
              </label>
            ))}
          </div>
          <ParameterSlider label="total steps T" value={T} min={20} max={500} step={10} onChange={(v) => { setT(v); setProbe(Math.min(probe, v)); }} />
          <ParameterSlider label="base η₀" value={base} min={1e-4} max={1e-2} step={1e-4} onChange={setBase} format={(v) => v.toExponential(1)} />
          <ParameterSlider label="warmup steps w" value={warm} min={1} max={Math.round(T / 2)} step={1} onChange={setWarm} />
          <ParameterSlider label="step decay γ" value={gamma} min={0.1} max={0.9} step={0.05} onChange={setGamma} />
          <ParameterSlider label="inspect step t" value={probe} min={0} max={T} step={1} onChange={setProbe} />
        </>
      }
      readouts={<Readouts items={S.filter((s) => on.includes(s.id)).map((s) => ({ label: s.name, value: fmt(lrSchedule(s.id, probe, T, base, s.id === "exponential" ? optsExp : opts), 6), color: s.color }))} />}
      calc={<CalcSteps title="Formulas" steps={S.filter((s) => on.includes(s.id)).map((s) => ({ tex: s.tex, note: s.name }))} />}
      note="Warmup avoids huge early updates while Adam's moment estimates are still noisy (standard for Transformers); decay lets the optimiser settle into a minimum. The exponential curve here uses γ chosen so η_T = 0.05·η₀."
    />
  );
}
