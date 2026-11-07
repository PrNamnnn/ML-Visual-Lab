"use client";
import { useEffect, useMemo, useState } from "react";
import { Plot, HeatmapCanvas, Polyline, Dot, C, useThemeRGB, Handle } from "@/components/visualization/Plot";
import { LabShell, Readouts } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, AnimationControls, Toggle } from "@/components/controls";
import { objectives, optimizerInfo, runOptimizer, type OptimizerId, type OptimizerConfig } from "@/lib/math/optim";
import { Contours, contourLevels } from "@/visualizers/calculus/Multivariable";
import { useTicker } from "@/hooks/useSize";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, cn } from "@/lib/utilities/format";

const DEFAULT_LR: Record<string, Partial<Record<OptimizerId, number>>> = {
  bowl: { gd: 0.1, momentum: 0.05, nesterov: 0.05, adagrad: 0.8, rmsprop: 0.05, adam: 0.15, adamw: 0.15, newton: 1, bfgs: 1, coord: 0.1 },
  ellipse: { gd: 0.15, momentum: 0.06, nesterov: 0.06, adagrad: 1, rmsprop: 0.06, adam: 0.2, adamw: 0.2, newton: 1, bfgs: 1, coord: 0.15 },
  rosenbrock: { gd: 0.0015, momentum: 0.0008, nesterov: 0.0008, adagrad: 0.5, rmsprop: 0.01, adam: 0.06, adamw: 0.06, newton: 1, bfgs: 1, coord: 0.0015 },
  saddle: { gd: 0.05, momentum: 0.03, nesterov: 0.03, adagrad: 0.5, rmsprop: 0.02, adam: 0.08, adamw: 0.08, newton: 1, bfgs: 1, coord: 0.05 },
  multimodal: { gd: 0.1, momentum: 0.05, nesterov: 0.05, adagrad: 0.6, rmsprop: 0.03, adam: 0.1, adamw: 0.1, newton: 1, bfgs: 1, coord: 0.1 },
};

export default function OptimizerRace({ optimizers = ["gd", "momentum", "adam"], objective = "ellipse", lab = false, clipping = false }: { optimizers?: OptimizerId[]; objective?: string; lab?: boolean; clipping?: boolean }) {
  const [oid, setOid] = useState(objective);
  const obj = objectives.find((o) => o.id === oid) ?? objectives[0];
  const [active, setActive] = useState<OptimizerId[]>(optimizers);
  const [lrs, setLrs] = useState<Record<OptimizerId, number>>({ ...(DEFAULT_LR[oid] as Record<OptimizerId, number>) });
  const [beta1, setBeta1] = useState(0.9);
  const [clip, setClip] = useState(clipping ? 2 : 0);
  const [start, setStart] = useState<[number, number]>(obj.start);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [wd, setWd] = useState(0.1);
  const N = 300;
  const th = useThemeRGB();

  useEffect(() => { setLrs({ ...(DEFAULT_LR[oid] as Record<OptimizerId, number>) }); setStart(obj.start); setT(0); }, [oid]); // eslint-disable-line

  const runs = useMemo(
    () => active.map((id) => {
      const cfg: OptimizerConfig = { id, lr: lrs[id] ?? 0.05, beta1, clip, weightDecay: wd };
      return { id, ...runOptimizer(obj, cfg, start, N) };
    }),
    [active, lrs, beta1, clip, obj, start, wd],
  );
  useTicker(playing, 30, () => setT((v) => { if (v >= N) { setPlaying(false); return N; } return v + 2; }));

  const [[x0, x1], [y0, y1]] = obj.domain;
  const fv = obj.id === "rosenbrock" ? (x: number, y: number) => Math.log1p(obj.f(x, y)) : obj.f;
  const levels = useMemo(() => contourLevels(fv, Math.min(x0, y0), Math.max(x1, y1), 16), [obj]); // eslint-disable-line
  const lo = levels[0], hi = levels[levels.length - 1];
  const all: OptimizerId[] = ["gd", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamw", "newton", "bfgs", "coord"];

  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <Plot x={[x0, x1]} y={[y0, y1]} equal aspect={0.7} maxH={500} ariaLabel={`optimizer trajectories on ${obj.name}`}
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={4} deps={[oid, th.panel[0]]} f={(x, y) => {
              const v = Math.max(0, Math.min(1, (fv(x, y) - lo) / (hi - lo || 1)));
              const k = 1 - v;
              return [th.panel[0] + (th.blue[0] - th.panel[0]) * k * 0.35, th.panel[1] + (th.blue[1] - th.panel[1]) * k * 0.35, th.panel[2] + (th.blue[2] - th.panel[2]) * k * 0.35, 255];
            }} />}>
            <Contours f={fv} levels={levels} />
            {obj.minimum && <Dot p={obj.minimum} r={4} color={C.fg} />}
            {runs.map((r) => {
              const pts = r.path.slice(0, Math.min(r.path.length, t + 1));
              const last = pts[pts.length - 1];
              return (
                <g key={r.id}>
                  <Polyline pts={pts} style={{ stroke: optimizerInfo[r.id].color, strokeWidth: 1.8 }} />
                  {last && <Dot p={last} r={4.5} color={optimizerInfo[r.id].color} stroke={C.bg} />}
                </g>
              );
            })}
            <Handle p={start} onDrag={(q) => { setStart([Math.max(x0, Math.min(x1, q[0])), Math.max(y0, Math.min(y1, q[1]))]); setT(0); }} color="fg" r={6} label="starting point" />
          </Plot>
          <LossCurves runs={runs} t={t} f={obj.f} log={!["saddle","multimodal"].includes(obj.id)} />
        </div>
      }
      controls={
        <>
          <Select label="Loss landscape" value={oid} onChange={setOid} options={objectives.map((o) => ({ value: o.id, label: o.name }))} />
          <div className="space-y-2">
            <div className="label">Optimizers and learning rates</div>
            {(lab ? all : optimizers).map((id) => {
              const on = active.includes(id);
              return (
                <div key={id} className="space-y-1">
                  <label className="flex cursor-pointer items-center gap-2 text-[12.5px]">
                    <input type="checkbox" checked={on} onChange={() => setActive(on ? active.filter((a) => a !== id) : [...active, id])} className="accent-[rgb(var(--blue))]" />
                    <span className="inline-block h-2 w-2 rounded-full" style={{ background: optimizerInfo[id].color }} />
                    <span className={cn(on ? "text-fg" : "text-faint")}>{optimizerInfo[id].name}</span>
                  </label>
                  {on && id !== "newton" && id !== "bfgs" && <ParameterSlider label="η" tex value={lrs[id] ?? 0.05} min={0.0002} max={id === "adagrad" ? 2 : 0.5} step={0.0002} onChange={(v) => { setLrs({ ...lrs, [id]: v }); setT(0); }} format={(v) => v.toFixed(4)} />}
                  {on && (id === "newton" || id === "bfgs") && <ParameterSlider label={id === "bfgs" ? "initial step (backtracking)" : "damping (step size)"} value={lrs[id] ?? 1} min={0.05} max={1} step={0.05} onChange={(v) => { setLrs({ ...lrs, [id]: v }); setT(0); }} />}
                </div>
              );
            })}
          </div>
          {active.some((a) => ["momentum", "nesterov", "adam", "adamw"].includes(a)) && <ParameterSlider label="β₁ (momentum)" value={beta1} min={0} max={0.99} step={0.01} onChange={(v) => { setBeta1(v); setT(0); }} />}
          {active.includes("adamw") && <ParameterSlider label="weight decay λ (AdamW)" value={wd} min={0} max={0.5} step={0.01} onChange={(v) => { setWd(v); setT(0); }} />}
          <Toggle label="Gradient clipping" checked={clip > 0} onChange={(v) => { setClip(v ? 2 : 0); setT(0); }} />
          {clip > 0 && <ParameterSlider label="clip norm c" value={clip} min={0.1} max={10} step={0.1} onChange={(v) => { setClip(v); setT(0); }} />}
          <AnimationControls playing={playing} onPlay={() => { if (!playing && t >= N) setT(0); setPlaying(!playing); }} onStep={() => setT((v) => Math.min(N, v + 1))} canStep={t < N} onReset={() => { setPlaying(false); setT(0); }} />
        </>
      }
      readouts={
        <Readouts items={runs.map((r) => {
          const i = Math.min(r.path.length - 1, t);
          const p = r.path[i];
          return { label: optimizerInfo[r.id].name, value: r.diverged && t >= r.path.length - 1 ? "diverged" : `f = ${fmt(obj.f(p[0], p[1]))}`, color: optimizerInfo[r.id].color };
        })} />
      }
      calc={
        <div className="space-y-2">
          <div className="label">Update rules ({obj.name}: <Tex tex={obj.tex} />)</div>
          <div className="grid gap-1.5 md:grid-cols-2">
            {runs.map((r) => (
              <div key={r.id} className="flex items-baseline gap-2 overflow-x-auto text-[12.5px]">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: optimizerInfo[r.id].color }} />
                <Tex tex={optimizerInfo[r.id].tex} />
              </div>
            ))}
          </div>
          {clip > 0 && <div className="text-[12.5px]"><Tex tex={`\\text{clipping: }g\\leftarrow g\\cdot\\min\\!\\left(1,\\tfrac{${clip}}{\\|g\\|}\\right)`} /></div>}
        </div>
      }
      note={obj.notes + " Drag the white handle to change the starting point. Each optimizer uses its own learning rate — comparisons are only meaningful when each is reasonably tuned."}
    />
  );
}

function LossCurves({ runs, t, f, log }: { runs: { id: OptimizerId; path: [number, number][] }[]; t: number; f: (x: number, y: number) => number; log: boolean }) {
  const tr = (v: number) => (log ? Math.log10(Math.max(1e-12, v)) : v);
  const series = runs.map((r) => ({ id: r.id, ys: r.path.slice(0, t + 1).map((p) => tr(f(p[0], p[1]))) }));
  const all = series.flatMap((s) => s.ys).filter(Number.isFinite);
  const ylo = log ? Math.min(-4, ...all) : Math.min(0, ...all), yhi = Math.max(log ? 1 : 0.5, ...all);
  return (
    <div>
      <div className="mb-1 text-[11.5px] text-faint">{log ? "log₁₀ f(θₜ)" : "f(θₜ)"} vs iteration t</div>
      <Plot x={[0, 300]} y={[Math.max(-12, ylo), yhi]} aspect={0.22} minH={130} maxH={180} ariaLabel="loss curves" margin={{ l: 36, b: 24 }}>
        {series.map((s) => <Polyline key={s.id} pts={s.ys.map((y, i) => [i, y])} style={{ stroke: optimizerInfo[s.id].color, strokeWidth: 1.5 }} />)}
      </Plot>
    </div>
  );
}
