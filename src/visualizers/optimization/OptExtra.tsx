"use client";
import { useMemo, useState } from "react";
import { Plot, FnPath, Dot, C, Polyline, usePlot } from "@/components/visualization/Plot";
import { VLine } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle } from "@/components/controls";
import { objectives } from "@/lib/math/optim";
import { fmt, tex } from "@/lib/utilities/format";

/* ------------------------------ objective functions & constraints ------------------------------ */
// house size (100 m²) vs price (100k): fit price ≈ w · size
const XS = [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.5];
const YS = [1.9, 2.3, 3.1, 3.4, 4.6, 5.1, 6.4, 7.2];

function Band({ lo, hi }: { lo: number; hi: number }) {
  const { sx, inner } = usePlot();
  return <rect x={sx(lo)} width={Math.max(0, sx(hi) - sx(lo))} y={inner.top} height={inner.bottom - inner.top} style={{ fill: C.a("green", 0.08) }} />;
}

export function ObjectiveViz() {
  const [lam, setLam] = useState(0);
  const [useCons, setUseCons] = useState(true);
  const [lo, setLo] = useState(1.5);
  const [hi, setHi] = useState(2.6);
  const n = XS.length;
  const sxx = XS.reduce((s, x) => s + x * x, 0), sxy = XS.reduce((s, x, i) => s + x * YS[i], 0), syy = YS.reduce((s, y) => s + y * y, 0);
  const mse = (w: number) => (syy - 2 * w * sxy + w * w * sxx) / n;
  const J = (w: number) => mse(w) + lam * w * w;
  const wUn = sxy / (sxx + n * lam); // minimiser of J (closed form)
  const wCon = useCons ? Math.min(hi, Math.max(lo, wUn)) : wUn; // J is convex in w → clip to the interval
  const active = useCons && (wUn < lo || wUn > hi);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[0, 5]} y={[0, Math.max(6, J(0) * 1.05)]} aspect={0.45} maxH={320} xLabel="slope w (price per 100 m²)" yLabel="objective J(w)" ariaLabel="objective function with a feasible interval">
            {useCons && <Band lo={lo} hi={hi} />}
            <FnPath f={mse} style={{ stroke: C.a("fg", 0.3), strokeWidth: 1.2, strokeDasharray: "4 3" }} />
            <FnPath f={J} />
            <Dot p={[wUn, J(wUn)]} r={5} color={C.blue} />
            <Dot p={[wCon, J(wCon)]} r={6} color={C.pink} stroke={C.bg} />
          </Plot>
          <Plot x={[0, 3]} y={[0, 8.5]} aspect={0.3} minH={140} maxH={200} ariaLabel="data and fitted line">
            {XS.map((x, i) => <Dot key={i} p={[x, YS[i]]} r={3.5} color={C.fg} />)}
            <Polyline pts={[[0, 0], [3, 3 * wCon]]} style={{ stroke: C.pink, strokeWidth: 2 }} />
          </Plot>
          <Legend items={[{ label: "objective J(w) = MSE + λw²", color: C.blue }, { label: "MSE alone", color: C.a("fg", 0.4), dashed: true }, ...(useCons ? [{ label: "feasible set lo ≤ w ≤ hi", color: C.a("green", 0.6) }] : []), { label: "chosen w (pink) / unconstrained minimum (blue)", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="regularisation λ" tex={false} value={lam} min={0} max={2} step={0.01} onChange={setLam} />
          <Toggle label="Constrain w to an interval" checked={useCons} onChange={setUseCons} />
          {useCons && <><ParameterSlider label="lower bound" value={lo} min={0} max={4} step={0.05} onChange={(v) => setLo(Math.min(v, hi))} /><ParameterSlider label="upper bound" value={hi} min={0.2} max={5} step={0.05} onChange={(v) => setHi(Math.max(v, lo))} /></>}
        </>
      }
      readouts={<Readouts items={[
        { label: "w_{\\text{unconstrained}}", tex: true, value: fmt(wUn, 4), color: C.blue },
        { label: "w^*", tex: true, value: fmt(wCon, 4), color: C.pink },
        { label: "J(w^*)", tex: true, value: fmt(J(wCon), 4) },
        { label: "\\text{constraint}", tex: true, value: !useCons ? "none" : active ? "active (at a bound)" : "inactive" },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`J(w)=\frac1n\sum_i(y_i-wx_i)^2+\lambda w^2,\qquad \frac{dJ}{dw}=0\Rightarrow w=\frac{\sum x_iy_i}{\sum x_i^2+n\lambda}=${tex(wUn, 4)}` },
        { tex: useCons ? String.raw`\min_{${tex(lo, 2)}\le w\le${tex(hi, 2)}}J(w)\Rightarrow w^*=\operatorname{clip}(${tex(wUn, 4)})=${tex(wCon, 4)}` : String.raw`\text{unconstrained: }w^*=${tex(wUn, 4)}`, note: "J is a convex parabola, so the constrained minimum is the nearest feasible point" },
      ]} />}
      note="An optimisation problem = objective (what to minimise: error + penalty) + decision variables (w) + constraints (feasible set). Penalties trade fit for simplicity; constraints forbid regions outright. (Synthetic house-price data.)"
    />
  );
}

/* ------------------------------ backtracking line search ------------------------------ */
export function LineSearchViz() {
  const [oid, setOid] = useState("rosenbrock");
  const obj = objectives.find((o) => o.id === oid)!;
  const [a0, setA0] = useState(1);
  const [beta, setBeta] = useState(0.5);
  const [c, setC] = useState(1e-4);
  const x = obj.start;
  const g = obj.grad(x[0], x[1]);
  const gn = Math.hypot(g[0], g[1]);
  const d: [number, number] = [-g[0] / gn, -g[1] / gn]; // unit steepest-descent direction
  const slope = g[0] * d[0] + g[1] * d[1]; // = −‖g‖
  const f0 = obj.f(x[0], x[1]);
  const phi = (a: number) => obj.f(x[0] + a * d[0], x[1] + a * d[1]);
  const trials = useMemo(() => {
    const out: { a: number; ok: boolean }[] = [];
    let a = a0;
    for (let i = 0; i < 30; i++) {
      const ok = phi(a) <= f0 + c * a * slope;
      out.push({ a, ok });
      if (ok) break;
      a *= beta;
    }
    return out;
  }, [a0, beta, c, oid]); // eslint-disable-line
  const acc = trials[trials.length - 1];
  const amax = Math.max(a0 * 1.1, 0.05);
  const ys = Array.from({ length: 100 }, (_, i) => phi((amax * i) / 99));
  const ylo = Math.min(...ys, f0 + c * amax * slope), yhi = Math.min(Math.max(...ys), f0 * 3 + 5);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, amax]} y={[ylo - 0.05 * Math.abs(yhi - ylo), yhi]} aspect={0.5} maxH={360} xLabel="step size α" yLabel="φ(α) = f(x + α d)" ariaLabel="line search along the descent direction">
            <FnPath f={phi} n={400} />
            <Polyline pts={[[0, f0], [amax, f0 + c * amax * slope]]} style={{ stroke: C.green, strokeWidth: 1.4, strokeDasharray: "5 3" }} />
            <Polyline pts={[[0, f0], [Math.min(amax, f0 / Math.max(1e-9, -slope)), f0 + slope * Math.min(amax, f0 / Math.max(1e-9, -slope))]]} style={{ stroke: C.a("fg", 0.3), strokeWidth: 1 }} />
            {trials.map((t, i) => <Dot key={i} p={[t.a, phi(t.a)]} r={t.ok ? 6 : 4} color={t.ok ? C.green : C.red} stroke={C.bg} />)}
            {acc && <VLine x={acc.a} color={C.green} label="accepted α" />}
          </Plot>
          <Legend items={[{ label: "φ(α) along d = −∇f/‖∇f‖", color: C.blue }, { label: "Armijo line f(x) + c·α·∇f·d", color: C.green, dashed: true }, { label: "rejected trial", color: C.red }, { label: "accepted step", color: C.green }]} />
        </div>
      }
      controls={
        <>
          <Select label="Function" value={oid} onChange={setOid} options={objectives.map((o) => ({ value: o.id, label: o.name }))} />
          <ParameterSlider label="initial step α₀" value={a0} min={0.01} max={3} step={0.01} onChange={setA0} />
          <ParameterSlider label="shrink factor β" value={beta} min={0.1} max={0.9} step={0.05} onChange={setBeta} />
          <ParameterSlider label="sufficient-decrease constant c" value={c} min={0.0001} max={0.9} step={0.0001} onChange={setC} format={(v) => v.toPrecision(2)} />
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{trials}", tex: true, value: trials.length },
        { label: "\\alpha", tex: true, value: acc ? fmt(acc.a, 5) : "—", color: C.green },
        { label: "f(\\mathbf x)", tex: true, value: fmt(f0, 4) },
        { label: "f(\\mathbf x+\\alpha\\mathbf d)", tex: true, value: acc ? fmt(phi(acc.a), 4) : "—" },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{accept }\alpha\text{ if }f(\mathbf x+\alpha\mathbf d)\le f(\mathbf x)+c\,\alpha\,\nabla f^\top\mathbf d,\ \text{else }\alpha\leftarrow\beta\alpha`, note: "Armijo / sufficient-decrease condition" },
        { tex: String.raw`\nabla f^\top\mathbf d=-\|\nabla f\|=${tex(slope, 4)}` },
      ]} />}
      note="Instead of a fixed learning rate, a line search picks the step along a chosen direction by trying a large step and shrinking it until the function decreases enough. BFGS and L-BFGS rely on it."
    />
  );
}
