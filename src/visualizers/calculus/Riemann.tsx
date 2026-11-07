"use client";
import { useState } from "react";
import { Plot, FnPath, C, usePlot, Polyline } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented, AnimationControls } from "@/components/controls";
import { functions1D, fnById, riemann, simpson, type RiemannRule } from "@/lib/math/calculus";
import { distributionById } from "@/lib/statistics/distributions";
import { useTicker } from "@/hooks/useSize";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, tex } from "@/lib/utilities/format";

function Rects({ rects, rule }: { rects: { x0: number; x1: number; h: number; h1?: number }[]; rule: RiemannRule }) {
  const { sx, sy } = usePlot();
  return (
    <g>
      {rects.map((r, i) => {
        const pos = rule === "trapezoid" ? (r.h + (r.h1 ?? 0)) / 2 >= 0 : r.h >= 0;
        const style = { fill: pos ? C.a("blue", 0.22) : C.a("pink", 0.22), stroke: pos ? C.a("blue", 0.7) : C.a("pink", 0.7), strokeWidth: 0.8 };
        if (rule === "trapezoid") return <path key={i} d={`M${sx(r.x0)},${sy(0)}L${sx(r.x0)},${sy(r.h)}L${sx(r.x1)},${sy(r.h1 ?? 0)}L${sx(r.x1)},${sy(0)}Z`} style={style} />;
        return <rect key={i} x={sx(r.x0)} width={Math.max(0, sx(r.x1) - sx(r.x0))} y={Math.min(sy(r.h), sy(0))} height={Math.abs(sy(r.h) - sy(0))} style={style} />;
      })}
    </g>
  );
}

function AreaFill({ f, a, b }: { f: (x: number) => number; a: number; b: number }) {
  const { sx, sy } = usePlot();
  const n = 200;
  let d = `M${sx(a)},${sy(0)}`;
  for (let i = 0; i <= n; i++) { const x = a + ((b - a) * i) / n; d += `L${sx(x)},${sy(f(x))}`; }
  d += `L${sx(b)},${sy(0)}Z`;
  return <path d={d} style={{ fill: C.a("blue", 0.25) }} />;
}

export default function Riemann({ mode = "sum" }: { mode?: "sum" | "ftc" | "probability" }) {
  const [fid, setFid] = useState(mode === "ftc" ? "quadratic" : "sin");
  const [n, setN] = useState(8);
  const [rule, setRule] = useState<RiemannRule>("midpoint");
  const [playing, setPlaying] = useState(false);
  const F = fnById(fid);
  const [a, setA] = useState(mode === "ftc" ? 0 : 0);
  const [b, setB] = useState(mode === "ftc" ? 3 : 3);
  const [mu, setMu] = useState(0);
  const [sig, setSig] = useState(1);
  const [pa, setPa] = useState(-1);
  const [pb, setPb] = useState(1.5);
  useTicker(playing, 280, () => setN((v) => { if (v >= 200) { setPlaying(false); return 200; } return Math.min(200, Math.ceil(v * 1.35)); }));

  if (mode === "probability") {
    const N = distributionById("normal")!;
    const p = { mu, sigma: sig };
    const lo = Math.min(pa, pb), hi = Math.max(pa, pb);
    const exact = N.cdf(hi, p) - N.cdf(lo, p);
    const R = riemann((x) => N.density(x, p), lo, hi, n, "midpoint");
    return (
      <LabShell
        viz={
          <Plot x={[-5, 5]} y={[-0.03, Math.min(1.5, 0.45 / sig + 0.05)]} aspect={0.55} maxH={420} ariaLabel="probability as area under a density">
            <AreaFill f={(x) => N.density(x, p)} a={lo} b={hi} />
            <Rects rects={R.rects} rule="midpoint" />
            <FnPath f={(x) => N.density(x, p)} />
          </Plot>
        }
        controls={
          <>
            <ParameterSlider label="μ" tex value={mu} min={-2} max={2} step={0.1} onChange={setMu} />
            <ParameterSlider label="σ" tex value={sig} min={0.4} max={2} step={0.05} onChange={setSig} />
            <ParameterSlider label="a" tex value={pa} min={-4.5} max={4.5} step={0.05} onChange={setPa} />
            <ParameterSlider label="b" tex value={pb} min={-4.5} max={4.5} step={0.05} onChange={setPb} />
            <ParameterSlider label="rectangles n" value={n} min={1} max={200} step={1} onChange={setN} />
          </>
        }
        readouts={<Readouts items={[{ label: "P(a\\le X\\le b)\\ \\text{exact}", tex: true, value: fmt(exact, 5) }, { label: "\\text{Riemann }(n)", tex: true, value: fmt(R.sum, 5) }, { label: "\\text{error}", tex: true, value: fmt(Math.abs(R.sum - exact), 6) }]} />}
        calc={<CalcSteps steps={[{ tex: `P(${tex(lo)}\\le X\\le ${tex(hi)})=\\int_{${tex(lo)}}^{${tex(hi)}}f(x)\\,dx=\\Phi\\!\\left(\\tfrac{${tex(hi)}-${tex(mu)}}{${tex(sig)}}\\right)-\\Phi\\!\\left(\\tfrac{${tex(lo)}-${tex(mu)}}{${tex(sig)}}\\right)=${tex(exact, 5)}` }]} />}
        note="The normal density has no elementary antiderivative; Φ is computed with a high-accuracy series for the error function."
      />
    );
  }

  const R = riemann(F.f, a, b, n, rule);
  const exact = F.F ? F.F(b) - F.F(a) : simpson(F.f, a, b, 2000);
  const lo = F.domain[0], hi = F.domain[1];
  const valid = !F.defined || (F.defined(a) && F.defined(b));
  // accumulation function for FTC
  const acc = (x: number) => (F.F ? F.F(x) - F.F(a) : simpson(F.f, a, x, 200));
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={F.domain} y={F.yRange} aspect={mode === "ftc" ? 0.42 : 0.6} maxH={440} originAxes ariaLabel="Riemann sum rectangles under a curve">
            {valid && <Rects rects={R.rects} rule={rule} />}
            <FnPath f={F.f} />
            <Polyline pts={[[a, F.yRange[0]], [a, F.yRange[1]]]} style={{ stroke: C.faint, strokeDasharray: "3 3", strokeWidth: 1 }} />
            <Polyline pts={[[b, F.yRange[0]], [b, F.yRange[1]]]} style={{ stroke: C.faint, strokeDasharray: "3 3", strokeWidth: 1 }} />
          </Plot>
          {mode === "ftc" && (
            <div>
              <div className="mb-1 text-[12px] text-muted"><Tex tex={`A(x)=\\int_{${tex(a)}}^{x}f(t)\\,dt\\quad\\text{and}\\quad A'(x)=f(x)`} /></div>
              <Plot x={F.domain} y={[F.yRange[0] * 2, F.yRange[1] * 2]} aspect={0.32} maxH={240} originAxes ariaLabel="accumulation function">
                <FnPath f={(x) => (x >= a ? acc(x) : NaN)} style={{ stroke: C.green }} />
                <FnPath f={F.f} style={{ stroke: C.a("blue", 0.35), strokeWidth: 1 }} />
              </Plot>
            </div>
          )}
        </div>
      }
      controls={
        <>
          <Select label="Function" value={fid} onChange={(v) => { setFid(v); const g = fnById(v); setA(Math.max(g.domain[0], g.id === "log" ? 0.5 : 0)); setB(Math.min(g.domain[1], 2)); }} options={functions1D.map((g) => ({ value: g.id, label: g.name }))} />
          <Segmented value={rule} onChange={setRule} options={[{ value: "left", label: "Left" }, { value: "midpoint", label: "Mid" }, { value: "right", label: "Right" }, { value: "trapezoid", label: "Trap" }]} />
          <ParameterSlider label="rectangles n" value={n} min={1} max={200} step={1} onChange={setN} />
          <ParameterSlider label="lower limit a" value={a} min={lo} max={hi} step={0.05} onChange={setA} />
          <ParameterSlider label="upper limit b" value={b} min={lo} max={hi} step={0.05} onChange={setB} />
          <AnimationControls playing={playing} onPlay={() => { if (!playing && n >= 200) setN(1); setPlaying(!playing); }} onReset={() => { setPlaying(false); setN(4); }} />
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{Riemann sum}", tex: true, value: valid ? fmt(R.sum, 6) : "undefined" },
        { label: `\\int_a^b f\\,dx\\ ${F.F ? "(exact)" : "(Simpson)"}`, tex: true, value: valid ? fmt(exact, 6) : "undefined" },
        { label: "\\text{error}", tex: true, value: valid ? fmt(Math.abs(R.sum - exact), 6) : "—" },
        { label: "\\Delta x", tex: true, value: fmt(R.width, 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `\\sum_{i=1}^{${n}}f(x_i^*)\\,\\Delta x,\\quad\\Delta x=\\tfrac{${tex(b)}-${tex(a)}}{${n}}=${tex(R.width, 4)}` },
        F.F ? { tex: `\\int_{${tex(a)}}^{${tex(b)}}f(x)\\,dx=F(${tex(b)})-F(${tex(a)})=${tex(exact, 5)}`, note: "fundamental theorem of calculus" } : { tex: `\\int\\approx${tex(exact, 5)}\\ \\text{(Simpson, n=2000)}` },
      ]} />}
      note={!valid ? "The interval includes points where f is undefined." : "Blue area counts positively, pink (below the axis) negatively. Midpoint and trapezoid errors shrink like 1/n², left/right like 1/n."}
    />
  );
}
