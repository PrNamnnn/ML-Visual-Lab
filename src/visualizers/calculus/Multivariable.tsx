"use client";
import { useMemo, useState } from "react";
import { Plot, Arrow, Handle, HeatmapCanvas, C, Polyline, useThemeRGB, usePlot } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle } from "@/components/controls";
import { functions2D } from "@/lib/math/calculus";
import { eigSym } from "@/lib/math/linalg";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, tex, texMatrix } from "@/lib/utilities/format";

type Mode = "gradient" | "directional" | "contour" | "hessian" | "partial" | "differential";

/** Marching-squares contour lines for f on the current plot domain. */
function Contours({ f, levels }: { f: (x: number, y: number) => number; levels: number[] }) {
  const { sx, sy } = usePlot();
  const [x0, x1] = sx.domain();
  const [y0, y1] = sy.domain();
  const N = 70;
  const segs = useMemo(() => {
    const out: [number, number, number, number][] = [];
    const xs = Array.from({ length: N + 1 }, (_, i) => x0 + ((x1 - x0) * i) / N);
    const ys = Array.from({ length: N + 1 }, (_, j) => y0 + ((y1 - y0) * j) / N);
    const v = ys.map((y) => xs.map((x) => f(x, y)));
    for (const L of levels) {
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const a = v[j][i] - L, b = v[j][i + 1] - L, c = v[j + 1][i + 1] - L, d = v[j + 1][i] - L;
        const pts: [number, number][] = [];
        const edge = (p: number, q: number, xa: number, ya: number, xb: number, yb: number) => {
          if ((p > 0) !== (q > 0)) { const t = p / (p - q); pts.push([xa + (xb - xa) * t, ya + (yb - ya) * t]); }
        };
        edge(a, b, xs[i], ys[j], xs[i + 1], ys[j]);
        edge(b, c, xs[i + 1], ys[j], xs[i + 1], ys[j + 1]);
        edge(c, d, xs[i + 1], ys[j + 1], xs[i], ys[j + 1]);
        edge(d, a, xs[i], ys[j + 1], xs[i], ys[j]);
        if (pts.length >= 2) out.push([pts[0][0], pts[0][1], pts[1][0], pts[1][1]]);
        if (pts.length === 4) out.push([pts[2][0], pts[2][1], pts[3][0], pts[3][1]]);
      }
    }
    return out;
  }, [f, levels, x0, x1, y0, y1]);
  return <path d={segs.map((s) => `M${sx(s[0])},${sy(s[1])}L${sx(s[2])},${sy(s[3])}`).join("")} style={{ stroke: C.a("fg", 0.28), strokeWidth: 0.8, fill: "none" }} />;
}

export function contourLevels(f: (x: number, y: number) => number, lo: number, hi: number, k = 14) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i <= 30; i++) for (let j = 0; j <= 30; j++) {
    const v = f(lo + ((hi - lo) * i) / 30, lo + ((hi - lo) * j) / 30);
    if (Number.isFinite(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  }
  return Array.from({ length: k }, (_, i) => mn + ((mx - mn) * (i + 0.5)) / k);
}

export { Contours };

export default function Multivariable({ mode: initial = "gradient", fn = "bowl" }: { mode?: Mode; fn?: string }) {
  const [mode, setMode] = useState<Mode>(initial);
  const [fid, setFid] = useState(fn);
  const F = functions2D.find((g) => g.id === fid) ?? functions2D[0];
  const [p, setP] = useState<[number, number]>([1.2, 0.8]);
  const [theta, setTheta] = useState(30);
  const [field, setField] = useState(mode === "gradient");
  const [ddx, setDdx] = useState(0.4);
  const [ddy, setDdy] = useState(-0.3);
  const th = useThemeRGB();
  const R = F.domain[1];
  const levels = useMemo(() => contourLevels(F.f, -R, R), [F, R]);
  const range = useMemo(() => { const l = levels; return [l[0], l[l.length - 1]]; }, [levels]);
  const g = F.grad(p[0], p[1]);
  const gn = Math.hypot(g[0], g[1]);
  const u: [number, number] = [Math.cos((theta * Math.PI) / 180), Math.sin((theta * Math.PI) / 180)];
  const Du = g[0] * u[0] + g[1] * u[1];
  const H = F.hess(p[0], p[1]);
  const he = eigSym(H);
  const scaleG = gn > 0 ? Math.min(1.5, 1.2 / Math.max(1, gn)) * 1 : 0;

  const fieldArrows: React.ReactNode[] = [];
  if (field) {
    const k = 9;
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      const x = -R + ((2 * R) * (i + 0.5)) / k, y = -R + ((2 * R) * (j + 0.5)) / k;
      const gg = F.grad(x, y);
      const n = Math.hypot(gg[0], gg[1]);
      if (n < 1e-9) continue;
      const len = (0.55 * R) / k * Math.min(1, 0.4 + n / 4);
      fieldArrows.push(<Arrow key={`${i}-${j}`} from={[x, y]} to={[x + (gg[0] / n) * len * 1.6, y + (gg[1] / n) * len * 1.6]} color="muted" width={1} />);
    }
  }

  const steps: { tex: string; note?: string }[] = [{ tex: `${F.gradTex}\\;\\Rightarrow\\;\\nabla f(${tex(p[0], 2)},${tex(p[1], 2)})=(${tex(g[0])},${tex(g[1])})` }];
  if (mode === "partial") steps.push({ tex: `\\dfrac{\\partial f}{\\partial x}=${tex(g[0])}\\ \\text{(slope along }x\\text{, }y\\text{ fixed)},\\quad\\dfrac{\\partial f}{\\partial y}=${tex(g[1])}` });
  if (mode === "directional") steps.push({ tex: `D_{\\mathbf u}f=\\nabla f\\cdot\\mathbf u=(${tex(g[0])})(${tex(u[0])})+(${tex(g[1])})(${tex(u[1])})=${tex(Du)}` }, { tex: `\\max_{\\|\\mathbf u\\|=1}D_{\\mathbf u}f=\\|\\nabla f\\|=${tex(gn)}\\ \\text{(when }\\mathbf u\\parallel\\nabla f)` });
  if (mode === "hessian") steps.push({ tex: `H=${texMatrix(H, 3)},\\quad\\lambda=${tex(he.values[0])},\\ ${tex(he.values[1])}`, note: he.values.every((v) => v > 1e-9) ? "positive definite: curving up in every direction" : he.values.every((v) => v < -1e-9) ? "negative definite: curving down" : he.values.some((v) => v > 1e-9) && he.values.some((v) => v < -1e-9) ? "indefinite: saddle-like" : "degenerate" });
  if (mode === "differential") {
    const actual = F.f(p[0] + ddx, p[1] + ddy) - F.f(p[0], p[1]);
    const lin = g[0] * ddx + g[1] * ddy;
    steps.push({ tex: String.raw`df=f_x\,dx+f_y\,dy=(${tex(g[0])})(${tex(ddx, 2)})+(${tex(g[1])})(${tex(ddy, 2)})=${tex(lin, 4)}` }, { tex: String.raw`\Delta f=f(x+dx,y+dy)-f(x,y)=${tex(actual, 4)},\quad|\Delta f-df|=${tex(Math.abs(actual - lin), 4)}`, note: "the error shrinks like (dx² + dy²) as the step gets smaller" });
  }
  if (mode === "contour") steps.push({ tex: "\\nabla f\\perp\\text{level curve }\\{f(x,y)=c\\}", note: "gradient crosses contours at right angles" });

  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-R, R]} y={[-R, R]} equal aspect={0.75} maxH={500} ariaLabel={`contour map of ${F.name} with gradient`}
            underlay={(ctx) => (
              <HeatmapCanvas ctx={ctx} res={4} deps={[fid, th.blue[0]]} f={(x, y) => {
                const t = Math.max(0, Math.min(1, (F.f(x, y) - range[0]) / (range[1] - range[0] || 1)));
                const lo = th.panel, hiC = t > 0.5 ? th.pink : th.blue;
                const k = t > 0.5 ? (t - 0.5) * 2 : 1 - t * 2;
                return [lo[0] + (hiC[0] - lo[0]) * k * 0.45, lo[1] + (hiC[1] - lo[1]) * k * 0.45, lo[2] + (hiC[2] - lo[2]) * k * 0.45, 255];
              }} />
            )}>
            <Contours f={F.f} levels={levels} />
            {fieldArrows}
            {mode === "partial" && (
              <>
                <Polyline pts={[[-R, p[1]], [R, p[1]]]} style={{ stroke: C.a("blue", 0.6), strokeDasharray: "4 3" }} />
                <Polyline pts={[[p[0], -R], [p[0], R]]} style={{ stroke: C.a("pink", 0.6), strokeDasharray: "4 3" }} />
                <Arrow from={p} to={[p[0] + Math.sign(g[0]) * Math.min(1.2, Math.abs(g[0]) * 0.4), p[1]]} color="blue" label="∂f/∂x" />
                <Arrow from={p} to={[p[0], p[1] + Math.sign(g[1]) * Math.min(1.2, Math.abs(g[1]) * 0.4)]} color="pink" label="∂f/∂y" />
              </>
            )}
            {mode === "directional" && <Arrow from={p} to={[p[0] + u[0] * 1.2, p[1] + u[1] * 1.2]} color="amber" width={2.4} label="u" />}
            {mode === "differential" && <Arrow from={p} to={[p[0] + ddx, p[1] + ddy]} color="amber" width={2.4} label="(dx, dy)" />}
            {mode === "hessian" && he.vectors && [0, 1].map((i) => {
              const v = [he.vectors[0][i], he.vectors[1][i]];
              return <Polyline key={i} pts={[[p[0] - v[0], p[1] - v[1]], [p[0] + v[0], p[1] + v[1]]]} style={{ stroke: i ? C.amber : C.green, strokeWidth: 2.4 }} />;
            })}
            {mode !== "partial" && gn > 1e-9 && <Arrow from={p} to={[p[0] + g[0] * scaleG, p[1] + g[1] * scaleG]} color="green" width={2.6} label="∇f" />}
            {mode !== "partial" && gn > 1e-9 && (mode === "gradient" || mode === "directional") && <Arrow from={p} to={[p[0] - g[0] * scaleG, p[1] - g[1] * scaleG]} color="red" width={1.6} dashed label="−∇f" />}
            <Handle p={p} onDrag={(q) => setP([Math.max(-R, Math.min(R, Math.round(q[0] * 100) / 100)), Math.max(-R, Math.min(R, Math.round(q[1] * 100) / 100))])} color="fg" label="point (x, y)" />
          </Plot>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
            <span className="text-[12.5px]"><Tex tex={F.tex} /></span>
            <Legend items={[{ label: "∇f (steepest ascent)", color: C.green }, ...(mode === "gradient" || mode === "directional" ? [{ label: "−∇f (steepest descent)", color: C.red, dashed: true }] : []), { label: "level curves", color: C.a("fg", 0.4) }, ...(mode === "hessian" ? [{ label: "Hessian eigen-directions", color: C.amber }] : [])]} />
          </div>
        </div>
      }
      controls={
        <>
          <Select label="Function" value={fid} onChange={setFid} options={functions2D.map((q) => ({ value: q.id, label: q.name }))} />
          <Select label="View" value={mode} onChange={setMode} options={[{ value: "partial", label: "Partial derivatives" }, { value: "gradient", label: "Gradient" }, { value: "directional", label: "Directional derivative" }, { value: "contour", label: "Level curves" }, { value: "hessian", label: "Hessian curvature" }, { value: "differential", label: "Total differential" }]} />
          {mode === "directional" && <ParameterSlider label="direction θ (degrees)" value={theta} min={0} max={359} step={1} onChange={setTheta} />}
          {mode === "differential" && (<><ParameterSlider label="dx" tex value={ddx} min={-1} max={1} step={0.01} onChange={setDdx} /><ParameterSlider label="dy" tex value={ddy} min={-1} max={1} step={0.01} onChange={setDdy} /></>)}
          <Toggle label="Show gradient field" checked={field} onChange={setField} />
          <p className="text-[12px] text-faint">Drag the white point.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "f(x,y)", tex: true, value: fmt(F.f(p[0], p[1])) },
        { label: "\\nabla f", tex: true, value: `(${fmt(g[0])}, ${fmt(g[1])})`, color: C.green },
        { label: "\\|\\nabla f\\|", tex: true, value: fmt(gn) },
        ...(mode === "directional" ? [{ label: "D_{\\mathbf u}f", tex: true, value: fmt(Du), color: C.amber }] : []),
        ...(mode === "hessian" ? [{ label: "\\lambda_1,\\lambda_2", tex: true, value: `${fmt(he.values[0])}, ${fmt(he.values[1])}` }] : []),
      ]} />}
      calc={<CalcSteps steps={steps} />}
    />
  );
}
