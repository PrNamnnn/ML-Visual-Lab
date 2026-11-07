"use client";
import { useState } from "react";
import { Plot, Polyline, Dot, Handle, C, FnPath, HeatmapCanvas, useThemeRGB } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select } from "@/components/controls";
import { fmt, tex, texMatrix } from "@/lib/utilities/format";

/* ------------------------------ implicit differentiation ------------------------------ */
/** Ellipse x²/a² + y²/b² = 1: dy/dx = −(b² x)/(a² y), checked against the explicit branch. */
export function ImplicitViz() {
  const [a, setA] = useState(2.2);
  const [b, setB] = useState(1.4);
  const [t, setT] = useState(0.9); // angle parameter of the point
  const x = a * Math.cos(t), y = b * Math.sin(t);
  const slope = Math.abs(y) < 1e-9 ? Infinity : -(b * b * x) / (a * a * y);
  const explicit = y >= 0 ? -(b * x) / (a * Math.sqrt(Math.max(1e-12, a * a - x * x))) : (b * x) / (a * Math.sqrt(Math.max(1e-12, a * a - x * x)));
  const ell = Array.from({ length: 181 }, (_, i) => { const s = (i / 180) * 2 * Math.PI; return [a * Math.cos(s), b * Math.sin(s)] as [number, number]; });
  const L = 1.8;
  const tangent: [number, number][] = Number.isFinite(slope) ? [[x - L, y - L * slope], [x + L, y + L * slope]] : [[x, y - L], [x, y + L]];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3.5, 3.5]} y={[-2.6, 2.6]} equal originAxes aspect={0.65} maxH={440} ariaLabel="ellipse with tangent from implicit differentiation">
            <Polyline pts={ell} style={{ stroke: C.blue, strokeWidth: 2.2 }} />
            <Polyline pts={tangent} style={{ stroke: C.pink, strokeWidth: 1.8 }} />
            <Handle p={[x, y]} onDrag={(q) => setT(Math.atan2(q[1] / b, q[0] / a))} color="pink" label="point on the curve" />
          </Plot>
          <Legend items={[{ label: "x²/a² + y²/b² = 1 (not a function y(x))", color: C.blue }, { label: "tangent from dy/dx", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="a (x semi-axis)" value={a} min={0.8} max={3.2} step={0.05} onChange={setA} />
          <ParameterSlider label="b (y semi-axis)" value={b} min={0.6} max={2.4} step={0.05} onChange={setB} />
          <ParameterSlider label="position angle" value={t} min={-Math.PI} max={Math.PI} step={0.01} onChange={setT} />
        </>
      }
      readouts={<Readouts items={[
        { label: "(x,y)", tex: true, value: `(${fmt(x)}, ${fmt(y)})` },
        { label: "dy/dx\\ \\text{(implicit)}", tex: true, value: fmt(slope), color: C.pink },
        { label: "dy/dx\\ \\text{(explicit branch)}", tex: true, value: fmt(explicit) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\frac{d}{dx}\Big(\frac{x^2}{a^2}+\frac{y^2}{b^2}\Big)=0\ \Rightarrow\ \frac{2x}{a^2}+\frac{2y}{b^2}\frac{dy}{dx}=0`, note: "differentiate both sides, treating y as y(x) (chain rule)" },
        { tex: String.raw`\frac{dy}{dx}=-\frac{b^2x}{a^2y}=-\frac{${tex(b * b)}\cdot${tex(x)}}{${tex(a * a)}\cdot${tex(y)}}=${tex(slope)}` },
      ]} />}
      note="At the left and right extremes (y = 0) the tangent is vertical and dy/dx is undefined — exactly where the curve cannot be written locally as y(x)."
    />
  );
}

/* ------------------------------ Jacobian ------------------------------ */
/** Non-linear map f(x, y) = (x + k sin y, y + k sin x). A small square near p maps to a curved patch ≈ J·square. */
export function JacobianViz() {
  const [k, setK] = useState(0.5);
  const [p, setP] = useState<[number, number]>([0.6, 0.4]);
  const [h, setH] = useState(0.5);
  const f = (x: number, y: number): [number, number] => [x + k * Math.sin(y), y + k * Math.sin(x)];
  const J = [[1, k * Math.cos(p[1])], [k * Math.cos(p[0]), 1]];
  const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
  const fp = f(p[0], p[1]);
  const sq: [number, number][] = [];
  const lin: [number, number][] = [];
  const per = 20;
  const corners: [number, number][] = [[0, 0], [h, 0], [h, h], [0, h], [0, 0]];
  for (let e = 0; e < 4; e++) for (let i = 0; i < per; i++) {
    const s = i / per;
    const dx = corners[e][0] + (corners[e + 1][0] - corners[e][0]) * s;
    const dy = corners[e][1] + (corners[e + 1][1] - corners[e][1]) * s;
    sq.push(f(p[0] + dx, p[1] + dy));
    lin.push([fp[0] + J[0][0] * dx + J[0][1] * dy, fp[1] + J[1][0] * dx + J[1][1] * dy]);
  }
  // exact image area by the shoelace formula
  const area = Math.abs(sq.reduce((s, q, i) => { const r2 = sq[(i + 1) % sq.length]; return s + q[0] * r2[1] - r2[0] * q[1]; }, 0)) / 2;
  const grid: [number, number][][] = [];
  for (let g = -3; g <= 3; g += 0.5) {
    grid.push(Array.from({ length: 61 }, (_, i) => f(g, -3 + i * 0.1)));
    grid.push(Array.from({ length: 61 }, (_, i) => f(-3 + i * 0.1, g)));
  }
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-3, 3]} equal originAxes aspect={0.75} maxH={480} ariaLabel="non-linear map and its Jacobian approximation">
            {grid.map((l, i) => <Polyline key={i} pts={l} style={{ stroke: C.a("fg", 0.12), strokeWidth: 1 }} />)}
            <Polyline pts={corners.map(([dx, dy]) => [p[0] + dx, p[1] + dy])} style={{ stroke: C.muted, strokeWidth: 1.2, strokeDasharray: "3 3" }} />
            <Polyline pts={sq} closed style={{ stroke: C.blue, strokeWidth: 2 }} />
            <Polyline pts={lin} closed style={{ stroke: C.pink, strokeWidth: 1.6, strokeDasharray: "5 3" }} />
            <Handle p={p} onDrag={(q) => setP([Math.max(-2.5, Math.min(2.5, q[0])), Math.max(-2.5, Math.min(2.5, q[1]))])} color="fg" label="base point p" />
          </Plot>
          <Legend items={[{ label: "small square at p (before)", color: C.muted, dashed: true }, { label: "exact image f(square)", color: C.blue }, { label: "linear approximation J·square", color: C.pink, dashed: true }, { label: "image of the grid", color: C.a("fg", 0.3) }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="non-linearity k" value={k} min={0} max={1.2} step={0.01} onChange={setK} />
          <ParameterSlider label="square size h" value={h} min={0.05} max={1} step={0.01} onChange={setH} />
          <p className="text-[12px] text-faint">Drag p. Shrink h: the blue and pink shapes converge — the Jacobian is the best local linear approximation.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\det J(\\mathbf p)", tex: true, value: fmt(det, 4), color: det < 0 ? C.red : C.fg },
        { label: "\\text{area ratio }f(\\square)/\\square", tex: true, value: fmt(area / (h * h), 4) },
        { label: "\\text{error vs }\\det J", tex: true, value: fmt(Math.abs(area / (h * h) - det), 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`J=\begin{bmatrix}\partial f_1/\partial x&\partial f_1/\partial y\\\partial f_2/\partial x&\partial f_2/\partial y\end{bmatrix}=\begin{bmatrix}1&k\cos y\\k\cos x&1\end{bmatrix}=` + texMatrix(J, 3) },
        { tex: String.raw`f(\mathbf p+\boldsymbol\delta)\approx f(\mathbf p)+J\boldsymbol\delta,\qquad \frac{\text{area}(f(\square))}{\text{area}(\square)}\to|\det J|=${tex(Math.abs(det), 4)}` },
      ]} />}
      note="The determinant of the Jacobian is the local area-scaling factor of a non-linear map — the term normalising flows track as log|det J| to compute exact likelihoods."
    />
  );
}

/* ------------------------------ double integrals ------------------------------ */
const DI: Record<string, { name: string; tex: string; f: (x: number, y: number) => number; exact: (a: number) => number; texExact: string }> = {
  quad: { name: "x² + y", tex: String.raw`f(x,y)=x^2+y`, f: (x, y) => x * x + y, exact: (a) => (a ** 4) / 3 + (a ** 3) / 2, texExact: String.raw`\int_0^a\!\!\int_0^a(x^2+y)\,dy\,dx=\tfrac{a^4}{3}+\tfrac{a^3}{2}` },
  prod: { name: "x·y", tex: String.raw`f(x,y)=xy`, f: (x, y) => x * y, exact: (a) => (a ** 4) / 4, texExact: String.raw`\int_0^a\!\!\int_0^a xy\,dy\,dx=\tfrac{a^4}{4}` },
  gauss: { name: "e^{−(x²+y²)}", tex: String.raw`f(x,y)=e^{-(x^2+y^2)}`, f: (x, y) => Math.exp(-(x * x + y * y)), exact: (a) => { const e = (t: number) => { let s = 0; const N = 2000; for (let i = 0; i < N; i++) { const u = ((i + 0.5) / N) * t; s += Math.exp(-u * u); } return (s * t) / N; }; const v = e(a); return v * v; }, texExact: String.raw`\Big(\int_0^ae^{-x^2}dx\Big)^2\ \text{(separable)}` },
};

export function DoubleIntegralViz() {
  const [fid, setFid] = useState("quad");
  const [n, setN] = useState(6);
  const [a, setA] = useState(2);
  const th = useThemeRGB();
  const F = DI[fid];
  const w = a / n;
  let sum = 0;
  let mx = 0;
  const cells: { x: number; y: number; v: number }[] = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = F.f((i + 0.5) * w, (j + 0.5) * w);
    sum += v * w * w;
    mx = Math.max(mx, v);
    cells.push({ x: i * w, y: j * w, v });
  }
  const exact = F.exact(a);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, a]} y={[0, a]} equal aspect={0.8} maxH={440} ariaLabel="midpoint grid for a double integral"
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={3} deps={[fid, n, a, th.blue[0]]} f={(x, y) => {
              const i = Math.min(n - 1, Math.floor(x / w)), j = Math.min(n - 1, Math.floor(y / w));
              const v = F.f((i + 0.5) * w, (j + 0.5) * w) / (mx || 1);
              return [th.panel[0] + (th.blue[0] - th.panel[0]) * v * 0.75, th.panel[1] + (th.blue[1] - th.panel[1]) * v * 0.75, th.panel[2] + (th.blue[2] - th.panel[2]) * v * 0.75, 255];
            }} />}>
            {Array.from({ length: n + 1 }, (_, i) => <Polyline key={`v${i}`} pts={[[i * w, 0], [i * w, a]]} style={{ stroke: C.a("fg", 0.25), strokeWidth: 0.8 }} />)}
            {Array.from({ length: n + 1 }, (_, i) => <Polyline key={`h${i}`} pts={[[0, i * w], [a, i * w]]} style={{ stroke: C.a("fg", 0.25), strokeWidth: 0.8 }} />)}
            {n <= 12 && cells.map((c, i) => <Dot key={i} p={[c.x + w / 2, c.y + w / 2]} r={2} color={C.pink} />)}
          </Plot>
          <Legend items={[{ label: "cell colour = f at the cell midpoint (pink dots)", color: C.blue }, { label: "each cell contributes f(midpoint)·ΔA (a column of that height)", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <Select label="Function" value={fid} onChange={setFid} options={Object.entries(DI).map(([k, v]) => ({ value: k, label: v.name }))} />
          <ParameterSlider label="cells per side n" value={n} min={1} max={60} step={1} onChange={setN} />
          <ParameterSlider label="square [0, a]²: a" value={a} min={0.5} max={3} step={0.1} onChange={setA} />
        </>
      }
      readouts={<Readouts items={[
        { label: "\\sum f\\,\\Delta A", tex: true, value: fmt(sum, 6) },
        { label: "\\text{exact}", tex: true, value: fmt(exact, 6) },
        { label: "\\text{error}", tex: true, value: fmt(Math.abs(sum - exact), 6) },
        { label: "\\Delta A", tex: true, value: fmt(w * w, 5) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\iint_R f\,dA\approx\sum_{i=1}^{${n}}\sum_{j=1}^{${n}}f(x_i^*,y_j^*)\,\Delta A,\quad\Delta A=(${tex(a)}/${n})^2` },
        { tex: F.texExact + "=" + tex(exact, 5), note: fid === "gauss" ? "1-D factor computed numerically (no elementary antiderivative)" : "Fubini: integrate in y, then in x" },
      ]} />}
      note="The double integral is the volume under the surface z = f(x, y). Midpoint sums converge like 1/n². In probability, integrating a joint density over a region gives the probability of that region."
    />
  );
}

/* ------------------------------ convexity ------------------------------ */
const CF: Record<string, { name: string; tex: string; f: (x: number) => number; d2: (x: number) => number; convex: boolean }> = {
  sq: { name: "x² (convex)", tex: "x^2", f: (x) => x * x, d2: () => 2, convex: true },
  exp: { name: "eˣ (convex)", tex: "e^x", f: Math.exp, d2: Math.exp, convex: true },
  abs: { name: "|x| (convex, not smooth)", tex: "|x|", f: Math.abs, d2: () => 0, convex: true },
  logl: { name: "log(1+eˣ) (convex)", tex: String.raw`\log(1+e^x)`, f: (x) => Math.log1p(Math.exp(x)), d2: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, convex: true },
  wave: { name: "x² − 2cos(3x) (non-convex)", tex: String.raw`x^2-2\cos 3x`, f: (x) => x * x - 2 * Math.cos(3 * x), d2: (x) => 2 + 18 * Math.cos(3 * x), convex: false },
  cubic: { name: "x³ − 3x (non-convex)", tex: "x^3-3x", f: (x) => x ** 3 - 3 * x, d2: (x) => 6 * x, convex: false },
};

export function ConvexityViz() {
  const [fid, setFid] = useState("wave");
  const [x1, setX1] = useState(-1.4);
  const [x2, setX2] = useState(1.1);
  const [lam, setLam] = useState(0.5);
  const F = CF[fid];
  const xm = lam * x1 + (1 - lam) * x2;
  const chord = lam * F.f(x1) + (1 - lam) * F.f(x2);
  const fx = F.f(xm);
  const holds = fx <= chord + 1e-12;
  const ys = Array.from({ length: 200 }, (_, i) => F.f(-2.5 + (5 * i) / 199));
  const ylo = Math.min(...ys), yhi = Math.min(12, Math.max(...ys));
  // scan for violations of f'' >= 0
  let minD2 = Infinity;
  for (let i = 0; i <= 400; i++) minD2 = Math.min(minD2, F.d2(-2.5 + (5 * i) / 400));
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-2.5, 2.5]} y={[ylo - 0.5, yhi + 0.5]} aspect={0.6} maxH={440} originAxes ariaLabel="chord test for convexity">
            <FnPath f={F.f} />
            <Polyline pts={[[x1, F.f(x1)], [x2, F.f(x2)]]} style={{ stroke: holds ? C.green : C.red, strokeWidth: 1.8 }} />
            <Polyline pts={[[xm, fx], [xm, chord]]} style={{ stroke: C.amber, strokeWidth: 1.4, strokeDasharray: "3 3" }} />
            <Dot p={[xm, chord]} r={4.5} color={holds ? C.green : C.red} />
            <Dot p={[xm, fx]} r={4.5} color={C.blue} />
            <Handle p={[x1, F.f(x1)]} onDrag={(q) => setX1(Math.max(-2.5, Math.min(2.5, q[0])))} color="fg" label="first point" />
            <Handle p={[x2, F.f(x2)]} onDrag={(q) => setX2(Math.max(-2.5, Math.min(2.5, q[0])))} color="fg" label="second point" />
          </Plot>
          <Legend items={[{ label: "f", color: C.blue }, { label: holds ? "chord above the curve here" : "chord dips below the curve: not convex", color: holds ? C.green : C.red }]} />
        </div>
      }
      controls={
        <>
          <Select label="Function" value={fid} onChange={setFid} options={Object.entries(CF).map(([k, v]) => ({ value: k, label: v.name }))} />
          <ParameterSlider label="λ (position between the points)" tex={false} value={lam} min={0} max={1} step={0.01} onChange={setLam} />
          <p className="text-[12px] text-faint">Drag the two white handles. For a convex function no choice of points and λ can make the chord go below the curve.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "f(\\lambda x_1+(1-\\lambda)x_2)", tex: true, value: fmt(fx, 4) },
        { label: "\\lambda f(x_1)+(1-\\lambda)f(x_2)", tex: true, value: fmt(chord, 4) },
        { label: "\\text{Jensen here}", tex: true, value: holds ? "holds" : "violated", color: holds ? C.green : C.red },
        { label: "\\min f''\\ \\text{on }[-2.5,2.5]", tex: true, value: fmt(minD2, 3), color: minD2 >= -1e-12 ? C.green : C.red },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`f(\lambda x_1+(1-\lambda)x_2)=${tex(fx, 4)}\ ${holds ? "\\le" : ">"}\ ${tex(chord, 4)}=\lambda f(x_1)+(1-\lambda)f(x_2)` },
        { tex: String.raw`f''(x)\ge0\ \forall x\iff f\text{ convex};\quad f(x)=${F.tex}`, note: F.convex ? "convex: every local minimum is global" : "non-convex: several local minima are possible" },
      ]} />}
      note="Convexity is why logistic regression, linear regression and SVM training always reach the global optimum, while neural-network losses (non-convex) depend on initialisation."
    />
  );
}

