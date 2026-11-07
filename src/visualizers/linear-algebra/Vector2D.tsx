"use client";
import { useState } from "react";
import { Plot, Arrow, Handle, C, Label, usePlot, Polyline } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle, Presets } from "@/components/controls";
import { dot, norm2, angleBetween, project, unit, sub, add, scale } from "@/lib/math/linalg";
import { fmt, tex, texMatrix } from "@/lib/utilities/format";

type Mode = "inner-product" | "components" | "add" | "scale" | "magnitude" | "dot" | "angle" | "project" | "orthogonal" | "gram-schmidt";
type V = [number, number];

const modes: { value: Mode; label: string }[] = [
  { value: "components", label: "Components" },
  { value: "add", label: "Addition / subtraction" },
  { value: "scale", label: "Scalar multiplication" },
  { value: "magnitude", label: "Magnitude & unit vector" },
  { value: "dot", label: "Dot product" },
  { value: "angle", label: "Angle between vectors" },
  { value: "project", label: "Projection" },
  { value: "orthogonal", label: "Orthogonality" },
  { value: "gram-schmidt", label: "Gram–Schmidt" },
  { value: "inner-product", label: "Weighted inner product" },
];

const snap = (p: number[]): V => [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10];
const clampV = (p: number[]): V => snap([Math.max(-4.8, Math.min(4.8, p[0])), Math.max(-4.8, Math.min(4.8, p[1]))]);

function AngleArc({ a, b, r = 0.7 }: { a: V; b: V; r?: number }) {
  const { sx, sy } = usePlot();
  const t1 = Math.atan2(a[1], a[0]);
  let t2 = Math.atan2(b[1], b[0]);
  let d = t2 - t1;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  t2 = t1 + d;
  const pts: string[] = [];
  for (let i = 0; i <= 30; i++) {
    const t = t1 + (d * i) / 30;
    pts.push(`${i ? "L" : "M"}${sx(r * Math.cos(t))},${sy(r * Math.sin(t))}`);
  }
  return <path d={pts.join("")} fill="none" style={{ stroke: C.amber, strokeWidth: 1.5 }} />;
}

function RightAngle({ at, dir1, dir2, s = 0.25 }: { at: V; dir1: V; dir2: V; s?: number }) {
  const { sx, sy } = usePlot();
  const u1 = unit(dir1);
  const u2 = unit(dir2);
  if (!u1 || !u2) return null;
  const p1 = add(at, scale(u1, s));
  const p3 = add(at, scale(u2, s));
  const p2 = add(p1, scale(u2, s));
  return <path d={`M${sx(p1[0])},${sy(p1[1])}L${sx(p2[0])},${sy(p2[1])}L${sx(p3[0])},${sy(p3[1])}`} fill="none" style={{ stroke: C.muted, strokeWidth: 1 }} />;
}

export default function Vector2D({ mode: initial = "dot", lab = false }: { mode?: Mode; lab?: boolean }) {
  const [mode, setMode] = useState<Mode>(initial);
  const [u, setU] = useState<V>([3, 1]);
  const [v, setV] = useState<V>([1, 2.5]);
  const [c, setC] = useState(1.5);
  const [subtract, setSubtract] = useState(false);
  const [snapPerp, setSnapPerp] = useState(true);
  const [w11, setW11] = useState(2);
  const [w22, setW22] = useState(0.6);
  const [w12, setW12] = useState(0.3);

  const reset = () => { setU([3, 1]); setV([1, 2.5]); setC(1.5); };
  const vEff: V = mode === "orthogonal" && snapPerp ? [-u[1] * 0.8, u[0] * 0.8] : v;

  const d = dot(u, vEff);
  const nu = norm2(u);
  const nv = norm2(vEff);
  const th = angleBetween(u, vEff);
  const proj = project(u, vEff);

  let viz: React.ReactNode = null;
  let readouts: { label: string; value: React.ReactNode; tex?: boolean; color?: string }[] = [];
  let steps: { tex: string; note?: string }[] = [];
  const legend: { label: string; color: string; dashed?: boolean }[] = [];

  const H = (p: V, set: (p: V) => void, color: string, label: string) => <Handle p={p} onDrag={(q) => set(clampV(q))} color={color} label={label} />;

  switch (mode) {
    case "components": {
      viz = (
        <>
          <Arrow from={[0, 0]} to={[u[0], 0]} color="pink" width={1.5} dashed />
          <Arrow from={[u[0], 0]} to={u} color="green" width={1.5} dashed />
          <Arrow from={[0, 0]} to={u} color="blue" label="v" />
          <Label p={[u[0] / 2, 0]} dy={u[1] >= 0 ? 14 : -6} anchor="middle" color={C.pink}>v₁ = {fmt(u[0], 1)}</Label>
          <Label p={[u[0], u[1] / 2]} dx={6} color={C.green}>v₂ = {fmt(u[1], 1)}</Label>
          {H(u, setU, "blue", "vector v")}
        </>
      );
      readouts = [{ label: "\\mathbf v", tex: true, value: `(${fmt(u[0], 1)}, ${fmt(u[1], 1)})` }, { label: "\\|\\mathbf v\\|", tex: true, value: fmt(nu) }, { label: "direction", value: `${fmt((Math.atan2(u[1], u[0]) * 180) / Math.PI, 1)}°` }];
      steps = [{ tex: `\\mathbf v=${tex(u[0], 1)}\\,\\mathbf e_1+${tex(u[1], 1)}\\,\\mathbf e_2` }];
      break;
    }
    case "add": {
      const r = subtract ? sub(u, v) : add(u, v);
      viz = subtract ? (
        <>
          <Arrow from={[0, 0]} to={u} color="blue" label="u" />
          <Arrow from={[0, 0]} to={v} color="pink" label="v" />
          <Arrow from={v} to={u} color="green" label="u − v" width={2.5} />
          <Arrow from={[0, 0]} to={r} color="green" dashed width={1.4} />
          {H(u, setU, "blue", "vector u")}
          {H(v, setV, "pink", "vector v")}
        </>
      ) : (
        <>
          <Arrow from={u} to={r} color="pink" dashed width={1.3} />
          <Arrow from={v} to={r} color="blue" dashed width={1.3} />
          <Arrow from={[0, 0]} to={u} color="blue" label="u" />
          <Arrow from={[0, 0]} to={v} color="pink" label="v" />
          <Arrow from={[0, 0]} to={r} color="green" label="u + v" width={2.6} />
          {H(u, setU, "blue", "vector u")}
          {H(v, setV, "pink", "vector v")}
        </>
      );
      readouts = [
        { label: "\\mathbf u", tex: true, value: `(${fmt(u[0], 1)}, ${fmt(u[1], 1)})`, color: C.blue },
        { label: "\\mathbf v", tex: true, value: `(${fmt(v[0], 1)}, ${fmt(v[1], 1)})`, color: C.pink },
        { label: subtract ? "\\mathbf u-\\mathbf v" : "\\mathbf u+\\mathbf v", tex: true, value: `(${fmt(r[0], 1)}, ${fmt(r[1], 1)})`, color: C.green },
        { label: "\\|\\text{result}\\|", tex: true, value: fmt(norm2(r)) },
      ];
      steps = [{ tex: `${subtract ? "\\mathbf u-\\mathbf v" : "\\mathbf u+\\mathbf v"}=(${tex(u[0], 1)}${subtract ? "-" : "+"}(${tex(v[0], 1)}),\\;${tex(u[1], 1)}${subtract ? "-" : "+"}(${tex(v[1], 1)}))=(${tex(r[0], 2)},\\;${tex(r[1], 2)})` }];
      break;
    }
    case "scale": {
      const r = scale(u, c);
      viz = (
        <>
          <Arrow from={[-u[0] * 6, -u[1] * 6]} to={[u[0] * 6, u[1] * 6]} color="muted" width={0.6} dashed />
          <Arrow from={[0, 0]} to={r} color="pink" label={`${fmt(c, 2)}·v`} width={3} />
          <Arrow from={[0, 0]} to={u} color="blue" label="v" width={1.8} />
          {H(u, setU, "blue", "vector v")}
        </>
      );
      readouts = [{ label: "c", tex: true, value: fmt(c, 2) }, { label: "c\\mathbf v", tex: true, value: `(${fmt(r[0], 2)}, ${fmt(r[1], 2)})`, color: C.pink }, { label: "\\|c\\mathbf v\\|", tex: true, value: fmt(norm2(r)) }, { label: "|c|\\,\\|\\mathbf v\\|", tex: true, value: fmt(Math.abs(c) * nu) }];
      steps = [{ tex: `${tex(c, 2)}\\cdot(${tex(u[0], 1)},${tex(u[1], 1)})=(${tex(r[0], 2)},${tex(r[1], 2)})` }, { tex: `\\|c\\mathbf v\\|=|${tex(c, 2)}|\\cdot${tex(nu)}=${tex(Math.abs(c) * nu)}`, note: c < 0 ? "direction reversed" : c === 0 ? "zero vector" : undefined }];
      break;
    }
    case "magnitude": {
      const uh = unit(u);
      viz = (
        <>
          <UnitCircle />
          <Arrow from={[0, 0]} to={u} color="blue" label="v" />
          {uh && <Arrow from={[0, 0]} to={uh} color="pink" label="v̂" width={3} />}
          {H(u, setU, "blue", "vector v")}
        </>
      );
      readouts = [{ label: "\\|\\mathbf v\\|", tex: true, value: fmt(nu) }, { label: "\\hat{\\mathbf v}", tex: true, value: uh ? `(${fmt(uh[0])}, ${fmt(uh[1])})` : "undefined (v = 0)", color: C.pink }, { label: "\\|\\hat{\\mathbf v}\\|", tex: true, value: uh ? fmt(norm2(uh)) : "—" }];
      steps = [
        { tex: `\\|\\mathbf v\\|=\\sqrt{${tex(u[0], 1)}^2+${tex(u[1], 1)}^2}=\\sqrt{${tex(u[0] ** 2 + u[1] ** 2, 2)}}=${tex(nu)}` },
        uh ? { tex: `\\hat{\\mathbf v}=\\tfrac{1}{${tex(nu)}}(${tex(u[0], 1)},${tex(u[1], 1)})=(${tex(uh[0])},${tex(uh[1])})` } : { tex: "\\mathbf v=\\mathbf 0\\Rightarrow\\hat{\\mathbf v}\\text{ undefined}" },
      ];
      break;
    }
    case "dot":
    case "angle":
    case "orthogonal": {
      const pv = proj;
      viz = (
        <>
          {mode === "dot" && pv && nv > 0 && (
            <>
              <Arrow from={[0, 0]} to={project(u, vEff) ?? [0, 0]} color="amber" width={4} />
              <Arrow from={u} to={pv} color="muted" width={1} dashed />
            </>
          )}
          {th !== null && mode !== "orthogonal" && <AngleArc a={u} b={vEff} />}
          {mode === "orthogonal" && Math.abs(d) < 0.05 && <RightAngle at={[0, 0]} dir1={u} dir2={vEff} />}
          <Arrow from={[0, 0]} to={u} color="blue" label="u" />
          <Arrow from={[0, 0]} to={vEff} color="pink" label="v" />
          {H(u, setU, "blue", "vector u")}
          {!(mode === "orthogonal" && snapPerp) && H(v, setV, "pink", "vector v")}
        </>
      );
      if (mode === "dot") legend.push({ label: "projection of u onto v (length ‖u‖cos θ)", color: C.amber });
      readouts = [
        { label: "\\mathbf u\\cdot\\mathbf v", tex: true, value: fmt(d), color: d > 1e-9 ? C.green : d < -1e-9 ? C.red : C.fg },
        { label: "\\|\\mathbf u\\|", tex: true, value: fmt(nu) },
        { label: "\\|\\mathbf v\\|", tex: true, value: fmt(nv) },
        { label: "\\cos\\theta", tex: true, value: th === null ? "undefined" : fmt(Math.cos(th)) },
        { label: "\\theta", tex: true, value: th === null ? "undefined" : `${fmt((th * 180) / Math.PI, 2)}°` },
      ];
      steps = [
        { tex: `\\mathbf u\\cdot\\mathbf v=(${tex(u[0], 1)})(${tex(vEff[0], 2)})+(${tex(u[1], 1)})(${tex(vEff[1], 2)})=${tex(d)}` },
        th === null ? { tex: "\\text{zero vector: angle undefined}" } : { tex: `\\cos\\theta=\\dfrac{${tex(d)}}{${tex(nu)}\\cdot${tex(nv)}}=${tex(Math.cos(th))}\\;\\Rightarrow\\;\\theta=${tex((th * 180) / Math.PI, 2)}^\\circ` },
      ];
      if (mode === "orthogonal") steps.push({ tex: Math.abs(d) < 1e-6 ? "\\mathbf u\\cdot\\mathbf v=0\\Rightarrow\\mathbf u\\perp\\mathbf v" : "\\mathbf u\\cdot\\mathbf v\\neq0\\Rightarrow\\text{not orthogonal}" });
      break;
    }
    case "project": {
      const pv = proj;
      const resid = pv ? sub(u, pv) : null;
      viz = (
        <>
          <Arrow from={[-vEff[0] * 6, -vEff[1] * 6]} to={[vEff[0] * 6, vEff[1] * 6]} color="muted" width={0.6} dashed />
          {pv && resid && (
            <>
              <Arrow from={pv} to={u} color="green" dashed width={1.6} />
              <RightAngle at={pv as V} dir1={scale(vEff, -1) as V} dir2={resid as V} s={0.22} />
              <Arrow from={[0, 0]} to={pv} color="amber" width={3.5} label="proj" labelOffset={[6, 14]} />
            </>
          )}
          <Arrow from={[0, 0]} to={vEff} color="pink" label="v" />
          <Arrow from={[0, 0]} to={u} color="blue" label="u" />
          {H(u, setU, "blue", "vector u")}
          {H(v, setV, "pink", "vector v")}
        </>
      );
      legend.push({ label: "projection", color: C.amber }, { label: "residual u − proj (⊥ v)", color: C.green, dashed: true });
      const k = nv > 0 ? d / (nv * nv) : NaN;
      readouts = [
        { label: "\\operatorname{proj}_{\\mathbf v}\\mathbf u", tex: true, value: pv ? `(${fmt(pv[0])}, ${fmt(pv[1])})` : "undefined", color: C.amber },
        { label: "\\text{coefficient}", tex: true, value: fmt(k) },
        { label: "\\text{scalar proj}", tex: true, value: nv ? fmt(d / nv) : "undefined" },
        { label: "\\text{residual}\\cdot\\mathbf v", tex: true, value: pv && resid ? fmt(dot(resid, vEff), 6) : "—" },
      ];
      steps = nv
        ? [
            { tex: `\\dfrac{\\mathbf u\\cdot\\mathbf v}{\\mathbf v\\cdot\\mathbf v}=\\dfrac{${tex(d)}}{${tex(nv * nv)}}=${tex(k)}` },
            { tex: `\\operatorname{proj}_{\\mathbf v}\\mathbf u=${tex(k)}\\,(${tex(vEff[0], 1)},${tex(vEff[1], 1)})=(${tex(pv![0])},${tex(pv![1])})` },
          ]
        : [{ tex: "\\mathbf v=\\mathbf 0\\Rightarrow\\text{projection undefined}" }];
      break;
    }
    case "inner-product": {
      const W = [[w11, w12], [w12, w22]];
      const ipW = (a: V, b: V) => a[0] * (W[0][0] * b[0] + W[0][1] * b[1]) + a[1] * (W[1][0] * b[0] + W[1][1] * b[1]);
      const detW = w11 * w22 - w12 * w12;
      const valid = w11 > 0 && detW > 0;
      const nuW = Math.sqrt(Math.max(0, ipW(u, u))), nvW = Math.sqrt(Math.max(0, ipW(v, v)));
      const cosW = valid && nuW > 0 && nvW > 0 ? ipW(u, v) / (nuW * nvW) : NaN;
      // unit "circle" {x : xᵀWx = 1}: x = W^{-1/2}(cos t, sin t) computed by solving along each direction
      const ball: V[] = Array.from({ length: 121 }, (_, i) => { const t = (i / 120) * 2 * Math.PI; const dvec: V = [Math.cos(t), Math.sin(t)]; const q = ipW(dvec, dvec); return q > 0 ? [dvec[0] / Math.sqrt(q), dvec[1] / Math.sqrt(q)] : [0, 0]; });
      viz = (
        <>
          {valid && <Polyline pts={ball} closed style={{ stroke: C.amber, strokeWidth: 1.6 }} />}
          <UnitCircle />
          <Arrow from={[0, 0]} to={u} color="blue" label="u" />
          <Arrow from={[0, 0]} to={v} color="pink" label="v" />
          {H(u, setU, "blue", "vector u")}
          {H(v, setV, "pink", "vector v")}
        </>
      );
      legend.push({ label: "unit ball of the W-norm: xᵀWx = 1", color: C.amber }, { label: "ordinary unit circle", color: C.lineStrong, dashed: true });
      readouts = [
        { label: "\\langle\\mathbf u,\\mathbf v\\rangle_W", tex: true, value: valid ? fmt(ipW(u, v)) : "W not positive definite", color: valid && Math.abs(ipW(u, v)) < 0.05 ? C.green : C.fg },
        { label: "\\mathbf u\\cdot\\mathbf v", tex: true, value: fmt(dot(u, v)) },
        { label: "\\|\\mathbf u\\|_W", tex: true, value: valid ? fmt(nuW) : "—" },
        { label: "\theta_W", tex: true, value: Number.isFinite(cosW) ? `${fmt((Math.acos(Math.max(-1, Math.min(1, cosW))) * 180) / Math.PI, 2)}°` : "—" },
      ];
      steps = [
        { tex: String.raw`\langle\mathbf u,\mathbf v\rangle_W=\mathbf u^\top W\mathbf v=` + texMatrix([[u[0], u[1]]], 1) + texMatrix(W, 2) + texMatrix([[v[0]], [v[1]]], 1) + "=" + tex(ipW(u, v)) },
        { tex: String.raw`\det W=${tex(detW)}\ ${valid ? ">0" : "\le0"}\Rightarrow ${valid ? String.raw`\text{a valid inner product}` : String.raw`\text{not an inner product (not positive definite)}`}` },
      ];
      break;
    }
    case "gram-schmidt": {
      const a1 = u, a2 = v;
      const q1 = unit(a1);
      const comp = q1 ? dot(a2, q1) : 0;
      const u2 = q1 ? sub(a2, scale(q1, comp)) : a2;
      const q2 = unit(u2);
      const dependent = !q1 || norm2(u2) < 1e-6;
      viz = (
        <>
          <UnitCircle />
          {q1 && <Arrow from={[0, 0]} to={scale(q1, comp)} color="amber" width={2} dashed />}
          {q1 && <Arrow from={scale(q1, comp)} to={a2} color="green" width={1.6} dashed />}
          <Arrow from={[0, 0]} to={a1} color="blue" label="a₁" width={1.6} />
          <Arrow from={[0, 0]} to={a2} color="pink" label="a₂" width={1.6} />
          {q1 && <Arrow from={[0, 0]} to={q1} color="blue" label="q₁" width={3.4} />}
          {q2 && !dependent && <Arrow from={[0, 0]} to={q2} color="green" label="q₂" width={3.4} />}
          {H(a1, setU, "blue", "vector a1")}
          {H(a2, setV, "pink", "vector a2")}
        </>
      );
      legend.push({ label: "(a₂·q₁) q₁ — removed", color: C.amber, dashed: true }, { label: "u₂ = a₂ − (a₂·q₁)q₁", color: C.green, dashed: true });
      readouts = [
        { label: "\\mathbf q_1", tex: true, value: q1 ? `(${fmt(q1[0])}, ${fmt(q1[1])})` : "undefined" },
        { label: "\\mathbf q_2", tex: true, value: q2 && !dependent ? `(${fmt(q2[0])}, ${fmt(q2[1])})` : "undefined (dependent)" },
        { label: "\\mathbf q_1\\cdot\\mathbf q_2", tex: true, value: q1 && q2 && !dependent ? fmt(dot(q1, q2), 8) : "—" },
        { label: "R", tex: true, value: q1 && q2 && !dependent ? `[[${fmt(norm2(a1))}, ${fmt(comp)}], [0, ${fmt(norm2(u2))}]]` : "—" },
      ];
      steps = q1
        ? [
            { tex: `\\mathbf q_1=\\dfrac{\\mathbf a_1}{\\|\\mathbf a_1\\|}=\\dfrac{(${tex(a1[0], 1)},${tex(a1[1], 1)})}{${tex(norm2(a1))}}=(${tex(q1[0])},${tex(q1[1])})` },
            { tex: `\\mathbf u_2=\\mathbf a_2-(${tex(comp)})\\,\\mathbf q_1=(${tex(u2[0])},${tex(u2[1])})` },
            dependent ? { tex: "\\mathbf u_2\\approx\\mathbf 0\\Rightarrow\\mathbf a_2\\text{ depends on }\\mathbf a_1" } : { tex: `\\mathbf q_2=\\mathbf u_2/${tex(norm2(u2))}=(${tex(q2![0])},${tex(q2![1])})` },
          ]
        : [{ tex: "\\mathbf a_1=\\mathbf 0:\\text{ cannot normalise}" }];
      break;
    }
  }

  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-5, 5]} y={[-5, 5]} equal originAxes aspect={0.75} maxH={520} ariaLabel={`2D vector visualization: ${mode}`}>
            {viz}
          </Plot>
          {legend.length > 0 && <Legend items={legend} />}
        </div>
      }
      controls={
        <>
          {lab && <Select label="Operation" value={mode} options={modes} onChange={setMode} />}
          {mode === "add" && <Toggle label="Show difference u − v" checked={subtract} onChange={setSubtract} />}
          {mode === "scale" && <ParameterSlider label="c" tex value={c} min={-3} max={3} step={0.05} onChange={setC} />}
          {mode === "inner-product" && (<><ParameterSlider label="W₁₁" value={w11} min={0.1} max={3} step={0.05} onChange={setW11} /><ParameterSlider label="W₂₂" value={w22} min={0.1} max={3} step={0.05} onChange={setW22} /><ParameterSlider label="W₁₂ = W₂₁" value={w12} min={-2} max={2} step={0.05} onChange={setW12} /></>)}
          {mode === "orthogonal" && <Toggle label="Keep v ⟂ u (rotate 90°)" checked={snapPerp} onChange={setSnapPerp} />}
          <Presets
            presets={[
              { label: "Acute", value: [[3, 1], [1, 2.5]] as [V, V] },
              { label: "Right angle", value: [[3, 0], [0, 2]] as [V, V] },
              { label: "Obtuse", value: [[3, 1], [-2, 1.5]] as [V, V] },
              { label: "Parallel", value: [[2, 1], [4, 2]] as [V, V] },
            ]}
            onPick={([a, b]) => { setU(a); setV(b); }}
          />
          <button className="btn w-full" onClick={reset}>Reset vectors</button>
          <p className="text-[12px] leading-relaxed text-faint">Drag the coloured handles (or focus them and use arrow keys). Values snap to 0.1.</p>
        </>
      }
      readouts={<Readouts items={readouts} />}
      calc={<CalcSteps steps={steps} />}
    />
  );
}

function UnitCircle() {
  const { sx, sy } = usePlot();
  return <circle cx={sx(0)} cy={sy(0)} r={Math.abs(sx(1) - sx(0))} fill="none" style={{ stroke: C.lineStrong, strokeDasharray: "3 3" }} />;
}
