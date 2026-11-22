"use client";
import { useState } from "react";
import { Plot, FnPath, Dot, Polyline, C } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Toggle, NumberInput } from "@/components/controls";
import { useSize } from "@/hooks/useSize";
import { mulberry32, randn } from "@/lib/math/random";
import { fmt, tex, texMatrix } from "@/lib/utilities/format";
import { MiniTable } from "@/visualizers/shared";

/* ------------------------------------------------------------------ zero-one loss and its convex surrogates */
const LOSSES = [
  { id: "zero", name: "0-1", color: C.fg, f: (u: number) => (u < 0 ? 1 : 0), d: () => 0, tex: String.raw`\mathbb 1[u<0]`, convex: false },
  { id: "sq", name: "squared", color: C.amber, f: (u: number) => (1 - u) ** 2, d: (u: number) => -2 * (1 - u), tex: String.raw`(1-u)^2`, convex: true },
  { id: "hinge", name: "hinge", color: C.green, f: (u: number) => Math.max(0, 1 - u), d: (u: number) => (u < 1 ? -1 : 0), tex: String.raw`\max(0,\,1-u)`, convex: true },
  { id: "log", name: "logistic", color: C.pink, f: (u: number) => Math.log(1 + Math.exp(-u)), d: (u: number) => -1 / (1 + Math.exp(u)), tex: String.raw`\log(1+e^{-u})`, convex: true },
  { id: "perc", name: "perceptron", color: C.blue, f: (u: number) => Math.max(0, -u), d: (u: number) => (u < 0 ? -1 : 0), tex: String.raw`\max(0,\,-u)`, convex: true },
] as const;

export default function Surrogates() {
  const [u, setU] = useState(-0.5);
  const [on, setOn] = useState<Record<string, boolean>>({ zero: true, sq: true, hinge: true, log: true, perc: true });
  const [scaled, setScaled] = useState(false);
  const val = (l: (typeof LOSSES)[number], x: number) => (l.id === "log" && scaled ? l.f(x) / Math.LN2 : l.f(x));
  const der = (l: (typeof LOSSES)[number], x: number) => (l.id === "log" && scaled ? l.d(x) / Math.LN2 : l.d(x));
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3.5]} y={[0, 4.2]} aspect={0.56} maxH={430} xLabel="u = y · wᵀx" yLabel="loss" ariaLabel="classification losses as functions of the margin" onPointerDown={(p) => setU(Number(Math.max(-3, Math.min(3.5, p[0])).toFixed(2)))}>
            <Polyline pts={[[0, 0], [0, 9]]} style={{ stroke: C.a("fg", 0.22), strokeWidth: 1 }} />
            {LOSSES.filter((l) => on[l.id]).map((l) => l.id === "zero"
              ? <g key={l.id}><Polyline pts={[[-3, 1], [0, 1]]} style={{ stroke: l.color, strokeWidth: 2.4 }} /><Polyline pts={[[0, 0], [3.5, 0]]} style={{ stroke: l.color, strokeWidth: 2.4 }} /></g>
              : <FnPath key={l.id} f={(x) => val(l, x)} n={400} style={{ stroke: l.color, strokeWidth: 2 }} />)}
            <Polyline pts={[[u, 0], [u, 9]]} style={{ stroke: C.a("fg", 0.4), strokeWidth: 1, strokeDasharray: "3 3" }} />
            {LOSSES.filter((l) => on[l.id]).map((l) => <Dot key={l.id} p={[u, val(l, u)]} r={4.5} color={l.color} stroke={C.bg} />)}
          </Plot>
          <Legend items={LOSSES.filter((l) => on[l.id]).map((l) => ({ label: l.name + (l.id === "log" && scaled ? " ÷ ln 2" : ""), color: l.color }))} />
          <p className="text-[12.5px] text-faint">Left of the vertical axis (u &lt; 0) the prediction is wrong; right of it the prediction is correct, with more confidence the larger u is.</p>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="margin u = y · wᵀx (or click the plot)" value={u} min={-3} max={3.5} step={0.05} onChange={setU} />
          <div className="space-y-2">
            {LOSSES.map((l) => <Toggle key={l.id} label={`${l.name} loss`} checked={!!on[l.id]} onChange={(v) => setOn({ ...on, [l.id]: v })} />)}
            <Toggle label="Scale logistic by 1/ln 2 (passes through 1 at u = 0)" checked={scaled} onChange={setScaled} />
          </div>
          <MiniTable head={["loss", "value at u", "slope at u", "convex", "≥ 0-1 loss"]} rows={LOSSES.map((l) => [l.name, fmt(val(l, u), 4), l.id === "zero" ? "0" : fmt(der(l, u), 4), l.convex ? "yes" : "no", l.id === "zero" ? "—" : l.id === "perc" ? "no" : l.id === "log" && !scaled ? "no (0.693 at u = 0)" : "yes"])} />
        </>
      }
      readouts={<Readouts items={LOSSES.map((l) => ({ label: l.tex, tex: true, value: fmt(val(l, u), 4), color: l.color }))} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`u=y\,\mathbf w^\top\mathbf x=${tex(u, 2)}\ \Rightarrow\ \text{prediction is }${u >= 0 ? String.raw`\text{correct}` : String.raw`\text{wrong}`}`, note: "every loss here depends on the example only through u" },
        { tex: String.raw`(\mathbf w^\top\mathbf x-y)^2=y^2\,(y\,\mathbf w^\top\mathbf x-1)^2=(1-u)^2=${tex((1 - u) ** 2, 4)}`, note: "squared error in terms of u, using y² = 1" },
        { tex: String.raw`-\log\sigma(u)=\log(1+e^{-u})=${tex(Math.log(1 + Math.exp(-u)), 4)},\qquad\max(0,1-u)=${tex(Math.max(0, 1 - u), 4)},\qquad\max(0,-u)=${tex(Math.max(0, -u), 4)}` },
      ]} />}
      note="The 0-1 loss is flat on both sides of its jump, so its slope column is always zero: no gradient to follow. Each surrogate is convex and slopes downwards wherever the prediction is wrong, which gives an optimiser something to do. Note the squared loss rising again for u > 1: it penalises predictions for being too correct."
    />
  );
}

/* ------------------------------------------------------------------ forward pass through a small network */
type Act = "relu" | "sigmoid";
const g = (a: Act, z: number) => (a === "relu" ? Math.max(0, z) : 1 / (1 + Math.exp(-z)));
const sig = (z: number) => 1 / (1 + Math.exp(-z));

const INIT = {
  W1: [[0.5, -1], [1, 0.5], [-0.5, 1]],   // 3 hidden units × 2 inputs (row j holds the weights into hidden unit j)
  b1: [0.2, -0.3, 0.1],
  W2: [[1, -1, 0.5]],                      // 1 output × 3 hidden
  b2: [0.1],
};

export function Forward() {
  const [x, setX] = useState([2, 1]);
  const [W1, setW1] = useState(INIT.W1);
  const [b1, setB1] = useState(INIT.b1);
  const [W2, setW2] = useState(INIT.W2);
  const [b2, setB2] = useState(INIT.b2);
  const [act, setAct] = useState<Act>("relu");
  const [out, setOut] = useState<"regression" | "classification">("regression");
  const [y, setY] = useState(1);
  const { ref, width } = useSize<HTMLDivElement>(0.5);
  const z1 = W1.map((r, j) => r[0] * x[0] + r[1] * x[1] + b1[j]);
  const a1 = z1.map((z) => g(act, z));
  const z2 = W2[0].reduce((s, w, j) => s + w * a1[j], b2[0]);
  const yhat = out === "regression" ? z2 : sig(z2);
  const yy = out === "classification" ? (y >= 0.5 ? 1 : 0) : y;
  const loss = out === "regression" ? 0.5 * (yhat - yy) ** 2 : -(yy * Math.log(Math.max(1e-12, yhat)) + (1 - yy) * Math.log(Math.max(1e-12, 1 - yhat)));
  const H = 300, Wd = width;
  const col = [70, Wd * 0.5, Wd - 80];
  const yIn = [H * 0.33, H * 0.67], yH = [H * 0.18, H * 0.5, H * 0.82], yO = H * 0.5;
  const wc = (w: number) => (w >= 0 ? C.blue : C.pink);
  const sw = (w: number) => 0.6 + Math.min(4, Math.abs(w) * 2.2);
  const randomise = () => {
    const r = mulberry32(Math.floor(Math.random() * 1e9));
    const q = () => Math.round(randn(r) * 10) / 10;
    setW1(W1.map((row) => row.map(q))); setB1(b1.map(q)); setW2([W2[0].map(q)]); setB2([q()]);
  };
  const params = W1.length * 2 + b1.length + W2[0].length + 1;
  return (
    <LabShell
      wideControls
      viz={
        <div ref={ref} className="space-y-2">
          <svg width={Wd} height={H} role="img" aria-label="network diagram with the value computed at every unit">
            {W1.map((row, j) => row.map((w, i) => <line key={`a${j}${i}`} x1={col[0] + 24} y1={yIn[i]} x2={col[1] - 34} y2={yH[j]} style={{ stroke: wc(w), strokeWidth: sw(w), opacity: 0.7 }} />))}
            {W2[0].map((w, j) => <line key={`b${j}`} x1={col[1] + 34} y1={yH[j]} x2={col[2] - 34} y2={yO} style={{ stroke: wc(w), strokeWidth: sw(w), opacity: 0.7 }} />)}
            {x.map((v, i) => (
              <g key={`i${i}`}>
                <circle cx={col[0]} cy={yIn[i]} r={24} style={{ fill: C.panel, stroke: C.lineStrong }} />
                <text x={col[0]} y={yIn[i] - 3} textAnchor="middle" style={{ fill: C.muted, fontSize: 10.5 }}>x{i + 1}</text>
                <text x={col[0]} y={yIn[i] + 11} textAnchor="middle" className="num" style={{ fill: C.fg, fontSize: 12 }}>{fmt(v, 2)}</text>
              </g>
            ))}
            {z1.map((z, j) => (
              <g key={`h${j}`}>
                <circle cx={col[1]} cy={yH[j]} r={34} style={{ fill: C.panel, stroke: a1[j] === 0 && act === "relu" ? C.lineStrong : C.amber, strokeWidth: 1.6 }} />
                <text x={col[1]} y={yH[j] - 5} textAnchor="middle" className="num" style={{ fill: C.muted, fontSize: 10.5 }}>z = {fmt(z, 3)}</text>
                <text x={col[1]} y={yH[j] + 11} textAnchor="middle" className="num" style={{ fill: C.fg, fontSize: 12 }}>a = {fmt(a1[j], 3)}</text>
              </g>
            ))}
            <circle cx={col[2]} cy={yO} r={34} style={{ fill: C.panel, stroke: C.green, strokeWidth: 1.8 }} />
            <text x={col[2]} y={yO - 5} textAnchor="middle" className="num" style={{ fill: C.muted, fontSize: 10.5 }}>z = {fmt(z2, 3)}</text>
            <text x={col[2]} y={yO + 11} textAnchor="middle" className="num" style={{ fill: C.fg, fontSize: 12 }}>ŷ = {fmt(yhat, 3)}</text>
            <text x={col[0]} y={18} textAnchor="middle" style={{ fill: C.faint, fontSize: 11 }}>input layer</text>
            <text x={col[1]} y={18} textAnchor="middle" style={{ fill: C.faint, fontSize: 11 }}>hidden layer ({act === "relu" ? "ReLU" : "sigmoid"})</text>
            <text x={col[2]} y={18} textAnchor="middle" style={{ fill: C.faint, fontSize: 11 }}>output ({out === "regression" ? "linear" : "sigmoid"})</text>
          </svg>
          <Legend items={[{ label: "positive weight", color: C.blue }, { label: "negative weight (thickness ∝ |w|)", color: C.pink }, { label: "active hidden unit", color: C.amber }, { label: "output unit", color: C.green }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Hidden activation" value={act} onChange={setAct} options={[{ value: "relu", label: "ReLU" }, { value: "sigmoid", label: "Sigmoid" }]} />
          <Segmented label="Task (output unit and loss)" value={out} onChange={setOut} options={[{ value: "regression", label: "Regression" }, { value: "classification", label: "Binary classification" }]} />
          <div className="grid grid-cols-2 gap-x-3 gap-y-3">
            <ParameterSlider label="x_1" tex value={x[0]} min={-3} max={3} step={0.1} onChange={(v) => setX([v, x[1]])} />
            <ParameterSlider label="x_2" tex value={x[1]} min={-3} max={3} step={0.1} onChange={(v) => setX([x[0], v])} />
          </div>
          {out === "regression" ? <ParameterSlider label="target y" value={y} min={-3} max={3} step={0.1} onChange={setY} /> : <Segmented label="Target y" value={String(yy)} onChange={(v) => setY(Number(v))} options={[{ value: "0", label: "y = 0" }, { value: "1", label: "y = 1" }]} />}
          <div>
            <div className="label mb-1.5">Hidden layer: W₁ (3 × 2) and b₁</div>
            <div className="grid grid-cols-3 gap-1">
              {W1.map((row, j) => [
                <NumberInput key={`w${j}0`} ariaLabel={`weight from x1 to hidden unit ${j + 1}`} value={row[0]} step={0.1} onChange={(v) => setW1(W1.map((r, a) => (a === j ? [v, r[1]] : r)))} />,
                <NumberInput key={`w${j}1`} ariaLabel={`weight from x2 to hidden unit ${j + 1}`} value={row[1]} step={0.1} onChange={(v) => setW1(W1.map((r, a) => (a === j ? [r[0], v] : r)))} />,
                <NumberInput key={`b${j}`} ariaLabel={`bias of hidden unit ${j + 1}`} value={b1[j]} step={0.1} className="border-dashed" onChange={(v) => setB1(b1.map((q, a) => (a === j ? v : q)))} />,
              ])}
            </div>
          </div>
          <div>
            <div className="label mb-1.5">Output layer: W₂ (1 × 3) and b₂</div>
            <div className="grid grid-cols-4 gap-1">
              {W2[0].map((w, j) => <NumberInput key={j} ariaLabel={`weight from hidden unit ${j + 1} to the output`} value={w} step={0.1} onChange={(v) => setW2([W2[0].map((q, a) => (a === j ? v : q))])} />)}
              <NumberInput ariaLabel="output bias" value={b2[0]} step={0.1} className="border-dashed" onChange={(v) => setB2([v])} />
            </div>
          </div>
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={randomise}>Random weights</button>
            <button className="btn flex-1" onClick={() => { setW1(INIT.W1); setB1(INIT.b1); setW2(INIT.W2); setB2(INIT.b2); setX([2, 1]); }}>Worked example</button>
          </div>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\hat y`, tex: true, value: fmt(yhat, 4), color: C.green },
        { label: "y", tex: true, value: fmt(yy, 2) },
        { label: out === "regression" ? String.raw`L=\tfrac12(\hat y-y)^2` : String.raw`L=-y\log\hat y-(1-y)\log(1-\hat y)`, tex: true, value: fmt(loss, 4), color: C.pink },
        { label: String.raw`\partial L/\partial z^{(2)}=\hat y-y`, tex: true, value: fmt(yhat - yy, 4) },
        { label: String.raw`\text{parameters}`, tex: true, value: params },
        { label: String.raw`\text{active hidden units}`, tex: true, value: act === "relu" ? `${a1.filter((a) => a > 0).length} of 3` : "3 of 3" },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\mathbf z^{(1)}=W_1\mathbf x+\mathbf b_1=${texMatrix(W1, 2)}${texMatrix([[x[0]], [x[1]]], 2)}+${texMatrix(b1.map((v) => [v]), 2)}=${texMatrix(z1.map((v) => [v]), 3)}` },
        { tex: String.raw`\mathbf a^{(1)}=${act === "relu" ? String.raw`\operatorname{ReLU}` : String.raw`\sigma`}(\mathbf z^{(1)})=${texMatrix(a1.map((v) => [v]), 3)}`, note: act === "relu" ? "negative pre-activations are set to 0" : "each entry squashed into (0, 1)" },
        { tex: String.raw`z^{(2)}=W_2\mathbf a^{(1)}+b_2=${W2[0].map((w, j) => `(${tex(w, 2)})(${tex(a1[j], 3)})`).join("+")}+(${tex(b2[0], 2)})=${tex(z2, 4)}` },
        { tex: out === "regression" ? String.raw`\hat y=z^{(2)}=${tex(yhat, 4)},\qquad L=\tfrac12(${tex(yhat, 4)}-${tex(yy, 2)})^2=${tex(loss, 4)}` : String.raw`\hat y=\sigma(z^{(2)})=${tex(yhat, 4)}=P(y=1\mid\mathbf x),\qquad L=-\log(${tex(yy ? yhat : 1 - yhat, 4)})=${tex(loss, 4)}`, note: out === "regression" ? "linear output, squared-error loss" : "sigmoid output, cross-entropy loss" },
      ]} />}
      note="Forward propagation is the same two operations repeated layer after layer: an affine map, then an element-wise non-linearity. Only the last layer changes with the task. With a linear activation in the hidden layer the whole network would collapse to a single linear model."
    />
  );
}
