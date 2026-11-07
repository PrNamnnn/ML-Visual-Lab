"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Plot, HeatmapCanvas, Dot, C, useThemeRGB, Polyline, FnPath } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, AnimationControls, Segmented } from "@/components/controls";
import { createMLP, forward, trainStep, activations, type MLP, type ActId } from "@/lib/nn";
import { makeDataset, type DatasetId } from "@/lib/ml/models";
import { useSize, useTicker } from "@/hooks/useSize";
import { fmt, tex, cn } from "@/lib/utilities/format";

/* ------------------------------ single neuron ------------------------------ */
export function NeuronViz({ mode = "neuron" }: { mode?: "neuron" | "perceptron" }) {
  const [x, setX] = useState([1, 0.5, -1]);
  const [w, setW] = useState([0.8, -0.4, 0.3]);
  const [b, setB] = useState(0.1);
  const [act, setAct] = useState(mode === "perceptron" ? "step" : "sigmoid");
  const z = x.reduce((s, xi, i) => s + xi * w[i], b);
  const A = activations.find((a) => a.id === act);
  const out = act === "step" ? (z >= 0 ? 1 : 0) : A ? A.f(z) : z;
  const { ref, width } = useSize<HTMLDivElement>(0.4);
  const H = 240, W = width;
  const ix = 60, nx = W * 0.55, ox = W - 50;
  return (
    <LabShell
      viz={
        <div ref={ref} className="space-y-3">
          <svg width={W} height={H} role="img" aria-label="artificial neuron diagram">
            {x.map((xi, i) => {
              const y = 40 + i * 80;
              const wc = w[i] >= 0 ? C.blue : C.pink;
              return (
                <g key={i}>
                  <line x1={ix + 22} y1={y} x2={nx - 34} y2={H / 2} style={{ stroke: wc, strokeWidth: 1 + Math.abs(w[i]) * 3, opacity: 0.75 }} />
                  <text x={(ix + nx) / 2 - 8} y={(y + H / 2) / 2 - 6} className="num" style={{ fill: wc, fontSize: 11 }}>w{i + 1}={fmt(w[i], 2)}</text>
                  <circle cx={ix} cy={y} r={22} style={{ fill: C.panel, stroke: C.lineStrong }} />
                  <text x={ix} y={y + 4} textAnchor="middle" className="num" style={{ fill: C.fg, fontSize: 12 }}>{fmt(xi, 2)}</text>
                  <text x={ix - 30} y={y + 4} textAnchor="end" style={{ fill: C.muted, fontSize: 11 }}>x{i + 1}</text>
                </g>
              );
            })}
            <circle cx={nx} cy={H / 2} r={34} style={{ fill: C.panel, stroke: C.amber, strokeWidth: 2 }} />
            <text x={nx} y={H / 2 - 6} textAnchor="middle" style={{ fill: C.muted, fontSize: 11 }}>Σ + b → {act === "step" ? "step" : A?.name}</text>
            <text x={nx} y={H / 2 + 12} textAnchor="middle" className="num" style={{ fill: C.fg, fontSize: 12 }}>z={fmt(z, 3)}</text>
            <line x1={nx + 34} y1={H / 2} x2={ox - 24} y2={H / 2} style={{ stroke: C.green, strokeWidth: 2 }} />
            <circle cx={ox} cy={H / 2} r={22} style={{ fill: C.panel, stroke: C.green }} />
            <text x={ox} y={H / 2 + 4} textAnchor="middle" className="num" style={{ fill: C.fg, fontSize: 12 }}>{fmt(out, 3)}</text>
          </svg>
          <Legend items={[{ label: "positive weight", color: C.blue }, { label: "negative weight (thickness ∝ |w|)", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <Select label="Activation" value={act} onChange={setAct} options={[{ value: "step", label: "Step (perceptron)" }, ...activations.map((a) => ({ value: a.id, label: a.name }))]} />
          {x.map((v, i) => <ParameterSlider key={`x${i}`} label={`x_${i + 1}`} tex value={v} min={-2} max={2} step={0.05} onChange={(nv) => setX(x.map((q, j) => (j === i ? nv : q)))} />)}
          {w.map((v, i) => <ParameterSlider key={`w${i}`} label={`w_${i + 1}`} tex value={v} min={-2} max={2} step={0.05} onChange={(nv) => setW(w.map((q, j) => (j === i ? nv : q)))} />)}
          <ParameterSlider label="b" tex value={b} min={-2} max={2} step={0.05} onChange={setB} />
        </>
      }
      readouts={<Readouts items={[{ label: "z=\\mathbf w^\\top\\mathbf x+b", tex: true, value: fmt(z, 4) }, { label: "a=f(z)", tex: true, value: fmt(out, 4), color: C.green }]} />}
      calc={<CalcSteps steps={[{ tex: `z=${x.map((xi, i) => `(${tex(w[i], 2)})(${tex(xi, 2)})`).join("+")}+(${tex(b, 2)})=${tex(z, 4)}` }, { tex: act === "step" ? `a=\\mathbb 1[z\\ge0]=${out}` : `a=${A?.name ?? "f"}(${tex(z, 3)})=${tex(out, 4)}` }]} />}
      note={mode === "perceptron" ? "A perceptron is a linear classifier: the decision boundary w·x + b = 0 is a hyperplane. Rosenblatt's update w ← w + η(y − ŷ)x converges only if the data are linearly separable." : "Every unit in a neural network computes this: a weighted sum plus bias, passed through a non-linearity."}
    />
  );
}

/* ------------------------------ network diagram ------------------------------ */
function NetDiagram({ net, x }: { net: MLP; x: number[] }) {
  const { ref, width } = useSize<HTMLDivElement>(0.4);
  const H = 220;
  const { as } = forward(net, x);
  const L = net.sizes.length;
  const pos = (l: number, j: number) => [30 + (l / (L - 1)) * (width - 60), H / 2 + (j - (net.sizes[l] - 1) / 2) * Math.min(34, (H - 30) / Math.max(...net.sizes))] as const;
  const maxW = Math.max(0.5, ...net.W.flat(2).map(Math.abs));
  return (
    <div ref={ref}>
      <svg width={width} height={H} role="img" aria-label="network weights and activations">
        {net.W.map((Wl, l) => Wl.map((row, j) => row.map((wv, i) => {
          const [x1, y1] = pos(l, i), [x2, y2] = pos(l + 1, j);
          return <line key={`${l}-${j}-${i}`} x1={x1} y1={y1} x2={x2} y2={y2} style={{ stroke: wv >= 0 ? C.blue : C.pink, strokeWidth: 0.3 + (Math.abs(wv) / maxW) * 3, opacity: 0.25 + 0.6 * (Math.abs(wv) / maxW) }} />;
        })))}
        {net.sizes.map((n, l) => Array.from({ length: n }, (_, j) => {
          const [cx, cy] = pos(l, j);
          const a = as[l][j];
          return <circle key={`${l}-${j}`} cx={cx} cy={cy} r={8} style={{ fill: a >= 0 ? `rgb(var(--blue) / ${Math.min(1, Math.abs(a))})` : `rgb(var(--pink) / ${Math.min(1, Math.abs(a))})`, stroke: C.lineStrong }} />;
        }))}
      </svg>
    </div>
  );
}

/* ------------------------------ MLP playground ------------------------------ */
export function MLPPlayground({ lab = false }: { lab?: boolean }) {
  const [ds, setDs] = useState<DatasetId>("circles");
  const [hidden, setHidden] = useState([6, 4]);
  const [act, setAct] = useState<ActId>("tanh");
  const [lr, setLr] = useState(0.3);
  const [l2, setL2] = useState(0);
  const [seed, setSeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState<number[]>([]);
  const [tick, setTick] = useState(0);
  const th = useThemeRGB();
  const pts = useMemo(() => makeDataset(ds, 160, 0.12, 5), [ds]);
  const X = useMemo(() => pts.map((p) => [p.x, p.y]), [pts]);
  const Y = useMemo(() => pts.map((p) => p.label), [pts]);
  const netRef = useRef<MLP>(createMLP([2, ...hidden, 1], act, seed));
  useEffect(() => { netRef.current = createMLP([2, ...hidden, 1], act, seed); setEpoch(0); setLoss([]); setTick((t) => t + 1); }, [hidden, act, seed, ds]);
  const train = (steps: number) => {
    let last = 0;
    for (let i = 0; i < steps; i++) last = trainStep(netRef.current, X, Y, lr, l2).loss;
    setEpoch((e) => e + steps);
    setLoss((h) => [...h.slice(-299), last]);
    setTick((t) => t + 1);
  };
  useTicker(playing, 60, () => train(5));
  const net = netRef.current;
  const acc = pts.filter((p) => (forward(net, [p.x, p.y]).out >= 0.5 ? 1 : 0) === p.label).length / pts.length;
  const params = net.W.reduce((s, Wl) => s + Wl.length * Wl[0].length, 0) + net.b.reduce((s, bl) => s + bl.length, 0);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
            <Plot x={[-2.6, 2.6]} y={[-2.4, 2.4]} equal aspect={0.85} maxH={400} ariaLabel="neural network decision boundary"
              underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={5} deps={[tick, th.blue[0]]} f={(x, y) => {
                const pr = forward(net, [x, y]).out;
                const c = pr >= 0.5 ? th.pink : th.blue;
                const k = 0.1 + 0.4 * Math.abs(pr - 0.5) * 2;
                return [th.panel[0] + (c[0] - th.panel[0]) * k, th.panel[1] + (c[1] - th.panel[1]) * k, th.panel[2] + (c[2] - th.panel[2]) * k, 255];
              }} />}>
              {pts.map((p, i) => <Dot key={i} p={[p.x, p.y]} r={3.2} color={p.label ? C.pink : C.blue} stroke={C.bg} />)}
            </Plot>
            <div className="space-y-2">
              <NetDiagram net={net} x={[0.5, 0.5]} />
              <div>
                <div className="mb-1 text-[11.5px] text-faint">Training loss (binary cross-entropy)</div>
                <Plot x={[0, Math.max(10, loss.length)]} y={[0, Math.max(0.8, ...loss.slice(0, 5))]} aspect={0.35} minH={110} maxH={150} margin={{ l: 32, b: 20 }} ariaLabel="loss curve">
                  <Polyline pts={loss.map((l, i) => [i, l])} style={{ stroke: C.amber, strokeWidth: 1.6 }} />
                </Plot>
              </div>
            </div>
          </div>
        </div>
      }
      controls={
        <>
          <Select label="Dataset" value={ds} onChange={setDs} options={[{ value: "blobs", label: "Blobs" }, { value: "moons", label: "Moons" }, { value: "circles", label: "Circles" }, { value: "xor", label: "XOR" }, { value: "spiral", label: "Spiral (hard)" }]} />
          <div className="space-y-1.5">
            <div className="text-[12.5px] text-muted">Hidden layers</div>
            <div className="flex flex-wrap items-center gap-1.5">
              {hidden.map((h, i) => (
                <div key={i} className="flex items-center rounded-md border border-line">
                  <button className="px-2 py-1 text-muted hover:text-fg" aria-label="fewer units" onClick={() => setHidden(hidden.map((v, j) => (j === i ? Math.max(1, v - 1) : v)))}>−</button>
                  <span className="num w-5 text-center text-[12.5px]">{h}</span>
                  <button className="px-2 py-1 text-muted hover:text-fg" aria-label="more units" onClick={() => setHidden(hidden.map((v, j) => (j === i ? Math.min(12, v + 1) : v)))}>+</button>
                </div>
              ))}
              <button className="chip" onClick={() => hidden.length < 4 && setHidden([...hidden, 4])}>+ layer</button>
              {hidden.length > 0 && <button className="chip" onClick={() => setHidden(hidden.slice(0, -1))}>− layer</button>}
            </div>
          </div>
          <Segmented label="Activation" value={act} onChange={setAct} options={[{ value: "tanh", label: "tanh" }, { value: "relu", label: "ReLU" }, { value: "sigmoid", label: "sigmoid" }, { value: "linear", label: "linear" }]} />
          <ParameterSlider label="learning rate" value={lr} min={0.01} max={2} step={0.01} onChange={setLr} />
          <ParameterSlider label="L2 regularisation" value={l2} min={0} max={0.05} step={0.001} onChange={setL2} />
          <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onStep={() => train(1)} onReset={() => { setPlaying(false); setSeed((s) => s + 1); }} />
          {act === "linear" && <p className="text-[12px] text-amber">With linear activations the whole network is a single linear map — it cannot separate circles or XOR.</p>}
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{epoch}", tex: true, value: epoch }, { label: "\\text{loss}", tex: true, value: loss.length ? fmt(loss[loss.length - 1], 4) : "—" }, { label: "\\text{train accuracy}", tex: true, value: `${fmt(acc * 100, 1)}%` }, { label: "\\text{parameters}", tex: true, value: params }]} />}
      calc={<CalcSteps steps={[{ tex: `\\mathbf h^{(l)}=${act === "relu" ? "\\operatorname{ReLU}" : act === "tanh" ? "\\tanh" : act === "sigmoid" ? "\\sigma" : ""}\\big(W^{(l)}\\mathbf h^{(l-1)}+\\mathbf b^{(l)}\\big),\\quad\\hat y=\\sigma(W^{(L)}\\mathbf h^{(L-1)}+b^{(L)})` }, { tex: "W\\leftarrow W-\\eta\\,\\big(\\nabla_WL+\\lambda W\\big)", note: "full-batch gradient descent, exact backpropagation" }]} />}
      note={lab ? "Everything runs in your browser: a real multilayer perceptron with exact gradients (verified against finite differences in the test suite)." : "Watch the hidden layers bend the input space until a single sigmoid output can separate the classes."}
    />
  );
}

/* ------------------------------ backprop step-by-step ------------------------------ */
export function BackpropViz() {
  // 2-2-1 network, sigmoid everywhere, squared loss — every number shown is computed live
  const [x1, setX1] = useState(0.05);
  const [x2, setX2] = useState(0.1);
  const [t, setT] = useState(0.9);
  const [lr, setLr] = useState(0.5);
  const [W, setW] = useState({ w1: 0.15, w2: 0.2, w3: 0.25, w4: 0.3, w5: 0.4, w6: 0.45, b1: 0.35, b2: 0.6 });
  const [stage, setStage] = useState<"forward" | "backward" | "update">("forward");
  const s = (z: number) => 1 / (1 + Math.exp(-z));
  const zh1 = W.w1 * x1 + W.w2 * x2 + W.b1, zh2 = W.w3 * x1 + W.w4 * x2 + W.b1;
  const h1 = s(zh1), h2 = s(zh2);
  const zo = W.w5 * h1 + W.w6 * h2 + W.b2, o = s(zo);
  const L = 0.5 * (t - o) ** 2;
  const dLdo = o - t, dodz = o * (1 - o), dO = dLdo * dodz;
  const g5 = dO * h1, g6 = dO * h2;
  const dh1 = dO * W.w5, dh2 = dO * W.w6;
  const d1 = dh1 * h1 * (1 - h1), d2 = dh2 * h2 * (1 - h2);
  const g1 = d1 * x1, g2 = d1 * x2, g3 = d2 * x1, g4 = d2 * x2;
  const grads = { w1: g1, w2: g2, w3: g3, w4: g4, w5: g5, w6: g6, b1: d1 + d2, b2: dO };
  const step = () => setW(Object.fromEntries(Object.entries(W).map(([k, v]) => [k, v - lr * grads[k as keyof typeof grads]])) as typeof W);
  const { ref, width } = useSize<HTMLDivElement>(0.4);
  const H = 230;
  const P = { x1: [50, 60], x2: [50, 170], h1: [width / 2, 60], h2: [width / 2, 170], o: [width - 60, 115] } as Record<string, [number, number]>;
  const edges: [string, string, keyof typeof W][] = [["x1", "h1", "w1"], ["x2", "h1", "w2"], ["x1", "h2", "w3"], ["x2", "h2", "w4"], ["h1", "o", "w5"], ["h2", "o", "w6"]];
  const vals: Record<string, number> = { x1, x2, h1, h2, o };
  return (
    <LabShell
      wideControls
      viz={
        <div ref={ref} className="space-y-3">
          <svg width={width} height={H} role="img" aria-label="two-layer network with forward values and gradients">
            {edges.map(([a, b, k]) => (
              <g key={k}>
                <line x1={P[a][0]} y1={P[a][1]} x2={P[b][0]} y2={P[b][1]} style={{ stroke: stage === "forward" ? C.blue : C.pink, strokeWidth: 1.4, opacity: 0.7 }} />
                <text x={(P[a][0] * 0.4 + P[b][0] * 0.6)} y={(P[a][1] * 0.4 + P[b][1] * 0.6) - 6} textAnchor="middle" className="num" style={{ fill: stage === "forward" ? C.muted : C.pink, fontSize: 10.5 }}>
                  {stage === "forward" ? `${k}=${fmt(W[k], 3)}` : `∂L/∂${k}=${fmt(grads[k], 4)}`}
                </text>
              </g>
            ))}
            {Object.entries(P).map(([k, [cx, cy]]) => (
              <g key={k}>
                <circle cx={cx} cy={cy} r={24} style={{ fill: C.panel, stroke: k === "o" ? C.green : k.startsWith("h") ? C.amber : C.lineStrong, strokeWidth: 1.6 }} />
                <text x={cx} y={cy - 2} textAnchor="middle" style={{ fill: C.muted, fontSize: 10 }}>{k}</text>
                <text x={cx} y={cy + 11} textAnchor="middle" className="num" style={{ fill: C.fg, fontSize: 11 }}>{fmt(vals[k], 4)}</text>
              </g>
            ))}
          </svg>
          <div className="flex flex-wrap gap-1.5">
            {(["forward", "backward", "update"] as const).map((s2) => <button key={s2} className={cn("btn", stage === s2 && "border-blue text-blue")} onClick={() => setStage(s2)}>{s2 === "forward" ? "1 · Forward pass" : s2 === "backward" ? "2 · Backward pass" : "3 · Update"}</button>)}
            {stage === "update" && <button className="btn-primary h-8" onClick={step}>Apply one gradient step</button>}
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="x_1" tex value={x1} min={-1} max={1} step={0.01} onChange={setX1} />
          <ParameterSlider label="x_2" tex value={x2} min={-1} max={1} step={0.01} onChange={setX2} />
          <ParameterSlider label="target t" value={t} min={0} max={1} step={0.01} onChange={setT} />
          <ParameterSlider label="learning rate η" value={lr} min={0.05} max={5} step={0.05} onChange={setLr} />
          <button className="btn w-full" onClick={() => setW({ w1: 0.15, w2: 0.2, w3: 0.25, w4: 0.3, w5: 0.4, w6: 0.45, b1: 0.35, b2: 0.6 })}>Reset weights</button>
          <p className="text-[12px] leading-relaxed text-faint">Sigmoid activations, loss L = ½(t − o)². The hidden layer shares bias b₁, the output uses b₂.</p>
        </>
      }
      readouts={<Readouts items={[{ label: "o", tex: true, value: fmt(o, 5) }, { label: "L", tex: true, value: fmt(L, 6), color: C.pink }, { label: "\\delta_o", tex: true, value: fmt(dO, 5) }, { label: "\\delta_{h_1},\\ \\delta_{h_2}", tex: true, value: `${fmt(d1, 5)}, ${fmt(d2, 5)}` }]} />}
      calc={<CalcSteps steps={stage === "forward" ? [
        { tex: `z_{h_1}=w_1x_1+w_2x_2+b_1=${tex(zh1, 4)},\\quad h_1=\\sigma(z_{h_1})=${tex(h1, 5)}` },
        { tex: `z_{h_2}=w_3x_1+w_4x_2+b_1=${tex(zh2, 4)},\\quad h_2=${tex(h2, 5)}` },
        { tex: `z_o=w_5h_1+w_6h_2+b_2=${tex(zo, 4)},\\quad o=${tex(o, 5)},\\quad L=\\tfrac12(${tex(t, 2)}-${tex(o, 4)})^2=${tex(L, 6)}` },
      ] : stage === "backward" ? [
        { tex: `\\delta_o=\\frac{\\partial L}{\\partial z_o}=(o-t)\\,o(1-o)=(${tex(dLdo, 4)})(${tex(dodz, 4)})=${tex(dO, 5)}` },
        { tex: `\\frac{\\partial L}{\\partial w_5}=\\delta_o h_1=${tex(g5, 5)},\\quad\\frac{\\partial L}{\\partial w_6}=\\delta_o h_2=${tex(g6, 5)}` },
        { tex: `\\delta_{h_1}=\\delta_o w_5\\,h_1(1-h_1)=${tex(d1, 6)},\\quad\\frac{\\partial L}{\\partial w_1}=\\delta_{h_1}x_1=${tex(g1, 6)}`, note: "chain rule through the hidden layer" },
      ] : [
        { tex: `w_5\\leftarrow w_5-\\eta\\,\\tfrac{\\partial L}{\\partial w_5}=${tex(W.w5, 4)}-${tex(lr, 2)}\\cdot${tex(g5, 5)}=${tex(W.w5 - lr * g5, 5)}` },
        { tex: `w_1\\leftarrow ${tex(W.w1, 4)}-${tex(lr, 2)}\\cdot${tex(g1, 6)}=${tex(W.w1 - lr * g1, 6)}` },
      ]} />}
      note="All values are computed live from the current weights. Press 'Apply one gradient step' repeatedly and watch L fall."
    />
  );
}

/* ------------------------------ vanishing / exploding gradients ------------------------------ */
export function VanishingViz() {
  const [depth, setDepth] = useState(10);
  const [act, setAct] = useState<ActId>("sigmoid");
  const [init, setInit] = useState<"small" | "xavier" | "he" | "large">("xavier");
  const [width] = useState(16);
  // measure per-layer gradient norms with a deep MLP on random data
  const norms = useMemo(() => {
    const sizes = [2, ...Array(depth).fill(width), 1];
    const net = createMLP(sizes, act, 7, init === "large" ? "he" : init);
    if (init === "large") net.W = net.W.map((L) => L.map((r) => r.map((w) => w * 2.5)));
    const pts = makeDataset("circles", 64, 0.1, 3);
    const r = trainStep(net, pts.map((p) => [p.x, p.y]), pts.map((p) => p.label), 0);
    return r.gradNorms;
  }, [depth, act, init, width]);
  const logs = norms.map((n) => Math.log10(Math.max(1e-16, n)));
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <div className="text-[11.5px] text-faint">log₁₀ ‖∂L/∂W⁽ˡ⁾‖ for each layer at initialisation (layer 1 = closest to input)</div>
          <Plot x={[0.5, norms.length + 0.5]} y={[Math.min(-8, ...logs) - 0.5, Math.max(1, ...logs) + 0.5]} aspect={0.5} maxH={360} xLabel="layer" ariaLabel="gradient norm by layer">
            <Polyline pts={logs.map((v, i) => [i + 1, v])} style={{ stroke: C.pink, strokeWidth: 2 }} />
            {logs.map((v, i) => <Dot key={i} p={[i + 1, v]} r={3.5} color={C.pink} />)}
            <Polyline pts={[[0.5, 0], [norms.length + 0.5, 0]]} style={{ stroke: C.faint, strokeDasharray: "4 3" }} />
          </Plot>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="hidden layers" value={depth} min={1} max={30} step={1} onChange={setDepth} />
          <Segmented label="Activation" value={act} onChange={setAct} options={[{ value: "sigmoid", label: "sigmoid" }, { value: "tanh", label: "tanh" }, { value: "relu", label: "ReLU" }]} />
          <Select label="Initialisation" value={init} onChange={setInit} options={[{ value: "small", label: "Small (σ = 0.1)" }, { value: "xavier", label: "Xavier / Glorot" }, { value: "he", label: "He (Kaiming)" }, { value: "large", label: "Too large (2.5 × He)" }]} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\|g_1\\|\\ \\text{(first layer)}", tex: true, value: fmt(norms[0]) }, { label: "\\|g_L\\|\\ \\text{(last layer)}", tex: true, value: fmt(norms[norms.length - 1]) }, { label: "\\text{ratio}", tex: true, value: fmt(norms[0] / norms[norms.length - 1]) }]} />}
      calc={<CalcSteps steps={[{ tex: "\\frac{\\partial L}{\\partial\\mathbf h^{(1)}}=\\Big(\\prod_{l=2}^{L}W^{(l)\\top}\\operatorname{diag}(f'(\\mathbf z^{(l)}))\\Big)\\frac{\\partial L}{\\partial\\mathbf h^{(L)}}", note: "a product of many factors: shrinks or grows exponentially with depth" }, { tex: "\\sigma'(z)\\le0.25\\Rightarrow\\text{each sigmoid layer can shrink the signal by }4\\times" }]} />}
      note="Fixes: ReLU-family activations, careful initialisation (Xavier for tanh, He for ReLU), normalisation layers, residual connections and gradient clipping. Measured from a real backward pass of a randomly initialised network."
    />
  );
}

export { FnPath };
