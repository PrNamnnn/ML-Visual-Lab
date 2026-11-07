"use client";
import { useState } from "react";
import { Plot, FnPath, C, Dot, Handle } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle } from "@/components/controls";
import { activations, losses, lossFns, softmax, klDivergence, crossEntropy } from "@/lib/nn";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, tex } from "@/lib/utilities/format";

/* ------------------------------ activation functions ------------------------------ */
export function ActivationViz({ id: initial = "relu" }: { id?: string }) {
  const [id, setId] = useState(initial);
  const [x, setX] = useState(0.8);
  const [compare, setCompare] = useState(false);
  const [logits, setLogits] = useState([2, 1, 0.1, -1]);
  const [temp, setTemp] = useState(1);
  const [param, setParam] = useState<number | null>(null);
  if (id === "softmax") {
    const p = softmax(logits, temp);
    return (
      <LabShell
        viz={
          <div className="space-y-3 py-2">
            {logits.map((z, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="num w-16 text-[12px] text-muted">z{i + 1} = {fmt(z, 2)}</span>
                <div className="h-6 flex-1 rounded bg-raised"><div className="h-6 rounded" style={{ width: `${p[i] * 100}%`, background: "rgb(var(--blue))" }} /></div>
                <span className="num w-14 text-right text-[12.5px]">{fmt(p[i], 3)}</span>
              </div>
            ))}
            <div className="text-[13px]"><Tex tex={"\\operatorname{softmax}(\\mathbf z)_i=\\dfrac{e^{z_i/T}}{\\sum_je^{z_j/T}}"} block /></div>
          </div>
        }
        controls={
          <>
            <Select label="Function" value={id} onChange={setId} options={[...activations.map((a) => ({ value: a.id, label: a.name })), { value: "softmax", label: "Softmax (vector)" }]} />
            {logits.map((z, i) => <ParameterSlider key={i} label={`z_${i + 1}`} tex value={z} min={-5} max={5} step={0.1} onChange={(v) => setLogits(logits.map((q, j) => (j === i ? v : q)))} />)}
            <ParameterSlider label="temperature T" tex value={temp} min={0.1} max={5} step={0.05} onChange={setTemp} />
          </>
        }
        readouts={<Readouts items={[{ label: "\\sum p_i", tex: true, value: fmt(p.reduce((a, b) => a + b, 0), 6) }, { label: "\\arg\\max", tex: true, value: `class ${p.indexOf(Math.max(...p)) + 1}` }, { label: "\\text{entropy}", tex: true, value: fmt(-p.reduce((s, q) => s + (q > 0 ? q * Math.log(q) : 0), 0), 4) }]} />}
        calc={<CalcSteps steps={[{ tex: `\\text{stable form: subtract }\\max_j z_j=${tex(Math.max(...logits), 2)}\\text{ before exponentiating}` }, { tex: `p=(${p.map((q) => tex(q, 3)).join(",\\;")})` }]} />}
        note="Softmax is invariant to adding a constant to every logit. Low temperature sharpens toward a one-hot argmax; high temperature flattens toward uniform."
      />
    );
  }
  const A = activations.find((a) => a.id === id) ?? activations[0];
  const a = param ?? A.param?.default;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-5, 5]} y={[-2, 3.5]} aspect={0.55} maxH={420} originAxes ariaLabel={`${A.name} and its derivative`}>
            {compare && activations.filter((b) => b.id !== A.id && ["sigmoid", "tanh", "relu", "gelu"].includes(b.id)).map((b) => <FnPath key={b.id} f={(t) => b.f(t)} style={{ stroke: C.a("fg", 0.18), strokeWidth: 1 }} />)}
            <FnPath f={(t) => A.df(t, a)} style={{ stroke: C.pink, strokeWidth: 1.6, strokeDasharray: "5 3" }} />
            <FnPath f={(t) => A.f(t, a)} n={500} />
            <Dot p={[x, A.df(x, a)]} r={4} color={C.pink} />
            <Handle p={[x, A.f(x, a)]} onDrag={(q) => setX(Math.max(-5, Math.min(5, q[0])))} color="blue" label="input x" />
          </Plot>
          <Legend items={[{ label: "f(x)", color: C.blue }, { label: "f′(x) — the gradient that flows back", color: C.pink, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <Select label="Function" value={id} onChange={(v) => { setId(v); setParam(null); }} options={[...activations.map((b) => ({ value: b.id, label: b.name })), { value: "softmax", label: "Softmax (vector)" }]} />
          <ParameterSlider label="input x" value={x} min={-5} max={5} step={0.01} onChange={setX} />
          {A.param && <ParameterSlider label={A.param.label} tex value={a ?? A.param.default} min={A.param.min} max={A.param.max} step={A.param.step} onChange={setParam} />}
          <Toggle label="Faint overlay of other activations" checked={compare} onChange={setCompare} />
          <div className="space-y-1 text-[12.5px]"><Tex tex={A.tex} block /><Tex tex={A.dtex} block /></div>
        </>
      }
      readouts={<Readouts items={[{ label: "f(x)", tex: true, value: fmt(A.f(x, a), 5) }, { label: "f'(x)", tex: true, value: fmt(A.df(x, a), 5), color: C.pink }, { label: "\\text{range}", tex: true, value: <Tex tex={A.range} /> }, { label: "\\text{zero-centred?}", tex: true, value: ["tanh", "linear", "elu", "selu", "leaky-relu", "prelu"].includes(A.id) ? "approximately" : "no" }]} />}
      calc={<CalcSteps steps={[{ tex: `f(${tex(x, 2)})=${tex(A.f(x, a), 5)},\\quad f'(${tex(x, 2)})=${tex(A.df(x, a), 5)}` }]} />}
      note={A.notes}
    />
  );
}

/* ------------------------------ loss functions ------------------------------ */
export function LossViz({ id: initial = "bce" }: { id?: string }) {
  const [id, setId] = useState(initial);
  const L = losses.find((l) => l.id === id) ?? losses[0];
  const [y, setY] = useState(1);
  const [pred, setPred] = useState(0.7);
  const [delta, setDelta] = useState(1);
  const [gamma, setGamma] = useState(2);
  const [margin, setMargin] = useState(1);
  const [P, setP] = useState([0.6, 0.3, 0.1]);
  const [Q, setQ] = useState([0.4, 0.4, 0.2]);
  if (L.kind === "multiclass" || L.kind === "distribution") {
    const norm = (v: number[]) => { const s = v.reduce((a, b) => a + b, 0) || 1; return v.map((x) => x / s); };
    const p = norm(P), q = norm(Q);
    const ce = crossEntropy(p, q), kl = klDivergence(p, q), H = -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);
    return (
      <LabShell
        viz={
          <div className="space-y-2 py-2">
            {p.map((pv, i) => (
              <div key={i} className="grid grid-cols-[60px_1fr] items-center gap-2 text-[12px]">
                <span className="text-muted">class {i + 1}</span>
                <div className="space-y-1">
                  <div className="h-3 rounded bg-raised"><div className="h-3 rounded" style={{ width: `${pv * 100}%`, background: "rgb(var(--green))" }} /></div>
                  <div className="h-3 rounded bg-raised"><div className="h-3 rounded" style={{ width: `${q[i] * 100}%`, background: "rgb(var(--blue))" }} /></div>
                </div>
              </div>
            ))}
            <Legend items={[{ label: L.kind === "multiclass" ? "target y (P)" : "P (true)", color: C.green }, { label: L.kind === "multiclass" ? "prediction p (Q)" : "Q (model)", color: C.blue }]} />
          </div>
        }
        controls={
          <>
            <Select label="Loss" value={id} onChange={setId} options={losses.map((l) => ({ value: l.id, label: l.name }))} />
            <div className="label">Target P</div>
            {P.map((v, i) => <ParameterSlider key={`p${i}`} label={`P_${i + 1}`} tex value={v} min={0} max={1} step={0.01} onChange={(nv) => setP(P.map((x, j) => (j === i ? nv : x)))} />)}
            <div className="label">Prediction Q</div>
            {Q.map((v, i) => <ParameterSlider key={`q${i}`} label={`Q_${i + 1}`} tex value={v} min={0.001} max={1} step={0.01} onChange={(nv) => setQ(Q.map((x, j) => (j === i ? nv : x)))} />)}
            <button className="btn w-full" onClick={() => setP([1, 0, 0])}>Make target one-hot</button>
          </>
        }
        readouts={<Readouts items={[{ label: "H(P,Q)", tex: true, value: fmt(ce, 4) }, { label: "H(P)", tex: true, value: fmt(H, 4) }, { label: "D_{KL}(P\\|Q)", tex: true, value: fmt(kl, 4), color: C.pink }, { label: "D_{KL}(Q\\|P)", tex: true, value: fmt(klDivergence(q, p), 4) }]} />}
        calc={<CalcSteps steps={[{ tex: L.tex }, { tex: `H(P,Q)=H(P)+D_{KL}(P\\|Q):\\ ${tex(ce, 4)}=${tex(H, 4)}+${tex(kl, 4)}` }]} />}
        note={L.notes + " Sliders are renormalised so each vector sums to 1."}
      />
    );
  }
  const isBin = L.kind === "binary";
  const isMetric = L.kind === "metric";
  const t = y ? 1 : -1;
  const f = (v: number): number => {
    switch (L.id) {
      case "mse": return lossFns.mse(y, v);
      case "mae": return lossFns.mae(y, v);
      case "huber": return lossFns.huber(y, v, delta);
      case "bce": return lossFns.bce(y, v);
      case "focal": return lossFns.focal(y, v, gamma);
      case "hinge": return lossFns.hinge(t, v);
      case "contrastive": return lossFns.contrastive(y, v, margin);
      case "triplet": return lossFns.triplet(v, 1, margin);
      default: return NaN;
    }
  };
  const xr: [number, number] = L.id === "hinge" ? [-3, 3] : isBin ? [0.001, 0.999] : isMetric ? [0, 3] : [-3, 3];
  const yv = f(pred);
  const dl = (f(pred + 1e-5) - f(pred - 1e-5)) / 2e-5;
  const xLabel = L.id === "hinge" ? "score s" : isBin ? "predicted p(y=1)" : L.id === "contrastive" ? "distance d" : L.id === "triplet" ? "d(anchor, positive) — d(anchor, negative) fixed at 1" : "prediction ŷ";
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={xr} y={[0, isBin ? 5 : 6]} aspect={0.55} maxH={400} xLabel={xLabel} ariaLabel={`${L.name} as a function of the prediction`}>
            {L.id === "focal" && <FnPath f={(v) => lossFns.bce(y, v)} style={{ stroke: C.a("fg", 0.3), strokeWidth: 1, strokeDasharray: "4 3" }} />}
            {(L.id === "huber" || L.id === "mae") && <FnPath f={(v) => lossFns.mse(y, v)} style={{ stroke: C.a("fg", 0.25), strokeWidth: 1, strokeDasharray: "4 3" }} />}
            <FnPath f={f} n={500} />
            <Handle p={[pred, Math.min(6, yv)]} onDrag={(q) => setPred(Math.max(xr[0], Math.min(xr[1], q[0])))} color="pink" label="prediction" />
          </Plot>
          <Legend items={[{ label: L.name, color: C.blue }, ...(L.id === "focal" ? [{ label: "plain cross-entropy (γ = 0)", color: C.a("fg", 0.4), dashed: true }] : []), ...(L.id === "huber" || L.id === "mae" ? [{ label: "squared error for comparison", color: C.a("fg", 0.4), dashed: true }] : [])]} />
        </div>
      }
      controls={
        <>
          <Select label="Loss" value={id} onChange={setId} options={losses.map((l) => ({ value: l.id, label: l.name }))} />
          {(isBin || L.id === "contrastive") && <Select label={L.id === "contrastive" ? "pair label y" : "true label y"} value={String(y)} onChange={(v) => setY(Number(v))} options={[{ value: "1", label: L.id === "contrastive" ? "1 — similar pair" : "1 (positive)" }, { value: "0", label: L.id === "contrastive" ? "0 — dissimilar pair" : L.id === "hinge" ? "0 (t = −1)" : "0 (negative)" }]} />}
          {L.kind === "regression" && <ParameterSlider label="true value y" value={y} min={-2} max={2} step={0.1} onChange={setY} />}
          <ParameterSlider label={xLabel.split(" —")[0]} value={pred} min={xr[0]} max={xr[1]} step={0.001} onChange={setPred} />
          {L.id === "huber" && <ParameterSlider label="δ" tex value={delta} min={0.1} max={3} step={0.05} onChange={setDelta} />}
          {L.id === "focal" && <ParameterSlider label="γ" tex value={gamma} min={0} max={5} step={0.1} onChange={setGamma} />}
          {(L.id === "contrastive" || L.id === "triplet") && <ParameterSlider label="margin m" value={margin} min={0.1} max={2} step={0.05} onChange={setMargin} />}
        </>
      }
      readouts={<Readouts items={[{ label: "L", tex: true, value: fmt(yv, 5), color: C.pink }, { label: "\\partial L/\\partial\\hat y", tex: true, value: fmt(dl, 4) }]} />}
      calc={<CalcSteps steps={[{ tex: L.tex }, { tex: `L(${tex(pred, 3)})=${tex(yv, 5)}` }]} />}
      note={L.notes}
    />
  );
}
