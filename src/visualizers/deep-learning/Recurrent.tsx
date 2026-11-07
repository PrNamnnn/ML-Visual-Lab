"use client";
import { useState } from "react";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented } from "@/components/controls";
import { Plot, Polyline, Dot, C } from "@/components/visualization/Plot";
import { useSize } from "@/hooks/useSize";
import { fmt, tex } from "@/lib/utilities/format";

const sig = (z: number) => 1 / (1 + Math.exp(-z));
const SEQ = [1, 0.5, -0.5, 0, 0, 0, 0, 0, 0.8, -1, 0, 0];

/** Scalar RNN / LSTM unrolled over a fixed input sequence; all values computed live. */
export default function Recurrent({ cell: initial = "rnn" }: { cell?: "rnn" | "lstm" | "gru" }) {
  const [cell, setCell] = useState(initial);
  const [wh, setWh] = useState(0.9);
  const [wx, setWx] = useState(1);
  const [fb, setFb] = useState(1.5); // forget-gate bias
  const T = SEQ.length;
  const hs: number[] = [], cs: number[] = [], gates: { f: number; i: number; o: number; g: number; z?: number; r?: number }[] = [], dhdh: number[] = [];
  let h = 0, c = 0;
  for (let t = 0; t < T; t++) {
    const x = SEQ[t];
    if (cell === "rnn") {
      const a = wh * h + wx * x;
      h = Math.tanh(a);
      dhdh.push(wh * (1 - h * h)); // ∂h_t/∂h_{t-1}
    } else if (cell === "lstm") {
      const f = sig(0.5 * h + 0.3 * x + fb), i = sig(0.4 * h + wx * x - 0.5), o = sig(0.3 * h + 0.5 * x + 0.5), g = Math.tanh(wh * h + wx * x);
      c = f * c + i * g;
      h = o * Math.tanh(c);
      gates.push({ f, i, o, g });
      dhdh.push(f); // dominant path ∂c_t/∂c_{t-1} = f_t
    } else {
      const z = sig(0.5 * h + 0.6 * x - fb + 1), r = sig(0.4 * h + 0.5 * x), hh = Math.tanh(wh * r * h + wx * x);
      h = (1 - z) * h + z * hh;
      gates.push({ f: 1 - z, i: z, o: r, g: hh, z, r });
      dhdh.push(1 - z);
    }
    hs.push(h); cs.push(c);
  }
  // gradient of h_T w.r.t. h_t via product of local Jacobians (dominant path)
  const grad = hs.map((_, t) => dhdh.slice(t + 1).reduce((p, v) => p * v, 1));
  const { ref, width } = useSize<HTMLDivElement>(0.3);
  const cw = Math.max(44, (width - 20) / T);
  return (
    <LabShell
      viz={
        <div ref={ref} className="space-y-3">
          <div className="overflow-x-auto">
            <svg width={Math.max(width, T * cw)} height={cell === "rnn" ? 150 : 190} role="img" aria-label="unrolled recurrent network">
              {SEQ.map((x, t) => {
                const cx = 10 + t * cw + cw / 2;
                return (
                  <g key={t}>
                    {t > 0 && <line x1={cx - cw + 18} y1={70} x2={cx - 18} y2={70} style={{ stroke: C.amber, strokeWidth: 1 + Math.min(4, Math.abs(grad[t - 1]) * 3), opacity: 0.4 + 0.6 * Math.min(1, Math.abs(grad[t - 1])) }} />}
                    <text x={cx} y={128} textAnchor="middle" className="num" style={{ fill: C.muted, fontSize: 10.5 }}>x={fmt(x, 1)}</text>
                    <line x1={cx} y1={118} x2={cx} y2={88} style={{ stroke: C.lineStrong }} />
                    <rect x={cx - 18} y={52} width={36} height={36} rx={6} style={{ fill: `rgb(var(--${hs[t] >= 0 ? "blue" : "pink"}) / ${Math.min(0.8, Math.abs(hs[t]))})`, stroke: C.lineStrong }} />
                    <text x={cx} y={74} textAnchor="middle" className="num" style={{ fill: C.fg, fontSize: 10.5 }}>{fmt(hs[t], 2)}</text>
                    <text x={cx} y={42} textAnchor="middle" style={{ fill: C.faint, fontSize: 10 }}>t={t + 1}</text>
                    {cell !== "rnn" && (
                      <text x={cx} y={150} textAnchor="middle" className="num" style={{ fill: C.green, fontSize: 9.5 }}>
                        {cell === "lstm" ? `f=${fmt(gates[t].f, 2)}` : `z=${fmt(gates[t].z ?? 0, 2)}`}
                      </text>
                    )}
                    {cell === "lstm" && <text x={cx} y={166} textAnchor="middle" className="num" style={{ fill: C.amber, fontSize: 9.5 }}>c={fmt(cs[t], 2)}</text>}
                  </g>
                );
              })}
            </svg>
          </div>
          <div>
            <div className="mb-1 text-[11.5px] text-faint">Gradient signal |∂h_T / ∂h_t| reaching step t from the last step (log scale)</div>
            <Plot x={[0.5, T + 0.5]} y={[-8, 0.5]} aspect={0.22} minH={120} maxH={160} margin={{ l: 32, b: 22 }} ariaLabel="gradient magnitude by time step">
              <Polyline pts={grad.map((g, t) => [t + 1, Math.log10(Math.max(1e-8, Math.abs(g)))])} style={{ stroke: C.amber, strokeWidth: 2 }} />
              {grad.map((g, t) => <Dot key={t} p={[t + 1, Math.log10(Math.max(1e-8, Math.abs(g)))]} r={3} color={C.amber} />)}
            </Plot>
          </div>
          <Legend items={[{ label: "hidden state hₜ (colour = sign, opacity = magnitude)", color: C.blue }, { label: "recurrent edge: width ∝ gradient that flows back through it", color: C.amber }]} />
        </div>
      }
      controls={
        <>
          <Segmented value={cell} onChange={setCell} options={[{ value: "rnn", label: "Vanilla RNN" }, { value: "lstm", label: "LSTM" }, { value: "gru", label: "GRU" }]} />
          <ParameterSlider label="recurrent weight w_h" tex value={wh} min={0} max={3} step={0.05} onChange={setWh} />
          <ParameterSlider label="input weight w_x" tex value={wx} min={0} max={3} step={0.05} onChange={setWx} />
          {cell !== "rnn" && <ParameterSlider label={cell === "lstm" ? "forget-gate bias b_f" : "update-gate bias offset"} tex value={fb} min={-3} max={4} step={0.1} onChange={setFb} />}
          <p className="text-[12px] leading-relaxed text-faint">Input sequence: an early signal, a long gap of zeros, then more inputs. Can information from t = 1 still influence the final state?</p>
        </>
      }
      readouts={<Readouts items={[{ label: "h_T", tex: true, value: fmt(hs[T - 1], 4) }, { label: "|\\partial h_T/\\partial h_1|", tex: true, value: fmt(Math.abs(grad[0]), 6), color: C.amber }, ...(cell === "lstm" ? [{ label: "c_T", tex: true, value: fmt(cs[T - 1], 4) }] : [])]} />}
      calc={<CalcSteps steps={cell === "rnn" ? [
        { tex: `h_t=\\tanh(w_hh_{t-1}+w_xx_t),\\quad\\frac{\\partial h_t}{\\partial h_{t-1}}=w_h(1-h_t^2)` },
        { tex: `\\frac{\\partial h_T}{\\partial h_1}=\\prod_{t=2}^{T}w_h(1-h_t^2)=${tex(grad[0], 6)}`, note: "|w_h| < 1 vanishes, large w_h can explode before tanh saturates" },
      ] : cell === "lstm" ? [
        { tex: "f_t=\\sigma(\\cdot),\\ i_t=\\sigma(\\cdot),\\ o_t=\\sigma(\\cdot),\\ \\tilde c_t=\\tanh(\\cdot)" },
        { tex: "c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde c_t,\\quad h_t=o_t\\odot\\tanh(c_t)" },
        { tex: `\\frac{\\partial c_T}{\\partial c_1}\\approx\\prod_t f_t=${tex(grad[0], 6)}`, note: "the additive cell path: gradients survive when forget gates stay near 1" },
      ] : [
        { tex: "z_t=\\sigma(\\cdot),\\ r_t=\\sigma(\\cdot),\\ \\tilde h_t=\\tanh(w_h(r_t\\odot h_{t-1})+w_xx_t)" },
        { tex: "h_t=(1-z_t)\\odot h_{t-1}+z_t\\odot\\tilde h_t", note: "1 − z acts like an LSTM forget gate" },
      ]} />}
      note="Gate pre-activations use fixed illustrative weights (shown values are computed exactly from them). Backpropagation through time multiplies one local Jacobian per step — the root of vanishing/exploding gradients that LSTMs and GRUs mitigate with gated additive paths."
    />
  );
}
