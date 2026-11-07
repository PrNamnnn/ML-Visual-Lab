"use client";
import { useState } from "react";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { MatrixInput, Toggle, ParameterSlider, Segmented, Presets } from "@/components/controls";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { Tex } from "@/components/mathematics/Tex";
import { softmax } from "@/lib/nn";
import { fmt, tex, texMatrix, cn } from "@/lib/utilities/format";

const TOK = ["The", "cat", "sat", "down"];
const mm = (A: number[][], B: number[][]) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const tr = (A: number[][]) => A[0].map((_, j) => A.map((r) => r[j]));

/** Scaled dot-product attention with directly editable Q, K, V (d_k = 3). */
export default function Attention({ mode: initial = "self", lab = false }: { mode?: "self" | "causal" | "multihead" | "cross"; lab?: boolean }) {
  const [mode, setMode] = useState(initial);
  const [Q, setQ] = useState([[1, 0, 1], [0, 2, 0], [1, 1, 0], [0, 1, 1]]);
  const [K, setK] = useState([[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]]);
  const [V, setV] = useState([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]);
  const [causal, setCausal] = useState(initial === "causal");
  const [scale, setScale] = useState(true);
  const [temp, setTemp] = useState(1);
  const [row, setRow] = useState(1);
  const dk = 3;
  const S = mm(Q, tr(K)).map((r) => r.map((v) => (scale ? v / Math.sqrt(dk) : v) / temp));
  const Sm = S.map((r, i) => r.map((v, j) => (causal && j > i ? -Infinity : v)));
  const A = Sm.map((r) => softmax(r.map((v) => (Number.isFinite(v) ? v : -1e9))));
  const O = mm(A, V);
  // second head: a different (fixed) projection of the same Q/K to show heads attending differently
  const P2 = [[0, 1, 0], [1, 0, 0], [0, 0, -1]];
  const Q2 = mm(Q, P2), K2 = mm(K, P2);
  const A2 = mm(Q2, tr(K2)).map((r, i) => softmax(r.map((v, j) => (causal && j > i ? -1e9 : v / Math.sqrt(dk) / temp))));
  const O2 = mm(A2, V);
  const qi = Math.min(row, TOK.length - 1);
  const ent = A[qi].reduce((s, p) => s - (p > 0 ? p * Math.log(p) : 0), 0);
  const Heat = ({ W, title }: { W: number[][]; title: string }) => (
    <div className="inline-flex flex-col gap-1">
      <div className="text-[12px] text-muted">{title}</div>
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: `56px repeat(${TOK.length}, 44px)` }}>
        <span />
        {TOK.map((t) => <span key={t} className="text-center text-[11px] text-faint">{t}</span>)}
        {W.map((r, i) => (
          <div key={i} className="contents">
            <button className={cn("pr-1 text-right text-[11px]", i === qi ? "text-fg" : "text-faint")} onClick={() => setRow(i)}>{TOK[i]}</button>
            {r.map((v, j) => (
              <div key={j} className={cn("num flex h-9 items-center justify-center rounded-[3px] text-[11px]", i === qi && "ring-1 ring-amber")} style={{ background: causal && j > i ? "rgb(var(--line) / 0.5)" : `rgb(var(--pink) / ${v * 0.85})` }}>
                {causal && j > i ? "—" : fmt(v, 2)}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-4 overflow-x-auto">
          <div className="flex flex-wrap items-start justify-center gap-5">
            <Heat W={A} title={mode === "multihead" ? "head 1 attention weights" : "attention weights softmax(QKᵀ/√dₖ)"} />
            {mode === "multihead" ? <Heat W={A2} title="head 2 (different projection)" /> : <MatrixView M={S.map((r, i) => r.map((v, j) => (causal && j > i ? NaN : v)))} label="scores QKᵀ/√dₖ" digits={2} cell={44} />}
          </div>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <MatrixView M={A} label="A" digits={2} cell={40} sequential />
            <span className="text-faint">×</span>
            <MatrixView M={V} label="V" digits={1} cell={36} />
            <span className="text-faint">=</span>
            <MatrixView M={O} label={mode === "multihead" ? "head 1 output" : "output = AV"} digits={2} cell={44} hl={(i) => (i === qi ? "out" : null)} />
            {mode === "multihead" && <MatrixView M={O2} label="head 2 output" digits={2} cell={44} />}
          </div>
          {mode === "multihead" && <p className="text-center text-[12px] text-faint">Head outputs are concatenated ([head₁ ‖ head₂], width 6) and mixed by W_O.</p>}
        </div>
      }
      controls={
        <>
          {lab && <Segmented value={mode} onChange={setMode} options={[{ value: "self", label: "Self" }, { value: "causal", label: "Masked" }, { value: "multihead", label: "Multi-head" }]} />}
          <div className="grid grid-cols-1 gap-3">
            <MatrixInput label="Q (queries, one row per token)" value={Q} onChange={setQ} step={0.5} />
            <MatrixInput label="K (keys)" value={K} onChange={setK} step={0.5} />
            <MatrixInput label="V (values)" value={V} onChange={setV} step={0.5} />
          </div>
          <Toggle label="Causal mask (no attending to future tokens)" checked={causal} onChange={setCausal} />
          <Toggle label="Scale by 1/√dₖ" checked={scale} onChange={setScale} />
          <ParameterSlider label="softmax temperature" value={temp} min={0.1} max={4} step={0.05} onChange={setTemp} />
          <Presets presets={[{ label: "Sharp (×4 queries)", value: 4 }, { label: "Uniform (zero queries)", value: 0 }]} onPick={(k) => setQ(Q.map((r) => r.map((v) => (k === 0 ? 0 : v * k))))} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{query token}", tex: true, value: TOK[qi] }, { label: "\\text{attends most to}", tex: true, value: TOK[A[qi].indexOf(Math.max(...A[qi]))] }, { label: "\\sum_j\\alpha_{ij}", tex: true, value: fmt(A[qi].reduce((a, b) => a + b, 0), 6) }, { label: "\\text{entropy}", tex: true, value: fmt(ent, 3) }]} />}
      calc={<CalcSteps steps={[
        { tex: `s_{${qi + 1}j}=\\frac{\\mathbf q_{${qi + 1}}\\cdot\\mathbf k_j}{\\sqrt{${dk}}}:\\ (${S[qi].map((v, j) => (causal && j > qi ? "-\\infty" : tex(v, 3))).join(",\\;")})` },
        { tex: `\\boldsymbol\\alpha_{${qi + 1}}=\\operatorname{softmax}(\\mathbf s_{${qi + 1}})=(${A[qi].map((v) => tex(v, 3)).join(",\\;")})` },
        { tex: `\\mathbf o_{${qi + 1}}=\\sum_j\\alpha_{${qi + 1}j}\\mathbf v_j=(${O[qi].map((v) => tex(v, 3)).join(",\\;")})` },
      ]} />}
      note={<span><Tex tex={String.raw`\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V`} />. Click a token name to inspect its row. Without the √dₖ scaling, large dot products push softmax into saturation and gradients vanish.</span>}
    />
  );
}

export { texMatrix };
