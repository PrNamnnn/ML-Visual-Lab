"use client";
import { useEffect, useMemo, useState } from "react";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { MatrixInput, Select, Segmented, AnimationControls, Toggle } from "@/components/controls";
import { useTicker } from "@/hooks/useSize";
import { fmt, tex, cn } from "@/lib/utilities/format";

const KERNELS: Record<string, number[][]> = {
  "edge (Laplacian)": [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
  "Sobel x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  "Sobel y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
  "box blur": [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
};

const INPUT: number[][] = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 1, 1, 0, 0, 1, 1, 0],
  [0, 1, 0, 0, 0, 0, 1, 0],
  [0, 1, 0, 0, 0, 0, 1, 0],
  [0, 1, 1, 0, 0, 1, 1, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
];

export function conv2d(X: number[][], K: number[][], pad: number, stride: number) {
  const n = X.length, k = K.length;
  const P = Array.from({ length: n + 2 * pad }, (_, i) => Array.from({ length: n + 2 * pad }, (_, j) => X[i - pad]?.[j - pad] ?? 0));
  const out = Math.floor((n + 2 * pad - k) / stride) + 1;
  const Y = Array.from({ length: out }, (_, i) => Array.from({ length: out }, (_, j) => {
    let s = 0;
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) s += P[i * stride + a][j * stride + b] * K[a][b];
    return s;
  }));
  return { P, Y, out };
}

function Grid({ M, cell = 30, hl, color, title, digits = 1, onClick }: { M: number[][]; cell?: number; hl?: (i: number, j: number) => boolean; color: "blue" | "pink" | "green"; title: string; digits?: number; onClick?: (i: number, j: number) => void }) {
  const max = Math.max(1e-9, ...M.flat().map(Math.abs));
  return (
    <div className="inline-flex flex-col items-center gap-1">
      <div className="text-[12px] text-muted">{title}</div>
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${M[0].length}, ${cell}px)` }}>
        {M.map((r, i) => r.map((v, j) => (
          <button
            key={`${i}-${j}`}
            onClick={onClick ? () => onClick(i, j) : undefined}
            tabIndex={onClick ? 0 : -1}
            aria-label={`${title} cell ${i + 1},${j + 1}: ${fmt(v, 2)}`}
            className={cn("num flex items-center justify-center rounded-[3px] text-[10px]", hl?.(i, j) && "ring-2 ring-amber", !onClick && "cursor-default")}
            style={{ height: cell, background: v >= 0 ? `rgb(var(--${color}) / ${(Math.abs(v) / max) * 0.7})` : `rgb(var(--pink) / ${(Math.abs(v) / max) * 0.7})` }}
          >
            {cell >= 26 ? fmt(v, digits) : ""}
          </button>
        )))}
      </div>
    </div>
  );
}

export default function Convolution({ mode = "conv", lab = false }: { mode?: "conv" | "pooling"; lab?: boolean }) {
  const [view, setView] = useState(mode);
  const [X, setX] = useState(INPUT);
  const [kname, setKname] = useState("edge (Laplacian)");
  const [K, setK] = useState(KERNELS["edge (Laplacian)"]);
  const [pad, setPad] = useState(0);
  const [stride, setStride] = useState(1);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [relu, setRelu] = useState(false);
  const [pool, setPool] = useState<"max" | "avg">("max");
  const { P, Y, out } = useMemo(() => conv2d(X, K, pad, stride), [X, K, pad, stride]);
  const Yr = relu ? Y.map((r) => r.map((v) => Math.max(0, v))) : Y;
  const total = out * out;
  useEffect(() => setPos(0), [pad, stride, K]);
  useTicker(playing, 450, () => setPos((p) => { if (p + 1 >= total) { setPlaying(false); return p; } return p + 1; }));
  const oi = Math.floor(pos / out), oj = pos % out;
  const k = K.length;
  const inWin = (i: number, j: number) => i >= oi * stride && i < oi * stride + k && j >= oj * stride && j < oj * stride + k;
  const terms = K.flatMap((r, a) => r.map((kv, b) => ({ x: P[oi * stride + a][oj * stride + b], k: kv }))).filter((t) => t.k !== 0);

  // pooling of the (ReLU'd) feature map, 2×2 stride 2
  const pooled = useMemo(() => {
    const m = Math.floor(Yr.length / 2);
    return Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => {
      const w = [Yr[2 * i][2 * j], Yr[2 * i][2 * j + 1], Yr[2 * i + 1][2 * j], Yr[2 * i + 1][2 * j + 1]];
      return pool === "max" ? Math.max(...w) : w.reduce((a, b) => a + b, 0) / 4;
    }));
  }, [Yr, pool]);
  const pm = pooled.length;
  const [pp, setPp] = useState(0);
  const pi = Math.floor(Math.min(pp, pm * pm - 1) / Math.max(1, pm)), pj = Math.min(pp, pm * pm - 1) % Math.max(1, pm);

  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-4 overflow-x-auto">
          {view === "conv" ? (
            <div className="flex flex-wrap items-start justify-center gap-5">
              <Grid M={P} color="blue" title={`input ${X.length}×${X.length}${pad ? ` (+ padding ${pad})` : ""} — click to toggle`} hl={inWin} digits={0}
                onClick={(i, j) => { const ii = i - pad, jj = j - pad; if (ii >= 0 && jj >= 0 && ii < X.length && jj < X.length) setX(X.map((r, a) => r.map((v, b) => (a === ii && b === jj ? (v ? 0 : 1) : v)))); }} />
              <Grid M={K} color="green" title="kernel 3×3" digits={2} cell={34} />
              <Grid M={Yr} color="pink" title={`feature map ${out}×${out}`} hl={(i, j) => i === oi && j === oj} digits={1} onClick={(i, j) => setPos(i * out + j)} />
            </div>
          ) : (
            <div className="flex flex-wrap items-start justify-center gap-6">
              <Grid M={Yr} color="pink" title={`feature map ${out}×${out}${relu ? " (after ReLU)" : ""}`} hl={(i, j) => Math.floor(i / 2) === pi && Math.floor(j / 2) === pj && i < 2 * pm && j < 2 * pm} digits={1} />
              <Grid M={pooled} color="blue" title={`${pool} pool 2×2, stride 2 → ${pm}×${pm}`} hl={(i, j) => i === pi && j === pj} digits={2} cell={36} onClick={(i, j) => setPp(i * pm + j)} />
            </div>
          )}
        </div>
      }
      controls={
        <>
          {lab && <Segmented value={view} onChange={setView} options={[{ value: "conv", label: "Convolution" }, { value: "pooling", label: "Pooling" }]} />}
          <Select label="Kernel" value={kname} onChange={(v) => { setKname(v); if (v !== "custom") setK(KERNELS[v]); }} options={[...Object.keys(KERNELS).map((k2) => ({ value: k2, label: k2 })), { value: "custom", label: "custom (edit below)" }]} />
          <MatrixInput label="Kernel weights" value={K.map((r) => r.map((v) => Math.round(v * 1000) / 1000))} onChange={(m) => { setKname("custom"); setK(m); }} step={0.1} />
          <Segmented label="Padding" value={String(pad)} onChange={(v) => setPad(Number(v))} options={[{ value: "0", label: "valid (0)" }, { value: "1", label: "same (1)" }]} />
          <Segmented label="Stride" value={String(stride)} onChange={(v) => setStride(Number(v))} options={[{ value: "1", label: "1" }, { value: "2", label: "2" }]} />
          <Toggle label="Apply ReLU to feature map" checked={relu} onChange={setRelu} />
          {view === "pooling" && <Segmented label="Pooling" value={pool} onChange={setPool} options={[{ value: "max", label: "Max" }, { value: "avg", label: "Average" }]} />}
          {view === "conv" && <AnimationControls playing={playing} onPlay={() => { if (!playing && pos >= total - 1) setPos(0); setPlaying(!playing); }} onStep={() => setPos((p) => Math.min(total - 1, p + 1))} canStep={pos < total - 1} onReset={() => { setPlaying(false); setPos(0); }} />}
          <button className="btn w-full" onClick={() => setX(INPUT)}>Reset input image</button>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{output size}", tex: true, value: `${out} × ${out}` },
        { label: "\\text{position}", tex: true, value: `(${oi + 1}, ${oj + 1})` },
        { label: "y_{ij}", tex: true, value: fmt(Y[oi]?.[oj] ?? 0, 3) },
        { label: "\\text{weights (incl. bias)}", tex: true, value: k * k + 1 },
      ]} />}
      calc={<CalcSteps steps={view === "conv" ? [
        { tex: `y_{${oi + 1},${oj + 1}}=\\sum_{a,b}x_{${oi * stride}+a,\\,${oj * stride}+b}\\,k_{ab}=${terms.length ? terms.map((t) => `(${tex(t.x, 0)})(${tex(t.k, 2)})`).join("+") : "0"}=${tex(Y[oi]?.[oj] ?? 0, 3)}`, note: "zero-weight terms omitted" },
        { tex: `\\text{out}=\\left\\lfloor\\frac{n+2p-k}{s}\\right\\rfloor+1=\\left\\lfloor\\frac{${X.length}+${2 * pad}-${k}}{${stride}}\\right\\rfloor+1=${out}` },
      ] : [{ tex: pool === "max" ? "y_{ij}=\\max\\{x_{2i,2j},x_{2i,2j+1},x_{2i+1,2j},x_{2i+1,2j+1}\\}" : "y_{ij}=\\tfrac14\\sum x_{\\text{window}}", note: "no learnable parameters; halves spatial size, adds small translation invariance" }]} />}
      note="Deep-learning 'convolution' is technically cross-correlation (the kernel is not flipped). The same 3×3 weights are reused at every position — this weight sharing is why CNNs need far fewer parameters than dense layers."
    />
  );
}
