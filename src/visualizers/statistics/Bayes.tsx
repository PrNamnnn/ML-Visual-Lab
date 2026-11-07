"use client";
import { useState } from "react";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Presets, Segmented } from "@/components/controls";
import { C } from "@/components/visualization/Plot";
import { useSize } from "@/hooks/useSize";
import { fmt, pct, tex } from "@/lib/utilities/format";

/** Bayes' theorem as a diagnostic test: a population grid and a probability tree, both exact. */
export default function Bayes({ mode = "test" }: { mode?: "test" | "venn" }) {
  if (mode === "venn") return <Venn />;
  return <Diagnostic />;
}

function Diagnostic() {
  const [prior, setPrior] = useState(0.02);
  const [sens, setSens] = useState(0.95);
  const [spec, setSpec] = useState(0.9);
  const [view, setView] = useState<"grid" | "tree">("grid");
  const N = 1000;
  const tp = prior * sens, fn = prior * (1 - sens), fp = (1 - prior) * (1 - spec), tn = (1 - prior) * spec;
  const pPos = tp + fp;
  const post = pPos > 0 ? tp / pPos : NaN;
  const npv = fn + tn > 0 ? tn / (fn + tn) : NaN;
  // integer counts for the grid (rounded consistently)
  const nSick = Math.round(prior * N);
  const nTP = Math.round(nSick * sens);
  const nFP = Math.round((N - nSick) * (1 - spec));
  const cells = Array.from({ length: N }, (_, i) => (i < nTP ? "tp" : i < nSick ? "fn" : i < nSick + nFP ? "fp" : "tn"));
  const col: Record<string, string> = { tp: "rgb(var(--pink))", fn: "rgb(var(--pink) / 0.3)", fp: "rgb(var(--blue))", tn: "rgb(var(--line-strong) / 0.6)" };
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Segmented value={view} onChange={setView} options={[{ value: "grid", label: "1,000 people" }, { value: "tree", label: "Probability tree" }]} />
          {view === "grid" ? (
            <div className="grid gap-[2px]" style={{ gridTemplateColumns: "repeat(50, minmax(0, 1fr))" }} role="img" aria-label={`${nTP} true positives, ${nFP} false positives out of 1000`}>
              {cells.map((c, i) => <div key={i} className="aspect-square rounded-[1px]" style={{ background: col[c] }} />)}
            </div>
          ) : (
            <Tree prior={prior} sens={sens} spec={spec} />
          )}
          <Legend items={[{ label: `sick & positive (TP) ≈ ${nTP}`, color: col.tp }, { label: `sick & negative (FN) ≈ ${nSick - nTP}`, color: col.fn }, { label: `healthy & positive (FP) ≈ ${nFP}`, color: col.fp }, { label: "healthy & negative (TN)", color: col.tn }]} />
          <p className="text-[13px] text-muted">Of the <span className="text-fg">{nTP + nFP}</span> people who test positive, only <span className="text-pink">{nTP}</span> are actually sick → P(sick | +) ≈ <span className="text-fg">{pct(nTP / Math.max(1, nTP + nFP))}</span> (grid counts are rounded; the exact value is {pct(post, 2)}).</p>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="prior P(sick)" value={prior} min={0.001} max={0.5} step={0.001} onChange={setPrior} format={(v) => pct(v, 1)} />
          <ParameterSlider label="sensitivity P(+ | sick)" value={sens} min={0.5} max={1} step={0.005} onChange={setSens} format={(v) => pct(v, 1)} />
          <ParameterSlider label="specificity P(− | healthy)" value={spec} min={0.5} max={1} step={0.005} onChange={setSpec} format={(v) => pct(v, 1)} />
          <Presets presets={[{ label: "Rare disease", value: [0.01, 0.99, 0.95] }, { label: "Common condition", value: [0.3, 0.9, 0.85] }, { label: "Spam filter", value: [0.4, 0.97, 0.99] }]} onPick={([a, b, c]) => { setPrior(a); setSens(b); setSpec(c); }} />
        </>
      }
      readouts={<Readouts items={[
        { label: "P(+)", tex: true, value: fmt(pPos, 4) },
        { label: "P(\\text{sick}\\mid +)", tex: true, value: pct(post, 2), color: C.pink },
        { label: "P(\\text{healthy}\\mid -)", tex: true, value: pct(npv, 2) },
        { label: "\\text{likelihood ratio}", tex: true, value: fmt(sens / (1 - spec || 1e-12)) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `P(+)=P(+\\mid S)P(S)+P(+\\mid \\bar S)P(\\bar S)=${tex(sens)}\\cdot${tex(prior, 4)}+${tex(1 - spec)}\\cdot${tex(1 - prior, 4)}=${tex(pPos, 5)}`, note: "law of total probability" },
        { tex: `P(S\\mid +)=\\dfrac{P(+\\mid S)P(S)}{P(+)}=\\dfrac{${tex(tp, 5)}}{${tex(pPos, 5)}}=${tex(post, 4)}`, note: "Bayes' theorem" },
      ]} />}
      note="Base-rate neglect: with a rare condition, even an accurate test produces mostly false positives. The same arithmetic governs precision for rare classes in ML."
    />
  );
}

function Tree({ prior, sens, spec }: { prior: number; sens: number; spec: number }) {
  const { ref, width } = useSize<HTMLDivElement>(0.5);
  const H = 240;
  const W = width;
  const n = (x: number, y: number, label: string, p: string, color: string) => (
    <g>
      <rect x={x - 62} y={y - 15} width={124} height={30} rx={5} style={{ fill: C.panel, stroke: color }} />
      <text x={x} y={y + 4} textAnchor="middle" style={{ fill: C.fg, fontSize: 11.5 }}>{label} <tspan style={{ fill: color }} className="num">{p}</tspan></text>
    </g>
  );
  const edge = (x1: number, y1: number, x2: number, y2: number, label: string) => (
    <g>
      <line x1={x1 + 62} y1={y1} x2={x2 - 62} y2={y2} style={{ stroke: C.lineStrong }} />
      <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 5} textAnchor="middle" className="num" style={{ fill: C.muted, fontSize: 10.5 }}>{label}</text>
    </g>
  );
  const x0 = 70, x1 = W * 0.42, x2 = W - 70;
  return (
    <div ref={ref}>
      <svg width={W} height={H} role="img" aria-label="probability tree">
        {edge(x0, H / 2, x1, 60, fmt(prior, 3))}
        {edge(x0, H / 2, x1, H - 60, fmt(1 - prior, 3))}
        {edge(x1, 60, x2, 25, fmt(sens, 3))}
        {edge(x1, 60, x2, 95, fmt(1 - sens, 3))}
        {edge(x1, H - 60, x2, H - 95, fmt(1 - spec, 3))}
        {edge(x1, H - 60, x2, H - 25, fmt(spec, 3))}
        {n(x0, H / 2, "all", "1", C.lineStrong)}
        {n(x1, 60, "sick", fmt(prior, 3), C.pink)}
        {n(x1, H - 60, "healthy", fmt(1 - prior, 3), C.blue)}
        {n(x2, 25, "+ (TP)", fmt(prior * sens, 4), C.pink)}
        {n(x2, 95, "− (FN)", fmt(prior * (1 - sens), 4), C.a("pink", 0.5))}
        {n(x2, H - 95, "+ (FP)", fmt((1 - prior) * (1 - spec), 4), C.blue)}
        {n(x2, H - 25, "− (TN)", fmt((1 - prior) * spec, 4), C.lineStrong)}
      </svg>
    </div>
  );
}

function Venn() {
  const [pa, setPa] = useState(0.4);
  const [pb, setPb] = useState(0.3);
  const [pab, setPab] = useState(0.12);
  const { ref, width } = useSize<HTMLDivElement>(0.5);
  const lo = Math.max(0, pa + pb - 1), hi = Math.min(pa, pb);
  const ab = Math.min(hi, Math.max(lo, pab));
  // circle radii ∝ sqrt(area); find center distance by bisection so lens area matches P(A∩B)
  const S = Math.min(width, 520) * 0.62;
  const rA = Math.sqrt(pa / Math.PI) * S, rB = Math.sqrt(pb / Math.PI) * S;
  const lens = (d: number) => {
    if (d >= rA + rB) return 0;
    if (d <= Math.abs(rA - rB)) return Math.PI * Math.min(rA, rB) ** 2;
    const a1 = rA * rA * Math.acos((d * d + rA * rA - rB * rB) / (2 * d * rA));
    const a2 = rB * rB * Math.acos((d * d + rB * rB - rA * rA) / (2 * d * rB));
    const a3 = 0.5 * Math.sqrt((-d + rA + rB) * (d + rA - rB) * (d - rA + rB) * (d + rA + rB));
    return a1 + a2 - a3;
  };
  const target = ab * S * S;
  let l = Math.abs(rA - rB), h = rA + rB;
  for (let i = 0; i < 60; i++) { const m = (l + h) / 2; if (lens(m) > target) l = m; else h = m; }
  const d = (l + h) / 2;
  const H = 260, cx = width / 2;
  const indep = pa * pb;
  return (
    <LabShell
      viz={
        <div ref={ref} className="space-y-2">
          <svg width={width} height={H} role="img" aria-label="area-proportional Venn diagram">
            <rect x={cx - S / 2 - 10} y={H / 2 - S / 2 + 4} width={S + 20} height={S - 8} rx={6} style={{ fill: "none", stroke: C.line }} />
            <circle cx={cx - d / 2} cy={H / 2} r={rA} style={{ fill: C.a("blue", 0.3), stroke: C.blue }} />
            <circle cx={cx + d / 2} cy={H / 2} r={rB} style={{ fill: C.a("pink", 0.3), stroke: C.pink }} />
            <text x={cx - d / 2 - rA * 0.5} y={H / 2 + 4} textAnchor="middle" style={{ fill: C.fg, fontSize: 13 }}>A</text>
            <text x={cx + d / 2 + rB * 0.5} y={H / 2 + 4} textAnchor="middle" style={{ fill: C.fg, fontSize: 13 }}>B</text>
            <text x={cx - S / 2} y={H / 2 - S / 2 + 20} style={{ fill: C.faint, fontSize: 11 }}>Ω (area 1)</text>
          </svg>
          <p className="text-[12px] text-faint">Areas are proportional to probabilities (the rectangle is approximate).</p>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="P(A)" value={pa} min={0.05} max={0.8} step={0.01} onChange={setPa} />
          <ParameterSlider label="P(B)" value={pb} min={0.05} max={0.8} step={0.01} onChange={setPb} />
          <ParameterSlider label="P(A ∩ B)" value={ab} min={lo} max={hi} step={0.005} onChange={setPab} />
          <button className="btn w-full" onClick={() => setPab(indep)}>Make A and B independent</button>
        </>
      }
      readouts={<Readouts items={[
        { label: "P(A\\mid B)", tex: true, value: fmt(ab / pb), color: C.pink },
        { label: "P(B\\mid A)", tex: true, value: fmt(ab / pa), color: C.blue },
        { label: "P(A\\cup B)", tex: true, value: fmt(pa + pb - ab) },
        { label: "\\text{independent?}", tex: true, value: Math.abs(ab - indep) < 0.003 ? "yes" : "no", color: Math.abs(ab - indep) < 0.003 ? C.green : undefined },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `P(A\\mid B)=\\dfrac{P(A\\cap B)}{P(B)}=\\dfrac{${tex(ab)}}{${tex(pb)}}=${tex(ab / pb)}` },
        { tex: `P(A)P(B)=${tex(indep)}\\ ${Math.abs(ab - indep) < 0.003 ? "=" : "\\neq"}\\ P(A\\cap B)=${tex(ab)}`, note: "independence test" },
        { tex: `P(A\\cup B)=P(A)+P(B)-P(A\\cap B)=${tex(pa + pb - ab)}` },
      ]} />}
    />
  );
}
