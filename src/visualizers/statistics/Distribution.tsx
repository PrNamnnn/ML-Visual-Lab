"use client";
import { useMemo, useState, useEffect } from "react";
import { Plot, FnPath, C, Polyline } from "@/components/visualization/Plot";
import { Bars, Stems, AreaUnder, VLine } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend, Warn } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented, AnimationControls, Toggle } from "@/components/controls";
import { distributions, distributionById, defaultParams } from "@/lib/statistics/distributions";
import { histogram, mean as smean, variance as svar, skewness, kurtosis } from "@/lib/statistics/descriptive";
import { mulberry32 } from "@/lib/math/random";
import { useTicker } from "@/hooks/useSize";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, tex } from "@/lib/utilities/format";

export default function Distribution({ id: initial = "normal", lab = false, showMoments = false }: { id?: string; lab?: boolean; showMoments?: boolean }) {
  const [id, setId] = useState(initial);
  const D = distributionById(id) ?? distributions[0];
  const [p, setP] = useState<Record<string, number>>(defaultParams(D));
  const [view, setView] = useState<"pdf" | "cdf">("pdf");
  const [samples, setSamples] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);
  const [seed, setSeed] = useState(1);
  const [qx, setQx] = useState<number | null>(null);
  const [compare, setCompare] = useState(false);
  const rng = useMemo(() => mulberry32(seed), [seed]);

  useEffect(() => { setP(defaultParams(D)); setSamples([]); setQx(null); }, [id]); // eslint-disable-line
  useEffect(() => setSamples([]), [p]);
  useTicker(playing, 60, () => setSamples((s) => { if (s.length >= 5000) { setPlaying(false); return s; } const add = Array.from({ length: Math.max(1, Math.round(s.length / 20)) }, () => D.sample(rng, p)); return [...s, ...add]; }));

  const err = D.validate?.(p) ?? null;
  const [lo, hi] = D.range(p);
  const disc = D.kind === "discrete";
  const xs = disc ? Array.from({ length: Math.round(hi - lo) + 1 }, (_, i) => lo + i) : [];
  const yMax = useMemo(() => {
    if (view === "cdf") return 1.05;
    if (disc) return Math.max(...xs.map((k) => D.density(k, p))) * 1.15 || 1;
    let m = 0;
    for (let i = 0; i <= 300; i++) { const v = D.density(lo + ((hi - lo) * i) / 300, p); if (Number.isFinite(v)) m = Math.max(m, v); }
    return Math.min(m * 1.15, 6) || 1;
  }, [D, p, view, lo, hi]); // eslint-disable-line
  const pad = disc ? 0.7 : 0;
  const bins = useMemo(() => {
    if (!samples.length) return [];
    if (disc) {
      const counts = new Map<number, number>();
      for (const s of samples) counts.set(s, (counts.get(s) ?? 0) + 1);
      return xs.map((k) => ({ x0: k - 0.4, x1: k + 0.4, count: counts.get(k) ?? 0, density: (counts.get(k) ?? 0) / samples.length }));
    }
    return histogram(samples, lo, hi, 40);
  }, [samples, disc, lo, hi]); // eslint-disable-line

  const mu = D.mean(p), v = D.variance(p);
  const q = qx ?? (Number.isFinite(mu) ? mu : (lo + hi) / 2);
  const Fq = D.cdf(q, p);
  const other = distributionById(D.id === "normal" ? "student-t" : "normal")!;
  const otherP: Record<string, number> = D.id === "normal" ? { nu: 3 } : { mu: Number.isFinite(mu) ? mu : 0, sigma: Number.isFinite(v) && v > 0 && v !== Infinity ? Math.sqrt(v) : 1 };

  return (
    <LabShell
      viz={
        <div className="space-y-2">
          {err ? <Warn>{err}</Warn> : (
            <Plot x={[lo - pad, hi + pad]} y={[0, yMax]} aspect={0.55} maxH={420} ariaLabel={`${D.name} ${view === "pdf" ? (disc ? "PMF" : "PDF") : "CDF"}`}
              onPointerDown={(pt) => setQx(disc ? Math.round(pt[0]) : pt[0])}>
              {view === "pdf" && samples.length > 0 && <Bars bins={bins} color={C.a("pink", 0.25)} stroke={C.a("pink", 0.6)} />}
              {view === "pdf" ? (
                disc ? (
                  <>
                    <Stems pts={xs.filter((k) => k <= q).map((k) => [k, D.density(k, p)])} color={C.a("blue", 0.9)} width={samples.length ? 0.25 : 0.6} />
                    <Stems pts={xs.filter((k) => k > q).map((k) => [k, D.density(k, p)])} color={C.a("blue", 0.35)} width={samples.length ? 0.25 : 0.6} />
                  </>
                ) : (
                  <>
                    <AreaUnder f={(x) => D.density(x, p)} a={lo} b={Math.min(q, hi)} />
                    {compare && <FnPath f={(x) => other.density(x, otherP)} style={{ stroke: C.amber, strokeWidth: 1.4, strokeDasharray: "5 3" }} />}
                    <FnPath f={(x) => D.density(x, p)} n={400} />
                  </>
                )
              ) : disc ? (
                <Polyline pts={xs.flatMap((k) => [[k, D.cdf(k, p)], [k + 1, D.cdf(k, p)]] as [number, number][])} style={{ stroke: C.blue, strokeWidth: 2 }} />
              ) : (
                <FnPath f={(x) => D.cdf(x, p)} />
              )}
              {view === "cdf" && samples.length > 0 && (
                <Polyline pts={samples.slice().sort((a, b) => a - b).map((s, i, arr) => [s, (i + 1) / arr.length] as [number, number])} style={{ stroke: C.pink, strokeWidth: 1.2 }} />
              )}
              {Number.isFinite(mu) && <VLine x={mu} color={C.green} label="μ" />}
              <VLine x={q} color={C.amber} label={`x = ${fmt(q, 2)}`} />
            </Plot>
          )}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
            <span className="text-[12.5px]"><Tex tex={D.formula} /></span>
            <Legend items={[{ label: view === "pdf" ? (disc ? "PMF" : "PDF") : "CDF", color: C.blue }, ...(samples.length ? [{ label: `${samples.length} samples ${view === "cdf" ? "(empirical CDF)" : "(histogram)"}`, color: C.pink }] : []), { label: "mean", color: C.green, dashed: true }, ...(compare && view === "pdf" && !disc ? [{ label: D.id === "normal" ? "Student t (ν=3) for comparison" : "normal with same mean & variance", color: C.amber, dashed: true }] : [])]} />
          </div>
        </div>
      }
      controls={
        <>
          {lab && (
            <Select label="Distribution" value={id} onChange={setId} options={[
              ...distributions.filter((d) => d.kind === "discrete").map((d) => ({ value: d.id, label: `${d.name} (discrete)` })),
              ...distributions.filter((d) => d.kind === "continuous").map((d) => ({ value: d.id, label: d.name })),
            ]} />
          )}
          <Segmented value={view} onChange={setView} options={[{ value: "pdf", label: disc ? "PMF" : "PDF" }, { value: "cdf", label: "CDF" }]} />
          {D.params.map((s) => (
            <ParameterSlider key={s.key} label={s.label} tex value={p[s.key] ?? s.default} min={s.min} max={s.max} step={s.step} onChange={(val) => setP({ ...p, [s.key]: val })} />
          ))}
          {!disc && <Toggle label="Compare with another distribution" checked={compare} onChange={setCompare} />}
          <div className="space-y-1.5">
            <div className="label">Sampling</div>
            <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onStep={() => setSamples((s) => [...s, ...Array.from({ length: 50 }, () => D.sample(rng, p))])} onReset={() => { setPlaying(false); setSamples([]); setSeed((x) => x + 1); }} />
          </div>
          <p className="text-[12px] text-faint">Click the plot to set the query point x.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "E[X]", tex: true, value: fmt(mu), color: C.green },
        { label: "\\operatorname{Var}(X)", tex: true, value: fmt(v) },
        { label: "\\sigma", tex: true, value: fmt(Math.sqrt(v)) },
        { label: disc ? `P(X=${fmt(q, 0)})` : `f(${fmt(q, 2)})`, tex: true, value: fmt(D.density(q, p), 5) },
        { label: `P(X\\le ${fmt(q, 2)})`, tex: true, value: fmt(Fq, 5), color: C.amber },
        ...(samples.length > 1 ? [{ label: "\\bar x\\ \\text{(sample)}", tex: true, value: fmt(smean(samples)) }, { label: "s^2\\ \\text{(sample)}", tex: true, value: fmt(svar(samples)) }] : []),
        ...((showMoments || lab) && samples.length > 3 ? [{ label: "\\text{skewness}", tex: true, value: fmt(skewness(samples)) }, { label: "\\text{excess kurtosis}", tex: true, value: fmt(kurtosis(samples)) }] : []),
      ]} />}
      calc={<CalcSteps steps={[
        { tex: D.formula + ",\\quad " + D.support },
        { tex: `F(${tex(q, 2)})=P(X\\le ${tex(q, 2)})=${tex(Fq, 5)}` },
      ]} />}
      note={`${D.notes} ML use: ${D.mlUse}`}
    />
  );
}
