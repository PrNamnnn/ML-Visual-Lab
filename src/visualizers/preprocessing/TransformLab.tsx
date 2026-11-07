"use client";
import { useEffect, useMemo, useState } from "react";
import { Plot, C, FnPath } from "@/components/visualization/Plot";
import { Bars } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Warn } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle } from "@/components/controls";
import { transforms, transformById, fitLambda, boxCoxLogLik, yeoJohnsonLogLik } from "@/lib/transformations";
import { histogram, skewness, kurtosis, mean, std, extent } from "@/lib/statistics/descriptive";
import { sample1D, sampleInfo, type SampleId } from "@/lib/statistics/datasets";
import { Tex } from "@/components/mathematics/Tex";
import { fmt, tex } from "@/lib/utilities/format";

function Hist({ data, label, color, overlayNormal }: { data: number[]; label: string; color: "blue" | "pink"; overlayNormal?: boolean }) {
  const finite = data.filter(Number.isFinite);
  const [lo0, hi0] = extent(finite);
  const pad = (hi0 - lo0) * 0.05 || 1;
  const lo = lo0 - pad, hi = hi0 + pad;
  const bins = histogram(finite, lo, hi, 36);
  const m = mean(finite), s = std(finite);
  const ymax = Math.max(...bins.map((b) => b.density), overlayNormal && s > 0 ? 1 / (s * Math.sqrt(2 * Math.PI)) : 0) * 1.12 || 1;
  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-baseline justify-between text-[12px]">
        <span className="text-muted">{label}</span>
        <span className="num text-faint">skew {fmt(skewness(finite), 2)}</span>
      </div>
      <Plot x={[lo, hi]} y={[0, ymax]} aspect={0.62} minH={180} maxH={280} margin={{ l: 36 }} ariaLabel={`${label} histogram`}>
        <Bars bins={bins} color={C.a(color, 0.35)} stroke={C.a(color, 0.75)} />
        {overlayNormal && s > 0 && <FnPath f={(x) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))} style={{ stroke: C.a("fg", 0.55), strokeWidth: 1.2, strokeDasharray: "4 3" }} />}
      </Plot>
    </div>
  );
}

export default function TransformLab({ id: initial = "log", lab = false }: { id?: string; lab?: boolean }) {
  const [id, setId] = useState(initial);
  const T = transformById(id) ?? transforms[0];
  const [sid, setSid] = useState<SampleId>(T.id === "cbrt" || T.id === "yeo-johnson" ? "negatives" : T.id === "log1p" || T.id === "sqrt" ? "counts" : T.id === "exp" ? "left-skewed" : "lognormal");
  const [param, setParam] = useState(T.param?.default ?? 0);
  const [auto, setAuto] = useState(!!T.fitLambda);
  const [normal, setNormal] = useState(true);
  useEffect(() => { setParam(T.param?.default ?? 0); setAuto(!!T.fitLambda); }, [id]); // eslint-disable-line
  const raw = useMemo(() => {
    const x = sample1D(sid, 500, 7);
    return T.id === "exp" ? x.map((v) => v / 15) : x; // keep exp numerically sane
  }, [sid, T.id]);
  const lambdaHat = useMemo(() => {
    if (!T.fitLambda) return null;
    if (T.id === "box-cox") { const pos = raw.filter((v) => v > 0); return pos.length > 2 ? fitLambda((l) => boxCoxLogLik(pos, l)) : null; }
    return fitLambda((l) => yeoJohnsonLogLik(raw, l));
  }, [raw, T]);
  const p = auto && lambdaHat !== null ? lambdaHat : param;
  const res = useMemo(() => T.apply(raw, p), [T, raw, p]);
  const valid = res.values.filter(Number.isFinite);
  const kept = raw.filter((_, i) => Number.isFinite(res.values[i]));
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <div className="grid gap-4 md:grid-cols-2">
            <Hist data={raw} label="Original" color="blue" overlayNormal={normal} />
            {valid.length > 2 ? <Hist data={res.values} label={`After ${T.name.toLowerCase()}`} color="pink" overlayNormal={normal} /> : <Warn>No valid values after this transformation.</Warn>}
          </div>
          {res.invalid.length > 0 && <Warn>{res.invalid.length} of {raw.length} values are invalid for this transform (e.g. ≤ 0 for a log) and were excluded. {T.zeroNegative}</Warn>}
          <div className="text-[13px]"><Tex tex={T.formula} block /></div>
        </div>
      }
      controls={
        <>
          {lab && <Select label="Transformation" value={id} onChange={setId} options={transforms.map((t) => ({ value: t.id, label: t.name }))} />}
          <Select label="Dataset" value={sid} onChange={setSid} options={Object.entries(sampleInfo).map(([k, v]) => ({ value: k as SampleId, label: v }))} />
          {T.fitLambda && <Toggle label="Estimate λ by maximum likelihood" checked={auto} onChange={setAuto} />}
          {T.param && <ParameterSlider label={T.param.label} tex value={p} min={T.param.min} max={T.param.max} step={T.param.step} onChange={(v) => { setAuto(false); setParam(v); }} disabled={auto && !!T.fitLambda} />}
          <Toggle label="Overlay fitted normal curve" checked={normal} onChange={setNormal} />
          <div className="space-y-2 text-[12.5px] leading-relaxed">
            <div><span className="label">Intuition</span><p className="mt-0.5 text-muted">{T.intuition}</p></div>
            <div><span className="label">Zeros & negatives</span><p className="mt-0.5 text-muted">{T.zeroNegative}</p></div>
          </div>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{skewness before}", tex: true, value: fmt(skewness(kept)) },
        { label: "\\text{skewness after}", tex: true, value: fmt(skewness(valid)), color: Math.abs(skewness(valid)) < Math.abs(skewness(kept)) ? C.green : C.amber },
        { label: "\\text{excess kurtosis after}", tex: true, value: fmt(kurtosis(valid)) },
        { label: "\\text{valid / total}", tex: true, value: `${valid.length} / ${raw.length}` },
        ...(T.fitLambda ? [{ label: "\\hat\\lambda_{\\text{MLE}}", tex: true, value: lambdaHat === null ? "—" : fmt(lambdaHat) }] : []),
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `x_1=${tex(raw[0], 3)}\\ \\mapsto\\ y_1=${Number.isFinite(res.values[0]) ? tex(res.values[0], 4) : "\\text{invalid}"}`, note: "first observation" },
        { tex: `\\text{skew}=\\tfrac{\\sqrt{n(n-1)}}{n-2}\\cdot\\tfrac{m_3}{m_2^{3/2}}:\\ ${tex(skewness(kept))}\\to${tex(skewness(valid))}` },
        ...(T.fitLambda && lambdaHat !== null ? [{ tex: `\\hat\\lambda=\\arg\\max_\\lambda\\ \\ell(\\lambda)=${tex(lambdaHat)}`, note: "profile log-likelihood under normality, golden-section search" }] : []),
      ]} />}
      note={`Use cases: ${T.useCases} Limitations: ${T.limitations}`}
    />
  );
}
