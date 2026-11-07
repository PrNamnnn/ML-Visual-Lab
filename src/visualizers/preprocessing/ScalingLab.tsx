"use client";
import { useMemo, useState } from "react";
import { Plot, C, Dot } from "@/components/visualization/Plot";
import { Bars } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { Select, Toggle } from "@/components/controls";
import { scalers } from "@/lib/transformations";
import { histogram, mean, std, median, quantile, extent } from "@/lib/statistics/descriptive";
import { sample1D } from "@/lib/statistics/datasets";
import { Tex } from "@/components/mathematics/Tex";
import { fmt } from "@/lib/utilities/format";

function Strip({ data, label, color }: { data: number[]; label: string; color: "blue" | "pink" }) {
  const [lo0, hi0] = extent(data);
  const pad = (hi0 - lo0) * 0.06 || 1;
  const lo = lo0 - pad, hi = hi0 + pad;
  const bins = histogram(data, lo, hi, 32);
  const ymax = Math.max(...bins.map((b) => b.density)) * 1.15 || 1;
  return (
    <div className="min-w-0">
      <div className="mb-1 text-[12px] text-muted">{label} <span className="num text-faint">range [{fmt(lo0, 2)}, {fmt(hi0, 2)}]</span></div>
      <Plot x={[lo, hi]} y={[-ymax * 0.18, ymax]} aspect={0.5} minH={170} maxH={240} margin={{ l: 36 }} ariaLabel={`${label} distribution`}>
        <Bars bins={bins} color={C.a(color, 0.35)} stroke={C.a(color, 0.75)} />
        {data.filter((_, i) => i % 3 === 0).map((v, i) => <Dot key={i} p={[v, -ymax * 0.09]} r={1.6} color={C.a(color, 0.6)} />)}
      </Plot>
    </div>
  );
}

export default function ScalingLab({ method: initial = "standard" }: { method?: string }) {
  const [m, setM] = useState(initial);
  const [outliers, setOutliers] = useState(true);
  const S = scalers.find((s) => s.id === m) ?? scalers[0];
  const raw = useMemo(() => sample1D(outliers ? "outliers" : "bimodal", 300, 5), [outliers]);
  const r = useMemo(() => S.apply(raw), [S, raw]);
  const v = r.values;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div className="grid gap-4 md:grid-cols-2">
            <Strip data={raw} label="Original feature" color="blue" />
            <Strip data={v} label={`After ${S.name.toLowerCase()}`} color="pink" />
          </div>
          <div className="text-[13px]"><Tex tex={S.formula} block /></div>
        </div>
      }
      controls={
        <>
          <Select label="Scaler" value={m} onChange={setM} options={scalers.map((s) => ({ value: s.id, label: s.name }))} />
          <Toggle label="Include outliers" checked={outliers} onChange={setOutliers} />
          <p className="text-[12.5px] leading-relaxed text-muted">{S.description}</p>
          {r.note && <p className="text-[12px] text-amber">{r.note}</p>}
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{mean}", tex: true, value: `${fmt(mean(raw))} → ${fmt(mean(v))}` },
        { label: "\\text{std}", tex: true, value: `${fmt(std(raw))} → ${fmt(std(v))}` },
        { label: "\\text{median}", tex: true, value: `${fmt(median(raw))} → ${fmt(median(v))}` },
        { label: "\\text{IQR}", tex: true, value: `${fmt(quantile(raw, 0.75) - quantile(raw, 0.25))} → ${fmt(quantile(v, 0.75) - quantile(v, 0.25))}` },
      ]} />}
      calc={<CalcSteps steps={[{ tex: `x_1=${fmt(raw[0], 3)}\\ \\mapsto\\ x_1'=${fmt(v[0], 4)}`.replace(/−/g, "-") }]} />}
      note="Linear scalers change location and spread but never the shape (skewness is unchanged). Fit the scaler on training data only and reuse its statistics on validation/test data to avoid leakage."
    />
  );
}
