"use client";
import { useMemo, useState } from "react";
import { Plot, C, Polyline, Dot, FnPath } from "@/components/visualization/Plot";
import { Bars, VLine, AreaUnder } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Select } from "@/components/controls";
import { confusion, classificationMetrics, rocCurve, prCurve, logLoss, brierScore, calibrationBins, regressionMetrics } from "@/lib/ml/metrics";
import { histogram } from "@/lib/statistics/descriptive";
import { mulberry32, randn, shuffle } from "@/lib/math/random";
import { fmt, pct, tex, cn } from "@/lib/utilities/format";

type Focus = "confusion" | "roc" | "pr" | "calibration" | "regression" | "cv";

export default function Evaluation({ focus = "confusion" }: { focus?: Focus }) {
  if (focus === "regression") return <RegressionMetrics />;
  if (focus === "cv") return <CrossValidation />;
  return <ThresholdLab focus={focus} />;
}

function ThresholdLab({ focus }: { focus: Focus }) {
  const [sep, setSep] = useState(1.6);
  const [prev, setPrev] = useState(0.3);
  const [thr, setThr] = useState(0.5);
  const [beta, setBeta] = useState(1);
  const [miscal, setMiscal] = useState(1);
  const [view, setView] = useState<"roc" | "pr" | "calibration">(focus === "pr" ? "pr" : focus === "calibration" ? "calibration" : "roc");
  const { y, s } = useMemo(() => {
    const r = mulberry32(9);
    const N = 400;
    const y: number[] = [], s: number[] = [];
    for (let i = 0; i < N; i++) {
      const lab = r() < prev ? 1 : 0;
      // logit score ~ N(±sep/2, 1); calibrated probability uses the true posterior, then optionally distorted
      const z = (lab ? sep / 2 : -sep / 2) + randn(r);
      const logit = z * sep + Math.log(prev / (1 - prev));
      y.push(lab);
      s.push(1 / (1 + Math.exp(-logit * miscal)));
    }
    return { y, s };
  }, [sep, prev, miscal]);
  const c = confusion(y, s, thr);
  const m = classificationMetrics(c, beta);
  const roc = useMemo(() => rocCurve(y, s), [y, s]);
  const pr = useMemo(() => prCurve(y, s), [y, s]);
  const cal = useMemo(() => calibrationBins(y, s, 10), [y, s]);
  const ll = logLoss(y, s), br = brierScore(y, s);
  const pos = s.filter((_, i) => y[i] === 1), neg = s.filter((_, i) => y[i] === 0);
  const hp = histogram(pos, 0, 1, 25), hn = histogram(neg, 0, 1, 25);
  const ymax = Math.max(...hp.map((b) => b.count), ...hn.map((b) => b.count)) * 1.1;
  const cell = (label: string, v: number, kind: string) => (
    <div className={cn("rounded-md border p-2.5 text-center", kind === "good" ? "border-green/40 bg-green/10" : "border-red/40 bg-red/10")}>
      <div className="text-[11px] text-muted">{label}</div>
      <div className="num text-[20px] font-semibold">{v}</div>
    </div>
  );
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <div>
            <div className="mb-1 text-[11.5px] text-faint">Score distributions by true class — everything right of the threshold is predicted positive</div>
            <Plot x={[0, 1]} y={[0, ymax]} aspect={0.3} minH={150} maxH={200} ariaLabel="score histograms per class">
              <Bars bins={hn} field="count" color={C.a("blue", 0.35)} stroke={C.a("blue", 0.7)} />
              <Bars bins={hp} field="count" color={C.a("pink", 0.35)} stroke={C.a("pink", 0.7)} />
              <VLine x={thr} color={C.fg} dashed={false} label={`t = ${fmt(thr, 2)}`} />
            </Plot>
          </div>
          <div className="grid gap-4 md:grid-cols-[minmax(0,240px)_minmax(0,1fr)]">
            <div className="space-y-1.5">
              <div className="grid grid-cols-[auto_1fr_1fr] items-center gap-1.5 text-[11px]">
                <span />
                <span className="text-center text-muted">pred +</span>
                <span className="text-center text-muted">pred −</span>
                <span className="text-muted [writing-mode:vertical-rl]">actual +</span>
                {cell("TP", c.tp, "good")}
                {cell("FN", c.fn, "bad")}
                <span className="text-muted [writing-mode:vertical-rl]">actual −</span>
                {cell("FP", c.fp, "bad")}
                {cell("TN", c.tn, "good")}
              </div>
            </div>
            <div>
              <Segmented value={view} onChange={setView} options={[{ value: "roc", label: `ROC (AUC ${fmt(roc.auc, 3)})` }, { value: "pr", label: `PR (AP ${fmt(pr.ap, 3)})` }, { value: "calibration", label: "Calibration" }]} />
              <div className="mt-2">
                {view === "roc" && (
                  <Plot x={[0, 1]} y={[0, 1]} equal aspect={0.8} maxH={260} xLabel="FPR" yLabel="TPR" ariaLabel="ROC curve" margin={{ l: 34, b: 26 }}>
                    <Polyline pts={[[0, 0], [1, 1]]} style={{ stroke: C.faint, strokeDasharray: "4 3" }} />
                    <Polyline pts={roc.points.map((p) => [p.fpr, p.tpr])} style={{ stroke: C.blue, strokeWidth: 2 }} />
                    <Dot p={[m.fpr, m.tpr]} r={5} color={C.pink} />
                  </Plot>
                )}
                {view === "pr" && (
                  <Plot x={[0, 1]} y={[0, 1.02]} equal aspect={0.8} maxH={260} xLabel="recall" yLabel="precision" ariaLabel="precision-recall curve" margin={{ l: 34, b: 26 }}>
                    <Polyline pts={[[0, prev], [1, prev]]} style={{ stroke: C.faint, strokeDasharray: "4 3" }} />
                    <Polyline pts={pr.points.map((p) => [p.recall, p.precision])} style={{ stroke: C.green, strokeWidth: 2 }} />
                    {Number.isFinite(m.precision) && <Dot p={[m.recall, m.precision]} r={5} color={C.pink} />}
                  </Plot>
                )}
                {view === "calibration" && (
                  <Plot x={[0, 1]} y={[0, 1]} equal aspect={0.8} maxH={260} xLabel="mean predicted" yLabel="fraction positive" ariaLabel="reliability diagram" margin={{ l: 34, b: 26 }}>
                    <Polyline pts={[[0, 0], [1, 1]]} style={{ stroke: C.faint, strokeDasharray: "4 3" }} />
                    <Polyline pts={cal.filter((b) => b.n > 0).map((b) => [b.meanPred, b.fracPos])} style={{ stroke: C.amber, strokeWidth: 2 }} />
                    {cal.filter((b) => b.n > 0).map((b, i) => <Dot key={i} p={[b.meanPred, b.fracPos]} r={2 + Math.sqrt(b.n) / 2} color={C.a("amber", 0.7)} />)}
                  </Plot>
                )}
              </div>
            </div>
          </div>
          <Legend items={[{ label: "negatives", color: C.blue }, { label: "positives", color: C.pink }, { label: "current threshold operating point", color: C.pink }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="decision threshold t" value={thr} min={0.01} max={0.99} step={0.01} onChange={setThr} />
          <ParameterSlider label="class separation" value={sep} min={0.2} max={4} step={0.05} onChange={setSep} />
          <ParameterSlider label="prevalence P(y=1)" value={prev} min={0.02} max={0.8} step={0.01} onChange={setPrev} />
          <ParameterSlider label="F-β: β" value={beta} min={0.25} max={4} step={0.05} onChange={setBeta} />
          <ParameterSlider label="calibration distortion (1 = calibrated)" value={miscal} min={0.3} max={3} step={0.05} onChange={setMiscal} />
          <p className="text-[12px] leading-relaxed text-faint">Raising t trades recall for precision. ROC is insensitive to prevalence; PR is not — try a rare positive class.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{accuracy}", tex: true, value: pct(m.accuracy) },
        { label: "\\text{precision}", tex: true, value: pct(m.precision) },
        { label: "\\text{recall (TPR)}", tex: true, value: pct(m.recall) },
        { label: "\\text{specificity}", tex: true, value: pct(m.specificity) },
        { label: "F_1", tex: true, value: fmt(m.f1, 3) },
        { label: `F_{${tex(beta, 2)}}`, tex: true, value: fmt(m.fbeta, 3) },
        { label: "\\text{log loss}", tex: true, value: fmt(ll, 4) },
        { label: "\\text{Brier}", tex: true, value: fmt(br, 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: `\\text{precision}=\\frac{TP}{TP+FP}=\\frac{${c.tp}}{${c.tp + c.fp}}=${tex(m.precision)},\\quad\\text{recall}=\\frac{TP}{TP+FN}=\\frac{${c.tp}}{${c.tp + c.fn}}=${tex(m.recall)}` },
        { tex: `F_\\beta=\\frac{(1+\\beta^2)\\,TP}{(1+\\beta^2)TP+\\beta^2FN+FP}=${tex(m.fbeta)}`, note: "β > 1 weights recall more" },
        { tex: `\\text{AUC}=P(s^+>s^-)=${tex(roc.auc, 4)},\\quad\\text{AP}=\\sum_k(R_k-R_{k-1})P_k=${tex(pr.ap, 4)}` },
      ]} />}
      note="Accuracy is misleading under class imbalance: predicting 'negative' for everyone scores 1 − prevalence. Choose the threshold from the costs of false positives versus false negatives."
    />
  );
}

function RegressionMetrics() {
  const [noise, setNoise] = useState(0.5);
  const [outlier, setOutlier] = useState(0);
  const [bias, setBias] = useState(0);
  const { y, yh, x } = useMemo(() => {
    const r = mulberry32(4);
    const x = Array.from({ length: 40 }, (_, i) => i / 4);
    const y = x.map((v) => 1 + 0.8 * v + randn(r) * noise);
    if (outlier) y[30] += outlier;
    const yh = x.map((v) => 1 + 0.8 * v + bias);
    return { y, yh, x };
  }, [noise, outlier, bias]);
  const m = regressionMetrics(y, yh, 1);
  return (
    <LabShell
      viz={
        <Plot x={[-0.5, 10.5]} y={[-2, 16]} aspect={0.55} maxH={400} ariaLabel="predictions and residuals">
          {x.map((v, i) => <Polyline key={`r${i}`} pts={[[v, y[i]], [v, yh[i]]]} style={{ stroke: C.a("pink", 0.6), strokeWidth: 1 }} />)}
          <FnPath f={(v) => 1 + 0.8 * v + bias} />
          {x.map((v, i) => <Dot key={i} p={[v, y[i]]} r={3.2} color={i === 30 && outlier ? C.red : C.fg} />)}
        </Plot>
      }
      controls={
        <>
          <ParameterSlider label="noise σ" value={noise} min={0} max={3} step={0.05} onChange={setNoise} />
          <ParameterSlider label="outlier size" value={outlier} min={0} max={15} step={0.5} onChange={setOutlier} />
          <ParameterSlider label="systematic bias of model" value={bias} min={-3} max={3} step={0.1} onChange={setBias} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{MAE}", tex: true, value: fmt(m.mae) }, { label: "\\text{MSE}", tex: true, value: fmt(m.mse) }, { label: "\\text{RMSE}", tex: true, value: fmt(m.rmse) }, { label: "R^2", tex: true, value: fmt(m.r2) }, { label: "\\bar R^2\\ (p=1)", tex: true, value: fmt(m.adjR2) }]} />}
      calc={<CalcSteps steps={[{ tex: `\\text{MAE}=\\tfrac1n\\sum|y_i-\\hat y_i|=${tex(m.mae)},\\quad\\text{RMSE}=\\sqrt{\\tfrac1n\\sum(y_i-\\hat y_i)^2}=${tex(m.rmse)}` }, { tex: `R^2=1-\\frac{SS_{res}}{SS_{tot}}=1-\\frac{${tex(m.sse, 2)}}{${tex(m.sst, 2)}}=${tex(m.r2)},\\quad\\bar R^2=1-(1-R^2)\\frac{n-1}{n-p-1}` }]} />}
      note="RMSE punishes the single outlier far more than MAE does. R² can be negative when the model is worse than predicting the mean."
    />
  );
}

function CrossValidation() {
  const [k, setK] = useState(5);
  const [kind, setKind] = useState<"kfold" | "stratified" | "timeseries">("kfold");
  const [fold, setFold] = useState(0);
  const n = 30;
  const labels = useMemo(() => Array.from({ length: n }, (_, i) => (i % 5 === 0 || i % 7 === 0 ? 1 : 0)), []);
  const order = useMemo(() => (kind === "timeseries" ? Array.from({ length: n }, (_, i) => i) : shuffle(Array.from({ length: n }, (_, i) => i), mulberry32(3))), [kind]);
  const assign = useMemo(() => {
    const f = Array(n).fill(-1);
    if (kind === "kfold") order.forEach((idx, pos) => (f[idx] = pos % k));
    else if (kind === "stratified") { let p0 = 0, p1 = 0; order.forEach((idx) => { if (labels[idx]) f[idx] = p1++ % k; else f[idx] = p0++ % k; }); }
    else { const size = Math.floor(n / (k + 1)); for (let i = 0; i < n; i++) f[i] = Math.min(k, Math.floor(i / size)) - 1; }
    return f;
  }, [kind, k, order, labels]);
  const fk = Math.min(fold, k - 1);
  const role = (i: number) => (kind === "timeseries" ? (assign[i] === fk ? "val" : assign[i] < fk ? "train" : "unused") : assign[i] === fk ? "val" : "train");
  const valPos = labels.filter((l, i) => role(i) === "val" && l === 1).length;
  const valN = labels.filter((_, i) => role(i) === "val").length;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          {Array.from({ length: k }, (_, f) => (
            <div key={f} className={cn("flex items-center gap-2", f === fk && "rounded bg-raised/60")}>
              <button className="w-14 shrink-0 text-left text-[11.5px] text-muted hover:text-fg" onClick={() => setFold(f)}>fold {f + 1}</button>
              <div className="grid flex-1 gap-[2px]" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
                {Array.from({ length: n }, (_, i) => {
                  const r = kind === "timeseries" ? (assign[i] === f ? "val" : assign[i] < f ? "train" : "unused") : assign[i] === f ? "val" : "train";
                  return <div key={i} className="relative h-6 rounded-[2px]" style={{ background: r === "val" ? "rgb(var(--pink) / 0.75)" : r === "train" ? "rgb(var(--blue) / 0.5)" : "rgb(var(--line))" }}>{labels[i] === 1 && <span className="absolute inset-x-0 bottom-0.5 mx-auto block h-1 w-1 rounded-full bg-white" />}</div>;
                })}
              </div>
            </div>
          ))}
          <Legend items={[{ label: "training", color: "rgb(var(--blue))" }, { label: "validation", color: "rgb(var(--pink))" }, ...(kind === "timeseries" ? [{ label: "future (unused)", color: "rgb(var(--line-strong))" }] : []), { label: "white dot = positive class", color: "#fff" }]} />
        </div>
      }
      controls={
        <>
          <Select label="Scheme" value={kind} onChange={setKind} options={[{ value: "kfold", label: "K-fold (shuffled)" }, { value: "stratified", label: "Stratified K-fold" }, { value: "timeseries", label: "Time-series (expanding window)" }]} />
          <ParameterSlider label="k folds" value={k} min={2} max={8} step={1} onChange={(v) => { setK(v); setFold(0); }} />
          <ParameterSlider label="highlight fold" value={fk + 1} min={1} max={k} step={1} onChange={(v) => setFold(v - 1)} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{validation size}", tex: true, value: valN }, { label: "\\text{positives in validation}", tex: true, value: `${valPos} (${pct(valN ? valPos / valN : NaN, 0)})` }, { label: "\\text{overall prevalence}", tex: true, value: pct(labels.filter(Boolean).length / n, 0) }]} />}
      calc={<CalcSteps steps={[{ tex: "\\widehat{\\text{err}}_{CV}=\\frac1k\\sum_{j=1}^{k}\\text{err}\\big(\\hat f^{(-j)},\\,D_j\\big)", note: "train on all folds but j, evaluate on fold j, average" }]} />}
      note={kind === "timeseries" ? "Never train on the future: each validation block comes strictly after its training data." : kind === "stratified" ? "Stratification keeps the class ratio in each fold close to the overall ratio — important for small or imbalanced datasets." : "Plain K-fold can produce folds with noticeably different class balance by chance."}
    />
  );
}

export { AreaUnder };
