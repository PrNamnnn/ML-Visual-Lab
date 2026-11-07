"use client";
import { useMemo, useState } from "react";
import { Plot, FnPath, Dot, C, Polyline } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, AnimationControls, Segmented } from "@/components/controls";
import { gradientBoost, stumpPredict, fitStump } from "@/lib/ml/models";
import { mulberry32, randn, randInt } from "@/lib/math/random";
import { useTicker } from "@/hooks/useSize";
import { fmt, tex } from "@/lib/utilities/format";

/** Gradient boosting with depth-1 regression stumps on a 1-D problem, plus a bagging comparison. */
export default function Boosting({ mode: initial = "boosting" }: { mode?: "boosting" | "bagging" }) {
  const [mode, setMode] = useState(initial);
  const [M, setM] = useState(1);
  const [lr, setLr] = useState(0.3);
  const [playing, setPlaying] = useState(false);
  const data = useMemo(() => {
    const r = mulberry32(17);
    return Array.from({ length: 60 }, () => { const x = r() * 6; return [x, Math.sin(x) * 1.5 + 0.3 * x + randn(r) * 0.25] as [number, number]; });
  }, []);
  const xs = data.map((d) => d[0]), ys = data.map((d) => d[1]);
  const gb = useMemo(() => gradientBoost(xs, ys, 100, lr), [lr]); // eslint-disable-line
  const bag = useMemo(() => {
    const r = mulberry32(3);
    return Array.from({ length: 100 }, () => { const idx = Array.from({ length: xs.length }, () => randInt(r, 0, xs.length - 1)); return fitStump(idx.map((i) => xs[i]), idx.map((i) => ys[i])); });
  }, []); // eslint-disable-line
  useTicker(playing, 220, () => setM((m) => { if (m >= 100) { setPlaying(false); return 100; } return m + 1; }));
  const st = gb.stages[M];
  const mse = (pred: number[]) => pred.reduce((s, p, i) => s + (ys[i] - p) ** 2, 0) / ys.length;
  const bagPred = (x: number) => bag.slice(0, M).reduce((s, b) => s + stumpPredict(b, x), 0) / M;
  const lastStump = gb.stumps[M - 1];
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[0, 6]} y={[-2, 3.5]} aspect={0.5} maxH={360} ariaLabel={`${mode} ensemble prediction`}>
            {data.map((p, i) => <Dot key={i} p={p} r={3} color={C.a("fg", 0.7)} />)}
            {mode === "boosting" ? (
              <>
                <FnPath f={(x) => gb.predict(x, M - 1)} style={{ stroke: C.a("blue", 0.35), strokeWidth: 1.2 }} />
                <FnPath f={(x) => gb.predict(x, M)} n={600} style={{ stroke: C.blue, strokeWidth: 2.2 }} />
              </>
            ) : (
              <>
                {bag.slice(0, Math.min(M, 15)).map((b, i) => <FnPath key={i} f={(x) => stumpPredict(b, x)} n={300} style={{ stroke: C.a("pink", 0.2), strokeWidth: 1 }} />)}
                <FnPath f={bagPred} n={600} style={{ stroke: C.pink, strokeWidth: 2.2 }} />
              </>
            )}
          </Plot>
          {mode === "boosting" && (
            <div>
              <div className="mb-1 text-[11.5px] text-faint">Residuals rₘ = y − Fₘ₋₁(x) and the stump fitted to them at stage {M}</div>
              <Plot x={[0, 6]} y={[-2, 2]} aspect={0.28} minH={120} maxH={180} ariaLabel="residuals and fitted stump">
                {data.map((p, i) => <Dot key={i} p={[p[0], gb.stages[M - 1].resid[i]]} r={2.6} color={C.a("amber", 0.85)} />)}
                {lastStump && <Polyline pts={[[0, lastStump.left], [Math.min(6, lastStump.t), lastStump.left], [Math.min(6, lastStump.t), lastStump.right], [6, lastStump.right]]} style={{ stroke: C.green, strokeWidth: 2 }} />}
              </Plot>
            </div>
          )}
          <Legend items={mode === "boosting" ? [{ label: `F_${M}(x)`, color: C.blue }, { label: `F_${M - 1}(x)`, color: C.a("blue", 0.4) }, { label: "residuals", color: C.amber }, { label: "new stump hₘ", color: C.green }] : [{ label: `average of ${M} bootstrapped stumps`, color: C.pink }, { label: "individual stumps (first 15)", color: C.a("pink", 0.4) }]} />
        </div>
      }
      controls={
        <>
          <Segmented value={mode} onChange={setMode} options={[{ value: "boosting", label: "Boosting (sequential)" }, { value: "bagging", label: "Bagging (parallel)" }]} />
          <ParameterSlider label={mode === "boosting" ? "stages M" : "estimators"} value={M} min={1} max={100} step={1} onChange={setM} />
          {mode === "boosting" && <ParameterSlider label="learning rate ν (shrinkage)" value={lr} min={0.05} max={1} step={0.05} onChange={setLr} />}
          <AnimationControls playing={playing} onPlay={() => { if (!playing && M >= 100) setM(1); setPlaying(!playing); }} onStep={() => setM((m) => Math.min(100, m + 1))} onReset={() => { setPlaying(false); setM(1); }} />
        </>
      }
      readouts={<Readouts items={mode === "boosting" ? [
        { label: "\\text{train MSE}", tex: true, value: fmt(mse(st.pred), 4) },
        { label: "F_0=\\bar y", tex: true, value: fmt(gb.f0) },
        { label: "h_M\\ \\text{split}", tex: true, value: lastStump ? `x ≤ ${fmt(lastStump.t)}` : "—" },
      ] : [{ label: "\\text{train MSE}", tex: true, value: fmt(mse(xs.map(bagPred)), 4) }]} />}
      calc={<CalcSteps steps={mode === "boosting" ? [
        { tex: "r_i^{(m)}=y_i-F_{m-1}(x_i)=-\\dfrac{\\partial}{\\partial F}\\tfrac12(y_i-F)^2\\Big|_{F_{m-1}}", note: "residual = negative gradient of squared loss" },
        { tex: `F_m(x)=F_{m-1}(x)+\\nu\\,h_m(x),\\quad\\nu=${tex(lr)}` },
        ...(lastStump ? [{ tex: `h_{${M}}(x)=\\begin{cases}${tex(lastStump.left)}&x\\le${tex(lastStump.t)}\\\\${tex(lastStump.right)}&\\text{otherwise}\\end{cases}` }] : []),
      ] : [{ tex: "\\hat f(x)=\\frac1B\\sum_{b=1}^Bh_b(x),\\quad h_b\\text{ fit on bootstrap sample }b", note: "averaging reduces variance but not bias: stumps stay too simple" }]} />}
      note={mode === "boosting" ? "Boosting reduces bias by fitting each new weak learner to what the ensemble still gets wrong. Smaller ν needs more stages but generalises better. XGBoost, LightGBM and CatBoost add second-order gradients, regularisation and fast tree building to this same idea." : "Bagging averages many high-variance learners trained independently. With weak stumps it barely helps — it shines with deep trees (random forests)."}
    />
  );
}
