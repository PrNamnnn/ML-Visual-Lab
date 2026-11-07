"use client";
import { useMemo, useState } from "react";
import { Plot, FnPath, Dot, C, Polyline } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Toggle } from "@/components/controls";
import { polyDesign, ridgeFit, elasticNetFit, predictLinear } from "@/lib/ml/models";
import { mulberry32, randn } from "@/lib/math/random";
import { fmt, tex } from "@/lib/utilities/format";

type Mode = "degree" | "ridge" | "lasso" | "elastic" | "bias-variance";
const truth = (x: number) => Math.sin(2.2 * x) + 0.3 * x;

function makeData(n: number, noise: number, seed: number) {
  const r = mulberry32(seed);
  return Array.from({ length: n }, () => { const x = r() * 2.8 - 1.4; return [x, truth(x) + noise * randn(r)] as [number, number]; });
}

export default function PolyRegression({ mode: initial = "degree", lab = false }: { mode?: Mode; lab?: boolean }) {
  const [mode, setMode] = useState<Mode>(initial);
  const [deg, setDeg] = useState(initial === "degree" ? 3 : 12);
  const [logLam, setLogLam] = useState(initial === "lasso" ? -2.5 : initial === "ridge" ? -2 : -2.5);
  const [alpha, setAlpha] = useState(0.5);
  const [n, setN] = useState(24);
  const [noise, setNoise] = useState(0.25);
  const [seed, setSeed] = useState(3);
  const [showTruth, setShowTruth] = useState(false);
  const lam = mode === "degree" || mode === "bias-variance" ? 0 : Math.pow(10, logLam);
  const train = useMemo(() => makeData(n, noise, seed), [n, noise, seed]);
  const test = useMemo(() => makeData(200, noise, 999), [noise]);

  // standardise polynomial features (except bias) so regularisation treats them comparably
  const fitModel = (data: [number, number][], d: number) => {
    const X0 = polyDesign(data.map((p) => p[0]), d);
    const mu = X0[0].map((_, j) => (j === 0 ? 0 : X0.reduce((s, r) => s + r[j], 0) / X0.length));
    const sd = X0[0].map((_, j) => (j === 0 ? 1 : Math.sqrt(X0.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / X0.length) || 1));
    const tf = (X: number[][]) => X.map((r) => r.map((v, j) => (j === 0 ? 1 : (v - mu[j]) / sd[j])));
    const X = tf(X0);
    const y = data.map((p) => p[1]);
    let w: number[] | null;
    if (mode === "lasso") w = elasticNetFit(X, y, lam, 1, 3000);
    else if (mode === "elastic") w = elasticNetFit(X, y, lam, alpha, 3000);
    else {
      // ridge objective here is ‖y − Xw‖²/(2n) + λ/2‖w‖² to match the coordinate-descent scaling → (XᵀX + nλI)w = Xᵀy
      w = ridgeFit(X, y, lam * data.length) ?? ridgeFit(X, y, 1e-8 * data.length);
    }
    const ww = w ?? X[0].map(() => 0);
    return { w: ww, predict: (x: number) => predictLinear(tf(polyDesign([x], d)), ww)[0] };
  };

  const model = useMemo(() => fitModel(train, deg), [train, deg, lam, alpha, mode]); // eslint-disable-line
  const mse = (data: [number, number][], f: (x: number) => number) => data.reduce((s, [x, y]) => s + (y - f(x)) ** 2, 0) / data.length;
  const trainErr = mse(train, model.predict), testErr = mse(test, model.predict);
  const curve = useMemo(() => Array.from({ length: 15 }, (_, d) => { const m = fitModel(train, d); return { d, tr: mse(train, m.predict), te: mse(test, m.predict) }; }), [train, test, lam, alpha, mode]); // eslint-disable-line
  const bvFits = useMemo(() => (mode === "bias-variance" ? Array.from({ length: 20 }, (_, k) => fitModel(makeData(n, noise, 100 + k), deg)) : []), [mode, n, noise, deg]); // eslint-disable-line
  const bv = useMemo(() => {
    if (!bvFits.length) return null;
    const xs = Array.from({ length: 50 }, (_, i) => -1.4 + (2.8 * i) / 49);
    let b2 = 0, v = 0;
    for (const x of xs) {
      const ps = bvFits.map((m) => m.predict(x));
      const mp = ps.reduce((a, c) => a + c, 0) / ps.length;
      b2 += (mp - truth(x)) ** 2;
      v += ps.reduce((a, c) => a + (c - mp) ** 2, 0) / ps.length;
    }
    return { bias2: b2 / xs.length, variance: v / xs.length };
  }, [bvFits]);
  const regularised = mode === "ridge" || mode === "lasso" || mode === "elastic";
  const nz = model.w.slice(1).filter((w) => Math.abs(w) > 1e-6).length;
  const maxW = Math.max(1, ...model.w.slice(1).map(Math.abs));

  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <Plot x={[-1.5, 1.5]} y={[-2.2, 2.2]} aspect={0.55} maxH={400} ariaLabel="polynomial regression fit">
            {showTruth && <FnPath f={truth} style={{ stroke: C.green, strokeWidth: 1.4, strokeDasharray: "5 4" }} />}
            {bvFits.map((m, i) => <FnPath key={i} f={m.predict} style={{ stroke: C.a("pink", 0.22), strokeWidth: 1 }} />)}
            {mode !== "bias-variance" && <FnPath f={model.predict} n={400} />}
            {mode !== "bias-variance" && train.map((p, i) => <Dot key={i} p={p} r={3.5} color={C.fg} />)}
          </Plot>
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className="mb-1 text-[11.5px] text-faint">Train vs test MSE by degree{regularised ? ` (λ = ${lam.toExponential(1)})` : ""}</div>
              <Plot x={[0, 14]} y={[0, Math.min(2, Math.max(...curve.map((c) => Math.min(2, c.te))) * 1.1)]} aspect={0.45} minH={140} maxH={190} margin={{ l: 34, b: 22 }} ariaLabel="error versus polynomial degree">
                <Polyline pts={curve.map((c) => [c.d, c.tr])} style={{ stroke: C.blue, strokeWidth: 1.6 }} />
                <Polyline pts={curve.map((c) => [c.d, Math.min(2, c.te)])} style={{ stroke: C.pink, strokeWidth: 1.6 }} />
                <Dot p={[deg, Math.min(2, testErr)]} r={4} color={C.pink} />
              </Plot>
            </div>
            <div>
              <div className="mb-1 text-[11.5px] text-faint">Coefficients (standardised features){regularised ? ` — ${nz} non-zero` : ""}</div>
              <div className="flex h-[150px] items-center gap-[3px] border-l border-line pl-1">
                {model.w.slice(1).map((w, i) => (
                  <div key={i} className="flex h-full flex-1 flex-col justify-center" title={`w${i + 1} = ${fmt(w)}`}>
                    <div className="flex flex-1 items-end"><div className="w-full rounded-t-sm" style={{ height: `${w > 0 ? (w / maxW) * 100 : 0}%`, background: "rgb(var(--blue))" }} /></div>
                    <div className="h-px bg-line-strong" />
                    <div className="flex flex-1 items-start"><div className="w-full rounded-b-sm" style={{ height: `${w < 0 ? (-w / maxW) * 100 : 0}%`, background: "rgb(var(--pink))" }} /></div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <Legend items={[{ label: "train MSE", color: C.blue }, { label: "test MSE", color: C.pink }, ...(showTruth ? [{ label: "true function", color: C.green, dashed: true }] : [])]} />
        </div>
      }
      controls={
        <>
          {lab && <Select label="Model" value={mode} onChange={setMode} options={[{ value: "degree", label: "Polynomial (OLS)" }, { value: "ridge", label: "Ridge (L2)" }, { value: "lasso", label: "Lasso (L1)" }, { value: "elastic", label: "Elastic Net" }, { value: "bias-variance", label: "Bias–variance (20 refits)" }]} />}
          <ParameterSlider label="polynomial degree" value={deg} min={0} max={14} step={1} onChange={setDeg} />
          {regularised && <ParameterSlider label="regularisation log₁₀ λ" value={logLam} min={-6} max={1} step={0.05} onChange={setLogLam} />}
          {mode === "elastic" && <ParameterSlider label="L1 ratio α" value={alpha} min={0} max={1} step={0.05} onChange={setAlpha} />}
          <ParameterSlider label="training points n" value={n} min={8} max={100} step={1} onChange={setN} />
          <ParameterSlider label="noise σ" value={noise} min={0} max={0.8} step={0.01} onChange={setNoise} />
          <Toggle label="Show true function" checked={showTruth} onChange={setShowTruth} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>Resample training set</button>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{train MSE}", tex: true, value: fmt(trainErr, 4), color: C.blue },
        { label: "\\text{test MSE}", tex: true, value: fmt(testErr, 4), color: C.pink },
        { label: "\\text{gap}", tex: true, value: fmt(testErr - trainErr, 4) },
        ...(bv ? [{ label: "\\text{bias}^2", tex: true, value: fmt(bv.bias2, 4) }, { label: "\\text{variance}", tex: true, value: fmt(bv.variance, 4) }, { label: "\\sigma^2\\ \\text{(irreducible)}", tex: true, value: fmt(noise * noise, 4) }] : []),
        { label: "\\|\\mathbf w\\|_2", tex: true, value: fmt(Math.hypot(...model.w.slice(1))) },
      ]} />}
      calc={<CalcSteps steps={[
        mode === "ridge" ? { tex: "\\min_{\\mathbf w}\\ \\tfrac{1}{2n}\\|\\mathbf y-X\\mathbf w\\|^2+\\tfrac{\\lambda}{2}\\|\\mathbf w\\|_2^2\\;\\Rightarrow\\;(X^\\top X+n\\lambda I)\\mathbf w=X^\\top\\mathbf y" }
          : mode === "lasso" ? { tex: "\\min_{\\mathbf w}\\ \\tfrac{1}{2n}\\|\\mathbf y-X\\mathbf w\\|^2+\\lambda\\|\\mathbf w\\|_1", note: "solved by coordinate descent with soft-thresholding" }
            : mode === "elastic" ? { tex: "\\min_{\\mathbf w}\\ \\tfrac{1}{2n}\\|\\mathbf y-X\\mathbf w\\|^2+\\lambda\\big(\\alpha\\|\\mathbf w\\|_1+\\tfrac{1-\\alpha}{2}\\|\\mathbf w\\|_2^2\\big)" }
              : mode === "bias-variance" ? { tex: "E[(y-\\hat f(x))^2]=\\text{Bias}[\\hat f(x)]^2+\\operatorname{Var}[\\hat f(x)]+\\sigma^2" }
                : { tex: `\\hat y=\\sum_{k=0}^{${deg}}w_kx^k,\\quad\\mathbf w=\\arg\\min\\|\\mathbf y-X\\mathbf w\\|^2` },
        { tex: `\\text{features}=${deg}\\ \\text{(+ bias)},\\ n=${n}` },
      ]} />}
      note={mode === "bias-variance" ? "Each faint curve is the same model fitted to a fresh training sample. Low degree: curves agree but are wrong (bias). High degree: curves disagree wildly (variance)." : regularised ? "The intercept is never penalised. Features are standardised first so the penalty treats all powers of x fairly." : "Training error always falls as degree rises; test error falls then rises — the signature of overfitting."}
    />
  );
}

export { tex };
