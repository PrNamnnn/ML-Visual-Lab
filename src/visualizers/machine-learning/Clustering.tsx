"use client";
import { useEffect, useMemo, useState } from "react";
import { Plot, Dot, C, Polyline, usePlot } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented, AnimationControls } from "@/components/controls";
import { makeClusters, makeDataset, kmeansInit, kmeansAssign, kmeansUpdate, inertia, dbscan, gmmEStep, gmmMStep, agglomerative, cutTree, type P2, type GMMComp } from "@/lib/ml/models";
import { eigSym } from "@/lib/math/linalg";
import { useTicker, useSize } from "@/hooks/useSize";
import { fmt, tex } from "@/lib/utilities/format";

type Algo = "kmeans" | "dbscan" | "gmm" | "hierarchical";
const PAL = ["#5aa9ff", "#ff6fae", "#6ecf97", "#e6c35c", "#c58bff", "#f59e5b", "#38d4c4", "#e88fc0"];

function useData(kind: "blobs" | "moons" | "aniso" | "varied", seed: number): P2[] {
  return useMemo(() => {
    if (kind === "moons") return makeDataset("moons", 200, 0.08, seed).map((p) => [p.x * 1.2, p.y * 1.2] as P2);
    if (kind === "aniso") return makeClusters(3, 60, 0.3, seed).map(([x, y]) => [x * 0.6 + y * 0.9, y * 0.4 - x * 0.2] as P2);
    if (kind === "varied") return [...makeClusters(1, 80, 0.15, seed).map(([x, y]) => [x - 0.5, y] as P2), ...makeClusters(1, 60, 0.6, seed + 1).map(([x, y]) => [x + 1.2, y - 0.4] as P2)];
    return makeClusters(4, 45, 0.32, seed);
  }, [kind, seed]);
}

function CovEllipse({ c, color }: { c: GMMComp; color: string }) {
  const { sx, sy } = usePlot();
  const { values, vectors } = eigSym(c.cov as unknown as number[][]);
  const pts: string[] = [];
  for (let i = 0; i <= 60; i++) {
    const t = (i / 60) * 2 * Math.PI;
    const a = 2 * Math.sqrt(Math.max(0, values[0])) * Math.cos(t), b = 2 * Math.sqrt(Math.max(0, values[1])) * Math.sin(t);
    const x = c.mu[0] + a * vectors[0][0] + b * vectors[0][1], y = c.mu[1] + a * vectors[1][0] + b * vectors[1][1];
    pts.push(`${i ? "L" : "M"}${sx(x)},${sy(y)}`);
  }
  return <path d={pts.join("") + "Z"} style={{ fill: "none", stroke: color, strokeWidth: 1.8 }} />;
}

export default function Clustering({ algo: initial = "kmeans", lab = false }: { algo?: Algo; lab?: boolean }) {
  const [algo, setAlgo] = useState<Algo>(initial);
  const [kind, setKind] = useState<"blobs" | "moons" | "aniso" | "varied">(initial === "dbscan" ? "moons" : initial === "gmm" ? "aniso" : "blobs");
  const [seed, setSeed] = useState(2);
  const pts = useData(kind, seed);
  const sub = useMemo(() => pts.filter((_, i) => i % 4 === 0), [pts]);
  const sel = AlgoSel();
  return (
    <>
      {algo === "kmeans" && <KMeans pts={pts} lab={lab} algoSel={sel} />}
      {algo === "dbscan" && <DBSCAN pts={pts} algoSel={sel} />}
      {algo === "gmm" && <GMM pts={pts} algoSel={sel} />}
      {algo === "hierarchical" && <Hierarchical pts={sub} algoSel={sel} />}
    </>
  );
  function AlgoSel() {
    return (
      <>
        {lab && <Select label="Algorithm" value={algo} onChange={setAlgo} options={[{ value: "kmeans", label: "K-Means / K-Means++" }, { value: "dbscan", label: "DBSCAN" }, { value: "gmm", label: "Gaussian mixture (EM)" }, { value: "hierarchical", label: "Hierarchical (agglomerative)" }]} />}
        <Select label="Dataset" value={kind} onChange={setKind} options={[{ value: "blobs", label: "4 round blobs" }, { value: "aniso", label: "Elongated clusters" }, { value: "varied", label: "Different densities" }, { value: "moons", label: "Moons (non-convex)" }]} />
        <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>New data</button>
      </>
    );
  }
}

function KMeans({ pts, algoSel }: { pts: P2[]; lab: boolean; algoSel: React.ReactNode }) {
  const [k, setK] = useState(4);
  const [init, setInit] = useState<"random" | "kmeans++">("kmeans++");
  const [iseed, setIseed] = useState(1);
  const [C0, setC0] = useState<P2[]>([]);
  const [hist, setHist] = useState<{ C: P2[]; a: number[] }[]>([]);
  const [phase, setPhase] = useState<"assign" | "update">("assign");
  const [playing, setPlaying] = useState(false);
  useEffect(() => { const c = kmeansInit(pts, k, init, iseed); setC0(c); setHist([{ C: c, a: kmeansAssign(pts, c) }]); setPhase("update"); setPlaying(false); }, [pts, k, init, iseed]);
  const cur = hist[hist.length - 1];
  const converged = hist.length > 2 && JSON.stringify(hist[hist.length - 1].a) === JSON.stringify(hist[hist.length - 2].a) && phase === "update";
  const step = () => {
    if (!cur) return;
    if (phase === "update") { setHist([...hist, { C: kmeansUpdate(pts, cur.a, cur.C), a: cur.a }]); setPhase("assign"); }
    else { setHist([...hist, { C: cur.C, a: kmeansAssign(pts, cur.C) }]); setPhase("update"); }
  };
  useTicker(playing && !converged, 500, step);
  useEffect(() => { if (converged) setPlaying(false); }, [converged]);
  if (!cur) return null;
  const J = inertia(pts, cur.a, cur.C);
  const iters = Math.floor((hist.length - 1) / 2);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-2.6, 2.6]} equal aspect={0.72} maxH={480} ariaLabel="k-means clustering">
            {pts.map((p, i) => <Polyline key={`l${i}`} pts={[p, cur.C[cur.a[i]]]} style={{ stroke: PAL[cur.a[i] % 8] + "22", strokeWidth: 0.7 }} />)}
            {pts.map((p, i) => <Dot key={i} p={p} r={3} color={PAL[cur.a[i] % 8]} />)}
            {hist.filter((_, i) => i % 2 === 0 || i === hist.length - 1).length > 1 && cur.C.map((_, j) => <Polyline key={`tr${j}`} pts={hist.map((h) => h.C[j])} style={{ stroke: C.a("fg", 0.6), strokeWidth: 1, strokeDasharray: "3 2" }} />)}
            {cur.C.map((c, j) => <g key={j}><Dot p={c} r={9} color={C.bg} /><Dot p={c} r={7} color={PAL[j % 8]} stroke={C.fg} /></g>)}
          </Plot>
          <Legend items={[{ label: "centroids (dashed = path)", color: C.fg }, { label: `next step: ${phase === "update" ? "move centroids to cluster means" : "reassign points to nearest centroid"}`, color: C.muted }]} />
        </div>
      }
      controls={
        <>
          {algoSel}
          <ParameterSlider label="k clusters" value={k} min={1} max={8} step={1} onChange={setK} />
          <Segmented label="Initialisation" value={init} onChange={setInit} options={[{ value: "random", label: "Random points" }, { value: "kmeans++", label: "K-Means++" }]} />
          <button className="btn w-full" onClick={() => setIseed((s) => s + 1)}>Re-initialise centroids</button>
          <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onStep={step} canStep={!converged} onReset={() => { setPlaying(false); setHist([{ C: C0, a: kmeansAssign(pts, C0) }]); setPhase("update"); }} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{iteration}", tex: true, value: iters }, { label: "J=\\sum\\|\\mathbf x_i-\\boldsymbol\\mu_{c_i}\\|^2", tex: true, value: fmt(J) }, { label: "\\text{status}", tex: true, value: converged ? "converged" : "running" }, ...cur.C.map((_, j) => ({ label: `|C_${j + 1}|`, tex: true, value: cur.a.filter((a) => a === j).length, color: PAL[j % 8] }))]} />}
      calc={<CalcSteps steps={[{ tex: "c_i\\leftarrow\\arg\\min_j\\|\\mathbf x_i-\\boldsymbol\\mu_j\\|^2", note: "assignment step" }, { tex: "\\boldsymbol\\mu_j\\leftarrow\\frac{1}{|C_j|}\\sum_{i\\in C_j}\\mathbf x_i", note: "update step — each step can only decrease J" }]} />}
      note="K-Means assumes round, similarly sized clusters and finds a local optimum that depends on initialisation; K-Means++ spreads initial centroids (sampling ∝ squared distance) to reduce bad starts. Try the moons dataset to see it fail."
    />
  );
}

function DBSCAN({ pts, algoSel }: { pts: P2[]; algoSel: React.ReactNode }) {
  const [eps, setEps] = useState(0.25);
  const [minPts, setMinPts] = useState(5);
  const [showEps, setShowEps] = useState(true);
  const r = useMemo(() => dbscan(pts, eps, minPts), [pts, eps, minPts]);
  const counts = { core: r.role.filter((x) => x === "core").length, border: r.role.filter((x) => x === "border").length, noise: r.role.filter((x) => x === "noise").length };
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-2.6, 2.6]} equal aspect={0.72} maxH={480} ariaLabel="DBSCAN clustering">
            {showEps && <EpsCircles pts={pts.filter((_, i) => r.role[i] === "core")} eps={eps} />}
            {pts.map((p, i) => <Dot key={i} p={p} r={r.role[i] === "core" ? 3.6 : r.role[i] === "border" ? 3 : 3} color={r.role[i] === "noise" ? "transparent" : PAL[r.label[i] % 8]} stroke={r.role[i] === "noise" ? C.muted : r.role[i] === "border" ? C.fg : undefined} />)}
          </Plot>
          <Legend items={[{ label: "core (filled)", color: C.blue }, { label: "border (outlined)", color: C.fg }, { label: "noise (hollow grey)", color: C.muted }]} />
        </div>
      }
      controls={
        <>
          {algoSel}
          <ParameterSlider label="ε (neighbourhood radius)" value={eps} min={0.05} max={0.8} step={0.01} onChange={setEps} />
          <ParameterSlider label="minPts" value={minPts} min={2} max={20} step={1} onChange={setMinPts} />
          <Segmented value={showEps ? "on" : "off"} onChange={(v) => setShowEps(v === "on")} options={[{ value: "on", label: "Show ε-balls" }, { value: "off", label: "Hide" }]} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{clusters}", tex: true, value: r.clusters }, { label: "\\text{core}", tex: true, value: counts.core }, { label: "\\text{border}", tex: true, value: counts.border }, { label: "\\text{noise}", tex: true, value: counts.noise }]} />}
      calc={<CalcSteps steps={[{ tex: `|N_\\varepsilon(\\mathbf x)|\\ge${minPts}\\Rightarrow\\mathbf x\\text{ is a core point},\\quad N_\\varepsilon(\\mathbf x)=\\{\\mathbf y:\\|\\mathbf x-\\mathbf y\\|\\le${tex(eps, 2)}\\}` }, { tex: "\\text{clusters}=\\text{connected components of core points (+ reachable border points)}" }]} />}
      note="DBSCAN finds arbitrarily shaped clusters and labels outliers as noise, without choosing k. It struggles when clusters have very different densities (one ε cannot fit all)."
    />
  );
}

function EpsCircles({ pts, eps }: { pts: P2[]; eps: number }) {
  const { sx, sy } = usePlot();
  const r = Math.abs(sx(eps) - sx(0));
  return <g>{pts.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={r} style={{ fill: C.a("blue", 0.035), stroke: C.a("blue", 0.12) }} />)}</g>;
}

function GMM({ pts, algoSel }: { pts: P2[]; algoSel: React.ReactNode }) {
  const [K, setK] = useState(3);
  const [seed, setSeed] = useState(1);
  const init = useMemo<GMMComp[]>(() => kmeansInit(pts, K, "random", seed).map((m) => ({ pi: 1 / K, mu: m, cov: [[0.3, 0], [0, 0.3]] })), [pts, K, seed]);
  const [comps, setComps] = useState<GMMComp[]>(init);
  const [hist, setHist] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);
  useEffect(() => { setComps(init); setHist([]); setPlaying(false); }, [init]);
  const { R, logLik } = useMemo(() => gmmEStep(pts, comps), [pts, comps]);
  const step = () => {
    setHist((h) => [...h, logLik]);
    setComps(gmmMStep(pts, R));
  };
  const conv = hist.length > 2 && Math.abs(hist[hist.length - 1] - hist[hist.length - 2]) < 1e-4;
  useTicker(playing && !conv, 400, step);
  const mix = (r: number[]) => {
    let rr = 0, gg = 0, bb = 0;
    r.forEach((w, k) => { const c = PAL[k % 8]; rr += w * parseInt(c.slice(1, 3), 16); gg += w * parseInt(c.slice(3, 5), 16); bb += w * parseInt(c.slice(5, 7), 16); });
    return `rgb(${rr | 0},${gg | 0},${bb | 0})`;
  };
  const { ref, width } = useSize<HTMLDivElement>(0.2);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3, 3]} y={[-2.6, 2.6]} equal aspect={0.72} maxH={460} ariaLabel="Gaussian mixture model">
            {pts.map((p, i) => <Dot key={i} p={p} r={3} color={mix(R[i])} />)}
            {comps.map((c, k) => <CovEllipse key={k} c={c} color={PAL[k % 8]} />)}
            {comps.map((c, k) => <Dot key={`m${k}`} p={c.mu} r={5} color={PAL[k % 8]} stroke={C.fg} />)}
          </Plot>
          <div ref={ref}>
            <div className="mb-1 text-[11.5px] text-faint">Log-likelihood per EM iteration (never decreases)</div>
            {hist.length > 1 && (
              <svg width={width} height={60} role="img" aria-label="log-likelihood curve">
                {(() => { const lo = Math.min(...hist), hi = Math.max(...hist); const d = hist.map((v, i) => `${i ? "L" : "M"}${(i / Math.max(1, hist.length - 1)) * (width - 8) + 4},${56 - ((v - lo) / (hi - lo || 1)) * 50}`).join(""); return <path d={d} style={{ stroke: C.pink, fill: "none", strokeWidth: 1.6 }} />; })()}
              </svg>
            )}
          </div>
        </div>
      }
      controls={
        <>
          {algoSel}
          <ParameterSlider label="components K" value={K} min={1} max={6} step={1} onChange={setK} />
          <button className="btn w-full" onClick={() => setSeed((s) => s + 1)}>Re-initialise</button>
          <AnimationControls playing={playing} onPlay={() => setPlaying(!playing)} onStep={step} canStep={!conv} onReset={() => { setPlaying(false); setComps(init); setHist([]); }} />
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{EM iterations}", tex: true, value: hist.length }, { label: "\\log p(X\\mid\\theta)", tex: true, value: fmt(logLik, 2) }, ...comps.map((c, k) => ({ label: `\\pi_${k + 1}`, tex: true, value: fmt(c.pi), color: PAL[k % 8] }))]} />}
      calc={<CalcSteps steps={[{ tex: "\\text{E: }\\ r_{ik}=\\dfrac{\\pi_k\\,\\mathcal N(\\mathbf x_i\\mid\\boldsymbol\\mu_k,\\Sigma_k)}{\\sum_j\\pi_j\\,\\mathcal N(\\mathbf x_i\\mid\\boldsymbol\\mu_j,\\Sigma_j)}", note: "soft assignments (point colours blend)" }, { tex: "\\text{M: }\\ N_k=\\sum_ir_{ik},\\ \\boldsymbol\\mu_k=\\tfrac1{N_k}\\sum_ir_{ik}\\mathbf x_i,\\ \\Sigma_k=\\tfrac1{N_k}\\sum_ir_{ik}(\\mathbf x_i-\\boldsymbol\\mu_k)(\\mathbf x_i-\\boldsymbol\\mu_k)^\\top,\\ \\pi_k=N_k/n" }]} />}
      note="Each step shown is one full E+M iteration. A tiny ridge (1e-4·I) is added to each covariance to prevent collapse onto a single point."
    />
  );
}

function Hierarchical({ pts, algoSel }: { pts: P2[]; algoSel: React.ReactNode }) {
  const [linkage, setLinkage] = useState<"single" | "complete" | "average" | "ward">("ward");
  const [k, setK] = useState(3);
  const merges = useMemo(() => agglomerative(pts, linkage), [pts, linkage]);
  const labels = useMemo(() => cutTree(pts.length, merges, k), [pts, merges, k]);
  const n = pts.length;
  // dendrogram layout: order leaves by traversal
  const order: number[] = [];
  const children = new Map<number, [number, number]>();
  merges.forEach((m) => children.set(m.id, [m.a, m.b]));
  const walk = (id: number) => { const c = children.get(id); if (!c) order.push(id); else { walk(c[0]); walk(c[1]); } };
  if (merges.length) walk(merges[merges.length - 1].id);
  const xpos = new Map<number, number>(order.map((id, i) => [id, i]));
  const height = new Map<number, number>();
  for (let i = 0; i < n; i++) height.set(i, 0);
  merges.forEach((m) => { xpos.set(m.id, (xpos.get(m.a)! + xpos.get(m.b)!) / 2); height.set(m.id, m.dist); });
  const maxH = merges.length ? merges[merges.length - 1].dist : 1;
  const cutH = k <= 1 ? maxH * 1.02 : merges.length >= k ? (merges[n - k - 1]?.dist ?? 0) / 2 + (merges[n - k]?.dist ?? maxH) / 2 : 0;
  return (
    <LabShell
      viz={
        <div className="grid gap-3 md:grid-cols-2">
          <Plot x={[-3, 3]} y={[-2.6, 2.6]} equal aspect={0.85} maxH={380} ariaLabel="clusters from cutting the dendrogram">
            {pts.map((p, i) => <Dot key={i} p={p} r={4} color={PAL[labels[i] % 8]} />)}
          </Plot>
          <Plot x={[-1, n]} y={[0, maxH * 1.05]} aspect={0.85} maxH={380} grid={false} ariaLabel="dendrogram" margin={{ l: 36, b: 16 }}>
            {merges.map((m) => {
              const xa = xpos.get(m.a)!, xb = xpos.get(m.b)!, ha = height.get(m.a)!, hb = height.get(m.b)!;
              const col = m.dist <= cutH ? PAL[labels[order[Math.round(xa)]] % 8] : "rgb(var(--muted))";
              return <Polyline key={m.id} pts={[[xa, ha], [xa, m.dist], [xb, m.dist], [xb, hb]]} style={{ stroke: col, strokeWidth: 1.2 }} />;
            })}
            <Polyline pts={[[-1, cutH], [n, cutH]]} style={{ stroke: C.red, strokeWidth: 1.2, strokeDasharray: "5 3" }} />
          </Plot>
        </div>
      }
      controls={
        <>
          {algoSel}
          <Select label="Linkage" value={linkage} onChange={setLinkage} options={[{ value: "single", label: "Single (min distance)" }, { value: "complete", label: "Complete (max distance)" }, { value: "average", label: "Average" }, { value: "ward", label: "Ward (min variance increase)" }]} />
          <ParameterSlider label="clusters k (cut height)" value={k} min={1} max={8} step={1} onChange={setK} />
        </>
      }
      readouts={<Readouts items={[{ label: "n", tex: true, value: n }, { label: "\\text{merges}", tex: true, value: merges.length }, { label: "\\text{cut height}", tex: true, value: fmt(cutH) }]} />}
      calc={<CalcSteps steps={[{ tex: linkage === "single" ? "d(A,B)=\\min_{a\\in A,b\\in B}\\|a-b\\|" : linkage === "complete" ? "d(A,B)=\\max_{a\\in A,b\\in B}\\|a-b\\|" : linkage === "average" ? "d(A,B)=\\frac{1}{|A||B|}\\sum_{a,b}\\|a-b\\|" : "d(A,B)=\\sqrt{\\tfrac{2|A||B|}{|A|+|B|}}\\,\\|\\bar a-\\bar b\\|", note: "repeatedly merge the two closest clusters" }]} />}
      note="The dendrogram records every merge and its distance; cutting it at a height (red line) yields a flat clustering. Single linkage chains through thin bridges; Ward favours compact, equal-sized clusters. (Subsampled to 1 in 4 points for an O(n³) algorithm.)"
    />
  );
}
