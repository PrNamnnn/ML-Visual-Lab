"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, Handle, Polyline, HeatmapCanvas, C, useThemeRGB } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Toggle, AnimationControls, Select } from "@/components/controls";
import { useTicker } from "@/hooks/useSize";
import { mulberry32, randn } from "@/lib/math/random";
import { makeClusters, makeDataset, kmeansAssign, kmeansUpdate, kmeansInit, inertia, type P2 } from "@/lib/ml/models";
import { fmt, tex, clamp } from "@/lib/utilities/format";
import { clsColor, BarList, MiniTable } from "@/visualizers/shared";
import { Trace, LineWB } from "./common";

type Mode = "objective" | "lloyd" | "geometry" | "plusplus";
type DS = "blobs" | "close" | "rings" | "moons" | "stretched";
const NAMES = ["blue", "pink", "green", "amber", "red", "muted"] as const;

function dataset(id: DS, seed = 3): P2[] {
  if (id === "blobs") return makeClusters(3, 40, 0.38, seed);
  if (id === "close") return makeClusters(4, 32, 0.5, seed + 4);
  if (id === "rings") return makeDataset("circles", 150, 0.07, seed).map((p) => [p.x, p.y] as P2);
  if (id === "moons") return makeDataset("moons", 150, 0.1, seed).map((p) => [p.x, p.y] as P2);
  const rng = mulberry32(seed);
  return Array.from({ length: 120 }, (_, i) => [randn(rng) * 1.5, (i % 2 ? 0.75 : -0.75) + randn(rng) * 0.16] as P2);
}
const DS_OPTS: { value: DS; label: string }[] = [{ value: "blobs", label: "Three blobs" }, { value: "close", label: "Four overlapping blobs" }, { value: "rings", label: "Two rings" }, { value: "moons", label: "Two moons" }, { value: "stretched", label: "Two long strips" }];

/** Runs Lloyd's algorithm to convergence and returns the final state. */
function lloyd(pts: P2[], C0: P2[], maxIter = 100) {
  let Cs = C0, z = kmeansAssign(pts, Cs), it = 0;
  for (; it < maxIter; it++) {
    Cs = kmeansUpdate(pts, z, Cs);
    const nz = kmeansAssign(pts, Cs);
    if (nz.every((v, i) => v === z[i])) break;
    z = nz;
  }
  return { C: Cs, z, F: inertia(pts, z, Cs), iters: it + 1 };
}

function Regions({ ctx, Cs, deps }: { ctx: Parameters<typeof HeatmapCanvas>[0]["ctx"]; Cs: P2[]; deps: unknown[] }) {
  const th = useThemeRGB();
  const cols = [th.blue, th.pink, th.green, th.amber, th.red, th.muted];
  return (
    <HeatmapCanvas ctx={ctx} res={3} deps={[...deps, th.blue[0], th.panel[0]]} f={(x, y) => {
      let b = 0, bd = Infinity;
      Cs.forEach((c, j) => { const d = (x - c[0]) ** 2 + (y - c[1]) ** 2; if (d < bd) { bd = d; b = j; } });
      const c = cols[b % cols.length], p = th.panel, a = 0.16;
      return [p[0] + (c[0] - p[0]) * a, p[1] + (c[1] - p[1]) * a, p[2] + (c[2] - p[2]) * a, 255];
    }} />
  );
}

const Centroid = ({ p, i }: { p: P2; i: number }) => (
  <g>
    <Dot p={p} r={8} color={C.bg} stroke={clsColor(i)} />
    <Dot p={p} r={4} color={clsColor(i)} />
  </g>
);

/* ------------------------------------------------------------------ the objective function */
function ObjectiveViz() {
  const pts = useMemo(() => dataset("blobs"), []);
  const k = 3;
  const [Cs, setCs] = useState<P2[]>([[-1.6, 1.5], [1.6, 1.4], [0.2, -1.8]]);
  const [live, setLive] = useState(true);
  const [frozen, setFrozen] = useState<number[]>(() => kmeansAssign(pts, [[-1.6, 1.5], [1.6, 1.4], [0.2, -1.8]]));
  const [sel, setSel] = useState(0);
  const z = live ? kmeansAssign(pts, Cs) : frozen;
  const per = Array.from({ length: k }, (_, j) => pts.reduce((s, p, i) => (z[i] === j ? s + (p[0] - Cs[j][0]) ** 2 + (p[1] - Cs[j][1]) ** 2 : s), 0));
  const F = per.reduce((a, b) => a + b, 0);
  const means = kmeansUpdate(pts, z, Cs);
  const Fbest = inertia(pts, z, means);
  const m = z.filter((v) => v === sel).length;
  const within = pts.reduce((s, p, i) => (z[i] === sel ? s + (p[0] - means[sel][0]) ** 2 + (p[1] - means[sel][1]) ** 2 : s), 0);
  const shift = (means[sel][0] - Cs[sel][0]) ** 2 + (means[sel][1] - Cs[sel][1]) ** 2;
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-3.4, 3.4]} y={[-3, 3]} equal aspect={0.72} maxH={470} ariaLabel="points, centroids and their distances">
            {pts.map((p, i) => <Polyline key={`l${i}`} pts={[p, Cs[z[i]]]} style={{ stroke: clsColor(z[i]), strokeWidth: 0.7, opacity: 0.35 }} />)}
            {pts.map((p, i) => <Dot key={i} p={p} r={3.2} color={clsColor(z[i])} />)}
            {means.map((p, j) => <Dot key={`m${j}`} p={p} r={4} color="transparent" stroke={C.fg} />)}
            {Cs.map((c, j) => <Handle key={j} p={c} color={NAMES[j]} r={8} label={`centroid ${j + 1}`} onDrag={(q) => { setSel(j); setCs(Cs.map((o, t) => (t === j ? [clamp(q[0], -3.2, 3.2), clamp(q[1], -2.8, 2.8)] : o))); }} />)}
          </Plot>
          <Legend items={[{ label: "points coloured by cluster zᵢ", color: C.blue }, { label: "centroids μₖ (drag)", color: C.pink }, { label: "mean of each cluster (ring)", color: C.fg }]} />
        </div>
      }
      controls={
        <>
          <Toggle label="Reassign points to the nearest centroid" checked={live} onChange={(v) => { if (!v) setFrozen(kmeansAssign(pts, Cs)); setLive(v); }} />
          <button className="btn w-full" onClick={() => setCs(means)}>Move centroids to the cluster means</button>
          <button className="btn w-full" onClick={() => { setCs([[-1.6, 1.5], [1.6, 1.4], [0.2, -1.8]]); setLive(true); }}>Reset</button>
          <div>
            <div className="label mb-2">Contribution of each cluster to F</div>
            <BarList digits={2} items={per.map((v, j) => ({ label: `cluster ${j + 1} (${z.filter((t) => t === j).length} pts)`, value: v, color: clsColor(j) }))} />
          </div>
          <p className="text-[12.5px] leading-relaxed text-muted">Switch reassignment off and drag one centroid: with the assignments fixed, F is smallest exactly when the centroid sits on the ring (the cluster mean).</p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`F(\mathbf z,\boldsymbol\mu)`, tex: true, value: fmt(F, 3), color: C.fg },
        { label: String.raw`F\ \text{with means as centroids}`, tex: true, value: fmt(Fbest, 3), color: C.green },
        { label: String.raw`F-F_{\text{means}}`, tex: true, value: fmt(F - Fbest, 3) },
        { label: "n", tex: true, value: pts.length },
        { label: "k", tex: true, value: k },
      ]} />}
      calc={<CalcSteps title={`Live calculation for cluster ${sel + 1}`} steps={[
        { tex: String.raw`F=\sum_{i=1}^{n}\|\mathbf x_i-\boldsymbol\mu_{z_i}\|^2=${per.map((v) => tex(v, 2)).join("+")}=${tex(F, 3)}` },
        { tex: String.raw`\sum_{i:z_i=${sel + 1}}\|\mathbf x_i-\boldsymbol\mu_{${sel + 1}}\|^2=\underbrace{\sum_{i:z_i=${sel + 1}}\|\mathbf x_i-\bar{\mathbf x}\|^2}_{${tex(within, 3)}}+\underbrace{m\,\|\bar{\mathbf x}-\boldsymbol\mu_{${sel + 1}}\|^2}_{${m}\times${tex(shift, 3)}}=${tex(within + m * shift, 3)}`, note: `measured directly: ${fmt(per[sel], 3)}` },
      ]} />}
      note="The second line is why the mean is the best centre: the cost of a cluster splits into a part that does not depend on the centroid and a penalty m‖x̄ − μ‖² that is zero only at the mean."
    />
  );
}

/* ------------------------------------------------------------------ Lloyd's algorithm, half-step by half-step */
interface LState { C: P2[]; z: number[]; F: number; phase: "init" | "assign" | "update"; moved: number }

function LloydViz() {
  const [ds, setDs] = useState<DS>("close");
  const [k, setK] = useState(4);
  const [seed, setSeed] = useState(2);
  const [playing, setPlaying] = useState(false);
  const pts = useMemo(() => dataset(ds), [ds]);
  const start = useMemo((): LState[] => {
    const C0 = kmeansInit(pts, k, "random", seed);
    return [{ C: C0, z: pts.map(() => -1), F: NaN, phase: "init", moved: 0 }];
  }, [pts, k, seed]);
  const [hist, setHist] = useState<LState[] | null>(null);
  const H = hist && hist[0] === start[0] ? hist : start;
  const cur = H[H.length - 1];
  const done = H.length > 2 && cur.phase === "assign" && cur.moved === 0;
  const step = () => {
    if (done) return;
    const last = H[H.length - 1];
    if (last.phase === "init" || last.phase === "update") {
      const z = kmeansAssign(pts, last.C);
      // ties are resolved in favour of the lowest index, so an unchanged point keeps its cluster
      const moved = last.phase === "init" ? pts.length : z.filter((v, i) => v !== last.z[i]).length;
      setHist([...H, { C: last.C, z, F: inertia(pts, z, last.C), phase: "assign", moved }]);
    } else {
      const Cn = kmeansUpdate(pts, last.z, last.C);
      setHist([...H, { C: Cn, z: last.z, F: inertia(pts, last.z, Cn), phase: "update", moved: 0 }]);
    }
  };
  useTicker(playing && !done, 650, step);
  const Fs = H.slice(1).map((h) => h.F);
  const iter = Math.floor((H.length - 1) / 2);
  const prev = H.length > 2 ? H[H.length - 2] : null;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-3.4, 3.4]} y={[-3, 3]} equal aspect={0.68} maxH={430} ariaLabel="k-means clustering state" underlay={(ctx) => <Regions ctx={ctx} Cs={cur.C} deps={[cur.C]} />}>
            {prev && cur.phase === "update" && prev.C.map((c, j) => <Polyline key={`mv${j}`} pts={[c, cur.C[j]]} style={{ stroke: C.fg, strokeWidth: 1.2, strokeDasharray: "3 3" }} />)}
            {pts.map((p, i) => <Dot key={i} p={p} r={3.2} color={cur.z[i] < 0 ? C.a("fg", 0.5) : clsColor(cur.z[i])} stroke={prev && cur.phase === "assign" && prev.z[i] >= 0 && prev.z[i] !== cur.z[i] ? C.fg : undefined} />)}
            {cur.C.map((c, j) => <Centroid key={j} p={c} i={j} />)}
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Objective F after every half-step (assign, update, assign, …) — it never goes up</div>
            <Trace ariaLabel="k-means objective after each half step" xLabel="half-step" series={[{ pts: Fs.length ? Fs : [0], color: C.green, dots: true }]} xMax={Math.max(6, Fs.length - 1)} />
          </div>
          <Legend items={[{ label: "centroids", color: C.fg }, { label: "points that just changed cluster (ringed)", color: C.muted }, { label: "shaded cells: nearest-centroid regions", color: C.a("blue", 0.5) }]} />
        </div>
      }
      controls={
        <>
          <Select label="Dataset" value={ds} onChange={(v) => { setPlaying(false); setDs(v); }} options={DS_OPTS} />
          <ParameterSlider label="clusters k" value={k} min={2} max={6} step={1} onChange={(v) => { setPlaying(false); setK(v); }} />
          <AnimationControls playing={playing && !done} onPlay={() => setPlaying(!playing)} onStep={step} canStep={!done} onReset={() => { setPlaying(false); setHist(null); }} />
          <button className="btn w-full" onClick={() => { setPlaying(false); setSeed((s) => s + 1); }}>New random initialisation</button>
          <p className="text-[12.5px] leading-relaxed text-muted">
            {cur.phase === "init" ? "Centroids are k randomly chosen data points. Step to assign every point to its nearest centroid." : done ? `Converged after ${iter} iteration${iter === 1 ? "" : "s"}: the last assignment step moved no point.` : cur.phase === "assign" ? `Assignment step: ${cur.moved} point${cur.moved === 1 ? "" : "s"} changed cluster. Next, each centroid moves to the mean of its points.` : "Update step: centroids moved to the cluster means. Next, points are reassigned."}
          </p>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{iteration } t`, tex: true, value: iter },
        { label: String.raw`F(\mathbf z^t,\boldsymbol\mu^t)`, tex: true, value: Number.isFinite(cur.F) ? fmt(cur.F, 3) : "—", color: C.green },
        { label: String.raw`\Delta F\ \text{(last half-step)}`, tex: true, value: Fs.length > 1 ? fmt(Fs[Fs.length - 1] - Fs[Fs.length - 2], 4) : "—" },
        { label: String.raw`\text{points moved}`, tex: true, value: cur.phase === "assign" ? cur.moved : "—" },
        { label: String.raw`\text{status}`, tex: true, value: done ? "converged" : "running", color: done ? C.green : undefined },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{assign: } z_i^{t+1}=\arg\min_{k}\|\mathbf x_i-\boldsymbol\mu_k^{t}\|^2\ \Rightarrow\ F(\mathbf z^{t+1},\boldsymbol\mu^{t})\le F(\mathbf z^{t},\boldsymbol\mu^{t})`, note: "each point can only get closer to its centre" },
        { tex: String.raw`\text{update: }\boldsymbol\mu_k^{t+1}=\frac{1}{|\{i:z_i^{t+1}=k\}|}\sum_{i:z_i^{t+1}=k}\mathbf x_i\ \Rightarrow\ F(\mathbf z^{t+1},\boldsymbol\mu^{t+1})\le F(\mathbf z^{t+1},\boldsymbol\mu^{t})`, note: "the mean minimises the within-cluster sum" },
        ...(Fs.length > 1 ? [{ tex: String.raw`F:\ ${Fs.slice(-5).map((v) => tex(v, 2)).join(String.raw`\ \to\ `)}`, note: "most recent values" }] : []),
      ]} />}
      note="Different initialisations can stop at different objective values: Lloyd's algorithm always converges, but only to a local minimum. Try several initialisations on the overlapping blobs."
    />
  );
}

/* ------------------------------------------------------------------ Voronoi geometry, choice of k */
function GeometryViz() {
  const [ds, setDs] = useState<DS>("blobs");
  const [k, setK] = useState(3);
  const [pair, setPair] = useState(0);
  const pts = useMemo(() => dataset(ds), [ds]);
  const best = (kk: number) => {
    let b: ReturnType<typeof lloyd> | null = null;
    for (let s = 1; s <= 8; s++) { const r = lloyd(pts, kmeansInit(pts, kk, "kmeans++", s * 13 + kk)); if (!b || r.F < b.F) b = r; }
    return b!;
  };
  const fit = useMemo(() => best(k), [pts, k]); // eslint-disable-line react-hooks/exhaustive-deps
  const elbow = useMemo(() => [NaN, ...Array.from({ length: 8 }, (_, i) => best(i + 1).F)], [pts]); // eslint-disable-line react-hooks/exhaustive-deps
  const pairs: [number, number][] = [];
  for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) pairs.push([a, b]);
  const [a, b] = pairs[Math.min(pair, pairs.length - 1)] ?? [0, 1];
  const ma = fit.C[a], mb = fit.C[b] ?? fit.C[a];
  const w: P2 = [2 * (mb[0] - ma[0]), 2 * (mb[1] - ma[1])];
  const rhs = mb[0] ** 2 + mb[1] ** 2 - ma[0] ** 2 - ma[1] ** 2;
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-3.4, 3.4]} y={[-3, 3]} equal aspect={0.68} maxH={430} ariaLabel="k-means cluster regions" underlay={(ctx) => <Regions ctx={ctx} Cs={fit.C} deps={[fit.C]} />}>
            {k > 1 && <LineWB w={w} b={-rhs} style={{ stroke: C.fg, strokeWidth: 1.6, strokeDasharray: "6 4" }} />}
            {k > 1 && <Polyline pts={[ma, mb]} style={{ stroke: C.a("fg", 0.6), strokeWidth: 1 }} />}
            {pts.map((p, i) => <Dot key={i} p={p} r={3.1} color={clsColor(fit.z[i])} />)}
            {fit.C.map((c, j) => <Centroid key={j} p={c} i={j} />)}
          </Plot>
          <div>
            <div className="mb-1 text-[12px] text-faint">Best objective found for each k (8 restarts). F always falls as k grows, so it cannot choose k by itself</div>
            <Trace ariaLabel="objective versus number of clusters" xLabel="k" xMax={8} marker={k} series={[{ pts: elbow, color: C.green, dots: true }]} />
          </div>
          <Legend items={[{ label: "cluster regions (Voronoi cells)", color: C.a("blue", 0.5) }, { label: `boundary between clusters ${a + 1} and ${b + 1}`, color: C.fg, dashed: true }]} />
        </div>
      }
      controls={
        <>
          <Select label="Dataset" value={ds} onChange={(v) => { setDs(v); setPair(0); }} options={DS_OPTS} />
          <ParameterSlider label="clusters k" value={k} min={1} max={8} step={1} onChange={(v) => { setK(v); setPair(0); }} />
          {pairs.length > 1 && <ParameterSlider label="boundary to inspect" value={Math.min(pair, pairs.length - 1) + 1} min={1} max={pairs.length} step={1} onChange={(v) => setPair(v - 1)} format={() => `${a + 1} | ${b + 1}`} />}
          <p className="text-[12.5px] leading-relaxed text-muted">
            {ds === "rings" || ds === "moons" ? "These clusters are not convex. Every k-means region is an intersection of half-planes, so no choice of centroids can trace a ring or a crescent." : ds === "stretched" ? "K-means measures plain Euclidean distance, so it prefers to cut the long strips across rather than separate them." : "Each region is the set of points closer to one centroid than to any other: a convex polygon bounded by perpendicular bisectors."}
          </p>
        </>
      }
      readouts={<Readouts items={[
        { label: "F", tex: true, value: fmt(fit.F, 3), color: C.green },
        { label: String.raw`\text{iterations}`, tex: true, value: fit.iters },
        { label: String.raw`\boldsymbol\mu_{${a + 1}}`, tex: true, value: `(${fmt(ma[0], 2)}, ${fmt(ma[1], 2)})`, color: clsColor(a) },
        { label: String.raw`\boldsymbol\mu_{${b + 1}}`, tex: true, value: k > 1 ? `(${fmt(mb[0], 2)}, ${fmt(mb[1], 2)})` : "—", color: clsColor(b) },
      ]} />}
      calc={k > 1 ? <CalcSteps steps={[
        { tex: String.raw`\|\mathbf x-\boldsymbol\mu_{${a + 1}}\|^2=\|\mathbf x-\boldsymbol\mu_{${b + 1}}\|^2\iff 2(\boldsymbol\mu_{${b + 1}}-\boldsymbol\mu_{${a + 1}})^\top\mathbf x=\|\boldsymbol\mu_{${b + 1}}\|^2-\|\boldsymbol\mu_{${a + 1}}\|^2` },
        { tex: String.raw`${tex(w[0], 3)}\,x_1+${tex(w[1], 3)}\,x_2=${tex(rhs, 3)}`, note: "a straight line: the perpendicular bisector of the two centroids" },
        { tex: String.raw`\text{midpoint }\tfrac12(\boldsymbol\mu_{${a + 1}}+\boldsymbol\mu_{${b + 1}})=(${tex((ma[0] + mb[0]) / 2, 3)},\ ${tex((ma[1] + mb[1]) / 2, 3)})\ \Rightarrow\ ${tex((w[0] * (ma[0] + mb[0])) / 2 + (w[1] * (ma[1] + mb[1])) / 2, 3)}=${tex(rhs, 3)}`, note: "the midpoint satisfies the equation" },
      ]} /> : undefined}
      note="The elbow heuristic looks for the k after which F stops dropping quickly. For the three blobs the bend is at k = 3; for rings and moons there is no clean elbow because the model itself does not fit."
    />
  );
}

/* ------------------------------------------------------------------ k-means++ seeding */
function PlusPlusViz() {
  const [ds, setDs] = useState<DS>("close");
  const [k, setK] = useState(4);
  const [seed, setSeed] = useState(1);
  const [chosen, setChosen] = useState<number[] | null>(null);
  const pts = useMemo(() => dataset(ds), [ds]);
  const first = useMemo(() => Math.floor(mulberry32(seed * 977)() * pts.length), [seed, pts]);
  const ch = chosen && chosen[0] === first ? chosen : [first];
  const D2 = pts.map((p) => Math.min(...ch.map((c) => (p[0] - pts[c][0]) ** 2 + (p[1] - pts[c][1]) ** 2)));
  const tot = D2.reduce((a, b) => a + b, 0) || 1;
  const prob = D2.map((v) => v / tot);
  const full = ch.length >= k;
  const next = () => {
    if (full) return;
    let r = mulberry32(seed * 131 + ch.length * 17)() * tot, i = 0;
    for (; i < D2.length - 1; i++) { r -= D2[i]; if (r <= 0) break; }
    setChosen([...ch, i]);
  };
  const compare = useMemo(() => {
    const run = (method: "random" | "kmeans++") => Array.from({ length: 60 }, (_, s) => lloyd(pts, kmeansInit(pts, k, method, s + 1)).F);
    const R = run("random"), P = run("kmeans++");
    const opt = Math.min(...R, ...P);
    const stats = (v: number[]) => ({ mean: v.reduce((x, y) => x + y, 0) / v.length, best: Math.min(...v), worst: Math.max(...v), hit: v.filter((x) => x <= opt * 1.02).length / v.length });
    return { r: stats(R), p: stats(P), opt };
  }, [pts, k]);
  const top = prob.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 4);
  const pMax = Math.max(...prob, 1e-9);
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-2">
          <Plot x={[-3.4, 3.4]} y={[-3, 3]} equal aspect={0.7} maxH={450} ariaLabel="k-means plus plus seeding probabilities">
            {pts.map((p, i) => <Dot key={i} p={p} r={full ? 3 : 2 + 7 * Math.sqrt(prob[i] / pMax)} color={full ? C.a("fg", 0.45) : C.a("blue", 0.25 + 0.6 * (prob[i] / pMax))} />)}
            {ch.map((c, j) => <Centroid key={j} p={pts[c]} i={j} />)}
          </Plot>
          <Legend items={[{ label: "chosen centres", color: C.fg }, { label: "dot size ∝ probability of being chosen next", color: C.blue }]} />
        </div>
      }
      controls={
        <>
          <Select label="Dataset" value={ds} onChange={(v) => { setDs(v); setChosen(null); }} options={DS_OPTS} />
          <ParameterSlider label="clusters k" value={k} min={2} max={6} step={1} onChange={(v) => { setK(v); setChosen(null); }} />
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={next} disabled={full}>Choose centre {Math.min(k, ch.length + 1)}</button>
            <button className="btn flex-1" onClick={() => { setSeed((s) => s + 1); setChosen(null); }}>Restart</button>
          </div>
          <div>
            <div className="label mb-2">Final objective over 60 runs of each initialisation</div>
            <MiniTable head={["initialisation", "mean F", "best", "worst", "runs within 2% of best"]} rows={[
              ["uniform random", fmt(compare.r.mean, 2), fmt(compare.r.best, 2), fmt(compare.r.worst, 2), `${Math.round(compare.r.hit * 100)}%`],
              ["k-means++", fmt(compare.p.mean, 2), fmt(compare.p.best, 2), fmt(compare.p.worst, 2), `${Math.round(compare.p.hit * 100)}%`],
            ]} />
          </div>
        </>
      }
      readouts={<Readouts items={[
        { label: String.raw`\text{centres chosen}`, tex: true, value: `${ch.length} / ${k}` },
        { label: String.raw`\textstyle\sum_j D(\mathbf x_j)^2`, tex: true, value: fmt(tot, 3) },
        { label: String.raw`\max_i p_i`, tex: true, value: fmt(pMax, 4) },
        { label: String.raw`\text{uniform would give}`, tex: true, value: fmt(1 / pts.length, 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`D(\mathbf x_i)^2=\min_{c\in\text{chosen}}\|\mathbf x_i-\boldsymbol\mu_c\|^2,\qquad p_i=\frac{D(\mathbf x_i)^2}{\sum_jD(\mathbf x_j)^2}` },
        ...top.slice(0, 2).map((t) => ({ tex: String.raw`\mathbf x_{${t.i + 1}}=(${tex(pts[t.i][0], 2)},${tex(pts[t.i][1], 2)}):\quad p=\frac{${tex(D2[t.i], 3)}}{${tex(tot, 3)}}=${tex(t.p, 4)}`, note: "among the most likely next centres" })),
        { tex: String.raw`\mathbb E[F_{\text{k-means++}}]\le 8(\ln k+2)\,F_{\text{opt}}=${tex(8 * (Math.log(k) + 2), 2)}\,F_{\text{opt}}`, note: "guarantee of Arthur and Vassilvitskii, in expectation over the seeding" },
      ]} />}
      note="Already chosen centres and their neighbours have D² ≈ 0, so the next centre almost always lands in a region that has none yet. The table reruns the complete algorithm 60 times per method on this dataset."
    />
  );
}

export default function KMeansWeek({ mode = "lloyd" }: { mode?: Mode }) {
  if (mode === "objective") return <ObjectiveViz />;
  if (mode === "geometry") return <GeometryViz />;
  if (mode === "plusplus") return <PlusPlusViz />;
  return <LloydViz />;
}
