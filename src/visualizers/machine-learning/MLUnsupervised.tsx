"use client";
import { useMemo, useState } from "react";
import { Plot, Dot, C, Polyline, HeatmapCanvas, useThemeRGB } from "@/components/visualization/Plot";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Toggle, AnimationControls } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { makeDataset, makeClusters, kmeansAssign, kmeansUpdate, type P2 } from "@/lib/ml/models";
import { useTicker } from "@/hooks/useSize";
import { fmt, pct, tex } from "@/lib/utilities/format";
import { jacobiEig, Heat, MiniTable, clsColor } from "@/visualizers/shared";

type RGB = [number, number, number];
const mix = (base: RGB, c: RGB, k: number): [number, number, number, number] => [base[0] + (c[0] - base[0]) * k, base[1] + (c[1] - base[1]) * k, base[2] + (c[2] - base[2]) * k, 255];
const kmeans2 = (P: P2[], init: P2[], it = 25) => { let Cc = init, a = kmeansAssign(P, Cc); for (let t = 0; t < it; t++) { Cc = kmeansUpdate(P, a, Cc); a = kmeansAssign(P, Cc); } return a; };
const agree = (a: number[], truth: number[]) => { const s = a.reduce((n, v, i) => n + (v === truth[i] ? 1 : 0), 0) / a.length; return Math.max(s, 1 - s); };

/* ------------------------------ spectral clustering ------------------------------ */
export function SpectralViz() {
  const [ds, setDs] = useState<"circles" | "moons">("circles");
  const [sigma, setSigma] = useState(0.2);
  const [show, setShow] = useState<"spectral" | "kmeans">("spectral");
  const pts = useMemo(() => makeDataset(ds, 120, ds === "circles" ? 0.08 : 0.08, 5), [ds]);
  const P: P2[] = pts.map((p) => [p.x, p.y]);
  const res = useMemo(() => {
    const n = P.length;
    const W = P.map((a, i) => P.map((b, j) => (i === j ? 0 : Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * sigma * sigma)))));
    const d = W.map((r) => r.reduce((s, x) => s + x, 0) + 1e-12);
    // normalised Laplacian L = I − D^{-1/2} W D^{-1/2}; take its two smallest eigenvectors (exact Jacobi solver)
    const Lsym = W.map((r, i) => r.map((x, j) => (i === j ? 1 : 0) - x / Math.sqrt(d[i] * d[j])));
    const { values, vectors } = jacobiEig(Lsym);
    const emb: P2[] = Array.from({ length: n }, (_, i) => { const v: P2 = [vectors[0][i], vectors[1][i]]; const nv = Math.hypot(v[0], v[1]) || 1; return [v[0] / nv, v[1] / nv]; });
    const a = kmeans2(emb, [emb.reduce((b, p) => (p[1] < b[1] ? p : b)), emb.reduce((b, p) => (p[1] > b[1] ? p : b))]);
    return { emb, a, lap: values.slice(0, 3) };
  }, [P, sigma]);  
  const km = useMemo(() => kmeans2(P, [P[0], P[1]]), [P]);  
  const a = show === "spectral" ? res.a : km;
  const truth = pts.map((p) => p.label);
  return (
    <LabShell
      viz={
        <div className="grid gap-3 md:grid-cols-[1.3fr_1fr]">
          <Plot x={[-2, 2.6]} y={[-2, 2]} equal aspect={0.8} maxH={360} ariaLabel={`${show} clustering of the ${ds} data`}>
            {P.map((p, i) => <Dot key={i} p={p} r={3.4} color={a[i] ? C.pink : C.blue} />)}
          </Plot>
          <Plot x={[-1.15, 1.15]} y={[-1.15, 1.15]} equal aspect={0.95} maxH={300} xLabel="eigenvector 1 (row-normalised)" yLabel="eigenvector 2" ariaLabel="spectral embedding">
            {res.emb.map((p, i) => <Dot key={i} p={p} r={3} color={res.a[i] ? C.pink : C.blue} opacity={0.8} />)}
          </Plot>
        </div>
      }
      controls={
        <>
          <Segmented label="Data" value={ds} onChange={setDs} options={[{ value: "circles", label: "Rings" }, { value: "moons", label: "Moons" }]} />
          <Segmented label="Colour by" value={show} onChange={setShow} options={[{ value: "spectral", label: "Spectral" }, { value: "kmeans", label: "Plain k-means" }]} />
          <ParameterSlider label="affinity width σ" value={sigma} min={0.05} max={1.5} step={0.01} onChange={setSigma} />
          <p className="text-[12px] leading-relaxed text-faint">Points become nodes of a similarity graph. The bottom eigenvectors of its Laplacian give each point new coordinates (right) in which connected groups are compact, so k-means there finds the rings. σ too large connects everything; too small breaks rings apart.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{agreement (spectral)}", tex: true, value: pct(agree(res.a, truth)), color: C.green },
        { label: "\\text{agreement (k-means)}", tex: true, value: pct(agree(km, truth)) },
        { label: "\\lambda_1,\\lambda_2,\\lambda_3\\ (L_{sym})", tex: true, value: res.lap.map((v) => fmt(Math.max(0, v), 4)).join(", "), hint: "near-zero eigenvalues ↔ nearly disconnected components" },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`W_{ij}=\exp\!\Big(-\frac{\|\mathbf x_i-\mathbf x_j\|^2}{2\cdot${tex(sigma, 2)}^2}\Big),\qquad L_{sym}=I-D^{-1/2}WD^{-1/2}` },
        { tex: String.raw`\text{rows of }[\mathbf u_1\ \mathbf u_2]\ (\text{smallest eigenvectors, normalised})\ \to\ k\text{-means}`, note: "Ng–Jordan–Weiss algorithm" },
      ]} />}
    />
  );
}

/* ------------------------------ mean shift and OPTICS ------------------------------ */
function optics(P: P2[], minPts: number) {
  const n = P.length, D = P.map((a) => P.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1])));
  const core = D.map((r) => r.slice().sort((x, y) => x - y)[minPts - 1]);
  const reach = Array(n).fill(Infinity), done = Array(n).fill(false), order: number[] = [];
  for (let s = 0; s < n; s++) {
    if (done[s]) continue;
    let cur = s;
    while (cur !== -1) {
      done[cur] = true; order.push(cur);
      for (let j = 0; j < n; j++) if (!done[j]) reach[j] = Math.min(reach[j], Math.max(core[cur], D[cur][j]));
      let nxt = -1, best = Infinity;
      for (let j = 0; j < n; j++) if (!done[j] && reach[j] < best) { best = reach[j]; nxt = j; }
      cur = nxt;
    }
  }
  return { order, reach, core };
}
export function DensityClusterViz({ mode: initial = "meanshift" }: { mode?: "meanshift" | "optics" }) {
  const [mode, setMode] = useState<"meanshift" | "optics">(initial);
  const [bw, setBw] = useState(0.45);
  const [eps, setEps] = useState(0.35);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const P = useMemo(() => { const base = makeClusters(3, 35, 0.28, 8); const r = mulberry32(2); return [...base, ...Array.from({ length: 12 }, () => [4 * r() - 2, 4 * r() - 2] as P2)]; }, []);
  // mean shift: move a copy of every point to the kernel-weighted mean of the data, repeatedly
  const traj = useMemo(() => {
    const out: P2[][] = [P.map((p) => [p[0], p[1]] as P2)];
    for (let t = 0; t < 30; t++) {
      out.push(out[t].map((q) => {
        let sx = 0, sy = 0, sw = 0;
        for (const p of P) { const w = Math.exp(-((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) / (2 * bw * bw)); sx += w * p[0]; sy += w * p[1]; sw += w; }
        return [sx / sw, sy / sw] as P2;
      }));
    }
    return out;
  }, [P, bw]);
  useTicker(playing && step < 30, 120, () => setStep((s) => s + 1));
  const cur = traj[Math.min(step, 30)];
  const modes: P2[] = [];
  const msLabel = traj[30].map((q) => { let k = modes.findIndex((m) => Math.hypot(m[0] - q[0], m[1] - q[1]) < bw / 2); if (k < 0) { modes.push(q); k = modes.length - 1; } return k; });
  const op = useMemo(() => optics(P, 5), [P]);
  // extract DBSCAN-like clusters at threshold ε from the OPTICS ordering
  const opLabel = Array(P.length).fill(-1);
  let cid = -1;
  op.order.forEach((i) => { if (op.reach[i] > eps) { if (op.core[i] <= eps) { cid++; opLabel[i] = cid; } } else if (cid >= 0) opLabel[i] = cid; });
  const nOp = cid + 1;
  const rmax = Math.max(...op.order.map((i) => (Number.isFinite(op.reach[i]) ? op.reach[i] : 0))) * 1.05;
  return (
    <LabShell
      viz={mode === "meanshift" ? (
        <div className="space-y-2">
          <Plot x={[-2.4, 2.4]} y={[-2.4, 2.4]} equal aspect={0.75} maxH={440} ariaLabel="mean shift trajectories">
            {P.map((p, i) => <Dot key={"d" + i} p={p} r={2.4} color={C.a("fg", 0.25)} />)}
            {step > 0 && P.map((_, i) => <Polyline key={"t" + i} pts={traj.slice(0, step + 1).map((s) => s[i])} style={{ stroke: C.a("fg", 0.15), strokeWidth: 1 }} />)}
            {cur.map((q, i) => <Dot key={i} p={q} r={3.2} color={step >= 30 ? clsColor(msLabel[i]) : C.blue} />)}
          </Plot>
          <Legend items={[{ label: "data", color: C.a("fg", 0.4) }, { label: "points climbing the density (coloured by mode when converged)", color: C.blue }]} />
        </div>
      ) : (
        <div className="space-y-3">
          <Plot x={[-2.4, 2.4]} y={[-2.4, 2.4]} equal aspect={0.6} maxH={320} ariaLabel="OPTICS clusters at the chosen threshold">
            {P.map((p, i) => <Dot key={i} p={p} r={3.2} color={clsColor(opLabel[i])} />)}
          </Plot>
          <Plot x={[0, P.length]} y={[0, rmax]} aspect={0.28} minH={130} maxH={180} xLabel="points in OPTICS order" yLabel="reachability" ariaLabel="OPTICS reachability plot">
            {op.order.map((i, k) => <Polyline key={k} pts={[[k + 0.5, 0], [k + 0.5, Number.isFinite(op.reach[i]) ? op.reach[i] : rmax]]} style={{ stroke: clsColor(opLabel[i]), strokeWidth: 3 }} />)}
            <Polyline pts={[[0, eps], [P.length, eps]]} style={{ stroke: C.fg, strokeDasharray: "5 3", strokeWidth: 1.2 }} />
          </Plot>
          <Legend items={[{ label: "valleys in the reachability plot = clusters", color: C.blue }, { label: "noise (above ε)", color: C.faint }, { label: "cut ε", color: C.fg, dashed: true }]} />
        </div>
      )}
      controls={
        <>
          <Segmented label="Algorithm" value={mode} onChange={setMode} options={[{ value: "meanshift", label: "Mean shift" }, { value: "optics", label: "OPTICS" }]} />
          {mode === "meanshift" ? <>
            <ParameterSlider label="bandwidth h" value={bw} min={0.15} max={1.2} step={0.01} onChange={(v) => { setBw(v); setStep(0); }} />
            <AnimationControls playing={playing} onPlay={() => setPlaying((p) => !p)} onStep={() => setStep((s) => Math.min(30, s + 1))} onReset={() => { setStep(0); setPlaying(false); }} />
          </> : <ParameterSlider label="cut threshold ε" value={eps} min={0.05} max={1.2} step={0.01} onChange={setEps} />}
          <p className="text-[12px] leading-relaxed text-faint">{mode === "meanshift" ? "Every point repeatedly moves to the weighted average of the data around it, climbing to a peak of the estimated density. Points that reach the same peak form a cluster; the number of clusters follows from the bandwidth." : "OPTICS orders points so that dense neighbourhoods are visited together and records each point's reachability distance (minPts = 5). One run supports any DBSCAN threshold ε: cut the plot and the valleys become clusters."}</p>
        </>
      }
      readouts={<Readouts items={mode === "meanshift" ? [
        { label: "\\text{iteration}", tex: true, value: step },
        { label: "\\text{modes found}", tex: true, value: step >= 30 ? modes.length : "run to 30" },
      ] : [
        { label: "\\text{clusters at }\\varepsilon", tex: true, value: nOp },
        { label: "\\text{noise points}", tex: true, value: opLabel.filter((l) => l < 0).length },
      ]} />}
      calc={<CalcSteps steps={mode === "meanshift" ? [
        { tex: String.raw`\mathbf m(\mathbf x)=\frac{\sum_i K_h(\mathbf x-\mathbf x_i)\,\mathbf x_i}{\sum_i K_h(\mathbf x-\mathbf x_i)},\qquad K_h(\mathbf u)=e^{-\|\mathbf u\|^2/2h^2}`, note: "x ← m(x) is gradient ascent on the kernel density estimate" },
      ] : [
        { tex: String.raw`\text{reach}(p,o)=\max\big(\text{core-dist}(o),\ d(o,p)\big)`, note: "core-dist = distance to the 5th nearest neighbour" },
      ]} />}
    />
  );
}

/* ------------------------------ isolation forest ------------------------------ */
type ITree = { leaf: true; size: number } | { leaf: false; f: 0 | 1; t: number; l: ITree; r: ITree };
const cN = (n: number) => (n <= 1 ? 0 : 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1)) / n);
function iTree(P: P2[], depth: number, maxD: number, r: () => number): ITree {
  if (depth >= maxD || P.length <= 1) return { leaf: true, size: P.length };
  const f = (r() < 0.5 ? 0 : 1) as 0 | 1;
  const lo = Math.min(...P.map((p) => p[f])), hi = Math.max(...P.map((p) => p[f]));
  if (hi - lo < 1e-12) return { leaf: true, size: P.length };
  const t = lo + r() * (hi - lo);
  return { leaf: false, f, t, l: iTree(P.filter((p) => p[f] < t), depth + 1, maxD, r), r: iTree(P.filter((p) => p[f] >= t), depth + 1, maxD, r) };
}
const pathLen = (node: ITree, x: P2, d = 0): number => (node.leaf ? d + cN(node.size) : pathLen(x[node.f] < node.t ? node.l : node.r, x, d + 1));
export function IsolationForestViz() {
  const [nTrees, setNTrees] = useState(100);
  const [contam, setContam] = useState(0.06);
  const th = useThemeRGB();
  const P = useMemo(() => { const base = makeClusters(2, 60, 0.32, 3); const r = mulberry32(14); return [...base, ...Array.from({ length: 8 }, () => [4.4 * r() - 2.2, 4.4 * r() - 2.2] as P2)]; }, []);
  const forest = useMemo(() => {
    const r = mulberry32(7), psi = 64, maxD = Math.ceil(Math.log2(psi));
    return Array.from({ length: nTrees }, () => { const sub = Array.from({ length: psi }, () => P[Math.floor(r() * P.length)]); return iTree(sub, 0, maxD, r); });
  }, [P, nTrees]);
  const score = (x: P2) => Math.pow(2, -forest.reduce((s, t) => s + pathLen(t, x), 0) / forest.length / cN(64));
  const scores = useMemo(() => P.map((p) => score(p)), [forest]); // eslint-disable-line react-hooks/exhaustive-deps
  const k = Math.max(1, Math.round(contam * P.length));
  const thr = scores.slice().sort((a, b) => b - a)[k - 1];
  const most = scores.indexOf(Math.max(...scores)), least = scores.indexOf(Math.min(...scores));
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-2.6, 2.6]} y={[-2.6, 2.6]} equal aspect={0.75} maxH={440} ariaLabel="isolation forest anomaly score map"
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={6} deps={[forest, th]} f={(x, y) => { const s = score([x, y]); return mix(th.panel, th.red, Math.max(0, Math.min(1, (s - 0.35) * 2.2)) * 0.6); }} />}>
            {P.map((p, i) => <Dot key={i} p={p} r={scores[i] >= thr ? 5 : 3} color={scores[i] >= thr ? C.red : C.blue} stroke={scores[i] >= thr ? C.fg : undefined} />)}
          </Plot>
          <Legend items={[{ label: "background: anomaly score (redder = isolated faster)", color: C.red }, { label: `flagged top ${k} points`, color: C.red }, { label: "normal", color: C.blue }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="number of trees" value={nTrees} min={5} max={200} step={5} onChange={setNTrees} />
          <ParameterSlider label="contamination (share flagged)" value={contam} min={0.01} max={0.2} step={0.01} onChange={setContam} format={(v) => pct(v)} />
          <p className="text-[12px] leading-relaxed text-faint">Each tree splits a random feature at a random value until points are alone. Anomalies are few and far away, so they get isolated after very few splits — a short average path means a high anomaly score.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{score of most isolated}", tex: true, value: fmt(scores[most], 3), color: C.red },
        { label: "\\text{score of most typical}", tex: true, value: fmt(scores[least], 3), color: C.blue },
        { label: "c(64)", tex: true, value: fmt(cN(64), 3), hint: "average path length of an unsuccessful search in a BST of 64 points" },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`s(\mathbf x)=2^{-\mathbb E[h(\mathbf x)]/c(\psi)},\quad c(n)=2H(n-1)-\frac{2(n-1)}{n},\ \psi=64`, note: "s near 1 → anomaly, s ≲ 0.5 → normal" }]} />}
    />
  );
}

/* ------------------------------ association rules (Apriori) ------------------------------ */
const BASKETS = [
  ["bread", "milk"], ["bread", "diapers", "beer", "eggs"], ["milk", "diapers", "beer", "cola"], ["bread", "milk", "diapers", "beer"], ["bread", "milk", "diapers", "cola"],
  ["bread", "milk", "eggs"], ["milk", "diapers", "beer"], ["bread", "eggs"], ["bread", "milk", "diapers"], ["diapers", "beer", "cola"],
];
export function AprioriViz() {
  const [minSup, setMinSup] = useState(0.3);
  const [minConf, setMinConf] = useState(0.6);
  const n = BASKETS.length;
  const sup = (s: string[]) => BASKETS.filter((b) => s.every((i) => b.includes(i))).length / n;
  const { levels, rules } = useMemo(() => {
    const items = [...new Set(BASKETS.flat())].sort();
    const levels: { set: string[]; s: number }[][] = [];
    let cur = items.map((i) => [i]).filter((s) => sup(s) >= minSup);

    while (cur.length) {
      levels.push(cur.map((s) => ({ set: s, s: sup(s) })));
      const next: string[][] = [];
      for (let i = 0; i < cur.length; i++) for (let j = i + 1; j < cur.length; j++) {
        const a = cur[i], b = cur[j];
        if (a.slice(0, -1).join() !== b.slice(0, -1).join()) continue; // join step: same prefix
        const cand = [...a, b[b.length - 1]].sort();
        // prune step: every subset must already be frequent (downward closure)
        const ok = cand.every((_, k) => cur.some((c) => c.join() === cand.filter((__, m) => m !== k).join()));
        if (!ok) continue;
        if (sup(cand) >= minSup) next.push(cand);
      }
      cur = next;
    }
    const rules: { a: string[]; c: string[]; s: number; conf: number; lift: number }[] = [];
    levels.slice(1).flat().forEach(({ set, s }) => {
      for (let mask = 1; mask < (1 << set.length) - 1; mask++) {
        const a = set.filter((_, k) => mask & (1 << k)), c = set.filter((_, k) => !(mask & (1 << k)));
        const conf = s / sup(a);
        if (conf >= minConf) rules.push({ a, c, s, conf, lift: conf / sup(c) });
      }
    });
    rules.sort((x, y) => y.lift - x.lift);
    return { levels, rules };
  }, [minSup, minConf]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1.5 text-[11.5px]">{BASKETS.map((b, i) => <span key={i} className="rounded-md border border-line bg-raised px-2 py-1 text-muted">#{i + 1} {b.join(", ")}</span>)}</div>
          <div className="grid gap-3 lg:grid-cols-2">
            <div className="space-y-1.5">
              <div className="text-[12px] text-faint">Frequent itemsets (support ≥ {pct(minSup)})</div>
              <MiniTable head={["size", "itemset", "support"]} rows={levels.flatMap((L, k) => L.map((x) => [k + 1, `{${x.set.join(", ")}}`, pct(x.s)]))} />
            </div>
            <div className="space-y-1.5">
              <div className="text-[12px] text-faint">Rules (confidence ≥ {pct(minConf)}), sorted by lift</div>
              <MiniTable head={["rule", "support", "confidence", "lift"]} rows={rules.slice(0, 10).map((r) => [`{${r.a.join(", ")}} → {${r.c.join(", ")}}`, pct(r.s), pct(r.conf), fmt(r.lift, 2)])} rowClass={(i) => (rules[i] && rules[i].lift > 1 ? undefined : "opacity-60")} />
            </div>
          </div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="minimum support" value={minSup} min={0.1} max={0.8} step={0.1} onChange={setMinSup} format={(v) => pct(v)} />
          <ParameterSlider label="minimum confidence" value={minConf} min={0.3} max={1} step={0.05} onChange={setMinConf} format={(v) => pct(v)} />
          <p className="text-[12px] leading-relaxed text-faint">Apriori grows itemsets one item at a time and discards any candidate with an infrequent subset — if {"{beer}"} is rare, {"{beer, diapers}"} must be rarer. Lift &gt; 1 means the items appear together more often than if they were independent.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{frequent itemsets}", tex: true, value: levels.flat().length },
        { label: "\\text{rules}", tex: true, value: rules.length },
        { label: "\\text{baskets}", tex: true, value: n },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\text{supp}(X)=\frac{|\{t:X\subseteq t\}|}{N},\quad \text{conf}(X\Rightarrow Y)=\frac{\text{supp}(X\cup Y)}{\text{supp}(X)},\quad \text{lift}=\frac{\text{conf}}{\text{supp}(Y)}` },
        ...(rules[0] ? [{ tex: String.raw`\text{top rule: conf}=\frac{${fmt(rules[0].s, 2)}}{${fmt(rules[0].s / rules[0].conf, 2)}}=${tex(rules[0].conf, 3)},\ \text{lift}=${tex(rules[0].lift, 3)}` }] : []),
      ]} />}
    />
  );
}

/* ------------------------------ collaborative filtering (matrix factorisation) ------------------------------ */
const USERS = ["Ada", "Ben", "Cy", "Dee", "Eli", "Fay"];
const MOVIES = ["Alien", "Blade", "Coco", "Dumbo", "Edge", "Frozen"];
// 0 = unknown. Two taste groups: sci-fi (Alien, Blade, Edge) and animation (Coco, Dumbo, Frozen)
const RATINGS = [
  [5, 4, 1, 0, 5, 1],
  [4, 0, 1, 2, 4, 0],
  [1, 2, 5, 4, 0, 5],
  [0, 1, 4, 5, 2, 4],
  [5, 5, 0, 1, 0, 2],
  [2, 0, 5, 0, 1, 5],
];
const HELD: [number, number][] = [[0, 4], [2, 5], [4, 1]]; // known ratings hidden from training
export function MatrixFactorViz() {
  const [k, setK] = useState(2);
  const [lam, setLam] = useState(0.1);
  const res = useMemo(() => {
    const train = RATINGS.map((r, i) => r.map((v, j) => (HELD.some(([a, b]) => a === i && b === j) ? 0 : v)));
    const r = mulberry32(5);
    let U = USERS.map(() => Array.from({ length: k }, () => 0.5 + 0.3 * r())), V = MOVIES.map(() => Array.from({ length: k }, () => 0.5 + 0.3 * r()));
    // alternating least squares: solve for each user with V fixed, then each item with U fixed
    const solveRow = (ys: number[], rows: number[][]) => {
      const M = Array.from({ length: k }, (_, a) => Array.from({ length: k }, (_, b) => rows.reduce((s, x) => s + x[a] * x[b], 0) + (a === b ? lam : 0)));
      const rhs = Array.from({ length: k }, (_, a) => rows.reduce((s, x, i) => s + x[a] * ys[i], 0));
      // Gaussian elimination (k ≤ 4)
      const n = k; const Mx = M.map((row, i) => [...row, rhs[i]]);
      for (let c = 0; c < n; c++) { let p = c; for (let q = c + 1; q < n; q++) if (Math.abs(Mx[q][c]) > Math.abs(Mx[p][c])) p = q; [Mx[c], Mx[p]] = [Mx[p], Mx[c]]; for (let q = c + 1; q < n; q++) { const m = Mx[q][c] / Mx[c][c]; for (let z = c; z <= n; z++) Mx[q][z] -= m * Mx[c][z]; } }
      const x = Array(n).fill(0); for (let c = n - 1; c >= 0; c--) x[c] = (Mx[c][n] - x.reduce((s, v, z) => s + (z > c ? Mx[c][z] * v : 0), 0)) / Mx[c][c];
      return x;
    };
    for (let it = 0; it < 30; it++) {
      U = U.map((_, i) => { const js = MOVIES.map((__, j) => j).filter((j) => train[i][j] > 0); return solveRow(js.map((j) => train[i][j]), js.map((j) => V[j])); });
      V = V.map((_, j) => { const is = USERS.map((__, i) => i).filter((i) => train[i][j] > 0); return solveRow(is.map((i) => train[i][j]), is.map((i) => U[i])); });
    }
    const pred = USERS.map((_, i) => MOVIES.map((__, j) => U[i].reduce((s, u, z) => s + u * V[j][z], 0)));
    const obs = train.flatMap((row, i) => row.map((v, j) => [i, j, v]).filter((t) => t[2] > 0));
    const rmse = (pairs: number[][]) => Math.sqrt(pairs.reduce((s, [i, j, v]) => s + (pred[i][j] - v) ** 2, 0) / pairs.length);
    return { pred, train, trainRmse: rmse(obs), heldRmse: rmse(HELD.map(([i, j]) => [i, j, RATINGS[i][j]])) };
  }, [k, lam]);
  return (
    <LabShell
      viz={
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-1.5"><div className="text-[12px] text-faint">Observed ratings R (? = unknown, ringed = hidden for testing)</div><Heat M={res.train.map((r) => r.map((v) => (v > 0 ? v : NaN)))} rowLabels={USERS} colLabels={MOVIES} digits={0} max={5} highlight={(i, j) => HELD.some(([a, b]) => a === i && b === j)} /></div>
          <div className="space-y-1.5"><div className="text-[12px] text-faint">Predicted ratings U Vᵀ (rank {k})</div><Heat M={res.pred} rowLabels={USERS} colLabels={MOVIES} digits={1} max={5} highlight={(i, j) => HELD.some(([a, b]) => a === i && b === j)} /></div>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="latent factors k" value={k} min={1} max={4} step={1} onChange={setK} />
          <ParameterSlider label="regularisation λ" value={lam} min={0} max={3} step={0.05} onChange={setLam} />
          <p className="text-[12px] leading-relaxed text-faint">Each user and each movie gets a k-dimensional vector; a rating is predicted by their dot product. The factors are learned only from the known cells (alternating least squares) and then fill in the unknown ones. Compare the ringed predictions with the true hidden ratings: 5, 5, 5.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{train RMSE}", tex: true, value: fmt(res.trainRmse, 3) },
        { label: "\\text{held-out RMSE}", tex: true, value: fmt(res.heldRmse, 3), color: C.green },
        { label: "\\text{parameters}", tex: true, value: `${(USERS.length + MOVIES.length) * k} vs 36 cells` },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\min_{U,V}\sum_{(i,j)\in\Omega}(r_{ij}-\mathbf u_i^\top\mathbf v_j)^2+\lambda\big(\|\mathbf u_i\|^2+\|\mathbf v_j\|^2\big)`, note: "Ω = observed cells only" },
        { tex: String.raw`\mathbf u_i\leftarrow\Big(\sum_{j\in\Omega_i}\mathbf v_j\mathbf v_j^\top+\lambda I\Big)^{-1}\sum_{j\in\Omega_i}r_{ij}\mathbf v_j`, note: "one ALS half-step; then the same for every v_j" },
      ]} />}
    />
  );
}

/* ------------------------------ Gaussian processes ------------------------------ */
export function cholesky(A: number[][]) {
  const n = A.length, L = A.map(() => Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j];
    for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
  }
  return L;
}
export const cholSolve = (L: number[][], b: number[]) => {
  const n = L.length, y = Array(n).fill(0), x = Array(n).fill(0);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
  return x;
};
export function GPViz() {
  const [ell, setEll] = useState(0.8);
  const [sf, setSf] = useState(1);
  const [noise, setNoise] = useState(0.1);
  const [samples, setSamples] = useState(true);
  const [pts, setPts] = useState<P2[]>([[-3, -0.6], [-1.6, 0.9], [-0.4, 0.2], [1.2, -0.8], [2.6, 0.5]]);
  const k = (a: number, b: number) => sf * sf * Math.exp(-((a - b) ** 2) / (2 * ell * ell));
  const res = useMemo(() => {
    const X = pts.map((p) => p[0]), y = pts.map((p) => p[1]);
    const K = X.map((a, i) => X.map((b, j) => k(a, b) + (i === j ? noise * noise : 0)));
    const L = cholesky(K), alpha = cholSolve(L, y);
    const grid = Array.from({ length: 81 }, (_, i) => -4 + (8 * i) / 80);
    const mean = grid.map((g) => X.reduce((s, x, i) => s + k(g, x) * alpha[i], 0));
    const vr = grid.map((g) => { const ks = X.map((x) => k(g, x)); const v = cholSolve(L, ks); return Math.max(0, k(g, g) - ks.reduce((s, q, i) => s + q * v[i], 0)); });
    // posterior samples on the grid: mean + chol(Σ_post) z
    const Ks = grid.map((g) => X.map((x) => k(g, x)));
    const Vs = Ks.map((ks) => cholSolve(L, ks));
    const Sp = grid.map((a, i) => grid.map((b, j) => k(a, b) - Ks[i].reduce((s, q, m) => s + q * Vs[j][m], 0) + (i === j ? 1e-6 : 0)));
    const Lp = cholesky(Sp);
    const r = mulberry32(3);
    const draws = [0, 1, 2].map(() => { const z = grid.map(() => randn(r)); return grid.map((_, i) => mean[i] + Lp[i].reduce((s, l, m) => s + l * z[m], 0)); });
    const lml = -0.5 * y.reduce((s, v, i) => s + v * alpha[i], 0) - L.reduce((s, row, i) => s + Math.log(row[i]), 0) - (X.length / 2) * Math.log(2 * Math.PI);
    return { grid, mean, sd: vr.map(Math.sqrt), draws, lml };
  }, [pts, ell, sf, noise]); // eslint-disable-line react-hooks/exhaustive-deps
  const band: [number, number][] = [...res.grid.map((g, i) => [g, res.mean[i] + 2 * res.sd[i]] as [number, number]), ...res.grid.map((g, i) => [g, res.mean[i] - 2 * res.sd[i]] as [number, number]).reverse()];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-4, 4]} y={[-3, 3]} aspect={0.5} maxH={380} ariaLabel="Gaussian process posterior mean and uncertainty" onPointerDown={(p) => setPts((ps) => (ps.length < 15 ? [...ps, p] : ps))}>
            <Polyline pts={band} closed style={{ fill: C.a("blue", 0.14), stroke: "none" }} />
            {samples && res.draws.map((d, i) => <Polyline key={i} pts={res.grid.map((g, j) => [g, d[j]])} style={{ stroke: C.a("pink", 0.55), strokeWidth: 1 }} />)}
            <Polyline pts={res.grid.map((g, i) => [g, res.mean[i]])} style={{ stroke: C.blue, strokeWidth: 2.4 }} />
            {pts.map((p, i) => <Dot key={i} p={p} r={5} color={C.fg} stroke={C.bg} />)}
          </Plot>
          <Legend items={[{ label: "posterior mean", color: C.blue }, { label: "±2 standard deviations", color: C.a("blue", 0.4) }, ...(samples ? [{ label: "functions sampled from the posterior", color: C.pink }] : [])]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="length-scale ℓ" value={ell} min={0.1} max={3} step={0.05} onChange={setEll} />
          <ParameterSlider label="signal std σ_f" value={sf} min={0.2} max={2} step={0.05} onChange={setSf} />
          <ParameterSlider label="noise std σ_n" value={noise} min={0.01} max={1} step={0.01} onChange={setNoise} />
          <Toggle label="Show posterior samples" checked={samples} onChange={setSamples} />
          <div className="flex gap-2"><button className="btn" onClick={() => setPts([])}>Clear points</button></div>
          <p className="text-[12px] leading-relaxed text-faint">Click the plot to add observations. Uncertainty shrinks near data and grows back to the prior away from it. ℓ controls how quickly functions can wiggle.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "n", tex: true, value: pts.length },
        { label: "\\log p(\\mathbf y\\mid X)", tex: true, value: pts.length ? fmt(res.lml, 3) : "—", hint: "log marginal likelihood — maximise it to choose ℓ, σ_f, σ_n" },
        { label: "\\max\\sigma(x)", tex: true, value: fmt(Math.max(...res.sd), 3) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`k(x,x')=${tex(sf * sf, 2)}\exp\!\Big(-\frac{(x-x')^2}{2\cdot${tex(ell, 2)}^2}\Big)` },
        { tex: String.raw`\mu(x_*)=\mathbf k_*^\top(K+\sigma_n^2I)^{-1}\mathbf y,\qquad \sigma^2(x_*)=k(x_*,x_*)-\mathbf k_*^\top(K+\sigma_n^2I)^{-1}\mathbf k_*`, note: "solved with a Cholesky factorisation, never an explicit inverse" },
      ]} />}
    />
  );
}

/* ------------------------------ hyperparameter search ------------------------------ */
// Synthetic validation-accuracy surface over (log10 learning rate, log10 regularisation)
const score2 = (a: number, b: number) => 0.62 + 0.3 * Math.exp(-(((a + 2.2) / 0.55) ** 2) - (((b + 3) / 1.1) ** 2) - 0.5 * (a + 2.2) * (b + 3)) + 0.04 * Math.exp(-(((a + 0.7) / 0.3) ** 2) - (((b + 0.8) / 0.4) ** 2));
const A_R: [number, number] = [-5, 0], B_R: [number, number] = [-6, 0];
function bayesOpt(budget: number) {
  const pts: { a: number; b: number; y: number }[] = [];
  const r = mulberry32(4);
  for (let i = 0; i < Math.min(3, budget); i++) { const a = A_R[0] + 5 * r(), b = B_R[0] + 6 * r(); pts.push({ a, b, y: score2(a, b) }); }
  const ell = [0.8, 1.0], sf2 = 0.01, sn2 = 1e-6;
  const k = (p: { a: number; b: number }, q: { a: number; b: number }) => sf2 * Math.exp(-((p.a - q.a) ** 2) / (2 * ell[0] ** 2) - ((p.b - q.b) ** 2) / (2 * ell[1] ** 2));
  while (pts.length < budget) {
    const ym = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    const K = pts.map((p, i) => pts.map((q, j) => k(p, q) + (i === j ? sn2 : 0)));
    const L = cholesky(K), alpha = cholSolve(L, pts.map((p) => p.y - ym));
    const best = Math.max(...pts.map((p) => p.y));
    let arg = { a: 0, b: 0, ei: -1 };
    for (let i = 0; i <= 30; i++) for (let j = 0; j <= 30; j++) {
      const c = { a: A_R[0] + (5 * i) / 30, b: B_R[0] + (6 * j) / 30 };
      const ks = pts.map((p) => k(c, p));
      const mu = ym + ks.reduce((s, v, m) => s + v * alpha[m], 0);
      const v = cholSolve(L, ks);
      const sd = Math.sqrt(Math.max(1e-12, sf2 - ks.reduce((s, q, m) => s + q * v[m], 0)));
      const z = (mu - best) / sd;
      const pdf = Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI), cdf = 0.5 * (1 + erf(z / Math.SQRT2));
      const ei = (mu - best) * cdf + sd * pdf; // expected improvement
      if (ei > arg.ei) arg = { ...c, ei };
    }
    pts.push({ a: arg.a, b: arg.b, y: score2(arg.a, arg.b) });
  }
  return pts;
}
function erf(x: number) { // Abramowitz–Stegun 7.1.26
  const s = Math.sign(x), t = 1 / (1 + 0.3275911 * Math.abs(x));
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
}
export function HyperSearchViz() {
  const [method, setMethod] = useState<"grid" | "random" | "bayes">("random");
  const [budget, setBudget] = useState(16);
  const th = useThemeRGB();
  const runs = useMemo(() => {
    const g = Math.max(1, Math.floor(Math.sqrt(budget)));
    const grid = Array.from({ length: g * g }, (_, i) => { const a = A_R[0] + (5 * ((i % g) + 0.5)) / g, b = B_R[0] + (6 * (Math.floor(i / g) + 0.5)) / g; return { a, b, y: score2(a, b) }; });
    const r = mulberry32(11);
    const rnd = Array.from({ length: budget }, () => { const a = A_R[0] + 5 * r(), b = B_R[0] + 6 * r(); return { a, b, y: score2(a, b) }; });
    return { grid, random: rnd, bayes: bayesOpt(budget) };
  }, [budget]);
  const cur = runs[method];
  const bestOf = (p: { y: number }[]) => Math.max(...p.map((q) => q.y));
  const bestPt = cur.reduce((b, p) => (p.y > b.y ? p : b));
  const curve = (p: { y: number }[]) => p.map((_, i) => [i + 1, Math.max(...p.slice(0, i + 1).map((q) => q.y))] as [number, number]);
  return (
    <LabShell
      viz={
        <div className="grid gap-3 lg:grid-cols-[1.3fr_1fr]">
          <Plot x={A_R} y={B_R} aspect={0.85} maxH={380} xLabel="log₁₀ learning rate" yLabel="log₁₀ regularisation" ariaLabel="hyperparameter search points over the validation score surface"
            underlay={(ctx) => <HeatmapCanvas ctx={ctx} res={5} deps={[th]} f={(a, b) => mix(th.panel, th.green, Math.max(0, (score2(a, b) - 0.6) / 0.33) * 0.7)} />}>
            {cur.map((p, i) => <Dot key={i} p={[p.a, p.b]} r={p === bestPt ? 6.5 : 4} color={p === bestPt ? C.pink : C.fg} stroke={C.bg} />)}
          </Plot>
          <Plot x={[1, budget]} y={[0.6, 0.95]} aspect={0.85} maxH={300} xLabel="evaluations" yLabel="best validation score so far" ariaLabel="best score found against number of evaluations">
            <Polyline pts={curve(runs.grid)} style={{ stroke: C.amber, strokeWidth: 1.6 }} />
            <Polyline pts={curve(runs.random)} style={{ stroke: C.blue, strokeWidth: 1.6 }} />
            <Polyline pts={curve(runs.bayes)} style={{ stroke: C.pink, strokeWidth: 2 }} />
          </Plot>
          <div className="lg:col-span-2"><Legend items={[{ label: "grid", color: C.amber }, { label: "random", color: C.blue }, { label: "Bayesian (GP + expected improvement)", color: C.pink }, { label: "background: synthetic validation score (greener = better)", color: C.green }]} /></div>
        </div>
      }
      controls={
        <>
          <Segmented label="Search strategy" value={method} onChange={setMethod} options={[{ value: "grid", label: "Grid" }, { value: "random", label: "Random" }, { value: "bayes", label: "Bayesian" }]} />
          <ParameterSlider label="evaluation budget" value={budget} min={4} max={36} step={1} onChange={setBudget} />
          <p className="text-[12px] leading-relaxed text-faint">The good region is a narrow diagonal ridge. A grid wastes evaluations on repeated values of each axis; random search tries a new value of every hyperparameter each time; Bayesian optimisation fits a Gaussian process to past results and picks the point with the highest expected improvement.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{best (grid)}", tex: true, value: fmt(bestOf(runs.grid), 4), color: C.amber },
        { label: "\\text{best (random)}", tex: true, value: fmt(bestOf(runs.random), 4), color: C.blue },
        { label: "\\text{best (Bayesian)}", tex: true, value: fmt(bestOf(runs.bayes), 4), color: C.pink },
        { label: "\\text{optimum of surface}", tex: true, value: fmt(score2(-2.2, -3), 4) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\mathrm{EI}(\mathbf x)=(\mu-f^+)\,\Phi(z)+\sigma\,\varphi(z),\quad z=\frac{\mu(\mathbf x)-f^+}{\sigma(\mathbf x)}`, note: "μ, σ from the GP posterior; f⁺ = best score so far" },
        { tex: String.raw`\text{grid with budget }${budget}\text{ tests only }${Math.floor(Math.sqrt(budget))}\text{ distinct values per axis}` },
      ]} />}
    />
  );
}
