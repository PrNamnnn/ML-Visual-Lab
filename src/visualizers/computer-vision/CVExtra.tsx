"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { LabShell, Readouts, CalcSteps } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, Select, Toggle } from "@/components/controls";
import { C } from "@/components/visualization/Plot";
import { mulberry32, randn } from "@/lib/math/random";
import { fmt, pct, tex } from "@/lib/utilities/format";
import { BarList, Heat, MiniTable } from "@/visualizers/shared";

type RGB = [number, number, number];
const N = 48;

/** Small canvas that renders a grey (0..1), signed, label or RGB image, with optional box / grid overlay. */
export function ImgCanvas({ data, kind = "gray", label, size = 192, boxes, grid, onPick, highlight }: {
  data: number[][] | RGB[][]; kind?: "gray" | "signed" | "rgb" | "labels"; label: string; size?: number;
  boxes?: { x: number; y: number; w: number; h: number; color: string; dashed?: boolean; text?: string }[]; grid?: number;
  onPick?: (i: number, j: number) => void; highlight?: number[][];
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const H = data.length, W = data[0].length;
  useEffect(() => {
    const g = ref.current?.getContext("2d");
    if (!g) return;
    const im = g.createImageData(W, H);
    const pal: RGB[] = [[24, 24, 27], [96, 165, 250], [244, 114, 182], [74, 222, 128], [251, 191, 36], [248, 113, 113], [167, 139, 250], [45, 212, 191], [253, 186, 116], [161, 161, 170]];
    const mx = kind === "signed" ? Math.max(1e-9, ...(data as number[][]).flat().map(Math.abs)) : 1;
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const k = (i * W + j) * 4;
      let c: RGB;
      if (kind === "rgb") c = (data as RGB[][])[i][j];
      else if (kind === "labels") { const v = (data as number[][])[i][j]; c = v <= 0 ? pal[0] : pal[((v - 1) % (pal.length - 1)) + 1]; }
      else if (kind === "signed") { const t = (data as number[][])[i][j] / mx; c = [t < 0 ? 255 * -t : 0, 30 * Math.abs(t), t > 0 ? 255 * t : 0]; }
      else { const t = Math.max(0, Math.min(1, (data as number[][])[i][j])) * 255; c = [t, t, t]; }
      const hl = highlight ? highlight[i][j] : 0;
      im.data[k] = c[0] * (1 - hl) + 255 * hl; im.data[k + 1] = c[1] * (1 - hl) + 80 * hl; im.data[k + 2] = c[2] * (1 - hl) + 80 * hl; im.data[k + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }, [data, kind, W, H, highlight]);
  const s = size / W;
  return (
    <figure className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: size, height: (size * H) / W }}>
        <canvas ref={ref} width={W} height={H} role="img" aria-label={label} className="absolute inset-0 h-full w-full rounded border border-line" style={{ imageRendering: "pixelated", cursor: onPick ? "crosshair" : undefined }}
          onClick={onPick ? (e) => { const r = (e.target as HTMLCanvasElement).getBoundingClientRect(); onPick(Math.floor(((e.clientY - r.top) / r.height) * H), Math.floor(((e.clientX - r.left) / r.width) * W)); } : undefined} />
        <svg className="pointer-events-none absolute inset-0" width={size} height={(size * H) / W}>
          {grid && Array.from({ length: Math.floor(W / grid) + 1 }, (_, k) => <line key={"v" + k} x1={k * grid * s} x2={k * grid * s} y1={0} y2={H * s} stroke="rgb(var(--fg) / 0.35)" strokeWidth={1} />)}
          {grid && Array.from({ length: Math.floor(H / grid) + 1 }, (_, k) => <line key={"h" + k} y1={k * grid * s} y2={k * grid * s} x1={0} x2={W * s} stroke="rgb(var(--fg) / 0.35)" strokeWidth={1} />)}
          {boxes?.map((b, k) => <g key={k}><rect x={b.x * s} y={b.y * s} width={b.w * s} height={b.h * s} fill="none" stroke={b.color} strokeWidth={2} strokeDasharray={b.dashed ? "4 3" : undefined} />{b.text && <text x={b.x * s + 3} y={b.y * s + 12} fontSize={10.5} fill={b.color}>{b.text}</text>}</g>)}
        </svg>
      </div>
      <figcaption className="text-[12px] text-muted">{label}</figcaption>
    </figure>
  );
}

const conv = (G: number[][], K: number[][]) => {
  const k = K.length, h = (k - 1) / 2, H = G.length, W = G[0].length;
  return G.map((r, i) => r.map((_, j) => { let s = 0; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) s += G[Math.min(H - 1, Math.max(0, i + a - h))][Math.min(W - 1, Math.max(0, j + b - h))] * K[a][b]; return s; }));
};
const relu = (M: number[][]) => M.map((r) => r.map((v) => Math.max(0, v)));
const pool2 = (M: number[][]) => Array.from({ length: Math.floor(M.length / 2) }, (_, i) => Array.from({ length: Math.floor(M[0].length / 2) }, (_, j) => Math.max(M[2 * i][2 * j], M[2 * i + 1][2 * j], M[2 * i][2 * j + 1], M[2 * i + 1][2 * j + 1])));

/* ------------------------------ CNN feature hierarchy ------------------------------ */
type Shape = "square" | "circle" | "cross" | "stripes";
function shapeImage(shape: Shape, noise: number) {
  const r = mulberry32(5);
  return Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => {
    const x = j - 24, y = i - 24;
    let v = 0;
    if (shape === "square") v = Math.abs(x) < 12 && Math.abs(y) < 12 ? 1 : 0;
    else if (shape === "circle") v = x * x + y * y < 14 * 14 ? 1 : 0;
    else if (shape === "cross") v = (Math.abs(x) < 4 && Math.abs(y) < 16) || (Math.abs(y) < 4 && Math.abs(x) < 16) ? 1 : 0;
    else v = Math.abs(x) < 16 && Math.abs(y) < 16 && Math.floor((j + 100) / 4) % 2 === 0 ? 1 : 0;
    return Math.max(0, Math.min(1, v * 0.8 + 0.1 + noise * randn(r)));
  }));
}
const K_V = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], K_H = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
export function FeatureHierarchyViz() {
  const [shape, setShape] = useState<Shape>("square");
  const [noise, setNoise] = useState(0.05);
  const feats = useMemo(() => {
    const img = shapeImage(shape, noise);
    // layer 1: oriented edge detectors (both polarities via |·|), ReLU, then 2×2 max-pool
    // layer 1: Sobel gradients, then four orientation-tuned edge maps (like oriented filters in a trained first layer)
    const gx = conv(img, K_V), gy = conv(img, K_H);
    const tuned = (th: number) => gx.map((r, i) => r.map((v, j) => { const m = Math.hypot(v, gy[i][j]), a = Math.atan2(gy[i][j], v); return m * Math.max(0, Math.cos(2 * (a - th))) ** 4; }));
    const L1 = [0, Math.PI / 2, Math.PI / 4, -Math.PI / 4].map((th) => pool2(relu(tuned(th))));
    // layer 2: combine nearby layer-1 responses — corner = vertical AND horizontal; curve = slanted edges not explained by straight ones
    const blur = [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => v / 9));
    const bl = L1.map((M) => conv(M, blur));
    const corner = pool2(bl[0].map((r, i) => r.map((v, j) => Math.sqrt(v * bl[1][i][j]))));
    const curve = pool2(bl[2].map((r, i) => r.map((v, j) => Math.max(0, Math.max(v, bl[3][i][j]) - 0.5 * Math.max(bl[0][i][j], bl[1][i][j])))));
    const straight = pool2(bl[0].map((r, i) => r.map((v, j) => Math.max(0, Math.max(v, bl[1][i][j]) - Math.max(bl[2][i][j], bl[3][i][j])))));
    // layer 3: global average pooling → one number per feature map
    const gap = (M: number[][]) => M.flat().reduce((s, v) => s + v, 0) / M.flat().length;
    const vec = [...L1.map(gap), gap(corner), gap(curve), gap(straight)];
    return { img, L1, L2: [corner, curve, straight], vec };
  }, [shape, noise]);
  const names = ["L1 vertical edge", "L1 horizontal edge", "L1 diagonal ↗", "L1 diagonal ↘", "L2 corner", "L2 curve", "L2 straight run"];
  const mv = Math.max(...feats.vec);
  const l1max = Math.max(1e-9, ...feats.L1.flatMap((M) => M.flat())), l2max = Math.max(1e-9, ...feats.L2.flatMap((M) => M.flat()));
  return (
    <LabShell
      wideControls
      viz={
        <div className="space-y-3">
          <div className="flex flex-wrap items-end gap-3">
            <ImgCanvas data={feats.img} label={`input ${N}×${N}`} size={150} />
            {feats.L1.map((M, k) => <ImgCanvas key={k} data={M.map((r) => r.map((v) => v / l1max))} label={names[k]} size={100} />)}
          </div>
          <div className="flex flex-wrap items-end gap-3">
            {feats.L2.map((M, k) => <ImgCanvas key={k} data={M.map((r) => r.map((v) => v / l2max))} label={names[4 + k]} size={100} />)}
          </div>
          <div className="space-y-1"><div className="text-[12px] text-faint">Layer 3: global average pooling → feature vector (normalised)</div><BarList items={feats.vec.map((v, k) => ({ label: names[k], value: v / mv, color: k < 4 ? C.blue : C.pink }))} max={1} /></div>
        </div>
      }
      controls={
        <>
          <Select label="Input image" value={shape} onChange={setShape} options={[{ value: "square", label: "Square" }, { value: "circle", label: "Circle" }, { value: "cross", label: "Cross" }, { value: "stripes", label: "Stripes" }]} />
          <ParameterSlider label="pixel noise" value={noise} min={0} max={0.3} step={0.01} onChange={setNoise} />
          <p className="text-[12px] leading-relaxed text-faint">Layer 1 responds to edges of one orientation. Layer 2 combines nearby layer-1 responses into parts: corners (vertical and horizontal edges together), curves (slanted edges) and straight runs. Each pooling halves the resolution, so later maps see larger regions. Different shapes produce different final vectors — that vector is what a classifier reads. (Hand-set filters that mimic what trained CNNs learn.)</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{resolution L1 / L2}", tex: true, value: `${N / 2}² / ${N / 4}²` },
        { label: "\\text{receptive field (L2)}", tex: true, value: "10×10 px" },
        { label: "\\text{strongest part}", tex: true, value: names[4 + feats.vec.slice(4).indexOf(Math.max(...feats.vec.slice(4)))] },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`F^{(1)}_k=\operatorname{pool}\big(\|\nabla I\|\cos^4(2(\theta-\theta_k))\big),\quad \nabla I\text{ from }K_{\text{vert}}=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}`, note: "θ = gradient direction; θ_k = 0°, 90°, ±45°" },
        { tex: String.raw`F^{(2)}_{\text{corner}}=\operatorname{pool}\Big(\sqrt{\bar F^{(1)}_{\text{vert}}\cdot\bar F^{(1)}_{\text{horiz}}}\Big),\qquad f_m=\operatorname{mean}_{i,j}F_m(i,j)` },
      ]} />}
    />
  );
}

/* ------------------------------ object detection: two-stage vs YOLO ------------------------------ */
type Box = { x: number; y: number; w: number; h: number };
const OBJ: (Box & { cls: string; color: RGB })[] = [
  { x: 4, y: 6, w: 14, h: 18, cls: "person", color: [244, 114, 182] },
  { x: 26, y: 24, w: 18, h: 12, cls: "car", color: [96, 165, 250] },
  { x: 22, y: 4, w: 9, h: 9, cls: "ball", color: [251, 191, 36] },
];
const iouB = (a: Box, b: Box) => { const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)), iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)); const I = ix * iy; return I / (a.w * a.h + b.w * b.h - I); };
function sceneImage(): RGB[][] {
  const r = mulberry32(2);
  return Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => {
    const o = OBJ.find((b) => j >= b.x && j < b.x + b.w && i >= b.y && i < b.y + b.h);
    const base: RGB = o ? o.color : [40 + 10 * r(), 44 + 10 * r(), 52 + 10 * r()];
    return base.map((v) => Math.max(0, Math.min(255, v + 12 * randn(r)))) as RGB;
  }));
}
export function DetectionViz({ mode: initial = "two-stage" }: { mode?: "two-stage" | "yolo" }) {
  const [mode, setMode] = useState<"two-stage" | "yolo">(initial);
  const [stage, setStage] = useState<"proposals" | "scored" | "refined" | "nms">("nms");
  const [S, setS] = useState(4);
  const [thr, setThr] = useState(0.5);
  const img = useMemo(sceneImage, []);
  const fg = useMemo(() => img.map((r) => r.map(([R, G, B]) => (Math.max(R, G, B) > 90 ? 1 : 0))), [img]);
  const det = useMemo(() => {
    // stage 1: anchor / sliding-window proposals at 3 scales × 3 aspect ratios
    const props: Box[] = [];
    for (const sz of [8, 13, 19]) for (const ar of [0.6, 1, 1.6]) { const w = Math.round(sz * Math.sqrt(ar)), h = Math.round(sz / Math.sqrt(ar)); for (let y = 0; y + h <= N; y += 4) for (let x = 0; x + w <= N; x += 4) props.push({ x, y, w, h }); }
    // stage 2: score each proposal from the pixels it contains (a stand-in for the per-region classifier)
    const scored = props.map((b) => {
      let inside = 0; const counts = OBJ.map(() => 0);
      for (let i = b.y; i < b.y + b.h; i++) for (let j = b.x; j < b.x + b.w; j++) if (fg[i][j]) { inside++; const k = OBJ.findIndex((o) => j >= o.x && j < o.x + o.w && i >= o.y && i < o.y + o.h); if (k >= 0) counts[k]++; }
      const k = counts.indexOf(Math.max(...counts)), o = OBJ[k];
      const precision = inside / (b.w * b.h), recall = counts[k] / (o.w * o.h);
      return { ...b, s: precision * recall, k };
    }).filter((b) => b.s > 0.25);
    // box regression stand-in: an imperfect regressor that moves each proposal 80 % of the way to the extent of
    // the object it overlaps most (a trained regressor predicts these offsets t_x, t_y, t_w, t_h from features)
    const refined = scored.map((b) => {
      const o = OBJ[b.k], k = 0.8;
      return { ...b, x: b.x + k * (o.x - b.x), y: b.y + k * (o.y - b.y), w: b.w + k * (o.w - b.w), h: b.h + k * (o.h - b.h) };
    });
    const order = refined.map((_, i) => i).sort((a, b) => refined[b].s - refined[a].s), kept: number[] = [];
    for (const i of order) if (!kept.some((k) => iouB(refined[k], refined[i]) > thr)) kept.push(i);
    return { props, scored, refined, kept: kept.map((i) => refined[i]) };
  }, [fg, thr]);
  const col = (k: number) => ["rgb(var(--pink))", "rgb(var(--blue))", "rgb(var(--amber))"][k];
  // YOLO: the cell containing each object's centre is responsible for predicting it
  const cell = N / S;
  const resp = OBJ.map((o) => ({ ci: Math.floor((o.y + o.h / 2) / cell), cj: Math.floor((o.x + o.w / 2) / cell) }));
  const collisions = resp.filter((r, i) => resp.findIndex((q) => q.ci === r.ci && q.cj === r.cj) !== i).length;
  const hl = img.map((r, i) => r.map((_, j) => (resp.some((q) => Math.floor(i / cell) === q.ci && Math.floor(j / cell) === q.cj) ? 0.25 : 0)));
  const boxes2 = mode === "yolo"
    ? OBJ.map((o, k) => ({ x: o.x, y: o.y, w: o.w, h: o.h, color: col(k), text: `${o.cls} ← cell (${resp[k].ci},${resp[k].cj})` }))
    : (stage === "proposals" ? det.props.filter((_, i) => i % 9 === 0).map((b) => ({ ...b, color: "rgb(var(--fg) / 0.25)" }))
      : stage === "scored" ? det.scored.map((b) => ({ ...b, color: col(b.k), dashed: true }))
      : stage === "refined" ? det.refined.map((b) => ({ ...b, color: col(b.k), dashed: true }))
      : det.kept.map((b) => ({ ...b, color: col(b.k), text: `${OBJ[b.k].cls} ${fmt(b.s, 2)}` })));
  return (
    <LabShell
      viz={
        <div className="flex flex-wrap items-start gap-4">
          <ImgCanvas data={img} kind="rgb" label="synthetic scene (48×48)" size={300} boxes={boxes2} grid={mode === "yolo" ? cell : undefined} highlight={mode === "yolo" ? hl : undefined} />
          {mode === "yolo" && <MiniTable head={["object", "centre cell", "box (x, y, w, h) relative to cell"]} rows={OBJ.map((o, k) => [o.cls, `(${resp[k].ci}, ${resp[k].cj})`, `(${fmt(((o.x + o.w / 2) % cell) / cell, 2)}, ${fmt(((o.y + o.h / 2) % cell) / cell, 2)}, ${fmt(o.w / N, 2)}, ${fmt(o.h / N, 2)})`])} />}
        </div>
      }
      controls={
        <>
          <Segmented label="Detector family" value={mode} onChange={setMode} options={[{ value: "two-stage", label: "Two-stage (R-CNN)" }, { value: "yolo", label: "One-stage (YOLO)" }]} />
          {mode === "two-stage" ? <>
            <Segmented label="Pipeline stage" value={stage} onChange={setStage} options={[{ value: "proposals", label: "1 Proposals" }, { value: "scored", label: "2 Classify" }, { value: "refined", label: "3 Regress" }, { value: "nms", label: "4 NMS" }]} />
            <ParameterSlider label="NMS IoU threshold" value={thr} min={0.1} max={0.9} step={0.05} onChange={setThr} />
          </> : <ParameterSlider label="grid size S" value={S} min={2} max={12} step={1} onChange={setS} />}
          <p className="text-[12px] leading-relaxed text-faint">{mode === "two-stage" ? "Stage 1 proposes many candidate boxes (anchors at 3 scales × 3 shapes). Stage 2 classifies each region and regresses a tighter box; NMS removes duplicates. R-CNN ran a CNN per proposal; Fast R-CNN shares one feature map; Faster R-CNN also predicts the proposals with a network (RPN). Region scores here are computed from the pixels, standing in for a trained classifier." : "YOLO divides the image into an S × S grid and predicts all boxes in one forward pass. The cell containing an object's centre is responsible for it, predicting the offset inside the cell and the size relative to the image. With a coarse grid, two centres can fall into one cell."}</p>
        </>
      }
      readouts={<Readouts items={mode === "two-stage" ? [
        { label: "\\text{proposals}", tex: true, value: det.props.length },
        { label: "\\text{scored as objects}", tex: true, value: det.scored.length },
        { label: "\\text{after NMS}", tex: true, value: det.kept.length, color: C.green },
        { label: "\\text{CNN passes: R-CNN vs Fast}", tex: true, value: `${det.props.length} vs 1` },
      ] : [
        { label: "\\text{output tensor}", tex: true, value: `${S}×${S}×(2·5+3) = ${S * S * 13}` },
        { label: "\\text{cell size}", tex: true, value: `${fmt(cell, 1)} px` },
        { label: "\\text{centre collisions}", tex: true, value: collisions, color: collisions ? C.red : C.green },
      ]} />}
      calc={<CalcSteps steps={mode === "two-stage" ? [
        { tex: String.raw`t_x=\frac{x-x_a}{w_a},\ t_y=\frac{y-y_a}{h_a},\ t_w=\log\frac{w}{w_a},\ t_h=\log\frac{h}{h_a}`, note: "the box-regression targets relative to a proposal / anchor a" },
        { tex: String.raw`\text{NMS: keep the highest score, drop boxes with IoU}>${tex(thr, 2)}\text{ with it, repeat}` },
      ] : [
        { tex: String.raw`\text{output}\in\mathbb R^{S\times S\times(B\cdot5+C)},\quad(x,y,w,h,\text{conf})\ \text{per box},\ B=2,\ C=3` },
        { tex: String.raw`\text{loss}=\lambda_{\text{coord}}\sum(\text{box errors})+\sum(\text{conf errors})+\sum(\text{class errors})` },
      ]} />}
    />
  );
}

/* ------------------------------ semantic & instance segmentation ------------------------------ */
export function SegmentationViz() {
  const [noise, setNoise] = useState(40);
  const [smooth, setSmooth] = useState(true);
  const [view, setView] = useState<"semantic" | "instance">("semantic");
  const { img, gt } = useMemo(() => {
    const cells = [[12, 12, 6], [12, 30, 5], [30, 10, 7], [36, 36, 5]], rods = [[22, 22, 16, 4], [40, 4, 4, 14]];
    const gt: number[][] = [];
    for (let i = 0; i < N; i++) { gt.push([]); for (let j = 0; j < N; j++) {
      let c = 0;
      cells.forEach(([y, x, rr]) => { if ((i - y) ** 2 + (j - x) ** 2 < rr * rr) c = 1; });
      rods.forEach(([y, x, h, w]) => { if (i >= y && i < y + h && j >= x && j < x + w) c = 2; });
      gt[i].push(c);
    } }
    const means: RGB[] = [[50, 50, 60], [200, 90, 150], [90, 170, 230]];
    return { img: gt.map((row) => row.map((c) => means[c].slice() as RGB)), gt };
  }, []);
  const noisy = useMemo(() => { const r = mulberry32(3); return img.map((row) => row.map((c) => c.map((v) => Math.max(0, Math.min(255, v + noise * randn(r)))) as RGB)); }, [img, noise]);
  const res = useMemo(() => {
    const means: RGB[] = [[50, 50, 60], [200, 90, 150], [90, 170, 230]];
    // per-pixel classifier: nearest class colour
    let pred = noisy.map((row) => row.map((c) => { let b = 0, bd = Infinity; means.forEach((m, k) => { const d = (c[0] - m[0]) ** 2 + (c[1] - m[1]) ** 2 + (c[2] - m[2]) ** 2; if (d < bd) { bd = d; b = k; } }); return b; }));
    if (smooth) pred = pred.map((row, i) => row.map((_, j) => { const cnt = [0, 0, 0]; for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) cnt[pred[Math.min(N - 1, Math.max(0, i + a))][Math.min(N - 1, Math.max(0, j + b))]]++; return cnt.indexOf(Math.max(...cnt)); }));
    // instance segmentation: connected components of each foreground class (4-connectivity, flood fill)
    const inst = pred.map((r) => r.map(() => 0));
    let id = 0; const sizes: number[] = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (pred[i][j] > 0 && !inst[i][j]) {
      id++; let n = 0; const st = [[i, j]]; inst[i][j] = id;
      while (st.length) { const [a, b] = st.pop()!; n++; for (const [da, db] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const y = a + da, x = b + db; if (y >= 0 && y < N && x >= 0 && x < N && !inst[y][x] && pred[y][x] === pred[i][j]) { inst[y][x] = id; st.push([y, x]); } } }
      sizes.push(n);
    }
    const ious = [0, 1, 2].map((c) => { let I = 0, U = 0; for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const p = pred[i][j] === c, g = gt[i][j] === c; if (p && g) I++; if (p || g) U++; } return I / U; });
    return { pred, inst, ious, nInst: id, small: sizes.filter((s) => s < 6).length };
  }, [noisy, smooth, gt]);
  return (
    <LabShell
      viz={
        <div className="flex flex-wrap gap-3">
          <ImgCanvas data={noisy} kind="rgb" label="input image" size={190} />
          <ImgCanvas data={gt.map((r) => r.map((v) => v))} kind="labels" label="ground truth (semantic)" size={190} />
          {view === "semantic" ? <ImgCanvas data={res.pred} kind="labels" label="predicted classes per pixel" size={190} /> : <ImgCanvas data={res.inst} kind="labels" label={`instances (${res.nInst} components)`} size={190} />}
        </div>
      }
      controls={
        <>
          <Segmented label="Output" value={view} onChange={setView} options={[{ value: "semantic", label: "Semantic" }, { value: "instance", label: "Instance" }]} />
          <ParameterSlider label="image noise" value={noise} min={0} max={120} step={5} onChange={setNoise} />
          <Toggle label="3×3 majority smoothing" checked={smooth} onChange={setSmooth} />
          <p className="text-[12px] leading-relaxed text-faint">Semantic segmentation gives every pixel a class (cell, rod, background). Instance segmentation also separates individual objects of the same class — here by finding connected regions. Noise creates speckles, which become spurious tiny instances; local smoothing (what large receptive fields and CRFs achieve in real models) removes them.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{IoU background / cell / rod}", tex: true, value: res.ious.map((v) => fmt(v, 3)).join(" / ") },
        { label: "\\text{mIoU}", tex: true, value: fmt(res.ious.reduce((s, v) => s + v, 0) / 3, 3), color: C.green },
        { label: "\\text{instances (true: 6)}", tex: true, value: res.nInst, color: res.nInst === 6 ? C.green : C.red },
        { label: "\\text{speckle instances (<6 px)}", tex: true, value: res.small },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\hat y_{ij}=\arg\min_c\|\mathbf x_{ij}-\boldsymbol\mu_c\|^2,\qquad \text{IoU}_c=\frac{|\hat Y=c\ \cap\ Y=c|}{|\hat Y=c\ \cup\ Y=c|},\quad\text{mIoU}=\frac1C\sum_c\text{IoU}_c`, note: "a nearest-colour pixel classifier stands in for a trained network's per-pixel softmax" },
      ]} />}
    />
  );
}

/* ------------------------------ vision transformer patches ------------------------------ */
function vitImage(): RGB[][] {
  return Array.from({ length: 64 }, (_, i) => Array.from({ length: 64 }, (_, j) => {
    let c: RGB = [120 + i, 170 + i * 0.6, 235 - i * 0.4];
    if (i > 46) c = [70, 140 + ((i + j) % 6) * 4, 70];
    if ((i - 14) ** 2 + (j - 48) ** 2 < 49) c = [250, 210, 80];
    if (i > 26 && i <= 46 && j > 8 && j < 32) c = [190, 90, 70];
    if ((i - 34) ** 2 + (j - 50) ** 2 < 70) c = [40, 125, 50];
    return c.map((v) => Math.min(255, v)) as RGB;
  }));
}
export function ViTViz() {
  const [p, setP] = useState(8);
  const [temp, setTemp] = useState(0.15);
  const [sel, setSel] = useState<[number, number]>([4, 2]);
  const img = useMemo(vitImage, []);
  const g = 64 / p, n = g * g;
  // a patch embedding stand-in: mean colour + mean brightness gradient of the patch, normalised
  const emb = useMemo(() => Array.from({ length: n }, (_, k) => {
    const pi = Math.floor(k / g), pj = k % g; let r = 0, gg = 0, b = 0, gx = 0;
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) { const c = img[pi * p + i][pj * p + j]; r += c[0]; gg += c[1]; b += c[2]; if (j > 0) gx += Math.abs(c[0] - img[pi * p + i][pj * p + j - 1][0]); }
    const v = [r, gg, b].map((x) => x / (p * p * 255)).concat([gx / (p * p * 60)]);
    const m = v.reduce((s, x) => s + x, 0) / v.length; const c = v.map((x) => x - m); const nn = Math.hypot(...c) || 1;
    return c.map((x) => x / nn);
  }), [img, p, g, n]);
  const si = Math.min(g - 1, sel[0]), sj = Math.min(g - 1, sel[1]), q = si * g + sj;
  const scores = emb.map((e) => e.reduce((s, x, k) => s + x * emb[q][k], 0) / temp);
  const mx = Math.max(...scores), ex = scores.map((s) => Math.exp(s - mx)), Z = ex.reduce((s, x) => s + x, 0), att = ex.map((x) => x / Z);
  const amax = Math.max(...att);
  const hl = img.map((row, i) => row.map((_, j) => 0.75 * (att[Math.floor(i / p) * g + Math.floor(j / p)] / amax)));
  return (
    <LabShell
      viz={
        <div className="flex flex-wrap items-start gap-4">
          <ImgCanvas data={img} kind="rgb" label={`image split into ${n} patches of ${p}×${p} (click a patch)`} size={300} grid={p} onPick={(i, j) => setSel([Math.floor(i / p), Math.floor(j / p)])} boxes={[{ x: sj * p, y: si * p, w: p, h: p, color: "rgb(var(--fg))" }]} />
          <ImgCanvas data={img} kind="rgb" label="attention from the selected patch (red = high)" size={300} highlight={hl} grid={p} />
        </div>
      }
      controls={
        <>
          <Select label="Patch size" value={String(p)} onChange={(v) => setP(Number(v))} options={[4, 8, 16, 32].map((v) => ({ value: String(v), label: `${v}×${v}` }))} />
          <ParameterSlider label="softmax temperature" value={temp} min={0.05} max={1} step={0.01} onChange={setTemp} />
          <p className="text-[12px] leading-relaxed text-faint">A Vision Transformer cuts the image into patches, flattens each into a vector, projects it to an embedding and adds a position embedding. The patch sequence (plus a [CLS] token) goes through standard Transformer layers, so any patch can attend to any other in the very first layer. Smaller patches mean more tokens and quadratically more attention cost. Attention here uses simple colour/texture embeddings, not trained weights.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{tokens }N", tex: true, value: `${n} + 1 [CLS]` },
        { label: "\\text{attention scores per head}", tex: true, value: ((n + 1) ** 2).toLocaleString("en-US") },
        { label: "\\text{patch vector length}", tex: true, value: `${p}·${p}·3 = ${p * p * 3}` },
        { label: "\\text{embedding params (D=768)}", tex: true, value: (p * p * 3 * 768 + 768).toLocaleString("en-US") },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\mathbf z_0=[\mathbf x_{\text{cls}};\ \mathbf x_p^1E;\ \dots;\ \mathbf x_p^NE]+E_{\text{pos}},\quad N=\Big(\frac{64}{${p}}\Big)^2=${n}` },
        { tex: String.raw`\text{attention cost}\propto N^2:\ \text{halving the patch size multiplies it by }16` },
      ]} />}
    />
  );
}

/* ------------------------------ image embeddings & contrastive (CLIP-style) ------------------------------ */
const ITEMS = [
  { color: 0, shape: 0, cap: "a red circle" }, { color: 1, shape: 1, cap: "a blue square" }, { color: 2, shape: 2, cap: "a green triangle" },
  { color: 0, shape: 1, cap: "a red square" }, { color: 1, shape: 0, cap: "a blue circle" }, { color: 2, shape: 0, cap: "a green circle" },
];
const COLORS: RGB[] = [[240, 90, 90], [90, 150, 245], [90, 200, 120]];
function itemImage(c: number, s: number): RGB[][] {
  return Array.from({ length: 24 }, (_, i) => Array.from({ length: 24 }, (_, j) => {
    const x = j - 12, y = i - 12;
    const inside = s === 0 ? x * x + y * y < 64 : s === 1 ? Math.abs(x) < 7 && Math.abs(y) < 7 : y > -8 && y < 7 && Math.abs(x) < (y + 8) * 0.55;
    return inside ? COLORS[c] : ([28, 28, 32] as RGB);
  }));
}
export function ImageEmbedViz() {
  const [temp, setTemp] = useState(0.1);
  const [align, setAlign] = useState(0.8);
  const [query, setQuery] = useState(4);
  const { I, Tt } = useMemo(() => {
    const r = mulberry32(4);
    const unit = (v: number[]) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
    // image encoder stand-in: colour histogram + shape descriptors (6 dims) followed by a random rotation to 8 dims
    const R8 = Array.from({ length: 8 }, () => Array.from({ length: 6 }, () => randn(r)));
    const feat = (c: number, s: number) => unit(R8.map((row) => row.reduce((acc, w, k) => acc + w * (k < 3 ? (k === c ? 1 : 0) : k - 3 === s ? 1 : 0), 0)));
    const I = ITEMS.map((it) => feat(it.color, it.shape));
    // text encoder stand-in: same concepts, partially aligned (align = 1 → identical space), plus noise
    const noise = ITEMS.map(() => Array.from({ length: 8 }, () => randn(r)));
    const Tt = ITEMS.map((it, k) => unit(feat(it.color, it.shape).map((v, d) => align * v + (1 - align) * 0.6 * noise[k][d])));
    return { I, Tt };
  }, [align]);
  const S = I.map((a) => Tt.map((b) => a.reduce((s, x, k) => s + x * b[k], 0)));
  const soft = (row: number[]) => { const z = row.map((v) => v / temp), m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
  const Pi = S.map(soft), Pt = S[0].map((_, j) => soft(S.map((r) => r[j])));
  const loss = (-Pi.reduce((s, r, i) => s + Math.log(r[i]), 0) - Pt.reduce((s, r, j) => s + Math.log(r[j]), 0)) / (2 * ITEMS.length);
  const ranked = I.map((e, k) => ({ k, s: e.reduce((a, x, d) => a + x * Tt[query][d], 0) })).sort((a, b) => b.s - a.s);
  const top1 = Pi.filter((r, i) => r.indexOf(Math.max(...r)) === i).length / ITEMS.length;
  return (
    <LabShell
      wideControls
      viz={
        <div className="grid gap-4 xl:grid-cols-[auto_1fr]">
          <div className="space-y-1.5">
            <div className="text-[12px] text-faint">Cosine similarity: images (rows) × captions (columns)</div>
            <div className="flex gap-1 pl-[54px]">{ITEMS.map((_, j) => <span key={j} className="w-[40px] truncate text-center text-[10px] text-faint" title={ITEMS[j].cap}>T{j + 1}</span>)}</div>
            <div className="flex gap-2">
              <div className="flex flex-col gap-[2px]">{ITEMS.map((it, i) => <ImgCanvas key={i} data={itemImage(it.color, it.shape)} kind="rgb" label="" size={26} />)}</div>
              <Heat M={S} digits={2} cell={40} signed max={1} highlight={(i, j) => i === j} />
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-[12px] text-faint">Retrieval for “{ITEMS[query].cap}”</div>
            <div className="flex flex-wrap gap-2">{ranked.map((r, n) => <div key={r.k} className="flex flex-col items-center"><ImgCanvas data={itemImage(ITEMS[r.k].color, ITEMS[r.k].shape)} kind="rgb" label={`#${n + 1} · ${fmt(r.s, 2)}`} size={56} /></div>)}</div>
            <p className="text-[12px] text-faint">{ITEMS.map((it, j) => `T${j + 1}: ${it.cap}`).join(" · ")}</p>
          </div>
        </div>
      }
      controls={
        <>
          <Select label="Text query" value={String(query)} onChange={(v) => setQuery(Number(v))} options={ITEMS.map((it, k) => ({ value: String(k), label: it.cap }))} />
          <ParameterSlider label="image–text alignment" value={align} min={0} max={1} step={0.05} onChange={setAlign} />
          <ParameterSlider label="temperature τ" value={temp} min={0.02} max={1} step={0.01} onChange={setTemp} />
          <p className="text-[12px] leading-relaxed text-faint">CLIP trains an image encoder and a text encoder so that matching pairs (the diagonal) have high cosine similarity and mismatched pairs low. Retrieval is then just a nearest-neighbour search in the shared space. Here both encoders are hand-built from colour and shape; the alignment slider mimics how well training has pulled the two spaces together.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{InfoNCE loss}", tex: true, value: fmt(loss, 3) },
        { label: "\\text{image→text top-1}", tex: true, value: pct(top1), color: C.green },
        { label: "\\text{loss of random guessing}", tex: true, value: fmt(Math.log(ITEMS.length), 3), hint: "log N" },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`s_{ij}=\frac{\mathbf u_i^\top\mathbf v_j}{\|\mathbf u_i\|\|\mathbf v_j\|},\qquad L=-\frac1{2N}\sum_i\Big[\log\frac{e^{s_{ii}/\tau}}{\sum_je^{s_{ij}/\tau}}+\log\frac{e^{s_{ii}/\tau}}{\sum_je^{s_{ji}/\tau}}\Big]`, note: "symmetric contrastive loss over a batch of N pairs" },
      ]} />}
    />
  );
}
