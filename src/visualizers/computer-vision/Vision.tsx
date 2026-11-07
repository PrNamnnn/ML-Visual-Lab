"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Select, Segmented, Toggle } from "@/components/controls";
import { Plot, C, usePlot } from "@/components/visualization/Plot";
import { MatrixView } from "@/components/mathematics/MatrixView";
import { fmt, tex } from "@/lib/utilities/format";

const N = 64;
type RGB = [number, number, number];

/** Procedurally generated test image (no external assets): sky gradient, sun, house, tree, text-like bars. */
function makeImage(): RGB[][] {
  const img: RGB[][] = [];
  for (let i = 0; i < N; i++) {
    const row: RGB[] = [];
    for (let j = 0; j < N; j++) {
      let c: RGB = [110 + i * 1.6, 160 + i * 0.8, 235 - i * 0.5];
      if (i > 44) c = [70 + (j % 7) * 3, 140 + ((i + j) % 5) * 4, 70];
      if (Math.hypot(i - 12, j - 48) < 7) c = [250, 210, 80];
      if (i > 28 && i <= 44 && j > 10 && j < 34) c = [190, 90, 70];
      if (i > 34 && i <= 44 && j > 18 && j < 25) c = [70, 45, 35];
      if (i > 30 && i < 35 && j > 13 && j < 17) c = [230, 230, 180];
      if (i > 15 && i <= 28 && j - 9 > 28 - i && j - 35 < i - 28 + 0 && j > 6 && j < 38) c = [130, 40, 40];
      if (Math.hypot(i - 32, j - 50) < 8) c = [40, 120 + ((i * j) % 20), 50];
      if (i > 38 && i <= 44 && j > 48 && j < 52) c = [90, 60, 30];
      row.push([Math.min(255, c[0]), Math.min(255, c[1]), Math.min(255, c[2])]);
    }
    img.push(row);
  }
  return img;
}

const gray = (img: RGB[][]) => img.map((r) => r.map(([R, G, B]) => 0.299 * R + 0.587 * G + 0.114 * B));

function convolve(G: number[][], K: number[][]) {
  const k = K.length, h = (k - 1) / 2;
  return G.map((r, i) => r.map((_, j) => {
    let s = 0;
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
      const ii = Math.min(N - 1, Math.max(0, i + a - h)), jj = Math.min(N - 1, Math.max(0, j + b - h)); // replicate padding
      s += G[ii][jj] * K[a][b];
    }
    return s;
  }));
}

function gaussKernel(sigma: number) {
  const r = Math.max(1, Math.ceil(2.5 * sigma));
  const K = Array.from({ length: 2 * r + 1 }, (_, a) => Array.from({ length: 2 * r + 1 }, (_, b) => Math.exp(-((a - r) ** 2 + (b - r) ** 2) / (2 * sigma * sigma))));
  const s = K.flat().reduce((x, y) => x + y, 0);
  return K.map((row) => row.map((v) => v / s));
}

function Canvas({ data, label, mode = "gray", signed = false }: { data: number[][] | RGB[][]; label: string; mode?: "gray" | "rgb"; signed?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const g = ref.current?.getContext("2d");
    if (!g) return;
    const im = g.createImageData(N, N);
    let max = 1;
    if (mode === "gray") max = signed ? Math.max(1e-9, ...(data as number[][]).flat().map(Math.abs)) : 255;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const k = (i * N + j) * 4;
      if (mode === "rgb") { const [r, gg, b] = (data as RGB[][])[i][j]; im.data[k] = r; im.data[k + 1] = gg; im.data[k + 2] = b; }
      else {
        const v = (data as number[][])[i][j];
        if (signed) { const t = v / max; im.data[k] = t < 0 ? 255 * -t : 0; im.data[k + 1] = 40 * Math.abs(t); im.data[k + 2] = t > 0 ? 255 * t : 0; }
        else { const t = Math.max(0, Math.min(255, v)); im.data[k] = t; im.data[k + 1] = t; im.data[k + 2] = t; }
      }
      im.data[k + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }, [data, mode, signed]);
  return (
    <figure className="flex flex-col items-center gap-1">
      <canvas ref={ref} width={N} height={N} role="img" aria-label={label} className="h-[200px] w-[200px] rounded border border-line" style={{ imageRendering: "pixelated" }} />
      <figcaption className="text-[12px] text-muted">{label}</figcaption>
    </figure>
  );
}

export function ImageFilters({ filter: initial = "sobel" }: { filter?: "channels" | "grayscale" | "normalize" | "blur" | "sobel" | "gradient" }) {
  const [filter, setFilter] = useState(initial);
  const [sigma, setSigma] = useState(1.2);
  const [ch, setCh] = useState<"R" | "G" | "B">("R");
  const [blurFirst, setBlurFirst] = useState(true);
  const img = useMemo(makeImage, []);
  const G = useMemo(() => gray(img), [img]);
  const Gb = useMemo(() => convolve(G, gaussKernel(sigma)), [G, sigma]);
  const src = blurFirst ? Gb : G;
  const sx = useMemo(() => convolve(src, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), [src]);
  const sy = useMemo(() => convolve(src, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), [src]);
  const mag = useMemo(() => sx.map((r, i) => r.map((v, j) => Math.hypot(v, sy[i][j]))), [sx, sy]);
  const pi = 40, pj = 22;
  const pix = img[pi][pj];
  const flat = G.flat();
  const mu = flat.reduce((a, b) => a + b, 0) / flat.length;
  const sd = Math.sqrt(flat.reduce((a, b) => a + (b - mu) ** 2, 0) / flat.length);
  const patch = G.slice(pi - 1, pi + 2).map((r) => r.slice(pj - 1, pj + 2));
  let views: React.ReactNode = null;
  let steps: { tex: string; note?: string }[] = [];
  switch (filter) {
    case "channels": {
      const idx = { R: 0, G: 1, B: 2 }[ch];
      views = <><Canvas data={img} label="RGB image (64×64×3)" mode="rgb" /><Canvas data={img.map((r) => r.map((p) => p[idx]))} label={`${ch} channel only`} /></>;
      steps = [{ tex: `\\text{pixel }(${pi},${pj})=(R,G,B)=(${pix.map((v) => Math.round(v)).join(",")})`, note: "an image is a height × width × channels tensor of 0–255 integers" }];
      break;
    }
    case "grayscale":
      views = <><Canvas data={img} label="RGB" mode="rgb" /><Canvas data={G} label="grayscale (luma)" /></>;
      steps = [{ tex: `Y=0.299R+0.587G+0.114B=0.299\\cdot${Math.round(pix[0])}+0.587\\cdot${Math.round(pix[1])}+0.114\\cdot${Math.round(pix[2])}=${tex(G[pi][pj], 1)}`, note: "weights reflect human sensitivity to green" }];
      break;
    case "normalize":
      views = <><Canvas data={G} label="raw 0–255" /><Canvas data={G.map((r) => r.map((v) => (v - mu) / sd))} label="standardised (signed: blue +, red −)" mode="gray" signed /></>;
      steps = [{ tex: `x'=\\frac{x-\\mu}{\\sigma}=\\frac{${tex(G[pi][pj], 1)}-${tex(mu, 1)}}{${tex(sd, 1)}}=${tex((G[pi][pj] - mu) / sd)}`, note: "or simply x/255 to map into [0, 1]; CNNs typically use per-channel ImageNet mean/std" }];
      break;
    case "blur":
      views = <><Canvas data={G} label="grayscale" /><Canvas data={Gb} label={`Gaussian blur σ = ${fmt(sigma, 1)}`} /></>;
      steps = [{ tex: `K_{ab}\\propto\\exp\\!\\left(-\\frac{a^2+b^2}{2\\sigma^2}\\right),\\ \\text{size }${gaussKernel(sigma).length}\\times${gaussKernel(sigma).length},\\ \\sum K=1` }];
      break;
    case "sobel":
      views = <><Canvas data={sx} label="Sobel Gx (vertical edges)" signed /><Canvas data={sy} label="Sobel Gy (horizontal edges)" signed /></>;
      steps = [{ tex: `G_x=\\begin{bmatrix}-1&0&1\\\\-2&0&2\\\\-1&0&1\\end{bmatrix}*I,\\quad G_x(${pi},${pj})=${tex(sx[pi][pj], 1)}` }];
      break;
    case "gradient":
      views = <><Canvas data={src} label={blurFirst ? "blurred input" : "input"} /><Canvas data={mag.map((r) => r.map((v) => Math.min(255, v / 3)))} label="gradient magnitude ‖∇I‖" /></>;
      steps = [{ tex: `\\|\\nabla I\\|=\\sqrt{G_x^2+G_y^2}=\\sqrt{${tex(sx[pi][pj], 1)}^2+${tex(sy[pi][pj], 1)}^2}=${tex(mag[pi][pj], 1)},\\quad\\theta=\\operatorname{atan2}(G_y,G_x)=${tex((Math.atan2(sy[pi][pj], sx[pi][pj]) * 180) / Math.PI, 1)}^\\circ` }];
      break;
  }
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div className="flex flex-wrap justify-center gap-6">{views}</div>
          {(filter === "sobel" || filter === "gradient" || filter === "blur") && (
            <div className="flex flex-wrap items-center justify-center gap-4">
              <MatrixView M={patch} label={`3×3 patch around (${pi}, ${pj})`} digits={0} cell={42} heat={false} />
            </div>
          )}
        </div>
      }
      controls={
        <>
          <Select label="Operation" value={filter} onChange={setFilter} options={[{ value: "channels", label: "RGB channels" }, { value: "grayscale", label: "Grayscale conversion" }, { value: "normalize", label: "Normalisation" }, { value: "blur", label: "Gaussian blur" }, { value: "sobel", label: "Sobel edge filters" }, { value: "gradient", label: "Image gradient magnitude" }]} />
          {filter === "channels" && <Segmented value={ch} onChange={setCh} options={[{ value: "R", label: "R" }, { value: "G", label: "G" }, { value: "B", label: "B" }]} />}
          {(filter === "blur" || filter === "gradient" || filter === "sobel") && <ParameterSlider label="Gaussian σ" value={sigma} min={0.4} max={3} step={0.1} onChange={setSigma} />}
          {(filter === "gradient" || filter === "sobel") && <Toggle label="Blur before differentiating" checked={blurFirst} onChange={setBlurFirst} />}
        </>
      }
      readouts={<Readouts items={[{ label: "\\text{image}", tex: true, value: "64 × 64 × 3, synthetic" }, { label: "\\mu,\\ \\sigma\\ \\text{(gray)}", tex: true, value: `${fmt(mu, 1)}, ${fmt(sd, 1)}` }]} />}
      calc={<CalcSteps steps={steps} />}
      note="The test image is generated procedurally in the browser. Borders use replicate padding. Blurring before Sobel suppresses pixel noise — the same idea as the Canny edge detector."
    />
  );
}

/* ------------------------------ IoU & non-maximum suppression ------------------------------ */
type Box = { x: number; y: number; w: number; h: number; s: number };
const iou = (a: Box, b: Box) => {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy;
  return { inter, union: a.w * a.h + b.w * b.h - inter, iou: inter / (a.w * a.h + b.w * b.h - inter) };
};

function BoxR({ b, color, dashed, label }: { b: Box; color: string; dashed?: boolean; label?: string }) {
  const { sx, sy } = usePlot();
  return (
    <g>
      <rect x={sx(b.x)} y={sy(b.y + b.h)} width={sx(b.x + b.w) - sx(b.x)} height={sy(b.y) - sy(b.y + b.h)} style={{ fill: "none", stroke: color, strokeWidth: 2, strokeDasharray: dashed ? "5 3" : undefined }} />
      {label && <text x={sx(b.x) + 3} y={sy(b.y + b.h) + 13} style={{ fill: color, fontSize: 11 }}>{label}</text>}
    </g>
  );
}

export function IoUNMS({ mode = "iou" }: { mode?: "iou" | "nms" }) {
  const [view, setView] = useState(mode);
  const [pred, setPred] = useState<Box>({ x: 3.2, y: 2.4, w: 4, h: 3.5, s: 1 });
  const gt: Box = { x: 2.5, y: 2, w: 4.2, h: 3.8, s: 1 };
  const [thr, setThr] = useState(0.5);
  const [scoreThr, setScoreThr] = useState(0.3);
  const cands: Box[] = useMemo(() => [
    { x: 1, y: 1, w: 3, h: 3.4, s: 0.92 }, { x: 1.3, y: 1.2, w: 3.1, h: 3.2, s: 0.81 }, { x: 0.7, y: 0.8, w: 3.3, h: 3.6, s: 0.66 },
    { x: 5.4, y: 4, w: 3.2, h: 2.6, s: 0.88 }, { x: 5.8, y: 4.3, w: 2.9, h: 2.4, s: 0.74 }, { x: 5, y: 3.6, w: 3.6, h: 3, s: 0.4 },
    { x: 3, y: 5.4, w: 2, h: 1.6, s: 0.25 },
  ], []);
  const kept: number[] = [];
  const suppressedBy: Record<number, number> = {};
  const order = cands.map((_, i) => i).filter((i) => cands[i].s >= scoreThr).sort((a, b) => cands[b].s - cands[a].s);
  for (const i of order) {
    const sup = kept.find((k) => iou(cands[k], cands[i]).iou > thr);
    if (sup === undefined) kept.push(i); else suppressedBy[i] = sup;
  }
  const r = iou(pred, gt);
  const PAL = ["#5aa9ff", "#ff6fae", "#6ecf97", "#e6c35c", "#c58bff", "#f59e5b", "#38d4c4"];
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[0, 10]} y={[0, 8]} equal aspect={0.75} maxH={440} ariaLabel={view === "iou" ? "intersection over union" : "non-maximum suppression"}
            onPointerDown={view === "iou" ? (p) => setPred({ ...pred, x: Math.max(0, Math.min(10 - pred.w, p[0] - pred.w / 2)), y: Math.max(0, Math.min(8 - pred.h, p[1] - pred.h / 2)) }) : undefined}>
            {view === "iou" ? (
              <>
                <BoxR b={gt} color={C.green} label="ground truth" />
                <BoxR b={pred} color={C.pink} label="prediction" />
                {r.inter > 0 && <BoxR b={{ x: Math.max(pred.x, gt.x), y: Math.max(pred.y, gt.y), w: Math.min(pred.x + pred.w, gt.x + gt.w) - Math.max(pred.x, gt.x), h: Math.min(pred.y + pred.h, gt.y + gt.h) - Math.max(pred.y, gt.y), s: 1 }} color={C.amber} dashed />}
              </>
            ) : (
              cands.map((b, i) => {
                const isKept = kept.includes(i);
                const below = b.s < scoreThr;
                return <BoxR key={i} b={b} color={below ? C.a("fg", 0.15) : isKept ? PAL[kept.indexOf(i) % 7] : C.a("fg", 0.3)} dashed={!isKept} label={`${b.s.toFixed(2)}${isKept ? " ✓" : below ? "" : ` ✗ (IoU ${fmt(iou(cands[suppressedBy[i]], b).iou, 2)})`}`} />;
              })
            )}
          </Plot>
          <Legend items={view === "iou" ? [{ label: "ground truth", color: C.green }, { label: "prediction (click to move)", color: C.pink }, { label: "intersection", color: C.amber, dashed: true }] : [{ label: "kept detections", color: C.blue }, { label: "suppressed (dashed)", color: C.a("fg", 0.4), dashed: true }]} />
        </div>
      }
      controls={
        <>
          <Segmented value={view} onChange={setView} options={[{ value: "iou", label: "IoU" }, { value: "nms", label: "Non-max suppression" }]} />
          {view === "iou" ? (
            <>
              <ParameterSlider label="pred width" value={pred.w} min={0.5} max={7} step={0.1} onChange={(w) => setPred({ ...pred, w })} />
              <ParameterSlider label="pred height" value={pred.h} min={0.5} max={6} step={0.1} onChange={(h) => setPred({ ...pred, h })} />
              <ParameterSlider label="match threshold" value={thr} min={0.1} max={0.95} step={0.05} onChange={setThr} />
            </>
          ) : (
            <>
              <ParameterSlider label="IoU threshold for suppression" value={thr} min={0.1} max={0.9} step={0.05} onChange={setThr} />
              <ParameterSlider label="confidence threshold" value={scoreThr} min={0} max={0.9} step={0.05} onChange={setScoreThr} />
            </>
          )}
        </>
      }
      readouts={<Readouts items={view === "iou" ? [{ label: "|A\\cap B|", tex: true, value: fmt(r.inter) }, { label: "|A\\cup B|", tex: true, value: fmt(r.union) }, { label: "\\text{IoU}", tex: true, value: fmt(r.iou, 3), color: r.iou >= thr ? C.green : C.red }, { label: "\\text{match?}", tex: true, value: r.iou >= thr ? "true positive" : "miss / false positive" }] : [{ label: "\\text{candidates}", tex: true, value: cands.length }, { label: "\\text{above confidence}", tex: true, value: order.length }, { label: "\\text{kept}", tex: true, value: kept.length }]} />}
      calc={<CalcSteps steps={view === "iou" ? [{ tex: `\\text{IoU}=\\frac{|A\\cap B|}{|A\\cup B|}=\\frac{${tex(r.inter)}}{${tex(r.union)}}=${tex(r.iou)}` }] : [{ tex: "\\text{sort by score; repeatedly keep the best box and drop every remaining box with IoU}>\\tau\\text{ against it}", note: `τ = ${thr}` }]} />}
      note={view === "iou" ? "Detection benchmarks count a prediction as correct when IoU ≥ 0.5 (PASCAL VOC) or average over 0.5–0.95 (COCO mAP)." : "Detectors like YOLO and Faster R-CNN output many overlapping boxes per object; NMS keeps one per object. Too low a threshold merges nearby objects, too high leaves duplicates."}
    />
  );
}
