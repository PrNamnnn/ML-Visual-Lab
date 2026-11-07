"use client";
import { useMemo, useRef, useState } from "react";
import { Plot, Dot, C, Polyline, FnPath } from "@/components/visualization/Plot";
import { Bars } from "@/components/visualization/Bars";
import { LabShell, Readouts, CalcSteps, Legend } from "@/components/visualization/LabShell";
import { ParameterSlider, Segmented, AnimationControls } from "@/components/controls";
import { mulberry32, randn } from "@/lib/math/random";
import { sigmoid } from "@/lib/ml/models";
import { histogram } from "@/lib/statistics/descriptive";
import { useTicker } from "@/hooks/useSize";
import { fmt, pct, tex } from "@/lib/utilities/format";
import { Heat, MiniTable } from "@/visualizers/shared";

const gauss = (x: number, m: number, s: number) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));

/* ------------------------------ bidirectional RNN: who can influence whom ------------------------------ */
const WORDS = ["the", "bank", "of", "the", "river", "was", "muddy"];
export function BiRNNViz() {
  const [dir, setDir] = useState<"fwd" | "bwd" | "bi">("bi");
  const [wscale, setWscale] = useState(0.9);
  const [focus, setFocus] = useState(1);
  const T = WORDS.length, H = 8;
  const res = useMemo(() => {
    const r = mulberry32(4);
    const mk = (a: number, b: number, s: number) => Array.from({ length: a }, () => Array.from({ length: b }, () => (s * randn(r)) / Math.sqrt(b)));
    const Wx = mk(H, 4, 1), Whf = mk(H, H, wscale), Whb = mk(H, H, wscale);
    const X = WORDS.map(() => Array.from({ length: 4 }, () => randn(r)));
    const run = (Wh: number[][], order: number[]) => {
      // forward pass storing states, then the Jacobian norm ‖∂h_t/∂x_s‖ via the chain rule
      const h: number[][] = Array(T);
      let prev = Array(H).fill(0);
      for (const t of order) { h[t] = Wx.map((row, i) => Math.tanh(row.reduce((s, w, j) => s + w * X[t][j], 0) + Wh[i].reduce((s, w, j) => s + w * prev[j], 0))); prev = h[t]; }
      // J[t][s] = ‖∂h_t/∂x_s‖_F with ∂h_t/∂x_s = D_t W_h D_{u} W_h … D_s W_x along the processing order (D = diag(1 − h²))
      const J = Array.from({ length: T }, () => Array(T).fill(0));
      order.forEach((t, k) => {
        for (let m = 0; m <= k; m++) {
          const s = order[m];
          let A = Wx.map((row, i) => row.map((w) => w * (1 - h[s][i] ** 2)));
          for (let q = m + 1; q <= k; q++) {
            const u = order[q], prevA = A;
            A = Wh.map((whRow, i) => prevA[0].map((_, j) => whRow.reduce((acc, w, z) => acc + w * prevA[z][j], 0) * (1 - h[u][i] ** 2)));
          }
          J[t][s] = Math.hypot(...A.flat());
        }
      });
      return J;
    };
    const order = WORDS.map((_, i) => i);
    const Jf = run(Whf, order), Jb = run(Whb, order.slice().reverse());
    return { Jf, Jb };
  }, [wscale]); // eslint-disable-line react-hooks/exhaustive-deps
  const J = res.Jf.map((row, t) => row.map((v, s) => (dir === "fwd" ? v : dir === "bwd" ? res.Jb[t][s] : Math.hypot(v, res.Jb[t][s]))));
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1.5">{WORDS.map((w, s) => <button key={s} onClick={() => setFocus(s)} className="rounded-md border border-line px-2.5 py-1 text-[13px]" style={{ background: `rgb(var(--blue) / ${Math.min(0.75, J[focus][s] / Math.max(...J[focus], 1e-9) * 0.75)})`, outline: s === focus ? "2px solid rgb(var(--fg))" : undefined }}>{w}</button>)}</div>
          <div className="text-[12px] text-faint">Row t, column s: how strongly input word s can change the hidden state at position t, ‖∂h_t/∂x_s‖.</div>
          <Heat M={J} rowLabels={WORDS.map((w, i) => `h${i + 1} ${w}`)} colLabels={WORDS} digits={2} cell={46} highlight={(i) => i === focus} />
        </div>
      }
      controls={
        <>
          <Segmented label="Direction" value={dir} onChange={setDir} options={[{ value: "fwd", label: "Forward" }, { value: "bwd", label: "Backward" }, { value: "bi", label: "Bidirectional" }]} />
          <ParameterSlider label="recurrent weight scale" value={wscale} min={0.2} max={2} step={0.05} onChange={setWscale} />
          <p className="text-[12px] leading-relaxed text-faint">Click a word. A forward RNN&apos;s state for &quot;bank&quot; cannot see &quot;river&quot; yet (zeros above the diagonal). A backward RNN sees only the future. Concatenating both gives every position the whole sentence — exactly what is needed to tell a river bank from a money bank.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: `\\text{words reaching } h_{${focus + 1}}`, tex: true, value: J[focus].filter((v) => v > 1e-9).length + " / " + WORDS.length },
        { label: "\\text{state size}", tex: true, value: dir === "bi" ? `2 × ${H} = ${2 * H}` : H },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\overrightarrow{\mathbf h}_t=\tanh(W_x\mathbf x_t+\overrightarrow{W}_h\overrightarrow{\mathbf h}_{t-1}),\quad \overleftarrow{\mathbf h}_t=\tanh(W_x\mathbf x_t+\overleftarrow{W}_h\overleftarrow{\mathbf h}_{t+1}),\quad \mathbf h_t=[\overrightarrow{\mathbf h}_t;\overleftarrow{\mathbf h}_t]` },
        { tex: String.raw`\frac{\partial\overrightarrow{\mathbf h}_t}{\partial\mathbf x_s}=\Big(\prod_{u=s+1}^{t}D_uW_h\Big)D_sW_x,\quad D_u=\operatorname{diag}(1-\mathbf h_u^2)`, note: "exactly zero for s > t in the forward direction" },
      ]} />}
    />
  );
}

/* ------------------------------ seq2seq: the fixed-size bottleneck ------------------------------ */
export function Seq2SeqViz() {
  const [d, setD] = useState(32);
  const [len, setLen] = useState(6);
  const [attn, setAttn] = useState(false);
  const V = 20;
  const res = useMemo(() => {
    const r = mulberry32(2);
    const E = Array.from({ length: V }, () => { const v = Array.from({ length: d }, () => randn(r)); const n = Math.hypot(...v); return v.map((x) => x / n); });
    const Pm = Array.from({ length: 40 }, () => Array.from({ length: d }, () => Array.from({ length: d }, () => randn(r) / Math.sqrt(d)))); // a random "position" map per step
    const mv = (M: number[][], v: number[]) => M.map((row) => row.reduce((s, w, j) => s + w * v[j], 0));
    const mvT = (M: number[][], v: number[]) => M[0].map((_, j) => M.reduce((s, row, i) => s + row[j] * v[i], 0));
    const trial = (L: number, rs: () => number) => {
      const seq = Array.from({ length: L }, () => Math.floor(rs() * V));
      if (attn) return { seq, out: seq.slice() }; // with attention the decoder can read each encoder state directly
      const c = Array(d).fill(0);
      seq.forEach((tok, t) => mv(Pm[t], E[tok]).forEach((v, i) => (c[i] += v))); // encoder: everything summed into one vector
      const out = seq.map((_, t) => { const q = mvT(Pm[t], c); let best = 0, bv = -Infinity; E.forEach((e, v) => { const s = e.reduce((a, x, i) => a + x * q[i], 0); if (s > bv) { bv = s; best = v; } }); return best; });
      return { seq, out };
    };
    const rs = mulberry32(9);
    const curve = [2, 4, 8, 12, 16, 24, 32, 40].map((L) => { let ok = 0, n = 0; for (let k = 0; k < 40; k++) { const t = trial(L, rs); t.seq.forEach((s, i) => { ok += s === t.out[i] ? 1 : 0; n++; }); } return { L, acc: ok / n }; });
    const example = trial(len, mulberry32(len * 31 + d));
    return { curve, example };
  }, [d, len, attn]);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <div className="space-y-1">
            <div className="text-[12px] text-faint">Input sequence → copied back by the decoder (token ids 0–19)</div>
            <div className="flex flex-wrap gap-1">{res.example.seq.map((t, i) => <span key={i} className="num rounded border border-line bg-raised px-1.5 py-0.5 text-[12px]">{t}</span>)}</div>
            <div className="flex flex-wrap gap-1">{res.example.out.map((t, i) => <span key={i} className="num rounded px-1.5 py-0.5 text-[12px]" style={{ background: t === res.example.seq[i] ? "rgb(var(--green) / 0.22)" : "rgb(var(--red) / 0.3)" }}>{t}</span>)}</div>
          </div>
          <Plot x={[2, 40]} y={[0, 1.02]} aspect={0.4} maxH={260} xLabel="sequence length" yLabel="tokens recovered" ariaLabel="recovery accuracy against sequence length">
            <Polyline pts={res.curve.map((c) => [c.L, c.acc])} style={{ stroke: attn ? C.green : C.pink, strokeWidth: 2.2 }} />
            {res.curve.map((c, i) => <Dot key={i} p={[c.L, c.acc]} r={3.5} color={attn ? C.green : C.pink} />)}
          </Plot>
        </div>
      }
      controls={
        <>
          <ParameterSlider label="context vector size d" value={d} min={4} max={64} step={4} onChange={setD} />
          <ParameterSlider label="example length" value={len} min={2} max={40} step={1} onChange={setLen} />
          <Segmented label="Decoder reads" value={attn ? "attn" : "ctx"} onChange={(v) => setAttn(v === "attn")} options={[{ value: "ctx", label: "One context vector" }, { value: "attn", label: "All encoder states (attention)" }]} />
          <p className="text-[12px] leading-relaxed text-faint">A classic encoder–decoder squeezes the whole input into one fixed-size vector. Here the encoder adds position-coded token vectors into d numbers and the decoder tries to read each token back. Accuracy collapses once the sequence carries more information than d numbers can hold; attention removes the bottleneck.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: `\\text{recovered (this example)}`, tex: true, value: pct(res.example.seq.filter((s, i) => s === res.example.out[i]).length / res.example.seq.length) },
        { label: "\\text{context numbers}", tex: true, value: attn ? `${len} × ${d}` : d },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\mathbf c=\sum_{t=1}^{L}P_t\,\mathbf e_{x_t}\in\mathbb R^{${d}},\qquad\hat x_t=\arg\max_v\ \mathbf e_v^\top P_t^\top\mathbf c`, note: "a linear stand-in for an RNN encoder: other tokens add interference that grows with L" },
      ]} />}
    />
  );
}

/* ------------------------------ autoencoder (trained in the browser) ------------------------------ */
type Net = { W1: number[][]; b1: number[]; W2: number[]; b2: number; W3: number[]; b3: number[]; W4: number[][]; b4: number[] };
const HID = 8;
export function AutoencoderViz() {
  const [kind, setKind] = useState<"linear" | "nonlinear">("nonlinear");
  const [playing, setPlaying] = useState(true);
  const [epoch, setEpoch] = useState(0);
  const data = useMemo(() => { const r = mulberry32(5); return Array.from({ length: 120 }, () => { const t = Math.PI * (0.1 + 0.8 * r()); return [1.4 * Math.cos(t), 1.4 * Math.sin(t) - 0.6] as [number, number]; }).map(([x, y]) => [x + 0.04 * randn(r), y + 0.04 * randn(r)] as [number, number]); }, []);
  const init = (): Net => { const r = mulberry32(3); const g = () => randn(r) * 0.5; return { W1: Array.from({ length: HID }, () => [g(), g()]), b1: Array(HID).fill(0).map(g), W2: Array.from({ length: HID }, g), b2: 0, W3: Array.from({ length: HID }, g), b3: Array(HID).fill(0).map(g), W4: Array.from({ length: 2 }, () => Array.from({ length: HID }, g)), b4: [0, 0] }; };
  const net = useRef<Net>(init());
  const fwd = (n: Net, x: [number, number]) => {
    const a1 = n.W1.map((w, i) => Math.tanh(w[0] * x[0] + w[1] * x[1] + n.b1[i]));
    const z = n.W2.reduce((s, w, i) => s + w * a1[i], n.b2); // 1-D bottleneck (linear)
    const a3 = n.W3.map((w, i) => Math.tanh(w * z + n.b3[i]));
    const out = n.W4.map((w, k) => w.reduce((s, v, i) => s + v * a3[i], n.b4[k]));
    return { a1, z, a3, out };
  };
  // linear autoencoder optimum = projection onto the first principal component
  const pca = useMemo(() => { const mx = data.reduce((s, p) => s + p[0], 0) / data.length, my = data.reduce((s, p) => s + p[1], 0) / data.length; let sxx = 0, sxy = 0, syy = 0; data.forEach(([x, y]) => { sxx += (x - mx) ** 2; sxy += (x - mx) * (y - my); syy += (y - my) ** 2; }); const th = 0.5 * Math.atan2(2 * sxy, sxx - syy); return { mx, my, u: [Math.cos(th), Math.sin(th)] }; }, [data]);
  const recon = (x: [number, number]): [number, number] => {
    if (kind === "linear") { const t = (x[0] - pca.mx) * pca.u[0] + (x[1] - pca.my) * pca.u[1]; return [pca.mx + t * pca.u[0], pca.my + t * pca.u[1]]; }
    const o = fwd(net.current, x).out; return [o[0], o[1]];
  };
  const step = () => {
    const n = net.current, lr = 0.05;
    for (let rep = 0; rep < 20; rep++) {
      for (const x of data) {
        const { a1, z, a3, out } = fwd(n, x);
        const dOut = [2 * (out[0] - x[0]) / data.length * 30, 2 * (out[1] - x[1]) / data.length * 30];
        const dA3 = Array.from({ length: HID }, (_, i) => n.W4[0][i] * dOut[0] + n.W4[1][i] * dOut[1]);
        for (let k = 0; k < 2; k++) { for (let i = 0; i < HID; i++) n.W4[k][i] -= lr * dOut[k] * a3[i]; n.b4[k] -= lr * dOut[k]; }
        const dZ3 = dA3.map((g, i) => g * (1 - a3[i] ** 2));
        const dz = dZ3.reduce((s, g, i) => s + g * n.W3[i], 0);
        for (let i = 0; i < HID; i++) { n.W3[i] -= lr * dZ3[i] * z; n.b3[i] -= lr * dZ3[i]; }
        const dA1 = n.W2.map((w) => w * dz);
        for (let i = 0; i < HID; i++) n.W2[i] -= lr * dz * a1[i];
        n.b2 -= lr * dz;
        dA1.forEach((g, i) => { const d1 = g * (1 - a1[i] ** 2); n.W1[i][0] -= lr * d1 * x[0]; n.W1[i][1] -= lr * d1 * x[1]; n.b1[i] -= lr * d1; });
      }
    }
    setEpoch((e) => e + 20);
  };
  useTicker(playing && kind === "nonlinear" && epoch < 3000, 40, step);
  const mse = data.reduce((s, x) => { const rr = recon(x); return s + (rr[0] - x[0]) ** 2 + (rr[1] - x[1]) ** 2; }, 0) / data.length;
  const curve = kind === "linear"
    ? [[-2, pca.my + ((-2 - pca.mx) / pca.u[0]) * pca.u[1]], [2, pca.my + ((2 - pca.mx) / pca.u[0]) * pca.u[1]]] as [number, number][]
    : Array.from({ length: 80 }, (_, i) => { const z = -6 + (12 * i) / 79; const n = net.current; const a3 = n.W3.map((w, k) => Math.tanh(w * z + n.b3[k])); return n.W4.map((w, k) => w.reduce((s, v, q) => s + v * a3[q], n.b4[k])) as [number, number]; });
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-2, 2]} y={[-1.2, 1.4]} equal aspect={0.6} maxH={400} ariaLabel="autoencoder reconstructions">
            {data.map((x, i) => { const rr = recon(x); return <Polyline key={"l" + i} pts={[x, rr]} style={{ stroke: C.a("fg", 0.2), strokeWidth: 1 }} />; })}
            <Polyline pts={curve} style={{ stroke: C.pink, strokeWidth: 2.2 }} />
            {data.map((x, i) => <Dot key={i} p={x} r={2.8} color={C.blue} />)}
          </Plot>
          <Legend items={[{ label: "data (2-D points on a curved 1-D manifold)", color: C.blue }, { label: "everything the decoder can output (1-D code swept)", color: C.pink }, { label: "reconstruction error", color: C.a("fg", 0.35) }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Autoencoder" value={kind} onChange={setKind} options={[{ value: "linear", label: "Linear (= PCA)" }, { value: "nonlinear", label: "2-8-1-8-2 tanh" }]} />
          {kind === "nonlinear" && <AnimationControls playing={playing} onPlay={() => setPlaying((p) => !p)} onStep={step} onReset={() => { net.current = init(); setEpoch(0); }} />}
          <p className="text-[12px] leading-relaxed text-faint">The encoder compresses each 2-D point into a single number z; the decoder maps z back to 2-D. A linear autoencoder can only reconstruct onto a straight line (the first principal component). A non-linear one, trained here by backpropagation in your browser, learns to follow the arc.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{reconstruction MSE}", tex: true, value: fmt(mse, 4), color: C.pink },
        { label: "\\text{epochs}", tex: true, value: kind === "linear" ? "closed form" : epoch },
        { label: "\\text{code size}", tex: true, value: "1 (from 2)" },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`z=g_\phi(\mathbf x),\quad\hat{\mathbf x}=f_\theta(z),\quad\min_{\phi,\theta}\frac1n\sum_i\|\mathbf x_i-f_\theta(g_\phi(\mathbf x_i))\|^2` }]} />}
    />
  );
}

/* ------------------------------ VAE: the linear-Gaussian case solved exactly ------------------------------ */
export function VAEViz() {
  const [sig2, setSig2] = useState(0.1);
  const [px, setPx] = useState<[number, number]>([1.5, 1]);
  const r0 = useMemo(() => mulberry32(6), []);
  const data = useMemo(() => Array.from({ length: 150 }, () => { const z = randn(r0), e1 = randn(r0), e2 = randn(r0); return [1.6 * z * 0.85 + 0.35 * e1, 1.6 * z * 0.53 + 0.35 * e2] as [number, number]; }), [r0]);
  const fit = useMemo(() => {
    // sample covariance → PPCA: W = u1 sqrt(λ1 − σ²) maximises the ELBO for a linear decoder with noise σ² (q is exact)
    const n = data.length, mx = data.reduce((s, p) => s + p[0], 0) / n, my = data.reduce((s, p) => s + p[1], 0) / n;
    let a = 0, b = 0, c = 0; data.forEach(([x, y]) => { a += (x - mx) ** 2 / n; b += (x - mx) * (y - my) / n; c += (y - my) ** 2 / n; });
    const tr = a + c, det = a * c - b * b, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr - l1;
    const u = Math.abs(b) > 1e-12 ? [l1 - c, b] : [1, 0]; const un = Math.hypot(u[0], u[1]); const u1 = [u[0] / un, u[1] / un];
    const w = Math.sqrt(Math.max(1e-6, l1 - sig2));
    const W = [u1[0] * w, u1[1] * w];
    return { mx, my, W, l1, l2 };
  }, [data, sig2]);
  const { W } = fit;
  const M = W[0] ** 2 + W[1] ** 2 + sig2; // posterior: q(z|x) = N(Wᵀ(x−μ)/M, σ²/M)
  const post = (x: [number, number]) => ({ m: (W[0] * (x[0] - fit.mx) + W[1] * (x[1] - fit.my)) / M, s2: sig2 / M });
  const q = post(px);
  // ELBO terms averaged over data (closed form for Gaussians)
  const elbo = useMemo(() => {
    let rec = 0, kl = 0;
    data.forEach((x) => {
      const { m, s2 } = post(x);
      const dx = x[0] - fit.mx - W[0] * m, dy = x[1] - fit.my - W[1] * m;
      rec += -Math.log(2 * Math.PI * sig2) - (dx * dx + dy * dy + (W[0] ** 2 + W[1] ** 2) * s2) / (2 * sig2);
      kl += 0.5 * (m * m + s2 - 1 - Math.log(s2));
    });
    return { rec: rec / data.length, kl: kl / data.length };
  }, [data, sig2, W]); // eslint-disable-line react-hooks/exhaustive-deps
  const gen = useMemo(() => { const r = mulberry32(12); return Array.from({ length: 150 }, () => { const z = randn(r); return [fit.mx + W[0] * z + Math.sqrt(sig2) * randn(r), fit.my + W[1] * z + Math.sqrt(sig2) * randn(r)] as [number, number]; }); }, [W, sig2, fit]);
  const zs = Array.from({ length: 121 }, (_, i) => -3 + (6 * i) / 120);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[-4, 4]} y={[-3, 3]} equal aspect={0.6} maxH={380} ariaLabel="data, decoder line and generated samples" onPointerDown={(p) => setPx(p)}>
            {data.map((p, i) => <Dot key={i} p={p} r={2.5} color={C.blue} opacity={0.6} />)}
            {gen.map((p, i) => <Dot key={"g" + i} p={p} r={2.2} color={C.pink} opacity={0.55} />)}
            <Polyline pts={[[fit.mx - 3 * W[0], fit.my - 3 * W[1]], [fit.mx + 3 * W[0], fit.my + 3 * W[1]]]} style={{ stroke: C.green, strokeWidth: 2 }} />
            <Dot p={px} r={6} color={C.fg} stroke={C.bg} />
            <Polyline pts={[px, [fit.mx + W[0] * q.m, fit.my + W[1] * q.m]]} style={{ stroke: C.fg, strokeDasharray: "3 3" }} />
          </Plot>
          <Plot x={[-3, 3]} y={[0, Math.max(1.2, 1 / Math.sqrt(2 * Math.PI * q.s2)) * 1.05]} aspect={0.22} minH={120} maxH={160} xLabel="latent z" ariaLabel="prior and posterior over the latent variable">
            <FnPath f={(z) => gauss(z, 0, 1)} style={{ stroke: C.a("fg", 0.5), strokeDasharray: "4 3" }} />
            <Polyline pts={zs.map((z) => [z, gauss(z, q.m, Math.sqrt(q.s2))])} style={{ stroke: C.amber, strokeWidth: 2 }} />
          </Plot>
          <Legend items={[{ label: "data", color: C.blue }, { label: "samples: z ~ N(0,1) → decoder + noise", color: C.pink }, { label: "decoder mean μ + W z", color: C.green }, { label: "encoder q(z | clicked x)", color: C.amber }, { label: "prior p(z)", color: C.a("fg", 0.5), dashed: true }]} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="decoder noise σ²" value={sig2} min={0.01} max={Math.max(0.02, fit.l1 * 0.95)} step={0.01} onChange={setSig2} />
          <p className="text-[12px] leading-relaxed text-faint">Click anywhere to encode that point. With a linear decoder and Gaussian noise the best VAE is known exactly (probabilistic PCA), so every number here is computed in closed form. Small σ² means sharp reconstructions but a narrow, confident encoder; large σ² explains more of the data as noise, and generated samples become blurrier.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "q(z\\mid\\mathbf x)", tex: true, value: `N(${fmt(q.m, 3)}, ${fmt(q.s2, 4)})`, color: C.amber },
        { label: "\\mathbb E_q[\\log p(\\mathbf x\\mid z)]", tex: true, value: fmt(elbo.rec, 3) },
        { label: "\\mathrm{KL}(q\\,\\|\\,p)", tex: true, value: fmt(elbo.kl, 3) },
        { label: "\\text{ELBO}", tex: true, value: fmt(elbo.rec - elbo.kl, 3), color: C.green },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\log p(\mathbf x)\ \ge\ \mathbb E_{q(z|\mathbf x)}[\log p(\mathbf x\mid z)]-\mathrm{KL}\big(q(z\mid\mathbf x)\,\|\,\mathcal N(0,1)\big)` },
        { tex: String.raw`p(\mathbf x\mid z)=\mathcal N(\boldsymbol\mu+\mathbf wz,\ ${tex(sig2, 2)}I),\quad q(z\mid\mathbf x)=\mathcal N\Big(\frac{\mathbf w^\top(\mathbf x-\boldsymbol\mu)}{\|\mathbf w\|^2+\sigma^2},\ \frac{\sigma^2}{\|\mathbf w\|^2+\sigma^2}\Big)`, note: "reparameterisation in a real VAE: z = m + s·ε, ε ~ N(0, 1), so gradients flow through m and s" },
      ]} />}
    />
  );
}

/* ------------------------------ GAN in one dimension ------------------------------ */
export function GANViz() {
  const [target, setTarget] = useState<"uni" | "bi">("uni");
  const [playing, setPlaying] = useState(true);
  const [it, setIt] = useState(0);
  const st = useRef({ m: -2, s: 0.4, a: 0, b: 0, c: 0 });
  const hist = useRef<number[]>([]);
  const real = (r: () => number) => (target === "uni" ? 2 + 0.6 * randn(r) : (r() < 0.5 ? -1.5 : 2.2) + 0.35 * randn(r));
  const reset = () => { st.current = { m: -2, s: 0.4, a: 0, b: 0, c: 0 }; hist.current = []; setIt(0); };
  const D = (x: number) => sigmoid(st.current.a * x * x / 4 + st.current.b * x + st.current.c); // quadratic logistic discriminator
  const step = () => {
    const r = mulberry32(it + 1), p = st.current, lrD = 0.05, lrG = 0.03;
    for (let rep = 0; rep < 5; rep++) {
      // discriminator: ascend log D(real) + log(1 − D(fake))
      let ga = 0, gb = 0, gc = 0;
      for (let k = 0; k < 64; k++) {
        const xr = real(r), xf = p.m + p.s * randn(r);
        const dr = D(xr), df = D(xf);
        ga += (1 - dr) * xr * xr / 4 - df * xf * xf / 4; gb += (1 - dr) * xr - df * xf; gc += (1 - dr) - df;
      }
      p.a += lrD * ga / 64; p.b += lrD * gb / 64; p.c += lrD * gc / 64;
      // generator (non-saturating): ascend log D(G(z))
      let gm = 0, gs = 0;
      for (let k = 0; k < 64; k++) {
        const z = randn(r), x = p.m + p.s * z, d = D(x);
        const dlogD = (1 - d) * (p.a * x / 2 + p.b);
        gm += dlogD; gs += dlogD * z;
      }
      p.m += lrG * gm / 64; p.s = Math.max(0.05, p.s + lrG * gs / 64);
    }
    hist.current.push(p.m);
    setIt((i) => i + 1);
  };
  useTicker(playing && it < 600, 30, step);
  const samplesR = useMemo(() => { const r = mulberry32(77); return Array.from({ length: 3000 }, () => real(r)); }, [target]); // eslint-disable-line react-hooks/exhaustive-deps
  const samplesF = useMemo(() => { const r = mulberry32(78); return Array.from({ length: 3000 }, () => st.current.m + st.current.s * randn(r)); }, [it]); // eslint-disable-line react-hooks/exhaustive-deps
  const hr = histogram(samplesR, -4, 5, 60), hf = histogram(samplesF, -4, 5, 60);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-4, 5]} y={[0, 1.2]} aspect={0.45} maxH={340} xLabel="x" ariaLabel="real and generated distributions with the discriminator">
            <Bars bins={hr} color={C.a("blue", 0.3)} stroke={C.blue} />
            <Bars bins={hf} color={C.a("pink", 0.3)} stroke={C.pink} />
            <FnPath f={D} style={{ stroke: C.green, strokeWidth: 2 }} />
          </Plot>
          <Legend items={[{ label: "real data", color: C.blue }, { label: "generator samples G(z) = m + s·z", color: C.pink }, { label: "discriminator D(x) = P(real)", color: C.green }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Real data" value={target} onChange={(v) => { setTarget(v); reset(); }} options={[{ value: "uni", label: "One mode" }, { value: "bi", label: "Two modes" }]} />
          <AnimationControls playing={playing} onPlay={() => setPlaying((p) => !p)} onStep={step} onReset={reset} />
          <p className="text-[12px] leading-relaxed text-faint">The discriminator learns where real data is denser than fake; the generator follows the discriminator&apos;s gradient towards &quot;more real&quot;. At equilibrium D(x) ≈ 0.5 everywhere. With two real modes this tiny generator can produce only one bump, so it settles on a compromise or one mode — a miniature of mode collapse.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{step}", tex: true, value: it },
        { label: "m,\\ s", tex: true, value: `${fmt(st.current.m, 3)}, ${fmt(st.current.s, 3)}`, hint: target === "uni" ? "target 2, 0.6" : "target: two modes" },
        { label: "D(m)", tex: true, value: fmt(D(st.current.m), 3) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`\min_G\max_D\ \mathbb E_{x\sim p_{\text{data}}}[\log D(x)]+\mathbb E_{z}[\log(1-D(G(z)))]` },
        { tex: String.raw`\text{generator uses the non-saturating loss }-\mathbb E_z[\log D(G(z))]`, note: "5 alternating gradient steps per frame, batch 64" },
      ]} />}
    />
  );
}

/* ------------------------------ diffusion with an exact score ------------------------------ */
const MIX = [{ w: 0.35, m: -2, s: 0.35 }, { w: 0.65, m: 1.5, s: 0.5 }];
export function DiffusionViz() {
  const [t, setT] = useState(0);
  const [phase, setPhase] = useState<"forward" | "reverse">("forward");
  const [playing, setPlaying] = useState(false);
  const Tn = 100;
  const betas = Array.from({ length: Tn }, (_, i) => 1e-4 + ((0.05 - 1e-4) * i) / (Tn - 1));
  const abar = betas.reduce<number[]>((acc, b) => [...acc, (acc.length ? acc[acc.length - 1] : 1) * (1 - b)], []);
  const ab = (k: number) => (k <= 0 ? 1 : abar[k - 1]);
  // q(x_t) for a Gaussian-mixture x_0 is again a mixture: means √ᾱ m, variances ᾱ s² + 1 − ᾱ
  const pt = (x: number, k: number) => MIX.reduce((s, c) => s + c.w * gauss(x, Math.sqrt(ab(k)) * c.m, Math.sqrt(ab(k) * c.s * c.s + 1 - ab(k))), 0);
  const score = (x: number, k: number) => { let num = 0, den = 0; MIX.forEach((c) => { const m = Math.sqrt(ab(k)) * c.m, v = ab(k) * c.s * c.s + 1 - ab(k); const p = c.w * gauss(x, m, Math.sqrt(v)); num += p * (-(x - m) / v); den += p; }); return num / den; };
  const x0 = useMemo(() => { const r = mulberry32(3); return Array.from({ length: 2000 }, () => { const c = r() < MIX[0].w ? MIX[0] : MIX[1]; return c.m + c.s * randn(r); }); }, []);
  const fwd = useMemo(() => { const r = mulberry32(4); const eps = x0.map(() => randn(r)); return (k: number) => x0.map((x, i) => Math.sqrt(ab(k)) * x + Math.sqrt(1 - ab(k)) * eps[i]); }, [x0]); // eslint-disable-line react-hooks/exhaustive-deps
  // reverse (DDPM ancestral sampling) using the exact score of q(x_t)
  const rev = useMemo(() => {
    const r = mulberry32(5);
    let x = Array.from({ length: 2000 }, () => randn(r));
    const snaps: number[][] = Array(Tn + 1);
    snaps[Tn] = x.slice();
    for (let k = Tn; k >= 1; k--) {
      const b = betas[k - 1];
      x = x.map((v) => (v + b * score(v, k)) / Math.sqrt(1 - b) + (k > 1 ? Math.sqrt(b) * randn(r) : 0));
      snaps[k - 1] = x.slice();
    }
    return snaps;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useTicker(playing, 60, () => { if (phase === "forward") setT((k) => (k >= Tn ? (setPlaying(false), k) : k + 1)); else setT((k) => (k <= 0 ? (setPlaying(false), k) : k - 1)); });
  const xs = phase === "forward" ? fwd(t) : rev[t];
  const bins = histogram(xs, -4.5, 4.5, 60);
  return (
    <LabShell
      viz={
        <div className="space-y-2">
          <Plot x={[-4.5, 4.5]} y={[0, 0.65]} aspect={0.45} maxH={340} xLabel={`x at step t = ${t}`} ariaLabel="distribution of samples during diffusion">
            <Bars bins={bins} color={phase === "forward" ? C.a("blue", 0.3) : C.a("pink", 0.3)} stroke={phase === "forward" ? C.blue : C.pink} />
            <FnPath f={(x) => pt(x, t)} n={300} style={{ stroke: C.green, strokeWidth: 2 }} />
            <FnPath f={(x) => pt(x, 0)} n={300} style={{ stroke: C.a("fg", 0.35), strokeDasharray: "4 3" }} />
          </Plot>
          <Legend items={[{ label: phase === "forward" ? "noised data x_t" : "samples being denoised", color: phase === "forward" ? C.blue : C.pink }, { label: "exact density q(x_t)", color: C.green }, { label: "data density q(x_0)", color: C.a("fg", 0.45), dashed: true }]} />
        </div>
      }
      controls={
        <>
          <Segmented label="Process" value={phase} onChange={(v) => { setPhase(v); setT(v === "forward" ? 0 : Tn); setPlaying(false); }} options={[{ value: "forward", label: "Forward: add noise" }, { value: "reverse", label: "Reverse: generate" }]} />
          <ParameterSlider label="step t" value={t} min={0} max={Tn} step={1} onChange={setT} />
          <AnimationControls playing={playing} onPlay={() => setPlaying((p) => !p)} onReset={() => { setPlaying(false); setT(phase === "forward" ? 0 : Tn); }} />
          <p className="text-[12px] leading-relaxed text-faint">Forward: a little Gaussian noise is added at every step until the data is pure noise. Reverse: start from noise and step backwards, each step nudging samples up the score ∇ log q(x_t). A real diffusion model learns that score with a neural network; for this 1-D mixture it is computed exactly, so the reverse process recovers the two bumps.</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\bar\\alpha_t", tex: true, value: fmt(ab(t), 4) },
        { label: "\\text{signal }\\sqrt{\\bar\\alpha_t}", tex: true, value: fmt(Math.sqrt(ab(t)), 3) },
        { label: "\\text{noise }\\sqrt{1-\\bar\\alpha_t}", tex: true, value: fmt(Math.sqrt(1 - ab(t)), 3) },
      ]} />}
      calc={<CalcSteps steps={[
        { tex: String.raw`q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\big),\quad\bar\alpha_t=\prod_{s\le t}(1-\beta_s)` },
        { tex: String.raw`x_{t-1}=\frac{x_t+\beta_t\nabla_x\log q(x_t)}{\sqrt{1-\beta_t}}+\sqrt{\beta_t}\,z`, note: "learned models predict ε instead: ∇ log q = −ε / √(1 − ᾱ)" },
      ]} />}
    />
  );
}

/* ------------------------------ U-Net: why the skip connections ------------------------------ */
export function UNetViz() {
  const [depth, setDepth] = useState(3);
  const [skip, setSkip] = useState(true);
  const [size, setSize] = useState(128);
  const [base, setBase] = useState(32);
  const N = 64;
  const signal = useMemo(() => Array.from({ length: N }, (_, i) => (i > 10 && i < 22 ? 1 : 0) + (i > 34 && i < 37 ? 0.8 : 0) + (i > 46 ? 0.4 : 0) + 0.15 * Math.sin(i * 1.3)), []);
  // encoder: average-pool by 2 at each level; decoder: nearest upsample; skip adds back the detail lost at that level
  const enc: number[][] = [signal];
  for (let l = 0; l < depth; l++) { const p = enc[l]; enc.push(Array.from({ length: p.length / 2 }, (_, i) => (p[2 * i] + p[2 * i + 1]) / 2)); }
  let dec = enc[depth];
  for (let l = depth - 1; l >= 0; l--) {
    const up = enc[l].map((_, i) => dec[Math.floor(i / 2)]);
    dec = skip ? up.map((v, i) => v + (enc[l][i] - enc[l + 1][Math.floor(i / 2)])) : up;
  }
  const err = Math.sqrt(signal.reduce((s, v, i) => s + (v - dec[i]) ** 2, 0) / N);
  const rows = Array.from({ length: depth + 1 }, (_, l) => [l === depth ? "bottleneck" : `level ${l + 1}`, `${size >> l}×${size >> l}×${base << l}`, l < depth ? (skip ? "→ copied to decoder" : "—") : ""]);
  return (
    <LabShell
      viz={
        <div className="space-y-3">
          <Plot x={[0, N - 1]} y={[-0.4, 1.4]} aspect={0.35} maxH={260} xLabel="pixel along one row" ariaLabel="input row and decoder output">
            <Polyline pts={signal.map((v, i) => [i, v])} style={{ stroke: C.a("fg", 0.45), strokeWidth: 1.5 }} />
            <Polyline pts={enc[depth].map((v, i) => [i * (1 << depth) + (1 << depth) / 2 - 0.5, v])} style={{ stroke: C.amber, strokeWidth: 1.5, strokeDasharray: "4 3" }} />
            <Polyline pts={dec.map((v, i) => [i, v])} style={{ stroke: skip ? C.green : C.pink, strokeWidth: 2.2 }} />
          </Plot>
          <Legend items={[{ label: "input", color: C.a("fg", 0.5) }, { label: "bottleneck (coarse)", color: C.amber, dashed: true }, { label: skip ? "decoder output with skips" : "decoder output without skips", color: skip ? C.green : C.pink }]} />
          <MiniTable head={["encoder level", "feature map (H×W×C)", "skip"]} rows={rows} />
        </div>
      }
      controls={
        <>
          <ParameterSlider label="depth (downsampling levels)" value={depth} min={1} max={5} step={1} onChange={setDepth} />
          <Segmented label="Skip connections" value={skip ? "on" : "off"} onChange={(v) => setSkip(v === "on")} options={[{ value: "on", label: "On" }, { value: "off", label: "Off" }]} />
          <ParameterSlider label="input size" value={size} min={64} max={512} step={64} onChange={setSize} />
          <ParameterSlider label="base channels" value={base} min={8} max={64} step={8} onChange={setBase} />
          <p className="text-[12px] leading-relaxed text-faint">The encoder keeps halving resolution to gather context, so fine detail is lost by the bottleneck. Skip connections hand each decoder level the encoder features of the same resolution, so it can place sharp boundaries again. (Illustration with fixed pooling, not a trained network.)</p>
        </>
      }
      readouts={<Readouts items={[
        { label: "\\text{RMSE vs input}", tex: true, value: fmt(err, 4), color: skip ? C.green : C.pink },
        { label: "\\text{bottleneck}", tex: true, value: `${size >> depth}×${size >> depth}×${base << depth}` },
      ]} />}
      calc={<CalcSteps steps={[{ tex: String.raw`\mathbf d_l=\operatorname{up}(\mathbf d_{l+1})\ \oplus\ \mathbf e_l\quad(\oplus=\text{concatenate, then convolve})`, note: "here ⊕ adds the detail eₗ − up(eₗ₊₁), so reconstruction with skips is exact" }]} />}
    />
  );
}
