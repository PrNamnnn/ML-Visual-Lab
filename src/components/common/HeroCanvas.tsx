"use client";
import { useEffect, useRef } from "react";

/**
 * Lightweight hero background: a grid continuously transformed by a slowly rotating
 * 2×2 matrix, with its basis vectors and the unit-circle → ellipse image. Canvas 2D only;
 * pauses when off-screen / tab hidden and renders a single static frame for reduced motion.
 */
export function HeroCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const g = c.getContext("2d");
    if (!g) return;
    let raf = 0;
    let visible = true;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      c.width = c.clientWidth * dpr;
      c.height = c.clientHeight * dpr;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(c);
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting));
    io.observe(c);
    const css = getComputedStyle(document.documentElement);
    const col = (n: string, a: number) => `rgb(${css.getPropertyValue(`--${n}`).trim().split(/\s+/).join(",")},${a})`;
    const draw = (t: number) => {
      const w = c.clientWidth, h = c.clientHeight;
      g.clearRect(0, 0, w, h);
      const s = Math.min(w, h) / 9;
      const cx = w * 0.72, cy = h * 0.5;
      const th = t * 0.00012;
      const A = [[1.2 * Math.cos(th) + 0.35, -Math.sin(th) * 0.9], [Math.sin(th) * 0.8, 1 + 0.25 * Math.cos(th * 1.7)]];
      const P = (x: number, y: number) => [cx + s * (A[0][0] * x + A[0][1] * y), cy - s * (A[1][0] * x + A[1][1] * y)];
      g.lineWidth = 1;
      for (let k = -10; k <= 10; k++) {
        g.strokeStyle = col("blue", k === 0 ? 0.25 : 0.07);
        let a = P(k, -10), b = P(k, 10);
        g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
        a = P(-10, k); b = P(10, k);
        g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
      }
      g.strokeStyle = col("fg", 0.25);
      g.beginPath();
      for (let i = 0; i <= 80; i++) { const a = (i / 80) * Math.PI * 2; const p = P(Math.cos(a) * 1.5, Math.sin(a) * 1.5); if (i) g.lineTo(p[0], p[1]); else g.moveTo(p[0], p[1]); }
      g.stroke();
      const arrow = (x: number, y: number, color: string) => {
        const o = P(0, 0), p = P(x, y);
        g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2;
        g.beginPath(); g.moveTo(o[0], o[1]); g.lineTo(p[0], p[1]); g.stroke();
        const ang = Math.atan2(p[1] - o[1], p[0] - o[0]);
        g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(p[0] - 9 * Math.cos(ang - 0.4), p[1] - 9 * Math.sin(ang - 0.4)); g.lineTo(p[0] - 9 * Math.cos(ang + 0.4), p[1] - 9 * Math.sin(ang + 0.4)); g.fill();
      };
      arrow(2, 0, col("blue", 0.75));
      arrow(0, 2, col("pink", 0.75));
    };
    const loop = (t: number) => {
      if (visible && !document.hidden) draw(t);
      raf = requestAnimationFrame(loop);
    };
    if (reduce) draw(0);
    else raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); };
  }, []);
  return <canvas ref={ref} aria-hidden className="pointer-events-none absolute inset-0 h-full w-full [mask-image:linear-gradient(to_right,transparent_10%,black_55%)]" />;
}
