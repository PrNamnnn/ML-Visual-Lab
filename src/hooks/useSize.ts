"use client";
import { useEffect, useRef, useState } from "react";

/** Observe an element's content-box width; height derived from aspect ratio. */
export function useSize<T extends HTMLElement>(aspect = 0.62, minH = 220, maxH = 560) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(640);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const cw = Math.floor(entries[0].contentRect.width);
      if (cw > 0) setW(cw);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const h = Math.max(minH, Math.min(maxH, Math.round(w * aspect)));
  return { ref, width: w, height: h };
}

export function usePrefersReducedMotion() {
  const [r, setR] = useState(false);
  useEffect(() => {
    const m = window.matchMedia("(prefers-reduced-motion: reduce)");
    setR(m.matches);
    const on = () => setR(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return r;
}

/**
 * requestAnimationFrame-driven ticker. Calls `onTick` at most every `intervalMs`
 * while `running`; pauses automatically when the tab is hidden.
 */
export function useTicker(running: boolean, intervalMs: number, onTick: () => void) {
  const cb = useRef(onTick);
  cb.current = onTick;
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = performance.now();
    const loop = (t: number) => {
      if (!document.hidden && t - last >= intervalMs) {
        last = t;
        cb.current();
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, intervalMs]);
}
