"use client";
import { useEffect, useRef, useState } from "react";
import { useThemeRGB } from "./Plot";
import { LoadingState } from "./LoadingState";

type PlotlyModule = {
  react: (el: HTMLElement, data: unknown[], layout: unknown, config?: unknown) => Promise<unknown>;
  purge: (el: HTMLElement) => void;
};

let plotlyPromise: Promise<PlotlyModule> | null = null;
/** Plotly is ~1 MB gzipped: loaded only when a Plotly chart actually mounts. */
const loadPlotly = () =>
  (plotlyPromise ??= import("plotly.js-dist-min")
    .then((m) => (m as unknown as { default: PlotlyModule }).default ?? (m as unknown as PlotlyModule))
    .catch((e) => { plotlyPromise = null; throw e; })); // allow a retry after a failed download

export function PlotlyChart({ data, layout, height = 440, ariaLabel }: { data: unknown[]; layout?: Record<string, unknown>; height?: number; ariaLabel: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const th = useThemeRGB();
  const fg = `rgb(${th.muted.join(",")})`;
  const grid = `rgba(${th.fg.join(",")},0.12)`;

  useEffect(() => {
    let alive = true;
    const el = ref.current;
    loadPlotly().then((P) => {
      if (!alive || !el) return;
      const axis = { gridcolor: grid, zerolinecolor: grid, color: fg, backgroundcolor: "rgba(0,0,0,0)", showbackground: false };
      P.react(
        el,
        data,
        {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: fg, size: 11, family: "Inter, Segoe UI, sans-serif" },
          margin: { l: 0, r: 0, t: 0, b: 0 },
          showlegend: false,
          ...layout,
          scene: { xaxis: axis, yaxis: axis, zaxis: axis, camera: { eye: { x: 1.5, y: -1.6, z: 1.1 } }, aspectmode: "cube", ...(layout?.scene as object) },
        },
        { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["toImage", "resetCameraLastSave3d"] },
      ).then(() => alive && setReady(true));
    }).catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, [data, layout, fg, grid, attempt]);

  useEffect(() => {
    const el = ref.current;
    return () => { if (el) loadPlotly().then((P) => P.purge(el)); };
  }, []);

  return (
    <div className="relative" style={{ height }} role="img" aria-label={ariaLabel}>
      {!ready && !failed && <LoadingState label="Loading mathematical model…" className="absolute inset-0" />}
      {failed && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-[12.5px] text-faint">
          The 3D plotting engine could not be loaded.
          <button className="btn" onClick={() => { setFailed(false); setAttempt((a) => a + 1); }}>Try again</button>
        </div>
      )}
      <div ref={ref} className="h-full w-full" />
    </div>
  );
}
