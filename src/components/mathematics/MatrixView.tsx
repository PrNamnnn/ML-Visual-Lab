"use client";
import { cn, fmt } from "@/lib/utilities/format";

/** Diverging colour for a value relative to a max magnitude: pink (−) ↔ blue (+). */
export function divergingBg(v: number, max: number) {
  if (!Number.isFinite(v) || max <= 0) return "transparent";
  const a = Math.min(1, Math.abs(v) / max) * 0.55;
  return v >= 0 ? `rgb(var(--blue) / ${a})` : `rgb(var(--pink) / ${a})`;
}

export function sequentialBg(v: number, max: number) {
  if (!Number.isFinite(v) || max <= 0) return "transparent";
  return `rgb(var(--blue) / ${Math.min(1, Math.max(0, v / max)) * 0.7})`;
}

/**
 * Matrix rendered as a heatmap grid with bracket decoration.
 * `hl(i,j)` returns a highlight style key for cells.
 */
export function MatrixView({
  M, label, digits = 2, hl, cell = 40, scaleMax, heat = true, faded, onHover, sequential = false, title,
}: {
  M: number[][]; label?: React.ReactNode; digits?: number; hl?: (i: number, j: number) => "row" | "col" | "out" | "pivot" | null;
  cell?: number; scaleMax?: number; heat?: boolean; faded?: (i: number, j: number) => boolean; onHover?: (i: number, j: number) => void;
  sequential?: boolean; title?: string;
}) {
  const max = scaleMax ?? Math.max(1e-9, ...M.flat().map((x) => (Number.isFinite(x) ? Math.abs(x) : 0)));
  const cols = M[0]?.length ?? 0;
  return (
    <div className="inline-flex flex-col items-center gap-1" title={title}>
      {label && <div className="text-[12px] text-muted">{label}</div>}
      <div className="relative px-1.5">
        <span className="pointer-events-none absolute inset-y-0 left-0 w-1 rounded-l-sm border-y border-l border-line-strong" />
        <span className="pointer-events-none absolute inset-y-0 right-0 w-1 rounded-r-sm border-y border-r border-line-strong" />
        <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${cols}, ${cell}px)` }} onMouseLeave={() => onHover?.(-1, -1)}>
          {M.map((r, i) =>
            r.map((v, j) => {
              const h = hl?.(i, j);
              return (
                <div
                  key={`${i}-${j}`}
                  onMouseEnter={() => onHover?.(i, j)}
                  className={cn(
                    "num flex items-center justify-center rounded-[3px] text-[11.5px] transition-colors",
                    h === "row" && "ring-1 ring-blue",
                    h === "col" && "ring-1 ring-pink",
                    h === "out" && "ring-2 ring-green",
                    h === "pivot" && "ring-2 ring-amber",
                    faded?.(i, j) && "opacity-35",
                  )}
                  style={{ height: Math.round(cell * 0.78), background: heat ? (sequential ? sequentialBg(v, max) : divergingBg(v, max)) : "rgb(var(--raised))" }}
                >
                  {Number.isNaN(v) ? "—" : fmt(v, digits)}
                </div>
              );
            }),
          )}
        </div>
      </div>
    </div>
  );
}
