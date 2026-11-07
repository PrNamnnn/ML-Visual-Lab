"use client";
import type { ReactNode } from "react";
import { Tex } from "@/components/mathematics/Tex";
import { cn } from "@/lib/utilities/format";

/**
 * Standard layout for every visualizer: main canvas, a controls column, live readouts
 * and an optional live calculation strip. Collapses to one column on small screens.
 */
export function LabShell({
  viz, controls, readouts, calc, note, wideControls = false,
}: { viz: ReactNode; controls?: ReactNode; readouts?: ReactNode; calc?: ReactNode; note?: ReactNode; wideControls?: boolean }) {
  return (
    <div className="panel overflow-hidden">
      <div className={cn("grid", !!controls && (wideControls ? "lg:grid-cols-[minmax(0,1fr)_320px]" : "lg:grid-cols-[minmax(0,1fr)_272px]"))}>
        <div className="min-w-0 p-3 sm:p-4">{viz}</div>
        {controls && <div className="min-w-0 space-y-5 border-t border-line p-4 lg:border-l lg:border-t-0">{controls}</div>}
      </div>
      {readouts && <div className="border-t border-line bg-raised/40 px-4 py-3">{readouts}</div>}
      {calc && <div className="border-t border-line px-4 py-3">{calc}</div>}
      {note && <div className="border-t border-line px-4 py-2.5 text-[12.5px] leading-relaxed text-muted">{note}</div>}
    </div>
  );
}

export function Readouts({ items }: { items: { label: string; value: ReactNode; tex?: boolean; color?: string; hint?: string }[] }) {
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 xl:grid-cols-4">
      {items.map((it) => (
        <div key={it.label} className="min-w-0" title={it.hint}>
          <dt className="truncate text-[11.5px] text-faint">{it.tex ? <Tex tex={it.label} /> : it.label}</dt>
          <dd className="num truncate text-[13.5px]" style={{ color: it.color }}>{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function CalcSteps({ title = "Live calculation", steps }: { title?: string; steps: { tex: string; note?: string }[] }) {
  return (
    <div className="space-y-2">
      <div className="label">{title}</div>
      <ol className="space-y-1.5">
        {steps.map((s, i) => (
          <li key={i} className="flex min-w-0 flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-3">
            <span className="num w-5 shrink-0 text-[11px] text-faint">{i + 1}</span>
            <div className="min-w-0 overflow-x-auto text-[13px]"><Tex tex={s.tex} /></div>
            {s.note && <span className="text-[12px] text-faint">{s.note}</span>}
          </li>
        ))}
      </ol>
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11.5px] text-muted">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <span className="inline-block h-[2px] w-4" style={{ background: i.dashed ? `repeating-linear-gradient(90deg, ${i.color} 0 4px, transparent 4px 7px)` : i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export function Warn({ children }: { children: ReactNode }) {
  return <div className="rounded-md border border-amber/30 bg-amber/10 px-3 py-2 text-[12.5px] text-amber">{children}</div>;
}
