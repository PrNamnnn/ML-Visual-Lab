"use client";
import { useId, type ReactNode } from "react";
import { Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { Tex } from "@/components/mathematics/Tex";
import { cn, fmt } from "@/lib/utilities/format";
export { Select, Segmented } from "./basic";

export function ParameterSlider({
  label, value, min, max, step, onChange, tex = false, format, disabled,
}: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
  tex?: boolean; format?: (v: number) => string; disabled?: boolean;
}) {
  const id = useId();
  const fill = ((value - min) / (max - min || 1)) * 100;
  return (
    <div className={cn("space-y-1", disabled && "opacity-40")}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-[12.5px] text-muted">
          {tex ? <Tex tex={label} /> : label}
        </label>
        <span className="num text-[12px] text-fg">{format ? format(value) : fmt(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        style={{ ["--fill" as string]: `${fill}%` }}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}



export function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 text-[12.5px] text-muted">
      <span>{label}</span>
      <button
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={cn("relative h-[18px] w-8 shrink-0 rounded-full transition-colors", checked ? "bg-accent" : "bg-line-strong")}
      >
        <span className={cn("absolute top-[2px] h-[14px] w-[14px] rounded-full transition-all", checked ? "left-[16px] bg-bg" : "left-[2px] bg-white/90")} />
      </button>
    </label>
  );
}

export function NumberInput({
  value, onChange, step = 1, className, ariaLabel, min, max,
}: { value: number; onChange: (v: number) => void; step?: number; className?: string; ariaLabel: string; min?: number; max?: number }) {
  return (
    <input
      type="number"
      aria-label={ariaLabel}
      value={Number.isFinite(value) ? Number(value.toFixed(4)) : 0}
      step={step}
      min={min}
      max={max}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (Number.isFinite(v)) onChange(v);
      }}
      className={cn(
        "num h-8 w-full min-w-0 rounded-md border border-line bg-raised px-1.5 text-center text-[13px] text-fg outline-none hover:border-line-strong focus:border-line-strong",
        className,
      )}
    />
  );
}

export function MatrixInput({
  value, onChange, label, step = 0.1, highlight,
}: { value: number[][]; onChange: (m: number[][]) => void; label?: string; step?: number; highlight?: (i: number, j: number) => boolean }) {
  const cols = value[0]?.length ?? 0;
  return (
    <div className="space-y-1">
      {label && <div className="text-[12.5px] text-muted">{label}</div>}
      <div className="relative inline-grid gap-1 px-2" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 3.6rem))` }}>
        <span className="pointer-events-none absolute inset-y-0 left-0 w-1.5 rounded-l border-y border-l border-line-strong" />
        <span className="pointer-events-none absolute inset-y-0 right-0 w-1.5 rounded-r border-y border-r border-line-strong" />
        {value.map((row, i) =>
          row.map((v, j) => (
            <NumberInput
              key={`${i}-${j}`}
              ariaLabel={`${label ?? "matrix"} row ${i + 1} column ${j + 1}`}
              value={v}
              step={step}
              className={highlight?.(i, j) ? "border-pink" : undefined}
              onChange={(nv) => onChange(value.map((r, a) => r.map((x, b) => (a === i && b === j ? nv : x))))}
            />
          )),
        )}
      </div>
    </div>
  );
}

export function AnimationControls({
  playing, onPlay, onStep, onReset, canStep = true, extra,
}: { playing: boolean; onPlay: () => void; onStep?: () => void; onReset: () => void; canStep?: boolean; extra?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <button className="btn" onClick={onPlay} aria-label={playing ? "Pause" : "Play"}>
        {playing ? <Pause size={14} /> : <Play size={14} />} {playing ? "Pause" : "Play"}
      </button>
      {onStep && (
        <button className="btn" onClick={onStep} disabled={playing || !canStep} aria-label="Step forward">
          <StepForward size={14} /> Step
        </button>
      )}
      <button className="btn" onClick={onReset} aria-label="Reset">
        <RotateCcw size={14} /> Reset
      </button>
      {extra}
    </div>
  );
}

export function Presets<T>({ presets, onPick }: { presets: { label: string; value: T }[]; onPick: (v: T) => void }) {
  return (
    <div className="space-y-1">
      <div className="text-[12.5px] text-muted">Presets</div>
      <div className="flex flex-wrap gap-1">
        {presets.map((p) => (
          <button key={p.label} className="chip hover:border-line-strong hover:text-fg" onClick={() => onPick(p.value)}>
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function ControlGroup({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <div className="space-y-3">
      {title && <div className="label">{title}</div>}
      {children}
    </div>
  );
}
