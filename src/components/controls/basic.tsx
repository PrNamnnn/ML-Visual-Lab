"use client";
import { useId } from "react";
import { cn } from "@/lib/utilities/format";

export function Select<T extends string>({
  label, value, options, onChange,
}: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  const id = useId();
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-[12.5px] text-muted">{label}</label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="h-8 w-full rounded-md border border-line bg-raised px-2 text-[13px] text-fg outline-none hover:border-line-strong"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

export function Segmented<T extends string>({
  value, options, onChange, label,
}: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label?: string }) {
  return (
    <div className="space-y-1">
      {label && <div className="text-[12.5px] text-muted">{label}</div>}
      <div role="radiogroup" aria-label={label} className="flex flex-wrap gap-0.5 rounded-md border border-line bg-raised p-0.5">
        {options.map((o) => (
          <button
            key={o.value}
            role="radio"
            aria-checked={value === o.value}
            onClick={() => onChange(o.value)}
            className={cn(
              "h-7 flex-1 whitespace-nowrap rounded px-2 text-[12px] font-medium transition-colors",
              value === o.value ? "bg-panel text-fg shadow-sm ring-1 ring-line-strong" : "text-muted hover:text-fg",
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}
