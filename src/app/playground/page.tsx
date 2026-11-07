import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { labs } from "@/data/labs";

export const metadata: Metadata = { title: "Visual Playground" };

export default function Playground() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <div className="label">Visual Playground</div>
      <h1 className="mt-1 text-[28px] font-semibold tracking-tight">Free-form laboratories</h1>
      <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-muted">Sixteen labs for experimenting without a fixed lesson. Each has adjustable parameters, live mathematical values, presets and reset. Only the engine a lab needs is loaded when you open it.</p>
      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {labs.map((l, i) => (
          <Link key={l.slug} href={`/playground/${l.slug}`} className="group flex flex-col rounded-lg border border-line bg-panel p-4 transition-colors hover:border-line-strong">
            <div className="flex items-center justify-between">
              <span className="num text-[11px] text-faint">{String(i + 1).padStart(2, "0")}</span>
              <span className="chip">{l.engine}</span>
            </div>
            <div className="mt-3 text-[14.5px] font-medium">{l.title}</div>
            <p className="mt-1 flex-1 text-[13px] leading-relaxed text-muted">{l.blurb}</p>
            <span className="mt-3 inline-flex items-center gap-1 text-[12.5px] text-fg">Open lab <ArrowRight size={13} className="transition-transform group-hover:translate-x-0.5" /></span>
          </Link>
        ))}
      </div>
    </div>
  );
}
