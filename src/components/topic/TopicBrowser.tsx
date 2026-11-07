"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { Segmented, Select } from "@/components/controls/basic";
import { useProgress } from "@/store/progress";
import { cn } from "@/lib/utilities/format";

interface T { key: string; title: string; href: string; module: string; moduleTitle: string; section: string; status: "complete" | "planned"; difficulty: string; summary: string }

export function TopicBrowser({ topics, modules }: { topics: T[]; modules: { id: string; title: string }[] }) {
  const [q, setQ] = useState("");
  const [mod, setMod] = useState("all");
  const [status, setStatus] = useState<"all" | "complete" | "planned">("complete");
  const [diff, setDiff] = useState("all");
  const done = useProgress((s) => s.completed);
  const list = useMemo(() => topics.filter((t) =>
    (mod === "all" || t.module === mod) && (status === "all" || t.status === status) && (diff === "all" || t.difficulty === diff) &&
    (!q || `${t.title} ${t.summary} ${t.section}`.toLowerCase().includes(q.toLowerCase())),
  ), [topics, mod, status, diff, q]);
  return (
    <div className="mt-6 space-y-4">
      <div className="grid gap-3 md:grid-cols-[1fr_200px_160px_auto]">
        <label className="space-y-1">
          <span className="block text-[12.5px] text-muted">Filter</span>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Type to filter…" className="h-8 w-full rounded-md border border-line bg-raised px-2 text-[13px] outline-none focus:border-line-strong" />
        </label>
        <Select label="Module" value={mod} onChange={setMod} options={[{ value: "all", label: "All modules" }, ...modules.map((m) => ({ value: m.id, label: m.title }))]} />
        <Select label="Difficulty" value={diff} onChange={setDiff} options={[{ value: "all", label: "Any" }, { value: "beginner", label: "Beginner" }, { value: "intermediate", label: "Intermediate" }, { value: "advanced", label: "Advanced" }]} />
        <Segmented label="Status" value={status} onChange={setStatus} options={[{ value: "complete", label: "Interactive" }, { value: "planned", label: "Planned" }, { value: "all", label: "All" }]} />
      </div>
      <div className="text-[12px] text-faint">{list.length} topics</div>
      <div className="divide-y divide-line rounded-lg border border-line">
        {list.map((t) => (
          <Link key={t.key} href={t.href} className="flex items-center gap-4 px-4 py-2.5 hover:bg-raised/50">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 text-[13.5px]">
                <span className={cn(t.status === "planned" && "text-muted")}>{t.title}</span>
                {t.status === "planned" && <span className="chip py-0 text-[10px]">planned</span>}
                {done.includes(t.key) && <CheckCircle2 size={13} className="text-green" />}
              </div>
              <div className="truncate text-[11.5px] text-faint">{t.moduleTitle} · {t.section}</div>
            </div>
            <span className="hidden text-[11.5px] capitalize text-faint sm:block">{t.difficulty}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
