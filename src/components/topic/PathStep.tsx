"use client";
import Link from "next/link";
import { CheckCircle2, Circle } from "lucide-react";
import { useProgress } from "@/store/progress";

export function PathStep({ k, title, href }: { k: string; title: string; href: string }) {
  const done = useProgress((s) => s.completed.includes(k));
  const toggle = useProgress((s) => s.toggleCompleted);
  return (
    <div className="flex items-center gap-2 rounded-md border border-line px-2.5 py-1.5 hover:border-line-strong">
      <button onClick={() => toggle(k)} aria-label={done ? `Mark ${title} as not learned` : `Mark ${title} as learned`} aria-pressed={done} className="text-faint hover:text-fg">
        {done ? <CheckCircle2 size={15} className="text-green" /> : <Circle size={15} />}
      </button>
      <Link href={href} className="min-w-0 flex-1 truncate text-[13px] hover:text-fg">{title}</Link>
    </div>
  );
}
