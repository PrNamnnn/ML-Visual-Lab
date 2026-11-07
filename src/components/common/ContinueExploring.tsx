"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowUpRight, Bookmark, Clock } from "lucide-react";
import { useProgress } from "@/store/progress";
import { useNav } from "@/components/navigation/NavContext";

function relTime(at: number) {
  const m = Math.round((Date.now() - at) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  return `${Math.round(h / 24)} d ago`;
}

export function ContinueExploring({ compact = false }: { compact?: boolean }) {
  const { byKey } = useNav();
  const recent = useProgress((s) => s.recent);
  const bookmarks = useProgress((s) => s.bookmarks);
  const clear = useProgress((s) => s.clearHistory);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-24" />;
  const rec = recent.map((r) => ({ ...r, t: byKey.get(r.key) })).filter((r) => r.t);
  const bms = bookmarks.map((k) => byKey.get(k)).filter(Boolean);
  if (!rec.length && !bms.length) {
    return (
      <div className="rounded-lg border border-dashed border-line px-4 py-6 text-center text-[13px] text-faint">
        Topics you open and bookmark will appear here. Progress is stored only in this browser.
      </div>
    );
  }
  const Item = ({ href, title, meta, icon }: { href: string; title: string; meta: string; icon: React.ReactNode }) => (
    <Link href={href} className="group flex items-center gap-3 rounded-md border border-line px-3 py-2 hover:border-line-strong hover:bg-raised/50">
      <span className="text-faint">{icon}</span>
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px]">{title}</div>
        <div className="truncate text-[11.5px] text-faint">{meta}</div>
      </div>
      <ArrowUpRight size={13} className="text-faint group-hover:text-fg" />
    </Link>
  );
  return (
    <div className={compact ? "space-y-4" : "grid gap-6 md:grid-cols-2"}>
      {rec.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="label">Recently visited</div>
            <button className="text-[11.5px] text-faint hover:text-fg" onClick={clear}>Clear</button>
          </div>
          <div className="grid gap-1.5">{rec.slice(0, 6).map((r) => <Item key={r.key} href={r.t!.href} title={r.t!.title} meta={relTime(r.at)} icon={<Clock size={13} />} />)}</div>
        </div>
      )}
      {bms.length > 0 && (
        <div className="space-y-2">
          <div className="label">Bookmarks</div>
          <div className="grid gap-1.5">{bms.slice(0, 6).map((t) => <Item key={t!.key} href={t!.href} title={t!.title} meta={t!.section} icon={<Bookmark size={13} />} />)}</div>
        </div>
      )}
    </div>
  );
}
