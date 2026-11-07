"use client";
import { useProgress } from "@/store/progress";

export function ProgressBadge({ keys }: { keys: string[] }) {
  const done = useProgress((s) => s.completed.filter((k) => keys.includes(k)).length);
  if (!done) return null;
  return <span className="chip border-green/40 text-green">{done} / {keys.length} learned</span>;
}
