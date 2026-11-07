"use client";
import { useEffect } from "react";
import { Bookmark, BookmarkCheck, CheckCircle2, Circle } from "lucide-react";
import { useProgress } from "@/store/progress";

export function TopicActions({ topicKey }: { topicKey: string }) {
  const bookmarked = useProgress((s) => s.bookmarks.includes(topicKey));
  const done = useProgress((s) => s.completed.includes(topicKey));
  const toggleBookmark = useProgress((s) => s.toggleBookmark);
  const toggleCompleted = useProgress((s) => s.toggleCompleted);
  const visit = useProgress((s) => s.visit);

  useEffect(() => {
    // wait for hydration so the visit isn't overwritten by stored state
    const p = useProgress.persist;
    if (p.hasHydrated()) visit(topicKey);
    else return p.onFinishHydration(() => visit(topicKey));
  }, [topicKey, visit]);

  return (
    <div className="flex flex-wrap gap-2">
      <button className="btn" onClick={() => toggleBookmark(topicKey)} aria-pressed={bookmarked}>
        {bookmarked ? <BookmarkCheck size={14} className="text-fg" /> : <Bookmark size={14} />}
        {bookmarked ? "Bookmarked" : "Bookmark"}
      </button>
      <button className="btn" onClick={() => toggleCompleted(topicKey)} aria-pressed={done}>
        {done ? <CheckCircle2 size={14} className="text-green" /> : <Circle size={14} />}
        {done ? "Marked as learned" : "Mark as learned"}
      </button>
    </div>
  );
}
