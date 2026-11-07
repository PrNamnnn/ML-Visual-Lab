import type { Metadata } from "next";
import { ContinueExploring } from "@/components/common/ContinueExploring";

export const metadata: Metadata = { title: "Bookmarks" };

export default function Bookmarks() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <div className="label">Your library</div>
      <h1 className="mt-1 text-[28px] font-semibold tracking-tight">Bookmarks and history</h1>
      <p className="mt-2 max-w-2xl text-[15px] text-muted">Bookmark a topic from its page. Everything is stored locally in this browser — no account, no server.</p>
      <div className="mt-8"><ContinueExploring /></div>
    </div>
  );
}
