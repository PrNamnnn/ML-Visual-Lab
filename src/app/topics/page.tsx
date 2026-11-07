import type { Metadata } from "next";
import { topics, modules } from "@/data/curriculum";
import { TopicBrowser } from "@/components/topic/TopicBrowser";

export const metadata: Metadata = { title: "All Topics" };

export default function AllTopics() {
  const list = topics.map((t) => ({ key: t.key, title: t.title, href: t.href, module: t.module, moduleTitle: t.moduleTitle, section: t.section, status: t.status, difficulty: t.difficulty, summary: t.summary ?? "" }));
  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <div className="label">Curriculum</div>
      <h1 className="mt-1 text-[28px] font-semibold tracking-tight">All topics</h1>
      <p className="mt-2 max-w-2xl text-[15px] text-muted">{topics.length} topics across {modules.length} modules. Interactive lessons have a working visualization, formulas, and AI/ML applications; planned topics are listed to keep the curriculum complete and are clearly marked.</p>
      <TopicBrowser topics={list} modules={modules.map((m) => ({ id: m.id, title: m.title }))} />
    </div>
  );
}
