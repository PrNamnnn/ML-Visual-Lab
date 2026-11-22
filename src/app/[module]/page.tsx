import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowRight } from "lucide-react";
import { modules, moduleById, topics, moduleStats, topicByKey } from "@/data/curriculum";
import type { ModuleDef, SectionDef } from "@/types/curriculum";
import { ProgressBadge } from "@/components/common/ProgressBadge";

export const dynamicParams = false;
export function generateStaticParams() {
  return modules.map((m) => ({ module: m.id }));
}

type Params = Promise<{ module: string }>;
export async function generateMetadata({ params }: { params: Params }): Promise<Metadata> {
  const { module } = await params;
  return { title: moduleById(module)?.title ?? "Module" };
}

function TopicCard({ m, slug }: { m: ModuleDef; slug: string }) {
  const tp = topicByKey(`${m.id}/${slug}`)!;
  return (
    <Link href={tp.href} className="group flex min-w-0 items-start justify-between gap-3 rounded-md border border-line px-3.5 py-3 hover:border-line-strong hover:bg-raised/50">
      <div className="min-w-0">
        <div className="flex items-center gap-2 text-[14px]">
          <span className={tp.status === "complete" ? "text-fg" : "text-muted"}>{tp.title}</span>
          {tp.status === "planned" && <span className="chip py-0 text-[10px]">planned</span>}
        </div>
        {tp.summary && <div className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-faint">{tp.summary}</div>}
      </div>
      <ArrowRight size={14} className="mt-1 shrink-0 text-faint group-hover:text-fg" />
    </Link>
  );
}

function SectionBlock({ m, s, index }: { m: ModuleDef; s: SectionDef; index: string }) {
  return (
    <section id={s.id} className="scroll-mt-20">
      <h3 className="flex items-baseline gap-2.5 text-[16px] font-semibold tracking-tight">
        <span className="num text-[11.5px] font-medium text-faint">{index}</span>
        {s.title}
      </h3>
      {s.blurb && <p className="mt-1.5 max-w-[78ch] text-[13.5px] leading-relaxed text-muted">{s.blurb}</p>}
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        {s.topics.map((t) => <TopicCard key={t.slug} m={m} slug={t.slug} />)}
      </div>
      {s.syllabus && (
        <details className="group mt-3 rounded-md border border-line">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3.5 py-2 text-[12.5px] text-muted hover:text-fg">
            <span>Syllabus coverage — {s.syllabus.length} items</span>
            <span className="text-faint group-open:hidden">show</span>
            <span className="hidden text-faint group-open:inline">hide</span>
          </summary>
          <ul className="grid gap-x-6 gap-y-1.5 border-t border-line px-3.5 py-3 text-[13px] sm:grid-cols-2">
            {s.syllabus.map((it) => {
              const tp = topicByKey(`${m.id}/${it.slug}`);
              return (
                <li key={it.item} className="flex min-w-0 items-baseline gap-2">
                  <span className="mt-[2px] h-1 w-1 shrink-0 translate-y-[-3px] rounded-full bg-faint" />
                  <span className="min-w-0 text-muted">
                    {it.item}
                    {tp && <> — <Link href={tp.href} className="text-fg/90 underline decoration-line-strong underline-offset-[3px] hover:decoration-fg">{tp.title}</Link></>}
                  </span>
                </li>
              );
            })}
          </ul>
        </details>
      )}
    </section>
  );
}

export default async function ModulePage({ params }: { params: Params }) {
  const { module } = await params;
  const m = moduleById(module);
  if (!m) notFound();
  const st = moduleStats(m.id);
  const parts = m.parts?.length ? m.parts : null;
  const weeks = m.sections.filter((s) => s.week);
  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <div className="label">Module</div>
      <h1 className="mt-1 text-[28px] font-semibold tracking-tight sm:text-[30px]">{m.title}</h1>
      <p className="mt-2 max-w-[70ch] text-[15px] leading-relaxed text-muted">{m.description}</p>
      <div className="mt-3 flex flex-wrap gap-2 text-[12px]">
        <span className="chip">{st.complete} interactive lessons</span>
        {st.total - st.complete > 0 && <span className="chip">{st.total - st.complete} planned</span>}
        <ProgressBadge keys={topics.filter((t) => t.module === m.id).map((t) => t.key)} />
      </div>

      {parts && (
        <nav aria-label="Module contents" className="mt-6 rounded-lg border border-line">
          <ul className="divide-y divide-line">
            {parts.map((p) => {
              const secs = m.sections.filter((s) => s.part === p.id);
              return (
                <li key={p.id} className="flex flex-col gap-1.5 px-4 py-3 sm:flex-row sm:items-baseline sm:gap-4">
                  <a href={`#part-${p.id}`} className="shrink-0 text-[13.5px] font-medium hover:underline sm:w-[250px]">{p.title}</a>
                  <span className="flex flex-wrap gap-x-3 gap-y-1 text-[12.5px] text-muted">
                    {secs.map((s) => <a key={s.id} href={`#${s.id}`} className="hover:text-fg">{s.week ? `W${s.week}` : s.title}</a>)}
                  </span>
                </li>
              );
            })}
          </ul>
        </nav>
      )}

      {parts ? (
        <div className="mt-10 space-y-14">
          {parts.map((p) => {
            const secs = m.sections.filter((s) => s.part === p.id);
            if (!secs.length) return null;
            return (
              <div key={p.id} id={`part-${p.id}`} className="scroll-mt-20">
                <div className="border-b border-line pb-3">
                  <h2 className="text-[20px] font-semibold tracking-tight">{p.title}</h2>
                  {p.description && <p className="mt-1.5 max-w-[78ch] text-[14px] leading-relaxed text-muted">{p.description}</p>}
                </div>
                <div className="mt-6 space-y-9">
                  {secs.map((s, i) => <SectionBlock key={s.id} m={m} s={s} index={s.week ? `Week ${s.week}` : String(i + 1)} />)}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="mt-8 space-y-9">
          {m.sections.map((s, si) => <SectionBlock key={s.id} m={m} s={s} index={String(si + 1)} />)}
        </div>
      )}
      {weeks.length > 0 && <p className="mt-10 text-[12.5px] leading-relaxed text-faint">Week lessons follow the topic sequence of the IITM Machine Learning Techniques course notes. Explanations, examples, code and visualizations are written for this site.</p>}
    </div>
  );
}
