import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowRight } from "lucide-react";
import { modules, moduleById, topics, moduleStats } from "@/data/curriculum";
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

export default async function ModulePage({ params }: { params: Params }) {
  const { module } = await params;
  const m = moduleById(module);
  if (!m) notFound();
  const st = moduleStats(m.id);
  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <div className="label">Module</div>
      <h1 className="mt-1 text-[28px] font-semibold tracking-tight">{m.title}</h1>
      <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-muted">{m.description}</p>
      <div className="mt-3 flex flex-wrap gap-2 text-[12px]">
        <span className="chip">{st.complete} interactive lessons</span>
        <span className="chip">{st.total - st.complete} planned</span>
        <ProgressBadge keys={topics.filter((t) => t.module === m.id).map((t) => t.key)} />
      </div>
      <div className="mt-8 space-y-8">
        {m.sections.map((s, si) => (
          <section key={s.id}>
            <h2 className="mb-3 flex items-baseline gap-2 text-[15px] font-semibold">
              <span className="num text-[11px] text-faint">{si + 1}</span>
              {s.title}
            </h2>
            <div className="grid gap-2 sm:grid-cols-2">
              {s.topics.map((t) => {
                const tp = topics.find((x) => x.key === `${m.id}/${t.slug}`)!;
                return (
                  <Link key={t.slug} href={tp.href} className="group flex items-start justify-between gap-3 rounded-md border border-line px-3.5 py-3 hover:border-line-strong hover:bg-raised/50">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 text-[13.5px]">
                        <span className={tp.status === "complete" ? "text-fg" : "text-muted"}>{t.title}</span>
                        {tp.status === "planned" && <span className="chip py-0 text-[10px]">planned</span>}
                      </div>
                      {t.summary && <div className="mt-0.5 line-clamp-2 text-[12.5px] text-faint">{t.summary}</div>}
                    </div>
                    <ArrowRight size={14} className="mt-1 shrink-0 text-faint group-hover:text-fg" />
                  </Link>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
