import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ChevronRight } from "lucide-react";
import { labs, labBySlug } from "@/data/labs";
import { topicByKey } from "@/data/curriculum";
import { VisualSlot } from "@/visualizers/VisualSlot";

export const dynamicParams = false;
export function generateStaticParams() {
  return labs.map((l) => ({ lab: l.slug }));
}

type Params = Promise<{ lab: string }>;
export async function generateMetadata({ params }: { params: Params }): Promise<Metadata> {
  const { lab } = await params;
  return { title: labBySlug(lab)?.title ?? "Lab" };
}

export default async function LabPage({ params }: { params: Params }) {
  const { lab } = await params;
  const l = labBySlug(lab);
  if (!l) notFound();
  const idx = labs.indexOf(l);
  const prev = labs[idx - 1], next = labs[idx + 1];
  return (
    <div className="mx-auto max-w-[1180px] px-4 py-6 sm:px-6">
      <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-[12px] text-faint">
        <Link href="/playground" className="hover:text-fg">Visual Playground</Link>
        <ChevronRight size={12} />
        <span>{l.title}</span>
      </nav>
      <div className="mt-2 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-[26px] font-semibold tracking-tight">{l.title}</h1>
          <p className="mt-1 text-[14.5px] text-muted">{l.blurb}</p>
        </div>
        <span className="chip">{l.engine}</span>
      </div>
      <div className="mt-6">
        <VisualSlot key={l.slug} id={l.visual} props={l.props} />
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-[1fr_auto]">
        <div>
          <div className="label mb-2">Related lessons</div>
          <div className="flex flex-wrap gap-1.5">
            {l.topics.map((k) => { const t = topicByKey(k); return t ? <Link key={k} href={t.href} className="chip hover:border-line-strong hover:text-fg">{t.title}</Link> : null; })}
          </div>
        </div>
        <div className="flex gap-2 self-end">
          {prev && <Link href={`/playground/${prev.slug}`} className="btn">← {prev.title}</Link>}
          {next && <Link href={`/playground/${next.slug}`} className="btn">{next.title} →</Link>}
        </div>
      </div>
    </div>
  );
}
