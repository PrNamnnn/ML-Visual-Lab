import Link from "next/link";
import { ArrowUpRight, ChevronRight, Info } from "lucide-react";
import type { Topic } from "@/types/curriculum";
import { Tex } from "@/components/mathematics/Tex";
import { topicByKey, topics } from "@/data/curriculum";
import { TopicActions } from "./TopicActions";
import { VisualSlot } from "@/visualizers/VisualSlot";
import { CodeTabs } from "./CodeTabs";
import { highlight } from "@/lib/utilities/highlight";

const diffLabel = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced" } as const;

function Section({ id, letter, title, children }: { id: string; letter: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="scroll-mt-16">
      <h2 id={`${id}-h`} className="mb-3 flex items-baseline gap-2.5 text-[15px] font-semibold tracking-tight">
        <span className="num text-[11px] font-medium text-faint">{letter}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

function TopicLink({ k }: { k: string }) {
  const t = topicByKey(k);
  if (!t) return null;
  return (
    <Link href={t.href} className="group flex items-center justify-between gap-3 rounded-md border border-line px-3 py-2 hover:border-line-strong hover:bg-raised/50">
      <div className="min-w-0">
        <div className="truncate text-[13px] text-fg">{t.title}</div>
        <div className="truncate text-[11.5px] text-faint">{t.moduleTitle}{t.status === "planned" ? " · planned" : ""}</div>
      </div>
      <ArrowUpRight size={14} className="shrink-0 text-faint group-hover:text-fg" />
    </Link>
  );
}

export function Breadcrumb({ topic }: { topic: Topic }) {
  return (
    <nav aria-label="Breadcrumb" className="flex flex-wrap items-center gap-1 text-[12px] text-faint">
      <Link href={`/${topic.module}`} className="hover:text-fg">{topic.moduleTitle}</Link>
      <ChevronRight size={12} />
      <span>{topic.section}</span>
    </nav>
  );
}

export function TopicView({ topic }: { topic: Topic }) {
  const c = topic.content;
  const sameSection = topics.filter((t) => t.module === topic.module && t.section === topic.section && t.key !== topic.key);
  if (!c) {
    return (
      <article className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
        <Breadcrumb topic={topic} />
        <h1 className="mt-2 text-[26px] font-semibold tracking-tight">{topic.title}</h1>
        {topic.summary && <p className="mt-2 text-[15px] text-muted">{topic.summary}</p>}
        <div className="mt-6 flex gap-3 rounded-lg border border-line bg-panel p-4">
          <Info size={16} className="mt-0.5 shrink-0 text-amber" />
          <div className="space-y-1 text-[13.5px] text-muted">
            <p className="font-medium text-fg">Interactive lesson not built yet</p>
            <p>This concept is part of the curriculum, but its visualization, formulas and worked example have not been implemented. It is listed so the learning path stays complete — it is not presented as finished.</p>
          </div>
        </div>
        {sameSection.some((t) => t.status === "complete") && (
          <div className="mt-8">
            <div className="label mb-2">Available in {topic.section}</div>
            <div className="grid gap-2 sm:grid-cols-2">
              {sameSection.filter((t) => t.status === "complete").slice(0, 8).map((t) => <TopicLink key={t.key} k={t.key} />)}
            </div>
          </div>
        )}
      </article>
    );
  }

  const related = [...new Set([...(topic.related ?? []), ...sameSection.filter((t) => t.status === "complete").slice(0, 3).map((t) => t.key)])].filter((k) => topicByKey(k));

  return (
    <article className="mx-auto max-w-[1180px] px-4 py-6 sm:px-6 lg:py-8">
      {/* A. Concept introduction */}
      <header className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0">
          <Breadcrumb topic={topic} />
          <h1 className="mt-2 text-[26px] font-semibold leading-tight tracking-tight sm:text-[28px]">{topic.title}</h1>
          <p className="mt-3 max-w-[68ch] text-[15px] leading-relaxed text-fg/90">{c.definition}</p>
          <p className="mt-2 max-w-[68ch] text-[14px] leading-relaxed text-muted"><span className="text-fg/80">Why it exists. </span>{c.why}</p>
          <div className="mt-4"><TopicActions topicKey={topic.key} /></div>
        </div>
        <aside className="space-y-4 rounded-lg border border-line p-4 text-[13px]">
          <div className="flex flex-wrap gap-1.5">
            <span className="chip">{diffLabel[topic.difficulty]}</span>
            <span className="chip">{topic.moduleTitle}</span>
          </div>
          <div>
            <div className="label mb-1.5">Used in AI/ML</div>
            <ul className="space-y-1 text-muted">
              {c.usedIn.map((u) => <li key={u} className="flex gap-2"><span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-faint" />{u}</li>)}
            </ul>
          </div>
          {topic.prerequisites?.length ? (
            <div>
              <div className="label mb-1.5">Prerequisites</div>
              <div className="flex flex-wrap gap-1.5">
                {topic.prerequisites.map((p) => {
                  const t = topicByKey(p);
                  return t ? <Link key={p} href={t.href} className="chip hover:border-line-strong hover:text-fg">{t.title}</Link> : null;
                })}
              </div>
            </div>
          ) : null}
        </aside>
      </header>

      <div className="mt-8 space-y-10">
        {/* B + D. Visual intuition & interactive experiment */}
        <Section id="visual" letter="B · D" title="Visual intuition and experiment">
          <VisualSlot key={topic.key} id={c.visual.id} props={c.visual.props} />
          {c.visual.caption && <p className="mt-2 text-[12.5px] text-faint">{c.visual.caption}</p>}
        </Section>

        {/* C. Mathematical formulation */}
        <Section id="math" letter="C" title="Mathematical formulation">
          <div className="grid gap-3 md:grid-cols-2">
            {c.formulas.map((f) => (
              <div key={f.label} className="panel min-w-0 p-4">
                <div className="label mb-2">{f.label}</div>
                <div className="overflow-x-auto py-1"><Tex tex={f.tex} block /></div>
                <p className="mt-2 text-[13px] leading-relaxed text-muted">{f.explanation}</p>
                {f.variables && (
                  <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12.5px]">
                    {f.variables.map((v) => (
                      <div key={v.symbol} className="contents">
                        <dt><Tex tex={v.symbol} /></dt>
                        <dd className="text-muted">{v.meaning}</dd>
                      </div>
                    ))}
                  </dl>
                )}
              </div>
            ))}
          </div>
          {(c.assumptions || c.specialCases || c.limitations) && (
            <div className="mt-3 grid gap-3 md:grid-cols-3">
              {[["Assumptions", c.assumptions], ["Special cases", c.specialCases], ["Limitations", c.limitations]].map(([title, list]) =>
                list && (list as string[]).length ? (
                  <div key={title as string} className="rounded-lg border border-line p-4">
                    <div className="label mb-2">{title as string}</div>
                    <ul className="space-y-1.5 text-[13px] leading-relaxed text-muted">
                      {(list as string[]).map((a) => <li key={a}>{a}</li>)}
                    </ul>
                  </div>
                ) : null,
              )}
            </div>
          )}
        </Section>

        {/* E. Worked example */}
        {c.worked && (
          <Section id="worked" letter="E" title={`Worked example — ${c.worked.title}`}>
            <div className="panel p-4">
              <ol className="space-y-2">
                {c.worked.steps.map((s, i) => (
                  <li key={i} className="flex min-w-0 items-baseline gap-3">
                    <span className="num w-5 shrink-0 text-[11px] text-faint">{i + 1}</span>
                    <div className="min-w-0 overflow-x-auto"><Tex tex={s.tex} /></div>
                    {s.note && <span className="text-[12px] text-faint">{s.note}</span>}
                  </li>
                ))}
              </ol>
              {c.worked.result && (
                <div className="mt-3 flex items-center gap-3 border-t border-line pt-3">
                  <span className="label">Result</span>
                  <Tex tex={c.worked.result} />
                </div>
              )}
              <p className="mt-3 text-[11.5px] text-faint">Static example with fixed inputs. The interactive visualization above recomputes its values live.</p>
            </div>
          </Section>
        )}

        {/* H. Implementation */}
        {c.code && (
          <Section id="code" letter="H" title="Implementation in Python and Rust">
            <CodeTabs python={{ html: highlight(c.code.python, "python"), raw: c.code.python }} rust={{ html: highlight(c.code.rust, "rust"), raw: c.code.rust }} output={c.code.output} explanation={c.code.explanation} />
          </Section>
        )}

        {/* F. AI/ML application */}
        <Section id="apps" letter="F" title="Where it shows up in AI/ML">
          <div className="grid gap-3 md:grid-cols-2">
            {c.applications.map((a) => (
              <div key={a.title} className="rounded-lg border border-line p-4">
                <div className="text-[13.5px] font-medium">{a.title}</div>
                <p className="mt-1 text-[13px] leading-relaxed text-muted">{a.body}</p>
              </div>
            ))}
          </div>
        </Section>

        {/* G. Related */}
        {related.length > 0 && (
          <Section id="related" letter="G" title="Related concepts">
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {related.map((k) => <TopicLink key={k} k={k} />)}
            </div>
          </Section>
        )}
      </div>
    </article>
  );
}
