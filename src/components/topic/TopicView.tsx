import Link from "next/link";
import { ArrowLeft, ArrowRight, ArrowUpRight, ChevronRight, Info } from "lucide-react";
import type { Topic, WorkedStep } from "@/types/curriculum";
import { Tex } from "@/components/mathematics/Tex";
import { neighbours, sectionOf, topicByKey, topics } from "@/data/curriculum";
import { TopicActions } from "./TopicActions";
import { VisualSlot } from "@/visualizers/VisualSlot";
import { CodeTabs } from "./CodeTabs";
import { highlight } from "@/lib/utilities/highlight";

const diffLabel = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced" } as const;

function Section({ id, title, children, aside }: { id: string; title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="scroll-mt-20">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-line pb-2">
        <h2 id={`${id}-h`} className="text-[17px] font-semibold tracking-tight">{title}</h2>
        {aside && <div className="text-[12px] text-faint">{aside}</div>}
      </div>
      {children}
    </section>
  );
}

function TopicLink({ k }: { k: string }) {
  const t = topicByKey(k);
  if (!t) return null;
  return (
    <Link href={t.href} className="group flex min-w-0 items-center justify-between gap-3 rounded-md border border-line px-3 py-2 hover:border-line-strong hover:bg-raised/50">
      <div className="min-w-0">
        <div className="truncate text-[13.5px] text-fg">{t.title}</div>
        <div className="truncate text-[12px] text-faint">{t.week ? `${t.moduleTitle} · MLT Week ${t.week}` : t.moduleTitle}{t.status === "planned" ? " · planned" : ""}</div>
      </div>
      <ArrowUpRight size={14} className="shrink-0 text-faint group-hover:text-fg" />
    </Link>
  );
}

export function Breadcrumb({ topic }: { topic: Topic }) {
  return (
    <nav aria-label="Breadcrumb" className="flex flex-wrap items-center gap-1 text-[12.5px] text-faint">
      <Link href={`/${topic.module}`} className="hover:text-fg">{topic.moduleTitle}</Link>
      <ChevronRight size={12} />
      <Link href={`/${topic.module}#${topic.sectionId}`} className="hover:text-fg">{topic.section}</Link>
    </nav>
  );
}

/** Numbered sequence of equations with a short remark per step (derivations and worked examples). */
function Steps({ steps }: { steps: WorkedStep[] }) {
  return (
    <ol className="steps">
      {steps.map((s, i) => (
        <li key={i}>
          <span className="num step-n" aria-hidden>{i + 1}</span>
          <div className="min-w-0">
            <div className="math-scroll"><Tex tex={s.tex} block /></div>
            {s.note && <p className="mt-1 text-[13px] leading-relaxed text-muted">{s.note}</p>}
          </div>
        </li>
      ))}
    </ol>
  );
}

function PrevNext({ topic }: { topic: Topic }) {
  const { prev, next } = neighbours(topic.key);
  if (!prev && !next) return null;
  const card = (t: Topic, dir: "prev" | "next") => (
    <Link href={t.href} rel={dir} className={`group flex min-w-0 flex-1 items-center gap-3 rounded-lg border border-line px-4 py-3 hover:border-line-strong hover:bg-raised/50 ${dir === "next" ? "justify-end text-right" : ""}`}>
      {dir === "prev" && <ArrowLeft size={15} className="shrink-0 text-faint group-hover:text-fg" />}
      <div className="min-w-0">
        <div className="label">{dir === "prev" ? "Previous" : "Next"}{t.sectionId !== topic.sectionId ? ` · ${t.section}` : ""}</div>
        <div className="mt-0.5 truncate text-[14px] text-fg">{t.title}</div>
      </div>
      {dir === "next" && <ArrowRight size={15} className="shrink-0 text-faint group-hover:text-fg" />}
    </Link>
  );
  return (
    <nav aria-label="Lesson sequence" className="flex flex-col gap-2 sm:flex-row">
      {prev ? card(prev, "prev") : <span className="hidden flex-1 sm:block" />}
      {next ? card(next, "next") : <span className="hidden flex-1 sm:block" />}
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
  const section = sectionOf(topic);
  const covers = section?.syllabus?.filter((s) => s.slug === topic.slug).map((s) => s.item) ?? [];
  const notes: [string, string[] | undefined][] = [["Assumptions", c.assumptions], ["Special cases", c.specialCases], ["Limitations", c.limitations]];
  const shownNotes = notes.filter(([, l]) => l?.length);

  return (
    <article className="mx-auto max-w-[1180px] px-4 py-6 sm:px-6 lg:py-8">
      <header className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0">
          <Breadcrumb topic={topic} />
          <h1 className="mt-2 text-[26px] font-semibold leading-tight tracking-tight sm:text-[30px]">{topic.title}</h1>
          <div className="prose-lesson mt-4">
            {c.intuition && <p><span className="lead-in">Intuition. </span>{c.intuition}</p>}
            <p className={c.intuition ? undefined : "text-fg/90"}>{c.intuition && <span className="lead-in">Definition. </span>}{c.definition}</p>
            <p><span className="lead-in">Why it matters. </span>{c.why}</p>
          </div>
          <div className="mt-5"><TopicActions topicKey={topic.key} /></div>
        </div>
        <aside className="space-y-4 self-start rounded-lg border border-line p-4 text-[13px]">
          <div className="flex flex-wrap gap-1.5">
            <span className="chip">{diffLabel[topic.difficulty]}</span>
            <span className="chip">{topic.week ? `MLT Week ${topic.week}` : topic.moduleTitle}</span>
          </div>
          {covers.length > 0 && (
            <div>
              <div className="label mb-1.5">Syllabus items covered</div>
              <ul className="space-y-1 text-muted">
                {covers.map((u) => <li key={u} className="flex gap-2"><span className="mt-[8px] h-1 w-1 shrink-0 rounded-full bg-faint" />{u}</li>)}
              </ul>
            </div>
          )}
          <div>
            <div className="label mb-1.5">Used in AI/ML</div>
            <ul className="space-y-1 text-muted">
              {c.usedIn.map((u) => <li key={u} className="flex gap-2"><span className="mt-[8px] h-1 w-1 shrink-0 rounded-full bg-faint" />{u}</li>)}
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

      <div className="mt-10 space-y-12">
        <Section id="visual" title="Interactive experiment">
          <VisualSlot key={topic.key} id={c.visual.id} props={c.visual.props} />
          {c.visual.caption && <p className="mt-2.5 max-w-[80ch] text-[13px] leading-relaxed text-muted">{c.visual.caption}</p>}
        </Section>

        <Section id="math" title="Mathematical formulation">
          {c.notation && <p className="mb-4 max-w-[80ch] rounded-md border border-line bg-raised/40 px-3.5 py-2.5 text-[13px] leading-relaxed text-muted"><span className="font-medium text-fg/90">Notation. </span>{c.notation}</p>}
          <div className="grid gap-3 md:grid-cols-2">
            {c.formulas.map((f) => (
              <div key={f.label} className="panel min-w-0 p-4">
                <div className="label mb-2">{f.label}</div>
                <div className="math-scroll py-1"><Tex tex={f.tex} block /></div>
                <p className="mt-2.5 text-[13.5px] leading-relaxed text-muted">{f.explanation}</p>
                {f.variables && (
                  <dl className="mt-3 grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-1.5 border-t border-line pt-3 text-[13px]">
                    {f.variables.map((v) => (
                      <div key={v.symbol} className="contents">
                        <dt className="text-fg/90"><Tex tex={v.symbol} /></dt>
                        <dd className="leading-relaxed text-muted">{v.meaning}</dd>
                      </div>
                    ))}
                  </dl>
                )}
              </div>
            ))}
          </div>
          {shownNotes.length > 0 && (
            <div className={`mt-3 grid gap-3 ${shownNotes.length > 1 ? "md:grid-cols-2" : ""} ${shownNotes.length > 2 ? "lg:grid-cols-3" : ""}`}>
              {shownNotes.map(([title, list]) => (
                <div key={title} className="rounded-lg border border-line p-4">
                  <div className="label mb-2">{title}</div>
                  <ul className="space-y-2 text-[13.5px] leading-relaxed text-muted">
                    {list!.map((a) => <li key={a} className="flex gap-2"><span className="mt-[9px] h-1 w-1 shrink-0 rounded-full bg-faint" />{a}</li>)}
                  </ul>
                </div>
              ))}
            </div>
          )}
        </Section>

        {c.derivation && (
          <Section id="derivation" title={`Derivation — ${c.derivation.title}`}>
            <div className="panel p-4 sm:p-5">
              {c.derivation.setup && <p className="mb-4 max-w-[80ch] text-[13.5px] leading-relaxed text-muted">{c.derivation.setup}</p>}
              <Steps steps={c.derivation.steps} />
              {c.derivation.conclusion && <p className="mt-4 max-w-[80ch] border-t border-line pt-3 text-[13.5px] leading-relaxed text-fg/90">{c.derivation.conclusion}</p>}
            </div>
          </Section>
        )}

        {c.worked && (
          <Section id="worked" title={`Worked example — ${c.worked.title}`} aside="Fixed inputs; the experiment above recomputes live.">
            <div className="panel p-4 sm:p-5">
              <Steps steps={c.worked.steps} />
              {c.worked.result && (
                <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-line pt-3">
                  <span className="label">Result</span>
                  <div className="math-scroll min-w-0"><Tex tex={c.worked.result} /></div>
                </div>
              )}
            </div>
          </Section>
        )}

        {c.code && (
          <Section id="code" title={c.code.rust ? "Implementation in Python and Rust" : "Implementation in Python"}>
            <CodeTabs python={{ html: highlight(c.code.python, "python"), raw: c.code.python }} rust={c.code.rust ? { html: highlight(c.code.rust, "rust"), raw: c.code.rust } : undefined} output={c.code.output} explanation={c.code.explanation} />
          </Section>
        )}

        <Section id="apps" title="Where it shows up in AI/ML">
          <div className="grid gap-3 md:grid-cols-2">
            {c.applications.map((a) => (
              <div key={a.title} className="rounded-lg border border-line p-4">
                <div className="text-[14px] font-medium">{a.title}</div>
                <p className="mt-1.5 text-[13.5px] leading-relaxed text-muted">{a.body}</p>
              </div>
            ))}
          </div>
        </Section>

        {related.length > 0 && (
          <Section id="related" title="Related concepts">
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {related.map((k) => <TopicLink key={k} k={k} />)}
            </div>
          </Section>
        )}

        <PrevNext topic={topic} />
      </div>
    </article>
  );
}
