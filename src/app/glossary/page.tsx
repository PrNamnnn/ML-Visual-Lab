import type { Metadata } from "next";
import Link from "next/link";
import { topics } from "@/data/curriculum";

export const metadata: Metadata = { title: "Glossary" };

const firstSentence = (s: string) => { const m = s.match(/^.*?[.!?](\s|$)/); return (m ? m[0] : s).trim(); };

export default function Glossary() {
  const entries = topics
    .filter((t) => t.content)
    .map((t) => ({ term: t.title, def: firstSentence(t.content!.definition), href: t.href, module: t.moduleTitle }))
    .sort((a, b) => a.term.localeCompare(b.term));
  const letters = [...new Set(entries.map((e) => e.term[0].toUpperCase()))];
  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <div className="label">Resources</div>
      <h1 className="mt-1 text-[28px] font-semibold tracking-tight">Glossary</h1>
      <p className="mt-2 text-[15px] text-muted">{entries.length} terms, each linked to its interactive lesson.</p>
      <nav className="mt-5 flex flex-wrap gap-1" aria-label="Jump to letter">
        {letters.map((l) => <a key={l} href={`#letter-${l}`} className="chip w-7 justify-center hover:border-line-strong hover:text-fg">{l}</a>)}
      </nav>
      <div className="mt-8 space-y-8">
        {letters.map((l) => (
          <section key={l} id={`letter-${l}`} className="scroll-mt-16">
            <h2 className="mb-2 text-[17px] font-semibold text-fg">{l}</h2>
            <dl className="space-y-3">
              {entries.filter((e) => e.term[0].toUpperCase() === l).map((e) => (
                <div key={e.href}>
                  <dt><Link href={e.href} className="text-[14px] font-medium hover:text-fg">{e.term}</Link> <span className="text-[11.5px] text-faint">· {e.module}</span></dt>
                  <dd className="mt-0.5 text-[13px] leading-relaxed text-muted">{e.def}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </div>
  );
}
