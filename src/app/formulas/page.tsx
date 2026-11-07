import type { Metadata } from "next";
import Link from "next/link";
import { modules, topics } from "@/data/curriculum";
import { Tex } from "@/components/mathematics/Tex";

export const metadata: Metadata = { title: "Formula Library" };

export default function Formulas() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <div className="label">Resources</div>
      <h1 className="mt-1 text-[28px] font-semibold tracking-tight">Formula library</h1>
      <p className="mt-2 max-w-2xl text-[15px] text-muted">Every formula from the interactive lessons, grouped by module. Each links back to its visualization. Use Ctrl+F or the global search (Ctrl K).</p>
      <nav className="mt-5 flex flex-wrap gap-1.5" aria-label="Jump to module">
        {modules.map((m) => <a key={m.id} href={`#${m.id}`} className="chip hover:border-line-strong hover:text-fg">{m.title}</a>)}
      </nav>
      <div className="mt-8 space-y-10">
        {modules.map((m) => {
          const ts = topics.filter((t) => t.module === m.id && t.content);
          return (
            <section key={m.id} id={m.id} className="scroll-mt-16">
              <h2 className="mb-3 text-[17px] font-semibold">{m.title}</h2>
              <div className="divide-y divide-line rounded-lg border border-line">
                {ts.flatMap((t) => t.content!.formulas.map((f, i) => (
                  <div key={`${t.key}-${i}`} className="grid gap-2 px-4 py-3 md:grid-cols-[220px_1fr]">
                    <div>
                      <div className="text-[13px] font-medium">{f.label}</div>
                      <Link href={t.href} className="text-[11.5px] text-faint hover:text-fg">{t.title}</Link>
                    </div>
                    <div className="min-w-0">
                      <div className="overflow-x-auto"><Tex tex={f.tex} /></div>
                      <p className="mt-1 text-[12px] text-muted">{f.explanation}</p>
                    </div>
                  </div>
                )))}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
