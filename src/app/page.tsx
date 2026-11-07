import Link from "next/link";
import { ArrowRight, FlaskConical } from "lucide-react";
import { modules, moduleStats, topics } from "@/data/curriculum";
import { labs } from "@/data/labs";
import { HeroCanvas } from "@/components/common/HeroCanvas";
import { ContinueExploring } from "@/components/common/ContinueExploring";
import { ProgressBadge } from "@/components/common/ProgressBadge";

const diff = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced" } as const;
const featured = ["pca", "gradient-descent", "neural-network", "attention", "matrix-transformations", "transformer"];

export default function Home() {
  const complete = topics.filter((t) => t.status === "complete").length;
  return (
    <div className="pb-16">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-line">
        <HeroCanvas />
        <div className="relative mx-auto max-w-6xl px-4 py-16 sm:px-6 sm:py-24">
          <div className="max-w-xl">
            <div className="label mb-4 text-fg">ML Visual Lab</div>
            <h1 className="text-[34px] font-semibold leading-[1.1] tracking-tight sm:text-[44px]">Understand the Mathematics Behind Intelligence.</h1>
            <p className="mt-4 max-w-lg text-[16px] leading-relaxed text-muted">An interactive visual laboratory for exploring mathematics, machine learning, deep learning, and modern AI engineering.</p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link href="/playground" className="btn-primary"><FlaskConical size={16} /> Explore Visual Lab</Link>
              <Link href="/topics" className="btn h-9 px-4 text-sm">Browse Curriculum <ArrowRight size={15} /></Link>
            </div>
            <div className="mt-8 flex gap-6 text-[12.5px] text-faint">
              <span><span className="num text-fg">{complete}</span> interactive lessons</span>
              <span><span className="num text-fg">{labs.length}</span> playground labs</span>
              <span><span className="num text-fg">{modules.length}</span> modules</span>
            </div>
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-6xl space-y-14 px-4 pt-12 sm:px-6">
        {/* Continue exploring */}
        <section aria-labelledby="continue-h">
          <h2 id="continue-h" className="mb-4 text-[17px] font-semibold tracking-tight">Continue exploring</h2>
          <ContinueExploring />
        </section>

        {/* Featured visualizations */}
        <section aria-labelledby="featured-h">
          <div className="mb-4 flex items-baseline justify-between">
            <h2 id="featured-h" className="text-[17px] font-semibold tracking-tight">Featured visualizations</h2>
            <Link href="/playground" className="text-[13px] text-muted hover:text-fg">All labs →</Link>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {featured.map((s) => labs.find((l) => l.slug === s)!).map((l) => (
              <Link key={l.slug} href={`/playground/${l.slug}`} className="group rounded-lg border border-line bg-panel p-4 transition-colors hover:border-line-strong">
                <div className="flex items-center justify-between">
                  <span className="chip">{l.engine}</span>
                  <ArrowRight size={14} className="text-faint transition-transform group-hover:translate-x-0.5 group-hover:text-fg" />
                </div>
                <div className="mt-3 text-[14.5px] font-medium">{l.title}</div>
                <p className="mt-1 text-[13px] leading-relaxed text-muted">{l.blurb}</p>
              </Link>
            ))}
          </div>
        </section>

        {/* Learning categories */}
        <section aria-labelledby="modules-h">
          <h2 id="modules-h" className="mb-4 text-[17px] font-semibold tracking-tight">Learning modules</h2>
          {(["foundations", "data-ml", "advanced"] as const).map((g) => (
            <div key={g} className="mb-6">
              <div className="label mb-2">{g === "foundations" ? "Mathematical Foundations" : g === "data-ml" ? "Data & Machine Learning" : "Advanced AI"}</div>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {modules.filter((m) => m.group === g).map((m) => {
                  const st = moduleStats(m.id);
                  return (
                    <Link key={m.id} href={`/${m.id}`} className="group flex flex-col rounded-lg border border-line p-4 transition-colors hover:border-line-strong hover:bg-panel">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[14.5px] font-medium">{m.title}</span>
                        <span className="chip">{diff[m.difficulty]}</span>
                      </div>
                      <p className="mt-1.5 flex-1 text-[13px] leading-relaxed text-muted">{m.short}</p>
                      <div className="mt-3 flex flex-wrap items-center gap-2 text-[11.5px] text-faint">
                        <span><span className="num text-fg">{st.complete}</span> interactive / {st.total} topics</span>
                        <div className="h-1 flex-1 rounded bg-raised"><div className="h-1 rounded bg-accent/60" style={{ width: `${(st.complete / st.total) * 100}%` }} /></div>
                        <ProgressBadge keys={topics.filter((t) => t.module === m.id).map((t) => t.key)} />
                      </div>
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </section>
      </div>
    </div>
  );
}
