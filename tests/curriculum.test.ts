import { describe, it, expect } from "vitest";
import { topics, modules, topicByKey } from "@/data/curriculum";
import { labs } from "@/data/labs";
import { visualizerIds } from "@/visualizers/ids";
import { search } from "@/lib/utilities/search";
import katex from "katex";

const ids = new Set<string>(visualizerIds);

describe("curriculum registry", () => {
  it("has unique topic keys", () => {
    const keys = topics.map((t) => t.key);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("every prerequisite / related reference resolves", () => {
    const broken: string[] = [];
    for (const t of topics) for (const r of [...(t.prerequisites ?? []), ...(t.related ?? [])]) if (!topicByKey(r)) broken.push(`${t.key} → ${r}`);
    expect(broken).toEqual([]);
  });

  it("every complete topic references a registered visualizer and has formulas + applications", () => {
    const bad: string[] = [];
    for (const t of topics.filter((x) => x.content)) {
      const c = t.content!;
      if (!ids.has(c.visual.id)) bad.push(`${t.key}: unknown visual ${c.visual.id}`);
      if (!c.formulas.length) bad.push(`${t.key}: no formulas`);
      if (!c.applications.length) bad.push(`${t.key}: no applications`);
      if (!c.definition || !c.why) bad.push(`${t.key}: missing text`);
    }
    expect(bad).toEqual([]);
  });

  it("every formula renders in KaTeX without errors", () => {
    const bad: string[] = [];
    const check = (key: string, tex: string) => {
      try { katex.renderToString(tex, { throwOnError: true, strict: "ignore" }); } catch (e) { bad.push(`${key}: ${(e as Error).message.slice(0, 120)}`); }
    };
    for (const t of topics.filter((x) => x.content)) {
      for (const f of t.content!.formulas) { check(t.key, f.tex); f.variables?.forEach((v) => check(t.key, v.symbol)); }
      t.content!.worked?.steps.forEach((s) => check(t.key + " (worked)", s.tex));
      t.content!.derivation?.steps.forEach((s) => check(t.key + " (derivation)", s.tex));
      if (t.content!.worked?.result) check(t.key + " (result)", t.content!.worked.result);
    }
    expect(bad).toEqual([]);
  });

  it("every module has at least one complete topic", () => {
    for (const m of modules) expect(topics.some((t) => t.module === m.id && t.status === "complete")).toBe(true);
  });

  it("labs reference registered visualizers and existing topics", () => {
    for (const l of labs) {
      expect(ids.has(l.visual)).toBe(true);
      for (const k of l.topics) expect(topicByKey(k), `${l.slug} → ${k}`).toBeDefined();
    }
    expect(labs.length).toBe(16);
  });
});

describe("IITM MLT weeks inside Linear Algebra", () => {
  const la = modules.find((m) => m.id === "linear-algebra")!;
  const weeks = la.sections.filter((s) => s.week !== undefined);
  // number of syllabus lines per week in the course outline (91 in total)
  const expected = [9, 7, 7, 7, 8, 7, 7, 7, 9, 6, 8, 9];

  it("has the twelve weeks in order, all inside the Linear Algebra module", () => {
    expect(weeks.map((s) => s.week)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(weeks.every((s) => s.part === "mlt")).toBe(true);
    expect(modules.some((m) => m.id !== "linear-algebra" && m.sections.some((s) => s.week !== undefined))).toBe(false);
    expect(la.parts?.map((p) => p.id)).toEqual(["foundations", "mlt", "additional"]);
    expect(la.sections.every((s) => la.parts!.some((p) => p.id === s.part))).toBe(true);
  });

  it("maps every syllabus line of every week to a lesson of that week", () => {
    const bad: string[] = [];
    weeks.forEach((s, i) => {
      if (s.syllabus?.length !== expected[i]) bad.push(`week ${s.week}: ${s.syllabus?.length} syllabus lines, expected ${expected[i]}`);
      for (const it of s.syllabus ?? []) if (!s.topics.some((t) => t.slug === it.slug)) bad.push(`week ${s.week}: "${it.item}" → ${it.slug} is not a lesson of this week`);
      for (const t of s.topics) if (!s.syllabus?.some((it) => it.slug === t.slug)) bad.push(`week ${s.week}: lesson ${t.slug} covers no syllabus line`);
    });
    expect(bad).toEqual([]);
  });

  it("gives every week lesson intuition, variables, a derivation or worked example, limits, code and an existing visual", () => {
    const bad: string[] = [];
    for (const s of weeks) for (const t of s.topics) {
      const c = topicByKey(`linear-algebra/${t.slug}`)!.content;
      if (!c) { bad.push(`${t.slug}: no content`); continue; }
      if (!c.intuition) bad.push(`${t.slug}: no intuition`);
      if (!c.formulas.some((f) => f.variables?.length)) bad.push(`${t.slug}: no variable explanations`);
      if (!c.derivation && !c.worked) bad.push(`${t.slug}: neither derivation nor worked example`);
      if (!c.worked) bad.push(`${t.slug}: no worked example`);
      if (!c.limitations?.length && !c.assumptions?.length) bad.push(`${t.slug}: no assumptions or limitations`);
      if (!c.code?.python) bad.push(`${t.slug}: no Python implementation`);
      if (c.applications.length < 2) bad.push(`${t.slug}: fewer than two applications`);
    }
    expect(bad).toEqual([]);
  });

  it("keeps the original Linear Algebra lessons", () => {
    for (const slug of ["vectors", "dot-product", "matrix-multiplication", "determinants", "eigenvalues-eigenvectors", "svd", "least-squares"]) expect(topicByKey(`linear-algebra/${slug}`)?.status).toBe("complete");
    expect(la.sections.filter((s) => s.part === "foundations").flatMap((s) => s.topics).length).toBe(47);
  });
});

describe("search", () => {
  const items = topics.map((t) => ({ key: t.key, title: t.title, href: t.href, context: t.section, k: (t.keywords ?? []).join(" | ").toLowerCase(), summary: t.summary, status: t.status }));
  const top = (q: string) => search(items, q, 5).map((r) => r.key);
  it("answers natural-language queries", () => {
    expect(top("How does PCA work?")[0]).toBe("preprocessing/pca");
    expect(top("Show determinant geometrically.")).toContain("linear-algebra/determinants");
    expect(top("Log transformation.")[0]).toBe("preprocessing/log-transformation");
    expect(top("Adam optimizer.")[0]).toBe("optimization/adam");
    expect(top("How does attention work?")[0]).toBe("transformers/self-attention");
  });
});
