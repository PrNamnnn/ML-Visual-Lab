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
