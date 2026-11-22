import type { ModuleDef, ModuleId, Topic } from "@/types/curriculum";
import { linearAlgebra } from "./linear-algebra";
import { calculus } from "./calculus";
import { statistics } from "./statistics";
import { optimization } from "./optimization";
import { preprocessing, featureEngineering } from "./preprocessing";
import { machineLearning, modelEvaluation } from "./machine-learning";
import { deepLearning } from "./deep-learning";
import { computerVision } from "./computer-vision";
import { nlp, transformers } from "./nlp";
import { aiEngineering } from "./ai-engineering";
import { codeExamples } from "../code";
import { matrixCalculus, jointMarginal, convexSets } from "./math-extra";
import type { TopicDef } from "@/types/curriculum";

/**
 * Lessons added by the mathematics gap audit that belong to other modules than Linear Algebra.
 * They are appended to an existing section of their home module (once: the guard keeps this idempotent).
 */
function addTopic(m: ModuleDef, sectionId: string, topic: TopicDef, after?: string) {
  const sec = m.sections.find((x) => x.id === sectionId);
  if (!sec || sec.topics.some((t) => t.slug === topic.slug)) return;
  const i = after ? sec.topics.findIndex((t) => t.slug === after) : -1;
  if (i >= 0) sec.topics.splice(i + 1, 0, topic);
  else sec.topics.push(topic);
}
addTopic(calculus, "multivariable", matrixCalculus, "hessian");
addTopic(statistics, "random-variables", jointMarginal, "random-variables");
addTopic(optimization, "fundamentals", convexSets, "objective-functions");

export const modules: ModuleDef[] = [
  linearAlgebra,
  calculus,
  statistics,
  optimization,
  preprocessing,
  featureEngineering,
  machineLearning,
  modelEvaluation,
  deepLearning,
  computerVision,
  nlp,
  transformers,
  aiEngineering,
];

export const moduleById = (id: string) => modules.find((m) => m.id === id);

export const topics: Topic[] = modules.flatMap((m) =>
  m.sections.flatMap((s) =>
    s.topics.map((t) => ({
      ...t,
      content: t.content ? { ...t.content, code: t.content.code ?? codeExamples[`${m.id}/${t.slug}`] } : undefined,
      key: `${m.id}/${t.slug}`,
      module: m.id as ModuleId,
      moduleTitle: m.title,
      section: s.title,
      sectionId: s.id,
      week: s.week,
      href: `/${m.id}/${t.slug}`,
      status: t.content ? ("complete" as const) : ("planned" as const),
      difficulty: t.difficulty ?? m.difficulty,
    })),
  ),
);

const byKey = new Map(topics.map((t) => [t.key, t]));
export const topicByKey = (key: string) => byKey.get(key);

/** Previous / next lesson inside the same module (curriculum order). */
export function neighbours(key: string): { prev?: Topic; next?: Topic } {
  const t = byKey.get(key);
  if (!t) return {};
  const inModule = topics.filter((x) => x.module === t.module);
  const i = inModule.findIndex((x) => x.key === key);
  return { prev: inModule[i - 1], next: inModule[i + 1] };
}

export const sectionOf = (t: Topic) => moduleById(t.module)?.sections.find((s) => s.id === t.sectionId);

export function moduleStats(id: ModuleId) {
  const ts = topics.filter((t) => t.module === id);
  return { total: ts.length, complete: ts.filter((t) => t.status === "complete").length };
}

/** Lightweight index shipped to the client for navigation and search. */
export interface NavTopic {
  key: string;
  title: string;
  href: string;
  module: ModuleId;
  section: string;
  status: "complete" | "planned";
  difficulty: string;
  summary?: string;
  /** search text: keywords + formula labels */
  k: string;
}

export interface NavModule {
  id: ModuleId;
  title: string;
  group: ModuleDef["group"];
  parts?: { id: string; title: string }[];
  sections: { id: string; title: string; part?: string; week?: number; topics: string[] }[];
}

export function navIndex(): { topics: NavTopic[]; modules: NavModule[] } {
  return {
    topics: topics.map((t) => ({
      key: t.key,
      title: t.title,
      href: t.href,
      module: t.module,
      section: t.section,
      status: t.status,
      difficulty: t.difficulty,
      summary: t.summary,
      k: [...(t.keywords ?? []), ...(t.content?.formulas.map((f) => f.label) ?? [])].join(" | ").toLowerCase(),
    })),
    modules: modules.map((m) => ({
      id: m.id,
      title: m.title,
      group: m.group,
      parts: m.parts?.map((p) => ({ id: p.id, title: p.title })),
      sections: m.sections.map((s) => ({ id: s.id, title: s.title, part: s.part, week: s.week, topics: s.topics.map((t) => `${m.id}/${t.slug}`) })),
    })),
  };
}
