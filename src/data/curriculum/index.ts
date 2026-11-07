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
      href: `/${m.id}/${t.slug}`,
      status: t.content ? ("complete" as const) : ("planned" as const),
      difficulty: t.difficulty ?? m.difficulty,
    })),
  ),
);

const byKey = new Map(topics.map((t) => [t.key, t]));
export const topicByKey = (key: string) => byKey.get(key);

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
  sections: { title: string; topics: string[] }[];
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
    modules: modules.map((m) => ({ id: m.id, title: m.title, group: m.group, sections: m.sections.map((s) => ({ title: s.title, topics: s.topics.map((t) => `${m.id}/${t.slug}`) })) })),
  };
}
