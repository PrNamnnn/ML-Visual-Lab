export type Difficulty = "beginner" | "intermediate" | "advanced";

export type ModuleId =
  | "linear-algebra"
  | "calculus"
  | "statistics"
  | "optimization"
  | "preprocessing"
  | "feature-engineering"
  | "machine-learning"
  | "model-evaluation"
  | "deep-learning"
  | "computer-vision"
  | "nlp"
  | "transformers"
  | "ai-engineering";

export interface FormulaBlock {
  label: string;
  tex: string;
  explanation: string;
  variables?: { symbol: string; meaning: string }[];
}

export interface WorkedStep {
  tex: string;
  note?: string;
}

export interface VisualRef {
  /** key into the visualizer registry */
  id: string;
  /** preset/configuration passed to the visualizer */
  props?: Record<string, unknown>;
  caption?: string;
}

/** Python and Rust programs that print identical output (verified by tests/code-run.test.ts). */
export interface CodeExample {
  python: string;
  rust: string;
  /** exact expected stdout of both programs */
  output: string;
  /** what the code does and how it maps to the formula */
  explanation: string;
}

export interface TopicContent {
  definition: string;
  why: string;
  usedIn: string[];
  visual: VisualRef;
  formulas: FormulaBlock[];
  assumptions?: string[];
  specialCases?: string[];
  limitations?: string[];
  worked?: { title: string; steps: WorkedStep[]; result?: string };
  applications: { title: string; body: string }[];
  code?: CodeExample;
}

export interface TopicDef {
  slug: string;
  title: string;
  summary?: string;
  difficulty?: Difficulty;
  prerequisites?: string[]; // topic keys "module/slug"
  related?: string[];
  keywords?: string[];
  content?: TopicContent;
}

export interface SectionDef {
  id: string;
  title: string;
  topics: TopicDef[];
}

export interface ModuleDef {
  id: ModuleId;
  title: string;
  short: string;
  description: string;
  group: "foundations" | "data-ml" | "advanced";
  difficulty: Difficulty;
  sections: SectionDef[];
}

/** Flattened topic with resolved module/section information. */
export interface Topic extends TopicDef {
  key: string; // `${module}/${slug}`
  module: ModuleId;
  moduleTitle: string;
  section: string;
  href: string;
  status: "complete" | "planned";
  difficulty: Difficulty;
}
