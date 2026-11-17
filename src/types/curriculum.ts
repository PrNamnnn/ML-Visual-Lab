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

/**
 * Runnable reference implementation (verified by tests/code-run.test.ts).
 * Every example has Python; the original lessons also carry a Rust program printing identical output.
 */
export interface CodeExample {
  python: string;
  rust?: string;
  /** exact expected stdout of the program(s) */
  output: string;
  /** what the code does and how it maps to the formula */
  explanation: string;
}

/** A step-by-step derivation or proof, rendered between the formulas and the worked example. */
export interface DerivationBlock {
  title: string;
  /** what is assumed before the first step */
  setup?: string;
  steps: WorkedStep[];
  conclusion?: string;
}

export interface TopicContent {
  definition: string;
  /** plain-language picture of the idea, before any notation */
  intuition?: string;
  why: string;
  usedIn: string[];
  visual: VisualRef;
  formulas: FormulaBlock[];
  derivation?: DerivationBlock;
  /** notation conventions that differ from the rest of the site (e.g. the d × n data matrix of the MLT notes) */
  notation?: string;
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

/** One syllabus line of a course week and the lesson (slug) that teaches it. */
export interface SyllabusItem {
  item: string;
  slug: string;
}

export interface SectionDef {
  id: string;
  title: string;
  /** id of the module part this section belongs to (see ModuleDef.parts) */
  part?: string;
  /** course week number, for sections that follow an external weekly syllabus */
  week?: number;
  blurb?: string;
  /** syllabus coverage map: every line of the week's syllabus and where it is taught */
  syllabus?: SyllabusItem[];
  topics: TopicDef[];
}

/** A named group of sections inside one module. */
export interface ModulePart {
  id: string;
  title: string;
  description?: string;
}

export interface ModuleDef {
  id: ModuleId;
  title: string;
  short: string;
  description: string;
  group: "foundations" | "data-ml" | "advanced";
  difficulty: Difficulty;
  /** optional grouping of sections; sections reference a part by id */
  parts?: ModulePart[];
  sections: SectionDef[];
}

/** Flattened topic with resolved module/section information. */
export interface Topic extends TopicDef {
  key: string; // `${module}/${slug}`
  module: ModuleId;
  moduleTitle: string;
  section: string;
  sectionId: string;
  /** course week of the section, when it has one */
  week?: number;
  href: string;
  status: "complete" | "planned";
  difficulty: Difficulty;
}
