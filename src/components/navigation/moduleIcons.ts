import {
  Boxes, Brain, Calculator, Eye, FunctionSquare, Gauge, Languages, LineChart, Shapes, Sparkles, Table2, Workflow, Wrench, type LucideIcon,
} from "lucide-react";
import type { ModuleId } from "@/types/curriculum";

export const moduleIcon: Record<ModuleId, LucideIcon> = {
  "linear-algebra": Shapes,
  calculus: FunctionSquare,
  statistics: LineChart,
  optimization: Gauge,
  preprocessing: Table2,
  "feature-engineering": Wrench,
  "machine-learning": Boxes,
  "model-evaluation": Calculator,
  "deep-learning": Brain,
  "computer-vision": Eye,
  nlp: Languages,
  transformers: Sparkles,
  "ai-engineering": Workflow,
};

/** Search-result categories (palette groups). */
export const searchCategory: Record<ModuleId, string> = {
  "linear-algebra": "Mathematical Foundations",
  calculus: "Mathematical Foundations",
  optimization: "Mathematical Foundations",
  statistics: "Statistics",
  preprocessing: "Preprocessing",
  "feature-engineering": "Preprocessing",
  "machine-learning": "Machine Learning",
  "model-evaluation": "Machine Learning",
  "deep-learning": "Deep Learning",
  "computer-vision": "Deep Learning",
  nlp: "Deep Learning",
  transformers: "Deep Learning",
  "ai-engineering": "AI Engineering",
};

export const CATEGORY_ORDER = ["Mathematical Foundations", "Statistics", "Preprocessing", "Machine Learning", "Deep Learning", "AI Engineering", "Visual Playground", "Pages"];
