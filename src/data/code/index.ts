import type { CodeExample } from "@/types/curriculum";
import { linearAlgebraCode } from "./linear-algebra";
import { calculusCode } from "./calculus";
import { statisticsCode } from "./statistics";
import { optimizationCode } from "./optimization";
import { preprocessingCode, featureEngineeringCode } from "./preprocessing";
import { machineLearningCode, modelEvaluationCode } from "./machine-learning";
import { deepLearningCode } from "./deep-learning";
import { computerVisionCode, nlpCode, transformersCode } from "./vision-language";
import { aiEngineeringCode } from "./ai-engineering";

const byModule: Record<string, Record<string, CodeExample>> = {
  "linear-algebra": linearAlgebraCode,
  calculus: calculusCode,
  statistics: statisticsCode,
  optimization: optimizationCode,
  preprocessing: preprocessingCode,
  "feature-engineering": featureEngineeringCode,
  "machine-learning": machineLearningCode,
  "model-evaluation": modelEvaluationCode,
  "deep-learning": deepLearningCode,
  "computer-vision": computerVisionCode,
  nlp: nlpCode,
  transformers: transformersCode,
  "ai-engineering": aiEngineeringCode,
};

/** All code examples keyed by "module/slug". */
export const codeExamples: Record<string, CodeExample> = Object.fromEntries(
  Object.entries(byModule).flatMap(([m, rec]) => Object.entries(rec).map(([slug, c]) => [`${m}/${slug}`, c])),
);
