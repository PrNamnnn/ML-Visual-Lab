import type { Metadata } from "next";
import Link from "next/link";
import { topicByKey } from "@/data/curriculum";
import { PathStep } from "@/components/topic/PathStep";

export const metadata: Metadata = { title: "Learning Path" };

const STAGES: { title: string; goal: string; keys: string[] }[] = [
  { title: "1 · Vectors and matrices", goal: "Read data and layers as vectors and matrices; see matrices as transformations.", keys: ["linear-algebra/vectors", "linear-algebra/dot-product", "linear-algebra/projection", "linear-algebra/matrix-multiplication", "linear-algebra/linear-transformations", "linear-algebra/determinants", "linear-algebra/matrix-inverse"] },
  { title: "2 · Structure in matrices", goal: "Rank, eigenvectors and SVD — the tools behind PCA and compression.", keys: ["linear-algebra/rank", "linear-algebra/eigenvalues-eigenvectors", "linear-algebra/svd", "linear-algebra/covariance-matrix", "preprocessing/pca"] },
  { title: "3 · Calculus for learning", goal: "Derivatives, gradients and the chain rule that powers backpropagation.", keys: ["calculus/derivatives", "calculus/chain-rule", "calculus/gradient", "calculus/hessian", "calculus/computational-graphs"] },
  { title: "4 · Optimisation", goal: "How gradient-based training actually moves parameters.", keys: ["optimization/gradient-descent", "optimization/momentum", "optimization/adam", "optimization/learning-rate-scheduling"] },
  { title: "5 · Probability and statistics", goal: "Uncertainty, distributions, estimation and testing.", keys: ["statistics/conditional-probability", "statistics/bayes-theorem", "statistics/random-variables", "statistics/normal-distribution", "statistics/central-limit-theorem", "statistics/maximum-likelihood", "statistics/hypothesis-testing"] },
  { title: "6 · Working with data", goal: "Clean, scale, transform and encode features without leakage.", keys: ["preprocessing/missing-data", "preprocessing/feature-scaling", "preprocessing/log-transformation", "preprocessing/box-cox-transformation", "preprocessing/one-hot-encoding", "machine-learning/data-leakage"] },
  { title: "7 · Classical machine learning", goal: "Regression, classification, ensembles and clustering.", keys: ["machine-learning/linear-regression", "machine-learning/bias-variance", "machine-learning/ridge-regression", "machine-learning/logistic-regression", "machine-learning/decision-trees", "machine-learning/support-vector-machines", "machine-learning/gradient-boosting", "machine-learning/k-means"] },
  { title: "8 · Evaluating models", goal: "Choose metrics and validation schemes that reflect reality.", keys: ["model-evaluation/confusion-matrix", "model-evaluation/roc-auc", "model-evaluation/precision-recall-curve", "model-evaluation/cross-validation"] },
  { title: "9 · Deep learning", goal: "Neurons, backprop, training pathologies, CNNs and RNNs.", keys: ["deep-learning/artificial-neuron", "deep-learning/multilayer-perceptron", "deep-learning/backpropagation", "deep-learning/vanishing-gradients", "deep-learning/convolution", "deep-learning/lstm"] },
  { title: "10 · Transformers and LLMs", goal: "Tokens, embeddings, attention and the full Transformer block.", keys: ["nlp/tokenization", "nlp/word-embeddings", "nlp/positional-encoding", "transformers/self-attention", "transformers/multi-head-attention", "transformers/transformer-architecture", "transformers/sampling-strategies", "transformers/lora"] },
  { title: "11 · AI engineering", goal: "Retrieval, inference efficiency and monitoring in production.", keys: ["ai-engineering/cosine-similarity", "ai-engineering/rag", "ai-engineering/kv-caching", "ai-engineering/quantization", "ai-engineering/data-drift"] },
];

export default function LearningPath() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <div className="label">Learning path</div>
      <h1 className="mt-1 text-[28px] font-semibold tracking-tight">A route from vectors to LLMs</h1>
      <p className="mt-2 max-w-2xl text-[15px] text-muted">A suggested order through the interactive lessons. Each stage builds on the previous one; tick topics off as you learn them (saved in this browser).</p>
      <ol className="mt-8 space-y-6">
        {STAGES.map((s) => (
          <li key={s.title} className="rounded-lg border border-line p-4">
            <div className="text-[15px] font-semibold">{s.title}</div>
            <p className="mt-0.5 text-[13px] text-muted">{s.goal}</p>
            <div className="mt-3 grid gap-1.5 sm:grid-cols-2">
              {s.keys.map((k) => { const t = topicByKey(k); return t ? <PathStep key={k} k={k} title={t.title} href={t.href} /> : null; })}
            </div>
          </li>
        ))}
      </ol>
      <p className="mt-6 text-[13px] text-faint">Prefer to wander? Use <Link href="/topics" className="text-fg hover:underline">All topics</Link> or the <Link href="/playground" className="text-fg hover:underline">Visual Playground</Link>.</p>
    </div>
  );
}
