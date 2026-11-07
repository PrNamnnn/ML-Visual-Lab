/** Every registered visualizer id. Tests verify that curriculum and labs only reference these. */
export const visualizerIds = [
  "vector2d", "vector3d", "span", "norms", "matrix-ops", "matrix-transform", "gauss", "svd", "surface3d", "pca", "regression",
  "derivative", "riemann", "taylor", "multivar", "comp-graph", "functions", "gd1d",
  "optimizer-race", "lr-schedule", "lagrange",
  "distribution", "clt", "bayes", "bayes-update", "inference", "correlation",
  "transform-lab", "scaling-lab", "data-tools",
  "poly-regression", "classifier", "clustering", "boosting", "evaluation",
  "activation", "loss", "neuron", "mlp", "backprop", "vanishing", "convolution", "recurrent",
  "image-filters", "iou-nms",
  "tokenizer", "embeddings", "positional-encoding",
  "attention", "transformer", "sampling", "lora", "quantization", "kv-cache", "drift", "rag",
  "implicit", "jacobian", "double-integral", "convexity",
  "dice", "cond-indep", "population-sample", "chi-square-test", "anova", "multiple-testing", "permutation", "partial-corr", "multicollinearity",
  "objective", "line-search",
  "kernel-pca", "truncated-svd", "lda-projection", "embedding-2d", "aggregation", "lag-features", "domain-features", "feature-filter", "mutual-info", "rfe",
  "poisson-reg", "quantile-reg", "robust-reg", "data-splits", "paradigms", "multiclass", "ensemble", "discriminant", "spectral", "density-cluster", "isolation-forest", "apriori", "matrix-factorization", "gp", "hyper-search",
  "init", "normalization", "dropout", "early-stopping", "cnn-arch", "transfer", "birnn", "seq2seq", "autoencoder", "vae", "gan", "diffusion", "unet",
  "feature-hierarchy", "detection", "segmentation", "vit", "image-embed",
  "text-prep", "vocabulary", "word2vec", "glove", "beam-search", "fine-tune", "distillation",
  "data-validation", "feature-store", "pipelines", "versioning", "experiments", "eval-pipeline", "serving", "vector-db", "hybrid-search", "prompting", "tool-calling", "llm-eval", "context-window",
] as const;

export type VisualizerId = (typeof visualizerIds)[number];
