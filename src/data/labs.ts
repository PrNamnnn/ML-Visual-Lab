/** Visual Playground laboratories. Each maps to a registered visualizer in "lab" mode. */
export interface Lab {
  slug: string;
  title: string;
  blurb: string;
  visual: string;
  props?: Record<string, unknown>;
  keywords?: string;
  engine: "SVG" | "Canvas" | "WebGL" | "Plotly";
  topics: string[]; // related topic keys
}

export const labs: Lab[] = [
  { slug: "vectors", title: "Vector Laboratory", blurb: "Add, scale, project and measure vectors by dragging them.", visual: "vector2d", props: { mode: "dot", lab: true }, keywords: "vector dot product projection angle", engine: "SVG", topics: ["linear-algebra/dot-product", "linear-algebra/projection", "linear-algebra/vector-addition"] },
  { slug: "matrix-transformations", title: "Matrix Transformation Laboratory", blurb: "Watch any 2×2 matrix bend the plane — grid, basis vectors and area.", visual: "matrix-transform", props: { mode: "transform", lab: true }, keywords: "linear transformation matrix grid shear rotation", engine: "SVG", topics: ["linear-algebra/linear-transformations", "linear-algebra/determinants"] },
  { slug: "determinant-inverse", title: "Determinant and Inverse Explorer", blurb: "Area scaling, singular collapse and undoing a transformation.", visual: "matrix-transform", props: { mode: "inverse", lab: true }, keywords: "determinant inverse singular", engine: "SVG", topics: ["linear-algebra/determinants", "linear-algebra/matrix-inverse"] },
  { slug: "eigenvectors", title: "Eigenvector Explorer", blurb: "Find directions that a matrix only stretches.", visual: "matrix-transform", props: { mode: "eigen", lab: true }, keywords: "eigenvalue eigenvector", engine: "SVG", topics: ["linear-algebra/eigenvalues-eigenvectors"] },
  { slug: "pca", title: "PCA Playground", blurb: "Centre, find principal directions, project and reconstruct.", visual: "pca", props: { lab: true }, keywords: "pca principal component analysis dimensionality reduction", engine: "SVG", topics: ["preprocessing/pca", "linear-algebra/covariance-matrix"] },
  { slug: "gradient-descent", title: "Gradient Descent Playground", blurb: "Race optimisers across loss landscapes and tune learning rates.", visual: "optimizer-race", props: { lab: true }, keywords: "gradient descent optimizer adam momentum learning rate", engine: "Canvas", topics: ["optimization/gradient-descent", "optimization/adam"] },
  { slug: "distributions", title: "Probability Distribution Playground", blurb: "Every major distribution with PDF/PMF, CDF and live sampling.", visual: "distribution", props: { lab: true, id: "normal" }, keywords: "distribution pdf cdf normal binomial poisson", engine: "SVG", topics: ["statistics/normal-distribution", "statistics/binomial-distribution"] },
  { slug: "sampling", title: "Statistical Sampling Laboratory", blurb: "Central limit theorem: sample means converge to a normal.", visual: "clt", props: { lab: true }, keywords: "central limit theorem sampling distribution", engine: "SVG", topics: ["statistics/central-limit-theorem", "statistics/sampling-distributions"] },
  { slug: "feature-transformations", title: "Feature Transformation Laboratory", blurb: "Log, Box-Cox, Yeo-Johnson and more on skewed data.", visual: "transform-lab", props: { lab: true, id: "box-cox" }, keywords: "log transformation box cox yeo johnson skewness", engine: "SVG", topics: ["preprocessing/log-transformation", "preprocessing/box-cox-transformation"] },
  { slug: "regression", title: "Regression Playground", blurb: "Fit lines and polynomials; regularise with Ridge and Lasso.", visual: "poly-regression", props: { lab: true, mode: "ridge" }, keywords: "regression polynomial ridge lasso overfitting", engine: "SVG", topics: ["machine-learning/linear-regression", "machine-learning/ridge-regression"] },
  { slug: "classification", title: "Classification Boundary Playground", blurb: "Logistic regression, KNN, Naive Bayes, trees, forests and SVMs.", visual: "classifier", props: { lab: true, algo: "knn" }, keywords: "classification decision boundary knn svm tree", engine: "Canvas", topics: ["machine-learning/k-nearest-neighbors", "machine-learning/support-vector-machines"] },
  { slug: "clustering", title: "Clustering Playground", blurb: "K-Means iterations, DBSCAN density and Gaussian mixtures.", visual: "clustering", props: { lab: true }, keywords: "clustering kmeans dbscan gmm", engine: "SVG", topics: ["machine-learning/k-means", "machine-learning/dbscan"] },
  { slug: "neural-network", title: "Neural Network Playground", blurb: "Train a small MLP live and watch its decision boundary form.", visual: "mlp", props: { lab: true }, keywords: "neural network mlp training backpropagation", engine: "Canvas", topics: ["deep-learning/multilayer-perceptron", "deep-learning/backpropagation"] },
  { slug: "convolution", title: "Convolution Playground", blurb: "Slide kernels over an input, with padding, stride and pooling.", visual: "convolution", props: { lab: true }, keywords: "convolution kernel cnn feature map stride padding", engine: "SVG", topics: ["deep-learning/convolution", "computer-vision/convolution-filters"] },
  { slug: "attention", title: "Attention Mechanism Playground", blurb: "Edit queries and keys; watch scores, softmax and outputs.", visual: "attention", props: { lab: true }, keywords: "attention self attention query key value softmax", engine: "SVG", topics: ["transformers/self-attention", "transformers/scaled-dot-product-attention"] },
  { slug: "transformer", title: "Transformer Architecture Playground", blurb: "Walk a sentence through every stage of a tiny Transformer.", visual: "transformer", props: { lab: true }, keywords: "transformer architecture llm next token", engine: "SVG", topics: ["transformers/transformer-architecture"] },
];

export const labBySlug = (s: string) => labs.find((l) => l.slug === s);
