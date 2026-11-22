# ML Visual Lab — Project Status

_Last updated: 2026-11-14_

## Run it

```bash
npm install
npm run dev        # http://localhost:3000
npm run lint       # ESLint (next/core-web-vitals + next/typescript)
npm run typecheck  # strict TypeScript
npm test           # unit + integrity tests (vitest)
npm run test:code  # executes every Python and Rust snippet and checks its printed output
npm run build && npm start
```

No backend, no API keys, no network calls at runtime. Progress is stored in `localStorage` only.
`npm run test:code` needs `python` (3.10+) and `rustc` on the PATH.

## Summary

| | Count |
|---|---|
| Modules | 13 |
| Topics in curriculum | 371 |
| **Interactive lessons (complete)** | **371** |
| Placeholder lessons ("not built yet") | **0** |
| Topics with verified Python code | **371 / 371** |
| ...of which also verified in Rust | 312 (the 59 lessons added in 2.0 are Python only) |
| Visualizer components (registered ids) | 161 |
| Visual Playground labs | 16 / 16 |
| Statically generated pages | 409 |

A topic counts as **complete** only if it has: definition + motivation, KaTeX formulas with explanations, a working interactive visualization with controls, live calculations, a worked example, AI/ML applications, related-topic links, and Python + Rust code. `tests/curriculum.test.ts` enforces the structure, plus: every cross-reference resolves, every visualizer id is registered, and every formula renders in KaTeX without error.

## Version 2.0 — curriculum integration and visualization audit (2026-11-14)

- **IITM MLT inside Linear Algebra.** The Linear Algebra module now has three parts: *Foundations* (the original 47 lessons, unchanged), *IITM MLT — Weeks 1 to 12* (52 lessons, one section per week, in course order) and *Additional Mathematics* (4 lessons). There is no separate course, route or navigation item. Every week section carries a syllabus map (91 syllabus lines → lesson); `tests/curriculum.test.ts` fails if a line is unmapped or a week lesson lacks intuition, variable definitions, a worked example, limits or code.
- **Lesson format.** `TopicContent` gained `intuition`, `derivation` (numbered steps with remarks) and `notation`. Week lessons use the notes' d × n data-matrix convention and say so on the page.
- **New visualizers** (`src/visualizers/mlt/`, 27 registered ids, all lazily loaded): PCA (centring, direction sweep, change of basis, choosing k), XXᵀ/XᵀX duality, feature maps and kernels, k-means (objective, Lloyd half-steps, Voronoi geometry, k-means++), MLE, mixtures and EM, linear regression (normal equations, GD/SGD, projection and likelihood views), kernel regression, ridge/LASSO/MAP geometry, cross-validation, zero-one loss and margins, decision stumps, generative models, Naive Bayes, perceptron and its mistake bound, logistic regression, hard- and soft-margin SVM (exact dual coordinate ascent in `src/lib/ml/svmOrigin.ts`), bagging/random forests, AdaBoost, surrogate losses, forward propagation; plus subspaces, subspace projection, spectral decomposition, pseudoinverse, matrix calculus, joint/marginal tables and convex sets.
- **Mathematics gap audit.** Added: four fundamental subspaces, projection matrices, spectral theorem, pseudoinverse (Linear Algebra); matrix calculus (Calculus); joint/marginal/conditional distributions (Statistics); convex sets (Optimization). Everything else on the candidate list already had a lesson.
- **Fixes to existing visualizers.** The labels `\theta_W` and `\dim S` were written with single backslashes and rendered as "heta_W" / italic "dimS" (Vector2D inner-product mode, Span subspace mode); the inner-product validity line claimed "det W ≤ 0" even when only w₁₁ failed (now Sylvester's criterion); the power formula in the z-test view was the two-sided one for one-sided tests; the Beta–Binomial lab printed "undefined" for the MLE/MAP before any data. The escaping test now rejects any TeX command not followed by a letter boundary, which is what let the first two through.
- **UI.** Lesson typography (15 px / 1.7 body, ruled section headings, variable tables, left-aligned numbered display equations that scroll instead of overflowing), grouped and collapsible sidebar sections, module page with part index and per-week syllabus coverage, previous/next lesson links, breadcrumbs that link to the section.
- **Rust.** The 59 new lessons ship Python only (`CodeExample.rust` is optional); the original 312 keep both languages.

## Lesson completion checklist (by module)

| Module | Complete | Placeholders |
|---|---|---|
| Linear Algebra | 103 (47 foundations + 52 IITM MLT week lessons + 4 additions) | 0 |
| Calculus | 25 | 0 |
| Probability & Statistics | 47 | 0 |
| Optimization | 17 | 0 |
| Data Preprocessing | 31 | 0 |
| Feature Engineering | 9 | 0 |
| Machine Learning | 36 | 0 |
| Model Evaluation | 7 | 0 |
| Deep Learning | 46 | 0 |
| Computer Vision | 13 | 0 |
| NLP | 7 | 0 |
| Transformers & LLMs | 11 | 0 |
| AI Engineering | 19 | 0 |
| **Total** | **371** | **0** |

Lessons added in this round (94): vector spaces, sparse matrices, inner-product spaces; implicit differentiation, Jacobian, multivariable chain rule, total differential, double integrals, convex functions; sample spaces, axioms, conditional independence, population/sample, Bayesian estimation, χ² tests, ANOVA, multiple testing, permutation tests, partial correlation, multicollinearity; objective functions, constrained optimisation (KKT), quasi-Newton (BFGS), line search, coordinate descent; kernel PCA, truncated SVD, LDA projection, t-SNE, UMAP; aggregation, lag, domain, filter, mutual-information and wrapper (RFE) features; learning paradigms, train/val/test, regularisation, Poisson/quantile/robust regression, OvR/OvO, stacking, voting, spectral clustering, mean shift/OPTICS, LDA/QDA, Gaussian processes, isolation forest, Apriori, matrix factorisation, grid/random/Bayesian search; initialisation, batch/layer norm, dropout, early stopping, CNN architectures, transfer learning, bidirectional RNN, seq2seq, autoencoders, VAE, GAN, diffusion, U-Net; CNN feature hierarchy, R-CNN, YOLO, segmentation, ViT, CLIP-style embeddings; text preprocessing, vocabulary, word2vec, GloVe, cross-attention, beam search, fine-tuning, distillation; data validation, feature stores, pipelines, versioning, experiment tracking, evaluation gates, serving, vector DBs (IVF), hybrid search (BM25 + RRF), structured prompting, tool calling, LLM evaluation, context windows.

## Python + Rust implementations

- Every topic has a Python tab and a Rust tab (only these two languages), with syntax highlighting, a copy button, the expected output and an explanation. Highlighting is a ~100-line tokenizer (no editor dependency).
- Snippets use only the standard library and are self-contained (shared randomness uses the same tiny LCG in both languages so outputs match exactly).
- **Verification:** `tests/code-run.test.ts` (via `npm run test:code`) runs all 312 Python programs with `python -I`, compiles all 312 Rust programs into one crate with `rustc --edition 2021`, and requires both printed outputs to equal the documented output exactly. Last run: **312/312 Python and 312/312 Rust pass**.
- Every documented output and every number quoted in an explanation was checked against the real run; several explanations and two demos were corrected where the first draft's claims did not match the computed result.

## Verification done (this round)

- `npm run lint` — clean (0 errors, 0 warnings).
- `npx tsc --noEmit` — clean.
- `npx vitest run` — 66 passed, 2 skipped (the 2 skipped are the code-execution tests, run separately: 3/3 pass with `RUN_CODE=1`).
- `npm run build` — succeeds, 350 static pages.
- Browser sweep (dev server) of all 94 new lesson pages: every page renders its visualizer and readouts with no error-boundary trips. Several visualizers were corrected after checking their numbers (kernel PCA component choice, t-SNE optimiser bug, spectral clustering eigen-solver, RFE data size, ensemble base models, CNN feature detectors, R-CNN regression).
- R3F lifecycle fix verified in Strict Mode (navigation in/out of all 3D pages, browser back): no "synchronously unmount" or `removeChild` errors.

## Bugs fixed this round (root causes)

1. **R3F "synchronously unmount a root" + `removeChild`** — drei `<Html>` in `Vector3D.tsx` created a separate React root per label inside the Canvas wrapper. Replaced with disposed canvas-texture sprites; no `Html`/`createRoot`/`createPortal` remain.
2. **Malformed math (`egin{bmatrix}`, `imes`)** — single-backslash TeX in ordinary JS strings in `format.ts`. Fixed with `String.raw`; `tests/escaping.test.ts` prevents regressions.
3. **State leaking between topics** that share a visualizer — `VisualSlot` is keyed per topic/lab.
4. **t-SNE divergence** — points were updated in place mid-pass and the iteration counter lagged between ticks; gradients are now computed from one snapshot and the counter lives in a ref.
5. **Spectral clustering at 85 %** — power iteration could not separate near-equal Laplacian eigenvalues; replaced by an exact Jacobi solver (`jacobiEig`) and a default affinity width with a real eigen-gap.

## Known limitations

- Some demos use stand-ins for trained models, always stated in the UI: hand-placed embeddings (analogy lab, CLIP-style lab), untrained Transformer explorer weights, pixel-based region scores in the R-CNN demo, scripted tool-calling decisions (the tools themselves really run), hand-made concept vectors for dense retrieval, fixed pooling in the U-Net illustration.
- UMAP is a simplified implementation (fixed curve parameters, random initialisation).
- Animated visualizers pause when the browser tab is hidden or occluded (`requestAnimationFrame`); Step buttons always work.
- No Web Workers; the heaviest computations (t-SNE, GP surrogate, Jacobi on 120 × 120) are memoised and fast enough on the main thread.
- `npm audit` reports advisories in dev tooling and KaTeX ≤ 0.18 (KaTeX only renders trusted in-repo strings). Major upgrades are a follow-up.

## Architecture

```
src/
  app/                  routes: /, /[module], /[module]/[topic], /playground/[lab], resources
  components/           layout, navigation (sidebar, Ctrl-K palette), topic (incl. CodeTabs), controls, visualization, mathematics
  visualizers/          134 lazily loaded visualizers + registry.tsx / ids.ts, shared.tsx helpers
  lib/                  math, statistics, transformations, ml, nn, utilities (format, highlight)
  data/curriculum/      one file per module
  data/code/            Python + Rust examples per module (merged into topics by key)
  store/progress.ts     zustand + localStorage
tests/                  linalg, stats, ml, extra helpers, curriculum integrity, escaping, code execution
```

Adding a topic: add an entry to the module file and its code example to `src/data/code/<module>.ts`; adding a visualizer: create the component, add its id to `ids.ts` and a loader in `registry.tsx`. Tests catch missing ids, broken links, invalid TeX and wrong code output.
