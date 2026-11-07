# ML Visual Lab — Project Status

_Last updated: 2026-11-07_

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
| Topics in curriculum | 312 |
| **Interactive lessons (complete)** | **312** |
| Placeholder lessons ("not built yet") | **0** |
| Topics with verified Python + Rust code | **312 / 312** |
| Visualizer components (registered ids) | 134 |
| Visual Playground labs | 16 / 16 |
| Statically generated pages | 350 |

A topic counts as **complete** only if it has: definition + motivation, KaTeX formulas with explanations, a working interactive visualization with controls, live calculations, a worked example, AI/ML applications, related-topic links, and Python + Rust code. `tests/curriculum.test.ts` enforces the structure, plus: every cross-reference resolves, every visualizer id is registered, and every formula renders in KaTeX without error.

## Lesson completion checklist (by module)

| Module | Complete | Placeholders |
|---|---|---|
| Linear Algebra | 47 | 0 |
| Calculus | 24 | 0 |
| Probability & Statistics | 46 | 0 |
| Optimization | 16 | 0 |
| Data Preprocessing | 31 | 0 |
| Feature Engineering | 9 | 0 |
| Machine Learning | 36 | 0 |
| Model Evaluation | 7 | 0 |
| Deep Learning | 46 | 0 |
| Computer Vision | 13 | 0 |
| NLP | 7 | 0 |
| Transformers & LLMs | 11 | 0 |
| AI Engineering | 19 | 0 |
| **Total** | **312** | **0** |

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
