/** Classification and regression evaluation metrics. */

export interface Confusion {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export function confusion(yTrue: number[], scores: number[], threshold: number): Confusion {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const pred = scores[i] >= threshold ? 1 : 0;
    if (pred === 1 && yTrue[i] === 1) tp++;
    else if (pred === 1) fp++;
    else if (yTrue[i] === 0) tn++;
    else fn++;
  }
  return { tp, fp, tn, fn };
}

const safeDiv = (a: number, b: number) => (b === 0 ? NaN : a / b);

export function classificationMetrics(c: Confusion, beta = 1) {
  const { tp, fp, tn, fn } = c;
  const precision = safeDiv(tp, tp + fp);
  const recall = safeDiv(tp, tp + fn);
  const specificity = safeDiv(tn, tn + fp);
  const accuracy = safeDiv(tp + tn, tp + tn + fp + fn);
  const f1 = safeDiv(2 * tp, 2 * tp + fp + fn);
  const b2 = beta * beta;
  const fbeta = safeDiv((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp);
  const fpr = safeDiv(fp, fp + tn);
  return { precision, recall, specificity, accuracy, f1, fbeta, fpr, tpr: recall };
}

/** ROC curve points sorted by increasing FPR, plus trapezoidal AUC. */
export function rocCurve(yTrue: number[], scores: number[]) {
  const idx = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const P = yTrue.filter((y) => y === 1).length;
  const N = yTrue.length - P;
  const pts: { fpr: number; tpr: number; threshold: number }[] = [{ fpr: 0, tpr: 0, threshold: Infinity }];
  let tp = 0, fp = 0;
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    if (yTrue[i] === 1) tp++;
    else fp++;
    // only emit a point when the next score differs (handles ties)
    if (k === idx.length - 1 || scores[idx[k + 1]] !== scores[i]) pts.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0, threshold: scores[i] });
  }
  let auc = 0;
  for (let k = 1; k < pts.length; k++) auc += (pts[k].fpr - pts[k - 1].fpr) * (pts[k].tpr + pts[k - 1].tpr) / 2;
  return { points: pts, auc };
}

/** Precision-recall curve and average precision (step-wise, as in scikit-learn). */
export function prCurve(yTrue: number[], scores: number[]) {
  const idx = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const P = yTrue.filter((y) => y === 1).length;
  const pts: { recall: number; precision: number; threshold: number }[] = [];
  let tp = 0, fp = 0;
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    if (yTrue[i] === 1) tp++;
    else fp++;
    if (k === idx.length - 1 || scores[idx[k + 1]] !== scores[i]) pts.push({ recall: P ? tp / P : 0, precision: tp / (tp + fp), threshold: scores[i] });
  }
  let ap = 0;
  let prevR = 0;
  for (const p of pts) {
    ap += (p.recall - prevR) * p.precision;
    prevR = p.recall;
  }
  return { points: pts, ap };
}

export function logLoss(yTrue: number[], probs: number[], eps = 1e-15): number {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const p = Math.min(1 - eps, Math.max(eps, probs[i]));
    s += yTrue[i] * Math.log(p) + (1 - yTrue[i]) * Math.log(1 - p);
  }
  return -s / yTrue.length;
}

export function brierScore(yTrue: number[], probs: number[]): number {
  return yTrue.reduce((s, y, i) => s + (probs[i] - y) ** 2, 0) / yTrue.length;
}

/** Reliability diagram bins. */
export function calibrationBins(yTrue: number[], probs: number[], bins = 10) {
  const out = Array.from({ length: bins }, (_, b) => ({ lo: b / bins, hi: (b + 1) / bins, n: 0, meanPred: 0, fracPos: 0 }));
  probs.forEach((p, i) => {
    const b = Math.min(bins - 1, Math.floor(p * bins));
    out[b].n++;
    out[b].meanPred += p;
    out[b].fracPos += yTrue[i];
  });
  for (const o of out) if (o.n) {
    o.meanPred /= o.n;
    o.fracPos /= o.n;
  }
  return out;
}

export function regressionMetrics(y: number[], yhat: number[], p = 1) {
  const n = y.length;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sae = 0, sse = 0, sst = 0;
  for (let i = 0; i < n; i++) {
    const e = y[i] - yhat[i];
    sae += Math.abs(e);
    sse += e * e;
    sst += (y[i] - my) ** 2;
  }
  const mae = sae / n;
  const mse = sse / n;
  const rmse = Math.sqrt(mse);
  const r2 = sst === 0 ? NaN : 1 - sse / sst;
  const adjR2 = n - p - 1 > 0 ? 1 - ((1 - r2) * (n - 1)) / (n - p - 1) : NaN;
  return { mae, mse, rmse, r2, adjR2, sse, sst };
}
