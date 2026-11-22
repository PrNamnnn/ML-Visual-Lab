/** Raw template tag: TeX is written with single backslashes. */
export const r = String.raw;

/** Shown on every week lesson that uses the data-matrix convention of the course notes. */
export const NOTATION =
  "As in the IITM MLT notes, the n data points x₁, …, xₙ ∈ ℝᵈ are the columns of the d × n data matrix X. Elsewhere on this site samples are stored as rows (n × d), so XXᵀ here plays the role of XᵀX there.";

/** Convention for linear models in the supervised weeks. */
export const NOTATION_SUP =
  "As in the IITM MLT notes, X is the d × n matrix whose columns are the data points, y ∈ ℝⁿ collects the labels, and the model is wᵀx with no separate bias: a constant feature 1 can be appended to x to play that role.";
