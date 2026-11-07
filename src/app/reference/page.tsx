import type { Metadata } from "next";
import { Tex } from "@/components/mathematics/Tex";

export const metadata: Metadata = { title: "Mathematical Reference" };

const GROUPS: { title: string; rows: [string, string][] }[] = [
  { title: "Conventions", rows: [
    ["x,\\ \\alpha", "scalars (lower-case italic)"], ["\\mathbf x", "vectors (bold lower-case), column vectors by default"], ["X,\\ W", "matrices (upper-case)"],
    ["x_i,\\ X_{ij}", "i-th component; entry in row i, column j"], ["\\mathbf x^{(l)}", "quantity at layer l"], ["\\mathbb R^{m\\times n}", "real m×n matrices"],
    ["\\hat y,\\ \\hat\\theta", "prediction or estimate"], ["\\bar x", "sample mean"],
  ] },
  { title: "Linear algebra", rows: [
    ["\\mathbf u^\\top\\mathbf v", "dot product"], ["\\|\\mathbf x\\|_p", "Lp norm"], ["A^\\top,\\ A^{-1},\\ A^{+}", "transpose, inverse, pseudoinverse"], ["\\det A,\\ \\operatorname{tr}A", "determinant, trace"],
    ["\\odot", "Hadamard (element-wise) product"], ["\\operatorname{diag}(\\mathbf d)", "diagonal matrix"], ["A\\succ0,\\ A\\succeq0", "positive (semi)definite"], ["\\operatorname{span},\\ \\operatorname{rank}", "span, rank"],
  ] },
  { title: "Calculus and optimisation", rows: [
    ["f'(x),\\ \\frac{df}{dx}", "derivative"], ["\\frac{\\partial f}{\\partial x}", "partial derivative"], ["\\nabla f", "gradient"], ["\\nabla^2f,\\ H", "Hessian"],
    ["\\int_a^bf(x)\\,dx", "definite integral"], ["\\arg\\min_\\theta L(\\theta)", "minimiser"], ["\\eta", "learning rate"], ["\\lambda", "regularisation strength or eigenvalue (context)"],
  ] },
  { title: "Probability and statistics", rows: [
    ["P(A\\mid B)", "conditional probability"], ["X\\sim\\mathcal N(\\mu,\\sigma^2)", "X is distributed as a normal"], ["E[X],\\ \\operatorname{Var}(X)", "expectation, variance"], ["\\operatorname{Cov}(X,Y),\\ \\rho", "covariance, correlation"],
    ["\\Phi,\\ \\varphi", "standard normal CDF and PDF"], ["\\mathcal L(\\theta),\\ \\ell(\\theta)", "likelihood, log-likelihood"], ["\\alpha,\\ \\beta", "Type I and Type II error rates"], ["\\xrightarrow{d}", "convergence in distribution"],
  ] },
  { title: "Greek letters used", rows: [
    ["\\alpha\\ \\beta\\ \\gamma\\ \\delta\\ \\epsilon", "alpha, beta, gamma, delta, epsilon"], ["\\eta\\ \\theta\\ \\kappa\\ \\lambda\\ \\mu", "eta, theta, kappa, lambda, mu"], ["\\nu\\ \\pi\\ \\rho\\ \\sigma\\ \\tau", "nu, pi, rho, sigma, tau"], ["\\phi\\ \\chi\\ \\psi\\ \\omega\\ \\Sigma", "phi, chi, psi, omega, capital sigma"],
  ] },
];

export default function Reference() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <div className="label">Resources</div>
      <h1 className="mt-1 text-[28px] font-semibold tracking-tight">Mathematical reference</h1>
      <p className="mt-2 text-[15px] text-muted">Notation used throughout the lab.</p>
      <div className="mt-8 space-y-8">
        {GROUPS.map((g) => (
          <section key={g.title}>
            <h2 className="mb-2 text-[16px] font-semibold">{g.title}</h2>
            <div className="divide-y divide-line rounded-lg border border-line">
              {g.rows.map(([t, d]) => (
                <div key={t} className="grid grid-cols-[minmax(0,200px)_1fr] items-center gap-4 px-4 py-2">
                  <div className="overflow-x-auto"><Tex tex={t} /></div>
                  <div className="text-[13px] text-muted">{d}</div>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
