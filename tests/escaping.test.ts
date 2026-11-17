import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import katex from "katex";
import { tex, texMatrix } from "@/lib/utilities/format";

const CMDS = "begin|end|frac|dfrac|tfrac|times|text|mathbf|sum|int|sqrt|cdot|left|right|alpha|beta|lambda|sigma|theta|nabla|partial|hat|bar|quad|operatorname|infty|top|mathbb|boldsymbol|odot|det|exp|log|ln|sin|cos|max|min|arg|prod|lim|neq|approx|pm|mu|varphi|epsilon|eta|kappa|rho|tau|omega|Sigma|Phi|delta|gamma|underbrace|xrightarrow|mathcal|tilde|lfloor|rfloor|lceil|rceil|langle|rangle|Rightarrow|iff|forall|mathrel|textstyle|bmatrix|pmatrix|perp|parallel|succeq|bmod|lVert|rVert|checkmark|texttt|le|ge|nu|xi|pi|phi|psi|chi|zeta|binom|big|Big|ell|vdots|cdots|ldots|varepsilon|mathrm|mid|propto|sim|leq|geq|ne|cup|cap|subset|setminus|circ|overline|vec|widehat|overset|underset|qquad|Delta|Gamma|Lambda|Omega|Theta|Pi|Psi|rightarrow|leftarrow|mapsto|exists|varnothing|rank|dim|ker|sup|inf|Pr";
// A command name must not be followed by another letter. (A plain \b word boundary would let "\theta_W"
// and "\dim S"-style strings through, because "_" and digits count as word characters.)
const BAD = new RegExp(String.raw`(^|[^\\])\\(${CMDS})(?![a-zA-Z])`);

function walk(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    return e.isDirectory() ? walk(p) : /\.(ts|tsx)$/.test(e.name) ? [p] : [];
  });
}

describe("TeX escaping in source", () => {
  it("no single-backslash TeX commands inside ordinary JS strings", () => {
    const offenders: string[] = [];
    for (const f of walk(path.resolve(__dirname, "../src"))) {
      fs.readFileSync(f, "utf8").split("\n").forEach((line, i) => {
        if (line.includes("String.raw")) return;
        // `r` is the String.raw alias used by the data files (const r = String.raw): drop those raw
        // templates and check what is left of the line, so ordinary strings beside them stay guarded
        const rest = line.replace(/\br`[^`]*`/g, "");
        if (BAD.test(rest)) offenders.push(`${path.relative(process.cwd(), f)}:${i + 1}: ${line.trim().slice(0, 100)}`);
      });
    }
    expect(offenders).toEqual([]);
  });

  it("texMatrix and tex produce valid, control-character-free TeX", () => {
    const out = texMatrix([[1.1, -0.5], [-0.3, 1.5]]) + tex(-5.55e-17) + tex(Infinity);
    expect(/[\u0000-\u001f]/.test(out)).toBe(false);
    expect(out).toContain(String.raw`\begin{bmatrix}1.1 & -0.5 \\ -0.3 & 1.5\end{bmatrix}`);
    expect(tex(-5.55e-17)).toBe(String.raw`-5.55\times10^{-17}`);
    expect(tex(-5.55e-17, 3, 1e-9)).toBe("0");
    expect(() => katex.renderToString(out, { throwOnError: true })).not.toThrow();
  });
});
