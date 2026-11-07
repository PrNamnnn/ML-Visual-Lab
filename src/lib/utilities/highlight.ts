/**
 * Tiny dependency-free syntax highlighter for Python and Rust snippets.
 * Produces HTML with .tok-* spans (styled in globals.css). Runs on the server, so no
 * highlighter code ships to the browser.
 */
const PY_KW = new Set("def return if elif else for while in not and or is import from as class lambda None True False with try except finally raise yield pass break continue global print range len sum min max abs enumerate zip sorted list dict set tuple float int str round".split(" "));
const RS_KW = new Set("fn let mut return if else for while loop in match struct enum impl pub use mod as const static true false break continue where self Self ref move f64 f32 i32 i64 u32 u64 usize isize bool char str String Vec Option Some None Ok Err Box".split(" "));

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function highlight(code: string, lang: "python" | "rust"): string {
  const kw = lang === "python" ? PY_KW : RS_KW;
  const comment = lang === "python" ? "#" : "//";
  let out = "";
  let i = 0;
  const n = code.length;
  while (i < n) {
    const ch = code[i];
    if (code.startsWith(comment, i)) {
      const j = code.indexOf("\n", i);
      const e = j < 0 ? n : j;
      out += `<span class="tok-c">${esc(code.slice(i, e))}</span>`;
      i = e;
    } else if (ch === '"' || (lang === "python" && ch === "'")) {
      let j = i + 1;
      while (j < n && code[j] !== ch) j += code[j] === "\\" ? 2 : 1;
      out += `<span class="tok-s">${esc(code.slice(i, j + 1))}</span>`;
      i = j + 1;
    } else if (/[0-9]/.test(ch) && !/[A-Za-z_]/.test(code[i - 1] ?? "")) {
      const m = /^[0-9][0-9_]*(\.[0-9]+)?([eE][-+]?[0-9]+)?(f64|f32|usize|i32|i64|u32|u64)?/.exec(code.slice(i))!;
      out += `<span class="tok-n">${esc(m[0])}</span>`;
      i += m[0].length;
    } else if (/[A-Za-z_]/.test(ch)) {
      const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(code.slice(i))!;
      const w = m[0];
      const next = code[i + w.length];
      if (kw.has(w)) out += `<span class="tok-k">${w}</span>`;
      else if (next === "(" || (lang === "rust" && next === "!")) out += `<span class="tok-f">${w}</span>`;
      else if (/^[A-Z]/.test(w)) out += `<span class="tok-t">${w}</span>`;
      else out += w;
      i += w.length;
    } else {
      out += esc(ch);
      i++;
    }
  }
  return out;
}

/** Strip a leading newline and common indentation from template-literal snippets. */
export function dedent(s: string): string {
  const lines = s.replace(/^\n/, "").replace(/\s+$/, "").split("\n");
  const ind = Math.min(...lines.filter((l) => l.trim()).map((l) => l.match(/^ */)![0].length));
  return lines.map((l) => l.slice(ind)).join("\n");
}
