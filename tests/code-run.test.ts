/**
 * Executes every Python and Rust example and checks both print exactly the documented output.
 * Slow (one rustc build of all snippets), so it runs only with RUN_CODE=1:  npm run test:code
 */
import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { topics } from "@/data/curriculum";

const RUN = !!process.env.RUN_CODE;
const norm = (s: string) => s.replace(/\r\n/g, "\n").replace(/\s+$/, "");
const withCode = topics.filter((t) => t.content?.code);
/** Python outputs collected for the optional WRITE_ACTUAL dump (only used when both languages agree). */
const pyOut: Record<string, string> = {};

describe.runIf(RUN)("code examples execute", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "mlvl-code-"));

  it("every Python snippet prints its documented output", () => {
    const bad: string[] = [];
    withCode.forEach((t, i) => {
      const f = path.join(dir, `s${i}.py`);
      fs.writeFileSync(f, t.content!.code!.python);
      try {
        const out = execFileSync("python", ["-I", f], { encoding: "utf8", timeout: 20000 });
        pyOut[t.key] = norm(out);
        if (norm(out) !== norm(t.content!.code!.output)) bad.push(`${t.key}\n--- expected\n${t.content!.code!.output}\n--- got\n${out}`);
      } catch (e) {
        bad.push(`${t.key}: ${(e as Error).message.slice(0, 400)}`);
      }
    });
    expect(bad).toEqual([]);
  }, 600000);

  it("every Rust snippet compiles and prints its documented output", () => {
    // all snippets in one crate: each wrapped in its own module, main() runs them in turn
    const MARK = "@@SNIPPET@@";
    const mods = withCode.map((t, i) => `mod s${i} {\n${t.content!.code!.rust.replace(/^fn main\(\)/m, "pub fn main()")}\n}`).join("\n\n");
    const main = `fn main() {\n${withCode.map((_, i) => `    println!("${MARK}${i}");\n    s${i}::main();`).join("\n")}\n}`;
    const src = path.join(dir, "all.rs");
    fs.writeFileSync(src, `#![allow(warnings)]\n${mods}\n\n${main}\n`);
    const exe = path.join(dir, process.platform === "win32" ? "all.exe" : "all");
    try {
      execFileSync("rustc", ["--edition", "2021", "-o", exe, src], { encoding: "utf8", stdio: "pipe", timeout: 600000 });
    } catch (e) {
      const err = (e as { stderr?: string }).stderr ?? String(e);
      // map line numbers back to snippets for a readable failure
      throw new Error(err.slice(0, 6000));
    }
    const out = norm(execFileSync(exe, { encoding: "utf8", timeout: 60000 }));
    const parts = out.split(new RegExp(`^${MARK}\\d+\\n?`, "m")).slice(1);
    const bad: string[] = [];
    const agreed: Record<string, { expected: string; actual: string }> = {};
    withCode.forEach((t, i) => {
      if (process.env.WRITE_ACTUAL && pyOut[t.key] !== undefined && pyOut[t.key] === norm(parts[i] ?? "") && pyOut[t.key] !== norm(t.content!.code!.output)) agreed[t.key] = { expected: norm(t.content!.code!.output), actual: pyOut[t.key] };
      if (norm(parts[i] ?? "") !== norm(t.content!.code!.output)) bad.push(`${t.key}\n--- expected\n${t.content!.code!.output}\n--- got\n${parts[i]}`);
    });
    if (process.env.WRITE_ACTUAL) fs.writeFileSync(process.env.WRITE_ACTUAL, JSON.stringify(agreed, null, 1));
    expect(bad).toEqual([]);
  }, 900000);
});

describe("code example coverage", () => {
  it("lists coverage", () => {
    console.log(`code examples: ${withCode.length} / ${topics.filter((t) => t.content).length} complete topics`);
    expect(withCode.length).toBeGreaterThanOrEqual(0);
  });
});
