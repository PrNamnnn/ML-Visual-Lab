// Runs tests/code-run.test.ts with RUN_CODE=1 (cross-platform).
import { spawnSync } from "node:child_process";
const r = spawnSync("npx", ["vitest", "run", "tests/code-run.test.ts"], { stdio: "inherit", shell: true, env: { ...process.env, RUN_CODE: "1" } });
process.exit(r.status ?? 1);
