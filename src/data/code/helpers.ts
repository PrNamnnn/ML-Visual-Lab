import type { CodeExample } from "@/types/curriculum";
import { dedent } from "@/lib/utilities/highlight";

/** Raw template tag so backslashes in code are kept literally. */
export const r = String.raw;

/** Build a code example; both programs must print exactly `output`. */
export const ex = (python: string, rust: string, output: string, explanation: string): CodeExample => ({
  python: dedent(python),
  rust: dedent(rust),
  output: dedent(output),
  explanation,
});

/** Python-only example (used by the IITM MLT week lessons); the program must print exactly `output`. */
export const py = (python: string, output: string, explanation: string): CodeExample => ({
  python: dedent(python),
  output: dedent(output),
  explanation,
});
