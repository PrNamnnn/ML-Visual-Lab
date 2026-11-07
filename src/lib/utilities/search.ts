export interface Searchable {
  key: string;
  title: string;
  href: string;
  /** secondary text shown in results */
  context: string;
  /** lowercased extra search text */
  k: string;
  summary?: string;
  status?: "complete" | "planned";
}

const STOP = new Set([
  "how", "does", "do", "what", "is", "are", "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "show", "me", "explain",
  "work", "works", "working", "why", "with", "visualize", "visualise", "geometrically", "geometric", "about", "learn", "using", "use", "?",
]);

const norm = (s: string) => s.toLowerCase().replace(/[’']/g, "").replace(/[^a-z0-9+#\-\s]/g, " ");

export function tokens(q: string): string[] {
  return norm(q).split(/\s+/).filter((w) => w && !STOP.has(w));
}

/** crude stemming so "transformations" matches "transformation", "optimizers" → "optimizer" */
const stem = (w: string) => w.replace(/(ations|ation|ings|ing|ers|er|es|s)$/, "");

export function search(items: Searchable[], query: string, limit = 30): Searchable[] {
  const qt = tokens(query);
  if (!qt.length) return [];
  const phrase = qt.join(" ");
  const scored: { it: Searchable; s: number }[] = [];
  for (const it of items) {
    const title = norm(it.title);
    const hay = `${title} ${it.k} ${norm(it.summary ?? "")} ${norm(it.context)}`;
    let s = 0;
    let all = true;
    if (title === phrase) s += 100;
    else if (title.startsWith(phrase)) s += 60;
    else if (title.includes(phrase)) s += 40;
    if (it.k.includes(phrase)) s += 30;
    if (it.k.split("|").some((kw) => kw.trim() === phrase)) s += 25;
    for (const t of qt) {
      const st = stem(t);
      if (title.split(/\s+/).some((w) => w.startsWith(st))) s += 12;
      else if (it.k.includes(st)) s += 7;
      else if (hay.includes(st)) s += 3;
      else all = false;
    }
    if (!all) s *= 0.35;
    if (it.status === "planned") s *= 0.7;
    if (s > 4) scored.push({ it, s });
  }
  scored.sort((a, b) => b.s - a.s);
  return scored.slice(0, limit).map((x) => x.it);
}
