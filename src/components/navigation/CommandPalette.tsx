"use client";
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { CornerDownLeft, FileText, FlaskConical, Search, X, type LucideIcon } from "lucide-react";
import { useNav } from "./NavContext";
import { search, tokens, type Searchable } from "@/lib/utilities/search";
import { labs } from "@/data/labs";
import { moduleIcon, searchCategory, CATEGORY_ORDER } from "./moduleIcons";
import { cn } from "@/lib/utilities/format";
import type { ModuleId } from "@/types/curriculum";

type Item = Searchable & { category: string; icon: LucideIcon };

const pages: Omit<Item, "icon" | "category">[] = [
  { key: "p/overview", title: "Overview", href: "/", context: "Home page", k: "home overview start" },
  { key: "p/path", title: "Learning Path", href: "/learning-path", context: "Suggested order of lessons", k: "learning path roadmap order curriculum" },
  { key: "p/topics", title: "All Topics", href: "/topics", context: "Browse and filter the curriculum", k: "all topics curriculum list browse" },
  { key: "p/playground", title: "Visual Playground", href: "/playground", context: "16 free-form labs", k: "playground labs experiments" },
  { key: "p/bookmarks", title: "Bookmarks", href: "/bookmarks", context: "Saved topics and history", k: "bookmarks saved recent history" },
  { key: "p/formulas", title: "Formula Library", href: "/formulas", context: "Every formula, by module", k: "formulas equations library" },
  { key: "p/glossary", title: "Glossary", href: "/glossary", context: "Terms A–Z", k: "glossary terms definitions" },
  { key: "p/reference", title: "Mathematical Reference", href: "/reference", context: "Notation and symbols", k: "notation reference symbols greek" },
];

/** Wrap occurrences of query tokens in <mark>. */
function Highlight({ text, q }: { text: string; q: string[] }) {
  if (!q.length) return <>{text}</>;
  const re = new RegExp(`(${q.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "ig");
  const parts = text.split(re);
  return <>{parts.map((p, i) => (i % 2 ? <mark key={i} className="hl">{p}</mark> : <Fragment key={i}>{p}</Fragment>))}</>;
}

export default function CommandPalette({ onClose }: { onClose: () => void }) {
  const { topics, modules } = useNav();
  const router = useRouter();
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const [closing, setClosing] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const reduce = useRef(false);

  const items = useMemo<Item[]>(() => {
    const modTitle = Object.fromEntries(modules.map((m) => [m.id, m.title]));
    return [
      ...topics.map((t) => ({ key: t.key, title: t.title, href: t.href, context: t.summary || `${modTitle[t.module]} · ${t.section}`, k: t.k, summary: t.summary, status: t.status, category: searchCategory[t.module as ModuleId], icon: moduleIcon[t.module as ModuleId] })),
      ...labs.map((l) => ({ key: `lab/${l.slug}`, title: l.title, href: `/playground/${l.slug}`, context: l.blurb, k: (l.keywords ?? "").toLowerCase() + " lab playground", summary: l.blurb, category: "Visual Playground", icon: FlaskConical })),
      ...pages.map((p) => ({ ...p, category: "Pages", icon: FileText })),
    ];
  }, [topics, modules]);

  const qTokens = useMemo(() => tokens(q).filter((t) => t.length > 1), [q]);

  // Ranked results, then grouped by category in a fixed order (rank preserved within groups).
  const groups = useMemo(() => {
    const ranked = (q.trim() ? (search(items, q, 40) as Item[]) : items.filter((i) => i.category === "Visual Playground").slice(0, 8));
    const by = new Map<string, Item[]>();
    for (const r of ranked) by.set(r.category, [...(by.get(r.category) ?? []), r]);
    const firstRank = (c: string) => ranked.findIndex((r) => r.category === c);
    return [...by.entries()].sort((a, b) => (q.trim() ? firstRank(a[0]) - firstRank(b[0]) : CATEGORY_ORDER.indexOf(a[0]) - CATEGORY_ORDER.indexOf(b[0])));
  }, [items, q]);
  const flat = useMemo(() => groups.flatMap(([, g]) => g), [groups]);

  // Scroll lock: released as soon as closing starts (not only on unmount), so the page is never left locked.
  const unlockRef = useRef<() => void>(() => {});
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const finished = useRef(false);
  const finish = useCallback(() => {
    if (finished.current) return;
    finished.current = true;
    unlockRef.current();
    onCloseRef.current();
  }, []);

  const close = useCallback(() => {
    unlockRef.current();
    // No exit animation when it could not be seen or may never complete: unmount immediately.
    if (reduce.current || document.hidden) return finish();
    setClosing(true);
  }, [finish]);

  useEffect(() => {
    reduce.current = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const prev = document.activeElement as HTMLElement | null; // captured before focusing the input
    inputRef.current?.focus();
    const { overflow } = document.body.style;
    document.body.style.overflow = "hidden";
    let unlocked = false;
    unlockRef.current = () => { if (!unlocked) { unlocked = true; document.body.style.overflow = overflow; } };
    return () => { unlockRef.current(); prev?.focus?.(); };
  }, []);
  // Unmount when the fade-out ends (onAnimationEnd below); this timer is a fallback in case the event never
  // fires (interrupted animation, throttled background tab), so an invisible overlay can never remain.
  useEffect(() => {
    if (!closing) return;
    const t = window.setTimeout(finish, 250);
    return () => window.clearTimeout(t);
  }, [closing, finish]);
  useEffect(() => setSel(0), [q]);
  useEffect(() => { listRef.current?.querySelector<HTMLElement>(`[data-i="${sel}"]`)?.scrollIntoView({ block: "nearest" }); }, [sel]);

  const go = (r: Item | undefined) => { if (!r) return; router.push(r.href); onClose(); };

  let idx = -1;
  return (
    <div
      className={cn("fixed inset-0 z-50 flex items-start justify-center bg-black/60 px-3 pt-[10vh] backdrop-blur-[2px]", closing ? "fade-out pointer-events-none" : "fade-in")}
      onMouseDown={close}
      onAnimationEnd={(e) => { if (closing && e.target === e.currentTarget) finish(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Search"
        className={cn("w-full max-w-[640px] overflow-hidden rounded-xl border border-line-strong bg-panel shadow-[0_24px_80px_-12px_rgb(0_0_0/0.7)]", !closing && "pop-in")}
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === "Escape") { e.preventDefault(); close(); }
          else if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(flat.length - 1, s + 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
          else if (e.key === "Enter") { e.preventDefault(); go(flat[sel]); }
          else if (e.key === "Tab") e.preventDefault();
        }}
      >
        <div className="flex items-center gap-3 border-b border-line px-4">
          <Search size={17} className="shrink-0 text-faint" />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search concepts, algorithms, formulas…"
            className="h-14 w-full bg-transparent text-[15px] outline-none placeholder:text-faint focus-visible:outline-none"
            aria-label="Search query"
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-results"
            aria-activedescendant={flat[sel] ? `res-${sel}` : undefined}
            spellCheck={false}
          />
          {q && (
            <button className="rounded p-1 text-faint hover:bg-hover hover:text-fg" onClick={() => { setQ(""); inputRef.current?.focus(); }} aria-label="Clear search">
              <X size={15} />
            </button>
          )}
          <kbd className="hidden rounded border border-line px-1.5 py-0.5 font-mono text-[10.5px] text-faint sm:inline">Esc</kbd>
        </div>
        <div id="palette-results" ref={listRef} role="listbox" className="max-h-[min(60vh,520px)] overflow-y-auto p-2">
          {flat.length === 0 && (
            <div className="px-3 py-10 text-center">
              <div className="text-[13.5px] text-muted">No matches for “{q}”.</div>
              <div className="mt-1 text-[12px] text-faint">Try “eigenvectors”, “cross entropy”, “Adam” or “standard deviation”.</div>
            </div>
          )}
          {groups.map(([cat, list]) => (
            <div key={cat} role="group" aria-label={cat} className="mb-1">
              <div className="px-2.5 pb-1 pt-2 text-[10.5px] font-medium uppercase tracking-[0.08em] text-faint">{!q.trim() && cat === "Visual Playground" ? "Jump into a lab" : cat}</div>
              {list.map((r) => {
                idx++;
                const i = idx;
                const Icon = r.icon;
                return (
                  <div
                    key={r.key}
                    id={`res-${i}`}
                    data-i={i}
                    role="option"
                    aria-selected={i === sel}
                    onMouseMove={() => setSel(i)}
                    onClick={() => go(r)}
                    className={cn("flex cursor-pointer items-center gap-3 rounded-lg px-2.5 py-2 transition-colors duration-100", i === sel ? "bg-hover" : "hover:bg-raised")}
                  >
                    <span className={cn("flex h-7 w-7 shrink-0 items-center justify-center rounded-md border", i === sel ? "border-line-strong bg-raised text-fg" : "border-line text-faint")}>
                      <Icon size={14} strokeWidth={1.8} />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 truncate text-[13.5px] text-fg">
                        <span className="truncate"><Highlight text={r.title} q={qTokens} /></span>
                        {r.status === "planned" && <span className="chip py-0 text-[10px]">planned</span>}
                      </div>
                      <div className="truncate text-[12px] text-faint"><Highlight text={r.context} q={qTokens} /></div>
                    </div>
                    {i === sel && <CornerDownLeft size={13} className="shrink-0 text-faint" />}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <div className="flex items-center gap-4 border-t border-line bg-raised/40 px-4 py-2 text-[11px] text-faint">
          <span><kbd className="font-mono">↑↓</kbd> navigate</span>
          <span><kbd className="font-mono">↵</kbd> open</span>
          <span><kbd className="font-mono">Esc</kbd> close</span>
          <span className="ml-auto">{flat.length} result{flat.length === 1 ? "" : "s"}</span>
        </div>
      </div>
    </div>
  );
}
