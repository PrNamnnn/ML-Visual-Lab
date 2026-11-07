"use client";
import { useEffect, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { Menu, Search, Moon, Sun } from "lucide-react";
import type { NavModule, NavTopic } from "@/data/curriculum";
import { Sidebar } from "@/components/navigation/Sidebar";
import dynamic from "next/dynamic";
const CommandPalette = dynamic(() => import("@/components/navigation/CommandPalette"), { ssr: false });
import { hydrateProgress, useProgress } from "@/store/progress";
import { NavProvider } from "@/components/navigation/NavContext";
import { cn } from "@/lib/utilities/format";

export function AppShell({ nav, children }: { nav: { topics: NavTopic[]; modules: NavModule[] }; children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const collapsed = useProgress((s) => s.sidebarCollapsed);
  const theme = useProgress((s) => s.theme);
  const setTheme = useProgress((s) => s.setTheme);
  const pathname = usePathname();
  const [isMac, setIsMac] = useState(false);
  useEffect(() => setIsMac(/Mac|iPhone|iPad/.test(navigator.platform)), []);

  useEffect(() => {
    hydrateProgress();
  }, []);

  useEffect(() => {
    try { document.documentElement.dataset.theme = theme; } catch { /* ignore */ }
  }, [theme]);

  useEffect(() => setMobileOpen(false), [pathname]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !isTyping(e))) {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <NavProvider value={nav}>
      <div className="flex min-h-screen">
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-panel focus:px-3 focus:py-2">
          Skip to content
        </a>
        {/* desktop sidebar */}
        <aside className={cn("sticky top-0 hidden h-screen shrink-0 border-r border-line bg-panel transition-[width] duration-200 ease-out md:block", collapsed ? "w-[60px]" : "w-[272px]")}>
          <Sidebar onSearch={() => setPaletteOpen(true)} />
        </aside>
        {/* mobile drawer */}
        {mobileOpen && (
          <div className="fixed inset-0 z-40 md:hidden">
            <div className="fade-in absolute inset-0 bg-black/60" onClick={() => setMobileOpen(false)} />
            <aside className="slide-in absolute inset-y-0 left-0 w-[284px] max-w-[85vw] border-r border-line bg-panel">
              <Sidebar onSearch={() => setPaletteOpen(true)} forceExpanded />
            </aside>
          </div>
        )}
        <div className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b border-line bg-bg/80 px-3 backdrop-blur-md md:px-6">
            <button className="btn-ghost md:hidden" aria-label="Open navigation" onClick={() => setMobileOpen(true)}>
              <Menu size={18} />
            </button>
            <button
              onClick={() => setPaletteOpen(true)}
              className="group flex h-9 w-full max-w-[420px] items-center gap-2.5 rounded-lg border border-line bg-panel px-3 text-[13px] text-faint shadow-[0_1px_0_0_rgb(255_255_255/0.03)_inset] transition-colors duration-150 hover:border-line-strong hover:bg-raised hover:text-muted"
              aria-label="Search (Ctrl K)"
            >
              <Search size={15} className="shrink-0 transition-colors group-hover:text-fg" />
              <span className="truncate">Search concepts, algorithms, formulas…</span>
              <span className="ml-auto hidden items-center gap-0.5 sm:flex">
                <kbd className="rounded border border-line bg-raised px-1.5 py-0.5 font-mono text-[10.5px] text-muted">{isMac ? "⌘" : "Ctrl"}</kbd>
                <kbd className="rounded border border-line bg-raised px-1.5 py-0.5 font-mono text-[10.5px] text-muted">K</kbd>
              </span>
            </button>
            <div className="ml-auto flex items-center gap-1">
              <button className="btn-ghost" aria-label="Toggle colour theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
                {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
              </button>
            </div>
          </header>
          <main id="main" className="min-w-0 flex-1"><div key={pathname} className="page-enter">{children}</div></main>
        </div>
      </div>
      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} />}
    </NavProvider>
  );
}

function isTyping(e: KeyboardEvent) {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}
