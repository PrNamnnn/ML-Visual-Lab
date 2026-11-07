"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { BookMarked, BookA, ChevronRight, Compass, FlaskConical, Home, LibraryBig, ListTree, PanelLeftClose, PanelLeftOpen, Route, Search, Sigma } from "lucide-react";
import { useNav } from "./NavContext";
import { useProgress } from "@/store/progress";
import { cn } from "@/lib/utilities/format";
import { moduleIcon } from "./moduleIcons";


const groups: { id: "foundations" | "data-ml" | "advanced"; title: string }[] = [
  { id: "foundations", title: "Mathematical Foundations" },
  { id: "data-ml", title: "Data & Machine Learning" },
  { id: "advanced", title: "Advanced AI" },
];

export function Sidebar({ onSearch, forceExpanded = false }: { onSearch: () => void; forceExpanded?: boolean }) {
  const { modules, byKey } = useNav();
  const pathname = usePathname();
  const storeCollapsed = useProgress((s) => s.sidebarCollapsed);
  const setCollapsed = useProgress((s) => s.setSidebarCollapsed);
  const collapsed = forceExpanded ? false : storeCollapsed;
  const activeModule = pathname.split("/")[1];
  const [open, setOpen] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (activeModule) setOpen((o) => ({ ...o, [activeModule]: true }));
  }, [activeModule]);

  const main = [
    { href: "/", label: "Overview", icon: Home },
    { href: "/learning-path", label: "Learning Path", icon: Route },
    { href: "/topics", label: "All Topics", icon: ListTree },
    { href: "/playground", label: "Visual Playground", icon: FlaskConical },
    { href: "/bookmarks", label: "Bookmarks", icon: BookMarked },
  ];
  const resources = [
    { href: "/reference", label: "Mathematical Reference", icon: Sigma },
    { href: "/formulas", label: "Formula Library", icon: LibraryBig },
    { href: "/glossary", label: "Glossary", icon: BookA },
  ];

  const item = (href: string, label: string, Icon: typeof Home, active: boolean) => (
    <Link
      key={href}
      href={href}
      title={collapsed ? label : undefined}
      aria-current={active ? "page" : undefined}
      className={cn(
        "relative flex h-8 items-center gap-2.5 rounded-md px-2.5 text-[13px] transition-colors duration-150",
        active ? "bg-hover font-medium text-fg before:absolute before:-left-2 before:top-1.5 before:h-5 before:w-[2px] before:rounded-full before:bg-accent" : "text-muted hover:bg-raised hover:text-fg",
        collapsed && "justify-center px-0",
      )}
    >
      <Icon size={15} className="shrink-0" strokeWidth={1.8} />
      {!collapsed && <span className="truncate">{label}</span>}
    </Link>
  );

  return (
    <nav aria-label="Main navigation" className="flex h-full flex-col">
      <div className={cn("flex h-14 items-center border-b border-line px-4", collapsed && "justify-center px-0")}>
        {!collapsed && (
          <Link href="/" className="flex items-center gap-2 text-[13.5px] font-semibold tracking-tight">
            <Compass size={16} className="text-fg" />
            ML Visual Lab
          </Link>
        )}
        {!forceExpanded && (
          <button className={cn("btn-ghost h-7 px-1.5", !collapsed && "ml-auto")} onClick={() => setCollapsed(!collapsed)} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
            {collapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
          </button>
        )}
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto px-2.5 py-4">
        <div className="space-y-0.5">
          {collapsed ? (
            <button className="flex h-8 w-full items-center justify-center rounded-md text-muted hover:bg-raised hover:text-fg" onClick={onSearch} aria-label="Search">
              <Search size={15} />
            </button>
          ) : null}
          {main.map((m) => item(m.href, m.label, m.icon, m.href === "/" ? pathname === "/" : pathname.startsWith(m.href)))}
        </div>
        {groups.map((g) => (
          <div key={g.id} className="space-y-0.5">
            {!collapsed && <div className="label px-2 pb-1">{g.title}</div>}
            {collapsed && <div className="mx-2 border-t border-line" />}
            {modules
              .filter((m) => m.group === g.id)
              .map((m) => {
                const Icon = moduleIcon[m.id];
                const isActive = activeModule === m.id;
                if (collapsed) return item(`/${m.id}`, m.title, Icon, isActive);
                const isOpen = !!open[m.id];
                const count = m.sections.reduce((s, sec) => s + sec.topics.length, 0);
                return (
                  <div key={m.id}>
                    <div className={cn("flex h-8 items-center rounded-md text-[13px]", isActive ? "text-fg" : "text-muted")}>
                      <button
                        className="flex h-8 w-6 shrink-0 items-center justify-center rounded hover:bg-raised"
                        onClick={() => setOpen((o) => ({ ...o, [m.id]: !isOpen }))}
                        aria-expanded={isOpen}
                        aria-label={`${isOpen ? "Collapse" : "Expand"} ${m.title}`}
                      >
                        <ChevronRight size={13} className={cn("transition-transform", isOpen && "rotate-90")} />
                      </button>
                      <Link href={`/${m.id}`} className={cn("flex h-8 min-w-0 flex-1 items-center gap-2 rounded-md px-1.5 hover:bg-raised/70 hover:text-fg", isActive && pathname === `/${m.id}` && "bg-raised")}>
                        <Icon size={15} strokeWidth={1.8} className="shrink-0" />
                        <span className="truncate">{m.title}</span>
                        <span className="num ml-auto text-[10.5px] text-faint">{count}</span>
                      </Link>
                    </div>
                    {isOpen && (
                      <div className="mb-1.5 ml-[19px] border-l border-line pl-2.5">
                        {m.sections.map((s) => (
                          <div key={s.title} className="py-0.5">
                            {m.sections.length > 1 && <div className="truncate px-2 pb-0.5 pt-1.5 text-[11px] text-faint">{s.title}</div>}
                            {s.topics.map((k) => {
                              const t = byKey.get(k);
                              if (!t) return null;
                              const active = pathname === t.href;
                              return (
                                <Link
                                  key={k}
                                  href={t.href}
                                  aria-current={active ? "page" : undefined}
                                  className={cn(
                                    "flex items-center gap-2 rounded-md px-2 py-[5px] text-[12.5px] leading-tight transition-colors duration-150",
                                    active ? "bg-hover font-medium text-fg" : "text-muted hover:bg-raised hover:text-fg",
                                    t.status === "planned" && !active && "text-faint",
                                  )}
                                >
                                  <span className={cn("h-1 w-1 shrink-0 rounded-full", active ? "bg-accent" : t.status === "complete" ? "bg-line-strong" : "bg-transparent ring-1 ring-line-strong")} />
                                  <span className="truncate">{t.title}</span>
                                </Link>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        ))}
        <div className="space-y-0.5">
          {!collapsed && <div className="label px-2 pb-1">Resources</div>}
          {collapsed && <div className="mx-2 border-t border-line" />}
          {resources.map((r) => item(r.href, r.label, r.icon, pathname.startsWith(r.href)))}
        </div>
      </div>
    </nav>
  );
}
