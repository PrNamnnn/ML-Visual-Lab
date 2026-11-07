"use client";
import { createContext, useContext, useMemo, type ReactNode } from "react";
import type { NavModule, NavTopic } from "@/data/curriculum";

interface NavData {
  topics: NavTopic[];
  modules: NavModule[];
  byKey: Map<string, NavTopic>;
}

const Ctx = createContext<NavData | null>(null);

export function NavProvider({ value, children }: { value: { topics: NavTopic[]; modules: NavModule[] }; children: ReactNode }) {
  const v = useMemo(() => ({ ...value, byKey: new Map(value.topics.map((t) => [t.key, t])) }), [value]);
  return <Ctx.Provider value={v}>{children}</Ctx.Provider>;
}

export function useNav() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useNav outside NavProvider");
  return v;
}
