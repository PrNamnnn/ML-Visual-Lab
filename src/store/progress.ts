"use client";
import { create } from "zustand";
import { persist, createJSONStorage, type StateStorage } from "zustand/middleware";

/**
 * Local-only persistence. All access goes through `safeStorage` so private windows,
 * blocked storage or quota errors never break the app. Swapping this storage adapter
 * for a remote one is the only change needed to add a backend later.
 */
const safeStorage: StateStorage = {
  getItem: (k) => {
    try { return window.localStorage.getItem(k); } catch { return null; }
  },
  setItem: (k, v) => {
    try { window.localStorage.setItem(k, v); } catch { /* ignore */ }
  },
  removeItem: (k) => {
    try { window.localStorage.removeItem(k); } catch { /* ignore */ }
  },
};

interface ProgressState {
  bookmarks: string[];
  completed: string[];
  recent: { key: string; at: number }[];
  theme: "dark" | "light";
  sidebarCollapsed: boolean;
  toggleBookmark: (key: string) => void;
  toggleCompleted: (key: string) => void;
  visit: (key: string) => void;
  setTheme: (t: "dark" | "light") => void;
  setSidebarCollapsed: (v: boolean) => void;
  clearHistory: () => void;
}

export const useProgress = create<ProgressState>()(
  persist(
    (set) => ({
      bookmarks: [],
      completed: [],
      recent: [],
      theme: "dark",
      sidebarCollapsed: false,
      toggleBookmark: (key) => set((s) => ({ bookmarks: s.bookmarks.includes(key) ? s.bookmarks.filter((k) => k !== key) : [key, ...s.bookmarks] })),
      toggleCompleted: (key) => set((s) => ({ completed: s.completed.includes(key) ? s.completed.filter((k) => k !== key) : [...s.completed, key] })),
      visit: (key) => set((s) => ({ recent: [{ key, at: Date.now() }, ...s.recent.filter((r) => r.key !== key)].slice(0, 12) })),
      setTheme: (theme) => {
        try { document.documentElement.dataset.theme = theme; } catch { /* ignore */ }
        set({ theme });
      },
      setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
      clearHistory: () => set({ recent: [] }),
    }),
    { name: "mlvl-progress-v1", storage: createJSONStorage(() => safeStorage), skipHydration: true },
  ),
);

/** Hydrate once on the client (avoids SSR/client markup mismatch). */
export function hydrateProgress() {
  const p = useProgress.persist;
  if (!p.hasHydrated()) void p.rehydrate();
}
