import type { Metadata, Viewport } from "next";
import "katex/dist/katex.min.css";
import "@/styles/globals.css";
import { AppShell } from "@/components/layout/AppShell";
import { navIndex } from "@/data/curriculum";

export const metadata: Metadata = {
  title: { default: "ML Visual Lab", template: "%s · ML Visual Lab" },
  description: "An interactive visual laboratory for the mathematics of machine learning, deep learning and modern AI engineering.",
};

export const viewport: Viewport = { themeColor: "#090909", width: "device-width", initialScale: 1 };

// Applies the persisted theme before first paint to avoid a flash.
const themeScript = `try{var s=JSON.parse(localStorage.getItem('mlvl-progress-v1')||'{}');var t=s&&s.state&&s.state.theme;document.documentElement.dataset.theme=t==='light'?'light':'dark'}catch(e){document.documentElement.dataset.theme='dark'}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const nav = navIndex();
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <AppShell nav={nav}>{children}</AppShell>
      </body>
    </html>
  );
}
