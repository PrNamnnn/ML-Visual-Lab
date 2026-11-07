"use client";
import { useState } from "react";
import { Check, Copy, Terminal } from "lucide-react";
import { cn } from "@/lib/utilities/format";

interface Lang { id: "python" | "rust"; label: string; html: string; raw: string; file: string }

export function CodeTabs({ python, rust, output, explanation }: { python: { html: string; raw: string }; rust: { html: string; raw: string }; output: string; explanation: string }) {
  const langs: Lang[] = [
    { id: "python", label: "Python", file: "main.py", ...python },
    { id: "rust", label: "Rust", file: "main.rs", ...rust },
  ];
  const [active, setActive] = useState<Lang["id"]>("python");
  const [copied, setCopied] = useState(false);
  const cur = langs.find((l) => l.id === active)!;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(cur.raw);
    } catch {
      const t = document.createElement("textarea");
      t.value = cur.raw;
      document.body.appendChild(t);
      t.select();
      document.execCommand("copy");
      t.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };
  return (
    <div className="grid gap-3 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
      <div className="panel min-w-0 overflow-hidden">
        <div className="flex items-center gap-1 border-b border-line bg-raised/50 px-2">
          <div role="tablist" aria-label="Implementation language" className="flex">
            {langs.map((l) => (
              <button
                key={l.id}
                role="tab"
                aria-selected={active === l.id}
                onClick={() => setActive(l.id)}
                className={cn("relative h-10 px-3 text-[12.5px] font-medium transition-colors", active === l.id ? "text-fg after:absolute after:inset-x-2 after:bottom-0 after:h-[2px] after:rounded-full after:bg-accent" : "text-faint hover:text-muted")}
              >
                {l.label}
              </button>
            ))}
          </div>
          <span className="ml-auto hidden font-mono text-[11px] text-faint sm:inline">{cur.file}</span>
          <button onClick={copy} className="btn-ghost h-7 px-2 text-[12px]" aria-label={`Copy ${cur.label} code`}>
            {copied ? <Check size={13} className="text-green" /> : <Copy size={13} />}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <pre className="max-h-[460px] overflow-auto p-4 font-mono text-[12.5px] leading-[1.65]" role="tabpanel"><code dangerouslySetInnerHTML={{ __html: cur.html }} /></pre>
      </div>
      <div className="min-w-0 space-y-3">
        <div className="panel overflow-hidden">
          <div className="flex items-center gap-2 border-b border-line px-3 py-2 text-[11.5px] text-faint"><Terminal size={13} /> Output (both programs)</div>
          <pre className="overflow-x-auto p-3 font-mono text-[12.5px] leading-relaxed text-fg">{output}</pre>
        </div>
        <p className="text-[13px] leading-relaxed text-muted">{explanation}</p>
      </div>
    </div>
  );
}
