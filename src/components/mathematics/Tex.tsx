import katex from "katex";
import { memo } from "react";

/**
 * KaTeX renderer. Works in both server and client components (pure string render).
 * `throwOnError: false` renders invalid TeX in red instead of crashing the page.
 */
function TexImpl({ tex, block = false, className }: { tex: string; block?: boolean; className?: string }) {
  const html = katex.renderToString(tex, { displayMode: block, throwOnError: false, strict: "ignore", trust: false, output: "html" });
  const Tag = block ? "div" : "span";
  return <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

export const Tex = memo(TexImpl);
