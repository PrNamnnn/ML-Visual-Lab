import { cn } from "@/lib/utilities/format";

/** Small spinner in the theme's neutral tones. */
export function Spinner({ className }: { className?: string }) {
  return <span aria-hidden className={cn("inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-line-strong border-t-fg/70 motion-reduce:animate-none", className)} />;
}

/**
 * Loading placeholder. It fades in only after a short delay (`.loading-delayed`), so operations that
 * finish quickly never flash an indicator.
 */
export function LoadingState({ label = "Loading…", className }: { label?: string; className?: string }) {
  return (
    <div role="status" aria-live="polite" className={cn("loading-delayed flex items-center justify-center gap-2 text-[12.5px] text-faint", className)}>
      <Spinner />
      <span>{label}</span>
    </div>
  );
}

/** Grey bar used in route skeletons. */
export function SkeletonBar({ className }: { className?: string }) {
  return <div aria-hidden className={cn("skeleton rounded-md", className)} />;
}
