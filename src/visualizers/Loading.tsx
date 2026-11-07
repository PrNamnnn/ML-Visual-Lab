import { LoadingState } from "@/components/visualization/LoadingState";

/** Fallback shown while a visualizer's code chunk downloads (same footprint as before: no layout shift). */
export function VizLoading({ label = "Loading visualization…" }: { label?: string }) {
  return (
    <div className="panel flex h-[420px] items-center justify-center">
      <LoadingState label={label} />
    </div>
  );
}
