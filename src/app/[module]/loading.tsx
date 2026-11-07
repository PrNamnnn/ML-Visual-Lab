import { LoadingState, SkeletonBar } from "@/components/visualization/LoadingState";

/** Shown while a module overview is being fetched (fades in only after a short delay). */
export default function ModuleLoading() {
  return (
    <div className="loading-delayed mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <div className="space-y-3">
        <SkeletonBar className="h-7 w-1/3" />
        <SkeletonBar className="h-4 w-1/2" />
      </div>
      <div className="mt-8 grid gap-2 sm:grid-cols-2">
        {Array.from({ length: 6 }, (_, i) => <SkeletonBar key={i} className="h-14" />)}
      </div>
      <LoadingState label="Loading module…" className="mt-6" />
    </div>
  );
}
