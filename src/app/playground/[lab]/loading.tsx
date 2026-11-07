import { LoadingState, SkeletonBar } from "@/components/visualization/LoadingState";

/** Shown while a playground lab's route is being fetched (fades in only after a short delay). */
export default function LabLoading() {
  return (
    <div className="loading-delayed mx-auto max-w-[1180px] px-4 py-6 sm:px-6">
      <div className="space-y-3">
        <SkeletonBar className="h-3 w-32" />
        <SkeletonBar className="h-7 w-1/3" />
      </div>
      <div className="panel mt-6 flex h-[420px] items-center justify-center">
        <LoadingState label="Loading visualization…" />
      </div>
    </div>
  );
}
