import { LoadingState, SkeletonBar } from "@/components/visualization/LoadingState";

/** Shown while a lesson's route is being fetched; it only fades in if the wait exceeds ~200 ms. */
export default function TopicLoading() {
  return (
    <div className="loading-delayed mx-auto max-w-[1180px] px-4 py-6 sm:px-6 lg:py-8">
      <div className="mx-auto max-w-3xl space-y-3">
        <SkeletonBar className="h-3 w-40" />
        <SkeletonBar className="h-7 w-2/3" />
        <SkeletonBar className="h-4 w-1/2" />
      </div>
      <div className="panel mt-8 flex h-[420px] items-center justify-center">
        <LoadingState label="Preparing interactive lesson…" />
      </div>
    </div>
  );
}
