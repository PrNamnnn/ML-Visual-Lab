import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-xl px-4 py-24 text-center">
      <div className="num text-[13px] text-faint">404</div>
      <h1 className="mt-2 text-[24px] font-semibold">This page isn’t in the curriculum</h1>
      <p className="mt-2 text-[14px] text-muted">Try the search (Ctrl K), browse all topics, or open the playground.</p>
      <div className="mt-6 flex justify-center gap-2">
        <Link href="/topics" className="btn">All topics</Link>
        <Link href="/playground" className="btn">Visual Playground</Link>
      </div>
    </div>
  );
}
