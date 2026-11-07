"use client";
import { Component, type ReactNode } from "react";
import { registry } from "./registry";

class Boundary extends Component<{ children: ReactNode }, { err: Error | null }> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) {
    return { err };
  }
  render() {
    const err = this.state.err;
    if (err && (err.name === "ChunkLoadError" || /loading (css )?chunk|failed to fetch dynamically imported module/i.test(err.message)))
      return (
        <div className="panel flex h-[420px] flex-col items-center justify-center gap-2 text-[13px] text-faint" role="alert">
          This visualization could not be downloaded (network problem or an updated site).
          <button className="btn" onClick={() => window.location.reload()}>Reload page</button>
        </div>
      );
    if (err)
      return (
        <div className="panel p-6 text-[13px] text-red">
          This visualization hit an error: {err.message}. Try resetting its parameters or reloading the page.
        </div>
      );
    return this.props.children;
  }
}

export function VisualSlot({ id, props }: { id: string; props?: Record<string, unknown> }) {
  const C = registry[id as keyof typeof registry];
  if (!C) return <div className="panel p-6 text-[13px] text-red">Unknown visualizer “{id}”.</div>;
  return (
    <Boundary>
      <C {...(props ?? {})} />
    </Boundary>
  );
}
