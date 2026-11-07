import type { TopicDef, Difficulty } from "@/types/curriculum";

/** Planned topic (title only — rendered honestly as "in development"). */
export const planned = (slug: string, title: string, summary?: string, difficulty?: Difficulty): TopicDef => ({ slug, title, summary, difficulty });
