import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { topics, topicByKey } from "@/data/curriculum";
import { TopicView } from "@/components/topic/TopicView";

export const dynamicParams = false;

export function generateStaticParams() {
  return topics.map((t) => ({ module: t.module, topic: t.slug }));
}

type Params = Promise<{ module: string; topic: string }>;

export async function generateMetadata({ params }: { params: Params }): Promise<Metadata> {
  const { module, topic } = await params;
  const t = topicByKey(`${module}/${topic}`);
  return { title: t?.title ?? "Topic", description: t?.content?.definition ?? t?.summary };
}

export default async function TopicPage({ params }: { params: Params }) {
  const { module, topic } = await params;
  const t = topicByKey(`${module}/${topic}`);
  if (!t) notFound();
  return <TopicView topic={t} />;
}
