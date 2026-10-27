import { notFound, redirect } from "next/navigation";

// The app used to be one route per stage (/<runId>/preview, /running, /result,
// /inference). It is one page now; old links land on the matching stage.
const STAGE_FOR: Record<string, string> = {
  preview: "data",
  running: "training",
  result: "results",
  inference: "deploy",
};

export default async function LegacyStageRedirect({ params }: { params: Promise<{ runId: string; stage: string }> }) {
  const { runId, stage } = await params;
  // Only the four old stage routes redirect; any other two-part path is a 404.
  if (!(stage in STAGE_FOR)) notFound();
  redirect(`/?run=${encodeURIComponent(runId)}&stage=${STAGE_FOR[stage]}`);
}
