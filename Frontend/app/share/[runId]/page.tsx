"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowUpRight, FlaskConical } from "lucide-react";
import { getDeployment, getModelSchema, type Deployment, type ModelSchema } from "@/lib/api";
import { cn } from "@/lib/utils";
import { buttonVariants } from "@/components/ui/button";
import { SiteHeader } from "@/components/site-header";
import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel";
import { TryModel } from "@/components/try-model";

/**
 * The public test page a deployed model is shared through (QR code and link
 * on the Deploy stage). No account and no dashboard: one form and its answer.
 */
export default function PublicSharePage() {
  const { runId } = useParams<{ runId: string }>();
  const [schema, setSchema] = useState<ModelSchema | null>(null);
  const [deployment, setDeployment] = useState<Deployment | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoadError(null);
    setSchema(null);
    setDeployment(null);
    getModelSchema(runId)
      .then(setSchema)
      .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)));
    getDeployment(runId)
      .then(setDeployment)
      // No deployment record yet reads as not live, which the card explains.
      .catch(() => setDeployment({ run_id: runId, status: "not_deployed" }));
  }, [runId]);

  useEffect(() => {
    load();
  }, [load]);

  const deployed = deployment?.status === "succeeded" && !!deployment.predict_url;

  return (
    <div className="flex min-h-dvh flex-col">
      <SiteHeader
        start={
          <Link href="/" className={cn(buttonVariants({ variant: "outline", size: "sm" }), "h-9 lg:h-8")}>
            <span className="sm:hidden">Build</span>
            <span className="hidden sm:inline">Build your own</span>
            <ArrowUpRight data-icon="inline-end" />
          </Link>
        }
      />

      <main className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6 lg:py-12">
        <header className="mb-6 text-center">
          <p className="text-xs font-medium text-primary">Try this AI model</p>
          <h1 className="mt-2 text-3xl font-bold sm:text-4xl">
            {schema ? `Predict ${humanize(schema.feature_cols.length, schema.problem_type)}` : loadError ? "Model unavailable" : "Loading the model..."}
          </h1>
          <p className="mx-auto mt-2 max-w-xl text-sm text-muted-foreground">
            {schema ? (
              <>
                A <strong className="font-medium text-foreground">{schema.model_name ?? "machine learning"}</strong> model trained on
                this dataset. Fill in the values, press Predict, and it answers live.
              </>
            ) : loadError ? (
              "The link may be old, or the model was removed."
            ) : (
              "Fetching the model details."
            )}
          </p>
        </header>

        <Panel title="Try the model" caption="Pre-filled with a sample row">
          {loadError ? (
            <PanelError title="This model isn't available" message={loadError} onRetry={load} />
          ) : !schema || !deployment ? (
            <PanelSkeleton rows={6} />
          ) : !deployed ? (
            <PanelEmpty icon={<FlaskConical />} title="Not live right now" description="This model is not deployed at the moment. Check back shortly." />
          ) : (
            <TryModel runId={runId} schema={schema} variant="simple" layout="stack" />
          )}
        </Panel>
      </main>
    </div>
  );
}

function humanize(featureCount: number, problemType?: string): string {
  if (problemType === "classification") return "a class";
  if (problemType === "regression") return "a value";
  return `from ${featureCount} features`;
}
