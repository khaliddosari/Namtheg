"use client";

import { useEffect, useRef, useState } from "react";
import QRCode from "react-qr-code";
import { Check, CloudUpload, Copy, ExternalLink, FlaskConical, QrCode, RefreshCw, Rocket } from "lucide-react";
import { startDeploy } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button, buttonVariants } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Fact, Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel";
import { toast } from "@/components/ui/toast";
import { TryModel } from "@/components/try-model";
import type { RunState } from "./use-run";
import type { StageKey } from "./dashboard";

async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Some mobile browsers block the async clipboard API.
    const el = document.createElement("textarea");
    el.value = text;
    el.style.cssText = "position:fixed;opacity:0;pointer-events:none";
    document.body.appendChild(el);
    el.select();
    document.execCommand("copy");
    document.body.removeChild(el);
  }
}

export function DeployStage({
  runId,
  run,
  goToStage,
}: {
  runId: string | null;
  run: RunState;
  goToStage: (key: StageKey) => void;
}) {
  const { status, statusError, schema, schemaError, retryResults, deployment, setDeployment } = run;
  const trained = status?.status === "succeeded";
  // A linked run still loading, or a trained one whose model details are on their way.
  const loading = !!runId && !status && !statusError;
  const schemaLoading = trained && !schema && !schemaError;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shareUrl, setShareUrl] = useState("");

  // window.location only exists on the client, and the QR code needs a stable string.
  useEffect(() => {
    setShareUrl(runId ? `${window.location.origin}/share/${runId}` : "");
  }, [runId]);

  const deployed = deployment?.status === "succeeded" && !!deployment.predict_url;
  const deploying = busy || deployment?.status === "deploying";
  const deployError = error ?? (deployment?.status === "failed" ? (deployment.error ?? "Deployment failed.") : null);

  // Say so when a deployment that was building finishes, whichever way it went.
  const deployPhase = deployment?.status;
  const deployFailure = deployment?.error;
  const prevPhase = useRef(deployPhase);
  useEffect(() => {
    const before = prevPhase.current;
    prevPhase.current = deployPhase;
    if (before !== "deploying") return;
    if (deployPhase === "succeeded") {
      toast.add({ title: "The model is live", description: "Send it a prediction or share the test page.", type: "success" });
    } else if (deployPhase === "failed") {
      toast.add({ title: "Deployment failed", description: deployFailure ?? "The container did not start.", type: "error" });
    }
  }, [deployPhase, deployFailure]);

  const deploy = async () => {
    if (!runId) return;
    setError(null);
    setBusy(true);
    try {
      setDeployment(await startDeploy(runId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start the deployment.");
    } finally {
      setBusy(false);
    }
  };

  const notTrained = (
    <PanelEmpty
      icon={<Rocket />}
      title="No model yet"
      description="Deployment opens up once a run finishes training."
      action={
        <Button variant="outline" onClick={() => goToStage(status ? "training" : "data")} className="h-10 lg:h-8">
          {status ? "See training" : "Go to Data"}
        </Button>
      }
    />
  );

  // Locked: the deployment's facts at their own height, sharing below it, and
  // trying the model in the big area. On phones, trying it comes before sharing.
  return (
    <div className="grid flex-1 gap-3 md:grid-cols-2 lock:h-full lock:grid-cols-12 lock:grid-rows-[auto_minmax(0,1fr)]">
      {/* Deployment */}
      <Panel
        title="Deployment"
        caption={schema?.model_name ? `${schema.model_name} as a live API on Modal` : "The champion as a live API on Modal"}
        className="lock:col-span-4"
      >
        {loading || schemaLoading ? (
          <PanelSkeleton rows={4} />
        ) : !trained ? (
          notTrained
        ) : schemaError ? (
          <PanelError title="The model details didn't load" message={schemaError} onRetry={retryResults} />
        ) : (
          <>
            <p className="text-sm text-muted-foreground">
              {deployed
                ? "The model is live. Send it predictions here, or share the public test page."
                : deploying
                  ? "Packaging the model into a fresh container. First builds usually take 30 to 90 seconds."
                  : "Spin up a live container for the champion, then send it predictions from this page."}
            </p>

            {deployment && deployment.status !== "not_deployed" && (
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
                <Fact label="App" value={deployment.app_name ?? "-"} title={deployment.app_name} valueClassName="font-mono" />
                <Fact
                  label="Build time"
                  value={deployment.elapsed_seconds != null ? `${Math.round(deployment.elapsed_seconds)}s` : "-"}
                  valueClassName="font-mono tabular-nums"
                />
                {deployment.predict_url && (
                  <div className="col-span-2 min-w-0">
                    <dt className="text-xs text-muted-foreground">Endpoint</dt>
                    <dd className="mt-0.5 flex min-w-0 items-center gap-1">
                      <span className="min-w-0 flex-1 truncate font-mono text-xs" dir="ltr" title={deployment.predict_url}>
                        {deployment.predict_url}
                      </span>
                      <CopyButton text={deployment.predict_url} label="Copy endpoint URL" done="Endpoint URL copied" />
                    </dd>
                  </div>
                )}
              </dl>
            )}

            {!deployed && !deploying && !deployError && (
              <ul className="flex flex-col gap-1.5 text-sm">
                {["An HTTPS predict endpoint for the champion", "A public test page and QR code to share it", "Live predictions right here on the dashboard"].map((item) => (
                  <li key={item} className="flex items-start gap-2">
                    <Check aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-primary" />
                    {item}
                  </li>
                ))}
              </ul>
            )}

            {deployError && !deploying && (
              <div className="rounded-xl bg-red-500/10 p-3 ring-1 ring-red-400/25">
                <p className="text-xs font-medium text-red-200">Deployment failed</p>
                <p className="mt-1 line-clamp-3 font-mono text-xs break-words text-muted-foreground" title={deployError}>
                  {deployError}
                </p>
              </div>
            )}

            {!deployed && (
              <Button onClick={deploy} disabled={deploying || !schema} className="mt-auto h-10 lg:h-8">
                {deploying ? <Spinner data-icon="inline-start" /> : deployError ? <RefreshCw data-icon="inline-start" /> : <CloudUpload data-icon="inline-start" />}
                {deploying ? "Deploying..." : deployError ? "Retry deployment" : "Deploy to the cloud"}
              </Button>
            )}
          </>
        )}
      </Panel>

      {/* Try the model */}
      <Panel
        title="Try the model"
        caption={deployed ? "Pre-filled with a sample row. Change any value to see how the prediction moves." : "Live predictions from the deployed endpoint"}
        className="md:col-start-2 md:row-span-2 md:row-start-1 lock:col-span-8 lock:col-start-5 lock:row-span-2 lock:row-start-1"
      >
        {loading ? (
          <PanelSkeleton rows={6} />
        ) : !trained ? (
          notTrained
        ) : !deployed || !runId || !schema ? (
          <PanelEmpty
            icon={<FlaskConical />}
            title="Deploy to try the model"
            description="Once the model is live, send it a row and see the prediction here."
          />
        ) : (
          <TryModel runId={runId} schema={schema} variant="full" layout="split" />
        )}
      </Panel>

      {/* Share */}
      <Panel
        title="Share"
        caption="A public test page anyone can open, no account needed"
        className="lock:col-span-4 lock:col-start-1 lock:row-start-2"
      >
        {loading ? (
          <PanelSkeleton rows={3} />
        ) : !trained ? (
          notTrained
        ) : !deployed || !runId ? (
          <PanelEmpty icon={<QrCode />} title="Deploy to share" description="A QR code and a public test page link appear once the model is live." />
        ) : (
          // Side by side on a short page; on a tall one the code grows and centres.
          <div className="flex min-h-0 flex-1 flex-wrap items-center gap-4 lock:flex-col lock:flex-nowrap lock:justify-center lock-short:flex-row">
            {shareUrl && (
              <div className="shrink-0 rounded-xl bg-foreground p-2 text-background">
                <QRCode
                  value={shareUrl}
                  size={128}
                  fgColor="currentColor"
                  bgColor="transparent"
                  level="M"
                  className="size-28 lock:size-40 lock-short:size-24"
                  viewBox="0 0 256 256"
                />
              </div>
            )}
            <div className="flex min-w-0 flex-1 flex-col gap-2 lock:w-full lock:max-w-sm lock:flex-none lock-short:flex-1">
              <p className="truncate font-mono text-xs text-muted-foreground lock:text-center lock-short:text-start" dir="ltr" title={shareUrl}>
                {shareUrl}
              </p>
              <CopyButton text={shareUrl} label="Copy share link" done="Share link copied" showLabel />
              <a href={`/share/${runId}`} target="_blank" rel="noopener noreferrer" className={cn(buttonVariants({ variant: "outline", size: "sm" }), "h-9 lg:h-7")}>
                <ExternalLink data-icon="inline-start" />
                Open the test page
              </a>
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}

function CopyButton({ text, label, done, showLabel }: { text: string; label: string; done: string; showLabel?: boolean }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 1800);
    return () => clearTimeout(id);
  }, [copied]);
  const copy = async () => {
    await copyText(text);
    setCopied(true);
    toast.add({ title: done, description: text, type: "success" });
  };
  const Icon = copied ? Check : Copy;
  return showLabel ? (
    <Button variant="outline" size="sm" onClick={copy} className="h-9 lg:h-7">
      <Icon data-icon="inline-start" />
      {copied ? "Link copied" : label}
    </Button>
  ) : (
    <Button variant="ghost" size="icon-xs" onClick={copy} aria-label={copied ? "Copied" : label} className="size-8 lg:size-6">
      <Icon />
    </Button>
  );
}
