"use client";

import { useEffect, useId, useRef, useState } from "react";
import { LogIn, LogOut, Mail, RotateCw, UserRound } from "lucide-react";
import {
  ApiError,
  getAuth,
  getMyRuns,
  requestCode,
  saveRun,
  signOut,
  verifyCode,
  type Account,
  type AuthState,
  type MyRun,
  type RunStatus,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Field, FieldError, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { toast } from "@/components/ui/toast";
import { StatusDot, type Tone } from "@/components/panel";
import { Segmented } from "@/components/segmented";
import { formatScore, metricInfo } from "./metrics";

// The backend refuses a second code for an address sooner than this.
const RESEND_SECONDS = 60;
const EMAIL = /^[^@\s]{1,64}@[^@\s]+\.[^@\s.]{2,}$/;

/** The backend's own sentence, without the status code ApiError appends to it. */
const say = (e: unknown, fallback: string) =>
  e instanceof ApiError && e.answered ? e.message.replace(/ \(\d{3}\)$/, "") : e instanceof Error && e.message ? e.message : fallback;

const STATUS: Record<MyRun["status"], { label: string; tone: Tone }> = {
  uploading: { label: "Reading the file", tone: "running" },
  uploaded: { label: "Ready to train", tone: "idle" },
  queued: { label: "Queued", tone: "running" },
  running: { label: "Training", tone: "running" },
  succeeded: { label: "Trained", tone: "done" },
  failed: { label: "Failed", tone: "malfunction" },
  cancelled: { label: "Cancelled", tone: "idle" },
  unknown: { label: "Unknown", tone: "idle" },
};

const when = (seconds: number) =>
  new Date(seconds * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

/**
 * Optional sign-in, in the header: a name and an email, and a code by email
 * every time instead of a password. Renders nothing when the backend offers no
 * accounts. Signed in, the same button opens the account: its runs, and
 * signing out. Signing in also saves the run that is open, so a run started
 * before deciding to sign in isn't left out.
 */
export function AccountControl({
  runId,
  onOpenRun,
}: {
  runId: string | null;
  onOpenRun: (runId: string, status: RunStatus) => void;
}) {
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [signInOpen, setSignInOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  // A fresh dialog for every opening: the sign-in form starts at the address
  // again, and the account reloads its runs.
  const [opening, setOpening] = useState(0);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let cancelled = false;
    // A backend from before accounts has no /auth: then there is no sign-in.
    getAuth().then(
      (a) => !cancelled && setAuth(a),
      () => {}
    );
    return () => {
      cancelled = true;
    };
  }, []);

  if (!auth?.enabled) return null;
  const user = auth.user;

  const openDialog = () => {
    setOpening((n) => n + 1);
    if (user) setAccountOpen(true);
    else setSignInOpen(true);
  };

  const signedIn = (account: Account) => {
    setSignInOpen(false);
    setAuth({ enabled: true, user: account });
    if (runId) saveRun(runId).catch(() => {});
    toast.add({ title: `Signed in as ${account.name}`, description: "Runs you upload or start are listed in your account.", type: "success" });
  };

  const signedOut = () => {
    setAccountOpen(false);
    setAuth({ enabled: true, user: null });
    toast.add({ title: "Signed out", description: "Your runs stay in your account for next time.", type: "info" });
  };

  return (
    <>
      <Button
        ref={buttonRef}
        variant="outline"
        size="sm"
        onClick={openDialog}
        aria-label={user ? `Account: ${user.name}` : "Sign in"}
        className="size-9 px-0 sm:w-auto sm:max-w-44 sm:ps-2 sm:pe-2.5 lg:h-8"
      >
        {user ? <UserRound /> : <LogIn />}
        <span className="hidden truncate sm:inline">{user ? user.name.split(" ")[0] : "Sign in"}</span>
      </Button>
      <SignInDialog key={`in-${opening}`} open={signInOpen} onOpenChange={setSignInOpen} onSignedIn={signedIn} finalFocus={buttonRef} />
      {user && (
        <AccountDialog
          key={`account-${opening}`}
          user={user}
          open={accountOpen}
          onOpenChange={setAccountOpen}
          onSignedOut={signedOut}
          onOpenRun={(id, status) => {
            setAccountOpen(false);
            onOpenRun(id, status);
          }}
          finalFocus={buttonRef}
        />
      )}
    </>
  );
}

type Mode = "signin" | "signup";
type Problem = { field: "name" | "email" | "code"; message: string } | null;

/**
 * Two steps: the address (and a name, to create an account), then the code
 * from the email. Asking for a code never says whether an address has an
 * account, so the second step hedges when signing in.
 */
function SignInDialog({
  open,
  onOpenChange,
  onSignedIn,
  finalFocus,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSignedIn: (account: Account) => void;
  finalFocus: React.RefObject<HTMLButtonElement | null>;
}) {
  const [mode, setMode] = useState<Mode>("signin");
  const [step, setStep] = useState<"details" | "code">("details");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [sentTo, setSentTo] = useState("");
  const [sentAt, setSentAt] = useState(0);
  const [now, setNow] = useState(0);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<Problem>(null);
  const [notice, setNotice] = useState("");
  const nameRef = useRef<HTMLInputElement>(null);
  const emailRef = useRef<HTMLInputElement>(null);
  const codeRef = useRef<HTMLInputElement>(null);
  // Set when a step changes on the user's say-so, so focus follows to the new step.
  const moved = useRef(false);
  const id = useId();
  const errorId = `${id}-error`;

  const wait = Math.max(0, RESEND_SECONDS - Math.floor((now - sentAt) / 1000));
  const counting = step === "code" && wait > 0;
  useEffect(() => {
    if (!counting) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [counting]);

  useEffect(() => {
    if (!moved.current) return;
    moved.current = false;
    (step === "code" ? codeRef : mode === "signup" ? nameRef : emailRef).current?.focus();
  }, [step, mode]);

  const goToStep = (next: "details" | "code") => {
    moved.current = true;
    setProblem(null);
    setNotice("");
    setStep(next);
  };

  const send = async (again = false) => {
    if (busy) return;
    const address = again ? sentTo : email.trim();
    const fullName = name.trim();
    if (!again && mode === "signup" && !fullName) return setProblem({ field: "name", message: "Enter your name." });
    if (!again && !EMAIL.test(address)) return setProblem({ field: "email", message: "Enter a valid email address." });
    setBusy(true);
    setProblem(null);
    setNotice("");
    try {
      await requestCode(mode === "signup" ? { email: address, name: fullName } : { email: address });
      const sent = Date.now();
      setSentTo(address);
      setSentAt(sent);
      setNow(sent);
      setCode("");
      if (again) setNotice("A new code is on its way. The last one no longer works.");
      else goToStep("code");
    } catch (e) {
      setProblem({ field: again ? "code" : "email", message: say(e, "The code couldn't be sent. Try again.") });
    } finally {
      setBusy(false);
    }
  };

  const verify = async (digits: string) => {
    if (busy) return;
    if (digits.length !== 6) return setProblem({ field: "code", message: "Enter the 6-digit code from the email." });
    setBusy(true);
    setProblem(null);
    setNotice("");
    try {
      const { user } = await verifyCode({ email: sentTo, code: digits });
      onSignedIn(user);
    } catch (e) {
      setProblem({ field: "code", message: say(e, "The code couldn't be checked. Try again.") });
      codeRef.current?.select();
    } finally {
      setBusy(false);
    }
  };

  const invalid = (field: NonNullable<Problem>["field"]) =>
    problem?.field === field ? { "aria-invalid": true, "aria-describedby": errorId } : {};
  const error = (field: NonNullable<Problem>["field"]) =>
    problem?.field === field && <FieldError id={errorId}>{problem.message}</FieldError>;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent initialFocus={emailRef} finalFocus={finalFocus}>
        <DialogHeader>
          <DialogTitle>{step === "code" ? "Check your email" : mode === "signup" ? "Create an account" : "Sign in"}</DialogTitle>
          <DialogDescription>
            {step === "details" ? (
              "No password: we email you a 6-digit code each time."
            ) : mode === "signup" ? (
              <>
                We emailed a 6-digit code to <span className="font-medium text-foreground">{sentTo}</span>. It expires in 10 minutes.
              </>
            ) : (
              <>
                If <span className="font-medium text-foreground">{sentTo}</span> has an account, we emailed it a 6-digit code. It
                expires in 10 minutes.
              </>
            )}
          </DialogDescription>
        </DialogHeader>

        <form
          noValidate
          className="flex flex-col gap-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (step === "details") send();
            else verify(code);
          }}
        >
          {step === "details" ? (
            <>
              <Segmented
                value={mode}
                onChange={(m) => {
                  setMode(m);
                  setProblem(null);
                }}
                label="Sign in or create an account"
                options={[
                  { value: "signin", label: "Sign in" },
                  { value: "signup", label: "Create account" },
                ]}
                className="w-full"
                itemClassName="flex-1"
              />
              {mode === "signup" && (
                <Field className="gap-1.5">
                  <FieldLabel htmlFor={`${id}-name`}>Name</FieldLabel>
                  <Input
                    ref={nameRef}
                    id={`${id}-name`}
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    autoComplete="name"
                    maxLength={80}
                    className="h-10 lg:h-8"
                    {...invalid("name")}
                  />
                  {error("name")}
                </Field>
              )}
              <Field className="gap-1.5">
                <FieldLabel htmlFor={`${id}-email`}>Email</FieldLabel>
                <Input
                  ref={emailRef}
                  id={`${id}-email`}
                  type="email"
                  inputMode="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                  autoCapitalize="none"
                  spellCheck={false}
                  dir="ltr"
                  className="h-10 lg:h-8"
                  {...invalid("email")}
                />
                {error("email")}
              </Field>
              <Button type="submit" disabled={busy} focusableWhenDisabled className="h-10 lg:h-8">
                {busy ? <Spinner data-icon="inline-start" /> : <Mail data-icon="inline-start" />}
                {busy ? "Sending..." : "Email me a code"}
              </Button>
              <p className="text-xs text-muted-foreground">
                Optional. Your account lists the runs you upload or start. Run links still open for anyone who has them.
              </p>
            </>
          ) : (
            <>
              <Field className="gap-1.5">
                <FieldLabel htmlFor={`${id}-code`}>Code</FieldLabel>
                <Input
                  ref={codeRef}
                  id={`${id}-code`}
                  value={code}
                  onChange={(e) => {
                    const digits = e.target.value.replace(/\D/g, "").slice(0, 6);
                    setCode(digits);
                    if (problem) setProblem(null);
                    if (digits.length === 6) verify(digits);
                  }}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  dir="ltr"
                  className="h-11 text-center font-mono text-lg tracking-[0.4em] md:text-lg lg:h-10"
                  {...invalid("code")}
                />
                {error("code")}
              </Field>
              <Button type="submit" disabled={busy} focusableWhenDisabled className="h-10 lg:h-8">
                {busy ? <Spinner data-icon="inline-start" /> : <LogIn data-icon="inline-start" />}
                {busy ? "Checking..." : "Verify"}
              </Button>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Button type="button" variant="ghost" size="sm" onClick={() => goToStep("details")} className="h-9 lg:h-7">
                  Use a different email
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => send(true)}
                  disabled={wait > 0 || busy}
                  focusableWhenDisabled
                  className="h-9 lg:h-7"
                >
                  <RotateCw data-icon="inline-start" />
                  {wait > 0 ? `Resend in ${wait} s` : "Resend code"}
                </Button>
              </div>
              <p role="status" className={notice ? "text-xs text-muted-foreground" : "sr-only"}>
                {notice}
              </p>
              {mode === "signin" && (
                <p className="text-xs text-muted-foreground">
                  Nothing after a minute? Check your spam folder, or{" "}
                  <button
                    type="button"
                    onClick={() => {
                      setMode("signup");
                      setEmail(sentTo);
                      goToStep("details");
                    }}
                    className="rounded-sm text-primary underline-offset-4 outline-none hover:underline focus-visible:ring-3 focus-visible:ring-ring/50"
                  >
                    create an account
                  </button>
                  .
                </p>
              )}
            </>
          )}
        </form>
      </DialogContent>
    </Dialog>
  );
}

/** Who is signed in, the runs they uploaded or started, and signing out. */
function AccountDialog({
  user,
  open,
  onOpenChange,
  onSignedOut,
  onOpenRun,
  finalFocus,
}: {
  user: Account;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSignedOut: () => void;
  onOpenRun: (runId: string, status: RunStatus) => void;
  finalFocus: React.RefObject<HTMLButtonElement | null>;
}) {
  const [runs, setRuns] = useState<MyRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [leaving, setLeaving] = useState(false);
  const headingId = useId();
  // Focus starts on the dialog itself: while the runs load, the first button
  // is "Sign out", and Enter there would sign the user out.
  const popupRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setError(null);
    getMyRuns().then(
      (r) => !cancelled && setRuns(r.runs),
      (e) => !cancelled && setError(say(e, "Your runs couldn't be loaded."))
    );
    return () => {
      cancelled = true;
    };
  }, [open, attempt]);

  const leave = async () => {
    if (leaving) return;
    setLeaving(true);
    try {
      await signOut();
      onSignedOut();
    } catch (e) {
      setError(say(e, "You couldn't be signed out. Try again."));
      setLeaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent ref={popupRef} className="sm:max-w-lg" initialFocus={popupRef} finalFocus={finalFocus}>
        <DialogHeader>
          <DialogTitle>{user.name}</DialogTitle>
          <DialogDescription dir="ltr" className="text-start">
            {user.email}
          </DialogDescription>
        </DialogHeader>

        <section aria-labelledby={headingId} className="space-y-2">
          <h3 id={headingId} className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            My runs
          </h3>
          {error ? (
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-background/40 p-3 ring-1 ring-border">
              <p role="alert" className="text-sm text-red-300">
                {error}
              </p>
              <Button variant="outline" size="sm" onClick={() => setAttempt((n) => n + 1)} className="h-9 lg:h-7">
                <RotateCw data-icon="inline-start" />
                Try again
              </Button>
            </div>
          ) : !runs ? (
            <div className="space-y-2" aria-hidden="true">
              {[0, 1, 2].map((i) => (
                <Skeleton key={i} className="h-12 w-full rounded-lg" />
              ))}
            </div>
          ) : runs.length === 0 ? (
            <p className="rounded-lg bg-background/40 p-3 text-sm text-muted-foreground ring-1 ring-border">
              No runs yet. The runs you upload or start while signed in are listed here.
            </p>
          ) : (
            <ul aria-labelledby={headingId} className="-mx-1 max-h-[55vh] space-y-1 overflow-y-auto px-1 py-0.5">
              {runs.map((run) => (
                <li key={run.run_id}>
                  <RunRow
                    run={run}
                    onOpen={() =>
                      onOpenRun(run.run_id, {
                        run_id: run.run_id,
                        status: run.status === "unknown" ? "uploaded" : run.status,
                        task: run.task ?? undefined,
                        target: run.target,
                      })
                    }
                  />
                </li>
              ))}
            </ul>
          )}
        </section>

        <DialogFooter>
          <Button variant="outline" onClick={leave} disabled={leaving} focusableWhenDisabled className="h-10 lg:h-8">
            {leaving ? <Spinner data-icon="inline-start" /> : <LogOut data-icon="inline-start" />}
            Sign out
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function RunRow({ run, onOpen }: { run: MyRun; onOpen: () => void }) {
  const status = STATUS[run.status] ?? STATUS.unknown;
  const score =
    typeof run.score === "number" && run.score_metric
      ? `${metricInfo(run.score_metric).short} ${formatScore(run.score, run.score_metric)}`
      : null;
  const details = [run.target && `Predicts ${run.target}`, run.model_name, score, when(run.added_at)].filter(Boolean).join(" · ");
  return (
    <button
      type="button"
      onClick={onOpen}
      className="flex w-full flex-col gap-1 rounded-lg px-2.5 py-2 text-start transition-colors outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50"
    >
      <span className="flex w-full items-center gap-2">
        <span className="min-w-0 flex-1 truncate font-medium">{run.filename || "Untitled dataset"}</span>
        <span className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
          <StatusDot tone={status.tone} />
          {status.label}
        </span>
      </span>
      <span className="truncate text-xs text-muted-foreground">{details}</span>
    </button>
  );
}
