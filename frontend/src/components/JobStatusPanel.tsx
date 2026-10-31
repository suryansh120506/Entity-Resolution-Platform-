"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api/client";
import { deleteJob, exportUrl, getJob, getJobStats } from "@/lib/api/jobs";
import type { JobStats, JobStatusResponse, StageEntry, StageProgress } from "@/lib/api/types";
import { formatDateTime, formatDuration } from "@/lib/format";

import { Num } from "./motion";
import { Meter, Notice, Skeleton, StatusDot, StatusLines, Tag, Window, buttonClass, type Tone } from "./ui";

/** FIG 2.2, in order, with the words a user sees. */
const PIPELINE: readonly { status: string; label: string }[] = [
  { status: "PENDING", label: "Queued" },
  { status: "PREPROCESSING", label: "Reading" },
  { status: "EMBEDDING", label: "Embedding" },
  { status: "BLOCKING", label: "Candidates" },
  { status: "MATCHING", label: "Matching" },
  { status: "AWAITING_REVIEW", label: "Review" },
  { status: "RESOLVING", label: "Verifying" },
  { status: "COMPLETED", label: "Done" },
];
const PIPELINE_STATUSES = PIPELINE.map((step) => step.status);

const TERMINAL = new Set(["COMPLETED", "FAILED"]);
/** Nothing left for the pipeline to do: poll slowly, show results. */
const SETTLED = new Set(["COMPLETED", "FAILED", "AWAITING_REVIEW"]);
/** A worker may be writing this job right now; the backend refuses deletes. */
const ACTIVE = new Set(["PREPROCESSING", "EMBEDDING", "BLOCKING", "MATCHING", "RESOLVING"]);

/**
 * `job.status` -> the `stage_progress` key whose `done`/`total` represents
 * what is *currently* running under that status. PREPROCESSING covers both
 * `ingest` and `classify` (one task, two counters written in sequence).
 */
const ACTIVE_STAGE_KEY: Record<string, string | ((p: StageProgress) => string)> = {
  PREPROCESSING: (p) => (hasEntry(p.classify) ? "classify" : "ingest"),
  EMBEDDING: "embedding",
  BLOCKING: "blocking",
  MATCHING: "matching",
  // While RESOLVING, the work is automated triage (web search + LLM check
  // of the unsure groups) when there is any; otherwise the gate itself.
  RESOLVING: (p) => (hasEntry(p.triage) ? "triage" : "resolution"),
};

function hasEntry(entry: StageProgress[string]): entry is StageEntry {
  return entry !== undefined && "done" in entry;
}

/**
 * Relative wall time of each stage, summing to 1, measured on a real
 * 51,668-row upload: reading ~17s, embedding ~13 min, blocking and matching
 * under a minute together. The per-stage meter is the exact number; this one
 * is "roughly how far through the whole job".
 */
const STAGE_WEIGHT: Record<string, number> = {
  PENDING: 0,
  PREPROCESSING: 0.05,
  EMBEDDING: 0.8,
  BLOCKING: 0.04,
  MATCHING: 0.08,
  AWAITING_REVIEW: 0.02,
  RESOLVING: 0.01,
  COMPLETED: 0,
};

interface Props {
  jobId: string;
  onDeleted?: (jobId: string) => void;
}

/**
 * One job on the dashboard, as an ER_OS window: a status block (what it is,
 * where it is), the pipeline as a process line, a meter while it runs, and
 * the next action in the title bar.
 *
 * Polls every 2s while the pipeline is working and every 15s once settled.
 * Everything is read from Postgres via the API: TRD 2.7 requires a job to be
 * diagnosable from Postgres alone.
 */
export function JobStatusPanel({ jobId, onDeleted }: Props) {
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [stats, setStats] = useState<JobStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  // Re-rendered every second purely to keep elapsed time and ETA ticking.
  const [now, setNow] = useState(() => Date.now());

  // Fallback rate baseline for jobs whose progress predates server-side
  // timestamps: the first observation of each stage this component saw.
  const stageStart = useRef<Record<string, { time: number; done: number }>>({});

  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const [nextJob, nextStats] = await Promise.all([
          getJob(jobId),
          getJobStats(jobId).catch(() => null),
        ]);
        if (cancelled) return;

        setJob(nextJob);
        if (nextStats) setStats(nextStats);
        setError(null);
        recordStageStart(stageStart.current, nextJob);

        if (!TERMINAL.has(nextJob.status)) {
          timer = setTimeout(tick, SETTLED.has(nextJob.status) ? 15_000 : 2000);
        }
      } catch (caught) {
        if (cancelled) return;
        if (caught instanceof ApiError && caught.status === 404) {
          // Deleted elsewhere (another tab): drop it from the list.
          onDeleted?.(jobId);
          return;
        }
        setError(caught instanceof ApiError ? caught.userMessage : "Could not reach the API.");
        // Back off rather than hammering a backend that is already unhappy.
        timer = setTimeout(tick, 5000);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId, onDeleted]);

  if (error && !job) {
    return <Notice tone="err">{error}</Notice>;
  }

  if (!job) {
    return <Skeleton className="h-48" />;
  }

  const failure = failureOf(job.stage_progress);
  const pendingReview = stats?.clusters_by_status.PENDING_REVIEW ?? 0;
  const settled = SETTLED.has(job.status);
  const running = !settled && job.status !== "PENDING";
  const elapsedMs = (job.completed_at ? Date.parse(job.completed_at) : now) - Date.parse(job.created_at);

  const remove = async () => {
    const label = job.original_filename ?? "this job";
    if (!window.confirm(`Delete ${label} and all of its results? This cannot be undone.`)) return;
    setDeleting(true);
    try {
      await deleteJob(job.id);
      onDeleted?.(job.id);
    } catch (caught) {
      setDeleting(false);
      setError(caught instanceof ApiError ? caught.userMessage : "Could not delete the job.");
    }
  };

  const actions = (
    <>
      {!ACTIVE.has(job.status) && (
        <button
          type="button"
          onClick={() => void remove()}
          disabled={deleting}
          className={buttonClass("danger", "sm")}
          aria-label={`Delete ${job.original_filename ?? "job"}`}
        >
          {deleting ? "Deleting…" : "Delete"}
        </button>
      )}
      {job.status === "RESOLVING" && (
        <a
          href={exportUrl(job.id, "gold", { partial: true })}
          className={buttonClass("ghost", "sm")}
          title="The final database is ready when processing finishes. This raw dump lists undecided groups as unresolved."
        >
          Partial export
        </a>
      )}
      {settled && job.status !== "FAILED" && pendingReview > 0 && (
        <Link href={`/review?job=${job.id}`} className={buttonClass("secondary", "sm")}>
          Review <Tag>{pendingReview.toLocaleString()}</Tag>
        </Link>
      )}
      {settled && job.status !== "FAILED" && (
        <Link href={`/jobs/${job.id}`} className={buttonClass("secondary", "sm")}>
          Results
        </Link>
      )}
      {job.status === "COMPLETED" && (
        <a
          href={exportUrl(job.id, "client")}
          className={buttonClass("primary", "sm")}
          title="One row per entity: duplicates merged, readable names, web-found details, and every column of your original file."
        >
          <DownloadIcon />
          Download
        </a>
      )}
    </>
  );

  return (
    <Window
      title={<span title={job.original_filename ?? job.id}>{job.original_filename ?? "Untitled upload"}</span>}
      meta={job.total_rows !== null ? `${job.total_rows.toLocaleString()} rows` : undefined}
      status={<StatusDot tone={statusTone(job.status)} label={statusLabel(job.status)} />}
      right={actions}
    >
      <div className="flex flex-col gap-5">
        <Stepper current={job.status} />

        <div className="grid gap-x-10 gap-y-5 md:grid-cols-[minmax(0,17rem)_minmax(0,1fr)]">
          <StatusLines
            keyWidth="9ch"
            rows={[
              ["status", <StatusText key="s" status={job.status} />],
              ["started", formatDateTime(job.created_at)],
              [running ? "elapsed" : "took", formatDuration(elapsedMs)],
              ["job", <span key="i" title={job.id} className="text-muted">{job.id.slice(0, 8)}</span>],
            ]}
          />
          <div className="flex min-w-0 flex-col justify-center gap-3">
            {running && <LiveProgress job={job} now={now} stageStart={stageStart.current} />}
            {settled && stats && job.status !== "FAILED" && <Outcome stats={stats} />}
            {job.status === "PENDING" && (
              <Notice tone="wait">
                Waiting for a pipeline worker to pick this up. If it stays here, check the status
                at the top right.
              </Notice>
            )}
          </div>
        </div>

        <WebDetailsProgress job={job} />

        {(failure || error) && <Notice tone="err">{failure ?? error}</Notice>}
      </div>
    </Window>
  );
}

function DownloadIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3.5 w-3.5">
      <path
        d="M8 2.5v8m0 0L4.5 7M8 10.5 11.5 7M3 13.5h10"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function statusTone(status: string): Tone {
  if (status === "FAILED") return "err";
  if (status === "COMPLETED") return "ok";
  if (status === "AWAITING_REVIEW") return "wait";
  if (status === "PENDING") return "idle";
  return "active";
}

function statusLabel(status: string): string {
  if (status === "FAILED") return "Failed";
  return PIPELINE.find((step) => step.status === status)?.label ?? status.toLowerCase();
}

function StatusText({ status }: { status: string }) {
  const tone = statusTone(status);
  const color = tone === "err" ? "text-err" : tone === "ok" || tone === "active" ? "text-accent" : "text-muted";
  return <span className={color}>{statusLabel(status).toLowerCase()}</span>;
}

/**
 * The pipeline as a stepper. Done steps fill with the accent and the line to
 * the next one draws itself; the running step radiates and its name shimmers.
 * FAILED is not a step: the step it stopped at turns red.
 */
function Stepper({ current }: { current: string }) {
  const index = PIPELINE_STATUSES.indexOf(current);
  const failedAt = current === "FAILED" ? Math.max(0, index) : -1;
  // A job that goes straight from matching to done skips review; showing that
  // step as done would claim a review happened.
  const skipsReview = current === "RESOLVING" || current === "COMPLETED";
  const steps = PIPELINE.filter((step) => !(step.status === "AWAITING_REVIEW" && skipsReview));

  return (
    <div className="-mx-1 overflow-x-auto px-1 pb-1">
      <ol aria-label="Pipeline steps" className="flex min-w-[40rem] items-center">
        {steps.map((step, position) => {
          const at = PIPELINE_STATUSES.indexOf(step.status);
          const failed = failedAt >= 0 && at === failedAt;
          const active = !failed && step.status === current;
          const complete = active && step.status === "COMPLETED";
          const done = (index >= 0 && at < index) || complete;
          const last = position === steps.length - 1;
          return (
            <li
              key={step.status}
              className={`flex items-center ${last ? "" : "flex-1"}`}
              aria-current={active ? "step" : undefined}
            >
              <span className="flex items-center gap-2">
                <StepMark state={failed ? "failed" : done ? "done" : active ? "active" : "pending"} />
                <span
                  className={`whitespace-nowrap text-xs transition-colors duration-300 ${
                    failed
                      ? "text-err"
                      : active
                        ? "text-shimmer font-medium"
                        : done
                          ? "text-foreground"
                          : "text-faint"
                  }`}
                >
                  {step.label}
                </span>
              </span>
              {!last && (
                <span aria-hidden="true" className="mx-3 h-px flex-1 overflow-hidden bg-hairline">
                  <span
                    className="block h-full bg-accent-border transition-[width] duration-700 ease-[var(--ease-out)]"
                    style={{ width: done && !complete ? "100%" : "0%" }}
                  />
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function StepMark({ state }: { state: "done" | "active" | "failed" | "pending" }) {
  if (state === "done") {
    return (
      <span className="flex h-4 w-4 animate-pop items-center justify-center rounded-full bg-accent text-background">
        <svg aria-hidden="true" viewBox="0 0 12 12" className="h-2.5 w-2.5">
          <path d="M2.5 6.2 5 8.5l4.5-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
    );
  }
  if (state === "failed") {
    return (
      <span className="flex h-4 w-4 animate-pop items-center justify-center rounded-full bg-err font-mono text-[10px] leading-none text-background">
        ×
      </span>
    );
  }
  if (state === "active") {
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full border border-accent">
        <span className="ping h-1.5 w-1.5 rounded-full bg-accent text-accent" />
      </span>
    );
  }
  return <span className="h-4 w-4 rounded-full border border-hairline-strong" />;
}

/** What the job found, once it has finished: four numbers that count up. */
function Outcome({ stats }: { stats: JobStats }) {
  const valid = stats.records_by_classification.Valid ?? 0;
  const c = stats.clusters_by_status;

  const tiles: { label: string; value: number; accent?: boolean }[] = [
    { label: "usable rows", value: valid },
    { label: "dup. groups", value: stats.clusters_total },
    { label: "auto-merged", value: c.AUTO_MERGED, accent: true },
    { label: "checked", value: c.RESOLVED + c.REJECTED },
  ];

  return (
    <div className="flex flex-col gap-2">
      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-hairline bg-hairline sm:grid-cols-4">
        {tiles.map((tile) => (
          <div key={tile.label} className="flex flex-col gap-1.5 bg-surface px-3 py-2.5">
            <dt className="label">{tile.label}</dt>
            <dd className={`text-lg leading-none ${tile.accent ? "text-accent" : "text-foreground"}`}>
              <Num value={tile.value} />
            </dd>
          </div>
        ))}
      </dl>
      {(c.PENDING_API > 0 || c.PENDING_REVIEW > 0) && (
        <p className="text-xs text-muted">
          {c.PENDING_API > 0 && `${c.PENDING_API.toLocaleString()} still being verified. `}
          {c.PENDING_REVIEW > 0 && `${c.PENDING_REVIEW.toLocaleString()} waiting for your review.`}
        </p>
      )}
    </div>
  );
}

/**
 * The failure reason the worker wrote into `stage_progress.error`, shown
 * because a failed job must be diagnosable without grepping worker logs.
 */
function failureOf(progress: StageProgress | undefined): string | null {
  const entry = progress?.error;
  return entry && "message" in entry ? entry.message : null;
}

/** Which `stage_progress` key is "live" under this job's current status. */
function activeStageKey(job: JobStatusResponse): string | null {
  const resolver = ACTIVE_STAGE_KEY[job.status];
  if (!resolver) return null;
  return typeof resolver === "function" ? resolver(job.stage_progress) : resolver;
}

/** Client-side rate baseline, used only when the server sent no timestamps. */
function recordStageStart(
  baseline: Record<string, { time: number; done: number }>,
  job: JobStatusResponse,
): void {
  const key = activeStageKey(job);
  if (!key) return;
  const entry = job.stage_progress?.[key];
  if (!hasEntry(entry)) return;

  const existing = baseline[key];
  if (!existing || entry.done < existing.done) {
    baseline[key] = { time: Date.now(), done: entry.done };
  }
}

/** Overall completion, 0-1: fixed stage weights blended with in-stage progress. */
function overallProgress(job: JobStatusResponse): number {
  const index = PIPELINE_STATUSES.indexOf(job.status);
  if (index < 0) return 0;

  const before = PIPELINE_STATUSES.slice(0, index).reduce(
    (sum, s) => sum + (STAGE_WEIGHT[s] ?? 0),
    0,
  );
  const key = activeStageKey(job);
  const entry = key ? job.stage_progress?.[key] : undefined;
  const within =
    hasEntry(entry) && entry.total !== undefined && entry.total > 0
      ? Math.min(1, entry.done / entry.total)
      : 0;
  return Math.min(1, before + (STAGE_WEIGHT[job.status] ?? 0) * within);
}

/**
 * Remaining time for the running stage, in ms, or null if not yet knowable.
 * Prefers the server's own `started_at`/`updated_at`: a rate measured from
 * when the stage really began, not from when this tab happened to open.
 */
function stageEta(
  entry: StageEntry,
  baseline: { time: number; done: number } | undefined,
  now: number,
): number | null {
  if (entry.total === undefined || entry.done >= entry.total) return null;
  const remaining = entry.total - entry.done;

  if (entry.started_at && entry.updated_at && entry.done > 0) {
    const elapsed = Date.parse(entry.updated_at) - Date.parse(entry.started_at);
    if (elapsed > 1000) return (remaining * elapsed) / entry.done;
  }
  if (baseline) {
    const since = now - baseline.time;
    const done = entry.done - baseline.done;
    if (since > 500 && done > 0) return remaining / (done / since);
  }
  return null;
}

interface LiveProgressProps {
  job: JobStatusResponse;
  now: number;
  stageStart: Record<string, { time: number; done: number }>;
}

/** The running step: its own meter, its count, and either an ETA or why it waits. */
function LiveProgress({ job, now, stageStart }: LiveProgressProps) {
  const key = activeStageKey(job);
  const entry = key ? job.stage_progress?.[key] : undefined;
  const progress = hasEntry(entry) ? entry : null;
  const hasTotal = progress !== null && progress.total !== undefined && progress.total > 0;
  const etaMs = progress && key ? stageEta(progress, stageStart[key], now) : null;
  const waiting = key ? waitingOf(job.stage_progress, key) : null;
  const stepLabel = statusLabel(job.status);

  return (
    <div className="flex min-w-0 flex-col gap-2.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 text-xs">
        <span className="text-muted">
          {stepLabel}{" "}
          {progress ? (
            <span className="font-mono text-foreground">
              <Num value={progress.done} />
              {progress.total !== undefined && (
                <span className="text-faint"> / {progress.total.toLocaleString()}</span>
              )}
            </span>
          ) : (
            "starting…"
          )}
        </span>
        <span className="font-mono text-muted">
          {waiting
            ? "paused"
            : etaMs !== null
              ? `~${formatDuration(etaMs)} left`
              : "estimating…"}
        </span>
      </div>
      <Meter
        value={hasTotal ? progress.done / (progress.total as number) : overallProgress(job)}
        tone={waiting ? "wait" : "active"}
        live={!waiting}
        label={hasTotal ? `${stepLabel} progress` : "Overall progress"}
      />
      {waiting && <WaitingNotice waiting={waiting} />}
    </div>
  );
}

interface Waiting {
  text: string;
  until: string | null;
  premium: boolean;
}

/**
 * Why a stage has stopped moving, when the worker knows: "Standard AI: daily
 * limit reached". Written by the worker while it waits out a provider's rate
 * limit (`<stage>_waiting` in stage_progress). Without it a rate-limited job
 * shows only an ETA extrapolated from its recent speed, which for a daily cap
 * is a number that means nothing.
 */
function waitingOf(progress: StageProgress, stage: string): Waiting | null {
  const entry = progress?.[`${stage}_waiting`];
  const details = hasEntry(entry) ? entry.details : undefined;
  if (!details || typeof details.waiting !== "string") return null;
  return {
    text: details.waiting,
    until: typeof details.until === "string" ? details.until : null,
    premium: details.premium === true,
  };
}

function WaitingNotice({ waiting }: { waiting: Waiting }) {
  const resumes = waiting.until
    ? new Date(waiting.until).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : null;
  return (
    <Notice tone="wait">
      <span className="text-foreground">{waiting.text}.</span>
      {resumes && <span className="text-muted"> Trying again about {resumes}.</span>}
      {!waiting.premium && (
        <span className="text-muted">
          {" "}
          The free tier refills slowly.{" "}
          <Link href="/settings" className="link">
            Add your own key in settings
          </Link>{" "}
          for a larger daily budget.
        </span>
      )}
    </Notice>
  );
}

/**
 * Tier 3 runs after the job is COMPLETED and can take days on a free LLM tier,
 * so a finished job still shows web enrichment moving. The download works at
 * any point: it contains whatever has been found so far.
 */
function WebDetailsProgress({ job }: { job: JobStatusResponse }) {
  const entry = job.stage_progress.golden_enrichment;
  if (job.status !== "COMPLETED" || !hasEntry(entry)) return null;
  const waiting = waitingOf(job.stage_progress, "golden_enrichment");
  const skipped = typeof entry.details?.skipped === "string" ? entry.details.skipped : null;
  if (skipped) return <p className="text-xs text-muted">Web details skipped: {skipped}.</p>;
  const total = entry.total ?? 0;
  if (total === 0) return null;
  return (
    <div className="flex flex-col gap-2.5 border-t border-hairline pt-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 text-xs">
        <span className="text-foreground">Web details</span>
        <span className="text-muted">
          {entry.done >= total ? (
            `Added for all ${total.toLocaleString()} merged entities`
          ) : (
            <>
              <span className="font-mono text-foreground">
                <Num value={entry.done} />
              </span>
              <span className="font-mono text-faint"> / {total.toLocaleString()}</span>
              <span> · download any time, the file holds what is found so far</span>
            </>
          )}
        </span>
      </div>
      {entry.done < total && (
        <Meter
          value={entry.done / total}
          tone={waiting ? "wait" : "active"}
          live={!waiting}
          label="Web details progress"
        />
      )}
      {waiting && <WaitingNotice waiting={waiting} />}
    </div>
  );
}
