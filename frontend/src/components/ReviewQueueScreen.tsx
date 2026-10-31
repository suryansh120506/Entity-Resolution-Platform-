"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { ClusterQueue } from "@/components/ClusterQueue";
import { Num, Tabs } from "@/components/motion";
import { Notice, PageHeader, Skeleton, StatusDot, Tag, buttonClass } from "@/components/ui";
import { ApiError } from "@/lib/api/client";
import { getJob, getJobStats, listJobs } from "@/lib/api/jobs";
import type { ClusterStatus, JobStats, JobStatusResponse } from "@/lib/api/types";
import { formatDateTime, formatScore } from "@/lib/format";

/** The review queue's tabs, in the order a steward works through them. */
const TABS: { status: ClusterStatus; label: string; hint: string }[] = [
  {
    status: "PENDING_REVIEW",
    label: "To review",
    hint: "The matcher was unsure. Decide whether each group is one school.",
  },
  {
    status: "PENDING_API",
    label: "Being verified",
    hint: "The automatic check is searching the web and asking the LLM whether each group is one place. Only shown while any are waiting.",
  },
  {
    status: "AUTO_MERGED",
    label: "Merged automatically",
    hint: "Confident enough to merge without review. Spot-check them here.",
  },
  { status: "RESOLVED", label: "Confirmed", hint: "Merged by a reviewer." },
  { status: "REJECTED", label: "Rejected", hint: "A reviewer said these are different schools." },
];

export function ReviewQueueScreen() {
  const params = useSearchParams();
  const jobId = params.get("job");
  return jobId ? <JobQueue jobId={jobId} /> : <JobPicker />;
}

/** No job chosen: list the ones that have something to review. */
function JobPicker() {
  const router = useRouter();
  const [jobs, setJobs] = useState<JobStatusResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listJobs(50)
      .then((page) => {
        if (!cancelled) {
          setJobs(
            page.items.filter((job) => ["AWAITING_REVIEW", "COMPLETED"].includes(job.status)),
          );
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof ApiError ? caught.userMessage : "Could not load jobs.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="stagger mx-auto flex max-w-2xl flex-col gap-6">
      <PageHeader title="Review" description="Choose an upload to review its duplicate groups." />

      {error && <Notice tone="err">{error}</Notice>}

      {jobs === null && !error ? (
        <Skeleton className="h-40" />
      ) : jobs && jobs.length === 0 ? (
        <p className="rounded-lg border border-dashed border-hairline-strong px-6 py-12 text-center text-sm text-muted">
          No finished uploads yet.{" "}
          <Link href="/dashboard" className="link">
            Upload a file
          </Link>{" "}
          to get started.
        </p>
      ) : (
        <ul className="stagger flex flex-col overflow-hidden rounded-lg border border-hairline bg-surface">
          {jobs?.map((job) => (
            <li key={job.id} className="border-b border-hairline last:border-0">
              <button
                type="button"
                onClick={() => router.push(`/review?job=${job.id}`)}
                className="group flex w-full items-center gap-3 px-4 py-3.5 text-left transition-colors duration-150 hover:bg-surface-subtle"
              >
                <StatusDot tone={job.status === "AWAITING_REVIEW" ? "wait" : "ok"} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-foreground">
                    {job.original_filename ?? "Untitled upload"}
                  </span>
                  <span className="block font-mono text-xs text-muted">
                    {formatDateTime(job.created_at)}
                    {job.total_rows !== null && ` · ${job.total_rows.toLocaleString()} rows`}
                  </span>
                </span>
                <Tag tone={job.status === "AWAITING_REVIEW" ? "neutral" : "accent"}>
                  {job.status === "AWAITING_REVIEW" ? "Needs review" : "Done"}
                </Tag>
                <span
                  aria-hidden="true"
                  className="text-faint transition-[color,transform] duration-300 ease-[var(--ease-out)] group-hover:translate-x-1 group-hover:text-accent"
                >
                  →
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function JobQueue({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [stats, setStats] = useState<JobStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [tab, setTab] = useState<ClusterStatus>("PENDING_REVIEW");
  const [pageInfo, setPageInfo] = useState({ page: 1, count: 0, hasNext: false });

  useEffect(() => {
    let cancelled = false;

    // Cleared up front, so a previous job's numbers never sit beside a new
    // job's name while it loads — or indefinitely if the new request fails.
    setStats(null);
    setJob(null);
    setStatsError(null);
    setTab("PENDING_REVIEW");

    Promise.all([getJob(jobId), getJobStats(jobId)])
      .then(([nextJob, nextStats]) => {
        if (cancelled) return;
        setJob(nextJob);
        setStats(nextStats);
      })
      .catch((caught: unknown) => {
        if (cancelled) return;
        setStatsError(
          caught instanceof ApiError ? caught.userMessage : "Could not load this job.",
        );
      });

    return () => {
      cancelled = true;
    };
  }, [jobId]);

  // Stable identity so ClusterQueue's reporting effect does not re-fire every
  // render and loop.
  const onPageInfo = useCallback((info: typeof pageInfo) => setPageInfo(info), []);

  const counts = stats?.clusters_by_status;
  const visibleTabs = TABS.filter(
    (t) => t.status !== "PENDING_API" || (counts?.PENDING_API ?? 0) > 0,
  );
  const activeTab = TABS.find((t) => t.status === tab) ?? TABS[0];

  return (
    <section className="stagger flex flex-col gap-6">
      <PageHeader
        crumbs={[{ href: "/review", label: "Review" }]}
        title={job?.original_filename ?? "Review queue"}
        description={<span className="font-mono text-xs">{jobId.slice(0, 8)}</span>}
        actions={
          <Link href={`/jobs/${jobId}`} className={buttonClass("secondary", "sm")}>
            Results
          </Link>
        }
      />

      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-hairline bg-hairline sm:grid-cols-4">
        {[
          { label: "waiting for you", value: counts?.PENDING_REVIEW },
          { label: "auto-merged", value: counts?.AUTO_MERGED, accent: true },
          { label: "decided", value: counts ? counts.RESOLVED + counts.REJECTED : undefined },
        ].map((tile) => (
          <div key={tile.label} className="flex flex-col gap-2 bg-surface px-4 py-3">
            <dt className="label">{tile.label}</dt>
            <dd className={`text-xl leading-none ${tile.accent ? "text-accent" : "text-foreground"}`}>
              {tile.value === undefined ? <span className="font-mono text-faint">—</span> : <Num value={tile.value} />}
            </dd>
          </div>
        ))}
        <div className="flex flex-col gap-2 bg-surface px-4 py-3">
          <dt className="label">avg. confidence</dt>
          <dd className="font-mono text-xl leading-none text-foreground">
            {formatScore(stats?.mean_confidence ?? null)}
          </dd>
        </div>
      </dl>

      {statsError && <Notice tone="err">{statsError}</Notice>}

      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Tabs
            label="Duplicate groups"
            value={tab}
            onChange={setTab}
            items={visibleTabs.map((option) => ({
              value: option.status,
              label: option.label,
              count: counts?.[option.status],
            }))}
          />
          <span className="font-mono text-xs text-muted">
            page {pageInfo.page}
            {pageInfo.hasNext ? "+" : ""} · {pageInfo.count} shown
          </span>
        </div>
        <p key={tab} className="animate-rise text-xs text-muted">
          {activeTab.hint}
        </p>
      </div>

      {/* PENDING_REVIEW is the *cluster* status. AWAITING_REVIEW is the *job*
          status for the same moment; asking the clusters endpoint for it
          returns an empty page rather than an error. `clustersTotal` lets the
          zero-state tell "all decided" from "never produced any clusters". */}
      <ClusterQueue
        key={tab}
        jobId={jobId}
        status={tab}
        clustersTotal={stats?.clusters_total ?? null}
        onPageInfo={onPageInfo}
      />
    </section>
  );
}
