"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api/client";
import { exportUrl, getJobSummary, listRecords } from "@/lib/api/jobs";
import type { JobSummary, RecordItem, RecordView } from "@/lib/api/types";
import { formatDateTime } from "@/lib/format";

import { Num, Tabs } from "./motion";
import { Meter, Notice, PageHeader, SectionTitle, Skeleton, StatusLines, Tag, Window, buttonClass } from "./ui";

const PAGE_SIZE = 50;

const VIEWS: { view: RecordView; label: string; hint: string }[] = [
  {
    view: "needs_info",
    label: "Needs info",
    hint: "Real schools described too thinly to match. Fill in what's missing and re-upload.",
  },
  { view: "junk", label: "Junk", hint: "Rows that do not describe a school at all." },
  {
    view: "unique",
    label: "Unique",
    hint: "Usable records with no duplicate found. These are already clean.",
  },
  { view: "all", label: "All rows", hint: "Every uploaded row, cleaned." },
];

const FIELDS: { key: string; label: string }[] = [
  { key: "name", label: "name" },
  { key: "address", label: "address" },
  { key: "district", label: "district" },
  { key: "state", label: "state" },
  { key: "pincode", label: "pincode" },
];

/**
 * One job's results, laid out as the Medallion it is: Bronze (every row
 * uploaded), Silver (usable, cleaned, grouped) and Gold (one row per entity),
 * then the rows themselves.
 */
export function JobResults({ jobId }: { jobId: string }) {
  const [summary, setSummary] = useState<JobSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<RecordView>("needs_info");

  useEffect(() => {
    let cancelled = false;
    getJobSummary(jobId)
      .then((next) => {
        if (!cancelled) setSummary(next);
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof ApiError ? caught.userMessage : "Could not load results.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (error) {
    return (
      <div className="flex flex-col gap-3">
        <Notice tone="err">{error}</Notice>
        <Link href="/dashboard" className="link w-fit text-sm">
          Back to dashboard
        </Link>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="flex flex-col gap-8">
        <Skeleton className="h-14 w-2/3" />
        <Skeleton className="h-44" />
        <Skeleton className="h-72" />
      </div>
    );
  }

  const { job, funnel, clusters_by_status: c } = summary;
  const toReview = c.PENDING_REVIEW;
  const counts: Record<RecordView, number> = {
    needs_info: funnel.missing_context,
    junk: funnel.junk,
    unique: funnel.unique_records,
    all: funnel.total_rows,
  };

  return (
    <section className="stagger flex flex-col gap-10">
      <PageHeader
        crumbs={[{ href: "/dashboard", label: "Dashboard" }]}
        title={job.original_filename ?? "Untitled upload"}
        description={
          <span className="font-mono text-xs">
            {formatDateTime(job.created_at)} · {funnel.total_rows.toLocaleString()} rows ·{" "}
            <span className={job.status === "COMPLETED" ? "text-accent" : "text-foreground"}>
              {job.status.toLowerCase()}
            </span>
          </span>
        }
        actions={
          <>
            <AdvancedExports jobId={job.id} complete={job.status === "COMPLETED"} />
            {toReview > 0 && (
              <Link href={`/review?job=${job.id}`} className={buttonClass("secondary")}>
                Review <Tag>{toReview.toLocaleString()}</Tag>
              </Link>
            )}
            {job.status === "COMPLETED" ? (
              <a
                href={exportUrl(job.id, "client")}
                className={buttonClass("primary")}
                title="One row per entity: duplicates merged, readable names, web-found details, and every column of your original file."
              >
                Download database
              </a>
            ) : (
              <span
                className="inline-flex h-8 cursor-not-allowed items-center rounded-md border border-dashed border-hairline-strong px-3.5 text-xs text-muted"
                title="Available once processing finishes. A partial export is under Exports."
              >
                Database ready when processing finishes
              </span>
            )}
          </>
        }
      />

      <Medallion summary={summary} />

      <div className="grid gap-4 md:grid-cols-2">
        <IssueBreakdown
          title="Why rows need more info"
          counts={summary.issues_by_class.Missing_Context}
          total={funnel.missing_context}
          labels={summary.issue_labels}
        />
        <IssueBreakdown
          title="Why rows were treated as junk"
          counts={summary.issues_by_class.Junk}
          total={funnel.junk}
          labels={summary.issue_labels}
        />
      </div>

      <section className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <SectionTitle>Rows</SectionTitle>
          <Tabs
            label="Rows"
            value={view}
            onChange={setView}
            items={VIEWS.map((option) => ({
              value: option.view,
              label: option.label,
              count: counts[option.view],
            }))}
          />
        </div>
        <p key={view} className="animate-rise text-xs text-muted">
          {VIEWS.find((v) => v.view === view)?.hint}
        </p>
        <RecordTable jobId={job.id} view={view} labels={summary.issue_labels} />
      </section>
    </section>
  );
}

/**
 * Bronze -> Silver -> Gold: three stage cards joined by a line that draws
 * itself. Every number counts up when the page opens.
 */
function Medallion({ summary }: { summary: JobSummary }) {
  const { funnel, clusters_by_status: c } = summary;
  const pct = (n: number) => (funnel.total_rows ? `${Math.round((n / funnel.total_rows) * 100)}%` : "—");

  const decided = c.AUTO_MERGED + c.RESOLVED + c.REJECTED;
  const pending = c.PENDING_API + c.PENDING_REVIEW;

  return (
    <section className="flex flex-col gap-3">
      <SectionTitle right={<span className="font-mono">bronze → silver → gold</span>}>Pipeline</SectionTitle>
      <div className="stagger grid items-stretch gap-3 lg:grid-cols-[minmax(0,1fr)_1.5rem_minmax(0,1fr)_1.5rem_minmax(0,1.15fr)]">
        <StageCard tier="Bronze" caption="Every row uploaded" big={funnel.total_rows}>
          <StatusLines
            keyWidth="8ch"
            rows={[
              ["usable", <span key="v">{funnel.valid.toLocaleString()} <span className="text-faint">{pct(funnel.valid)}</span></span>],
              ["thin", <span key="m" className="text-muted">{funnel.missing_context.toLocaleString()} <span className="text-faint">{pct(funnel.missing_context)}</span></span>],
              ["junk", <span key="j" className="text-muted">{funnel.junk.toLocaleString()} <span className="text-faint">{pct(funnel.junk)}</span></span>],
            ]}
          />
        </StageCard>

        <Connector />

        <StageCard tier="Silver" caption="Duplicate groups found" big={funnel.clusters}>
          <StatusLines
            keyWidth="10ch"
            rows={[
              ["auto-merged", <span key="a" className="text-accent">{c.AUTO_MERGED.toLocaleString()}</span>],
              ["confirmed", c.RESOLVED.toLocaleString()],
              ["kept apart", c.REJECTED.toLocaleString()],
              ...(pending > 0
                ? [["pending", <span key="p" className="text-muted">{pending.toLocaleString()}</span>] as [string, React.ReactNode]]
                : []),
            ]}
          />
        </StageCard>

        <Connector />

        <StageCard tier="Gold" caption="Entities, one row each" big={funnel.unique_records} gold>
          <div className="flex flex-col gap-3">
            <StatusLines
              keyWidth="10ch"
              rows={[["rows folded", summary.records_merged_away.toLocaleString()]]}
            />
            {funnel.clusters > 0 && (
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-muted">
                  {decided.toLocaleString()} of {funnel.clusters.toLocaleString()} groups decided
                </span>
                <Meter
                  value={decided / funnel.clusters}
                  tone={pending > 0 ? "wait" : "active"}
                  live={pending > 0}
                  label="Duplicate groups decided"
                />
              </div>
            )}
          </div>
        </StageCard>
      </div>
    </section>
  );
}

function StageCard({
  tier,
  caption,
  big,
  gold = false,
  children,
}: {
  tier: string;
  caption: string;
  big: number;
  gold?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`relative flex flex-col gap-4 overflow-hidden rounded-lg border bg-surface p-4 transition-colors duration-300 ${
        gold ? "border-accent-border" : "border-hairline hover:border-hairline-strong"
      }`}
    >
      {gold && (
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 h-24"
          style={{ background: "linear-gradient(180deg, var(--accent-glow), transparent)" }}
        />
      )}
      <div className="relative flex items-center justify-between">
        <span className="label">{tier}</span>
        <span className="text-xs text-muted">{caption}</span>
      </div>
      <span className={`relative text-2xl font-medium leading-none ${gold ? "text-accent" : "text-foreground"}`}>
        <Num value={big} />
      </span>
      <div className="relative">{children}</div>
    </div>
  );
}

/** The line between two stages. It draws itself left to right on load. */
function Connector() {
  return (
    <div aria-hidden="true" className="flex items-center justify-center max-lg:h-4">
      <span className="relative block h-px w-full overflow-hidden bg-hairline max-lg:h-full max-lg:w-px">
        <span className="absolute inset-0 origin-left animate-[draw_0.9s_var(--ease-out)_0.3s_both] bg-accent-border max-lg:origin-top" />
      </span>
    </div>
  );
}

function IssueBreakdown({
  title,
  counts,
  total,
  labels,
}: {
  title: string;
  counts: Record<string, number> | undefined;
  total: number;
  labels: Record<string, string>;
}) {
  const entries = Object.entries(counts ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <Window title={title} meta={total.toLocaleString()}>
      {total === 0 ? (
        <p className="text-sm text-faint">None.</p>
      ) : entries.length === 0 ? (
        <p className="text-sm text-muted">
          {total.toLocaleString()} rows. A per-reason breakdown is recorded for uploads processed
          from now on.
        </p>
      ) : (
        <ul className="stagger flex flex-col gap-3">
          {entries.map(([code, n]) => (
            <li key={code} className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between gap-3 text-xs">
                <span className="min-w-0 truncate text-foreground">{labels[code] ?? code}</span>
                <span className="shrink-0 font-mono text-muted">{n.toLocaleString()}</span>
              </div>
              <Meter value={n / total} tone="wait" showValue={false} label={labels[code] ?? code} />
            </li>
          ))}
        </ul>
      )}
    </Window>
  );
}

function RecordTable({
  jobId,
  view,
  labels,
}: {
  jobId: string;
  view: RecordView;
  labels: Record<string, string>;
}) {
  const [items, setItems] = useState<RecordItem[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Bumped per loaded page, so the rows re-run their entrance.
  const [pageKey, setPageKey] = useState(0);
  // Keyset pages forward only, so "previous" is remembered, not computed.
  const history = useRef<(string | null)[]>([null]);
  const generation = useRef(0);

  const load = useCallback(
    async (cursor: string | null) => {
      const token = (generation.current += 1);
      setLoading(true);
      setError(null);
      try {
        const page = await listRecords(jobId, { view, limit: PAGE_SIZE, cursor });
        if (generation.current !== token) return;
        setItems(page.items);
        setNextCursor(page.next_cursor);
        setPageKey((key) => key + 1);
      } catch (caught) {
        if (generation.current !== token) return;
        setError(caught instanceof ApiError ? caught.userMessage : "Could not load rows.");
      } finally {
        if (generation.current === token) setLoading(false);
      }
    },
    [jobId, view],
  );

  useEffect(() => {
    history.current = [null];
    void load(null);
  }, [load]);

  if (error) {
    return <Notice tone="err">{error}</Notice>;
  }

  const page = history.current.length;

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto rounded-lg border border-hairline bg-surface">
        <table className="w-full min-w-[720px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-hairline text-left">
              <th scope="col" className="w-12 px-4 py-2.5 text-right font-normal">
                <span className="label">#</span>
              </th>
              {FIELDS.map((field) => (
                <th key={field.key} scope="col" className="px-3 py-2.5 font-normal">
                  <span className="label">{field.label}</span>
                </th>
              ))}
              <th scope="col" className="px-3 py-2.5 font-normal">
                <span className="label">issues</span>
              </th>
            </tr>
          </thead>
          <tbody
            key={pageKey}
            className={`stagger transition-opacity duration-200 ${loading ? "opacity-40" : ""}`}
          >
            {items.length === 0 && !loading ? (
              <tr>
                <td colSpan={FIELDS.length + 2} className="px-3 py-14 text-center text-sm text-muted">
                  No rows here.
                </td>
              </tr>
            ) : items.length === 0 ? (
              Array.from({ length: 6 }, (_, index) => (
                <tr key={index} className="border-b border-hairline last:border-0">
                  <td colSpan={FIELDS.length + 2} className="px-4 py-3">
                    <Skeleton className="h-3.5 rounded-sm" />
                  </td>
                </tr>
              ))
            ) : (
              items.map((item, index) => (
                <tr
                  key={item.record_id}
                  className="border-b border-hairline align-top transition-colors duration-150 last:border-0 hover:bg-surface-subtle"
                >
                  <td className="px-4 py-2 text-right font-mono text-xs text-faint">
                    {(page - 1) * PAGE_SIZE + index + 1}
                  </td>
                  {FIELDS.map((field) => (
                    <td key={field.key} className="px-3 py-2 text-foreground">
                      {item.clean_fields[field.key] || <span className="text-faint">—</span>}
                    </td>
                  ))}
                  <td className="px-3 py-2">
                    <div className="flex flex-wrap gap-1">
                      {item.issues.length === 0 ? (
                        <span className="text-faint">—</span>
                      ) : (
                        item.issues.map((issue) => <Tag key={issue}>{labels[issue] ?? issue}</Tag>)
                      )}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-end gap-2 text-xs">
        <span className="mr-auto font-mono text-muted">
          Page <span className="text-foreground">{page}</span>
        </span>
        <button
          type="button"
          onClick={() => {
            if (history.current.length < 2 || loading) return;
            history.current.pop();
            void load(history.current[history.current.length - 1]);
          }}
          disabled={page < 2 || loading}
          className={buttonClass("secondary", "sm")}
        >
          Previous
        </button>
        <button
          type="button"
          onClick={() => {
            if (!nextCursor || loading) return;
            history.current.push(nextCursor);
            void load(nextCursor);
          }}
          disabled={!nextCursor || loading}
          className={buttonClass("secondary", "sm")}
        >
          Next
        </button>
      </div>
    </div>
  );
}

/**
 * Engineer-facing exports, out of the way of the one a client needs: the raw
 * Gold dump (pipeline ids, scores, who decided, enrichment provenance), every
 * uploaded row cleaned, and the record -> master id mapping.
 *
 * A native <details>: keyboard and screen-reader behaviour for free, and it
 * closes itself when a link is followed into a download.
 */
function AdvancedExports({ jobId, complete }: { jobId: string; complete: boolean }) {
  const items = [
    {
      href: exportUrl(jobId, "gold", { partial: !complete }),
      label: complete ? "Raw dump" : "Raw dump (partial)",
      hint: "Pipeline ids, scores, who decided, where each web value came from.",
    },
    {
      href: exportUrl(jobId, "cleaned"),
      label: "Every row, cleaned",
      hint: "One row per uploaded row, with its outcome.",
    },
    {
      href: exportUrl(jobId, "mapping"),
      label: "ID mapping",
      hint: "Each record's ID and the ID of the record it was merged into.",
    },
  ];
  return (
    <details className="group relative">
      <summary className={`${buttonClass("ghost")} cursor-pointer list-none [&::-webkit-details-marker]:hidden`}>
        Exports
        <svg
          aria-hidden="true"
          viewBox="0 0 12 12"
          className="h-3 w-3 transition-transform duration-300 ease-[var(--ease-out)] group-open:rotate-180"
        >
          <path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </summary>
      <div className="menu-in absolute right-0 z-20 mt-2 w-80 overflow-hidden rounded-lg border border-hairline-strong bg-surface p-1 shadow-[0_12px_32px_-12px_rgb(0_0_0/0.35)]">
        {items.map((item) => (
          <a
            key={item.label}
            href={item.href}
            className="flex flex-col gap-0.5 rounded-md px-3 py-2 transition-colors duration-150 hover:bg-surface-raised"
          >
            <span className="text-sm text-foreground">{item.label}</span>
            <span className="text-xs text-muted">{item.hint}</span>
          </a>
        ))}
      </div>
    </details>
  );
}
