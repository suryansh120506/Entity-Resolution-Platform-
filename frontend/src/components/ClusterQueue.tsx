"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ConfidenceBadge, Kbd, Notice, buttonClass } from "@/components/ui";
import { getCluster, listClusters } from "@/lib/api/clusters";
import { ApiError } from "@/lib/api/client";
import { exportUrl } from "@/lib/api/jobs";
import type { ClusterStatus, ClusterSummary } from "@/lib/api/types";
import { canonicalValue, topDiscrepancy } from "@/lib/diff";
import { proposeMaster } from "@/lib/format";

const PAGE_SIZE = 25;

/** Concurrent detail fetches for the name/discrepancy columns. See `loadRowDetails`. */
const DETAIL_CONCURRENCY = 6;

/** What the name and discrepancy columns show, per cluster. */
interface RowDetail {
  name: string;
  place: string;
  discrepancy: string | null;
}

interface Props {
  jobId: string;
  /** Which band of clusters to page through. */
  status?: ClusterStatus;
  /** Total clusters in the job, used to tell "all resolved" from "none yet". */
  clustersTotal: number | null;
  onPageInfo?: (info: { page: number; count: number; hasNext: boolean }) => void;
}

export function ClusterQueue({
  jobId,
  status = "PENDING_REVIEW",
  clustersTotal,
  onPageInfo,
}: Props) {
  const router = useRouter();

  const [items, setItems] = useState<ClusterSummary[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, RowDetail | null>>({});
  const [active, setActive] = useState(0);

  // Keyset gives you the row *after* a position, so "previous" must be
  // remembered rather than computed — there is no backwards cursor to ask for.
  const history = useRef<(string | null)[]>([null]);
  const rowRefs = useRef<(HTMLTableRowElement | null)[]>([]);

  // Monotonic token identifying the current page load. Every async write
  // checks it before committing, so a slow batch from a page the steward has
  // already left cannot write into the page they are looking at now.
  const generation = useRef(0);

  useEffect(() => {
    // Invalidate any in-flight work when the component goes away.
    return () => {
      generation.current += 1;
    };
  }, []);

  const load = useCallback(
    async (cursor: string | null) => {
      const token = (generation.current += 1);
      setLoading(true);
      setError(null);
      try {
        const page = await listClusters({ jobId, status, limit: PAGE_SIZE, cursor });
        if (generation.current !== token) return;

        setItems(page.items);
        setNextCursor(page.next_cursor);
        setActive(0);
        setDetails({});
        void loadRowDetails(page.items, token, generation, setDetails);
      } catch (caught) {
        if (generation.current !== token) return;
        setError(caught instanceof ApiError ? caught.userMessage : "Could not load the queue.");
      } finally {
        if (generation.current === token) setLoading(false);
      }
    },
    [jobId, status],
  );

  useEffect(() => {
    history.current = [null];
    void load(null);
  }, [load]);

  useEffect(() => {
    onPageInfo?.({
      page: history.current.length,
      count: items.length,
      hasNext: Boolean(nextCursor),
    });
  }, [items.length, nextCursor, onPageInfo]);

  const goNext = useCallback(() => {
    if (!nextCursor || loading) return;
    history.current.push(nextCursor);
    void load(nextCursor);
  }, [load, loading, nextCursor]);

  const goPrevious = useCallback(() => {
    if (history.current.length < 2 || loading) return;
    history.current.pop();
    void load(history.current[history.current.length - 1]);
  }, [load, loading]);

  const open = useCallback(
    (cluster: ClusterSummary) => {
      router.push(`/review/${cluster.id}?job=${jobId}`);
    },
    [jobId, router],
  );

  // --- Keyboard navigation ---------------------------------------------------
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;

      // Never hijack typing.
      if (
        target &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
      ) {
        return;
      }

      // Never hijack activation of a focused control. A global Enter handler
      // that calls preventDefault() breaks every button and link on the page:
      // tabbing to "Next" and pressing Enter would open a cluster instead of
      // paging. Rows handle their own Enter via onKeyDown below.
      const interactive = target?.closest("button, a, [role='button']");
      if (interactive) return;

      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (items.length === 0) return;

      switch (event.key) {
        case "j":
        case "J":
        case "ArrowDown":
          event.preventDefault();
          setActive((index) => {
            const next = Math.min(index + 1, items.length - 1);
            rowRefs.current[next]?.focus();
            return next;
          });
          break;
        case "k":
        case "K":
        case "ArrowUp":
          event.preventDefault();
          setActive((index) => {
            const next = Math.max(index - 1, 0);
            rowRefs.current[next]?.focus();
            return next;
          });
          break;
        case "Enter":
          event.preventDefault();
          if (items[active]) open(items[active]);
          break;
        case "]":
          event.preventDefault();
          goNext();
          break;
        case "[":
          event.preventDefault();
          goPrevious();
          break;
        default:
          break;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [active, goNext, goPrevious, items, open]);

  useEffect(() => {
    rowRefs.current[active]?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const body = useMemo(
    () =>
      items.map((cluster, index) => {
        const isActive = index === active;
        const detail = details[cluster.id];

        return (
          <tr
            key={cluster.id}
            ref={(element) => {
              rowRefs.current[index] = element;
            }}
            // Rows are the primary control here, so they must be reachable by
            // Tab and operable by Enter/Space — not mouse-only. A roving
            // tabindex keeps the table a single tab stop instead of 25.
            tabIndex={isActive ? 0 : -1}
            aria-selected={isActive}
            onFocus={() => setActive(index)}
            onClick={() => open(cluster)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                open(cluster);
              }
            }}
            className={`cursor-pointer border-b border-hairline transition-colors duration-100 last:border-0 ${
              isActive ? "bg-surface-raised shadow-[inset_2px_0_0_var(--accent)]" : "hover:bg-surface-subtle"
            }`}
          >
            <td className="max-w-[22rem] px-3 py-2">
              {detail === undefined ? (
                <span
                  className="skeleton inline-block h-3 w-40 rounded-sm"
                  aria-label="Loading"
                />
              ) : (
                <>
                  <span className="block truncate text-xs text-foreground" title={detail?.name}>
                    {detail?.name || "--"}
                  </span>
                  <span className="block truncate text-[11px] text-muted">{detail?.place}</span>
                </>
              )}
            </td>
            <td className="px-3 py-2 font-mono text-[11px] text-muted">{cluster.member_count}</td>
            <td className="px-3 py-2 text-xs text-muted">
              {detail === undefined ? "" : (detail?.discrepancy ?? "--")}
            </td>
            <td className="px-3 py-2">
              <ConfidenceBadge value={cluster.confidence_score} />
            </td>
            <td className="px-3 py-2 text-right">
              <span className={`text-[11px] ${isActive ? "text-accent" : "text-faint"}`}>
                Open →
              </span>
            </td>
          </tr>
        );
      }),
    [active, details, items, open],
  );

  if (error) {
    return <Notice tone="err">{error}</Notice>;
  }

  if (!loading && items.length === 0 && history.current.length === 1) {
    return status === "PENDING_REVIEW" ? (
      <EmptyQueue jobId={jobId} clustersTotal={clustersTotal} />
    ) : (
      <p className="rounded-lg border border-dashed border-hairline-strong px-6 py-12 text-center text-sm text-muted">
        No duplicate groups in this list.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-hidden rounded-lg border border-hairline bg-surface">
        {/*
          role="grid" is what makes aria-selected on a row legal. On a plain
          <table> the rows are role="row" inside role="table", where
          aria-selected is not a supported attribute and assistive tech is
          entitled to ignore it.
        */}
        <table role="grid" className="w-full border-collapse text-xs">
          <caption className="sr-only">
            Duplicate groups. Use J and K to move between rows, Enter to open.
          </caption>
          <thead>
            <tr className="border-b border-hairline bg-surface-subtle text-left">
              {["school", "records", "differs in", "confidence", "action"].map(
                (heading, i) => (
                  <th
                    key={heading}
                    scope="col"
                    className={`px-3 py-2 font-normal ${i === 4 ? "text-right" : ""}`}
                  >
                    {i === 4 ? <span className="sr-only">{heading}</span> : <span className="label">{heading}</span>}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody className={loading ? "opacity-50" : ""}>{body}</tbody>
        </table>
      </div>

      <div className="flex items-center justify-between text-xs text-muted">
        <span className="flex items-center gap-1.5">
          <Kbd>J</Kbd>
          <Kbd>K</Kbd>
          move
          <span className="mx-1 text-faint">·</span>
          <Kbd>↵</Kbd>
          open
          <span className="mx-1 text-faint">·</span>
          <Kbd>[</Kbd>
          <Kbd>]</Kbd>
          page
        </span>

        <span className="flex items-center gap-2">
          <button
            type="button"
            onClick={goPrevious}
            disabled={history.current.length < 2 || loading}
            className={buttonClass("secondary", "sm")}
          >
            Previous
          </button>
          <button
            type="button"
            onClick={goNext}
            disabled={!nextCursor || loading}
            className={buttonClass("secondary", "sm")}
          >
            Next
          </button>
        </span>
      </div>
    </div>
  );
}

/**
 * Fill the "school" and "differs in" columns.
 *
 * KNOWN N+1. The queue endpoint returns `ClusterSummary`, which carries no
 * member data, so naming the school and working out which field disagrees
 * means one detail request per row. Bounded (a page is 25), capped at a few
 * in flight, and rendered progressively rather than blocking the table.
 *
 * `token` is the page generation this batch belongs to. Every commit checks
 * it first: without that, paging away mid-batch leaves ~19 requests still
 * running that write their results into the *next* page's state.
 */
async function loadRowDetails(
  clusters: ClusterSummary[],
  token: number,
  generation: { current: number },
  commit: (
    updater: (prev: Record<string, RowDetail | null>) => Record<string, RowDetail | null>,
  ) => void,
): Promise<void> {
  const queue = [...clusters];

  const worker = async (): Promise<void> => {
    for (;;) {
      if (generation.current !== token) return;
      const cluster = queue.shift();
      if (!cluster) return;
      try {
        const detail = await getCluster(cluster.id);
        if (generation.current !== token) return;
        const masterId = proposeMaster(detail.members, detail.master_record_id);
        const master = detail.members.find((m) => m.record_id === masterId) ?? detail.members[0];
        const row: RowDetail = {
          name: master ? canonicalValue(master, "name") : "",
          place: master
            ? [canonicalValue(master, "locality"), canonicalValue(master, "state")]
                .filter(Boolean)
                .join(", ")
            : "",
          discrepancy: topDiscrepancy(detail.members),
        };
        commit((prev) => ({ ...prev, [cluster.id]: row }));
      } catch {
        if (generation.current !== token) return;
        commit((prev) => ({ ...prev, [cluster.id]: null }));
      }
    }
  };

  await Promise.all(Array.from({ length: Math.min(DETAIL_CONCURRENCY, queue.length) }, worker));
}

/**
 * Zero state.
 *
 * An empty first page of PENDING_REVIEW does NOT by itself mean "everything
 * is resolved" — a job still running, or one where blocking found no
 * duplicates at all, also has nothing awaiting review. Claiming success in
 * those cases tells the steward the work is done when it has not started.
 * `clustersTotal` is what separates the two.
 */
function EmptyQueue({ jobId, clustersTotal }: { jobId: string; clustersTotal: number | null }) {
  const resolved = clustersTotal !== null && clustersTotal > 0;

  return (
    <div className="flex flex-col items-center gap-3 rounded-md border border-dashed border-hairline-strong bg-surface-subtle px-6 py-14 text-center">
      <div
        className={`flex h-9 w-9 animate-pop items-center justify-center rounded-full border ${
          resolved
            ? "border-accent-border bg-accent-dim text-accent"
            : "border-hairline-strong bg-surface-raised text-muted"
        }`}
      >
        <span className="text-sm">{resolved ? "✓" : "·"}</span>
      </div>

      <div>
        <p className="text-sm text-foreground">
          {resolved ? "Queue clear" : "Nothing awaiting review"}
        </p>
        <p className="mt-0.5 text-xs text-muted">
          {resolved
            ? `Every duplicate group in this upload has been decided or merged automatically (${clustersTotal.toLocaleString()} total).`
            : "No duplicate groups need a decision — none were found, or the upload is still processing."}
        </p>
      </div>

      {resolved && (
        <div className="mt-1 flex flex-wrap items-center justify-center gap-2">
          {/*
            Points at our own route handler, which attaches the bearer token
            server-side and pipes the CSV straight through. Linking at the
            API directly sends no Authorization header and 401s every time.
          */}
          <a
            href={exportUrl(jobId, "cleaned")}
            className={buttonClass("primary")}
          >
            Download cleaned CSV
          </a>
          <Link href={`/jobs/${jobId}`} className={buttonClass("secondary")}>
            View results
          </Link>
        </div>
      )}
    </div>
  );
}
