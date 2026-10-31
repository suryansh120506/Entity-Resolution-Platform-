import type { ClusterMemberSummary } from "@/lib/api/types";

/**
 * Parse a score the API sends as a string.
 *
 * `clusters.confidence_score` and `cluster_members.pairwise_score` are
 * `NUMERIC(4,3)`, and Pydantic serialises Decimal as a JSON *string* —
 * `"0.842"`, not `0.842`. Sorting or comparing them as-is silently gives
 * lexicographic order, where "0.9" sorts above "0.85". Confirmed against the
 * live API rather than assumed.
 */
export function toScore(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined) return null;
  const parsed = typeof value === "number" ? value : Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatScore(value: string | number | null | undefined): string {
  const score = toScore(value);
  return score === null ? "—" : score.toFixed(3);
}

/**
 * Tailwind classes for a confidence band. Mirrors the tenant thresholds:
 * accent at auto-merge, plain in the band the automated check settles, dim
 * below the floor. (Low confidence is not an error, so it is not red.)
 */
export function scoreTone(value: string | number | null | undefined): string {
  const score = toScore(value);
  if (score === null) return "text-faint";
  if (score >= 0.95) return "text-accent";
  if (score >= 0.8) return "text-foreground";
  return "text-muted";
}

export function formatTimestamp(iso: string): string {
  // Fixed locale so server and client render identically — `toLocaleString()`
  // with the ambient locale is a hydration mismatch waiting to happen.
  return new Date(iso).toISOString().replace("T", " ").slice(0, 19) + "Z";
}

const DATE_TIME = new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "short" });

/**
 * "26 Sept 2026, 11:16 am" in the viewer's own timezone. Only for values
 * rendered after a client-side fetch — never in server-rendered markup,
 * where the server's timezone would disagree with the browser's.
 */
export function formatDateTime(iso: string): string {
  return DATE_TIME.format(new Date(iso));
}

/** "1h 5m", "4m 12s", "9s". */
export function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

/** Fields shown first, in this order. Everything else follows alphabetically. */
const FIELD_PRIORITY = ["name", "address", "district", "state", "pincode"];

/** The union of every member's `raw_row` keys, priority fields first. */
export function orderedFields(members: ClusterMemberSummary[]): string[] {
  const keys = new Set<string>();
  for (const member of members) {
    for (const key of Object.keys(member.raw_row ?? {})) keys.add(key);
  }

  const priority = FIELD_PRIORITY.filter((field) => keys.has(field));
  const rest = [...keys].filter((key) => !priority.includes(key)).sort();
  return [...priority, ...rest];
}

export function fieldValue(member: ClusterMemberSummary, field: string): string {
  const value = member.raw_row?.[field];
  if (value === null || value === undefined) return "";
  return String(value);
}

/**
 * Which record should survive a merge.
 *
 * The backend sets `master_record_id` only once a cluster is resolved, so on
 * a PENDING_REVIEW cluster every member comes back with `is_master: false`
 * and there is nothing to "approve" — but `MERGE` *requires* a
 * master_record_id, so one has to be proposed.
 *
 * The rule mirrors `select_master_record` in the worker: strongest evidence
 * first, then the most complete row, then record id. The last tiebreak is
 * not cosmetic — without it the proposal could differ between two renders of
 * the same cluster, and a steward clicking Merge would be approving
 * something they had not necessarily looked at.
 */
export function proposeMaster(
  members: ClusterMemberSummary[],
  backendMasterId: string | null,
): string | null {
  if (backendMasterId) return backendMasterId;
  if (members.length === 0) return null;

  const ranked = [...members].sort((left, right) => {
    const byScore = (toScore(right.pairwise_score) ?? -1) - (toScore(left.pairwise_score) ?? -1);
    if (byScore !== 0) return byScore;

    const byCompleteness = completeness(right) - completeness(left);
    if (byCompleteness !== 0) return byCompleteness;

    return left.record_id.localeCompare(right.record_id);
  });

  return ranked[0].record_id;
}

function completeness(member: ClusterMemberSummary): number {
  return Object.values(member.raw_row ?? {}).filter(
    (value) => value !== null && value !== undefined && String(value).trim() !== "",
  ).length;
}
