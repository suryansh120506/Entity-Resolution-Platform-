import { apiFetch } from "./client";
import type {
  ClusterDetail,
  ClusterPage,
  ClusterStatus,
  FeedbackRequest,
  FeedbackResponse,
} from "./types";

export interface ListClustersParams {
  jobId: string;
  /**
   * Cluster-level status. Note this is NOT `AWAITING_REVIEW` — that is a
   * *job* status (FIG 2.2). A job sits at AWAITING_REVIEW precisely while
   * its clusters sit at PENDING_REVIEW (FIG 2.5), and the two enums are
   * easy to conflate because they describe the same moment from different
   * sides. Filtering clusters by AWAITING_REVIEW returns nothing, silently.
   */
  status?: ClusterStatus;
  limit?: number;
  /** Opaque. Pass `next_cursor` back exactly as received. */
  cursor?: string | null;
}

export function listClusters(
  { jobId, status = "PENDING_REVIEW", limit = 25, cursor }: ListClustersParams,
  init?: RequestInit,
): Promise<ClusterPage> {
  const query = new URLSearchParams({ status, limit: String(limit) });
  if (cursor) query.set("cursor", cursor);
  return apiFetch<ClusterPage>(`/jobs/${jobId}/clusters?${query}`, init);
}

export function getCluster(clusterId: string, init?: RequestInit): Promise<ClusterDetail> {
  return apiFetch<ClusterDetail>(`/clusters/${clusterId}`, init);
}

export function submitFeedback(
  clusterId: string,
  body: FeedbackRequest,
): Promise<FeedbackResponse> {
  return apiFetch<FeedbackResponse>(`/clusters/${clusterId}/feedback`, {
    method: "POST",
    body,
  });
}

/**
 * The next cluster a steward should see, or null if the queue is empty.
 *
 * Fetched with `limit: 1` and used for prefetching, so the jump after a
 * verdict is instant rather than a round trip the user waits through.
 */
export async function nextPendingCluster(
  jobId: string,
  excludeId?: string,
): Promise<string | null> {
  // Two, not one: the cluster just decided may still be in the page the
  // backend computed a moment ago, and skipping straight back to it would
  // loop the steward on the same record.
  const page = await listClusters({ jobId, limit: 2 });
  const next = page.items.find((cluster) => cluster.id !== excludeId);
  return next?.id ?? null;
}
