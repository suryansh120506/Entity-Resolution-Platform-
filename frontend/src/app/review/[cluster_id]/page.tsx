import { Suspense } from "react";

import { ClusterResolutionScreen } from "@/components/ClusterResolutionScreen";

/**
 * Cluster resolution — FIG 2.10 / FR-08.
 *
 * Only MERGE and REJECT are offered. SPLIT is the third verdict the backend
 * accepts and it needs a different interaction — the steward has to say
 * *which* records leave the cluster, and the API enforces that
 * `excluded_record_ids` is non-empty and SPLIT-only. Shipping a button that
 * cannot express that would produce 422s, so it is left out rather than
 * half-built.
 */
export default function ClusterDetailPage() {
  return (
    <Suspense fallback={<p className="text-xs text-muted">Loading cluster…</p>}>
      <ClusterResolutionScreen />
    </Suspense>
  );
}
