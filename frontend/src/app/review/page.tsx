import { Suspense } from "react";

import { ReviewQueueScreen } from "@/components/ReviewQueueScreen";

/**
 * Review queue — FIG 2.9.
 *
 * `?job=<uuid>` opens one upload's queue; without it the screen lists the
 * uploads that have something to review (`GET /v1/jobs`). The id stays in
 * the URL so a queue link is shareable.
 *
 * `useSearchParams` opts a route into client rendering, so the screen is
 * wrapped in Suspense — without it the build fails rather than degrading.
 */
export default function ReviewPage() {
  return (
    <Suspense fallback={<p className="text-xs text-muted">Loading…</p>}>
      <ReviewQueueScreen />
    </Suspense>
  );
}
