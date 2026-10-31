"use client";

import { useCallback, useEffect, useState } from "react";

import { JobStatusPanel } from "@/components/JobStatusPanel";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Notice, PageHeader, SectionTitle, Skeleton } from "@/components/ui";
import { ApiError } from "@/lib/api/client";
import { listJobs } from "@/lib/api/jobs";
import type { JobAccepted } from "@/lib/api/types";

/**
 * Upload and ingestion status.
 *
 * Job ids come from `GET /v1/jobs` on mount, not from in-tab state alone —
 * that used to be the only source, which meant a job "disappeared" the
 * moment a steward followed the review-queue link and navigated back, or
 * reloaded the page. A freshly-accepted upload is still pushed to the front
 * of the list immediately (optimistic — see `onAccepted`) so it shows up
 * without waiting on a refetch; the list endpoint is what makes it survive
 * after that.
 */
export default function DashboardPage() {
  const [jobIds, setJobIds] = useState<string[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    listJobs()
      .then((page) => {
        if (cancelled) return;
        setJobIds(page.items.map((job) => job.id));
      })
      .catch((caught: unknown) => {
        if (cancelled) return;
        setLoadError(caught instanceof ApiError ? caught.userMessage : "Could not load jobs.");
      })
      .finally(() => {
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const onAccepted = (job: JobAccepted) => {
    setJobIds((prev) => [job.job_id, ...prev.filter((id) => id !== job.job_id)]);
  };

  // Stable identity: JobStatusPanel's polling effect depends on it.
  const onDeleted = useCallback((jobId: string) => {
    setJobIds((prev) => prev.filter((id) => id !== jobId));
  }, []);

  return (
    <div className="stagger flex flex-col gap-10">
      <PageHeader
        title="Clean a list"
        description="Upload a CSV of schools. Duplicates are grouped, uncertain groups are checked against the web, and you get one clean row per school."
      />

      <UploadDropzone onAccepted={onAccepted} />

      <section className="flex flex-col gap-4">
        <SectionTitle count={loaded ? jobIds.length : undefined}>Jobs</SectionTitle>

        {loadError && <Notice tone="err">{loadError}</Notice>}

        {!loaded ? (
          <Skeleton className="h-48" />
        ) : jobIds.length === 0 ? (
          <p className="rounded-lg border border-dashed border-hairline px-6 py-12 text-center text-sm text-muted">
            No jobs yet. Uploads appear here with live progress.
          </p>
        ) : (
          <div className="stagger flex flex-col gap-4">
            {jobIds.map((id) => (
              <JobStatusPanel key={id} jobId={id} onDeleted={onDeleted} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
