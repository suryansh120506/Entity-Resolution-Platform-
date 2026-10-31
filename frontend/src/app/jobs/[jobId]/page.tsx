import { JobResults } from "@/components/JobResults";

/**
 * Results for one job: where every uploaded row ended up, why rows were set
 * aside, the rows themselves, and the cleaned download.
 */
export default async function JobResultsPage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = await params;
  return <JobResults jobId={jobId} />;
}
