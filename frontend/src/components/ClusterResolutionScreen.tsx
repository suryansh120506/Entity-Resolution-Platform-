"use client";

import { useParams, useSearchParams } from "next/navigation";

import { ClusterResolution } from "@/components/ClusterResolution";

/**
 * Reads the route params on the client.
 *
 * `params` is a Promise in a Next 15 server component, and this screen needs
 * `?job=` from the query string as well — which the queue attaches so that
 * "next cluster" has a queue to draw from. Both come from hooks here rather
 * than being threaded down as props.
 */
export function ClusterResolutionScreen() {
  const params = useParams<{ cluster_id: string }>();
  const search = useSearchParams();

  return (
    <ClusterResolution clusterId={params.cluster_id} jobId={search.get("job")} />
  );
}
