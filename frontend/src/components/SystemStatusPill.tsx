"use client";

import { useEffect, useState } from "react";

import { getSystemStatus } from "@/lib/api/jobs";
import type { SystemStatus } from "@/lib/api/types";

import { StatusDot } from "./ui";

const POLL_MS = 15_000;

/**
 * Header indicator: is everything a job needs actually running?
 *
 * The most common "it's broken" report this app has had was an upload that
 * sat at PENDING because no worker was running: the API accepted the file,
 * nothing ever picked it up, and nothing on screen said why. This says so
 * before anyone uploads. Written as ZUI's status line: "SYS : ready · 1 worker".
 */
export function SystemStatusPill() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const next = await getSystemStatus();
        if (cancelled) return;
        setStatus(next);
        setUnreachable(false);
      } catch {
        if (!cancelled) setUnreachable(true);
      } finally {
        if (!cancelled) timer = setTimeout(tick, POLL_MS);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  const problem = unreachable
    ? "backend offline"
    : status?.checks.postgres !== undefined && status.checks.postgres !== "ok"
      ? "database offline"
      : status && status.workers === 0
        ? "no worker"
        : null;

  const detail = unreachable
    ? "The API on port 8000 is not responding. Start it, then reload."
    : problem === "no worker"
      ? "Uploads will wait at 'pending' until a pipeline worker is started."
      : problem
        ? "Uploads and review will fail until this is fixed."
        : status
          ? `All systems running · ${status.workers} worker${status.workers === 1 ? "" : "s"}${
              status.enrichment_llm ? " · enrichment on" : ""
            }`
          : "Checking…";

  return (
    <span
      role="status"
      title={detail}
      className={[
        "inline-flex h-7 items-center gap-2 rounded-full border px-2.5 text-xs leading-none transition-colors duration-300",
        problem ? "border-err-border bg-err-dim text-err" : "border-hairline text-muted",
      ].join(" ")}
    >
      <StatusDot tone={problem ? "err" : status ? "active" : "idle"} />
      <span className={problem ? "" : "text-foreground"}>
        {problem ?? (status ? "Operational" : "Checking")}
      </span>
      {!problem && status && (
        <span className="font-mono text-[11px] text-faint">
          {status.workers}w
        </span>
      )}
      <span className="sr-only">. {detail}</span>
    </span>
  );
}
