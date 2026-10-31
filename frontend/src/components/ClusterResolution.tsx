"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ConfidenceBadge, Kbd, MonoId, StatusDot, buttonClass, fieldClass } from "@/components/ui";
import { getCluster, nextPendingCluster, submitFeedback } from "@/lib/api/clusters";
import { ApiError } from "@/lib/api/client";
import type {
  ClusterDetail,
  ClusterMemberSummary,
  CorrectableField,
  FeedbackDecision,
} from "@/lib/api/types";
import { CANONICAL_FIELDS, canonicalValue, diffWords, isDiffedField } from "@/lib/diff";
import { formatScore, proposeMaster } from "@/lib/format";

/** This screen's field keys -> the backend's correctable field names. */
const CORRECTABLE: Record<string, CorrectableField> = {
  name: "name",
  address: "address",
  locality: "district",
  state: "state",
  pincode: "pincode",
};

type Corrections = Partial<Record<CorrectableField, string>>;

interface Props {
  clusterId: string;
  jobId: string | null;
}

export function ClusterResolution({ clusterId, jobId }: Props) {
  const router = useRouter();

  const [cluster, setCluster] = useState<ClusterDetail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [masterId, setMasterId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState<FeedbackDecision | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [nextId, setNextId] = useState<string | null>(null);
  // Values the steward typed over the kept record's own. Sent with a MERGE
  // and stored beside the record — the uploaded row itself is never edited.
  const [corrections, setCorrections] = useState<Corrections>({});
  // `text` alone is not enough to make a screen reader re-announce two
  // identical decisions in a row: React bails out of a setState call whose
  // value is Object.is-equal to the current one, so the DOM text node for
  // "Merged. Opening next cluster." never mutates on the second merge, and
  // aria-live only fires on mutation. `seq` forces a real content change —
  // see the render below — even when `text` repeats.
  const [announcement, setAnnouncement] = useState({ text: "", seq: 0 });

  useEffect(() => {
    let cancelled = false;
    setCluster(null);
    setLoadError(null);
    setActionError(null);
    setSubmitting(null);
    setCorrections({});
    // Otherwise the live region still asserts "Merged. Opening next
    // cluster." while a fresh, undecided cluster is already on screen.
    setAnnouncement({ text: "", seq: 0 });

    getCluster(clusterId)
      .then((detail) => {
        if (cancelled) return;
        setCluster(detail);
        setMasterId(proposeMaster(detail.members, detail.master_record_id));
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setLoadError(
            caught instanceof ApiError ? caught.userMessage : "Could not load the cluster.",
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, [clusterId]);

  // Resolve where the steward goes next *while they are still reading*, so a
  // verdict is a navigation rather than a round trip.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;

    // Reset before resolving, not after. `ClusterResolutionScreen` renders
    // this component with no `key`, so navigating A -> B reuses the same
    // instance — if B turns out to be the last pending cluster,
    // `nextPendingCluster` resolves to null and a stale `nextId` left over
    // from A would otherwise survive untouched. `advance()` guards against
    // that value too (see below), but clearing it here means the tray's
    // "Next cluster ready" hint is never shown for a cluster that has none.
    setNextId(null);

    nextPendingCluster(jobId, clusterId)
      .then((id) => {
        if (cancelled) return;
        setNextId(id);
        if (id) router.prefetch(`/review/${id}?job=${jobId}`);
      })
      .catch(() => {
        /* A missed prefetch costs a slower jump, nothing else. */
      });

    return () => {
      cancelled = true;
    };
  }, [clusterId, jobId, router]);

  const advance = useCallback(() => {
    // `nextId !== clusterId` is the load-bearing guard, not decoration.
    // Without it, a stale `nextId` equal to the cluster just decided sends
    // `router.push` to the URL already on screen. Next.js no-ops a
    // navigation to the current URL, so the component neither re-renders nor
    // unmounts — and `submitting` (cleared only by the `[clusterId]` effect
    // or by unmounting) stays stuck on "Merging…" forever, with both buttons
    // disabled, after a merge that actually succeeded.
    router.push(
      nextId && nextId !== clusterId && jobId
        ? `/review/${nextId}?job=${jobId}`
        : jobId
          ? `/review?job=${jobId}`
          : "/review",
    );
  }, [clusterId, jobId, nextId, router]);

  const decide = useCallback(
    async (decision: FeedbackDecision) => {
      // Guards the HOTKEY path. The buttons are disabled when the cluster is
      // already decided, but a keypress never touches them — without this,
      // pressing M on a RESOLVED cluster fires a request that 409s.
      if (!cluster || submitting) return;
      if (cluster.status === "RESOLVED" || cluster.status === "REJECTED") return;
      // Belt-and-braces: the render guard below never shows the Merge button
      // without a proposed master, but the hotkey listener is bound
      // unconditionally (all hooks run before any early return), so this is
      // what actually stops M from posting a MERGE with no survivor if the
      // two ever disagree.
      if (decision === "MERGE" && !masterId) return;
      setSubmitting(decision);
      setActionError(null);

      const changed = Object.fromEntries(
        Object.entries(corrections).filter(([, value]) => value !== undefined && value.trim()),
      ) as Corrections;

      try {
        await submitFeedback(cluster.id, {
          decision,
          // MERGE needs a survivor; REJECT must not carry one.
          master_record_id: decision === "MERGE" ? masterId : null,
          field_corrections:
            decision === "MERGE" && Object.keys(changed).length > 0 ? changed : null,
        });
        setAnnouncement((prev) => ({
          text:
            decision === "MERGE" ? "Merged. Opening next cluster." : "Rejected. Opening next cluster.",
          seq: prev.seq + 1,
        }));
        advance();
      } catch (caught) {
        setSubmitting(null);
        setActionError(
          caught instanceof ApiError && caught.status === 409
            ? "Already reviewed by someone else — their decision stands."
            : caught instanceof ApiError
              ? caught.userMessage
              : "Could not record the decision.",
        );
      }
    },
    [advance, cluster, corrections, masterId, submitting],
  );

  // --- Hotkeys ---------------------------------------------------------------
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
      ) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      if (event.key === "m" || event.key === "M") {
        event.preventDefault();
        void decide("MERGE");
      } else if (event.key === "r" || event.key === "R") {
        event.preventDefault();
        void decide("REJECT");
      } else if (event.key === "Escape") {
        event.preventDefault();
        router.push(jobId ? `/review?job=${jobId}` : "/review");
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [decide, jobId, router]);

  const master = useMemo(
    () => cluster?.members.find((member) => member.record_id === masterId) ?? null,
    [cluster, masterId],
  );
  const variants = useMemo(
    () => cluster?.members.filter((member) => member.record_id !== masterId) ?? [],
    [cluster, masterId],
  );

  if (loadError) {
    return (
      <div className="flex flex-col gap-3 py-8">
        <p role="alert" className="text-sm text-err">
          {loadError}
        </p>
        <Link
          href={jobId ? `/review?job=${jobId}` : "/review"}
          className="text-xs text-muted underline-offset-4 hover:underline"
        >
          ← Back to queue
        </Link>
      </div>
    );
  }

  if (!cluster) {
    return <p className="py-8 text-sm text-muted">Loading cluster…</p>;
  }

  if (!master) {
    // Distinct from "still loading": the cluster fetched successfully but
    // has no member to propose as a survivor (an empty `members` array).
    // Collapsing this into the loading branch above leaves the steward
    // staring at a spinner-equivalent that will never resolve, with no link
    // back out — this one has one.
    return (
      <div className="flex flex-col gap-3 py-8">
        <p className="text-sm text-muted">
          This cluster has no records left to review.
        </p>
        <Link
          href={jobId ? `/review?job=${jobId}` : "/review"}
          className="text-xs text-muted underline-offset-4 hover:underline"
        >
          ← Back to queue
        </Link>
      </div>
    );
  }

  const decided = cluster.status === "RESOLVED" || cluster.status === "REJECTED";
  const canMerge = variants.length > 0;

  return (
    // Bottom padding clears the sticky tray so the last card is never trapped
    // behind it.
    <section className="flex flex-col gap-4 pb-24">
      {/*
        Decisions are made by hotkey and immediately navigate, so a sighted
        user sees the page change and a screen-reader user previously got
        nothing at all — the destination was indistinguishable by ear from a
        failed keypress. This announces the outcome before the route changes.
      */}
      <p aria-live="polite" role="status" className="sr-only">
        {announcement.text}
        {/* Zero-width space, present on odd sequence numbers only. Invisible
            and silent, but it changes the text node's actual content, which
            is what makes a screen reader re-announce two consecutive
            identical decisions instead of the second one going unheard. */}
        {announcement.seq % 2 === 1 ? "​" : ""}
      </p>

      <header className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <h1 className="text-xl font-semibold text-foreground">Same school?</h1>
        <MonoId value={cluster.id} length={12} />
        <ConfidenceBadge value={cluster.confidence_score} />
        <span className="text-xs text-muted">
          {cluster.member_count} records · {cluster.matcher_version ?? "unscored"}
        </span>
        <Link
          href={jobId ? `/review?job=${jobId}` : "/review"}
          className="ml-auto flex items-center gap-1.5 text-xs text-muted underline-offset-4 hover:underline"
        >
          <Kbd>Esc</Kbd> queue
        </Link>
      </header>

      {decided && (
        <p className="rounded-md border border-hairline-strong bg-surface-subtle px-3 py-2 text-xs text-foreground"><span className="mr-2 text-muted">!</span>
          Already {cluster.status.toLowerCase()} — the server will refuse a second verdict.
        </p>
      )}

      <p className="text-xs text-muted">
        Are these records the same school? The left side is the record that will be kept if you
        merge. You can correct its details before merging; your corrections are saved alongside
        it and the original upload is never changed.
      </p>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <MasterPane
          cluster={cluster}
          master={master}
          corrections={corrections}
          editable={!decided && submitting === null}
          onCorrect={(field, value) =>
            setCorrections((prev) => ({ ...prev, [field]: value }))
          }
        />
        <div className="flex flex-col gap-2">
          <DiffLegend />
          <VariantPane
            master={master}
            variants={variants}
            onPromote={setMasterId}
            disabled={submitting !== null || decided}
          />
        </div>
      </div>

      <ActionTray
        decided={decided}
        canMerge={canMerge}
        submitting={submitting}
        error={actionError}
        nextReady={Boolean(nextId)}
        masterLabel={
          corrections.name?.trim() ||
          canonicalValue(master, "name") ||
          master.record_id.slice(0, 8)
        }
        corrected={Object.values(corrections).some((value) => value?.trim())}
        onDecide={decide}
        onSkip={advance}
      />
    </section>
  );
}

/** Left pane: the record that survives a merge, correctable before merging. */
function MasterPane({
  cluster,
  master,
  corrections,
  editable,
  onCorrect,
}: {
  cluster: ClusterDetail;
  master: ClusterMemberSummary;
  corrections: Corrections;
  editable: boolean;
  onCorrect: (field: CorrectableField, value: string) => void;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const golden = cluster.golden_fields;

  return (
    <div className="overflow-hidden rounded-lg border border-accent-border bg-surface">
      <div className="flex items-center gap-2 border-b border-hairline px-3 py-2">
        <StatusDot tone="ok" />
        <span className="label text-accent">
          Record to keep
        </span>
        <span
          className="ml-auto text-[11px] text-muted"
          title="How strongly this record matches the others"
        >
          {formatScore(master.pairwise_score)}
        </span>
      </div>

      <dl className="divide-y divide-hairline">
        {CANONICAL_FIELDS.map((field) => {
          const target = CORRECTABLE[field.key];
          const original = canonicalValue(master, field.key);
          // "" means the edit was put back to the original: no correction.
          const corrected = target ? corrections[target] || undefined : undefined;
          const saved = target ? golden?.[target] : undefined;
          const shown = corrected ?? saved ?? original;
          const changed = shown !== original;

          return (
            <div key={field.key} className="grid grid-cols-[88px_minmax(0,1fr)] gap-2 px-3 py-2">
              <dt className="label">
                {field.label}
              </dt>
              <dd className="flex min-w-0 items-start gap-2 text-xs">
                {editing === field.key && target ? (
                  <input
                    autoFocus
                    defaultValue={shown}
                    aria-label={`Correct ${field.label}`}
                    onBlur={(event) => {
                      const value = event.currentTarget.value.trim();
                      onCorrect(target, value === original ? "" : value);
                      setEditing(null);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") event.currentTarget.blur();
                      if (event.key === "Escape") setEditing(null);
                    }}
                    className={`${fieldClass} px-1.5 py-1 text-xs`}
                  />
                ) : (
                  <>
                    <span className="min-w-0 flex-1 break-words">
                      {shown || <span className="text-muted">—</span>}
                      {changed && (
                        <span className="mt-0.5 block text-[10px] text-muted">
                          was: {original || "—"} ·{" "}
                          {corrected !== undefined ? "your correction" : "corrected by a reviewer"}
                        </span>
                      )}
                    </span>
                    {editable && target && (
                      <button
                        type="button"
                        onClick={() => setEditing(field.key)}
                        className="shrink-0 rounded px-1 text-[11px] text-faint transition-colors duration-100 hover:bg-surface-raised hover:text-foreground"
                        aria-label={`Edit ${field.label}`}
                      >
                        Edit
                      </button>
                    )}
                  </>
                )}
              </dd>
            </div>
          );
        })}
        <div className="grid grid-cols-[88px_minmax(0,1fr)] gap-2 px-3 py-2">
          <dt className="label">
            Record
          </dt>
          <dd>
            <MonoId value={master.record_id} length={18} />
          </dd>
        </div>
      </dl>

      <GoldenStatus
        status={cluster.status}
        curated={Boolean(
          cluster.golden_fields && Object.keys(cluster.golden_fields).length > 0,
        )}
      />
    </div>
  );
}

/**
 * Where this cluster's surviving values come from. `curated` means the
 * golden record carries explicit values (a steward's corrections, or an
 * address verified before the zero-cost migration) rather than the master
 * row's own cleaned fields.
 */
function GoldenStatus({ status, curated }: { status: string; curated: boolean }) {
  const label = curated
    ? "Golden record has curated values"
    : status === "AUTO_MERGED"
      ? "Auto-merged above threshold — values from the most complete record"
      : "Values from the master record";

  return (
    <p className="border-t border-hairline bg-surface-subtle px-3 py-2 text-[10px] text-muted">
      {label}
    </p>
  );
}

/**
 * What the highlighting means. The single most-asked question about this
 * screen was whether red marks "the wrong value" — it does not.
 */
function DiffLegend() {
  return (
    <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted">
      <span>
        <span className="diff-add">green</span> = what this record says
      </span>
      <span>
        <span className="diff-remove line-through decoration-1">red</span> = what the record to
        keep says instead
      </span>
      <span>Neither means “wrong” — they just differ.</span>
    </p>
  );
}

/** Right pane: each conflicting record, diffed against the master. */
function VariantPane({
  master,
  variants,
  onPromote,
  disabled,
}: {
  master: ClusterMemberSummary;
  variants: ClusterMemberSummary[];
  onPromote: (recordId: string) => void;
  disabled: boolean;
}) {
  if (variants.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-md border border-dashed border-hairline-strong px-4 py-10 text-xs text-muted">
        No other records in this cluster.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {variants.map((variant) => (
        <article key={variant.record_id} className="overflow-hidden rounded-lg border border-hairline bg-surface">
          <div className="flex items-center gap-2 border-b border-hairline px-3 py-2">
            <span className="label">
              Possible duplicate
            </span>
            <MonoId value={variant.record_id} />
            <span className="ml-auto flex items-center gap-2">
              <span className="text-[11px] text-muted">
                {formatScore(variant.pairwise_score)}
              </span>
              <button
                type="button"
                onClick={() => onPromote(variant.record_id)}
                disabled={disabled}
                className={buttonClass("ghost", "sm")}
              >
                Keep this one instead
              </button>
            </span>
          </div>

          <dl className="divide-y divide-hairline">
            {CANONICAL_FIELDS.map((field) => {
              const masterValue = canonicalValue(master, field.key);
              const variantValue = canonicalValue(variant, field.key);
              const identical = masterValue === variantValue;

              return (
                <div
                  key={field.key}
                  className="grid grid-cols-[88px_minmax(0,1fr)] gap-2 px-3 py-2"
                >
                  <dt className="label">
                    {field.label}
                  </dt>
                  <dd className="break-words text-xs">
                    {!variantValue && !masterValue ? (
                      <span className="text-muted">—</span>
                    ) : identical || !isDiffedField(field.key) ? (
                      <span className={identical ? "text-muted" : undefined}>
                        {variantValue || <span className="text-muted">—</span>}
                      </span>
                    ) : (
                      <DiffLine master={masterValue} variant={variantValue} />
                    )}
                  </dd>
                </div>
              );
            })}
          </dl>
        </article>
      ))}
    </div>
  );
}

/**
 * One field, rendered as a word-level diff.
 *
 * Green is what this variant says, red is what the master says instead — so a
 * steward reads the row as "this record claims X where the master claims Y"
 * without holding both strings in their head.
 */
function DiffLine({ master, variant }: { master: string; variant: string }) {
  const segments = diffWords(master, variant);

  return (
    <>
      {/*
        The visual diff interleaves both records' words — "Main Road Rd Block 7"
        reads correctly to an eye that can see one half struck through and the
        other tinted, and as nonsense to a screen reader, which flattens the
        markup and announces the merged string. So the interleaved version is
        hidden from assistive tech and a plain pair of values is provided
        instead. Colour and strikethrough are also not the only signal: the
        sr-only text states which value belongs to which record.
      */}
      <span aria-hidden="true">
        {segments.map((segment, index) => (
          <span
            key={index}
            className={
              segment.kind === "removed"
                ? "diff-remove line-through decoration-1"
                : segment.kind === "added"
                  ? "diff-add"
                  : undefined
            }
          >
            {segment.text}
          </span>
        ))}
      </span>
      <span className="sr-only">
        {variant || "empty"}, differs from master value {master || "empty"}
      </span>
    </>
  );
}

/** Sticky decision tray. Hotkeys are shown, not hidden in a help menu. */
function ActionTray({
  decided,
  canMerge,
  submitting,
  error,
  nextReady,
  masterLabel,
  corrected,
  onDecide,
  onSkip,
}: {
  decided: boolean;
  canMerge: boolean;
  submitting: FeedbackDecision | null;
  error: string | null;
  nextReady: boolean;
  masterLabel: string;
  corrected: boolean;
  onDecide: (decision: FeedbackDecision) => void;
  onSkip: () => void;
}) {
  const busy = submitting !== null;

  return (
    <div className="fixed inset-x-0 bottom-0 z-10 border-t border-hairline bg-surface-subtle/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-3 px-6 py-3">
        <button
          type="button"
          onClick={() => onDecide("MERGE")}
          disabled={busy || decided || !canMerge}
          title={canMerge ? undefined : "Nothing to merge — only one record in this cluster"}
          className={buttonClass("primary")}
        >
          {submitting === "MERGE" ? "Merging…" : "Same school, merge"}
          <Kbd>M</Kbd>
        </button>

        <button
          type="button"
          onClick={() => onDecide("REJECT")}
          disabled={busy || decided}
          className={buttonClass("secondary")}
        >
          {submitting === "REJECT" ? "Saving…" : "Different schools"}
          <Kbd>R</Kbd>
        </button>

        <p className="text-[11px] text-muted">
          {canMerge ? (
            <>
              Merging keeps <span className="text-accent">{masterLabel}</span>
              {corrected ? " with your corrections" : ""} and marks the others as its
              duplicates. &ldquo;Different schools&rdquo; keeps every record separate.
            </>
          ) : (
            "Only one record in this cluster — nothing to merge it with."
          )}
        </p>

        {error ? (
          <span
            role="alert"
            className="flex items-center gap-2 text-[11px] text-err"
          >
            {error}
            <button type="button" onClick={onSkip} className="underline underline-offset-2">
              Skip
            </button>
          </span>
        ) : decided ? (
          // Previously the only forward affordance here lived inside the
          // error branch, so a cluster that arrived already decided by
          // someone else — no error, just nothing left to do — showed "Next
          // cluster ready" as inert text with no control anywhere near it
          // that actually went there.
          <button
            type="button"
            onClick={onSkip}
            className={`${buttonClass("secondary")} ml-auto`}
          >
            {nextReady ? "Next cluster →" : "Back to queue"}
          </button>
        ) : (
          nextReady && <span className="ml-auto text-[11px] text-muted">Next cluster ready</span>
        )}
      </div>
    </div>
  );
}
