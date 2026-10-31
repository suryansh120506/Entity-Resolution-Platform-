"use client";

import { useEffect, useId, useState } from "react";

import { generateDomain, saveDomain } from "@/lib/api/ontologies";
import { getAiSettings } from "@/lib/api/settings";
import type {
  AiSettings,
  DomainDraftResponse,
  OntologyDraft,
  OntologyOption,
} from "@/lib/api/types";
import { samplesForAi } from "@/lib/csv-mapping";
import { messageOf } from "@/lib/errors";

import { Notice, Tag, buttonClass, fieldClass } from "../ui";

/**
 * "Describe your dataset" -> an AI-drafted data domain -> the user's review
 * -> saved. Nothing the AI proposes is used until the user saves it, and the
 * server re-checks the draft on save.
 *
 * Every rule is shown and can be removed, because each one changes how
 * records match: an abbreviation rewrites names, an identifier format decides
 * which values count as the same entity. The server's notes (what it removed
 * from the AI's proposal, and why) are shown as-is.
 */
export function CustomDomainBuilder({
  headers,
  rows,
  onSaved,
  onCancel,
}: {
  headers: string[];
  rows: Record<string, string>[];
  onSaved: (option: OntologyOption) => void;
  onCancel: () => void;
}) {
  const [description, setDescription] = useState("");
  const [ai, setAi] = useState<AiSettings | null>(null);
  const [result, setResult] = useState<DomainDraftResponse | null>(null);
  const [draft, setDraft] = useState<OntologyDraft | null>(null);
  const [busy, setBusy] = useState<"generating" | "saving" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const textareaId = useId();

  useEffect(() => {
    getAiSettings().then(setAi).catch(() => setAi(null));
  }, []);

  const tier =
    ai?.mode === "premium"
      ? `Premium AI (${ai.providers.find((p) => p.id === ai.provider)?.label}, ${ai.model})`
      : ai?.standard_model
        ? `Standard AI (${ai.standard_model})`
        : "the configured AI";

  async function generate() {
    setBusy("generating");
    setError(null);
    try {
      const response = await generateDomain(description.trim(), headers, samplesForAi(rows));
      setResult(response);
      setDraft(response.draft);
    } catch (caught) {
      setError(messageOf(caught));
    } finally {
      setBusy(null);
    }
  }

  async function save() {
    if (!draft) return;
    setBusy("saving");
    setError(null);
    try {
      onSaved(await saveDomain(draft, description.trim(), samplesForAi(rows)));
    } catch (caught) {
      setError(messageOf(caught));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="flex animate-rise flex-col gap-4 rounded-lg border border-hairline bg-surface-subtle p-4">
      {!draft ? (
        <>
          <div className="flex flex-col gap-1.5">
            <label htmlFor={textareaId} className="label">
              Describe your dataset
            </label>
            <textarea
              id={textareaId}
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              rows={3}
              maxLength={1000}
              placeholder="Car dealerships in India, one row per showroom, with GST numbers."
              className={`${fieldClass} resize-none py-2 leading-relaxed`}
            />
            <p className="text-xs text-muted">
              Your description, the column headers and up to 15 sample rows are sent to {tier} to
              draft the rules. Nothing is used until you review and save it.
            </p>
          </div>
          {error && <ErrorLine>{error}</ErrorLine>}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => void generate()}
              disabled={description.trim().length < 8 || busy !== null}
              className={buttonClass("primary")}
            >
              {busy === "generating" ? "Drafting…" : "Draft rules"}
            </button>
            <button type="button" onClick={onCancel} className={buttonClass("ghost")}>
              Cancel
            </button>
            {busy === "generating" && (
              <span className="text-xs text-muted">
                Usually under 20 seconds; up to a minute when the free tier is busy.
              </span>
            )}
          </div>
        </>
      ) : (
        <DraftReview
          draft={draft}
          notes={result?.notes ?? []}
          draftedBy={result?.drafted_by ?? ""}
          onChange={setDraft}
        >
          {error && <ErrorLine>{error}</ErrorLine>}
          <div className="flex items-center gap-2 border-t border-hairline pt-4">
            <button
              type="button"
              onClick={() => void save()}
              disabled={busy !== null || !draft.label.trim()}
              className={buttonClass("primary")}
            >
              {busy === "saving" ? "Saving…" : "Save domain"}
            </button>
            <button
              type="button"
              onClick={() => {
                setDraft(null);
                setResult(null);
              }}
              className={buttonClass("ghost")}
            >
              Start over
            </button>
          </div>
        </DraftReview>
      )}
    </div>
  );
}

function DraftReview({
  draft,
  notes,
  draftedBy,
  onChange,
  children,
}: {
  draft: OntologyDraft;
  notes: string[];
  draftedBy: string;
  onChange: (draft: OntologyDraft) => void;
  children: React.ReactNode;
}) {
  const abbreviations = Object.entries(draft.abbreviations);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <input
          value={draft.label}
          onChange={(event) => onChange({ ...draft, label: event.target.value.slice(0, 60) })}
          aria-label="Domain name"
          className="-mx-1 rounded-sm border border-transparent bg-transparent px-1 text-base text-foreground hover:border-hairline focus:border-accent-border focus:bg-surface focus:outline-none"
        />
        {draft.description && <p className="text-sm text-muted">{draft.description}</p>}
        {draftedBy && <p className="font-mono text-xs text-faint">Drafted by {draftedBy}</p>}
      </div>

      <Section
        title="Abbreviations"
        caption="Expanded in names before matching and in the standardised name."
        empty="None proposed."
        count={abbreviations.length}
      >
        {abbreviations.map(([short, long]) => (
          <Chip
            key={short}
            onRemove={() => {
              const next = { ...draft.abbreviations };
              delete next[short];
              onChange({ ...draft, abbreviations: next });
            }}
          >
            <span className="text-accent">{short}</span>
            <span className="text-muted"> → </span>
            {long}
          </Chip>
        ))}
      </Section>

      <Section
        title="Kind-of words"
        caption="Say what kind of entity it is, not which one; matching gives them less weight."
        empty="None proposed."
        count={draft.generic_name_tokens.length}
      >
        {draft.generic_name_tokens.map((token) => (
          <Chip
            key={token}
            onRemove={() =>
              onChange({
                ...draft,
                generic_name_tokens: draft.generic_name_tokens.filter((t) => t !== token),
              })
            }
          >
            {token}
          </Chip>
        ))}
      </Section>

      <Section
        title="Identifiers"
        caption="Recognise one entity across uploads. A generated identifier never merges records on its own."
        empty="None: identity will come from name and location."
        count={draft.natural_keys.length}
      >
        {draft.natural_keys.map((key) => (
          <Chip
            key={key.key}
            onRemove={() =>
              onChange({
                ...draft,
                natural_keys: draft.natural_keys.filter((k) => k.key !== key.key),
              })
            }
          >
            {key.label || key.key}
            <span className="ml-1.5 text-[11px] text-muted">{key.pattern}</span>
          </Chip>
        ))}
      </Section>

      <Section
        title="Find on the web"
        caption="Filled in for every merged record, only from text found word for word in a search result."
        empty="None proposed."
        count={draft.enrichment_targets.length}
      >
        {draft.enrichment_targets.map((target) => (
          <Chip
            key={target.key}
            title={target.description}
            onRemove={() =>
              onChange({
                ...draft,
                enrichment_targets: draft.enrichment_targets.filter((t) => t.key !== target.key),
              })
            }
          >
            {target.key.replaceAll("_", " ")}
            <span className="ml-1.5 font-mono text-[11px] text-faint">{target.type}</span>
          </Chip>
        ))}
      </Section>

      {notes.length > 0 && (
        <Notice tone="wait">
          <p className="mb-1 font-medium">Checked and adjusted</p>
          <ul className="space-y-0.5 text-muted">
            {notes.map((note) => (
              <li key={note}>
                <span className="text-faint">· </span>
                {note}
              </li>
            ))}
          </ul>
        </Notice>
      )}

      {children}
    </div>
  );
}

function Section({
  title,
  caption,
  empty,
  count,
  children,
}: {
  title: string;
  caption: string;
  empty: string;
  count: number;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline gap-2">
        <h4 className="text-sm font-medium text-foreground">{title}</h4>
        <Tag>{count}</Tag>
      </div>
      <p className="-mt-1 text-xs text-muted">{caption}</p>
      <div className="flex flex-wrap gap-1.5">
        {count === 0 ? <span className="text-xs text-faint">{empty}</span> : children}
      </div>
    </div>
  );
}

function Chip({
  onRemove,
  title,
  children,
}: {
  onRemove: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      title={title}
      className="group inline-flex animate-pop items-center gap-1 rounded-full border border-hairline bg-surface py-0.5 pl-2.5 pr-1 text-xs text-foreground transition-colors duration-100 hover:border-hairline-strong"
    >
      <span>{children}</span>
      <button
        type="button"
        onClick={onRemove}
        aria-label="Remove"
        className="rounded-full px-1 text-faint transition-colors duration-100 hover:text-err group-hover:text-muted"
      >
        ×
      </button>
    </span>
  );
}

function ErrorLine({ children }: { children: React.ReactNode }) {
  return <Notice tone="err">{children}</Notice>;
}
