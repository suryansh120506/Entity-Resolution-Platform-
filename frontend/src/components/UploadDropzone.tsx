"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getUploadOptions, uploadCsv, type UploadHandle } from "@/lib/api/jobs";
import type { ColumnMapping, JobAccepted, OntologyOption, UploadOptions } from "@/lib/api/types";
import { previewCsv, rebaseMapping, suggestMapping, type CsvPreview } from "@/lib/csv-mapping";
import { messageOf } from "@/lib/errors";

import { ColumnMapper } from "./upload/ColumnMapper";
import { CustomDomainBuilder } from "./upload/CustomDomainBuilder";
import { CUSTOM, DomainSelector } from "./upload/DomainSelector";
import { Meter, Notice, StatusDot, Window, buttonClass } from "./ui";

type Step = "pick" | "configure" | "uploading" | "accepted" | "error";

interface Props {
  onAccepted?: (job: JobAccepted) => void;
}

/**
 * Upload in three moves: choose a file, confirm what it means, send it.
 *
 * 1. **Pick.** The file is read *in the browser* (PapaParse, first 25 rows
 *    only), so the next step can show its real columns and values before a
 *    single byte is uploaded.
 * 2. **Configure.** The data domain (a shipped ontology, one of this
 *    workspace's custom domains, or a new one drafted from a sentence) and
 *    the column mapping, pre-filled and confirmed by the user. The mapping is
 *    authoritative: it is applied at ingestion (see column_mapping.py).
 * 3. **Upload.** As before: a 202 means *queued*, not done, so the final
 *    state says so and points at the job.
 */
export function UploadDropzone({ onAccepted }: Props) {
  const [step, setStep] = useState<Step>("pick");
  const [options, setOptions] = useState<UploadOptions | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [ontology, setOntology] = useState("education");
  const [lastDomain, setLastDomain] = useState("education");
  const [mapping, setMapping] = useState<ColumnMapping>({ fields: {} });
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [job, setJob] = useState<JobAccepted | null>(null);
  const [dragging, setDragging] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const handleRef = useRef<UploadHandle | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getUploadOptions({ signal: controller.signal })
      .then((loaded) => {
        setOptions(loaded);
        setOntology(loaded.default_ontology);
        setLastDomain(loaded.default_ontology);
      })
      .catch(() => {
        // The selector falls back to Education; the upload surfaces real errors.
      });
    return () => controller.abort();
  }, []);

  const domains = useMemo<OntologyOption[]>(
    () => options?.ontologies ?? [{ name: "education", label: "Education", description: "" }],
    [options],
  );
  const optionFor = useCallback((name: string) => domains.find((d) => d.name === name), [domains]);

  const fail = (text: string) => {
    setStep("error");
    setMessage(text);
  };

  const choose = useCallback(
    async (chosen: File) => {
      if (!chosen.name.toLowerCase().endsWith(".csv")) {
        fail(`${chosen.name} is not a .csv file.`);
        return;
      }
      if (options && chosen.size > options.max_upload_bytes) {
        fail(`${chosen.name} is larger than the ${formatBytes(options.max_upload_bytes)} limit.`);
        return;
      }
      try {
        const read = await previewCsv(chosen);
        setFile(chosen);
        setPreview(read);
        setMapping(suggestMapping(read.headers, optionFor(ontology === CUSTOM ? lastDomain : ontology)));
        setStep("configure");
        setMessage(null);
      } catch (error) {
        fail(`Could not read ${chosen.name}: ${messageOf(error)}`);
      } finally {
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [lastDomain, ontology, optionFor, options],
  );

  function changeDomain(name: string, option: OntologyOption | undefined = optionFor(name)) {
    setOntology(name);
    if (name === CUSTOM) return;
    setLastDomain(name);
    if (preview) setMapping((current) => rebaseMapping(current, preview.headers, option));
  }

  function domainSaved(option: OntologyOption) {
    setOptions((current) =>
      current ? { ...current, ontologies: [...current.ontologies, option] } : current,
    );
    changeDomain(option.name, option);
  }

  function upload() {
    if (!file) return;
    setStep("uploading");
    setProgress(0);
    setJob(null);
    const handle = uploadCsv(file, ontology, options?.direct_upload ?? false, setProgress, mapping);
    handleRef.current = handle;
    handle.promise
      .then((accepted) => {
        setStep("accepted");
        setProgress(1);
        setJob(accepted);
        onAccepted?.(accepted);
      })
      .catch((error: unknown) => fail(messageOf(error)))
      .finally(() => {
        handleRef.current = null;
      });
  }

  function reset() {
    setStep("pick");
    setFile(null);
    setPreview(null);
    setMessage(null);
    if (ontology === CUSTOM) setOntology(lastDomain);
  }

  const canUpload = step === "configure" && ontology !== CUSTOM && Boolean(mapping.fields.name);

  return (
    <div className="flex flex-col gap-4">
      {(step === "pick" || step === "error") && (
        <label
          htmlFor="csv-input"
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const dropped = event.dataTransfer.files?.[0];
            if (dropped) void choose(dropped);
          }}
          className={[
            "group relative flex cursor-pointer flex-col items-center gap-3 overflow-hidden rounded-lg border border-dashed px-6 py-12 text-center",
            "transition-[background-color,border-color] duration-300 ease-[var(--ease-out)]",
            dragging
              ? "marching bg-accent-dim/30"
              : "border-hairline-strong bg-surface hover:border-accent-border hover:bg-surface-subtle",
          ].join(" ")}
        >
          {/* A soft accent glow that blooms while a file is over the zone. */}
          <span
            aria-hidden="true"
            className={`pointer-events-none absolute inset-0 transition-opacity duration-500 ${
              dragging ? "opacity-100" : "opacity-0"
            }`}
            style={{ background: "radial-gradient(60% 80% at 50% 0%, var(--accent-glow), transparent)" }}
          />
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="sr-only"
            id="csv-input"
            onChange={(event) => {
              const picked = event.target.files?.[0];
              if (picked) void choose(picked);
            }}
          />
          <span
            className={`relative flex h-10 w-10 items-center justify-center rounded-full border transition-[transform,border-color,color] duration-300 ease-[var(--ease-out)] ${
              dragging
                ? "-translate-y-1 scale-110 border-accent-border text-accent"
                : "border-hairline text-muted group-hover:-translate-y-0.5 group-hover:border-accent-border group-hover:text-accent"
            }`}
          >
            <UploadGlyph />
          </span>
          <span className="relative text-sm text-foreground">
            {dragging ? "Drop to read it" : (
              <>
                Drop a CSV here, or <span className="link">choose a file</span>
              </>
            )}
          </span>
          <span className="relative font-mono text-xs text-muted">
            .csv up to {options ? formatBytes(options.max_upload_bytes) : "256 MB"} · read in your
            browser first
          </span>
        </label>
      )}

      {step === "configure" && file && preview && (
        <div className="animate-rise">
          <Window
            title={file.name}
            meta={`${formatBytes(file.size)} · ${preview.headers.length} columns`}
            status={<StatusDot tone="wait" />}
            right={
              <button type="button" onClick={reset} className={buttonClass("ghost", "sm")}>
                Change file
              </button>
            }
          >
            <div className="stagger flex flex-col gap-6">
              <DomainSelector
                options={domains}
                value={ontology}
                onChange={(name) => changeDomain(name)}
                customAvailable={options?.custom_domains_available ?? false}
              />

              {ontology === CUSTOM ? (
                <CustomDomainBuilder
                  headers={preview.headers}
                  rows={preview.rows}
                  onSaved={domainSaved}
                  onCancel={() => changeDomain(lastDomain)}
                />
              ) : (
                <ColumnMapper
                  headers={preview.headers}
                  rows={preview.rows}
                  ontology={optionFor(ontology)}
                  mapping={mapping}
                  onChange={setMapping}
                />
              )}

              <div className="flex items-center justify-end gap-3 border-t border-hairline pt-4">
                {ontology === CUSTOM && (
                  <span className="mr-auto text-xs text-muted">Save the domain to continue.</span>
                )}
                {ontology !== CUSTOM && !mapping.fields.name && (
                  <span className="mr-auto text-xs text-muted">Choose the name column to continue.</span>
                )}
                <button type="button" onClick={upload} disabled={!canUpload} className={buttonClass("primary")}>
                  Start processing
                  <span aria-hidden="true">→</span>
                </button>
              </div>
            </div>
          </Window>
        </div>
      )}

      {step === "uploading" && (
        <div className="animate-rise">
          <Window
            title={file?.name ?? "file"}
            meta="uploading"
            status={<StatusDot tone="active" />}
            right={
              <button
                type="button"
                onClick={() => handleRef.current?.cancel()}
                className={buttonClass("ghost", "sm")}
              >
                Cancel
              </button>
            }
          >
            <Meter value={progress} live label="Upload progress" />
          </Window>
        </div>
      )}

      {step === "accepted" && job && (
        <Notice tone="ok">
          <div className="flex flex-wrap items-center justify-between gap-2">
            {/* 202, not 200: queued, not done. */}
            <span>
              Queued. Job <code className="font-mono text-accent">{job.job_id.slice(0, 8)}</code> (
              {formatBytes(job.total_bytes)}) appears below as it runs.
            </span>
            <button type="button" onClick={reset} className={buttonClass("ghost", "sm")}>
              Upload another
            </button>
          </div>
        </Notice>
      )}

      {step === "error" && message && (
        <Notice tone="err">
          <span className="font-medium">Upload failed.</span> {message}
        </Notice>
      )}
    </div>
  );
}

function UploadGlyph() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-[18px] w-[18px]">
      <path
        d="M12 15V4m0 0-4 4m4-4 4 4M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}
