"use client";

import { useId } from "react";

import { MAPPABLE_FIELDS, type ColumnMapping, type MappableField, type OntologyOption } from "@/lib/api/types";
import { mappedHeaders, samplesOf } from "@/lib/csv-mapping";

import { fieldClass } from "../ui";

const FIELD_COPY: Record<MappableField, { label: string; hint: string }> = {
  name: { label: "Name", hint: "What the entity is called" },
  address: { label: "Address", hint: "Street, area, landmark" },
  district: { label: "District / city", hint: "" },
  state: { label: "State", hint: "Name or code" },
  pincode: { label: "PIN code", hint: "Six digits" },
};

type Target =
  | { kind: "field"; field: MappableField }
  | { kind: "key"; key: string }
  | { kind: "record_id" };

/**
 * Which CSV column means what. Pre-filled from the domain's header aliases
 * and a few narrow guesses (`schl_nm` -> Name); the user confirms or changes
 * every row. The sample values beside each choice are the check: a "Name"
 * column showing "Mrs Sharma" is the principal, not the school.
 *
 * One column per field: picking a column already used moves it here.
 */
export function ColumnMapper({
  headers,
  rows,
  ontology,
  mapping,
  onChange,
}: {
  headers: string[];
  rows: Record<string, string>[];
  ontology: OntologyOption | undefined;
  mapping: ColumnMapping;
  onChange: (mapping: ColumnMapping) => void;
}) {
  const keys = ontology?.natural_keys ?? [];
  const used = mappedHeaders(mapping);
  const unmapped = headers.filter((h) => !used.has(h)).length;

  function valueOf(target: Target): string {
    if (target.kind === "field") return mapping.fields[target.field] ?? "";
    if (target.kind === "key") return mapping.natural_keys?.[target.key] ?? "";
    return mapping.record_id ?? "";
  }

  function assign(target: Target, header: string) {
    // Free the column wherever else it was used, then place it here.
    const without = (value: string | undefined) => (value === header ? undefined : value);
    const fields = Object.fromEntries(
      Object.entries(mapping.fields).map(([f, h]) => [f, without(h)]),
    ) as ColumnMapping["fields"];
    const naturalKeys = Object.fromEntries(
      Object.entries(mapping.natural_keys ?? {}).filter(([, h]) => h !== header),
    );
    let recordId = without(mapping.record_id);
    const chosen = header || undefined;
    if (target.kind === "field") fields[target.field] = chosen;
    else if (target.kind === "key") {
      if (chosen) naturalKeys[target.key] = chosen;
      else delete naturalKeys[target.key];
    } else recordId = chosen;
    onChange({
      fields: Object.fromEntries(Object.entries(fields).filter(([, h]) => h)) as ColumnMapping["fields"],
      natural_keys: naturalKeys,
      ...(recordId ? { record_id: recordId } : {}),
    });
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-3">
        <span className="label">column map</span>
        <span className="text-xs text-muted">
          {headers.length} columns · {unmapped} kept as-is
        </span>
      </div>

      <div className="overflow-hidden rounded-lg border border-hairline">
        <div className="grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)_minmax(0,1.3fr)] gap-x-4 border-b border-hairline bg-surface-subtle px-4 py-1.5 max-sm:hidden">
          <span className="label">field</span>
          <span className="label">your column</span>
          <span className="label">sample values</span>
        </div>
        {MAPPABLE_FIELDS.map((field) => (
          <Row
            key={field}
            label={FIELD_COPY[field].label}
            hint={FIELD_COPY[field].hint}
            required={field === "name"}
            headers={headers}
            rows={rows}
            used={used}
            value={valueOf({ kind: "field", field })}
            onChange={(header) => assign({ kind: "field", field }, header)}
          />
        ))}
        <Row
          label="Record ID"
          hint="Your own row id, kept in exports"
          headers={headers}
          rows={rows}
          used={used}
          value={valueOf({ kind: "record_id" })}
          onChange={(header) => assign({ kind: "record_id" }, header)}
        />
        {keys.map((key) => (
          <Row
            key={key.key}
            label={prettify(key.key)}
            hint="Identifier"
            headers={headers}
            rows={rows}
            used={used}
            value={valueOf({ kind: "key", key: key.key })}
            onChange={(header) => assign({ kind: "key", key: key.key }, header)}
          />
        ))}
      </div>

      {!mapping.fields.name && (
        <p className="text-xs text-err">
          Choose the column that holds the name. It is the one field every match needs.
        </p>
      )}
    </div>
  );
}

function Row({
  label,
  hint,
  required = false,
  headers,
  rows,
  used,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  required?: boolean;
  headers: string[];
  rows: Record<string, string>[];
  used: Set<string>;
  value: string;
  onChange: (header: string) => void;
}) {
  const id = useId();
  const samples = value ? samplesOf(rows, value) : [];

  return (
    <div className="grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)_minmax(0,1.3fr)] items-center gap-x-4 gap-y-1.5 border-b border-hairline px-4 py-2 transition-colors duration-100 last:border-b-0 hover:bg-surface-subtle max-sm:grid-cols-1">
      <label htmlFor={id} className="flex flex-col">
        <span className="text-sm text-foreground">
          {label}
          {required && <span className="ml-1 text-accent">*</span>}
        </span>
        {hint && <span className="text-xs text-muted">{hint}</span>}
      </label>

      <div className="relative">
        <select
          id={id}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className={[
            fieldClass,
            "appearance-none truncate pr-8",
            value ? "" : "border-dashed border-hairline-strong text-muted",
          ].join(" ")}
        >
          <option value="">{required ? "Choose a column…" : "Not in this file"}</option>
          {headers.map((header) => (
            <option key={header} value={header}>
              {header}
              {used.has(header) && header !== value ? "  (in use)" : ""}
            </option>
          ))}
        </select>
        <svg
          aria-hidden="true"
          viewBox="0 0 12 12"
          className="pointer-events-none absolute right-3 top-1/2 h-3 w-3 -translate-y-1/2 text-muted"
        >
          <path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      <div className="flex min-w-0 flex-wrap gap-1">
        {samples.length === 0 ? (
          <span className="text-xs text-faint">{value ? "Empty in the first rows" : "—"}</span>
        ) : (
          samples.map((sample) => (
            <span
              key={sample}
              title={sample}
              className="max-w-[14rem] animate-rise truncate rounded-sm bg-surface-raised px-1.5 py-0.5 font-mono text-[11px] text-muted"
            >
              {sample}
            </span>
          ))
        )}
      </div>
    </div>
  );
}

function prettify(key: string): string {
  const label = key.replaceAll("_", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}
