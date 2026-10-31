import Papa from "papaparse";

import {
  MAPPABLE_FIELDS,
  type ColumnMapping,
  type MappableField,
  type OntologyOption,
} from "@/lib/api/types";

/** The first rows of a CSV, read in the browser. Nothing is uploaded yet. */
export interface CsvPreview {
  /** Headers exactly as written, as the server's CSV reader will see them. */
  headers: string[];
  /** Up to PREVIEW_ROWS rows keyed by header. */
  rows: Record<string, string>[];
}

const PREVIEW_ROWS = 25;

/**
 * Read the header row and a few data rows of a (possibly huge) CSV.
 *
 * `preview` makes PapaParse stop after the first rows, so a 200 MB file costs
 * one small read. `header: false` on purpose: PapaParse's header mode renames
 * duplicate headers ("name" -> "name_1"), which the server's reader does not,
 * and a mapping that names a renamed header points at a column the server
 * cannot find.
 */
export function previewCsv(file: File): Promise<CsvPreview> {
  return new Promise((resolve, reject) => {
    Papa.parse<string[]>(file, {
      header: false,
      preview: PREVIEW_ROWS + 1,
      skipEmptyLines: "greedy",
      complete: (result) => {
        const [first, ...rest] = result.data;
        if (!first || first.length === 0) {
          reject(new Error("The file has no header row."));
          return;
        }
        // The server strips a UTF-8 BOM; do the same so headers match.
        const headers = first.map((h, i) => (i === 0 ? h.replace(/^﻿/, "") : h));
        const rows = rest.map((cells) => {
          const row: Record<string, string> = {};
          headers.forEach((header, i) => {
            // Later duplicates win, as with Python's csv.DictReader.
            row[header] = cells[i] ?? "";
          });
          return row;
        });
        resolve({ headers: unique(headers), rows });
      },
      error: (error) => reject(error),
    });
  });
}

/** Same normalisation as the server's `normalize_key`. */
export function normalizeHeader(header: string): string {
  return Array.from(header.trim().toLowerCase())
    .filter((ch) => /[\p{L}\p{N}]/u.test(ch))
    .join("");
}

const PEOPLE = /(principal|owner|contact|person|father|mother|head|manager|ceo|director|proprietor)/;

/**
 * Second-chance matching for headers no alias table knows ("schl_nm",
 * "dealer_name", "pin_cd"). Deliberately narrow: a wrong suggestion the user
 * does not notice is worse than an empty one they fill in.
 */
const FUZZY: Record<MappableField, (n: string) => boolean> = {
  name: (n) => (n.includes("name") || n.endsWith("nm") || n.includes("title")) && !PEOPLE.test(n),
  address: (n) => n.includes("addr") || n.includes("street") || n.includes("location"),
  district: (n) => n.includes("district") || n.startsWith("dist") || n.includes("city"),
  state: (n) => n.includes("state") || n === "st" || n === "stcd" || n === "stnm",
  pincode: (n) =>
    n.startsWith("pin") || n.includes("pincode") || n.includes("zip") || n.includes("postal"),
};

const RECORD_ID = new Set(["id", "uid", "recordid", "rowid", "sno", "srno", "serialno"]);

/**
 * A first guess at the mapping: exact alias matches from the domain first,
 * then the narrow fuzzy rules, each column used at most once.
 */
export function suggestMapping(headers: string[], ontology: OntologyOption | undefined): ColumnMapping {
  const used = new Set<string>();
  const take = (header: string | undefined) => {
    if (header) used.add(header);
    return header;
  };
  const free = () => headers.filter((h) => !used.has(h));

  const fields: ColumnMapping["fields"] = {};
  const aliases = ontology?.field_aliases ?? {};
  for (const field of MAPPABLE_FIELDS) {
    const known = new Set(aliases[field] ?? [field]);
    fields[field] = take(free().find((h) => known.has(normalizeHeader(h))));
  }
  for (const field of MAPPABLE_FIELDS) {
    if (!fields[field]) fields[field] = take(free().find((h) => FUZZY[field](normalizeHeader(h))));
  }

  const natural_keys: Record<string, string> = {};
  for (const key of ontology?.natural_keys ?? []) {
    const spellings = new Set(key.headers);
    const wanted = normalizeHeader(key.key);
    const header = free().find((h) => {
      const n = normalizeHeader(h);
      return spellings.has(n) || n.includes(wanted);
    });
    if (take(header)) natural_keys[key.key] = header!;
  }

  const record_id = take(free().find((h) => RECORD_ID.has(normalizeHeader(h))));
  return { fields: compact(fields), natural_keys, ...(record_id ? { record_id } : {}) };
}

/** Re-target the natural keys when the domain changes; keep the user's field choices. */
export function rebaseMapping(
  current: ColumnMapping,
  headers: string[],
  ontology: OntologyOption | undefined,
): ColumnMapping {
  const keys = new Set((ontology?.natural_keys ?? []).map((k) => k.key));
  const kept = Object.fromEntries(
    Object.entries(current.natural_keys ?? {}).filter(([key]) => keys.has(key)),
  );
  const suggested = suggestMapping(
    headers.filter((h) => !mappedHeaders(current).has(h)),
    ontology,
  ).natural_keys;
  return { ...current, natural_keys: { ...suggested, ...kept } };
}

export function mappedHeaders(mapping: ColumnMapping): Set<string> {
  return new Set(
    [
      ...Object.values(mapping.fields),
      ...Object.values(mapping.natural_keys ?? {}),
      mapping.record_id,
    ].filter((h): h is string => Boolean(h)),
  );
}

/** First few non-empty values of a column, for the mapping screen. */
export function samplesOf(rows: Record<string, string>[], header: string, count = 3): string[] {
  const out: string[] = [];
  for (const row of rows) {
    const value = (row[header] ?? "").trim();
    if (value && !out.includes(value)) out.push(value);
    if (out.length === count) break;
  }
  return out;
}

/** A few rows trimmed for the AI provider: short values, few rows. */
export function samplesForAi(rows: Record<string, string>[], count = 15): Record<string, string>[] {
  return rows.slice(0, count).map((row) =>
    Object.fromEntries(Object.entries(row).map(([k, v]) => [k, String(v).slice(0, 80)])),
  );
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values));
}

function compact<T extends Record<string, string | undefined>>(record: T): Partial<Record<keyof T, string>> {
  return Object.fromEntries(Object.entries(record).filter(([, v]) => v)) as Partial<
    Record<keyof T, string>
  >;
}
