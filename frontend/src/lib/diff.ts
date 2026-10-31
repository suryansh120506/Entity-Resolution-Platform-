import type { ClusterMemberSummary } from "@/lib/api/types";
import { fieldValue } from "@/lib/format";

/**
 * Token-level diffing and discrepancy detection.
 *
 * The steward's job is to answer "is this the same school?" in a couple of
 * seconds. That is a reading task, and the thing that makes it slow is
 * scanning two near-identical strings for the one token that differs. So the
 * diff is computed at *word* granularity, not character: highlighting
 * "Karnat<mark>k</mark>a" makes the eye hunt for a letter, while highlighting
 * the whole token "Karnatka" is legible at a glance.
 */

export type DiffKind = "equal" | "removed" | "added";

export interface DiffSegment {
  kind: DiffKind;
  text: string;
}

/**
 * Canonical field names, and the `raw_row` keys that can carry them.
 *
 * CSV headers vary per tenant, and the spec names fields (`address_line1`)
 * that this data spells differently (`address`). Resolving through aliases
 * means the comparison grid shows the same five rows whatever the upload
 * called them, instead of silently omitting a field because of its header.
 */
export const CANONICAL_FIELDS = [
  { key: "name", label: "Name", aliases: ["name", "school_name", "schoolname", "institution_name"] },
  {
    key: "address",
    label: "Address",
    aliases: ["address", "address_line1", "addressline1", "street_address", "location"],
  },
  { key: "locality", label: "Locality", aliases: ["district", "city", "town", "taluk", "locality"] },
  { key: "state", label: "State", aliases: ["state", "statename"] },
  { key: "pincode", label: "Pincode", aliases: ["pincode", "pin", "postal_code", "postalcode", "zip"] },
] as const;

/** Fields the diff highlights. Others are shown but not tinted. */
const DIFFED_FIELDS = new Set(["name", "address", "pincode"]);

export function isDiffedField(canonicalKey: string): boolean {
  return DIFFED_FIELDS.has(canonicalKey);
}

/**
 * This module's field key -> the key the backend's `canonical_fields`
 * (`app/ml_pipeline/fields.py`) uses for the same field on `clean_fields`.
 * Only "locality" differs; everything else matches by name.
 */
const CLEAN_FIELD_KEY: Record<string, string> = { locality: "district" };

function normaliseKey(key: string): string {
  return key.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/**
 * Read a canonical field out of a record.
 *
 * Prefers `member.clean_fields` — alias-resolved and redundant-comma/
 * null-sentinel-cleaned server-side, by the same alias table the classifier
 * itself reads (`app/ml_pipeline/fields.py`), so this can never recognise a
 * header the classifier does not or vice versa. Falls back to parsing
 * `raw_row` directly only if a member somehow arrives without it.
 */
export function canonicalValue(member: ClusterMemberSummary, canonicalKey: string): string {
  const clean = member.clean_fields?.[CLEAN_FIELD_KEY[canonicalKey] ?? canonicalKey];
  if (clean) return clean;

  const field = CANONICAL_FIELDS.find((candidate) => candidate.key === canonicalKey);
  if (!field) return "";

  const row = member.raw_row ?? {};
  const byNormalised = new Map(
    Object.keys(row).map((key) => [normaliseKey(key), key] as const),
  );

  for (const alias of field.aliases) {
    const actual = byNormalised.get(normaliseKey(alias));
    if (actual) {
      const value = fieldValue(member, actual);
      if (value) return value;
    }
  }
  return "";
}

/**
 * Word-level diff of `variant` against `master`, via longest common
 * subsequence.
 *
 * LCS rather than a naive index-by-index walk because records differ by
 * *insertions*, not just substitutions: "Main Road Block 7" vs
 * "Main Rd Block 7" aligns only if the algorithm can skip a token on one
 * side. Positional comparison would mark every token after the first
 * difference as changed, which is noise dressed as signal.
 *
 * O(n·m) on token counts. Field values here are short — a school name is a
 * dozen words — so the quadratic table is a few hundred cells.
 */
export function diffWords(master: string, variant: string): DiffSegment[] {
  if (master === variant) {
    return variant ? [{ kind: "equal", text: variant }] : [];
  }
  if (!master) return variant ? [{ kind: "added", text: variant }] : [];
  if (!variant) return [{ kind: "removed", text: master }];

  const left = tokenise(master);
  const right = tokenise(variant);

  // lcs[i][j] = length of the longest common subsequence of left[i:], right[j:]
  const lcs: number[][] = Array.from({ length: left.length + 1 }, () =>
    new Array<number>(right.length + 1).fill(0),
  );

  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      lcs[i][j] = comparable(left[i]) === comparable(right[j])
        ? lcs[i + 1][j + 1] + 1
        : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }

  const segments: DiffSegment[] = [];
  let i = 0;
  let j = 0;

  while (i < left.length && j < right.length) {
    if (comparable(left[i]) === comparable(right[j])) {
      // Emit the *variant's* spelling for equal tokens: the pane being read
      // is the variant, and substituting the master's casing there would
      // misrepresent the record under review.
      push(segments, "equal", right[j]);
      i += 1;
      j += 1;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      push(segments, "removed", left[i]);
      i += 1;
    } else {
      push(segments, "added", right[j]);
      j += 1;
    }
  }
  while (i < left.length) push(segments, "removed", left[i++]);
  while (j < right.length) push(segments, "added", right[j++]);

  return segments.filter((segment) => segment.text.length > 0);
}

/** Split into words while keeping the whitespace, so the text reflows intact. */
function tokenise(value: string): string[] {
  return value.split(/(\s+)/).filter((token) => token !== "");
}

/** Case- and punctuation-insensitive comparison: "Rd." and "rd" are the same token. */
function comparable(token: string): string {
  return token.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
}

function push(segments: DiffSegment[], kind: DiffKind, text: string): void {
  const last = segments[segments.length - 1];
  if (last && last.kind === kind) {
    last.text += text;
    return;
  }
  segments.push({ kind, text });
}

/**
 * The field that disagrees most across a cluster — what the queue shows as
 * "Top Discrepancy".
 *
 * Ranked by how many members disagree with the plurality value, tie-broken by
 * the canonical field order (name before address before pincode), so the
 * column is stable rather than reshuffling between renders of the same data.
 */
export function topDiscrepancy(members: ClusterMemberSummary[]): string | null {
  if (members.length < 2) return null;

  let best: { label: string; disagreeing: number; rank: number } | null = null;

  CANONICAL_FIELDS.forEach((field, rank) => {
    const counts = new Map<string, number>();
    for (const member of members) {
      const value = comparable(canonicalValue(member, field.key));
      counts.set(value, (counts.get(value) ?? 0) + 1);
    }
    if (counts.size < 2) return;

    const plurality = Math.max(...counts.values());
    const disagreeing = members.length - plurality;
    if (disagreeing === 0) return;

    if (
      !best ||
      disagreeing > best.disagreeing ||
      (disagreeing === best.disagreeing && rank < best.rank)
    ) {
      best = { label: field.label, disagreeing, rank };
    }
  });

  return best ? `${(best as { label: string }).label} mismatch` : null;
}
