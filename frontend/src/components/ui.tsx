import type { ComponentPropsWithoutRef, ReactNode } from "react";
import Link from "next/link";

import { formatScore, toScore } from "@/lib/format";

/*
  ER_OS primitives. Server-safe: no hooks here, so pages that render on the
  server can use them. Anything that animates with state (count-up numbers,
  sliding tab indicators) lives in ./motion.tsx.

    PageHeader    breadcrumb, title, one line of description, actions.
    Window        a framed panel: header (status, title, actions) + body.
    SectionTitle  a quiet heading for a group of windows, with a count.
    StatusLines   aligned "KEY : value" rows, ZUI's system-status block.
    StatusDot     a dot that radiates while something is running.
    Meter         a progress bar that eases to its value and shimmers while live.
    Notice        a message block: ok, wait, err.
    buttonClass   the four button looks, for <button>, <a> and <Link> alike.
*/

// --- Frames and headings --------------------------------------------------------

export function PageHeader({
  crumbs = [],
  title,
  description,
  actions,
}: {
  crumbs?: { href: string; label: string }[];
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-end gap-x-6 gap-y-4">
      <div className="flex min-w-0 flex-1 flex-col gap-1.5">
        {crumbs.length > 0 && (
          <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 font-mono text-xs text-muted">
            {crumbs.map((crumb) => (
              <span key={crumb.href} className="flex items-center gap-1.5">
                <Link href={crumb.href} className="transition-colors duration-150 hover:text-foreground">
                  {crumb.label}
                </Link>
                <span aria-hidden="true" className="text-faint">
                  /
                </span>
              </span>
            ))}
          </nav>
        )}
        <h1 className="truncate text-xl font-semibold text-foreground">{title}</h1>
        {description && <p className="max-w-[70ch] text-sm text-muted">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

/**
 * A framed panel. The header carries a status dot, the title and actions; the
 * body gets the padding unless `flush` (tables run edge to edge).
 */
export function Window({
  title,
  meta,
  right,
  status,
  flush = false,
  className = "",
  children,
}: {
  title: ReactNode;
  meta?: ReactNode;
  right?: ReactNode;
  status?: ReactNode;
  flush?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section
      className={`overflow-hidden rounded-lg border border-hairline bg-surface transition-colors duration-300 hover:border-hairline-strong ${className}`}
    >
      <header className="flex min-h-11 flex-wrap items-center gap-x-3 gap-y-1 border-b border-hairline px-4 py-2">
        {status}
        <h3 className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
          {title}
          {meta && <span className="ml-2 font-mono text-xs font-normal text-muted">{meta}</span>}
        </h3>
        {right && <div className="flex shrink-0 flex-wrap items-center gap-1.5">{right}</div>}
      </header>
      <div className={flush ? "" : "p-4"}>{children}</div>
    </section>
  );
}

/** A group heading: title, an optional count, and something on the right. */
export function SectionTitle({
  children,
  count,
  right,
}: {
  children: ReactNode;
  count?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <h2 className="flex items-center gap-2 text-sm font-medium text-foreground">
        {children}
        {count !== undefined && (
          <span className="rounded-full bg-surface-raised px-2 py-0.5 font-mono text-[11px] text-muted">
            {count}
          </span>
        )}
      </h2>
      {right && <div className="flex items-center gap-2 text-xs text-muted">{right}</div>}
    </div>
  );
}

/**
 * ZUI's system-status block: "STATUS    : running". Keys share a column so
 * values line up. Mono, because this is the machine talking.
 */
export function StatusLines({
  rows,
  keyWidth = "11ch",
}: {
  rows: [ReactNode, ReactNode][];
  keyWidth?: string;
}) {
  return (
    <dl
      className="grid gap-y-1 font-mono text-xs"
      style={{ gridTemplateColumns: `minmax(0,${keyWidth}) minmax(0,1fr)` }}
    >
      {rows.map(([key, value], index) => (
        <div key={index} className="contents">
          <dt className="label truncate leading-6 after:text-faint after:content-['_:']">{key}</dt>
          <dd className="min-w-0 truncate leading-6 text-foreground">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

// --- Status --------------------------------------------------------------------

export type Tone = "ok" | "active" | "wait" | "err" | "idle";

const DOT_TONE: Record<Tone, string> = {
  ok: "bg-accent text-accent",
  active: "bg-accent text-accent ping",
  wait: "bg-muted text-muted ping",
  err: "bg-err text-err",
  idle: "bg-faint text-faint",
};

/** A 7px status dot. "active" and "wait" radiate a ring; the colour is the meaning. */
export function StatusDot({ tone, label }: { tone: Tone; label?: string }) {
  return (
    <span
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={`inline-block h-[7px] w-[7px] shrink-0 rounded-full transition-colors duration-300 ${DOT_TONE[tone]}`}
    />
  );
}

const NOTICE_TONE: Record<"ok" | "wait" | "err", { glyph: string; box: string; mark: string }> = {
  ok: { glyph: "✓", box: "border-accent-border bg-accent-dim/40", mark: "bg-accent text-background" },
  wait: { glyph: "!", box: "border-hairline bg-surface-subtle", mark: "bg-surface-raised text-muted" },
  err: { glyph: "×", box: "border-err-border bg-err-dim/50", mark: "bg-err text-background" },
};

/**
 * A message block. Warnings are quiet, not orange (the ZUI rule): the mark
 * and the words carry the weight, so colour stays for "live" and "broken".
 */
export function Notice({
  tone,
  children,
  className = "",
}: {
  tone: "ok" | "wait" | "err";
  children: ReactNode;
  className?: string;
}) {
  const { glyph, box, mark } = NOTICE_TONE[tone];
  return (
    <div
      role={tone === "err" ? "alert" : "status"}
      className={`flex animate-rise items-start gap-3 rounded-md border px-3 py-2.5 text-sm leading-relaxed text-foreground ${box} ${className}`}
    >
      <span
        aria-hidden="true"
        className={`mt-0.5 flex h-4 w-4 shrink-0 animate-pop items-center justify-center rounded-full font-mono text-[10px] leading-none ${mark}`}
      >
        {glyph}
      </span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

/**
 * A progress bar. The fill eases to each new value (so a polled count glides
 * rather than jumps), carries a travelling sheen while `live`, and slides a
 * short segment back and forth when there is no total yet (`value === null`).
 */
export function Meter({
  value,
  tone = "active",
  label,
  live = false,
  showValue = true,
}: {
  value: number | null;
  tone?: "active" | "wait" | "err";
  label: string;
  live?: boolean;
  showValue?: boolean;
}) {
  const fraction = value === null ? null : Math.max(0, Math.min(1, value));
  const fill = tone === "err" ? "bg-err" : tone === "wait" ? "bg-muted" : "bg-accent";
  return (
    <div className="flex min-w-0 items-center gap-3">
      <div
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}
        className="relative h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-surface-raised"
      >
        {fraction === null ? (
          <div className={`absolute inset-y-0 left-0 w-1/4 rounded-full indeterminate ${fill}`} />
        ) : (
          <div
            className={`h-full rounded-full transition-[width] duration-700 ease-[var(--ease-out)] ${fill} ${
              live && tone === "active" ? "sheen" : ""
            }`}
            style={{ width: `${Math.max(fraction > 0 ? 1.5 : 0, fraction * 100)}%` }}
          />
        )}
      </div>
      {showValue && fraction !== null && (
        <span className="w-10 shrink-0 text-right font-mono text-xs text-muted">
          {Math.round(fraction * 100)}%
        </span>
      )}
    </div>
  );
}

/** A loading block with a highlight sweeping across it. */
export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden="true" className={`skeleton rounded-lg ${className}`} />;
}

// --- Actions -------------------------------------------------------------------

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const BUTTON_VARIANT: Record<ButtonVariant, string> = {
  /* The one primary per screen: solid accent, a soft accent halo on hover. */
  primary:
    "bg-accent text-background font-medium hover:shadow-[0_0_0_4px_var(--accent-glow)] hover:brightness-110",
  secondary:
    "border border-hairline bg-surface text-foreground hover:border-hairline-strong hover:bg-surface-raised",
  ghost: "text-muted hover:bg-surface-raised hover:text-foreground",
  danger: "text-muted hover:bg-err-dim hover:text-err",
};

/**
 * The button look for any element: <button>, <a> or <Link>. Presses scale
 * down a touch; nothing else moves.
 */
export function buttonClass(variant: ButtonVariant = "secondary", size: "sm" | "md" = "md"): string {
  const sizing = size === "sm" ? "h-7 px-2.5 text-xs" : "h-8 px-3.5 text-sm";
  return [
    "inline-flex shrink-0 select-none items-center justify-center gap-1.5 rounded-md whitespace-nowrap",
    "transition-[color,background-color,border-color,box-shadow,transform,filter] duration-150 ease-[var(--ease-out)]",
    "active:scale-[0.97] disabled:pointer-events-none disabled:opacity-40",
    sizing,
    BUTTON_VARIANT[variant],
  ].join(" ");
}

export function Button({
  variant = "secondary",
  size = "md",
  className = "",
  ...props
}: ComponentPropsWithoutRef<"button"> & { variant?: ButtonVariant; size?: "sm" | "md" }) {
  return <button type="button" className={`${buttonClass(variant, size)} ${className}`} {...props} />;
}

/**
 * Text inputs, selects and textareas: a hairline well that takes a soft
 * accent halo on focus.
 */
export const fieldClass =
  "w-full rounded-md border border-hairline bg-surface px-3 py-1.5 text-sm text-foreground transition-[border-color,box-shadow] duration-150 hover:border-hairline-strong focus:border-accent-border focus:shadow-[0_0_0_3px_var(--accent-glow)] focus:outline-none disabled:opacity-40";

/** A small pill: counts, tags, chips. ZUI's 999px radius lives here. */
export function Tag({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "err";
}) {
  const color =
    tone === "accent"
      ? "bg-accent-dim text-accent"
      : tone === "err"
        ? "bg-err-dim text-err"
        : "bg-surface-raised text-muted";
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 font-mono text-[11px] leading-none ${color}`}>
      {children}
    </span>
  );
}

// --- Existing primitives, retoned ------------------------------------------------

/** A keyboard hint. Rendered as a real <kbd> so screen readers announce it. */
export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-5 min-w-5 items-center justify-center rounded-sm border border-hairline-strong bg-surface px-1 font-mono text-[11px] leading-none text-muted">
      {children}
    </kbd>
  );
}

/**
 * Confidence band. Accent at or above the auto-merge threshold, plain in the
 * band the automated check settles, dim below the floor.
 */
export function ConfidenceBadge({ value }: { value: string | number | null }) {
  const score = toScore(value);

  const tone =
    score === null
      ? "bg-surface-raised text-faint"
      : score >= 0.95
        ? "bg-accent-dim text-accent"
        : score >= 0.8
          ? "bg-surface-raised text-foreground"
          : "bg-surface-raised text-muted";

  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 font-mono text-[11px] leading-none ${tone}`}>
      {formatScore(value)}
    </span>
  );
}

/** Shortened identifier. Full value stays in the title for copy/paste. */
export function MonoId({ value, length = 8 }: { value: string; length?: number }) {
  return (
    <code title={value} className="font-mono text-[11px] text-muted">
      {value.slice(0, length)}
    </code>
  );
}

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="label">{label}</span>
      <span className="font-mono text-lg leading-none text-foreground">{value}</span>
    </div>
  );
}
