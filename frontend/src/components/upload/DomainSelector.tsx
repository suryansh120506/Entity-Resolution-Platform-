"use client";

import type { OntologyOption } from "@/lib/api/types";

export const CUSTOM = "__custom__";

/**
 * Which data domain (ontology) standardises the upload: the shipped ones, this
 * workspace's own custom domains, and "+ Custom domain" to draft a new one
 * from a sentence.
 *
 * A wrapping row of pills rather than a <select>: there are few options, and
 * seeing them all at once is what tells a user a custom domain is possible.
 */
export function DomainSelector({
  options,
  value,
  onChange,
  customAvailable,
  disabled = false,
}: {
  options: OntologyOption[];
  value: string;
  onChange: (value: string) => void;
  customAvailable: boolean;
  disabled?: boolean;
}) {
  const shipped = options.filter((o) => !o.custom);
  const custom = options.filter((o) => o.custom);
  const selected = options.find((o) => o.name === value);

  return (
    <div className="flex flex-col gap-2">
      <span className="label">data domain</span>
      <div role="radiogroup" aria-label="Data domain" className="flex flex-wrap gap-1.5">
        {shipped.map((option) => (
          <Pill
            key={option.name}
            active={option.name === value}
            disabled={disabled}
            onClick={() => onChange(option.name)}
          >
            {option.label}
          </Pill>
        ))}
        {custom.length > 0 && <span aria-hidden="true" className="mx-1 w-px self-stretch bg-hairline" />}
        {custom.map((option) => (
          <Pill
            key={option.name}
            active={option.name === value}
            disabled={disabled}
            onClick={() => onChange(option.name)}
          >
            <span aria-hidden="true" className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-accent align-middle" />
            {option.label}
          </Pill>
        ))}
        <Pill
          active={value === CUSTOM}
          disabled={disabled || !customAvailable}
          onClick={() => onChange(CUSTOM)}
          dashed
          title={
            customAvailable
              ? "Describe your dataset and let the AI draft its rules"
              : "Needs an AI: configure one in Settings"
          }
        >
          + Custom domain
        </Pill>
      </div>
      {selected?.description ? (
        <p className="text-xs text-muted">{selected.description}</p>
      ) : value === CUSTOM ? null : (
        <p className="text-xs text-muted">&nbsp;</p>
      )}
    </div>
  );
}

function Pill({
  active,
  disabled,
  onClick,
  dashed = false,
  title,
  children,
}: {
  active: boolean;
  disabled: boolean;
  onClick: () => void;
  dashed?: boolean;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      disabled={disabled}
      onClick={onClick}
      title={title}
      className={[
        "rounded-full px-3 py-1 text-xs transition-[color,background-color,border-color,transform] duration-150 active:scale-[0.97]",
        active
          ? "border border-accent-border bg-accent-dim text-accent"
          : dashed
            ? "border border-dashed border-hairline-strong text-muted hover:border-accent-border hover:text-foreground"
            : "border border-hairline bg-surface text-muted hover:border-hairline-strong hover:text-foreground",
        disabled ? "cursor-not-allowed opacity-40 hover:border-hairline" : "",
      ].join(" ")}
    >
      {children}
    </button>
  );
}
