"use client";

import { useEffect, useId, useState } from "react";

import { deleteAiKey, getAiSettings, updateAiSettings } from "@/lib/api/settings";
import type { AiMode, AiSettings, PremiumProvider } from "@/lib/api/types";
import { messageOf } from "@/lib/errors";
import { formatDateTime } from "@/lib/format";

import { useIndicator } from "./motion";
import { Notice, Skeleton, StatusDot, StatusLines, Tag, Window, buttonClass, fieldClass } from "./ui";

/**
 * Standard AI (the platform's free model) or Premium AI (your own key).
 *
 * The key is write-only from here: the API never sends it back, only a hint
 * ("…3f9a"). Saving a Premium key makes the server call the provider once
 * with a known-answer question, so a wrong key or an unsuitable model fails
 * on this screen, with the provider's reason, not hours later inside a job.
 */
export function AiTierSettings() {
  const [settings, setSettings] = useState<AiSettings | null>(null);
  const [mode, setMode] = useState<AiMode>("standard");
  const [provider, setProvider] = useState<PremiumProvider>("anthropic");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const keyId = useId();
  const modelId = useId();

  useEffect(() => {
    const controller = new AbortController();
    getAiSettings({ signal: controller.signal })
      .then((loaded) => {
        setSettings(loaded);
        setMode(loaded.mode);
        if (loaded.provider) setProvider(loaded.provider);
        setModel(loaded.model ?? "");
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) setError(messageOf(caught));
      });
    return () => controller.abort();
  }, []);

  if (!settings) {
    return error ? (
      <Notice tone="err">{error}</Notice>
    ) : (
      <Skeleton className="h-72" />
    );
  }

  const providerInfo = settings.providers.find((p) => p.id === provider);
  const keyStoredForProvider = settings.key_hint !== null && settings.provider === provider;
  const unchanged =
    mode === settings.mode &&
    (mode === "standard" ||
      (provider === settings.provider && model === (settings.model ?? "") && !apiKey));

  async function save() {
    setSaving(true);
    setError(null);
    setSaved(null);
    try {
      const next = await updateAiSettings(
        mode === "standard"
          ? { mode }
          : {
              mode,
              provider,
              ...(model.trim() ? { model: model.trim() } : {}),
              ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
            },
      );
      setSettings(next);
      setModel(next.model ?? "");
      setApiKey("");
      setSaved(
        next.mode === "premium"
          ? `Verified with ${providerInfo?.label ?? provider}. Your own key is now in use.`
          : "Standard AI is on.",
      );
    } catch (caught) {
      setError(messageOf(caught));
    } finally {
      setSaving(false);
    }
  }

  async function removeKey() {
    setSaving(true);
    setError(null);
    try {
      const next = await deleteAiKey();
      setSettings(next);
      setMode("standard");
      setModel("");
      setSaved("Key removed. Standard AI is on.");
    } catch (caught) {
      setError(messageOf(caught));
    } finally {
      setSaving(false);
    }
  }

  const activeProvider = settings.providers.find((p) => p.id === settings.provider);

  return (
    <Window
      title="AI model"
      status={<StatusDot tone={settings.mode === "premium" || settings.standard_model ? "ok" : "idle"} />}
    >
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-4">
          <p className="max-w-[68ch] text-sm text-muted">
            The model that settles uncertain duplicates, enriches merged records, and drafts custom
            data domains.
          </p>
          <StatusLines
            keyWidth="9ch"
            rows={[
              [
                "active",
                settings.mode === "premium" ? (
                  <span key="a">
                    <span className="text-accent">own key</span>{" "}
                    <span className="text-muted">
                      {activeProvider?.label} · {settings.model}
                    </span>
                  </span>
                ) : (
                  <span key="a">
                    <span className="text-accent">standard</span>{" "}
                    {settings.standard_model ? (
                      <span className="text-muted">{settings.standard_model}</span>
                    ) : (
                      <span className="text-err">no platform model</span>
                    )}
                  </span>
                ),
              ],
              [
                "key",
                settings.key_hint ? (
                  <span key="k" className="text-muted">••••{settings.key_hint.slice(1)}</span>
                ) : (
                  <span key="k" className="text-faint">none saved</span>
                ),
              ],
              ...(settings.verified_at && settings.mode === "premium"
                ? [["verified", formatDateTime(settings.verified_at)] as [string, string]]
                : []),
            ]}
          />
        </div>

        <div role="radiogroup" aria-label="AI tier" className="grid gap-3 sm:grid-cols-2">
          <TierCard
            selected={mode === "standard"}
            onSelect={() => setMode("standard")}
            title="Standard"
            badge="Free"
            lines={[
              settings.standard_model
                ? `Shared platform model (${settings.standard_model}).`
                : "No platform model is configured on this server.",
              "Slower at volume: about 1,000 AI calls a day, shared.",
            ]}
          />
          <TierCard
            selected={mode === "premium"}
            onSelect={() => setMode("premium")}
            disabled={!settings.premium_available}
            title="Your own key"
            badge="Free or paid"
            lines={
              settings.premium_available
                ? [
                    "Groq and Cerebras have free tiers with much larger daily budgets; OpenAI and Anthropic are paid.",
                    "Large jobs finish in hours to a day or two, not weeks.",
                  ]
                : ["Unavailable: this server cannot encrypt keys (SECRETS_ENCRYPTION_KEY)."]
            }
          />
        </div>

        {/* Expands and collapses by animating the grid row, so it can ease to
            its natural height without a measured max-height. */}
        <div
          className={`grid transition-[grid-template-rows,opacity] duration-500 ease-[var(--ease-out)] ${
            mode === "premium" ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
          }`}
          aria-hidden={mode !== "premium"}
          inert={mode !== "premium"}
        >
          <div className="min-h-0 overflow-hidden">
            <div className="flex flex-col gap-5 rounded-lg border border-hairline bg-surface-subtle p-4">
              <Field label="Provider" hint={providerInfo?.note}>
                <ProviderPicker
                  providers={settings.providers}
                  value={provider}
                  onChange={(id) => {
                    setProvider(id);
                    setModel(id === settings.provider ? (settings.model ?? "") : "");
                  }}
                />
              </Field>

              <Field
                label="Model"
                htmlFor={modelId}
                hint="Must answer without thinking first: triage asks for a one-word YES or NO."
              >
                <input
                  id={modelId}
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  placeholder={providerInfo?.default_model}
                  spellCheck={false}
                  className={`${fieldClass} font-mono`}
                />
              </Field>

              <Field
                label="API key"
                htmlFor={keyId}
                hint={
                  keyStoredForProvider
                    ? `A key ending ${settings.key_hint} is saved. Leave blank to keep it.`
                    : "Stored encrypted, never shown again, used only for this workspace's jobs."
                }
                link={
                  providerInfo?.key_url
                    ? { href: providerInfo.key_url, text: `Get a ${providerInfo.label} key` }
                    : undefined
                }
              >
                <div className="relative">
                  <input
                    id={keyId}
                    type={showKey ? "text" : "password"}
                    value={apiKey}
                    onChange={(event) => setApiKey(event.target.value)}
                    placeholder={
                      keyStoredForProvider
                        ? `••••••••${settings.key_hint?.slice(1)}`
                        : (providerInfo?.key_placeholder ?? "sk-…")
                    }
                    autoComplete="off"
                    spellCheck={false}
                    className={`${fieldClass} pr-16 font-mono`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey((shown) => !shown)}
                    aria-pressed={showKey}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-sm px-2 py-0.5 text-xs text-muted transition-colors duration-150 hover:bg-surface-raised hover:text-foreground"
                  >
                    {showKey ? "Hide" : "Show"}
                  </button>
                </div>
              </Field>

              <p className="text-xs leading-relaxed text-muted">
                Web search stays at one query every two seconds on every tier. DuckDuckGo limits by
                IP address, and every workspace searches from the same server, so a faster pace for
                one would get searches blocked for all.
              </p>
            </div>
          </div>
        </div>

        {error && <Notice tone="err">{error}</Notice>}
        {saved && !error && <Notice tone="ok">{saved}</Notice>}

        <div className="flex flex-wrap items-center gap-3 border-t border-hairline pt-4">
          <button
            type="button"
            onClick={() => void save()}
            disabled={saving || unchanged || (mode === "premium" && !apiKey.trim() && !keyStoredForProvider)}
            className={buttonClass("primary")}
          >
            {saving && <Spinner />}
            {saving
              ? mode === "premium"
                ? `Verifying with ${providerInfo?.label ?? "provider"}`
                : "Saving"
              : mode === "premium"
                ? "Save and verify"
                : "Save"}
          </button>
          {saving && mode === "premium" && (
            <span className="text-xs text-muted">Sending one test question…</span>
          )}
          {settings.key_hint && (
            <button
              type="button"
              onClick={() => void removeKey()}
              disabled={saving}
              className={`${buttonClass("danger", "sm")} ml-auto`}
            >
              Remove saved key
            </button>
          )}
        </div>
      </div>
    </Window>
  );
}

function Spinner() {
  return (
    <span
      aria-hidden="true"
      className="h-3 w-3 animate-spin rounded-full border-[1.5px] border-current border-r-transparent"
    />
  );
}

/** A radio card. The ring fills and the dot pops in when chosen. */
function TierCard({
  selected,
  onSelect,
  disabled = false,
  title,
  badge,
  lines,
}: {
  selected: boolean;
  onSelect: () => void;
  disabled?: boolean;
  title: string;
  badge: string;
  lines: string[];
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      disabled={disabled}
      onClick={onSelect}
      className={[
        "flex flex-col gap-2 rounded-lg border p-4 text-left",
        "transition-[border-color,background-color,box-shadow,transform] duration-300 ease-[var(--ease-out)] active:scale-[0.99]",
        selected
          ? "border-accent-border bg-accent-dim/25 shadow-[0_0_0_3px_var(--accent-glow)]"
          : "border-hairline hover:border-hairline-strong hover:bg-surface-subtle",
        disabled ? "cursor-not-allowed opacity-40" : "",
      ].join(" ")}
    >
      <span className="flex items-center gap-2.5">
        <span
          aria-hidden="true"
          className={`flex h-4 w-4 items-center justify-center rounded-full border transition-colors duration-300 ${
            selected ? "border-accent" : "border-hairline-strong"
          }`}
        >
          {selected && <span className="h-2 w-2 animate-pop rounded-full bg-accent" />}
        </span>
        <span className="text-sm font-medium text-foreground">{title}</span>
        <span className="ml-auto">
          <Tag tone={selected ? "accent" : "neutral"}>{badge}</Tag>
        </span>
      </span>
      {lines.map((line) => (
        <span key={line} className="pl-[26px] text-xs leading-relaxed text-muted">
          {line}
        </span>
      ))}
    </button>
  );
}

/** Provider choice as a segmented control; the highlight slides between them. */
function ProviderPicker({
  providers,
  value,
  onChange,
}: {
  providers: AiSettings["providers"];
  value: PremiumProvider;
  onChange: (id: PremiumProvider) => void;
}) {
  const { container, style } = useIndicator<HTMLDivElement>(value);
  return (
    <div
      ref={container}
      role="radiogroup"
      aria-label="Provider"
      className="relative inline-flex w-fit max-w-full flex-wrap rounded-lg border border-hairline bg-surface p-0.5"
    >
      <span
        aria-hidden="true"
        className="absolute inset-y-0.5 left-0 rounded-md bg-accent-dim"
        style={style}
      />
      {providers.map((option) => {
        const active = value === option.id;
        return (
          <button
            key={option.id}
            data-key={option.id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(option.id)}
            className={`relative z-10 flex h-7 items-center gap-1.5 rounded-md px-3 text-xs transition-colors duration-150 ${
              active ? "text-accent" : "text-muted hover:text-foreground"
            }`}
          >
            {option.label}
            {option.free && (
              <span className={`font-mono text-[10px] uppercase ${active ? "" : "text-faint"}`}>free</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

function Field({
  label,
  htmlFor,
  hint,
  link,
  children,
}: {
  label: string;
  htmlFor?: string;
  hint?: string;
  link?: { href: string; text: string };
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={htmlFor} className="label">
        {label}
      </label>
      {children}
      {(hint || link) && (
        <p className="text-xs text-muted">
          {hint}
          {link && (
            <>
              {hint ? " " : ""}
              <a href={link.href} target="_blank" rel="noreferrer" className="link">
                {link.text} ↗
              </a>
            </>
          )}
        </p>
      )}
    </div>
  );
}
