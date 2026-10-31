"use client";

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

/*
  The stateful half of the motion system (the CSS half is in globals.css):
  numbers that count to their value, and a selection pill that slides between
  tabs instead of blinking from one to the next.
*/

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * A number that eases from wherever it was to `target` (from 0 on first
 * render). Polled counts glide instead of jumping, which is also what makes a
 * change noticeable.
 */
export function useCountUp(target: number, duration = 800): number {
  const [shown, setShown] = useState(0);
  const current = useRef(0);

  useEffect(() => {
    const start = current.current;
    if (start === target) return;
    if (prefersReducedMotion()) {
      current.current = target;
      setShown(target);
      return;
    }
    let frame = 0;
    const began = performance.now();
    const step = (now: number) => {
      const progress = Math.min(1, (now - began) / duration);
      const eased = 1 - Math.pow(1 - progress, 4);
      current.current = start + (target - start) * eased;
      setShown(current.current);
      if (progress < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);

  return shown;
}

/** A count-up number in the mono face, grouped for the locale. */
export function Num({ value, className = "" }: { value: number; className?: string }) {
  const shown = useCountUp(value);
  return (
    <span className={`font-mono tabular-nums ${className}`}>
      {Math.round(shown).toLocaleString()}
    </span>
  );
}

/**
 * Position of the item marked `data-key={active}` inside the returned
 * container, as a style for an absolutely-positioned indicator. The first
 * placement is instant; every later one slides.
 */
export function useIndicator<T extends HTMLElement>(active: string | null) {
  const container = useRef<T>(null);
  const [box, setBox] = useState<{ left: number; width: number } | null>(null);
  const [animate, setAnimate] = useState(false);

  useLayoutEffect(() => {
    const root = container.current;
    if (!root) return;
    const measure = () => {
      const item = active
        ? root.querySelector<HTMLElement>(`[data-key="${CSS.escape(active)}"]`)
        : null;
      setBox(item ? { left: item.offsetLeft, width: item.offsetWidth } : null);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    return () => observer.disconnect();
  }, [active]);

  useEffect(() => {
    if (!box || animate) return;
    const frame = requestAnimationFrame(() => setAnimate(true));
    return () => cancelAnimationFrame(frame);
  }, [box, animate]);

  const style: CSSProperties = box
    ? {
        width: box.width,
        transform: `translateX(${box.left}px)`,
        transition: animate
          ? "transform 0.35s var(--ease-out), width 0.35s var(--ease-out)"
          : "none",
      }
    : { opacity: 0 };

  return { container, style };
}

/**
 * Segmented tabs with a sliding selection pill. Real tabs for assistive
 * technology (tablist / tab / aria-selected), arrow keys move between them.
 */
export function Tabs<V extends string>({
  items,
  value,
  onChange,
  label,
}: {
  items: { value: V; label: ReactNode; count?: number }[];
  value: V;
  onChange: (value: V) => void;
  label: string;
}) {
  const { container, style } = useIndicator<HTMLDivElement>(value);

  const move = (direction: 1 | -1) => {
    const index = items.findIndex((item) => item.value === value);
    const next = items[(index + direction + items.length) % items.length];
    onChange(next.value);
    container.current?.querySelector<HTMLElement>(`[data-key="${CSS.escape(next.value)}"]`)?.focus();
  };

  return (
    <div
      ref={container}
      role="tablist"
      aria-label={label}
      className="relative inline-flex w-fit max-w-full flex-wrap rounded-lg border border-hairline bg-surface-subtle p-0.5"
      onKeyDown={(event) => {
        if (event.key === "ArrowRight") move(1);
        if (event.key === "ArrowLeft") move(-1);
      }}
    >
      <span
        aria-hidden="true"
        className="absolute inset-y-0.5 left-0 rounded-md border border-hairline bg-surface"
        style={style}
      />
      {items.map((item) => {
        const active = item.value === value;
        return (
          <button
            key={item.value}
            data-key={item.value}
            role="tab"
            type="button"
            tabIndex={active ? 0 : -1}
            aria-selected={active}
            onClick={() => onChange(item.value)}
            className={`relative z-10 flex h-7 items-center gap-2 rounded-md px-3 text-xs transition-colors duration-150 ${
              active ? "text-foreground" : "text-muted hover:text-foreground"
            }`}
          >
            {item.label}
            {item.count !== undefined && (
              <span className={`font-mono text-[11px] ${active ? "text-accent" : "text-faint"}`}>
                {item.count.toLocaleString()}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
