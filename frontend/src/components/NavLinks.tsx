"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useIndicator } from "./motion";

const NAV = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/review", label: "Review" },
  { href: "/settings", label: "Settings" },
];

/** Main navigation. The highlight slides to the current page. */
export function NavLinks() {
  const pathname = usePathname();
  const active =
    NAV.find((item) => pathname === item.href || pathname.startsWith(`${item.href}/`))?.href ??
    // A job's results page belongs to the dashboard.
    (pathname.startsWith("/jobs/") ? "/dashboard" : null);
  const { container, style } = useIndicator<HTMLElement>(active);

  return (
    <nav ref={container} aria-label="Main" className="relative flex items-center gap-0.5 text-sm">
      <span aria-hidden="true" className="absolute inset-y-0 left-0 rounded-md bg-surface-raised" style={style} />
      {NAV.map((item) => {
        const current = item.href === active;
        return (
          <Link
            key={item.href}
            data-key={item.href}
            href={item.href}
            aria-current={current ? "page" : undefined}
            className={`relative z-10 rounded-md px-2.5 py-1 transition-colors duration-150 ${
              current ? "text-foreground" : "text-muted hover:text-foreground"
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
