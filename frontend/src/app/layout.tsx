import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";

import { NavLinks } from "@/components/NavLinks";
import { SystemStatusPill } from "@/components/SystemStatusPill";

import "./globals.css";

// Sans for reading, mono for data (see globals.css).
const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Entity Resolution Platform",
  description: "Multi-tenant entity resolution and record linkage (ER-PLATFORM-001).",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} min-h-screen`}>
        <header className="sticky top-0 z-20 border-b border-hairline bg-background/80 backdrop-blur-md">
          <div className="mx-auto flex h-12 max-w-6xl items-center gap-6 px-6">
            <Link href="/dashboard" className="group flex items-center gap-2.5">
              <Logo />
              <span className="text-sm font-semibold tracking-tight text-foreground">ER_OS</span>
            </Link>
            <NavLinks />
            <span className="ml-auto flex items-center gap-3">
              <SystemStatusPill />
              {/* Honest about what the session is: there is no login because
                  there is no identity provider yet (see lib/auth/dev-token.ts). */}
              <span className="hidden font-mono text-xs text-faint sm:inline">dev@tenant</span>
            </span>
          </div>
        </header>

        <main className="mx-auto max-w-6xl px-6 py-10">{children}</main>
      </body>
    </html>
  );
}

/** Two records resolving into one: two rings that overlap, one in the accent. */
function Logo() {
  return (
    <svg aria-hidden="true" viewBox="0 0 20 20" className="h-5 w-5">
      <circle
        cx="7.5"
        cy="10"
        r="5"
        fill="none"
        stroke="var(--muted)"
        strokeWidth="1.5"
        className="transition-transform duration-300 ease-[var(--ease-out)] group-hover:translate-x-[1.5px]"
      />
      <circle
        cx="12.5"
        cy="10"
        r="5"
        fill="none"
        stroke="var(--accent)"
        strokeWidth="1.5"
        className="transition-transform duration-300 ease-[var(--ease-out)] group-hover:-translate-x-[1.5px]"
      />
    </svg>
  );
}
