import { ApiError } from "@/lib/api/client";

/**
 * A sentence for the user. The server's own `detail` when it sent a string
 * (this API writes them for people: "Enter your Anthropic API key."), the
 * generic per-status wording otherwise.
 */
export function messageOf(error: unknown): string {
  if (error instanceof ApiError) {
    return typeof error.detail === "string" && error.detail ? error.detail : error.userMessage;
  }
  return error instanceof Error ? error.message : "Something went wrong.";
}
