import signs from "../../shared/signs.json";

type LibraryCategory = "all" | "letter" | "digit" | "word";
type ModelContext = {
  registerTool(
    tool: {
      name: string;
      description: string;
      inputSchema: object;
      annotations: { readOnlyHint: boolean };
      execute(input: unknown): unknown;
    },
    options: { signal: AbortSignal },
  ): void | Promise<void>;
};

// Optional browser capability; ordinary navigation does not depend on it.
export function registerLibraryTool(
  showLibrary: (query: string, category: LibraryCategory) => void,
) {
  const context = (document as Document & { modelContext?: ModelContext })
    .modelContext;
  if (!context?.registerTool) return;
  const lifecycle = new AbortController();
  try {
    void Promise.resolve(
      context.registerTool(
        {
          name: "show_sign_library",
          description:
            "Open the NSL label library and apply a search and category filter. Does not activate a camera or perform recognition.",
          inputSchema: {
            type: "object",
            properties: {
              query: { type: "string", maxLength: 100 },
              category: {
                type: "string",
                enum: ["all", "letter", "digit", "word"],
              },
            },
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false },
          execute(input: unknown) {
            if (lifecycle.signal.aborted)
              throw new Error("Workspace is closed.");
            if (!input || typeof input !== "object" || Array.isArray(input))
              throw new Error("Expected a filter object.");
            const args = input as Record<string, unknown>;
            if (
              Object.keys(args).some(
                (key) => key !== "query" && key !== "category",
              )
            )
              throw new Error("Unknown filter.");
            const query = args.query ?? "";
            const category = args.category ?? "all";
            if (typeof query !== "string" || query.length > 100)
              throw new Error(
                "Query must be a string of at most 100 characters.",
              );
            if (
              typeof category !== "string" ||
              !["all", "letter", "digit", "word"].includes(category)
            )
              throw new Error("Unknown category.");
            showLibrary(query, category as LibraryCategory);
            return {
              page: "Sign library",
              total: signs.filter(
                (sign) =>
                  (category === "all" || category === sign.category) &&
                  (sign.label.toLowerCase().includes(query.toLowerCase()) ||
                    sign.id.includes(query.toLowerCase())),
              ).length,
            };
          },
        },
        { signal: lifecycle.signal },
      ),
    ).catch(() => {
      /* Optional integration unavailable. */
    });
  } catch {
    /* Keep ordinary browser controls available. */
  }
  return () => lifecycle.abort();
}
