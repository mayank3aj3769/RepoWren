export type Repository = {
  id: number;
  name: string;
  root_path: string;
  is_active: boolean;
};

export type ServerStatus = {
  api: "ok";
  inference: "not_loaded" | "loading" | "ready" | "error";
  model: string;
};

export type SearchPlan = {
  embedding_query: string;
  lexical_queries: string[];
  cross_questions: string[];
};

export type SearchMatch = {
  path: string;
  start_line: number;
  end_line: number;
  text: string;
  score: number;
};

export type GraphSearchMatch = Omit<SearchMatch, "score"> & {
  relation: string;
  detail: string | null;
  seed_path: string;
  seed_start_line: number;
  seed_end_line: number;
  depth: number;
};

export type SearchResponse = {
  query_plan: SearchPlan;
  matches: SearchMatch[];
  graph_matches: GraphSearchMatch[];
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type StreamEvent =
  | { type: "token"; text: string }
  | { type: "status"; message: string }
  | ({ type: "retrieval" } & SearchResponse)
  | { type: "error"; message: string }
  | { type: "done" };

const API_ROOT = "/api/repowren";

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as {
      detail?: string | Array<{ msg?: string }>;
    };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg ?? "Invalid request").join("; ");
    }
  } catch {
    // The fallback below is enough when an upstream proxy returns plain text.
  }
  return `${response.status} ${response.statusText}`.trim();
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()) as T;
}

export const api = {
  status: () => request<ServerStatus>("/status"),
  repositories: () => request<Repository[]>("/v1/repositories"),
  registerRepository: (path: string, name?: string) =>
    request<Repository>("/v1/repositories", {
      method: "POST",
      body: JSON.stringify({ path, name: name || undefined }),
    }),
  selectRepository: (id: number) =>
    request<Repository>(`/v1/repositories/${id}/select`, { method: "POST" }),
};

export async function streamChat(
  messages: ChatMessage[],
  repositoryId: number | null,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const path = repositoryId
    ? `/v1/repositories/${repositoryId}/chat/stream`
    : "/v1/chat/stream";
  const response = await fetch(`${API_ROOT}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    cache: "no-store",
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  if (!response.body) throw new Error("The API returned no response stream.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";

  const emitLines = (final = false) => {
    const lines = buffered.split("\n");
    const tail = lines.pop() ?? "";
    buffered = final ? "" : tail;
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) onEvent(JSON.parse(trimmed) as StreamEvent);
    }
    if (final && tail.trim()) onEvent(JSON.parse(tail.trim()) as StreamEvent);
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffered += decoder.decode(value, { stream: true });
    emitLines();
  }
  buffered += decoder.decode();
  emitLines(true);
}
