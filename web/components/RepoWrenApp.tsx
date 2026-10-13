"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  ChatMessage,
  FileMetadata,
  Repository,
  SearchMatch,
  SearchResponse,
  ServerStatus,
  streamChat,
} from "@/lib/api";

type View = "chat" | "search" | "files";
type UiMessage = ChatMessage & { id: string };

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function formatBytes(value: number): string {
  if (value < 1_024) return `${value} B`;
  if (value < 1_048_576) return `${(value / 1_024).toFixed(1)} KB`;
  return `${(value / 1_048_576).toFixed(1)} MB`;
}

function WrenMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 44 44" className="brand-mark">
      <path d="M9 26c6-1 9-6 12-13 3 7 8 11 15 12-4 2-7 5-9 10l-6-6-5 6-1-8-6-1Z" />
      <circle cx="28.5" cy="21" r="1.5" />
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 20 20">
      <path d="M4 10h11M11 6l4 4-4 4" />
    </svg>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="empty-state">
      <WrenMark />
      <h2>{title}</h2>
      <p>{body}</p>
    </div>
  );
}

export function RepoWrenApp() {
  const [repositories, setRepositories] = useState<Repository[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [view, setView] = useState<View>("chat");
  const [notice, setNotice] = useState<string | null>(null);
  const [loadingRepositories, setLoadingRepositories] = useState(true);

  const [showAddRepository, setShowAddRepository] = useState(false);
  const [repositoryPath, setRepositoryPath] = useState("");
  const [repositoryName, setRepositoryName] = useState("");
  const [registering, setRegistering] = useState(false);

  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [chatStatus, setChatStatus] = useState("");
  const [contextMatches, setContextMatches] = useState<SearchMatch[]>([]);
  const [contextQuery, setContextQuery] = useState("");
  const messagesEnd = useRef<HTMLDivElement>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [searchBusy, setSearchBusy] = useState(false);

  const [fileFilter, setFileFilter] = useState("");
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [filesBusy, setFilesBusy] = useState(false);
  const [openFile, setOpenFile] = useState<{ path: string; content: string } | null>(
    null,
  );

  const selectedRepository = useMemo(
    () => repositories.find((repository) => repository.id === selectedId) ?? null,
    [repositories, selectedId],
  );

  async function refreshRepositories(preferredId?: number) {
    const items = await api.repositories();
    setRepositories(items);
    setSelectedId((current) => {
      if (preferredId && items.some((item) => item.id === preferredId)) {
        return preferredId;
      }
      if (current && items.some((item) => item.id === current)) return current;
      return items.find((item) => item.is_active)?.id ?? items[0]?.id ?? null;
    });
  }

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const [repositoriesResponse, statusResponse] = await Promise.all([
          api.repositories(),
          api.status(),
        ]);
        if (!active) return;
        setRepositories(repositoriesResponse);
        setSelectedId(
          repositoriesResponse.find((item) => item.is_active)?.id ??
            repositoriesResponse[0]?.id ??
            null,
        );
        setStatus(statusResponse);
      } catch (error) {
        if (active) {
          setNotice(error instanceof Error ? error.message : "Could not reach RepoWren.");
        }
      } finally {
        if (active) setLoadingRepositories(false);
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, chatStatus]);

  useEffect(() => {
    if (view === "files" && selectedId !== null && files.length === 0) {
      void loadFiles("");
    }
    // The selected repository reset already clears files, so this only runs once per view.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, selectedId]);

  function resetRepositoryWorkspace() {
    setMessages([]);
    setContextMatches([]);
    setContextQuery("");
    setSearchResult(null);
    setFiles([]);
    setOpenFile(null);
    setFileFilter("");
  }

  function clearChatContext() {
    if (chatBusy) return;
    setMessages([]);
    setContextMatches([]);
    setContextQuery("");
    setChatStatus("");
    setNotice(null);
  }

  async function selectRepository(repositoryId: number | null) {
    setNotice(null);
    if (repositoryId === null) {
      setSelectedId(null);
      resetRepositoryWorkspace();
      setView("chat");
      return;
    }
    try {
      await api.selectRepository(repositoryId);
      await refreshRepositories(repositoryId);
      resetRepositoryWorkspace();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not select repository.");
    }
  }

  async function registerRepository(event: FormEvent) {
    event.preventDefault();
    if (!repositoryPath.trim() || registering) return;
    setRegistering(true);
    setNotice(null);
    try {
      const repository = await api.registerRepository(
        repositoryPath.trim(),
        repositoryName.trim(),
      );
      await api.selectRepository(repository.id);
      await refreshRepositories(repository.id);
      resetRepositoryWorkspace();
      setRepositoryPath("");
      setRepositoryName("");
      setShowAddRepository(false);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not register repository.");
    } finally {
      setRegistering(false);
    }
  }

  async function sendMessage(event?: FormEvent) {
    event?.preventDefault();
    const content = draft.trim();
    if (!content || chatBusy) return;

    const userMessage: UiMessage = { id: makeId(), role: "user", content };
    const assistantId = makeId();
    const outbound = [
      ...messages.map(({ role, content: messageContent }) => ({
        role,
        content: messageContent,
      })),
      { role: "user" as const, content },
    ].slice(-32);

    setDraft("");
    setChatBusy(true);
    setChatStatus("Thinking with local context…");
    setNotice(null);
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "" },
    ]);

    try {
      if (selectedId !== null) {
        const context = await api.search(selectedId, content, 5);
        setContextQuery(content);
        setContextMatches(context.matches);
      } else {
        setContextQuery("");
        setContextMatches([]);
      }
      await streamChat(outbound, selectedId, (streamEvent) => {
        if (streamEvent.type === "token") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + streamEvent.text }
                : message,
            ),
          );
        } else if (streamEvent.type === "status") {
          setChatStatus(streamEvent.message);
        } else if (streamEvent.type === "error") {
          throw new Error(streamEvent.message);
        }
      });
      setChatStatus("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Chat request failed.";
      setNotice(message);
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId && !item.content
            ? { ...item, content: `Request failed: ${message}` }
            : item,
        ),
      );
    } finally {
      setChatBusy(false);
      setChatStatus("");
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  async function runSearch(event: FormEvent) {
    event.preventDefault();
    if (selectedId === null || !searchQuery.trim() || searchBusy) return;
    setSearchBusy(true);
    setNotice(null);
    try {
      setSearchResult(await api.search(selectedId, searchQuery.trim(), 10));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Search failed.");
    } finally {
      setSearchBusy(false);
    }
  }

  async function loadFiles(query = fileFilter) {
    if (selectedId === null || filesBusy) return;
    setFilesBusy(true);
    setNotice(null);
    try {
      setFiles(await api.files(selectedId, query.trim()));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not list files.");
    } finally {
      setFilesBusy(false);
    }
  }

  async function readFile(path: string) {
    if (selectedId === null) return;
    setNotice(null);
    try {
      setOpenFile(await api.file(selectedId, path));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not read file.");
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <WrenMark />
          <div>
            <strong>RepoWren</strong>
            <span>Local code companion</span>
          </div>
        </div>

        <section className="repository-section">
          <div className="section-heading">
            <span>Workspaces</span>
            <button
              className="icon-button"
              type="button"
              onClick={() => setShowAddRepository((value) => !value)}
              aria-label="Register repository"
              title="Register repository"
            >
              +
            </button>
          </div>

          {showAddRepository && (
            <form className="repository-form" onSubmit={registerRepository}>
              <label>
                Repository path
                <input
                  value={repositoryPath}
                  onChange={(event) => setRepositoryPath(event.target.value)}
                  placeholder="C:\\code\\project"
                  autoFocus
                />
              </label>
              <label>
                Display name <span>(optional)</span>
                <input
                  value={repositoryName}
                  onChange={(event) => setRepositoryName(event.target.value)}
                  placeholder="My project"
                />
              </label>
              <button className="primary-button compact" disabled={registering}>
                {registering ? "Indexing…" : "Add repository"}
              </button>
            </form>
          )}

          <div className="repository-list">
            <button
              type="button"
              className={`repository-item ${selectedId === null ? "selected" : ""}`}
              onClick={() => void selectRepository(null)}
            >
              <span className="repo-avatar general">AI</span>
              <span>
                <strong>General chat</strong>
                <small>No repository context</small>
              </span>
            </button>

            {loadingRepositories ? (
              <div className="sidebar-placeholder">Loading repositories…</div>
            ) : (
              repositories.map((repository) => (
                <button
                  type="button"
                  className={`repository-item ${selectedId === repository.id ? "selected" : ""}`}
                  key={repository.id}
                  onClick={() => void selectRepository(repository.id)}
                  title={repository.root_path}
                >
                  <span className="repo-avatar">
                    {repository.name.slice(0, 2).toUpperCase()}
                  </span>
                  <span>
                    <strong>{repository.name}</strong>
                    <small>{repository.root_path}</small>
                  </span>
                </button>
              ))
            )}
          </div>
        </section>

        <div className="connection-card">
          <span className={`status-dot ${status?.inference ?? "not_loaded"}`} />
          <div>
            <strong>{status?.inference === "ready" ? "Model ready" : "Model warming up"}</strong>
            <span>{status?.model?.split("/").pop() ?? "Checking local API…"}</span>
          </div>
        </div>
      </aside>

      <section className="workspace">
        <header className="workspace-header">
          <div>
            <span className="eyebrow">{selectedRepository ? "Repository" : "Assistant"}</span>
            <h1>{selectedRepository?.name ?? "General chat"}</h1>
            <p>{selectedRepository?.root_path ?? "Talk directly to the local model."}</p>
          </div>
          <nav className="view-tabs" aria-label="Workspace views">
            {(["chat", "search", "files"] as View[]).map((item) => (
              <button
                key={item}
                type="button"
                disabled={item !== "chat" && selectedId === null}
                className={view === item ? "active" : ""}
                onClick={() => setView(item)}
              >
                {item[0].toUpperCase() + item.slice(1)}
              </button>
            ))}
          </nav>
        </header>

        {notice && (
          <div className="notice" role="alert">
            <span>{notice}</span>
            <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}

        {view === "chat" && (
          <section className="chat-view">
            {selectedRepository && (
              <aside className="context-panel">
                <div className="context-panel-header">
                  <div>
                    <span className="eyebrow">Source context</span>
                    <strong>
                      {contextMatches.length
                        ? `${contextMatches.length} chunks sent for “${contextQuery}”`
                        : "No context loaded yet"}
                    </strong>
                  </div>
                  <button
                    className="text-button"
                    type="button"
                    onClick={clearChatContext}
                    disabled={chatBusy || (!messages.length && !contextMatches.length)}
                  >
                    Clear context
                  </button>
                </div>
                {contextMatches.length > 0 ? (
                  <div className="context-list">
                    {contextMatches.map((match) => (
                      <details className="context-card" key={`${match.path}-${match.start_line}`}>
                        <summary>
                          <span>{match.path}</span>
                          <small>
                            lines {match.start_line}-{match.end_line} · {match.score.toFixed(3)}
                          </small>
                        </summary>
                        <pre>{match.text}</pre>
                      </details>
                    ))}
                  </div>
                ) : (
                  <p className="context-empty">
                    Ask a repository question to preview the five source chunks used for the answer.
                  </p>
                )}
              </aside>
            )}
            <div className="conversation" aria-live="polite">
              {messages.length === 0 ? (
                <EmptyState
                  title={selectedRepository ? `Ask about ${selectedRepository.name}` : "Ask RepoWren"}
                  body={
                    selectedRepository
                      ? "Answers use the five highest-ranked source chunks from hybrid search."
                      : "Select a repository for source-aware answers, or chat with the local model."
                  }
                />
              ) : (
                messages.map((message) => (
                  <article className={`message ${message.role}`} key={message.id}>
                    <span className="message-label">
                      {message.role === "user" ? "You" : "RepoWren"}
                    </span>
                    <div>{message.content || (chatBusy ? <span className="typing">● ● ●</span> : "")}</div>
                  </article>
                ))
              )}
              {chatStatus && <div className="chat-status">{chatStatus}</div>}
              <div ref={messagesEnd} />
            </div>

            <form className="composer" onSubmit={sendMessage}>
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder={
                  selectedRepository
                    ? "Ask a question about this repository…"
                    : "Message the local model…"
                }
                rows={2}
              />
              <div className="composer-footer">
                <span>Enter to send · Shift+Enter for a new line</span>
                <div className="composer-actions">
                  {messages.length > 0 && (
                    <button className="text-button" type="button" onClick={clearChatContext}>
                      Clear chat
                    </button>
                  )}
                  <button className="send-button" disabled={!draft.trim() || chatBusy} aria-label="Send">
                    <ArrowIcon />
                  </button>
                </div>
              </div>
            </form>
          </section>
        )}

        {view === "search" && selectedId !== null && (
          <section className="tool-view">
            <div className="tool-intro">
              <span className="eyebrow">Hybrid retrieval</span>
              <h2>Find the code behind an idea</h2>
              <p>The local model plans semantic and exact-text queries, then RRF merges both rankings.</p>
            </div>
            <form className="search-form" onSubmit={runSearch}>
              <input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Where are failed payments retried?"
                autoFocus
              />
              <button className="primary-button" disabled={!searchQuery.trim() || searchBusy}>
                {searchBusy ? "Searching…" : "Search code"}
              </button>
            </form>

            {searchResult && (
              <div className="search-output">
                <div className="query-plan">
                  <div>
                    <span>Embedding query</span>
                    <strong>{searchResult.query_plan.embedding_query}</strong>
                  </div>
                  <div>
                    <span>Exact-text queries</span>
                    <div className="query-chips">
                      {searchResult.query_plan.lexical_queries.map((query) => (
                        <code key={query}>{query}</code>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="result-heading">
                  <h3>Aggregate results</h3>
                  <span>{searchResult.matches.length} matches · RRF ranked</span>
                </div>
                <div className="result-list">
                  {searchResult.matches.map((match, index) => (
                    <details className="result-card" key={`${match.path}-${match.start_line}`} open={index < 2}>
                      <summary>
                        <span className="result-rank">{String(index + 1).padStart(2, "0")}</span>
                        <span className="result-location">
                          <strong>{match.path}</strong>
                          <small>lines {match.start_line}–{match.end_line}</small>
                        </span>
                        <code>{match.score.toFixed(4)}</code>
                      </summary>
                      <pre>{match.text}</pre>
                    </details>
                  ))}
                  {searchResult.matches.length === 0 && <p className="muted">No matches found.</p>}
                </div>
              </div>
            )}
          </section>
        )}

        {view === "files" && selectedId !== null && (
          <section className="files-view">
            <div className="file-browser">
              <div className="file-browser-header">
                <div>
                  <span className="eyebrow">Safe repository files</span>
                  <h2>Browse source</h2>
                </div>
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    void loadFiles();
                  }}
                >
                  <input
                    value={fileFilter}
                    onChange={(event) => setFileFilter(event.target.value)}
                    placeholder="Filter paths…"
                  />
                </form>
              </div>
              <div className="file-list">
                {filesBusy ? (
                  <p className="muted">Loading files…</p>
                ) : (
                  files.map((file) => (
                    <button
                      type="button"
                      key={file.path}
                      className={openFile?.path === file.path ? "active" : ""}
                      onClick={() => void readFile(file.path)}
                    >
                      <span>{file.path}</span>
                      <small>{formatBytes(file.size_bytes)}</small>
                    </button>
                  ))
                )}
              </div>
            </div>
            <div className="file-preview">
              {openFile ? (
                <>
                  <div className="file-preview-header">
                    <strong>{openFile.path}</strong>
                    <button type="button" onClick={() => setOpenFile(null)}>
                      Close
                    </button>
                  </div>
                  <pre>{openFile.content}</pre>
                </>
              ) : (
                <EmptyState title="Select a file" body="Choose an allowed UTF-8 file to read it here." />
              )}
            </div>
          </section>
        )}
      </section>
    </main>
  );
}
