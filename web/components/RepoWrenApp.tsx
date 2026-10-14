"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  ChatMessage,
  Repository,
  SearchResponse,
  ServerStatus,
  streamChat,
} from "@/lib/api";

type UiMessage = ChatMessage & { id: string; retrieval?: SearchResponse };

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
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
  const messagesEnd = useRef<HTMLDivElement>(null);

  const selectedRepository = useMemo(
    () => repositories.find((repository) => repository.id === selectedId) ?? null,
    [repositories, selectedId],
  );
  const latestTrace = useMemo(
    () => [...messages].reverse().find((message) => message.retrieval)?.retrieval ?? null,
    [messages],
  );

  async function refreshRepositories(preferredId?: number) {
    const items = await api.repositories();
    setRepositories(items);
    setSelectedId((current) => {
      if (preferredId && items.some((item) => item.id === preferredId)) return preferredId;
      if (current && items.some((item) => item.id === current)) return current;
      return items.find((item) => item.is_active)?.id ?? items[0]?.id ?? null;
    });
  }

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const [repositoryItems, statusResponse] = await Promise.all([
          api.repositories(),
          api.status(),
        ]);
        if (!active) return;
        setRepositories(repositoryItems);
        setSelectedId(
          repositoryItems.find((item) => item.is_active)?.id ?? repositoryItems[0]?.id ?? null,
        );
        setStatus(statusResponse);
      } catch (error) {
        if (active) setNotice(error instanceof Error ? error.message : "Could not reach RepoWren.");
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

  function clearChat() {
    if (chatBusy) return;
    setMessages([]);
    setChatStatus("");
    setNotice(null);
  }

  async function selectRepository(repositoryId: number | null) {
    setNotice(null);
    if (repositoryId === null) {
      setSelectedId(null);
      setMessages([]);
      return;
    }
    try {
      await api.selectRepository(repositoryId);
      await refreshRepositories(repositoryId);
      setMessages([]);
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
      const repository = await api.registerRepository(repositoryPath.trim(), repositoryName.trim());
      await api.selectRepository(repository.id);
      await refreshRepositories(repository.id);
      setMessages([]);
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
      ...messages.map(({ role, content: messageContent }) => ({ role, content: messageContent })),
      { role: "user" as const, content },
    ].slice(-32);

    setDraft("");
    setChatBusy(true);
    setChatStatus(selectedId === null ? "Thinking…" : "Planning GraphRAG retrieval…");
    setNotice(null);
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "" },
    ]);

    try {
      await streamChat(outbound, selectedId, (streamEvent) => {
        if (streamEvent.type === "token") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + streamEvent.text }
                : message,
            ),
          );
        } else if (streamEvent.type === "retrieval") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    retrieval: {
                      query_plan: streamEvent.query_plan,
                      matches: streamEvent.matches,
                      graph_matches: streamEvent.graph_matches,
                    },
                  }
                : message,
            ),
          );
          setChatStatus("Answering from ranked and graph-linked code…");
        } else if (streamEvent.type === "status") {
          setChatStatus(streamEvent.message);
        } else if (streamEvent.type === "error") {
          throw new Error(streamEvent.message);
        }
      });
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

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <WrenMark />
          <div><strong>RepoWren</strong><span>Local code companion</span></div>
        </div>
        <section className="repository-section">
          <div className="section-heading">
            <span>Workspaces</span>
            <button className="icon-button" type="button" onClick={() => setShowAddRepository((value) => !value)} aria-label="Register repository" title="Register repository">+</button>
          </div>
          {showAddRepository && (
            <form className="repository-form" onSubmit={registerRepository}>
              <label>Repository path<input value={repositoryPath} onChange={(event) => setRepositoryPath(event.target.value)} placeholder="C:\\code\\project" autoFocus /></label>
              <label>Display name <span>(optional)</span><input value={repositoryName} onChange={(event) => setRepositoryName(event.target.value)} placeholder="My project" /></label>
              <button className="primary-button compact" disabled={registering}>{registering ? "Indexing…" : "Add repository"}</button>
            </form>
          )}
          <div className="repository-list">
            <button type="button" className={`repository-item ${selectedId === null ? "selected" : ""}`} onClick={() => void selectRepository(null)}>
              <span className="repo-avatar general">AI</span><span><strong>General chat</strong><small>No repository context</small></span>
            </button>
            {loadingRepositories ? <div className="sidebar-placeholder">Loading repositories…</div> : repositories.map((repository) => (
              <button type="button" className={`repository-item ${selectedId === repository.id ? "selected" : ""}`} key={repository.id} onClick={() => void selectRepository(repository.id)} title={repository.root_path}>
                <span className="repo-avatar">{repository.name.slice(0, 2).toUpperCase()}</span><span><strong>{repository.name}</strong><small>{repository.root_path}</small></span>
              </button>
            ))}
          </div>
        </section>
        <div className="connection-card"><span className={`status-dot ${status?.inference ?? "not_loaded"}`} /><div><strong>{status?.inference === "ready" ? "Model ready" : "Model warming up"}</strong><span>{status?.model?.split("/").pop() ?? "Checking local API…"}</span></div></div>
      </aside>

      <section className="workspace">
        <header className="workspace-header"><div><span className="eyebrow">{selectedRepository ? "Repository chat" : "Assistant"}</span><h1>{selectedRepository?.name ?? "General chat"}</h1><p>{selectedRepository?.root_path ?? "Talk directly to the local model."}</p></div></header>
        {notice && <div className="notice" role="alert"><span>{notice}</span><button type="button" onClick={() => setNotice(null)} aria-label="Dismiss">×</button></div>}
        <section className="chat-view">
          {selectedRepository && (
            <aside className="context-panel">
              <div className="context-panel-header"><div><span className="eyebrow">GraphRAG trace</span><strong>{latestTrace ? `${latestTrace.matches.length} ranked seeds · ${latestTrace.graph_matches.length} linked chunks` : "Ask a question to inspect retrieval"}</strong></div><button className="text-button" type="button" onClick={clearChat} disabled={chatBusy || !messages.length}>Clear chat</button></div>
              {latestTrace ? (
                <div className="context-list">
                  <details className="context-card"><summary><span>Query plan</span><small>{latestTrace.query_plan.cross_questions.length} cross-questions</small></summary><pre>{[latestTrace.query_plan.embedding_query, ...latestTrace.query_plan.cross_questions].join("\n")}</pre></details>
                  {latestTrace.matches.map((match) => <details className="context-card" key={`seed-${match.path}-${match.start_line}`}><summary><span>{match.path}</span><small>seed · lines {match.start_line}-{match.end_line}</small></summary><pre>{match.text}</pre></details>)}
                  {latestTrace.graph_matches.map((match) => <details className="context-card" key={`graph-${match.path}-${match.start_line}`}><summary><span>{match.path}</span><small>depth {match.depth} · {match.relation}{match.detail ? ` (${match.detail})` : ""}</small></summary><pre>{match.text}</pre></details>)}
                </div>
              ) : <p className="context-empty">Planning, hybrid retrieval, reciprocal-rank fusion, and relationship expansion now run inside the chat workflow.</p>}
            </aside>
          )}
          <div className="conversation" aria-live="polite">
            {messages.length === 0 ? <EmptyState title={selectedRepository ? `Ask about ${selectedRepository.name}` : "Ask RepoWren"} body={selectedRepository ? "Chat plans multiple searches, fuses their rankings, and follows code relationships before answering." : "Select a repository for source-aware answers, or chat with the local model."} /> : messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}><span className="message-label">{message.role === "user" ? "You" : "RepoWren"}</span><div>{message.content || (chatBusy ? <span className="typing">● ● ●</span> : "")}</div></article>
            ))}
            {chatStatus && <div className="chat-status">{chatStatus}</div>}
            <div ref={messagesEnd} />
          </div>
          <form className="composer" onSubmit={sendMessage}>
            <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder={selectedRepository ? "Ask a question about this repository…" : "Message the local model…"} rows={2} />
            <div className="composer-footer"><span>Enter to send · Shift+Enter for a new line</span><div className="composer-actions">{messages.length > 0 && <button className="text-button" type="button" onClick={clearChat}>Clear chat</button>}<button className="send-button" disabled={!draft.trim() || chatBusy} aria-label="Send"><ArrowIcon /></button></div></div>
          </form>
        </section>
      </section>
    </main>
  );
}
