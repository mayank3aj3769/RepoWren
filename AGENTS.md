# RepoWren contributor instructions

These instructions apply to the entire repository.

## Project direction

- Keep RepoWren local-first, understandable, and safe for an average developer.
- Use vLLM as the only chat-inference backend. Run vLLM through Docker Compose;
  do not add a manual WSL or host-Python vLLM installation path.
- Run the FastAPI API and terminal client from the Windows Python virtual
  environment. Run PostgreSQL with pgvector and vLLM in Docker.
- Do not add PowerShell scripts. Document ordinary cross-platform commands or
  direct executable commands instead.
- Prefer the simplest correct implementation. Do not add caching, incremental
  indexing, vector indexes, parsers, or abstraction layers without a measured
  need and explicit approval.

## Current repository retrieval design

- Use `sentence-transformers/all-MiniLM-L6-v2` on CPU for 384-dimensional
  embeddings so vLLM retains the GPU.
- Split allowed UTF-8 files into fixed, non-overlapping 20-line chunks.
- Store chunk text, JSONB metadata (`path`, `start_line`, `end_line`), and the
  embedding in PostgreSQL `VECTOR(384)` columns.
- Re-registering a repository performs a complete chunk rebuild. Keep this
  behavior simple until incremental indexing is explicitly requested.
- Perform exact pgvector cosine search and use the five closest chunks for chat
  context. Do not reintroduce keyword extraction or literal-search loops.

## Repository safety

- Treat the registered Git root as the filesystem security boundary.
- Resolve paths before access and reject absolute paths, `..` traversal,
  symlink escapes, missing files, oversized files, binaries, and invalid UTF-8.
- Apply the same exclusion rules to listing, reading, and indexing.
- Never read or embed `.env`, `.env.*`, `.git`, credentials, model weights,
  virtual environments, dependency trees, caches, or generated build output.
- Respect the repository root `.gitignore`. Keep built-in exclusions together
  in the clearly named list in `repowren/repositories/service.py`.
- Repository-aware features remain read-only unless a later approved phase
  explicitly adds editing, command execution, or Git mutation.

## Code and documentation style

- Use small typed functions, dataclasses, and narrow protocols at boundaries.
- Keep SQL in migrations or the PostgreSQL store; keep filesystem rules in the
  repository service and vLLM protocol details in the inference adapter.
- Add clear comments to non-obvious validation, search, context, and storage
  code. Explain behind-the-scenes behavior in simple language.
- For complicated transformations, include a small concrete object example,
  such as a chunk with its path and line-range metadata.
- Update `README.md`, `docs/architecture.md`, `docs/repo-guide.md`, and
  `docs/endpoint-flow-guide.md` when public behavior or architecture changes.
- Never place tokens, credentials, local paths containing secrets, model
  caches, or `.env` contents in source control or PR text.

## Testing and validation

- Keep normal tests deterministic and offline. Inject fake inference,
  embedding, and repository-store implementations instead of requiring Docker,
  a GPU, PostgreSQL, or model downloads.
- Add or update tests for validation boundaries, exclusions, chunk metadata,
  API response shapes, context construction, CLI formatting, and regressions.
- Before a PR, run:
  - `.venv\Scripts\python.exe -m pytest -q`
  - `.venv\Scripts\python.exe -m pip check`
  - `docker compose config --quiet`
  - `git diff --check`
- When persistence SQL changes, also verify migrations and one real pgvector
  insert/search round-trip against the Docker database when it is available.

## Git and PR practices

- Preserve unrelated and pre-existing user changes. Never use destructive
  commands such as `git reset --hard` or discard a dirty worktree.
- Use a focused `codex/` branch, review the complete diff, and keep commits and
  PR descriptions explicit about migrations, behavior changes, and validation.
- Do not commit, push, open a PR, merge, or otherwise change remote Git state
  without the user's explicit approval.
