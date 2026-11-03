# NeuroChat — Master Implementation Prompt

Act as a senior Python engineer and database systems mentor. Help implement **NeuroChat**, a context-aware chatbot backed by a custom hybrid database engine named **NeuroLinkDB**.

I am a student learning database systems and AI application development. Build the project in small, testable phases. Explain important design choices simply. Prioritize a working implementation over generating a large amount of untested code.

## Important: no external setup yet

**Do not perform external setup in this first pass.** Do not install packages, download models, request API keys, make network requests, or change system settings. First inspect the current project directory and produce a proposed implementation plan and file structure. Then implement the code scaffold for Phase 1 with configuration placeholders and mocked tests. We will handle installations, API keys, and other external setup later.

Do not delete or overwrite unrelated files. If an existing project is present, inspect and adapt it.

## Product goal

NeuroChat is a context-aware chatbot:
- Groq API will generate responses once credentials are configured later.
- A separate embedding model will encode messages and queries.
- NeuroLinkDB will store conversations and metadata using our own persistent storage.
- Relevant history will be retrieved through vector similarity and metadata filtering when useful.
- Exact vector search comes first; a custom IVFFlat-style index using K-Means comes later.

This is a database systems project. Do not substitute SQLite, PostgreSQL, Chroma, FAISS, or another existing database/vector-index engine for the core NeuroLinkDB storage or custom index.

## Architecture

- UI: Streamlit.
- Application logic: Python.
- LLM provider: Groq API, isolated behind an interface.
- Embeddings: separate lightweight local model, integrated in a later phase.
- Custom database: NeuroLinkDB.
- Testing: pytest.
- Configuration: environment variables and `.env.example`.

Do not assume a GPU exists.

## Phase 0 — Inspect and plan only

1. Inspect the working directory and existing files.
2. Identify the current OS and Python version if accessible without installing anything.
3. Check whether the project already has a virtual environment, Git repository, or dependency files.
4. Do not install dependencies or download anything.
5. Propose a minimal file structure and explain each file.
6. List prerequisites that we will set up later.
7. State assumptions and ask only essential questions.

## Phase 1 — Code scaffold, no external setup

Create a minimal, clean scaffold for the first chatbot milestone. Do not implement the entire database engine yet.

Suggested structure, adapted as needed:

```text
neurochat/
  app.py
  requirements.txt
  .env.example
  .gitignore
  README.md
  config/
    __init__.py
    settings.py
  llm/
    __init__.pycontim
    base.py
    groq_client.py
  services/
    __init__.py
    chat_service.py
  tests/
    test_chat_service.py
    test_settings.py
```

Do not create empty modules for features that have not been implemented. If the directory is not already named `neurochat`, do not rename it without asking.

Implement:
1. A simple Streamlit chat interface.
2. A small LLM provider interface.
3. A Groq client wrapper intended to use the official SDK, but do not install the SDK or make a live request now.
4. Configuration that reads `GROQ_API_KEY` and a configurable model name from environment variables.
5. Safe handling for a missing API key.
6. A chat service that can be tested with a mocked provider.
7. Unit tests that do not need internet access or a real Groq key.
8. A `.env.example` containing placeholders only.
9. A `.gitignore` excluding `.env`, virtual environments, caches, logs, local database files, and generated model/index files.
10. A README with future setup instructions clearly marked as not yet executed.

Never put a real API key in code. Do not hardcode a model name throughout the application. Keep provider-specific logic in `llm/groq_client.py`.

If the SDK API cannot be verified from locally available information, isolate the uncertain integration in the client wrapper and mark it for verification during setup. Do not invent successful execution.

## Future phases — plan only; do not implement yet

### Phase 2: NeuroLinkDB storage
- Documented custom binary page format targeting 4096-byte pages.
- Record serialization/deserialization.
- Persistent collections and basic insert/retrieve operations.
- Reopen database and verify persistence.
- Tests for multiple pages and corrupted data.

### Phase 3: Conversation persistence
- Conversation ID, message ID, role, content, timestamp, and optional embedding reference.
- Store and retrieve conversation history across restarts.

### Phase 4: Embeddings and exact vector search
- Separate embedding interface.
- Record embedding model identity and dimension.
- Validate vector dimensions.
- Implement exact cosine similarity.
- Add deterministic tests using synthetic vectors.

### Phase 5: Context-aware retrieval
- Decide when context retrieval is useful.
- Retrieve relevant records, apply filters, limit context size, and send query plus selected context to Groq.
- Store the assistant response.
- Use mocked LLM and embedding components in tests.

### Phase 6: Hybrid queries and custom IVFFlat
- Relational metadata filters combined with vector ranking.
- Custom K-Means clustering and IVFFlat-style approximate search.
- Persist centroids, assignments, and index metadata.
- Support index rebuilding.
- Compare recall@k, latency, build time, and storage overhead against exact search.

### Phase 7: Buffer pool and WAL
- Page cache and eviction policy.
- Dirty-page tracking.
- Write-ahead logging and documented recovery behavior.
- Tests for supported interrupted-write scenarios.
- Do not claim full ACID or production-grade crash safety without evidence.

### Phase 8: Optional document ingestion and benchmarks
- Text/PDF extraction, chunking, embeddings, source metadata, and retrieval.
- Benchmark storage and vector search using measured results.

## Engineering rules

- Implement only the current phase.
- Do not install dependencies, download models, call external APIs, or change machine configuration during this pass.
- Do not claim tests passed unless actually run.
- Keep code modular and reasonably simple.
- Use type hints and clear errors.
- Tests must not require internet access.
- Do not claim features are implemented when they are only planned.
- Keep a list of known limitations.
- Do not overwrite unrelated files.

## Required response after this pass

1. Summarize the environment you could inspect.
2. Show the final file tree.
3. Explain each implemented file's responsibility.
4. List files created or modified.
5. Report exactly which tests were run and their actual results.
6. List external setup to do later, including dependency installation, Groq API-key configuration, and model configuration.
7. Give the next command or action for the user, but do not execute external setup.

**Begin with Phase 0. Do not install or download anything.**
