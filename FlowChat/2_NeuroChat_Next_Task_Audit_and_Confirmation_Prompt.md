# NeuroChat — Current Implementation Audit & Next-Task Confirmation Prompt

Act as a senior software engineer and database systems mentor working on my existing **NeuroChat** project.

## Context

NeuroChat is a context-aware chatbot with:
- Streamlit frontend.
- Groq API for language-model responses.
- A custom database project named **NeuroLinkDB**.
- A planned hybrid architecture combining persistent record storage, relational metadata filtering, and vector similarity search.

We intend to implement NeuroLinkDB ourselves in Python. Do not use PostgreSQL, MySQL, MongoDB, SQLite, Chroma, or FAISS as replacements for our custom storage engine or custom vector index.

The planned roadmap is:
1. Inspect the existing application.
2. Implement custom persistent storage with a documented binary page format (target: 4096-byte pages).
3. Implement record and collection management.
4. Persist conversations and messages.
5. Add embedding storage and exact vector similarity search.
6. Add metadata filtering and a hybrid query interface.
7. Implement a custom K-Means/IVFFlat-style approximate vector index.
8. Later consider a buffer pool, WAL, and recovery.
9. Add benchmarks and optional document ingestion.

This is a roadmap, not proof that these features exist. Verify the repository.

---

# PART A — Read-only project audit

**Do not modify, create, delete, install, or overwrite anything during this audit.**

Inspect the current project and report its actual state.

## 1. Frontend and application
- Is the Streamlit frontend implemented?
- Which files implement it?
- What features currently work?
- How does it communicate with the backend?
- Is conversation state in memory or persisted?

## 2. Groq integration
- Is the Groq client implemented, and where?
- How are credentials configured? Do not print secrets.
- Is the model configurable?
- How are API errors and rate limits handled?
- Are there mocked tests?

## 3. NeuroLinkDB
Inspect the database code and classify each feature as:
- Implemented and verified
- Implemented but not verified
- Partially implemented
- Planned but not implemented
- Not found

Check:
- Database initialization and file format.
- Binary pages, page size, and page header.
- Page allocation and page read/write.
- Record serialization/deserialization.
- Collection/table management.
- Insert, get, update, delete, and scan.
- Persistence after closing and reopening.
- Conversation/message integration.
- Embedding storage.
- Exact vector search.
- Metadata filtering.
- K-Means and IVFFlat indexing.
- Buffer pool.
- WAL and recovery.

Only call something verified if code and actual test/execution evidence support that status.

## 4. Project quality and tests
- Inspect the project structure and dependency files.
- Identify existing tests and their coverage.
- Run only tests that are safe with the current environment and already-installed dependencies.
- Do not install missing dependencies or make network calls.
- Report exact commands and actual results.
- Identify important unfinished modules, integration issues, or duplicated logic.

---

# PART B — Status report

Return these sections:

## 1. Current project summary
Explain what NeuroChat can do right now.

## 2. Actual architecture
Show the architecture that exists in code, not the intended architecture. A Mermaid diagram is welcome. Distinguish implemented and planned components.

## 3. Implemented features
For each feature, include status, relevant file paths, what it does, and evidence/tests.

## 4. Incomplete and missing features
List partially implemented and unimplemented work.

## 5. Existing database specification
If database code exists, document its actual:
- File format and page size.
- Header fields and record layout.
- Serialization format.
- Collection layout.
- Page allocation strategy.
- Persistence behavior and limitations.

If a detail cannot be established from the code, say **not defined**. Do not invent specifications.

## 6. Test report
List tests discovered, tests executed, passed, failed, and not run (with reasons).

## 7. Risks and technical issues
Identify problems that should be resolved before adding features.

---

# PART C — Recommend exactly one next task

Based on the audit, recommend the single most logical next implementation task.

Prioritize:
1. Fix a critical issue preventing the existing app from working.
2. Complete an unfinished feature required for the current milestone.
3. If custom persistent storage is missing, build the NeuroLinkDB storage foundation.
4. If storage works, integrate conversation/message persistence.
5. If persistence works, implement embedding storage and exact vector search.
6. After exact search is correct, add metadata filtering and the custom approximate index.
7. Add buffer pool and WAL only after storage is stable.

For the recommended task, specify:
- Task name and why it is next.
- Exact objective and scope.
- Explicit exclusions.
- Expected behavior.
- Files likely to be created or modified.
- Database/API specifications that need decisions.
- Dependencies, if any.
- Tests and acceptance criteria.
- Risks, trade-offs, and implementation complexity.

Do not combine multiple major phases into one task.

---

# PART D — Ask for confirmation and missing specifications

**Stop after the audit and recommendation. Do not write or modify code yet.**

Present the proposed next task and ask me to confirm it.

Ask only questions that are necessary and cannot be answered from the repository. Consider:
- Must an existing database format be preserved?
- Is 4096-byte page size already implemented or still only a target?
- Is a slotted-page layout already present or should it be designed?
- How are record IDs and collection metadata represented?
- Are update/delete operations required in the next milestone?
- Should the next task be an independently tested database component or include chatbot integration?
- What unresolved design decisions did the audit reveal?

For each question:
1. Explain briefly why it matters.
2. Give a recommended default.
3. Let me accept the default or specify another choice.

Do not ask me to repeat information already clear from the repository.

End with a clear confirmation request:

**“I recommend implementing [specific next task]. Should I proceed with this scope and the proposed specifications?”**

Wait for my response. Do not begin implementation until I explicitly approve.

---

# Strict working rules

- Audit first; implement only after approval.
- Never assume a planned feature has been built.
- Never invent file paths, test results, database formats, or capabilities.
- Never replace NeuroLinkDB with an existing database.
- Do not unnecessarily change the working Groq integration or frontend.
- Do not install packages, download models, request API keys, or make network calls during this audit.
- Do not expose credentials, tokens, or private data.
- Preserve existing user data and files.
- Keep the implementation modular, educational, and testable.
- Explain database design decisions in beginner-friendly language.

**Begin with the read-only repository audit. Finish with the status report, one recommended next task, and questions for my confirmation.**
