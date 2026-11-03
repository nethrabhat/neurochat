# NeuroChat

A context-aware chatbot backed by **NeuroLinkDB** — a custom hybrid database engine built from scratch as a database-systems learning project.

---

## Architecture overview

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Application logic | Python 3.11+ |
| LLM provider | Groq API |
| Embeddings | Lightweight local deterministic / mock embeddings |
| Custom database | NeuroLinkDB (4096-byte pages, LRU Buffer Pool, WAL crash recovery) |
| Vector Indexing | Pure-Python Spherical K-Means + IVFFlat Approximate Nearest Neighbor |
| Tests | pytest (241 passing offline tests) |
| Config | Environment variables + `.env` |

---

## Current status — Phase 8 complete (All Phases Implemented)

NeuroChat is complete across all planned 8 phases of the PRD:

- **Phase 1: LLM Scaffold & UI**
  - Streamlit chat interface with graceful error handling
  - Pluggable LLM provider interface (`BaseLLMProvider`)
  - Groq API wrapper (`GroqProvider`) with SDK integration markers
  - Chat service with conversation-history management and context windowing
  - Configuration loaded from environment variables

- **Phase 2: NeuroLinkDB Storage Engine**
  - Custom binary page format (4096 bytes: 32-byte header with magic bytes `NLDB`, version, page IDs, record counts, free offsets, CRC32 checksums)
  - Variable-length record serialisation (key + JSON value, little-endian struct)
  - Persistent collections stored as chained page files (`.nldb`)
  - Full CRUD operations: `insert`, `get`, `get_all`, `scan`, `delete`, `count`
  - Automatic page allocation and corruption detection (magic bytes, CRC32, truncated files)

- **Phase 3: Conversation Persistence**
  - `StoredMessage` records containing conversation ID, message ID, role, content, timestamp, and metadata
  - `ConversationStore` persistence layer backed by NeuroLinkDB collections
  - Multi-conversation thread management in Streamlit sidebar (new conversation, switch conversation, delete thread)
  - Full state persistence across server restarts

- **Phase 4: Embeddings & Exact Vector Search**
  - Abstract `BaseEmbeddingProvider` and deterministic offline `MockEmbeddingProvider`
  - Pure-Python vector mathematics (cosine similarity, dot product, L2 norm, normalization, euclidean distance)
  - `VectorStore` engine storing embeddings, dimensions, and metadata in NeuroLinkDB
  - Exhaustive exact k-NN search with configurable `top_k`, similarity `threshold`, and relational filters
  - 100% offline, GPU-free, and FAISS/Chroma-free

- **Phase 5: Context-Aware Retrieval**
  - Intent detection heuristics (`should_retrieve`): filters trivial greetings, conversational fillers, and acknowledgments
  - Similarity threshold filtering (`min_threshold=0.25`) and prompt bounding (`max_context_chars=1200`)
  - De-duplication against current chat history window
  - Memory badges in Streamlit UI with expandable relevance explanations and user toggle

- **Phase 6: Hybrid Queries & Custom IVFFlat Index**
  - Pure-Python spherical K-Means clustering with deterministic K-Means++ centroid initialization
  - Custom IVFFlat (Inverted File Flat) partitioning index that scales query throughput by probing top `nprobe` cells
  - Persistent on-disk index layout in NeuroLinkDB collections (`.nldb`) with automated rebuild capability
  - Relational hybrid query engine supporting exact key-value filters, timestamp bounds (`min_timestamp`, `max_timestamp`), and custom lambda predicates
  - Evaluation metrics suite: Recall@K calculation against exact k-NN ground truth, latency speedup, and storage overhead measurement

- **Phase 7: Buffer Pool Manager & Write-Ahead Log (WAL)**
  - `BufferPoolManager` with configurable frame capacity, LRU replacement policy, and pin-count protection
  - Dirty-page tracking to ensure pages are flushed to disk before eviction
  - `WriteAheadLog` with 4-byte magic `NLWL`, monotonic LSNs, CRC32 record checksums, and torn-write protection
  - REDO crash recovery protocol restoring uncommitted in-memory writes upon database re-open
  - Seamless integration into `Collection` via `use_buffer_pool` and `use_wal` flags

- **Phase 8: Document Ingestion & Benchmarking**
  - `TextSplitter` supporting character chunking with configurable overlap (`chunk_size=512`, `overlap=64`) and paragraph-aware splitting
  - `DocumentStore` for persisting, embedding, and retrieving external document chunks (.txt and .pdf)
  - Streamlit sidebar UI for file upload, chunk ingestion, listing, and deletion
  - Automated benchmark suite measuring storage throughput, scan latency, exact k-NN vs IVFFlat Recall@K, and query speedups

- **Phase 8+: Slotted-Page Architecture & In-Place Record Updates**
  - Dual-ended slotted page layout: dynamic slot directory `(offset: uint16, length: uint16)` growing forward from byte 32, payload bytes growing backward from byte 4096.
  - In-place updates via `Collection.update(key, new_value)` preserving original `record_id`.
  - Reusable tombstone slots eliminating directory bloat on new inserts.
  - Page defragmentation via `Page.compact()` and collection-wide `Collection.vacuum()` reclaiming deleted record space.
  - Backward compatibility with legacy V1 sequential pages.

---

## Performance Benchmarks

The following are **real measured benchmark numbers** generated by running `python -m neurochat.benchmarks.run_all`:

### 1. Storage Engine Performance (500 records)

| Metric | Plain Collection | With Buffer Pool (BPM) |
|:---|---:|---:|
| **Insert throughput** | 45 records/sec | **48 records/sec** |
| **Scan latency** | 4.06 ms | 5.14 ms |

### 2. Vector Search Performance (1,000 vectors, dim=64, k=10, 50 queries)

| Method | Recall@10 | Avg Latency (ms) | Speedup vs Exact |
|:---|:---:|---:|:---:|
| **Exact Brute-Force** | **1.0000** | 14.403 ms | 1.0x (baseline) |
| **IVFFlat (`nprobe=1`)** | 0.3640 | **1.958 ms** | **7.4x faster** |
| **IVFFlat (`nprobe=2`)** | 0.5600 | 3.915 ms | **3.7x faster** |
| **IVFFlat (`nprobe=4`)** | 0.7800 | 8.604 ms | **1.7x faster** |

*Note: As `nprobe` approaches `nlist=8`, Recall@K reaches 1.00 while trading off latency, validating the approximate nearest neighbor tradeoff curve.*

---

## Setup & Running

### 1. Prerequisites

- Python 3.11+
- Virtual environment recommended

### 2. Install dependencies

```bash
cd neurochat
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:
```env
GROQ_API_KEY=gsk_your_real_key_here
MODEL_NAME=llama3-8b-8192
DB_DIR=./data/neurolinkdb
```

### 4. Run the full automated test suite (100% offline)

```bash
pytest neurochat/tests/ -v
```

All 226 tests run without network access, external services, or GPU dependencies.

### 5. Run the benchmarks

```bash
python -m neurochat.benchmarks.run_all
```

### 6. Start the Streamlit application

```bash
streamlit run neurochat/app.py
```

---

## Project structure

```
neurochat/
├── app.py                      # Streamlit UI with multi-conversation & document upload
├── requirements.txt            # Project dependencies
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
├── benchmarks/                 # Performance benchmarking suite (Phase 8)
│   ├── __init__.py
│   ├── storage_benchmark.py    # Insert throughput & scan latency benchmarks
│   ├── vector_benchmark.py     # Exact k-NN vs IVFFlat Recall@K & latency benchmarks
│   └── run_all.py              # CLI entry point running all benchmarks
├── config/
│   ├── __init__.py
│   └── settings.py             # Settings (API keys, models, paths)
├── embeddings/                 # Vector embeddings & similarity math (Phase 4)
│   ├── __init__.py
│   ├── base.py                 # BaseEmbeddingProvider & MockEmbeddingProvider
│   └── similarity.py           # Pure-Python cosine_similarity, dot_product, l2_norm
├── ingestion/                  # Document ingestion & chunking (Phase 8)
│   ├── __init__.py
│   ├── splitter.py             # TextSplitter (character windowing & paragraph splitting)
│   └── document_store.py       # DocumentStore (chunk persistence, embedding & retrieval)
├── llm/
│   ├── __init__.py
│   ├── base.py                 # BaseLLMProvider, Message, ChatResponse
│   └── groq_client.py          # Groq LLM API wrapper
├── neurolinkdb/                # Custom binary storage engine (Phases 2, 6, 7)
│   ├── __init__.py             # Exports NeuroLinkDB, Collection, Record, BufferPool, WAL
│   ├── constants.py            # Page sizing, headers, flags, magic bytes
│   ├── page.py                 # PageHeader & 4096-byte Page binary layout + CRC32
│   ├── record.py               # Variable-length record serialization
│   ├── collection.py           # Chained page collection (.nldb) with BPM & WAL hooks
│   ├── database.py             # Database handle & collection directory registry
│   ├── buffer_pool.py          # BufferPoolManager with LRU eviction & dirty tracking (Phase 7)
│   ├── wal.py                  # WriteAheadLog with REDO crash recovery & checksums (Phase 7)
│   └── index/                  # Vector indexing engine (Phase 6)
│       ├── __init__.py
│       ├── kmeans.py           # Spherical K-Means with K-Means++ seeding
│       ├── ivfflat.py          # IVFFlat approximate index (build, search, save, load)
│       └── evaluator.py        # Recall@K and latency performance evaluation
├── services/                   # Application services (Phases 1, 3, 5)
│   ├── __init__.py
│   ├── chat_service.py         # Multi-turn conversation orchestration & persistence
│   └── retrieval_service.py    # ContextRetriever: intent heuristics & prompt injection
├── storage/                    # Persistence abstractions (Phases 3, 4, 6)
│   ├── __init__.py
│   ├── conversation_store.py   # StoredMessage & ConversationStore
│   └── vector_store.py         # VectorStore: exact k-NN, hybrid filters, IVFFlat integration
└── tests/                      # 226 unit & integration tests (100% offline passing)
    ├── __init__.py
    ├── test_settings.py        # Settings validation (12 tests)
    ├── test_chat_service.py     # Conversation turns & windowing (12 tests)
    ├── test_neurolinkdb.py     # Binary pages, CRC32, collections, corruptions (50 tests)
    ├── test_conversation_store.py # Conversation storage & resume (14 tests)
    ├── test_embeddings.py      # Vector similarity & mock embeddings (31 tests)
    ├── test_context_retrieval.py # Intent detection & prompt injection (26 tests)
    ├── test_ivfflat_index.py   # K-Means, IVFFlat, Recall@K, hybrid queries (16 tests)
    ├── test_buffer_pool_wal.py # LRU eviction, dirty flush, WAL REDO recovery (13 tests)
    ├── test_document_ingestion.py # TextSplitter, DocumentStore, search (39 tests)
    ├── test_benchmarks.py      # Storage & vector benchmark validation (13 tests)
    └── test_slotted_page.py    # Slotted-page layout, updates, and compaction (15 tests)
```

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Inspect & plan | ✅ Done |
| 1 | LLM scaffold + Streamlit UI | ✅ Done |
| 2 | NeuroLinkDB storage (binary page format) | ✅ Done |
| 3 | Conversation persistence in NeuroLinkDB | ✅ Done |
| 4 | Embeddings + exact vector search | ✅ Done |
| 5 | Context-aware retrieval | ✅ Done |
| 6 | Hybrid queries + custom IVFFlat index | ✅ Done |
| 7 | Buffer pool + Write-Ahead Log (WAL) | ✅ Done |
| 8 | Document ingestion + benchmarks | ✅ Done |

---

## Known limitations & Future work

- **Variable-length record overflow:** Individual records are capped at `PAGE_SIZE - HEADER_SIZE - RECORD_HEADER_SIZE` (approx 4047 bytes). Multi-page overflow chains are a planned future enhancement.
- **Single-process concurrency:** Concurrency control is file-lock-free and designed for single-process desktop applications. Multi-writer transactions require a lock manager.
- **Pure-Python numeric calculations:** Vector mathematics and K-Means clustering are implemented in pure Python to eliminate compiled C/GPU dependencies, which limits scalability beyond ~100k vectors compared to BLAS-accelerated libraries.
- **PDF parsing:** PDF extraction requires optional `pdfminer.six`. If omitted, plain text `.txt` files are supported natively with zero third-party parsing dependencies.

---

## License

MIT
