"""
app.py
──────
NeuroChat — Streamlit entry point.

Run with:
    streamlit run neurochat/app.py

Environment variables are read from a .env file if python-dotenv is
installed, otherwise from the shell environment.
"""

from __future__ import annotations

import uuid
import streamlit as st

# ── optional dotenv support ──────────────────────────────────────────────────
try:
    from pathlib import Path
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv()  # reads .env in the working directory, if present
    load_dotenv(Path(__file__).parent / ".env")  # reads .env in package directory
except ImportError:
    pass  # python-dotenv not yet installed; env vars must be set manually

# ── project imports ──────────────────────────────────────────────────────────
from neurochat.config.settings import settings, MissingAPIKeyError
from neurochat.embeddings.base import MockEmbeddingProvider
from neurochat.llm.groq_client import GroqProvider
from neurochat.neurolinkdb.database import NeuroLinkDB
from neurochat.ingestion.document_store import DocumentStore
from neurochat.services.chat_service import ChatService
from neurochat.storage.conversation_store import ConversationStore
from neurochat.storage.vector_store import VectorStore


# ── page config (must be the first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title=settings.app_title,
    page_icon="🧠",
    layout="centered",
)


# ── session-state initialisation ─────────────────────────────────────────────
def _switch_conversation(new_conv_id: str) -> None:
    """Helper to switch active conversation thread."""
    provider = GroqProvider(settings)
    st.session_state["current_conversation_id"] = new_conv_id
    st.session_state["chat_service"] = ChatService(
        provider,
        conversation_store=st.session_state["conv_store"],
        conversation_id=new_conv_id,
    )
    st.session_state["display_history"] = [
        {"role": m.role, "content": m.content}
        for m in st.session_state["chat_service"].history
    ]


def _init_session() -> None:
    """Create session-state keys on first run."""
    if "db" not in st.session_state:
        st.session_state["db"] = NeuroLinkDB(settings.db_dir)

    if "embedder" not in st.session_state:
        st.session_state["embedder"] = MockEmbeddingProvider(dimension=64)

    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = VectorStore(
            st.session_state["db"],
            collection_name="chat_vectors",
            dimension=64,
            model_name="mock-embed-v1",
        )

    if "conv_store" not in st.session_state:
        st.session_state["conv_store"] = ConversationStore(
            st.session_state["db"],
            vector_store=st.session_state["vector_store"],
            embedding_provider=st.session_state["embedder"],
        )

    if "current_conversation_id" not in st.session_state:
        convs = st.session_state["conv_store"].list_conversations()
        if convs:
            st.session_state["current_conversation_id"] = convs[0]["conversation_id"]
        else:
            st.session_state["current_conversation_id"] = str(uuid.uuid4())

    if "chat_service" not in st.session_state:
        provider = GroqProvider(settings)
        st.session_state["chat_service"] = ChatService(
            provider,
            conversation_store=st.session_state["conv_store"],
            conversation_id=st.session_state["current_conversation_id"],
        )

    if "doc_store" not in st.session_state:
        st.session_state["doc_store"] = DocumentStore(settings.db_dir)

    if "display_history" not in st.session_state:
        st.session_state["display_history"] = [
            {"role": m.role, "content": m.content}
            for m in st.session_state["chat_service"].history
        ]


_init_session()
chat_service: ChatService = st.session_state["chat_service"]
conv_store: ConversationStore = st.session_state["conv_store"]
current_conv_id: str = st.session_state["current_conversation_id"]


# ── header ────────────────────────────────────────────────────────────────────
st.title(f"🧠 {settings.app_title}")
st.caption(
    f"Powered by Groq · model: `{settings.model_name}` · session: `{current_conv_id[:8]}…`"
)

if not settings.is_groq_configured():
    st.warning(
        "⚠️  **GROQ_API_KEY is not set.**  "
        "Copy `.env.example` to `.env`, add your key, then restart the app.  "
        "You can still browse the UI, but sending messages will fail.",
        icon="🔑",
    )

st.divider()

# ── conversation display ──────────────────────────────────────────────────────
for turn in st.session_state["display_history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("retrieved"):
            with st.expander(
                f"🔍 Retrieved Context ({len(turn['retrieved'])} memories)",
                expanded=False,
            ):
                for mem in turn["retrieved"]:
                    st.caption(f"• **{mem['role']}** ({mem['score']}% relevance): {mem['content']}")

# ── input handling ────────────────────────────────────────────────────
if user_input := st.chat_input("Type your message…"):
    # Show user bubble immediately
    st.session_state["display_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the LLM
    retrieved_summary = []
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = chat_service.send(user_input)
                reply = response.content
                if response.retrieved_context:
                    for item in response.retrieved_context:
                        retrieved_summary.append({
                            "role": item.message.role,
                            "score": int(round(item.score * 100)),
                            "content": item.message.content,
                        })
            except MissingAPIKeyError as exc:
                reply = f"❌ **Configuration error:** {exc}"
            except RuntimeError as exc:
                reply = f"❌ **Runtime error:** {exc}"
            except Exception as exc:  # noqa: BLE001
                reply = f"❌ **Unexpected error:** {exc}"

        st.markdown(reply)
        if retrieved_summary:
            with st.expander(
                f"🔍 Retrieved Context ({len(retrieved_summary)} memories)",
                expanded=False,
            ):
                for mem in retrieved_summary:
                    st.caption(f"• **{mem['role']}** ({mem['score']}% relevance): {mem['content']}")

    st.session_state["display_history"].append(
        {"role": "assistant", "content": reply, "retrieved": retrieved_summary}
    )

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("NeuroChat")
    st.markdown(
        """
        A context-aware chatbot backed by **NeuroLinkDB** (custom DB engine).

        **Status:** Phase 8 Complete (Production Ready)  
        **Provider:** Groq  
        **Storage:** NeuroLinkDB with Buffer Pool & WAL  
        **Vector Retrieval:** Exact k-NN & Custom IVFFlat Index  
        """
    )
    st.divider()

    if chat_service.retriever:
        retrieval_active = st.toggle(
            "🧠 Semantic Memory Retrieval",
            value=chat_service.retriever.enabled,
            help="Selectively search and inject relevant past conversation context into the prompt.",
        )
        chat_service.retriever.enabled = retrieval_active
        st.divider()

    # ── Document Ingestion (Phase 8) ──────────────────────────────────────────
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload text or PDF document",
        type=["txt", "pdf"],
        help="Chunks text and embeds for semantic search across documents.",
    )
    doc_store: DocumentStore = st.session_state["doc_store"]

    if uploaded_file is not None:
        file_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
        if not st.session_state.get(file_key, False):
            with st.spinner("Ingesting document…"):
                try:
                    if uploaded_file.name.lower().endswith(".txt"):
                        content = uploaded_file.read().decode("utf-8", errors="replace")
                        records = doc_store.ingest_text(
                            content,
                            filename=uploaded_file.name,
                            embedder=st.session_state.get("embedder"),
                            vector_store=st.session_state.get("vector_store"),
                        )
                    else:
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        records = doc_store.ingest_file(
                            tmp_path,
                            embedder=st.session_state.get("embedder"),
                            vector_store=st.session_state.get("vector_store"),
                        )
                    st.session_state[file_key] = True
                    st.success(f"Ingested {len(records)} chunks from {uploaded_file.name}!")
                except Exception as exc:
                    st.error(f"Failed to ingest: {exc}")

    doc_count = doc_store.count()
    docs_list = doc_store.list_documents()
    st.caption(f"📚 {doc_count} chunk(s) across {len(docs_list)} document(s)")
    if docs_list:
        with st.expander("Manage Ingested Documents"):
            for d in docs_list:
                col1, col2 = st.columns([3, 1])
                col1.caption(f"• **{d['filename']}** ({d['chunk_count']} chunks)")
                if col2.button("🗑️", key=f"del_doc_{d['filename']}", help="Delete chunks"):
                    doc_store.delete_document(d["filename"])
                    st.rerun()

    st.divider()

    if st.button("➕ New conversation", use_container_width=True):
        new_id = str(uuid.uuid4())
        _switch_conversation(new_id)
        st.rerun()

    if st.button("🗑️ Delete this conversation", use_container_width=True):
        conv_store.delete_conversation(current_conv_id)
        new_id = str(uuid.uuid4())
        _switch_conversation(new_id)
        st.rerun()

    st.divider()
    st.subheader("💬 Past Conversations")
    saved_convs = conv_store.list_conversations()

    if not saved_convs:
        st.caption("No saved conversations yet.")
    else:
        for c in saved_convs:
            cid = c["conversation_id"]
            short_id = cid[:8]
            count = c["message_count"]
            preview = c.get("last_message", "")[:28]
            label = f"📝 {short_id} ({count} msgs)\n{preview}..."
            is_active = (cid == current_conv_id)
            if st.button(
                label,
                key=f"conv_{cid}",
                use_container_width=True,
                disabled=is_active,
            ):
                _switch_conversation(cid)
                st.rerun()

    st.divider()
    st.caption("© NeuroChat — Phase 8 Complete")
