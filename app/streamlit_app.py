"""
Streamlit Web Interface for Multimodal AI Assistant.

Provides interactive document management, multimodal chat, source citation badges,
collapsible context inspection, and image reasoning.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Any
import streamlit as st
from PIL import Image

from mmassistant.config import AppConfig, get_config
from mmassistant.pipeline import MixedResult, MultimodalPipeline, RAGResult, VisionResult
from mmassistant.router import InputRouter, InputType

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("mmassistant.ui")


@st.cache_resource
def get_pipeline() -> MultimodalPipeline:
    """Initialize and cache the global MultimodalPipeline."""
    return MultimodalPipeline()


def initialize_session_state() -> None:
    """Initialize chat history, active document filters, and attachment states."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your **Multimodal AI Assistant**. "
                    "You can upload documents (PDF, DOCX, TXT, MD) to query with factual citations, "
                    "or attach images (charts, diagrams, screenshots) for visual reasoning and mixed cross-modal analysis."
                ),
                "citations": [],
                "sources": [],
                "image": None,
            }
        ]
    if "selected_doc_ids" not in st.session_state:
        st.session_state.selected_doc_ids = set()
    if "attached_image_bytes" not in st.session_state:
        st.session_state.attached_image_bytes = None
    if "attached_image_name" not in st.session_state:
        st.session_state.attached_image_name = None


def render_sidebar(pipeline: MultimodalPipeline, cfg: AppConfig) -> None:
    """Render configuration, document uploader, and library manager in the sidebar."""
    with st.sidebar:
        st.title("⚙️ System Control")

        # 1. API Status Banner
        if not cfg.has_api_key:
            st.error("⚠️ `GOOGLE_API_KEY` is not set in `.env`.")
        else:
            st.success(f"🟢 Connected: `{cfg.gemini_model_id}`")

        with st.expander("ℹ️ Architecture & Settings", expanded=False):
            st.caption(f"**Embedding Model:** `{cfg.embedding_model_id}` (384-dim, Local)")
            st.caption(f"**Vector Store:** ChromaDB (`storage/chroma`)")
            st.caption(f"**Top-K Chunks:** {cfg.top_k_chunks}")
            st.caption(f"**Similarity Threshold:** {cfg.similarity_threshold}")
            st.caption(f"**OCR Engine:** RapidOCR (ONNX Runtime, Local)")

        st.divider()

        # 2. Document Upload Section
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, or MD",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Files are indexed into ChromaDB with SHA-256 deduplication.",
        )

        col1, col2 = st.columns(2)
        with col1:
            extract_figures = st.checkbox(
                "Figure Captioning",
                value=cfg.enable_figure_extraction,
                help="Generates Gemini Vision descriptions of diagrams inside PDFs during ingestion.",
            )
        with col2:
            force_reindex = st.checkbox("Force Re-index", value=False)

        if uploaded_files:
            if st.button("📥 Index Uploaded Documents", use_container_width=True, type="primary"):
                for uploaded_file in uploaded_files:
                    save_path = cfg.upload_dir / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                        try:
                            res = pipeline.ingest_document(
                                save_path,
                                force_reindex=force_reindex,
                                extract_figures=extract_figures,
                            )
                            if res["status"] == "indexed":
                                st.success(f"Indexed **{res['filename']}** ({res['chunks']} chunks)")
                            elif res["status"] == "skipped":
                                st.info(f"**{res['filename']}** was already indexed.")
                        except Exception as exc:
                            st.error(f"Failed to index '{uploaded_file.name}': {exc}")
                st.rerun()

        st.divider()

        # 3. Document Library & Active Filter Manager
        st.subheader("📚 Document Library")
        docs = pipeline.list_documents()

        if not docs:
            st.caption("No documents indexed yet. Upload a document above or use sample files.")
        else:
            total_chunks = sum(d["chunk_count"] for d in docs)
            st.caption(f"**Total Indexed:** {len(docs)} documents | {total_chunks} chunks")

            # Selection / filter controls
            col_sel_all, col_clear_all = st.columns(2)
            if col_sel_all.button("Select All", use_container_width=True):
                st.session_state.selected_doc_ids = {d["doc_id"] for d in docs}
                st.rerun()
            if col_clear_all.button("Clear Filter", use_container_width=True):
                st.session_state.selected_doc_ids = set()
                st.rerun()

            for doc in docs:
                doc_id = doc["doc_id"]
                is_selected = doc_id in st.session_state.selected_doc_ids

                row_col1, row_col2 = st.columns([0.8, 0.2])
                with row_col1:
                    checked = st.checkbox(
                        f"**{doc['filename']}** ({doc['chunk_count']} chunks)",
                        value=is_selected,
                        key=f"chk_{doc_id}",
                    )
                    if checked and doc_id not in st.session_state.selected_doc_ids:
                        st.session_state.selected_doc_ids.add(doc_id)
                        st.rerun()
                    elif not checked and doc_id in st.session_state.selected_doc_ids:
                        st.session_state.selected_doc_ids.remove(doc_id)
                        st.rerun()

                with row_col2:
                    if st.button("🗑️", key=f"del_{doc_id}", help="Delete document from ChromaDB"):
                        pipeline.delete_document(doc_id)
                        st.session_state.selected_doc_ids.discard(doc_id)
                        st.rerun()

        st.divider()

        # 4. Image Attachment Section
        st.subheader("🖼️ Attach Image")
        img_upload = st.file_uploader(
            "Attach image for Vision / Mixed query",
            type=["png", "jpg", "jpeg", "webp"],
            key="img_attachment_uploader",
        )

        if img_upload is not None:
            st.session_state.attached_image_bytes = img_upload.getvalue()
            st.session_state.attached_image_name = img_upload.name
            st.image(
                st.session_state.attached_image_bytes,
                caption=f"Attached: {img_upload.name}",
                use_container_width=True,
            )
            if st.button("❌ Remove Image", use_container_width=True):
                st.session_state.attached_image_bytes = None
                st.session_state.attached_image_name = None
                st.rerun()

        mixed_mode = st.toggle(
            "🔀 Mixed Mode",
            value=True,
            help="When ON with an image attached, cross-references the image against indexed documents.",
        )
        return mixed_mode


def render_chat_history() -> None:
    """Render historical conversation messages with citations and expandable sources."""
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("image"):
                st.image(msg["image"], width=300, caption="User Attached Image")
            st.markdown(msg["content"])

            # Render citation pill badges if present
            citations = msg.get("citations", [])
            if citations:
                citation_html = " ".join(
                    f"<span style='background-color:#E0F2FE; color:#0369A1; padding:3px 8px; border-radius:12px; font-size:12px; font-weight:600; margin-right:4px;'>{c}</span>"
                    for c in citations
                )
                st.markdown(citation_html, unsafe_allow_html=True)

            # Render collapsible source inspection panel
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"🔍 Retrieved Source Context ({len(sources)} chunks)", expanded=False):
                    for s_idx, src in enumerate(sources, 1):
                        st.markdown(
                            f"**[{s_idx}] {src.get('citation', 'Source')}** — *Similarity: {src.get('similarity', 0.0):.3f} | Type: `{src.get('chunk_type', 'text')}`*"
                        )
                        st.code(src.get("text", ""), language="markdown")
                        st.divider()


def main() -> None:
    """Main application loop."""
    cfg = get_config()
    pipeline = get_pipeline()
    initialize_session_state()

    mixed_mode = render_sidebar(pipeline, cfg)

    # Main Page Header
    st.title("🤖 Multimodal AI Assistant")
    st.caption(
        "Intelligent Document Understanding (RAG) & Computer Vision powered by BGE-small, ChromaDB, RapidOCR, and Gemini."
    )

    # Render Chat History
    render_chat_history()

    # Chat Input
    query = st.chat_input("Ask a question about your documents or attached image...")

    if query:
        attached_img = st.session_state.attached_image_bytes
        attached_name = st.session_state.attached_image_name

        # Append User Message to UI
        user_msg = {
            "role": "user",
            "content": query,
            "image": attached_img,
        }
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            if attached_img:
                st.image(attached_img, width=300, caption=f"Attached: {attached_name}")
            st.markdown(query)

        # Route input
        docs = pipeline.list_documents()
        has_docs = len(docs) > 0
        input_type = InputRouter.route(
            query=query,
            image_bytes=attached_img,
            has_indexed_docs=has_docs,
            is_mixed_mode=mixed_mode,
        )

        filter_doc_ids = list(st.session_state.selected_doc_ids) if st.session_state.selected_doc_ids else None

        # Execute Pipeline based on route
        with st.chat_message("assistant"):
            with st.spinner("Analyzing and generating grounded response..."):
                try:
                    if input_type == InputType.IMAGE_QUERY and attached_img:
                        v_res: VisionResult = pipeline.ask_image(attached_img, query)
                        ans_content = v_res.answer
                        citations = []
                        sources_data = []

                    elif input_type == InputType.MIXED_QUERY and attached_img:
                        m_res: MixedResult = pipeline.ask_mixed(query, attached_img, doc_ids=filter_doc_ids)
                        ans_content = m_res.answer
                        citations = m_res.citations
                        sources_data = [
                            {
                                "citation": s.citation,
                                "similarity": s.similarity,
                                "chunk_type": s.chunk_type,
                                "text": s.text,
                            }
                            for s in m_res.sources
                        ]

                    else:  # TEXT_DOC_QUERY
                        r_res: RAGResult = pipeline.ask(query, doc_ids=filter_doc_ids)
                        ans_content = r_res.answer
                        citations = r_res.citations
                        sources_data = [
                            {
                                "citation": s.citation,
                                "similarity": s.similarity,
                                "chunk_type": s.chunk_type,
                                "text": s.text,
                            }
                            for s in r_res.sources
                        ]

                    st.markdown(ans_content)

                    if citations:
                        citation_html = " ".join(
                            f"<span style='background-color:#E0F2FE; color:#0369A1; padding:3px 8px; border-radius:12px; font-size:12px; font-weight:600; margin-right:4px;'>{c}</span>"
                            for c in citations
                        )
                        st.markdown(citation_html, unsafe_allow_html=True)

                    if sources_data:
                        with st.expander(f"🔍 Retrieved Source Context ({len(sources_data)} chunks)", expanded=False):
                            for s_idx, src in enumerate(sources_data, 1):
                                st.markdown(
                                    f"**[{s_idx}] {src['citation']}** — *Similarity: {src['similarity']:.3f} | Type: `{src['chunk_type']}`*"
                                )
                                st.code(src["text"], language="markdown")
                                st.divider()

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans_content,
                        "citations": citations,
                        "sources": sources_data,
                    })

                except Exception as exc:
                    st.error(f"Generation error: {exc}")
                    logger.exception("Chat generation failed: %s", exc)


if __name__ == "__main__":
    main()
