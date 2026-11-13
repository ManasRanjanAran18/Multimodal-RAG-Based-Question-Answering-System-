from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from process import process_input_folder
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_files"
TEXT_DIR = BASE_DIR / "text_files"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

for folder in (INPUT_DIR, TEXT_DIR, VECTOR_DB_DIR):
    folder.mkdir(parents=True, exist_ok=True)

FAISS_INDEX = VECTOR_DB_DIR / "faiss_index.bin"
FAISS_METADATA = VECTOR_DB_DIR / "metadata.json"

SUPPORTED_TYPES = [
    "pdf", "docx", "txt", "md",
    "mp3", "wav", "m4a", "flac",
    "mp4", "mkv", "avi", "mov",
]

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# SESSION STATE
# =========================================================

def init_state() -> None:
    defaults = {
        "theme": "Light",
        "messages": [],
        "rag": None,
        "last_process_report": None,
        "uploader_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# =========================================================
# THEME
# =========================================================

def apply_theme(theme: str) -> None:
    if theme == "Dark":
        c = {
            "bg": "#101827", "surface": "#182334", "surface2": "#243247",
            "text": "#F1F5F9", "muted": "#B8C4D4", "border": "#3A4A60",
            "input": "#202E42", "accent": "#438BE8",
            "button_text": "#F8FAFC",
        }
    else:
        c = {
            "bg": "#F5F7FB", "surface": "#FFFFFF", "surface2": "#E9EEF5",
            "text": "#172033", "muted": "#64748B", "border": "#DCE3EC",
            "input": "#FFFFFF", "accent": "#2563EB",
            "button_text": "#172033",
        }

    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: {"dark" if theme == "Dark" else "light"}; }}
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
            background: {c["bg"]} !important;
            color: {c["text"]} !important;
        }}
        [data-testid="stHeader"] {{ background: transparent !important; }}
        .block-container {{
            max-width: 1450px;
            padding-top: 0.65rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }}

        /* Text and headings */
        h1, h2, h3, h4, h5, h6,
        p, li, label, legend,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stText"],
        [data-testid="stWidgetLabel"] p {{
            color: {c["text"]} !important;
        }}
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        .stCaption, small {{
            color: {c["muted"]} !important;
        }}
        a {{ color: {c["accent"]} !important; }}
        hr {{ margin: 0.55rem 0 !important; border-color: {c["border"]} !important; }}

        /* Cards, upload area and expanders */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"],
        [data-testid="stChatMessage"],
        [data-testid="stFileUploader"] section {{
            background: {c["surface"]} !important;
            border-color: {c["border"]} !important;
            border-radius: 11px;
        }}
        [data-testid="stFileUploader"] section {{
            border-style: dashed !important;
            padding: 0.45rem 0.7rem;
        }}
        [data-testid="stFileUploader"] button {{
            background: {c["surface2"]} !important;
            color: {c["text"]} !important;
            border: 1px solid {c["border"]} !important;
        }}

        /* Text inputs and text areas */
        input, textarea,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div {{
            background: {c["input"]} !important;
            color: {c["text"]} !important;
            border-color: {c["border"]} !important;
            -webkit-text-fill-color: {c["text"]} !important;
        }}
        input::placeholder, textarea::placeholder {{
            color: {c["muted"]} !important;
            -webkit-text-fill-color: {c["muted"]} !important;
        }}

        /* All regular and download buttons: explicitly set both text and fill.
           This fixes the invisible labels seen in the previous screenshot. */
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button,
        div.stButton > button,
        div.stDownloadButton > button {{
            background: {c["surface2"]} !important;
            color: {c["button_text"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 8px !important;
            min-height: 2.15rem;
        }}
        div[data-testid="stButton"] > button *,
        div[data-testid="stDownloadButton"] > button *,
        div.stButton > button *,
        div.stDownloadButton > button * {{
            color: {c["button_text"]} !important;
            -webkit-text-fill-color: {c["button_text"]} !important;
        }}
        div[data-testid="stButton"] > button[kind="primary"],
        div.stButton > button[kind="primary"] {{
            background: {c["accent"]} !important;
            color: #FFFFFF !important;
            border-color: {c["accent"]} !important;
        }}
        div[data-testid="stButton"] > button[kind="primary"] *,
        div.stButton > button[kind="primary"] * {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}

        /* Horizontal theme selector */
        div[data-testid="stRadio"] [role="radiogroup"] {{
            display: flex;
            flex-direction: row;
            justify-content: flex-end;
            gap: 0.75rem;
        }}
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] label * {{
            color: {c["text"]} !important;
        }}
        div[data-testid="stRadio"] [role="radio"] {{
            border-color: {c["accent"]} !important;
        }}

        [data-testid="stStatusWidget"] {{ color: {c["text"]} !important; }}
        [data-testid="stMetric"] {{
            background: {c["surface"]} !important;
            color: {c["text"]} !important;
            padding: 0.5rem 0.7rem;
            border: 1px solid {c["border"]};
            border-radius: 9px;
        }}
        [data-testid="stChatInput"] {{ display: none; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # These selectors apply ONLY in Light mode. The existing Dark theme remains
    # exactly as it was.
    if theme == "Light":
        st.markdown(
            """
            <style>
            /* Upload formats, maximum file size, drag-and-drop helper text */
            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] p,
            [data-testid="stFileUploader"] span,
            [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"],
            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
                color: #475569 !important;
                -webkit-text-fill-color: #475569 !important;
            }

            /* Helper/caption text next to Process and Clear Chat buttons */
            [data-testid="stHorizontalBlock"] [data-testid="stCaptionContainer"],
            [data-testid="stHorizontalBlock"] [data-testid="stCaptionContainer"] *,
            [data-testid="stHorizontalBlock"] [data-testid="stMarkdownContainer"],
            [data-testid="stHorizontalBlock"] [data-testid="stMarkdownContainer"] p {
                color: #475569 !important;
                -webkit-text-fill-color: #475569 !important;
            }

            /* General muted helper text in the light theme */
            [data-testid="stCaptionContainer"],
            [data-testid="stCaptionContainer"] *,
            [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] * {
                color: #64748B !important;
                -webkit-text-fill-color: #64748B !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


# Header: theme selector at top-right, horizontal.
head_left, head_right = st.columns([6, 1.15], vertical_alignment="center")
with head_left:
    st.markdown("## 🧠 RAG Document Assistant")
    st.caption("Upload documents · Process content · Ask questions · Save files")
with head_right:
    st.radio(
        "Theme",
        options=["Light", "Dark"],
        horizontal=True,
        key="theme",
        label_visibility="collapsed",
    )

apply_theme(st.session_state.theme)


# =========================================================
# HELPERS
# =========================================================

def saved_files() -> list[Path]:
    return sorted(
        [p for p in INPUT_DIR.iterdir() if p.is_file()],
        key=lambda p: p.name.lower(),
    )


def save_uploads(files: list[Any]) -> tuple[list[str], list[str]]:
    saved, errors = [], []
    for uploaded in files:
        name = Path(uploaded.name).name
        destination = INPUT_DIR / name
        try:
            payload = uploaded.getvalue()
            if not payload:
                errors.append(f"{name}: empty file.")
                continue
            destination.write_bytes(payload)
            saved.append(name)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    return saved, errors


def clear_vector_database() -> None:
    for path in (FAISS_INDEX, FAISS_METADATA):
        if path.exists():
            path.unlink()
    st.session_state.rag = None


def rebuild_vector_database() -> tuple[int, list[str]]:
    """
    Run the project's existing extraction/chunking flow and persist a fresh
    FAISS index. Rebuilding from all remaining input files prevents deleted
    source files from remaining searchable in an old index.
    """
    chunks = process_input_folder()
    if not chunks:
        clear_vector_database()
        return 0, []

    vector_db = VectorStore()
    vector_db.create_vector_store(chunks)
    st.session_state.rag = None

    sources = sorted({
        str(chunk.get("source", "Unknown"))
        for chunk in chunks
        if isinstance(chunk, dict)
    })
    return len(chunks), sources


def get_rag() -> RAGPipeline:
    if st.session_state.rag is None:
        st.session_state.rag = RAGPipeline()
    return st.session_state.rag


def groq_complete(rag: RAGPipeline, prompt: str, max_tokens: int = 1200) -> str:
    response = rag.client.chat.completions.create(
        model=rag.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def summarize_document(filename: str) -> str:
    """
    Summarize the complete extracted text file, rather than only the top-k
    FAISS retrieval results. Uses a map/reduce-style summary for long files.
    """
    text_path = TEXT_DIR / f"{Path(filename).stem}.txt"
    if not text_path.exists():
        raise FileNotFoundError(
            f"Extracted text for {filename} was not found. Process the file first."
        )

    full_text = text_path.read_text(encoding="utf-8").strip()
    if not full_text:
        raise ValueError(f"No extractable text was found in {filename}.")

    rag = get_rag()
    # Keep each request moderate for context-window compatibility.
    segment_size = 10000
    segments = [
        full_text[i:i + segment_size]
        for i in range(0, len(full_text), segment_size)
    ]

    partial_summaries = []
    for number, segment in enumerate(segments, start=1):
        partial = groq_complete(
            rag,
            f"""Summarize the following part ({number} of {len(segments)})
of the document "{filename}". Preserve important facts, names, dates,
figures, definitions and conclusions. Do not add information not present.

DOCUMENT CONTENT:
{segment}

PART SUMMARY:""",
            max_tokens=900,
        )
        partial_summaries.append(partial)

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined = "\n\n".join(
        f"Part {i}: {summary}"
        for i, summary in enumerate(partial_summaries, start=1)
    )
    return groq_complete(
        rag,
        f"""Create one coherent, well-structured summary of the complete
document "{filename}" using the partial summaries below. Include key topics,
important facts, conclusions and actionable points. Avoid repetition.
Do not invent details.

PARTIAL SUMMARIES:
{combined}

FINAL DOCUMENT SUMMARY:""",
        max_tokens=1600,
    )


def format_source(source: dict[str, Any]) -> str:
    filename = (
        source.get("source")
        or source.get("source_file")
        or "Unknown source"
    )
    score = source.get("similarity_score")
    chunk_id = source.get("chunk_id")
    parts = [f"**{filename}**"]
    if chunk_id is not None:
        parts.append(f"Chunk: {chunk_id}")
    if isinstance(score, (int, float)):
        parts.append(f"Similarity: {score:.3f}")
    return " · ".join(parts)


def is_summary_request(prompt: str) -> bool:
    p = prompt.lower()
    return any(word in p for word in ("summarize", "summarise", "summary"))


def answer_question(prompt: str) -> tuple[str, list[dict[str, Any]]]:
    rag = get_rag()
    context = rag.retrieve_context(prompt)
    if not context:
        return (
            "I couldn't find relevant information in the processed documents.",
            [],
        )
    return rag.generate_answer(prompt, context), context


# =========================================================
# SECTION 1 — UPLOAD
# =========================================================

st.divider()
st.markdown("### 01 · Upload Documents")
upload_col, upload_hint = st.columns([3, 1.35], vertical_alignment="center")

with upload_col:
    uploads = st.file_uploader(
        "Choose files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.uploader_version}",
    )
    if uploads:
        names, errors = save_uploads(uploads)
        if names:
            st.success(f"Saved {len(names)} file(s): {', '.join(names)}")
        for error in errors:
            st.error(error)

with upload_hint:
    st.caption("PDF · DOCX · TXT · MD · Audio · Video")
    st.caption("Files are saved in `input_files/`. Processing is a separate step.")


# =========================================================
# SECTION 2 — PROCESS
# =========================================================

st.divider()
st.markdown("### 02 · Process Uploaded Files")

files_now = saved_files()
proc_left, proc_right = st.columns([1.25, 3], vertical_alignment="center")

with proc_left:
    process_clicked = st.button(
        "⚙️ Process documents",
        type="primary",
        use_container_width=True,
        disabled=not bool(files_now),
    )
with proc_right:
    st.caption(
        f"{len(files_now)} file(s) saved · "
        "Extraction → Chunking → Embeddings → FAISS"
    )

if process_clicked:
    with st.spinner("Extracting content and building the searchable index..."):
        try:
            chunk_count, indexed_sources = rebuild_vector_database()
            if chunk_count:
                st.session_state.last_process_report = {
                    "chunks": chunk_count,
                    "sources": indexed_sources,
                }
                st.success(
                    f"Processing complete: {chunk_count} chunks indexed "
                    f"from {len(indexed_sources)} source file(s)."
                )
            else:
                st.warning("No chunks were generated. Check the files and try again.")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")

if st.session_state.last_process_report:
    report = st.session_state.last_process_report
    st.caption(
        f"Last successful index: {report['chunks']} chunks · "
        f"{len(report['sources'])} source file(s)"
    )


# =========================================================
# SECTION 3 — CHAT
# =========================================================

st.divider()
st.markdown("### 03 · Chat with Your Documents")

# Render conversation history. It naturally expands the page after answers.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.markdown(format_source(source))

# Compact, in-section input; unlike st.chat_input this does not stick to
# the bottom edge of the browser window.
chat_files = [p.name for p in saved_files()]
chat_col, send_col = st.columns([6, 1], vertical_alignment="bottom")

with chat_col:
    prompt = st.text_input(
        "Your prompt",
        placeholder="Ask a question, or type: Summarize the uploaded documents...",
        label_visibility="collapsed",
        key="chat_prompt",
    )
with send_col:
    send_clicked = st.button(
        "Send ➤",
        type="primary",
        use_container_width=True,
        disabled=not bool(prompt.strip()),
    )

if send_clicked and prompt.strip():
    user_prompt = prompt.strip()
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    try:
        if is_summary_request(user_prompt):
            # Summarize all currently saved documents when the user asks
            # for a general summary. A filename in the prompt narrows it.
            matching_files = [
                name for name in chat_files
                if Path(name).stem.lower() in user_prompt.lower()
            ]
            targets = matching_files if matching_files else chat_files

            if not targets:
                raise ValueError("There are no uploaded documents to summarize.")

            summaries = []
            for filename in targets:
                summaries.append(
                    f"## Summary: {filename}\n\n{summarize_document(filename)}"
                )
            answer = "\n\n---\n\n".join(summaries)
            sources = [
                {"source": name, "chunk_id": "Full document"}
                for name in targets
            ]
        else:
            answer, sources = answer_question(user_prompt)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Unable to complete your request: {exc}",
            }
        )

    st.rerun()

clear_col, chat_tip = st.columns([1, 4], vertical_alignment="center")
with clear_col:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with chat_tip:
    st.caption("Question answering uses FAISS retrieval. Summarization reads the complete extracted document.")


# =========================================================
# SECTION 4 — SAVED DOCUMENTS
# =========================================================

st.divider()
st.markdown("### 04 · Saved Documents")

files_now = saved_files()
if not files_now:
    st.caption("No documents saved yet.")
else:
    for file_path in files_now:
        file_col, status_col, size_col, download_col, delete_col = st.columns(
            [3.5, 1.15, 0.8, 1.05, 0.8],
            vertical_alignment="center",
        )

        with file_col:
            st.markdown(f"📄 **{file_path.name}**")

        with status_col:
            # The current FAISS metadata stores source names. This is a
            # practical per-file indicator for files in the current index.
            indexed = False
            if FAISS_METADATA.exists():
                try:
                    import json
                    metadata = json.loads(FAISS_METADATA.read_text(encoding="utf-8"))
                    indexed = any(
                        item.get("source") == file_path.name
                        or item.get("source_file") == file_path.name
                        for item in metadata
                        if isinstance(item, dict)
                    )
                except Exception:
                    indexed = False
            if indexed:
                st.markdown("🟢 Processed")
            else:
                st.markdown("🟡 Uploaded")

        with size_col:
            st.caption(f"{file_path.stat().st_size / 1024:.0f} KB")

        with download_col:
            st.download_button(
                "Download",
                data=file_path.read_bytes(),
                file_name=file_path.name,
                key=f"download_{file_path.name}",
                use_container_width=True,
            )

        with delete_col:
            if st.button("Delete", key=f"delete_{file_path.name}", use_container_width=True):
                try:
                    file_path.unlink()
                    # Reset the uploader widget so its previous selection
                    # cannot silently upload the deleted file again.
                    st.session_state.uploader_version += 1
                    st.session_state.rag = None
                    st.session_state.last_process_report = None
                    remaining = saved_files()
                    if remaining:
                        # Rebuild from the remaining files so removed content
                        # is no longer present in the searchable FAISS index.
                        with st.spinner("Updating the knowledge base..."):
                            rebuild_vector_database()
                    else:
                        clear_vector_database()
                    st.success(f"{file_path.name} deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete {file_path.name}: {exc}")


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption("RAG Document Assistant · Groq · FAISS · Sentence Transformers · Whisper")
