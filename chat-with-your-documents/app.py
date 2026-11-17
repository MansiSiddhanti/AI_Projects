import time
from datetime import datetime

import streamlit as st

from rag import RAGPipeline
from rag.intent import is_summary_request
from rag.pipeline import parse_citations

st.set_page_config(page_title="Chat with Your Documents", layout="wide")


@st.cache_resource(show_spinner="Loading models and index...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings()


@st.cache_data(ttl=30, show_spinner=False)
def llm_status() -> tuple[bool, str]:
    return get_pipeline().llm.health()


try:
    pipeline = get_pipeline()
except Exception as exc:
    st.error(f"Could not initialize the application: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            scores = " · ".join(f"{k} {v:.3f}" for k, v in s["scores"].items())
            st.markdown(f"**[{i}] {s['source']}, p. {s['page']}**  \n:gray[{scores}]")
            st.caption(s["text"])


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if meta := message.get("meta"):
            st.caption(meta)
        render_sources(message.get("sources", []))


def transcript_markdown() -> str:
    lines = [f"# Chat transcript ({datetime.now():%Y-%m-%d %H:%M})\n"]
    for m in st.session_state.messages:
        lines.append(f"**{m['role'].capitalize()}:** {m['content']}\n")
        for i, s in enumerate(m.get("sources", []), start=1):
            lines.append(f"> [{i}] {s['source']}, page {s['page']}")
        lines.append("")
    return "\n".join(lines)


def run_summary(source: str) -> dict:
    bar = st.progress(0.0, text=f"Summarizing {source}...")
    summary = pipeline.summarize(
        source, progress=lambda i, n, msg: bar.progress(i / n, text=f"{msg} ({i}/{n})")
    )
    bar.empty()
    return {"role": "assistant", "content": f"**Summary of {source}**\n\n{summary}"}


with st.sidebar:
    st.header("Documents")

    ok, status = llm_status()
    (st.success if ok else st.error)(status)

    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        submitted = st.form_submit_button("Index", type="primary", use_container_width=True)
    if submitted and files:
        progress = st.progress(0.0)
        for i, f in enumerate(files, start=1):
            progress.progress((i - 1) / len(files), text=f"Indexing {f.name}...")
            try:
                result = pipeline.ingest_bytes(f.name, f.getvalue())
                if result.status == "indexed":
                    st.toast(f"Indexed {f.name}: {result.chunks} chunks")
                elif result.status == "unchanged":
                    st.toast(f"{f.name} is already indexed")
                else:
                    st.warning(f"No extractable text in {f.name} (scanned PDF?)")
            except Exception as exc:
                st.error(f"Failed to index {f.name}: {exc}")
        progress.empty()

    documents = pipeline.documents()
    names = [d.source for d in documents]

    if documents:
        st.caption(f"{len(documents)} documents · {sum(d.chunks for d in documents)} chunks")
        for doc in documents:
            col1, col2, col3 = st.columns([5, 2, 2], vertical_alignment="center")
            col1.markdown(f"**{doc.source}**  \n:gray[{doc.pages} pages · {doc.chunks} chunks]")
            if col2.button("Sum.", key=f"sum-{doc.source}", help="Summarize this document"):
                st.session_state.pending_summary = doc.source
            if col3.button("Del", key=f"del-{doc.source}", help="Remove from index"):
                pipeline.delete_document(doc.source)
                st.rerun()
    else:
        st.info("Upload a PDF to get started.")

    st.divider()
    st.subheader("Search settings")
    scope = st.multiselect("Search within", names, placeholder="All documents")
    with st.expander("Retrieval options"):
        mode = st.radio(
            "Strategy",
            ["hybrid", "dense", "bm25"],
            horizontal=True,
            index=["hybrid", "dense", "bm25"].index(pipeline.settings.retrieval_mode),
            help="Hybrid fuses semantic (dense) and keyword (BM25) search with RRF.",
        )
        rerank = st.toggle("Cross-encoder reranking", value=pipeline.settings.use_reranker)
        top_k = st.slider("Passages per answer", 1, 10, pipeline.settings.top_k)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    col2.download_button(
        "Export",
        transcript_markdown(),
        file_name="chat.md",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )


st.title("Chat with Your Documents")
st.caption(
    "Private, local RAG: hybrid search + reranking over your PDFs, answered by "
    f"`{pipeline.settings.llm_model}` via Ollama, with page-level citations."
)

if not st.session_state.messages:
    st.markdown(
        "**Try asking:**\n"
        "- *What techniques are used to handle imbalanced datasets?*\n"
        "- *How is AI being integrated with IoT devices?*\n"
        "- *Summarize Paper_01* (summaries are cached after the first run)"
    )

for message in st.session_state.messages:
    render_message(message)

# sidebar summary button
if source := st.session_state.pop("pending_summary", None):
    request = {"role": "user", "content": f"Summarize {source}"}
    st.session_state.messages.append(request)
    render_message(request)
    with st.chat_message("assistant"):
        try:
            reply = run_summary(source)
            st.markdown(reply["content"])
            st.session_state.messages.append(reply)
        except Exception as exc:
            st.error(f"Summarization failed: {exc}")

if question := st.chat_input("Ask about your documents, or ask for a summary..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        try:
            summary_source = pipeline.route(question, scope[0] if len(scope) == 1 else None)
            if summary_source:
                reply = run_summary(summary_source)
                st.markdown(reply["content"])
            elif is_summary_request(question) and names:
                reply = {
                    "role": "assistant",
                    "content": (
                        "Which document should I summarize? Name it (e.g. *Summarize "
                        f"{names[0]}*), or use the Sum. button in the sidebar."
                    ),
                }
                st.markdown(reply["content"])
            else:
                options = pipeline.default_options(
                    top_k=top_k, mode=mode, rerank=rerank, sources=scope or None
                )
                with st.spinner("Searching documents..."):
                    streamed = pipeline.stream_answer(question, history, options)
                t0 = time.perf_counter()
                text = st.write_stream(streamed.tokens)
                timings = streamed.timings | {"generate": time.perf_counter() - t0}

                _, invalid = parse_citations(text, len(streamed.sources))
                meta = " · ".join(f"{k} {v:.1f}s" for k, v in timings.items() if v > 0.05)
                if streamed.standalone_question != question:
                    meta = f"searched: “{streamed.standalone_question}” · {meta}"
                if invalid:
                    st.warning(f"The model cited non-existent sources: {invalid}")
                st.caption(meta)
                sources = [s.to_dict() for s in streamed.sources]
                render_sources(sources)
                reply = {"role": "assistant", "content": text, "sources": sources, "meta": meta}
            st.session_state.messages.append(reply)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
