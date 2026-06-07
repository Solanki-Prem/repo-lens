"""Streamlit front-end for repo-lens.

Run with: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from repo_lens.analyzer import list_js_like_functions, list_python_functions
from repo_lens.chains import (
    answer_question,
    architecture_summary,
    document_file,
    explain_function,
)
from repo_lens.config import Settings
from repo_lens.loader import _language_for, repo_id_for, walk_sources
from repo_lens.pipeline import file_tree, index_repo, sample_headers
from repo_lens.store import VectorStore

st.set_page_config(page_title="repo-lens", page_icon=None, layout="wide")


# ---- session helpers -------------------------------------------------------

def get_settings() -> Settings | None:
    try:
        return Settings.load()
    except RuntimeError as e:
        st.error(str(e))
        return None


def repo_root_for(settings: Settings, source: str) -> Path:
    rid = repo_id_for(source)
    cached = settings.repo_cache_dir / rid
    if cached.exists():
        return cached
    return Path(source).expanduser().resolve()


# ---- sidebar ---------------------------------------------------------------

st.sidebar.title("repo-lens")
st.sidebar.caption("Ask questions, generate docs, and map architecture for any repo.")

settings = get_settings()
if settings is None:
    st.stop()

st.sidebar.markdown("**Repository**")
default_source = st.session_state.get("source", "")
source = st.sidebar.text_input(
    "GitHub URL or local path",
    value=default_source,
    placeholder="https://github.com/owner/repo",
)
rebuild = st.sidebar.checkbox("Rebuild index", value=False)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    do_index = st.button("Index", use_container_width=True, type="primary")
with col_b:
    do_load = st.button("Load", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("**Previously indexed**")
indexed = VectorStore.list_indexed(settings)
if indexed:
    chosen = st.sidebar.selectbox("Switch to", options=["-"] + indexed, index=0)
    if chosen and chosen != "-":
        st.session_state["repo_id"] = chosen
        st.session_state["source"] = chosen
else:
    st.sidebar.caption("Nothing indexed yet.")


# ---- index / load actions --------------------------------------------------

if do_index and source:
    progress_area = st.empty()
    with st.spinner("Working..."):
        def _progress(msg: str):
            progress_area.info(msg)

        try:
            result = index_repo(settings, source, progress=_progress, rebuild=rebuild)
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            st.stop()
    st.session_state["repo_id"] = result.repo_id
    st.session_state["source"] = source
    st.success(
        f"Indexed `{result.repo_id}` — {result.files_indexed} files, "
        f"{result.chunks_indexed} chunks."
    )

if do_load and source:
    rid = repo_id_for(source)
    store = VectorStore(settings, rid)
    if store.count() == 0:
        st.warning("No index found for this source. Click Index first.")
    else:
        st.session_state["repo_id"] = rid
        st.session_state["source"] = source
        st.success(f"Loaded `{rid}` ({store.count()} chunks).")


repo_id = st.session_state.get("repo_id")
active_source = st.session_state.get("source")

if not repo_id:
    st.title("repo-lens")
    st.write(
        "Point the sidebar at a GitHub URL (or a local path), hit **Index**, "
        "and the project is embedded into a private Chroma collection. After "
        "that, you can ask questions, generate per-file documentation, get a "
        "one-page architecture summary, or have a single function explained."
    )
    st.stop()


# ---- main panel ------------------------------------------------------------

st.title(repo_id)
store = VectorStore(settings, repo_id)
st.caption(f"{store.count()} indexed chunks · source: `{active_source}`")

tab_ask, tab_arch, tab_docs, tab_fn = st.tabs(
    ["Ask", "Architecture", "File docs", "Explain function"]
)


with tab_ask:
    question = st.text_input(
        "Your question",
        placeholder="What does the authentication middleware do?",
        key="ask_q",
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        path_filter = st.text_input(
            "Restrict to path substring (optional)",
            placeholder="src/auth",
            key="ask_path",
        )
    with col2:
        k = st.slider("Snippets", 3, 12, 6, key="ask_k")
    if st.button("Ask", type="primary", key="ask_btn") and question.strip():
        with st.spinner("Searching the index and writing an answer..."):
            ans = answer_question(
                settings,
                store,
                question.strip(),
                k=k,
                path_prefix=path_filter.strip() or None,
            )
        st.markdown(ans.text)
        with st.expander("Sources"):
            for s in ans.sources:
                st.markdown(f"- `{s}`")


with tab_arch:
    st.write("Builds a one-page architecture summary from the file tree and headers.")
    if st.button("Generate summary", key="arch_btn"):
        root = repo_root_for(settings, active_source)
        with st.spinner("Reading the project layout..."):
            files = list(walk_sources(root, repo_id, settings.max_file_bytes))
            tree = file_tree(root)
            samples = sample_headers(files)
        with st.spinner("Writing the summary..."):
            md = architecture_summary(settings, repo_id=repo_id, tree=tree, samples=samples)
        st.markdown(md)
        with st.expander("File tree input"):
            st.code(tree)


with tab_docs:
    root = repo_root_for(settings, active_source)
    files = list(walk_sources(root, repo_id, settings.max_file_bytes))
    if not files:
        st.warning("No readable source files at the project root.")
    else:
        options = [f.path for f in files]
        choice = st.selectbox("File", options=options, key="doc_pick")
        if st.button("Generate documentation", key="doc_btn"):
            picked = next(f for f in files if f.path == choice)
            with st.spinner("Reading the file and writing the docs..."):
                md = document_file(
                    settings,
                    path=picked.path,
                    source=picked.text,
                    language=picked.language,
                )
            st.markdown(md)


with tab_fn:
    root = repo_root_for(settings, active_source)
    py_refs = list_python_functions(root, repo_id)
    js_refs = list_js_like_functions(root)
    refs = py_refs + js_refs
    if not refs:
        st.info("No Python/JS/TS functions detected.")
    else:
        labels = [f"{r.path}:{r.start_line}  {r.qualname}" for r in refs]
        idx = st.selectbox("Function", range(len(labels)), format_func=lambda i: labels[i])
        chosen = refs[idx]
        st.code(chosen.source, language=chosen.language)
        if st.button("Explain", key="fn_btn"):
            with st.spinner("Thinking..."):
                md = explain_function(
                    settings,
                    path=chosen.path,
                    name=chosen.qualname,
                    source=chosen.source,
                    language=chosen.language,
                )
            st.markdown(md)
