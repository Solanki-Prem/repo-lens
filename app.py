"""Streamlit front-end for repo-lens.

Run with: streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from repo_lens.analyzer import list_js_like_functions, list_python_functions
from repo_lens.chains import (
    answer_question,
    architecture_summary,
    document_file,
    explain_function,
)
from repo_lens.config import (
    CHAT_PROVIDERS,
    DEFAULT_CHAT_MODELS,
    DEFAULT_EMBED_MODELS,
    EMBED_PROVIDERS,
    Settings,
)
from repo_lens.loader import repo_id_for, walk_sources
from repo_lens.pipeline import file_tree, get_indexed_source, index_repo, sample_headers
from repo_lens.store import VectorStore

st.set_page_config(page_title="repo-lens", page_icon=None, layout="wide")


# ---- sidebar ---------------------------------------------------------------

st.sidebar.title("repo-lens")
st.sidebar.caption("Ask questions, generate docs, and map architecture for any repo.")

defaults = Settings.load()

st.sidebar.markdown("**Chat model**")
chat_provider = st.sidebar.selectbox(
    "Chat provider", list(CHAT_PROVIDERS),
    index=list(CHAT_PROVIDERS).index(defaults.chat_provider),
)
chat_model = st.sidebar.text_input(
    "Chat model name", value=DEFAULT_CHAT_MODELS[chat_provider],
    key=f"chat_model_{chat_provider}",
)
chat_api_key = st.sidebar.text_input(
    f"{chat_provider.title()} API key",
    value=defaults.chat_api_key if defaults.chat_provider == chat_provider else "",
    type="password", key=f"chat_key_{chat_provider}",
)

st.sidebar.markdown("**Embedding model**")
embed_provider = st.sidebar.selectbox(
    "Embed provider", list(EMBED_PROVIDERS),
    index=list(EMBED_PROVIDERS).index(defaults.embed_provider),
    help="HuggingFace uses the free Inference API — grab a token at huggingface.co/settings/tokens.",
)
embed_model = st.sidebar.text_input(
    "Embed model name", value=DEFAULT_EMBED_MODELS[embed_provider],
    key=f"embed_model_{embed_provider}",
)
embed_api_key = st.sidebar.text_input(
    "HuggingFace token" if embed_provider == "huggingface" else "OpenAI API key",
    value=defaults.embed_api_key if defaults.embed_provider == embed_provider else "",
    type="password", key=f"embed_key_{embed_provider}",
)

settings = Settings(
    chat_provider=chat_provider,
    chat_model=chat_model.strip(),
    chat_api_key=chat_api_key.strip(),
    embed_provider=embed_provider,
    embed_model=embed_model.strip(),
    embed_api_key=embed_api_key.strip(),
    chroma_dir=defaults.chroma_dir,
    repo_cache_dir=defaults.repo_cache_dir,
    max_file_bytes=defaults.max_file_bytes,
)

missing = []
if not settings.chat_api_key:
    missing.append(f"{chat_provider} API key")
if not settings.embed_api_key:
    missing.append("HuggingFace token" if embed_provider == "huggingface" else "OpenAI key")
ready = not missing

st.sidebar.divider()
st.sidebar.markdown("**Repository**")
source = st.sidebar.text_input(
    "GitHub URL or local path",
    value=st.session_state.get("source", ""),
    placeholder="https://github.com/owner/repo",
)
rebuild = st.sidebar.checkbox("Rebuild index", value=False)
col_a, col_b = st.sidebar.columns(2)
with col_a:
    do_index = st.button("Index", use_container_width=True, type="primary", disabled=not ready)
with col_b:
    do_load = st.button("Load", use_container_width=True, disabled=not ready)

st.sidebar.divider()
st.sidebar.markdown("**Previously indexed**")
if ready:
    indexed = VectorStore.list_indexed(settings)
    if indexed:
        current = st.session_state.get("repo_id")
        for rid in indexed:
            btn_type = "primary" if rid == current else "secondary"
            if st.sidebar.button(rid, key=f"switch_{rid}", type=btn_type,
                                 use_container_width=True) and rid != current:
                st.session_state["repo_id"] = rid
                # Restore the original URL/path so re-indexing and the tabs
                # work correctly. Fall back to the cached folder path for
                # older indexes that never wrote the source marker.
                original = get_indexed_source(settings, rid)
                if original:
                    st.session_state["source"] = original
                else:
                    cached_path = settings.repo_cache_dir / rid
                    st.session_state["source"] = (
                        str(cached_path) if cached_path.exists() else ""
                    )
    else:
        st.sidebar.caption("Nothing indexed yet.")

if not ready:
    st.title("repo-lens")
    st.info(f"Fill in the sidebar to continue. Missing: {', '.join(missing)}.")
    st.stop()


# ---- index / load actions --------------------------------------------------

def _repo_root(source: str, repo_id: str) -> Path:
    """Find the folder holding the repo's files.

    Prefers the cache directory keyed by the known repo_id, so we
    don't re-hash the source string. Falls back to treating source
    as a local path.
    """
    cached = settings.repo_cache_dir / repo_id
    if cached.exists():
        return cached
    return Path(source).expanduser().resolve()


if do_index and source:
    area = st.empty()
    try:
        with st.spinner("Working..."):
            result = index_repo(settings, source, progress=lambda m: area.info(m), rebuild=rebuild)
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")
        st.stop()
    st.session_state["repo_id"] = result.repo_id
    st.session_state["source"] = source
    st.success(f"Indexed `{result.repo_id}` — {result.files_indexed} files, {result.chunks_indexed} chunks.")

if do_load and source:
    rid = repo_id_for(source)
    if VectorStore(settings, rid).count() == 0:
        st.warning("No index found for this source. Click Index first.")
    else:
        st.session_state["repo_id"] = rid
        st.session_state["source"] = source
        st.success(f"Loaded `{rid}`.")


repo_id = st.session_state.get("repo_id")
active_source = st.session_state.get("source")

if not repo_id:
    st.title("repo-lens")
    st.write(
        "Point the sidebar at a GitHub URL (or a local path), hit **Index**, "
        "and the project is embedded into a private Chroma collection. Then "
        "pick a section above to ask questions, generate docs, or map the architecture."
    )
    st.stop()


# ---- main panel ------------------------------------------------------------

store = VectorStore(settings, repo_id)
st.title(repo_id)
st.caption(
    f"{store.count()} chunks · source: `{active_source}` · "
    f"chat: {settings.chat_provider}/{settings.chat_model} · "
    f"embed: {settings.embed_provider}/{settings.embed_model}"
)

# Radio-based navigation instead of st.tabs — persists the active
# section across reruns via session_state, so button clicks inside
# a section don't warp you back to the first one.
section = st.radio(
    "Section",
    ["Ask", "Architecture", "File docs", "Explain function"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_section",
)

if section == "Ask":
    question = st.text_input("Your question", key="ask_q",
                             placeholder="What does the authentication middleware do?")
    col1, col2 = st.columns([2, 1])
    with col1:
        path_filter = st.text_input("Restrict to path substring (optional)",
                                    key="ask_path", placeholder="src/auth")
    with col2:
        k = st.slider("Snippets", 3, 12, 6, key="ask_k")
    if st.button("Ask", type="primary", key="ask_btn") and question.strip():
        with st.spinner("Searching..."):
            ans = answer_question(settings, store, question.strip(),
                                  k=k, path_prefix=path_filter.strip() or None)
        st.markdown(ans.text)
        with st.expander("Sources"):
            for s in ans.sources:
                st.markdown(f"- `{s}`")

elif section == "Architecture":
    st.write("Builds a one-page architecture summary from the file tree and headers.")
    if st.button("Generate summary", key="arch_btn"):
        root = _repo_root(active_source, repo_id)
        with st.spinner("Reading..."):
            files = list(walk_sources(root, repo_id, settings.max_file_bytes))
            tree = file_tree(root)
            samples = sample_headers(files)
        with st.spinner("Writing..."):
            md = architecture_summary(settings, repo_id=repo_id, tree=tree, samples=samples)
        st.markdown(md)
        with st.expander("File tree input"):
            st.code(tree)

elif section == "File docs":
    root = _repo_root(active_source, repo_id)
    files = list(walk_sources(root, repo_id, settings.max_file_bytes))
    if not files:
        st.warning("No readable source files at the project root.")
    else:
        choice = st.selectbox("File", [f.path for f in files], key="doc_pick")
        if st.button("Generate documentation", key="doc_btn"):
            picked = next(f for f in files if f.path == choice)
            with st.spinner("Writing..."):
                md = document_file(settings, path=picked.path, source=picked.text,
                                   language=picked.language)
            st.markdown(md)

elif section == "Explain function":
    root = _repo_root(active_source, repo_id)
    refs = list_python_functions(root, repo_id) + list_js_like_functions(root)
    if not refs:
        st.info("No Python/JS/TS functions detected.")
    else:
        labels = [f"{r.path}:{r.start_line}  {r.qualname}" for r in refs]
        idx = st.selectbox("Function", range(len(labels)), format_func=lambda i: labels[i])
        chosen = refs[idx]
        st.code(chosen.source, language=chosen.language)
        if st.button("Explain", key="fn_btn"):
            with st.spinner("Thinking..."):
                md = explain_function(settings, path=chosen.path, name=chosen.qualname,
                                      source=chosen.source, language=chosen.language)
            st.markdown(md)
