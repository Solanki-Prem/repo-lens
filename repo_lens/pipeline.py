"""High level orchestration: clone, chunk, embed, store, summarise.

Kept separate from the chains so the Streamlit app and the CLI can share it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .chunker import chunk_files
from .config import Settings
from .loader import (
    SourceFile,
    clone_or_use_local,
    repo_id_for,
    walk_sources,
)
from .store import VectorStore


@dataclass
class IndexResult:
    repo_id: str
    root: Path
    files_indexed: int
    chunks_indexed: int


def index_repo(
    settings: Settings,
    source: str,
    progress: Optional[Callable[[str], None]] = None,
    rebuild: bool = False,
) -> IndexResult:
    rid = repo_id_for(source)
    _log(progress, f"Resolving source ({rid})...")
    root = clone_or_use_local(source, settings.repo_cache_dir)

    _log(progress, "Reading source files...")
    files: List[SourceFile] = list(
        walk_sources(root, rid, settings.max_file_bytes)
    )
    if not files:
        raise RuntimeError("No source files were found at that location.")

    store = VectorStore(settings, rid)
    if rebuild and store.count() > 0:
        _log(progress, "Wiping previous index...")
        store.reset()

    if store.count() > 0 and not rebuild:
        _log(progress, "Reusing existing index.")
        return IndexResult(
            repo_id=rid,
            root=root,
            files_indexed=len(files),
            chunks_indexed=store.count(),
        )

    _log(progress, f"Splitting {len(files)} files into chunks...")
    docs = chunk_files(files)

    _log(progress, f"Embedding {len(docs)} chunks into Chroma...")
    written = store.add(docs)

    return IndexResult(
        repo_id=rid,
        root=root,
        files_indexed=len(files),
        chunks_indexed=written,
    )


def file_tree(root: Path, max_entries: int = 200) -> str:
    """Compact tree string. Useful as model input for the architecture chain."""
    entries: List[str] = []
    for p in sorted(root.rglob("*")):
        if any(part.startswith(".") or part in {"node_modules", "dist", "build"} for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        if p.is_dir():
            rel += "/"
        entries.append(rel)
        if len(entries) >= max_entries:
            entries.append("...")
            break
    return "\n".join(entries)


def sample_headers(files: List[SourceFile], per_file_chars: int = 400, limit: int = 30) -> str:
    """First few hundred chars of up to N files, useful as architecture-chain input."""
    selected = files[:limit]
    blocks = []
    for f in selected:
        head = f.text[:per_file_chars]
        blocks.append(f"--- {f.path} ---\n{head}")
    return "\n\n".join(blocks)


def _log(progress, msg: str):
    if progress is not None:
        progress(msg)
