from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import Settings


class VectorStore:
    """Thin wrapper around a per-repo Chroma collection."""

    def __init__(self, settings: Settings, repo_id: str):
        self.settings = settings
        self.repo_id = repo_id
        self._embeddings = OpenAIEmbeddings(
            model=settings.embed_model,
            api_key=settings.openai_api_key,
        )
        self._dir = settings.chroma_dir / repo_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db = Chroma(
            collection_name=repo_id.replace("-", "_"),
            embedding_function=self._embeddings,
            persist_directory=str(self._dir),
        )

    def add(self, docs: List[Document], batch_size: int = 128) -> int:
        if not docs:
            return 0
        total = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self._db.add_documents(batch)
            total += len(batch)
        return total

    def search(self, query: str, k: int = 6, path_prefix: Optional[str] = None) -> List[Document]:
        filt = {"path": {"$contains": path_prefix}} if path_prefix else None
        # Chroma doesn't support $contains directly; use a post-filter when needed.
        results = self._db.similarity_search(query, k=k * 3 if path_prefix else k)
        if path_prefix:
            results = [r for r in results if path_prefix in r.metadata.get("path", "")]
            results = results[:k]
        return results

    def count(self) -> int:
        try:
            return self._db._collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        # Drop the collection by deleting the persistence dir.
        import shutil

        try:
            self._db.delete_collection()
        except Exception:
            pass
        if self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db = Chroma(
            collection_name=self.repo_id.replace("-", "_"),
            embedding_function=self._embeddings,
            persist_directory=str(self._dir),
        )

    @staticmethod
    def list_indexed(settings: Settings) -> List[str]:
        base: Path = settings.chroma_dir
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())
