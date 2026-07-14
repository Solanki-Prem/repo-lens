import shutil
import time
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import Settings


class _RetryingEmbeddings(Embeddings):
    """Retries embedding calls — helps with HF cold-model 500s."""

    def __init__(self, inner: Embeddings, retries: int = 5, delay: float = 3.0):
        self._inner = inner
        self._retries = retries
        self._delay = delay

    def _run(self, fn, *args):
        # Try up to _retries times. Wait between attempts. On the final
        # attempt, let any exception propagate naturally.
        for attempt in range(self._retries - 1):
            try:
                return fn(*args)
            except Exception:
                time.sleep(self._delay * (attempt + 1))
        return fn(*args)

    def embed_documents(self, texts):
        return self._run(self._inner.embed_documents, texts)

    def embed_query(self, text):
        return self._run(self._inner.embed_query, text)


def _build_embeddings(settings: Settings) -> Embeddings:
    if settings.embed_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        inner = HuggingFaceEndpointEmbeddings(
            model=settings.embed_model,
            task="feature-extraction",
            huggingfacehub_api_token=settings.embed_api_key or None,
        )
        return _RetryingEmbeddings(inner)
    if settings.embed_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embed_model, api_key=settings.embed_api_key)
    raise ValueError(f"Unsupported embed provider: {settings.embed_provider}")


class VectorStore:
    """Thin wrapper around a per-repo Chroma collection."""

    def __init__(self, settings: Settings, repo_id: str):
        self.settings = settings
        self.repo_id = repo_id
        self._dir = settings.chroma_dir / repo_id
        # Chroma 1.5.x can open an empty leftover dir in SQLite readonly mode
        # (usually a crashed prior init). If we see one, drop it so Chroma
        # gets a clean slate.
        if self._dir.exists() and not any(self._dir.iterdir()):
            shutil.rmtree(self._dir)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = _build_embeddings(settings)
        self._db = Chroma(
            collection_name=repo_id.replace("-", "_"),
            embedding_function=self._embeddings,
            persist_directory=str(self._dir),
        )

    def add(self, docs: List[Document], batch_size: int = 128) -> int:
        total = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self._db.add_documents(batch)
            total += len(batch)
        return total

    def search(self, query: str, k: int = 6, path_prefix: Optional[str] = None) -> List[Document]:
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

    @staticmethod
    def wipe(settings: Settings, repo_id: str) -> None:
        """Delete a repo's Chroma dir and drop the cached SystemClient.

        chromadb 1.x caches PersistentClient per persist_directory, so
        after an rmtree the next Chroma() call in the same process would
        hand back a stale client pointing at deleted SQLite files and
        error with 'attempt to write a readonly database'.
        """
        from chromadb.api.shared_system_client import SharedSystemClient

        target = settings.chroma_dir / repo_id
        if target.exists():
            shutil.rmtree(target)
        try:
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass

    @staticmethod
    def list_indexed(settings: Settings) -> List[str]:
        base = settings.chroma_dir
        if not base.exists():
            return []
        # A directory only counts as "indexed" if Chroma actually wrote its
        # sqlite file. Skips empty stale dirs and dirs containing only our
        # own _source.txt marker.
        return sorted(
            p.name for p in base.iterdir()
            if p.is_dir() and (p / "chroma.sqlite3").exists()
        )
