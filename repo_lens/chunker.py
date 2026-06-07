from __future__ import annotations

from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .loader import SourceFile

# Map our language tag to LangChain's Language enum where they overlap.
# Anything not listed falls back to plain text splitting.
_LANG_MAP = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "java": Language.JAVA,
    "kotlin": Language.KOTLIN,
    "scala": Language.SCALA,
    "go": Language.GO,
    "rust": Language.RUST,
    "ruby": Language.RUBY,
    "php": Language.PHP,
    "swift": Language.SWIFT,
    "c": Language.C,
    "cpp": Language.CPP,
    "csharp": Language.CSHARP,
    "html": Language.HTML,
    "markdown": Language.MARKDOWN,
    "rst": Language.RST,
    "sol": Language.SOL,
}


def _splitter_for(lang: str, chunk_size: int, overlap: int):
    enum = _LANG_MAP.get(lang)
    if enum is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=enum,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_files(
    files: Iterable[SourceFile],
    chunk_size: int = 1200,
    overlap: int = 150,
) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        splitter = _splitter_for(f.language, chunk_size, overlap)
        pieces = splitter.split_text(f.text)
        for i, piece in enumerate(pieces):
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "repo_id": f.repo_id,
                        "path": f.path,
                        "language": f.language,
                        "chunk": i,
                        "chunks_total": len(pieces),
                    },
                )
            )
    return docs
