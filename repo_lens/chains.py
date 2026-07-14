from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .config import Settings
from .prompts import (
    ARCHITECTURE_PROMPT,
    FILE_DOC_PROMPT,
    FUNCTION_EXPLAIN_PROMPT,
    QA_PROMPT,
)
from .store import VectorStore


@dataclass
class Answer:
    text: str
    sources: List[str]


def _llm(settings: Settings, temperature: float = 0.1) -> BaseChatModel:
    provider = settings.chat_provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.chat_api_key,
            temperature=temperature,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.chat_model,
            api_key=settings.chat_api_key,
            temperature=temperature,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.chat_api_key,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported chat provider: {provider}")


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        path = d.metadata.get("path", "?")
        lang = d.metadata.get("language", "")
        chunk = d.metadata.get("chunk", 0)
        blocks.append(f"--- {path} (chunk {chunk}) ---\n```{lang}\n{d.page_content}\n```")
    return "\n\n".join(blocks)


def answer_question(
    settings: Settings,
    store: VectorStore,
    question: str,
    k: int = 6,
    path_prefix: Optional[str] = None,
) -> Answer:
    docs = store.search(question, k=k, path_prefix=path_prefix)
    context = _format_context(docs)
    chain = QA_PROMPT | _llm(settings) | StrOutputParser()
    text = chain.invoke({"question": question, "context": context})
    sources = sorted({d.metadata.get("path", "?") for d in docs})
    return Answer(text=text, sources=sources)


def document_file(
    settings: Settings,
    path: str,
    source: str,
    language: str,
) -> str:
    chain = FILE_DOC_PROMPT | _llm(settings, temperature=0.2) | StrOutputParser()
    return chain.invoke({"path": path, "language": language, "source": source})


def explain_function(
    settings: Settings,
    path: str,
    name: str,
    source: str,
    language: str,
) -> str:
    chain = FUNCTION_EXPLAIN_PROMPT | _llm(settings) | StrOutputParser()
    return chain.invoke(
        {"path": path, "name": name, "language": language, "source": source}
    )


def architecture_summary(
    settings: Settings,
    repo_id: str,
    tree: str,
    samples: str,
) -> str:
    chain = ARCHITECTURE_PROMPT | _llm(settings, temperature=0.2) | StrOutputParser()
    return chain.invoke({"repo_id": repo_id, "tree": tree, "samples": samples})
