import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


CHAT_PROVIDERS = ("openai", "anthropic", "google")
EMBED_PROVIDERS = ("huggingface", "openai")

DEFAULT_CHAT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "google": "gemini-1.5-flash",
}

DEFAULT_EMBED_MODELS = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
}

CHAT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

EMBED_KEY_ENV = {
    "huggingface": "HUGGINGFACEHUB_API_TOKEN",
    "openai": "OPENAI_API_KEY",
}


@dataclass
class Settings:
    chat_provider: str
    chat_model: str
    chat_api_key: str
    embed_provider: str
    embed_model: str
    embed_api_key: str
    chroma_dir: Path
    repo_cache_dir: Path
    max_file_bytes: int = 204800

    @classmethod
    def load(cls) -> "Settings":
        """Env-based loader used by the CLI and as UI defaults."""
        chat_provider = os.getenv("CHAT_PROVIDER", "openai").strip().lower()
        embed_provider = os.getenv("EMBED_PROVIDER", "huggingface").strip().lower()
        return cls(
            chat_provider=chat_provider,
            chat_model=os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODELS.get(chat_provider, "")),
            chat_api_key=os.getenv(CHAT_KEY_ENV.get(chat_provider, ""), "").strip(),
            embed_provider=embed_provider,
            embed_model=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODELS.get(embed_provider, "")),
            embed_api_key=os.getenv(EMBED_KEY_ENV.get(embed_provider, ""), "").strip(),
            chroma_dir=Path(os.getenv("CHROMA_DIR", ".chroma")).resolve(),
            repo_cache_dir=Path(os.getenv("REPO_CACHE_DIR", ".repo_cache")).resolve(),
            max_file_bytes=int(os.getenv("MAX_FILE_BYTES", "204800")),
        )
