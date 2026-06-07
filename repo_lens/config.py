import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str
    chat_model: str
    embed_model: str
    chroma_dir: Path
    repo_cache_dir: Path
    max_file_bytes: int

    @classmethod
    def load(cls) -> "Settings":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return cls(
            openai_api_key=key,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            chroma_dir=Path(os.getenv("CHROMA_DIR", ".chroma")).resolve(),
            repo_cache_dir=Path(os.getenv("REPO_CACHE_DIR", ".repo_cache")).resolve(),
            max_file_bytes=int(os.getenv("MAX_FILE_BYTES", "204800")),
        )
