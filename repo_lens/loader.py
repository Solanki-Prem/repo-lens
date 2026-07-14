import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

from git import Repo

# Things we never want to index. Either binary, generated, or just noise.
SKIP_DIRS = {
    ".git", ".github", ".idea", ".vscode", ".vs",
    "node_modules", "vendor", "dist", "build", "out", "target",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", "site-packages",
    ".next", ".turbo", ".cache", ".chroma", ".repo_cache",
    "coverage", ".nyc_output",
}

# Extensions worth reading as source. Anything else gets skipped.
SOURCE_EXTS = {
    ".py", ".pyi",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".scala", ".groovy",
    ".go", ".rs", ".rb", ".php", ".swift",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cxx",
    ".cs", ".fs",
    ".sh", ".bash", ".zsh", ".ps1",
    ".sql",
    ".html", ".css", ".scss", ".sass", ".vue", ".svelte",
    ".md", ".rst", ".txt",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".json",
    ".dockerfile", ".tf",
}

# Some files have no extension but we still care about them.
SPECIAL_NAMES = {"Dockerfile", "Makefile", "Rakefile", "Gemfile", "Procfile"}


@dataclass
class SourceFile:
    repo_id: str
    path: str          # path relative to repo root, posix-style
    abs_path: Path
    language: str
    size: int
    text: str


def repo_id_for(url_or_path: str) -> str:
    """Stable, short id used as Chroma collection + cache dir name."""
    norm = url_or_path.strip().rstrip("/").lower()
    digest = hashlib.sha1(norm.encode()).hexdigest()[:10]
    base = re.sub(r"[^a-z0-9]+", "-", Path(norm).name)[:40].strip("-") or "repo"
    return f"{base}-{digest}"


def is_github_url(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("http://", "https://", "git@")):
        return True
    return value.endswith(".git")


def clone_or_use_local(source: str, cache_dir: Path) -> Path:
    """Either clone the repo into cache_dir or return source as a local path."""
    source = source.strip()
    if not is_github_url(source):
        p = Path(source).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Local path does not exist: {p}")
        return p

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / repo_id_for(source)
    if target.exists():
        # If a previous clone is there, just reuse it.
        return target

    # Normalize URLs of the form github.com/owner/repo
    url = source
    parsed = urlparse(url)
    if parsed.scheme == "" and url.startswith("github.com/"):
        url = "https://" + url

    Repo.clone_from(url, target, depth=1)
    return target


def language_for(path: Path) -> str:
    ext = path.suffix.lower()
    if path.name in SPECIAL_NAMES:
        return path.name.lower()
    return {
        ".py": "python", ".pyi": "python",
        ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".java": "java", ".kt": "kotlin", ".scala": "scala", ".groovy": "groovy",
        ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".swift": "swift",
        ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cxx": "cpp",
        ".cs": "csharp", ".fs": "fsharp",
        ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell",
        ".sql": "sql",
        ".html": "html", ".css": "css", ".scss": "scss", ".sass": "sass",
        ".vue": "vue", ".svelte": "svelte",
        ".md": "markdown", ".rst": "rst", ".txt": "text",
        ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".ini": "ini", ".cfg": "ini",
        ".json": "json",
        ".tf": "terraform", ".dockerfile": "dockerfile",
    }.get(ext, "text")


def walk_sources(
    root: Path,
    repo_id: str,
    max_bytes: int,
) -> Iterator[SourceFile]:
    for path in _iter_files(root):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > max_bytes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Binary or unreadable. Skip.
            continue
        rel = path.relative_to(root).as_posix()
        yield SourceFile(
            repo_id=repo_id,
            path=rel,
            abs_path=path,
            language=language_for(path),
            size=size,
            text=text,
        )


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts):
            continue
        if path.name in SPECIAL_NAMES or path.suffix.lower() in SOURCE_EXTS:
            yield path


def purge_cache(cache_dir: Path) -> int:
    """Remove the entire repo cache. Returns number of repos removed."""
    if not cache_dir.exists():
        return 0
    count = sum(1 for _ in cache_dir.iterdir())
    shutil.rmtree(cache_dir)
    return count
