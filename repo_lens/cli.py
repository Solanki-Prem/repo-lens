"""Command-line interface for repo-lens.

Use when you don't want the Streamlit UI — headless scripts,
piping the output somewhere, or just quick one-off questions.

Reads provider + API keys from environment variables (or a .env file).
See README for the variable names.

Examples:
    python -m repo_lens.cli index https://github.com/tiangolo/typer
    python -m repo_lens.cli ask https://github.com/tiangolo/typer "how does typer parse arguments?"
    python -m repo_lens.cli docs https://github.com/tiangolo/typer typer/main.py
    python -m repo_lens.cli arch https://github.com/tiangolo/typer
"""
from pathlib import Path

import click

from .chains import answer_question, architecture_summary, document_file
from .config import Settings
from .loader import language_for, repo_id_for, walk_sources
from .pipeline import file_tree, index_repo, sample_headers
from .store import VectorStore


def _repo_root(settings: Settings, source: str) -> Path:
    """Return the folder that holds the repo's files (cache dir or local path)."""
    cached = settings.repo_cache_dir / repo_id_for(source)
    if cached.exists():
        return cached
    return Path(source).expanduser().resolve()


@click.group()
def cli():
    """repo-lens CLI — clone a repo, index it, ask it questions."""


@cli.command()
@click.argument("source")
@click.option("--rebuild", is_flag=True, help="Delete the previous index first.")
def index(source, rebuild):
    """Clone SOURCE (or open a local path) and embed it into Chroma."""
    settings = Settings.load()
    result = index_repo(settings, source, progress=click.echo, rebuild=rebuild)
    click.echo(
        f"\nIndexed {result.files_indexed} files "
        f"({result.chunks_indexed} chunks) as '{result.repo_id}'."
    )


@cli.command()
@click.argument("source")
@click.argument("question")
@click.option("--k", default=6, show_default=True, help="How many snippets to retrieve.")
def ask(source, question, k):
    """Ask a QUESTION about an already-indexed SOURCE."""
    settings = Settings.load()
    rid = repo_id_for(source)
    store = VectorStore(settings, rid)
    if store.count() == 0:
        click.echo(f"'{rid}' has no chunks. Run 'index {source}' first.")
        return
    answer = answer_question(settings, store, question, k=k)
    click.echo(answer.text)
    click.echo("\nSources:")
    for path in answer.sources:
        click.echo(f"  - {path}")


@cli.command()
@click.argument("source")
@click.argument("file_path")
def docs(source, file_path):
    """Print Markdown docs for one FILE_PATH inside an indexed SOURCE."""
    settings = Settings.load()
    root = _repo_root(settings, source)
    target = root / file_path
    if not target.exists():
        click.echo(f"File not found: {target}")
        return
    text = target.read_text(encoding="utf-8")
    markdown = document_file(settings, file_path, text, language_for(target))
    click.echo(markdown)


@cli.command()
@click.argument("source")
def arch(source):
    """Print a one-page architecture summary for SOURCE."""
    settings = Settings.load()
    rid = repo_id_for(source)
    root = _repo_root(settings, source)
    files = list(walk_sources(root, rid, settings.max_file_bytes))
    markdown = architecture_summary(
        settings,
        repo_id=rid,
        tree=file_tree(root),
        samples=sample_headers(files),
    )
    click.echo(markdown)


if __name__ == "__main__":
    cli()
