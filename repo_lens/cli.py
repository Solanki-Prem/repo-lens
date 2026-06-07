from __future__ import annotations

from pathlib import Path

import click

from .analyzer import list_python_functions
from .chains import answer_question, architecture_summary, document_file
from .config import Settings
from .loader import purge_cache, repo_id_for, walk_sources
from .pipeline import file_tree, index_repo, sample_headers
from .store import VectorStore


@click.group()
def cli():
    """Command line interface for repo-lens."""


@cli.command()
@click.argument("source")
@click.option("--rebuild", is_flag=True, help="Drop the existing index and start fresh.")
def index(source: str, rebuild: bool):
    """Clone (or open) SOURCE and embed it into Chroma."""
    settings = Settings.load()
    result = index_repo(settings, source, progress=click.echo, rebuild=rebuild)
    click.echo(
        f"Indexed {result.files_indexed} files / {result.chunks_indexed} chunks "
        f"as `{result.repo_id}`."
    )


@cli.command()
@click.argument("source")
@click.argument("question", nargs=-1)
@click.option("--k", default=6, show_default=True, help="Snippets to retrieve.")
@click.option("--path", default=None, help="Restrict retrieval to this path substring.")
def ask(source: str, question, k: int, path: str | None):
    """Ask a question about an already-indexed SOURCE."""
    if not question:
        raise click.UsageError("Provide a question after the source.")
    settings = Settings.load()
    rid = repo_id_for(source)
    store = VectorStore(settings, rid)
    if store.count() == 0:
        raise click.UsageError(
            f"`{rid}` has no chunks. Run `repo-lens index {source}` first."
        )
    ans = answer_question(settings, store, " ".join(question), k=k, path_prefix=path)
    click.echo(ans.text)
    click.echo("\nsources:")
    for s in ans.sources:
        click.echo(f"  - {s}")


@cli.command()
@click.argument("source")
@click.argument("file_path")
def doc(source: str, file_path: str):
    """Generate Markdown documentation for one file inside an indexed repo."""
    settings = Settings.load()
    rid = repo_id_for(source)
    cache_root = settings.repo_cache_dir / rid
    root = cache_root if cache_root.exists() else Path(source).expanduser().resolve()
    target = root / file_path
    if not target.exists():
        raise click.UsageError(f"No such file: {target}")
    text = target.read_text(encoding="utf-8")
    from .loader import _language_for  # noqa: WPS437

    md = document_file(settings, file_path, text, _language_for(target))
    click.echo(md)


@cli.command()
@click.argument("source")
def arch(source: str):
    """Print an architecture summary for an indexed repo."""
    settings = Settings.load()
    rid = repo_id_for(source)
    cache_root = settings.repo_cache_dir / rid
    root = cache_root if cache_root.exists() else Path(source).expanduser().resolve()
    files = list(walk_sources(root, rid, settings.max_file_bytes))
    md = architecture_summary(
        settings,
        repo_id=rid,
        tree=file_tree(root),
        samples=sample_headers(files),
    )
    click.echo(md)


@cli.command(name="functions")
@click.argument("source")
def list_functions(source: str):
    """List Python functions in an indexed repo (path:line qualname)."""
    settings = Settings.load()
    rid = repo_id_for(source)
    cache_root = settings.repo_cache_dir / rid
    root = cache_root if cache_root.exists() else Path(source).expanduser().resolve()
    refs = list_python_functions(root, rid)
    for r in refs:
        click.echo(f"{r.path}:{r.start_line}  {r.qualname}")


@cli.command()
def cache_clear():
    """Remove all cloned repos from the local cache."""
    settings = Settings.load()
    n = purge_cache(settings.repo_cache_dir)
    click.echo(f"Removed {n} cached repo(s).")


if __name__ == "__main__":
    cli()
