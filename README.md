# repo-lens

A small tool that points an LLM at a GitHub repository so you can:

- ask questions about the codebase in plain English
- generate per-file documentation
- get a one-page architecture summary
- pull up any function and have it explained

It runs locally. Embeddings go into a Chroma collection on disk, so
re-opening a project you've already indexed is instant.

Built with LangChain and ChromaDB. OpenAI is the default LLM provider but
the chains are swappable.

## How it works

1. `loader.py` either clones the repo into a local cache or opens a local
   path. Binary, vendored and oversized files are filtered out.
2. `chunker.py` splits each file with LangChain's language-aware splitter
   so functions and blocks tend to stay together.
3. `store.py` writes those chunks into a per-repo Chroma collection. Each
   repo gets its own folder under `.chroma/`.
4. `chains.py` wires the prompts in `prompts.py` to a `ChatOpenAI` model.
   Retrieval happens against the same Chroma collection.
5. `analyzer.py` walks Python with the AST and JS/TS with a couple of
   small regexes so the UI can list functions and jump straight to one.

The Streamlit app in `app.py` ties it all together. There is also a
`repo-lens` CLI for headless use.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# put your OPENAI_API_KEY into .env
```

## Running

UI:

```bash
streamlit run app.py
```

Then paste a GitHub URL in the sidebar and hit **Index**. The first run
clones the repo and embeds it; subsequent runs reuse the cached
collection. The four tabs are:

- **Ask** — retrieval-augmented Q&A against the codebase.
- **Architecture** — one-page summary built from the file tree and
  sampled file headers.
- **File docs** — pick any file and get a Markdown reference for it.
- **Explain function** — pick a function and get a focused explanation
  with its source shown alongside.

CLI:

```bash
# index a public repo
python -m repo_lens.cli index https://github.com/tiangolo/fastapi

# ask something
python -m repo_lens.cli ask https://github.com/tiangolo/fastapi \
    "where is the dependency injection resolver implemented?"

# generate docs for one file
python -m repo_lens.cli doc https://github.com/tiangolo/fastapi fastapi/applications.py

# one-page architecture summary
python -m repo_lens.cli arch https://github.com/tiangolo/fastapi

# list python functions
python -m repo_lens.cli functions https://github.com/tiangolo/fastapi
```

`repo-lens cache-clear` deletes everything under `.repo_cache/` if you
want to start over.

## Configuration

All settings live in `.env`:

| Variable           | Default                    | Notes                                  |
| ------------------ | -------------------------- | -------------------------------------- |
| `OPENAI_API_KEY`   | _(required)_               |                                        |
| `OPENAI_CHAT_MODEL`| `gpt-4o-mini`              | Used by every chain.                   |
| `OPENAI_EMBED_MODEL`| `text-embedding-3-small`  | 1536-dim embeddings into Chroma.       |
| `CHROMA_DIR`       | `.chroma`                  | Per-repo collections live under here.  |
| `REPO_CACHE_DIR`   | `.repo_cache`              | Cloned repos go here.                  |
| `MAX_FILE_BYTES`   | `204800` (200 KB)          | Anything bigger is skipped.            |

## Project layout

```
repo-lens/
├── app.py                    # Streamlit UI
├── repo_lens/
│   ├── analyzer.py           # AST/regex function discovery
│   ├── chains.py             # LangChain chains for QA, docs, arch, explain
│   ├── chunker.py            # Language-aware splitting
│   ├── cli.py                # `repo-lens ...` command
│   ├── config.py             # Settings loader
│   ├── loader.py             # Clone + walk + filter
│   ├── pipeline.py           # index_repo() and a few helpers
│   ├── prompts.py            # Prompt templates
│   └── store.py              # Chroma wrapper
├── requirements.txt
└── .env.example
```

## Notes

The retrieval step is unfiltered by default but the `Ask` tab and
`--path` flag on the CLI take a substring you can use to scope answers
to a directory. That helps a lot on larger codebases where the same
phrase ("`config`", "`session`") shows up everywhere.

The architecture chain works from a truncated file tree plus the first
~400 characters of up to 30 files. That is intentionally crude — it
keeps the prompt cheap and the output grounded in things the listing
actually contains. If you need a deeper write-up, ask follow-up
questions in the `Ask` tab.

## License

MIT.
