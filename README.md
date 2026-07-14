# repo-lens

A small Streamlit app that points a language model at any GitHub repo so
you can:

- ask questions about the codebase in plain English
- generate per-file documentation
- get a one-page architecture summary
- pull up any Python/JS/TS function and have it explained

Everything runs locally. The repo gets cloned into `.repo_cache/`,
embedded into a per-repo Chroma collection under `.chroma/`, and reused
on subsequent runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

There are two ways to use it — pick whichever you prefer.

### Streamlit UI

```bash
streamlit run app.py
```

Then in the sidebar:

1. Pick a **chat provider** — OpenAI, Anthropic, or Google — and paste
   that provider's API key.
2. Leave **embed provider** on `huggingface` (uses the free HF
   Inference API — get a token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
   or switch to `openai`.
3. Paste a GitHub URL (or a local path), click **Index**, then use the
   tabs: **Ask**, **Architecture**, **File docs**, **Explain function**.

### CLI

For headless use — piping into scripts, quick one-off questions. The
CLI reads provider + API keys from environment variables (or a `.env`
file); see the Configuration section below for the variable names.

```bash
# Index a public repo (clones on first use, reused after)
python -m repo_lens.cli index https://github.com/tiangolo/typer

# Ask a question about it
python -m repo_lens.cli ask https://github.com/tiangolo/typer \
    "How does typer parse CLI arguments?"

# Generate docs for one file
python -m repo_lens.cli docs https://github.com/tiangolo/typer typer/main.py

# One-page architecture summary
python -m repo_lens.cli arch https://github.com/tiangolo/typer
```

Every command supports `--help` for details.

## Configuration

The sidebar values can be pre-seeded from a `.env` file so you don't
retype them every session:

```
CHAT_PROVIDER=openai            # or anthropic, google
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

EMBED_PROVIDER=huggingface      # or openai
HUGGINGFACEHUB_API_TOKEN=hf_...
```

## Project layout

```
repo-lens/
├── app.py                # Streamlit UI
├── repo_lens/
│   ├── config.py         # Settings dataclass + env loader
│   ├── loader.py         # Clone + walk + filter source files
│   ├── chunker.py        # Language-aware text splitting
│   ├── store.py          # Chroma vector-store wrapper + embeddings
│   ├── chains.py         # LangChain chains: ask, docs, explain, arch
│   ├── prompts.py        # Prompt templates
│   ├── analyzer.py       # AST/regex function discovery
│   ├── pipeline.py       # index_repo() glue
│   └── cli.py            # Command-line interface
├── requirements.txt
└── README.md
```

## License

MIT.
