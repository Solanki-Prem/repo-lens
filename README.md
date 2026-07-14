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

See `PROJECT_GUIDE.md` for a full walkthrough — architecture,
concepts, interview Q&A, and how to demo it.

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
│   └── pipeline.py       # index_repo() glue
├── requirements.txt
└── README.md
```

## License

MIT.
