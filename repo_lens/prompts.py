"""Prompt templates used by the chains.

These are intentionally direct. The model gets a short instruction, the
retrieved context, and the user's input. We keep the wording terse so the
model's answers stay close to the source rather than embroidering."""

from langchain_core.prompts import ChatPromptTemplate

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a software project by reading the snippets "
            "below. Be concrete. Quote file paths when you reference code. If the "
            "answer is not present in the snippets, say so plainly rather than "
            "guessing.",
        ),
        (
            "human",
            "Question:\n{question}\n\nRelevant snippets:\n{context}\n\n"
            "Write a focused answer. When you cite code, use the format `path/to/file.py`.",
        ),
    ]
)


FILE_DOC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write developer documentation for a single source file. Keep it "
            "practical: what the file is for, the public API it exposes, and the "
            "relationships to other parts of the project that are visible in the "
            "code. Do not invent behavior that isn't in the source.",
        ),
        (
            "human",
            "File: {path}\nLanguage: {language}\n\nSource:\n```{language}\n{source}\n```\n\n"
            "Write the documentation as Markdown. Start with a short summary "
            "paragraph, then an `## API` section listing exported names with a "
            "one-line description each, then a `## Notes` section for anything "
            "non-obvious. Skip sections that have nothing to say.",
        ),
    ]
)


FUNCTION_EXPLAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You explain code for a developer who is reading it for the first time. "
            "Explain what the function does, what its inputs and outputs mean, and "
            "any side effects. Mention edge cases the code clearly handles. Keep it "
            "short. Never restate the code line by line.",
        ),
        (
            "human",
            "File: {path}\nFunction: {name}\n\n```{language}\n{source}\n```",
        ),
    ]
)


ARCHITECTURE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write a one-page architecture summary of a codebase, given a file "
            "tree and a sample of file headers. Describe the layout, the main "
            "components, and how data or control flows between them. Stay grounded "
            "in what the listing actually shows.",
        ),
        (
            "human",
            "Project: {repo_id}\n\nFile tree (truncated):\n{tree}\n\n"
            "Sampled file headers:\n{samples}\n\n"
            "Produce Markdown with these sections: `## Overview`, `## Layout`, "
            "`## Components`, `## Data flow`, `## Notable choices`. Each section "
            "should be at most a few sentences.",
        ),
    ]
)
