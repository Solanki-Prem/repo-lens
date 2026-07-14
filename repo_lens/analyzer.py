from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class FunctionRef:
    path: str
    name: str
    qualname: str
    start_line: int
    end_line: int
    language: str
    source: str


def list_python_functions(root: Path, repo_id: str) -> List[FunctionRef]:
    """Walk a Python repo and return every top-level or class-level function."""
    out: List[FunctionRef] = []
    for py in root.rglob("*.py"):
        rel_parts = py.relative_to(root).parts
        if any(part.startswith(".") or part in {"node_modules", "venv"} for part in rel_parts):
            continue
        try:
            text = py.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        rel = py.relative_to(root).as_posix()
        lines = text.splitlines(keepends=True)
        out.extend(_visit_module(tree, lines, rel))
    return out


def _visit_module(tree: ast.Module, lines: List[str], rel: str) -> Iterable[FunctionRef]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield _make_ref(node, lines, rel, qualname=node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield _make_ref(sub, lines, rel, qualname=f"{node.name}.{sub.name}")


def _make_ref(node, lines: List[str], rel: str, qualname: str) -> FunctionRef:
    start = node.lineno
    end = getattr(node, "end_lineno", start)
    source = "".join(lines[start - 1 : end])
    return FunctionRef(
        path=rel,
        name=node.name,
        qualname=qualname,
        start_line=start,
        end_line=end,
        language="python",
        source=source,
    )


_JS_FUNC_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
    re.MULTILINE,
)
_JS_CONST_FN_RE = re.compile(
    r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(",
    re.MULTILINE,
)


def list_js_like_functions(root: Path, exts={".js", ".jsx", ".ts", ".tsx"}) -> List[FunctionRef]:
    """Crude function lister for JS/TS. Good enough for name + jump-to-line."""
    out: List[FunctionRef] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in exts:
            continue
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") or part in {"node_modules", "dist", "build"} for part in rel_parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(root).as_posix()
        for rx in (_JS_FUNC_RE, _JS_CONST_FN_RE):
            for m in rx.finditer(text):
                name = m.group(1)
                start_line = text.count("\n", 0, m.start()) + 1
                end_line = _find_block_end(text, m.start())
                src = "\n".join(text.splitlines()[start_line - 1 : end_line])
                out.append(
                    FunctionRef(
                        path=rel,
                        name=name,
                        qualname=name,
                        start_line=start_line,
                        end_line=end_line,
                        language="typescript" if path.suffix in {".ts", ".tsx"} else "javascript",
                        source=src,
                    )
                )
    return out


def _find_block_end(text: str, start: int) -> int:
    """Best-effort: walk forward from start, count braces, stop at depth 0."""
    depth = 0
    started = False
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
            started = True
        elif c == "}":
            depth -= 1
            if started and depth == 0:
                return text.count("\n", 0, i) + 1
        i += 1
    return text.count("\n", 0, len(text))
