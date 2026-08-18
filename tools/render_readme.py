#!/usr/bin/env python3
"""Execute README python blocks and refresh the following output fences."""

from __future__ import annotations

import ast
import io
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
sys.path.insert(0, str(ROOT / "src"))


def _format_value(value: Any) -> str:
    import pandas as pd

    if value is None:
        return ""
    buf = io.StringIO()
    if isinstance(value, (pd.DataFrame, pd.Series)):
        print(value, file=buf)
    elif hasattr(value, "get_figure"):
        return ""
    else:
        print(repr(value), file=buf)
    return buf.getvalue().strip()


def _clean_code(code: str) -> str:
    lines = []
    for line in code.splitlines():
        if line.strip().startswith("#>"):
            continue
        lines.append(line)
    return "\n".join(lines)


def run_python_block(code: str, namespace: dict[str, Any]) -> str:
    code = _clean_code(code)
    if not code.strip():
        return ""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: None  # noqa: ARG005

    tree = ast.parse(code)
    chunks: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Expr):
            value = eval(
                compile(ast.Expression(node.value), "<readme>", "eval"),
                namespace,
            )
            text = _format_value(value)
            if text:
                chunks.append(text)
        else:
            exec(
                compile(ast.Module([node], type_ignores=[]), "<readme>", "exec"),
                namespace,
            )
    return "\n\n".join(chunks).strip()


def split_fences(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    pos = 0
    for match in pattern.finditer(text):
        if match.start() > pos:
            parts.append(("text", text[pos : match.start()]))
        lang = match.group(1)
        parts.append((lang, match.group(2)))
        pos = match.end()
    if pos < len(text):
        parts.append(("text", text[pos:]))
    return parts


def render_readme(dry_run: bool = False) -> list[str]:
    import statassist as sa

    import numpy as np
    import pandas as pd

    raw = README.read_text(encoding="utf-8")
    parts = split_fences(raw)
    out: list[str] = []
    namespace: dict[str, Any] = {"sa": sa, "np": np, "pd": pd, "statassist": sa}
    errors: list[str] = []
    block_idx = 0
    i = 0

    while i < len(parts):
        lang, body = parts[i]
        if lang == "text":
            out.append(body)
            i += 1
            continue

        if lang == "python":
            block_idx += 1
            code = body.strip("\n")
            out.append(f"```python\n{body.rstrip()}\n```")
            try:
                captured = run_python_block(code, namespace)
            except Exception:
                errors.append(
                    f"Block {block_idx} failed:\n{code[:400]}\n{traceback.format_exc()}"
                )
                captured = None
            i += 1
            if i < len(parts) and parts[i][0] == "":
                if captured is not None:
                    out.append(f"```\n{captured}\n```")
                else:
                    out.append(f"```\n{parts[i][1].rstrip()}\n```")
                i += 1
            elif captured:
                out.append(f"```\n{captured}\n```")
            continue

        out.append(f"```{lang}\n{body.rstrip()}\n```" if lang else f"```\n{body.rstrip()}\n```")
        i += 1

    rendered = "".join(out)
    if not dry_run:
        README.write_text(rendered, encoding="utf-8")
    return errors


def main() -> None:
    dry = "--dry-run" in sys.argv
    errors = render_readme(dry_run=dry)
    if errors:
        print("Render errors:")
        for err in errors:
            print(err)
        if not dry:
            print(f"Wrote README with {len(errors)} block error(s); fix and re-run.")
        else:
            raise SystemExit(1)
    print("README rendered successfully." if not dry else "Dry run completed with no errors.")


if __name__ == "__main__":
    main()
