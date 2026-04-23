#!/usr/bin/env python3
"""Regenerate the auto-appendix section of structure.md.

The hand-written top portion of structure.md is left untouched.  Everything
between the `auto:begin`/`auto:end` markers is replaced with a deterministic
listing of tracked Python files (plus first-docstring-line and top-level
defs) so `structure.md` stays in sync as the repo grows.

Usage:
    python toolbox/refresh_structure.py          # rewrite in place
    python toolbox/refresh_structure.py --check  # exit 1 if out of date

Runs with the stdlib only on Python 3.6+ so it works on plain NERSC login
nodes (no module load required) and inside the shifter image.
"""

import argparse
import ast
import subprocess
import sys
from typing import List, Optional, Set, Tuple

try:
    from pathlib import Path
except ImportError:  # pragma: no cover - py<3.4
    raise

REPO = Path(__file__).resolve().parent.parent
STRUCT = REPO / "structure.md"
BEGIN = "<!-- auto:begin"
END = "<!-- auto:end -->"

# Only summarize files under these roots.  Anything else (pycache, tmp, etc.)
# is skipped.  Order here is the order it appears in the appendix.
ROOTS = [
    ("Top-level scripts",       ["*.py"], "."),
    ("toolbox/",                ["*.py"], "toolbox"),
    ("toolbox/jaxley_cells/",   ["*.py"], "toolbox/jaxley_cells"),
    ("toolbox/tests/",          ["*.py"], "toolbox/tests"),
    ("scripts/",                ["*.py", "*.sh"], "scripts"),
    ("packBBP3/",               ["*.py"], "packBBP3"),
]


class FileSummary(object):
    __slots__ = ("path", "doc", "defs")

    def __init__(self, path, doc, defs):
        # path: str   doc: str   defs: List[str]
        self.path = path
        self.doc = doc
        self.defs = defs


def tracked_files():
    # type: () -> Set[str]
    out = subprocess.check_output(
        ["git", "ls-files"], cwd=str(REPO)
    ).decode("utf-8").splitlines()
    return set(out)


def summarize(py_path):
    # type: (Path) -> FileSummary
    rel = py_path.relative_to(REPO).as_posix()
    try:
        src = py_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return FileSummary(rel, "", [])
    doc = ""
    defs = []  # type: List[str]
    try:
        tree = ast.parse(src, filename=rel)
    except SyntaxError:
        return FileSummary(rel, "(syntax error)", [])
    module_doc = ast.get_docstring(tree)
    if module_doc:
        doc = module_doc.strip().splitlines()[0].strip()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.append(node.name)
    return FileSummary(rel, doc, defs)


def collect():
    # type: () -> List[Tuple[str, List[FileSummary]]]
    tracked = tracked_files()
    sections = []  # type: List[Tuple[str, List[FileSummary]]]
    for title, patterns, sub in ROOTS:
        base = (REPO / sub).resolve()
        if not base.exists():
            continue
        hits = []  # type: List[FileSummary]
        for pattern in patterns:
            for p in sorted(base.glob(pattern)):
                rel = p.relative_to(REPO).as_posix()
                # Only include tracked files so untracked experiments don't
                # show up in commits.
                if rel not in tracked:
                    continue
                if p.suffix == ".py":
                    hits.append(summarize(p))
                else:
                    hits.append(FileSummary(rel, "", []))
        if hits:
            sections.append((title, hits))
    return sections


def render(sections):
    # type: (List[Tuple[str, List[FileSummary]]]) -> str
    lines = []
    lines.append("## Appendix - auto-generated file inventory")
    lines.append("")
    lines.append("_Regenerate with `python toolbox/refresh_structure.py`._")
    lines.append("")
    for title, files in sections:
        lines.append("### " + title)
        lines.append("")
        for f in files:
            head = "- `" + f.path + "`"
            if f.doc:
                head += " - " + f.doc
            lines.append(head)
            if f.defs:
                joined = ", ".join(f.defs)
                if len(joined) > 240:
                    joined = joined[:237] + "..."
                lines.append("    - defs: " + joined)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def splice(current, auto_body):
    # type: (str, str) -> str
    """Replace the region between `auto:begin`/`auto:end` with `auto_body`."""
    begin = current.find(BEGIN)
    end = current.find(END)
    if begin == -1 or end == -1 or end < begin:
        raise SystemExit(
            "structure.md is missing the 'auto:begin'/'auto:end' markers - "
            "refusing to write. Restore them and re-run."
        )
    begin_line_end = current.find("\n", begin) + 1
    end_line_start = current.rfind("\n", 0, end) + 1
    header = current[:begin_line_end]
    footer = current[end_line_start:]
    return header + auto_body + "\n" + footer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if structure.md would change")
    args = parser.parse_args()

    if not STRUCT.exists():
        sys.stderr.write(str(STRUCT) + " not found\n")
        return 2
    current = STRUCT.read_text(encoding="utf-8")
    sections = collect()
    auto_body = render(sections)
    new = splice(current, auto_body)

    if args.check:
        if new != current:
            sys.stderr.write(
                "structure.md is stale - run: python toolbox/refresh_structure.py\n"
            )
            return 1
        return 0

    if new == current:
        return 0
    STRUCT.write_text(new, encoding="utf-8")
    sys.stdout.write("refreshed " + STRUCT.relative_to(REPO).as_posix() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
