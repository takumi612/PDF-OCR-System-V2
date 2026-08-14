from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable


def iter_runtime_strings(root: Path) -> Iterable[tuple[Path, int, str]]:
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield path, getattr(node, "lineno", 0), node.value


def read_terms(values: list[str], terms_file: Path | None) -> list[str]:
    terms = [value.strip() for value in values if value.strip()]
    if terms_file is not None:
        terms.extend(
            line.strip()
            for line in terms_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    # Longest terms first makes the report easier to inspect.
    return sorted(set(terms), key=lambda value: (-len(value), value.casefold()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan runtime Python string literals for corpus-specific words or "
            "phrases supplied by the caller. The tool has no built-in blacklist."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("src/government_ocr_text_api"),
        help="Runtime source directory to scan",
    )
    parser.add_argument("--term", action="append", default=[])
    parser.add_argument("--terms-file", type=Path)
    args = parser.parse_args()

    terms = read_terms(args.term, args.terms_file)
    if not terms:
        parser.error("provide at least one --term or --terms-file")

    hits: list[tuple[Path, int, str]] = []
    folded_terms = [(term, term.casefold()) for term in terms]
    for path, line, value in iter_runtime_strings(args.root):
        folded_value = value.casefold()
        for term, folded_term in folded_terms:
            if folded_term in folded_value:
                hits.append((path, line, term))

    if not hits:
        print(f"PASS: no supplied corpus terms found in runtime literals under {args.root}")
        return 0

    print("FAIL: corpus terms found in runtime literals")
    for path, line, term in hits:
        print(f"{path}:{line}: {term}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
