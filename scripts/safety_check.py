from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PDF_ROOT = (ROOT / "data" / "demo").resolve()
FORBIDDEN_NAMES = {".env"}
FORBIDDEN_SUFFIXES = {".pem", ".key"}
GENERATED_DIRS = {"chroma", ".cache"}


def main() -> int:
    problems: list[str] = []

    for path in ROOT.rglob("*"):
        if any(part in GENERATED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.is_dir():
            continue

        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"secret-like file: {path.relative_to(ROOT)}")

        if path.suffix.lower() == ".pdf":
            resolved = path.resolve()
            if ALLOWED_PDF_ROOT not in resolved.parents:
                problems.append(f"PDF outside synthetic demo corpus: {path.relative_to(ROOT)}")

    if problems:
        print("Safety check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Safety check passed: synthetic PDFs only and no obvious secret files shipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
