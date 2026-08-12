from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
GENERATED_ROOTS = {"data", "outputs"}
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
SENSITIVE_PATTERNS = {
    "absolute_home_path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "email_address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}


def tracked_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
            and path.relative_to(ROOT).parts[0] not in GENERATED_ROOTS
        ]
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> None:
    failures: list[str] = []
    files = tracked_files()
    required = [
        ROOT / "README.md",
        ROOT / "README_KR.md",
        ROOT / "pyproject.toml",
        ROOT / "results" / "p1_qwen_sequential_100x3" / "metrics.json",
    ]
    for path in required:
        if not path.exists():
            failures.append(f"missing required file: {path.relative_to(ROOT)}")

    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append(f"file exceeds 10 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.name == ".env.example":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")

    if failures:
        print("Public-release preflight: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"Public-release preflight: PASS ({len(files)} files checked)")


if __name__ == "__main__":
    main()
