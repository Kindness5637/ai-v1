#!/usr/bin/env python3
"""Remove common Project Gutenberg front matter and license boilerplate."""

import argparse
import re
from pathlib import Path


CHAPTER_RE = re.compile(r"^CHAPTER\s+(?:[IVXLCDM]+|\d+)\b.*$", re.MULTILINE)
END_RE = re.compile(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*$", re.MULTILINE)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    start = CHAPTER_RE.search(text)
    if start:
        text = text[start.start():]
    end = END_RE.search(text)
    if end:
        text = text[:end.start()]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/cleaned"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in args.files:
        destination = args.output_dir / source.name
        destination.write_text(clean_text(source.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"{source} -> {destination}")


if __name__ == "__main__":
    main()
