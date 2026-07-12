"""Golden dataset schema, loader and reference validation (CLAUDE.md §8).

The golden set is the project's most valuable asset. Claude Code provides the
schema, loader and validator; the real 60-80 Q&A dataset is built and validated
by hand. Each example:

    {
      "id": "q001",
      "question": "...",
      "answer": "Respuesta de referencia concisa.",
      "references": [{"chapter": "1-13", "numeral": "2.1"}],
      "answerable": true,
      "difficulty": "medium",
      "source_excerpt": "Fragmento que sustenta la respuesta.",
      "tags": ["plazos"],
      "origin": "seed" | "human"
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chapter: str
    numeral: str

    def key(self) -> tuple[str, str]:
        return (self.chapter, self.numeral)


class GoldenExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    answer: str
    references: list[Reference] = []
    answerable: bool
    difficulty: Literal["easy", "medium", "hard"]
    source_excerpt: str = ""
    tags: list[str] = []
    origin: Literal["seed", "human"]

    @model_validator(mode="after")
    def _check_answerable_references(self) -> GoldenExample:
        if self.answerable and not self.references:
            raise ValueError("answerable=true requires at least one reference")
        if not self.answerable and self.references:
            raise ValueError("answerable=false requires references to be empty")
        return self


def load_golden(path: str | Path) -> list[GoldenExample]:
    """Load and validate a golden dataset from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    examples: list[GoldenExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            examples.append(GoldenExample.model_validate_json(line))
    return examples


def known_reference_keys(chunks: list[dict]) -> set[tuple[str, str]]:
    """Set of (chapter_code, numeral) that actually exist in the corpus."""
    return {(c["chapter_code"], c["numeral"]) for c in chunks}


def missing_references(
    example: GoldenExample, known: set[tuple[str, str]]
) -> list[tuple[str, str]]:
    """References of an example that do not exist in the corpus."""
    return [ref.key() for ref in example.references if ref.key() not in known]
