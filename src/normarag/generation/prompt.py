"""Prompt templates: citation-required and abstention-aware."""


def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    raise NotImplementedError("M2")


def build_baseline_prompt(question: str) -> str:
    raise NotImplementedError("M3")
