"""RAGAS wrappers + custom citation-accuracy metric."""


def citation_validity(citations: list[dict], known_chunks: list[dict]) -> float:
    raise NotImplementedError("M3")


def citation_exactness(citations: list[dict], references: list[dict]) -> dict:
    raise NotImplementedError("M3")


def abstention_score(abstained: bool, answerable: bool) -> int:
    raise NotImplementedError("M3")
