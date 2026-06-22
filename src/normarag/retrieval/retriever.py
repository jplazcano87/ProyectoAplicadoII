"""Dense + sparse hybrid retrieval with RRF fusion."""


class Retriever:
    def retrieve(self, query: str) -> list[dict]:
        raise NotImplementedError("M2/M4")
