from __future__ import annotations

import os
import sys

from .models import CountResponse, DocResponse, EngineResponse, FindResponse
from .cpp_engine import Engine

_REQUIRED_FILES             = ("data.fm9", "data_offset")
_REQUIRED_FILES_WITH_META   = ("data.fm9", "data_offset", "meta.fm9", "meta_offset")


class InfiniGramMiniEngine:
    """FM-index query engine over one or more pre-built index shards."""

    def __init__(
        self,
        index_dirs: list[str],
        load_to_ram: bool,
        get_metadata: bool,
    ) -> None:
        if sys.byteorder != "little":
            raise RuntimeError("InfiniGramMiniEngine requires a little-endian system.")
        if not isinstance(index_dirs, list) or not all(isinstance(d, str) for d in index_dirs):
            raise TypeError("index_dirs must be a list of strings.")
        if not index_dirs:
            raise ValueError("index_dirs must not be empty.")

        required = _REQUIRED_FILES_WITH_META if get_metadata else _REQUIRED_FILES
        for d in index_dirs:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Index directory not found: {d!r}")
            missing = [f for f in required if not os.path.exists(os.path.join(d, f))]
            if missing:
                raise FileNotFoundError(
                    f"Index directory {d!r} is missing required file(s): {missing}. "
                    "Has indexing completed successfully?"
                )

        self._index_dirs    = index_dirs
        self._load_to_ram   = load_to_ram
        self._get_metadata  = get_metadata
        self._engine        = Engine(index_dirs, load_to_ram, get_metadata)

    @property
    def num_shards(self) -> int:
        """Number of index shards loaded."""
        return len(self._index_dirs)

    def __repr__(self) -> str:
        return (
            f"InfiniGramMiniEngine("
            f"num_shards={self.num_shards}, "
            f"load_to_ram={self._load_to_ram}, "
            f"get_metadata={self._get_metadata})"
        )

    def __enter__(self) -> InfiniGramMiniEngine:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def find(self, query: str) -> EngineResponse[FindResponse]:
        """Return the suffix-array interval [lo, hi) per shard where *query* occurs."""
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__!r}")
        result = self._engine.find(query)
        return {"cnt": result.cnt, "segment_by_shard": result.segment_by_shard}

    def count(self, query: str) -> EngineResponse[CountResponse]:
        """Return the number of times *query* appears in the indexed corpus."""
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__!r}")
        result = self._engine.count(query)
        return {"count": result.count}

    def get_doc_by_rank(
        self,
        s: int,
        rank: int,
        needle_len: int,
        max_ctx_len: int,
    ) -> EngineResponse[DocResponse]:
        """Retrieve the document containing the occurrence at position *rank* in shard *s*."""
        if not 0 <= s < self.num_shards:
            raise ValueError(f"Shard index s={s} is out of range [0, {self.num_shards}).")
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}.")
        if needle_len < 0:
            raise ValueError(f"needle_len must be non-negative, got {needle_len}.")
        if max_ctx_len < 0:
            raise ValueError(f"max_ctx_len must be non-negative, got {max_ctx_len}.")

        result = self._engine.get_doc_by_rank(s, rank, needle_len, max_ctx_len)
        try:
            text = result.text
        except UnicodeDecodeError:
            return {
                "error": (
                    "Failed to decode document text as UTF-8. The context window may be "
                    "cut off mid-character. Try a different max_ctx_len."
                )
            }
        return {
            "doc_ix":        result.doc_ix,
            "doc_len":       result.doc_len,
            "disp_len":      result.disp_len,
            "needle_offset": result.needle_offset,
            "metadata":      result.metadata,
            "text":          text,
        }

    def count_batch(self, queries: list[str]) -> list[EngineResponse[CountResponse]]:
        """Count occurrences for multiple queries. Returns one result per query."""
        return [self.count(q) for q in queries]

    def find_batch(self, queries: list[str]) -> list[EngineResponse[FindResponse]]:
        """Find suffix-array intervals for multiple queries. Returns one result per query."""
        return [self.find(q) for q in queries]

    def search(
        self,
        query: str,
        max_results: int = 10,
        max_ctx_len: int = 200,
    ) -> list[EngineResponse[DocResponse]]:
        """Find occurrences and retrieve up to *max_results* documents round-robin across shards."""
        find_result = self.find(query)
        if "error" in find_result:
            return [find_result]  # type: ignore[list-item]

        needle_len = len(query.encode())
        docs: list[EngineResponse[DocResponse]] = []

        segments = find_result["segment_by_shard"]
        pointers = [lo for lo, _ in segments]
        ends     = [hi for _, hi in segments]

        while len(docs) < max_results:
            advanced = False
            for s in range(self.num_shards):
                if len(docs) >= max_results:
                    break
                if pointers[s] < ends[s]:
                    docs.append(self.get_doc_by_rank(s, pointers[s], needle_len, max_ctx_len))
                    pointers[s] += 1
                    advanced = True
            if not advanced:
                break

        return docs
