from __future__ import annotations

from typing import TypeAlias, TypedDict, TypeVar

__all__ = [
    "ErrorResponse",
    "EngineResponse",
    "FindResponse",
    "CountResponse",
    "DocResponse",
]


class ErrorResponse(TypedDict):
    error: str


T = TypeVar("T")
EngineResponse: TypeAlias = "ErrorResponse | T"


class FindResponse(TypedDict):
    cnt: int
    segment_by_shard: list[tuple[int, int]]


class CountResponse(TypedDict):
    count: int


class DocResponse(TypedDict):
    doc_ix: int
    doc_len: int
    disp_len: int
    needle_offset: int
    metadata: str
    text: str
