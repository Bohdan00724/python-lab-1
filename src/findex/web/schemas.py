from pydantic import BaseModel, Field

from findex.ranking import ScorerName


class SearchHitOut(BaseModel):
    doc_id: int
    score: float
    path: str
    snippet: str


class SearchResponseOut(BaseModel):
    query: str
    scorer: ScorerName
    page: int
    k: int
    total: int
    total_pages: int
    has_prev: bool
    has_next: bool
    elapsed_ms: float
    results: list[SearchHitOut]


class DocumentOut(BaseModel):
    doc_id: int
    path: str
    length: int
    content: str


class TermStatOut(BaseModel):
    term: str
    df: int


class StatsResponseOut(BaseModel):
    num_docs: int
    num_terms: int
    avg_doc_length: float
    store_positions: bool
    top_terms: list[TermStatOut] = Field(default_factory=list[TermStatOut])


class HealthResponseOut(BaseModel):
    status: str
    index_loaded: bool
    num_docs: int
