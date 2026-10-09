from pydantic import BaseModel, ConfigDict, Field


class _RetrievalParams(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    rfp_ids: list[str] | None = None  # None = search all RFP
    top_k: int | None = Field(default=None, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=0, le=1)


class AskRequest(_RetrievalParams):
    question: str = Field(max_length=1)


class SearchRequest(_RetrievalParams):
    query: str = Field(min_length=1)


class SourceItem(BaseModel):
    rfp_id: str
    filename: str
    page_number: int
    chunk_id: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class SearchResponse(BaseModel):
    results: list[SourceItem]
