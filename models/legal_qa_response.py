"""結構化法規解析回覆與完整查詢結果。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.retrieval_result import RetrievalResult


LEGAL_NOTICE = "本回答僅供內部初步法規解析，不構成正式法律意見。"
REQUIRED_RESPONSE_FIELDS = [
    "can_answer",
    "summary",
    "conditions",
    "exceptions",
    "missing_information",
    "citations",
    "notice",
]


class Citation(BaseModel):
    """模型回答中經 allowlist 驗證後的法條引用。"""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        json_schema_extra={"additionalProperties": False},
    )

    provision_id: int = Field(gt=0)
    document_name: str
    article_no: str


class LegalQaResponse(BaseModel):
    """LLM 必須產生且通過驗證的 JSON 結構。"""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        json_schema_extra={
            "additionalProperties": False,
            "required": REQUIRED_RESPONSE_FIELDS,
        },
    )

    can_answer: bool
    summary: str
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    notice: str = LEGAL_NOTICE


class LegalQaResult(BaseModel):
    """QA Service 提供給 UI 的單次處理結果。"""

    query_id: str
    question: str
    normalized_question: str
    response: LegalQaResponse | None = None
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    duration_ms: int = Field(default=0, ge=0)
    error: str | None = None
