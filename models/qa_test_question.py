"""Validated question-bank item used by offline QA evaluation."""

from __future__ import annotations

import unicodedata
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)


QuestionId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
    ),
]
QuestionText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=4_000,
    ),
]
ExpectedAnswer = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=20_000,
    ),
]
DocumentName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]
ArticleNo = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=100,
    ),
]
ExpectedKeyword = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=200,
    ),
]
ProvisionId = Annotated[int, Field(strict=True, gt=0)]


def _normalized_unique_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


class QaTestQuestion(BaseModel):
    """One expected QA case from an externally supplied question bank."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    question_id: QuestionId
    question: QuestionText
    expected_answer: ExpectedAnswer
    document_name: DocumentName
    article_no: ArticleNo
    expected_keywords: list[ExpectedKeyword] = Field(
        min_length=1,
        max_length=100,
    )
    expected_provision_ids: list[ProvisionId] = Field(
        min_length=1,
        max_length=100,
    )

    @field_validator("expected_keywords")
    @classmethod
    def validate_unique_keywords(cls, values: list[str]) -> list[str]:
        normalized = [_normalized_unique_text(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("expected_keywords 不可包含重複項目")
        return values

    @field_validator("expected_provision_ids")
    @classmethod
    def validate_unique_provision_ids(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)):
            raise ValueError("expected_provision_ids 不可包含重複項目")
        return values


__all__ = ["QaTestQuestion"]
