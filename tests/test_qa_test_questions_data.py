from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from models.legal_provision import LegalProvision
from models.qa_test_question import QaTestQuestion
from services.document_service import load_legal_provisions
from services.question_bank_service import load_question_bank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION_BANK_PATH = PROJECT_ROOT / "data" / "qa_test_questions.json"
LEGAL_PROVISIONS_PATH = PROJECT_ROOT / "data" / "legal_provisions.json"


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _merged_provision_text(provisions: list[LegalProvision]) -> str:
    return _normalize_text(
        " ".join(
            part
            for provision in provisions
            for part in (
                provision.document_name,
                provision.chapter_name,
                provision.section_name,
                provision.article_no,
                provision.title,
                provision.content,
            )
            if part
        )
    )


@pytest.fixture(scope="module")
def questions() -> list[QaTestQuestion]:
    return load_question_bank(QUESTION_BANK_PATH)


@pytest.fixture(scope="module")
def active_provisions_by_id() -> dict[int, LegalProvision]:
    return {
        provision.provision_id: provision
        for provision in load_legal_provisions(LEGAL_PROVISIONS_PATH)
        if provision.is_active
    }


def test_question_bank_contains_exactly_100_valid_questions(
    questions: list[QaTestQuestion],
) -> None:
    assert len(questions) == 100


def test_expected_provisions_are_active_and_primary_metadata_matches(
    questions: list[QaTestQuestion],
    active_provisions_by_id: dict[int, LegalProvision],
) -> None:
    for question in questions:
        missing_ids = [
            provision_id
            for provision_id in question.expected_provision_ids
            if provision_id not in active_provisions_by_id
        ]
        assert not missing_ids, (
            f"{question.question_id} 引用了不存在或非 active 的 "
            f"provision_id：{missing_ids}"
        )

        primary = active_provisions_by_id[question.expected_provision_ids[0]]
        assert question.document_name == primary.document_name, (
            f"{question.question_id} 的 document_name 與第一個引用條文不一致"
        )
        assert question.article_no == primary.article_no, (
            f"{question.question_id} 的 article_no 與第一個引用條文不一致"
        )


def test_questions_are_unique_and_cover_at_least_25_documents(
    questions: list[QaTestQuestion],
) -> None:
    normalized_questions = [
        _normalize_text(question.question).casefold()
        for question in questions
    ]

    assert len(normalized_questions) == len(set(normalized_questions))
    assert len({question.document_name for question in questions}) >= 25


def test_each_expected_keyword_is_supported_by_answer_or_cited_provisions(
    questions: list[QaTestQuestion],
    active_provisions_by_id: dict[int, LegalProvision],
) -> None:
    unsupported_keywords: list[str] = []

    for question in questions:
        normalized_answer = _normalize_text(question.expected_answer)
        cited_provisions = [
            active_provisions_by_id[provision_id]
            for provision_id in question.expected_provision_ids
        ]
        normalized_citations = _merged_provision_text(cited_provisions)

        for keyword in question.expected_keywords:
            normalized_keyword = _normalize_text(keyword)
            if not (
                normalized_keyword in normalized_answer
                or normalized_keyword in normalized_citations
            ):
                unsupported_keywords.append(
                    f"{question.question_id}: {keyword}"
                )

    assert not unsupported_keywords, (
        "下列 expected_keyword 未出現在 expected_answer 或引用條文中："
        + "、".join(unsupported_keywords)
    )
