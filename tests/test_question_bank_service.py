from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.qa_test_question import QaTestQuestion
from services.question_bank_service import (
    QuestionBankFormatError,
    QuestionBankReadError,
    QuestionBankService,
    load_question_bank,
)


def make_question(
    question_id: str = "Q-001",
    *,
    question: str = "學生因重病經核准，最多可以延期註冊多久？",
    expected_answer: str = "至多二星期。",
    document_name: str = "朝陽科技大學學則",
    article_no: str = "第十三條",
    expected_keywords: list[str] | None = None,
    expected_provision_ids: list[int] | None = None,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": question,
        "expected_answer": expected_answer,
        "document_name": document_name,
        "article_no": article_no,
        "expected_keywords": (
            ["重病", "延期註冊", "二星期"]
            if expected_keywords is None
            else expected_keywords
        ),
        "expected_provision_ids": (
            [53, 134]
            if expected_provision_ids is None
            else expected_provision_ids
        ),
    }


def encode(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_loads_root_array_from_json_string() -> None:
    questions = load_question_bank(encode([make_question()]))

    assert len(questions) == 1
    assert isinstance(questions[0], QaTestQuestion)
    assert questions[0].question_id == "Q-001"
    assert questions[0].document_name == "朝陽科技大學學則"
    assert questions[0].article_no == "第十三條"
    assert questions[0].expected_provision_ids == [53, 134]


def test_loads_questions_object_from_utf8_bom_bytes() -> None:
    payload = {"questions": [make_question()]}
    raw = b"\xef\xbb\xbf" + encode(payload).encode("utf-8")

    questions = load_question_bank(raw)

    assert [item.question_id for item in questions] == ["Q-001"]


def test_loads_path_as_utf8_and_preserves_question_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qa_questions.json"
    path.write_text(
        encode(
            [
                make_question("Q-001"),
                make_question(
                    "Q-002",
                    question="休學期限最多多久？",
                    expected_answer="至多二學年。",
                    article_no="第四十二條",
                    expected_keywords=["休學", "二學年"],
                    expected_provision_ids=[83],
                ),
            ]
        ),
        encoding="utf-8",
    )

    questions = QuestionBankService().load(path)

    assert [item.question_id for item in questions] == ["Q-001", "Q-002"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "根節點"),
        ({"items": []}, "questions"),
        ({"questions": "not-an-array"}, "必須是陣列"),
        ({"questions": [], "version": 1}, "只包含 questions"),
        ([], "不可為空"),
        ({"questions": []}, "不可為空"),
    ],
)
def test_rejects_invalid_root_shapes(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(QuestionBankFormatError, match=message):
        load_question_bank(encode(payload))


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        ({"question_id": ""}, "question_id"),
        ({"question_id": 1}, "question_id"),
        ({"question": "   "}, "question"),
        ({"expected_answer": ""}, "expected_answer"),
        ({"document_name": ""}, "document_name"),
        ({"article_no": ""}, "article_no"),
        ({"expected_keywords": []}, "expected_keywords"),
        ({"expected_keywords": "重病"}, "expected_keywords"),
        ({"expected_keywords": ["重病", 2]}, "expected_keywords"),
        (
            {"expected_keywords": ["ＡＢＣ", "abc"]},
            "expected_keywords",
        ),
        ({"expected_provision_ids": []}, "expected_provision_ids"),
        ({"expected_provision_ids": [0]}, "expected_provision_ids"),
        ({"expected_provision_ids": [True]}, "expected_provision_ids"),
        ({"expected_provision_ids": ["53"]}, "expected_provision_ids"),
        ({"expected_provision_ids": [53, 53]}, "expected_provision_ids"),
    ],
)
def test_strongly_validates_required_question_fields(
    updates: dict[str, object],
    field: str,
) -> None:
    payload = make_question()
    payload.update(updates)

    with pytest.raises(QuestionBankFormatError, match=field):
        load_question_bank(encode([payload]))


@pytest.mark.parametrize(
    "missing_field",
    [
        "question_id",
        "question",
        "expected_answer",
        "document_name",
        "article_no",
        "expected_keywords",
        "expected_provision_ids",
    ],
)
def test_all_schema_fields_are_required(missing_field: str) -> None:
    payload = make_question()
    del payload[missing_field]

    with pytest.raises(QuestionBankFormatError, match=missing_field):
        load_question_bank(encode([payload]))


def test_rejects_unknown_question_fields() -> None:
    payload = make_question()
    payload["model_answer"] = "不應出現在輸入格式"

    with pytest.raises(QuestionBankFormatError, match="model_answer"):
        load_question_bank(encode([payload]))


def test_rejects_duplicate_ids_after_unicode_and_case_normalization() -> None:
    payload = [
        make_question("Ｑ-001"),
        make_question(
            "q-001",
            question="另一道問題？",
        ),
    ]

    with pytest.raises(QuestionBankFormatError, match="question_id 不可重複"):
        load_question_bank(encode(payload))


def test_rejects_duplicate_questions_after_whitespace_normalization() -> None:
    payload = [
        make_question("Q-001", question="如何 延期註冊？"),
        make_question("Q-002", question="  如何　延期註冊？  "),
    ]

    with pytest.raises(QuestionBankFormatError, match="問題文字不可重複"):
        load_question_bank(encode(payload))


def test_enforces_configured_question_count_limit() -> None:
    payload = [
        make_question("Q-001"),
        make_question("Q-002", question="第二題？"),
        make_question("Q-003", question="第三題？"),
    ]

    with pytest.raises(QuestionBankFormatError, match="不可超過 2 題"):
        load_question_bank(encode(payload), max_questions=2)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_max_questions_must_be_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="大於 0 的整數"):
        QuestionBankService(max_questions=value)  # type: ignore[arg-type]


def test_invalid_json_error_does_not_echo_payload() -> None:
    secret = "do-not-expose-this-value"

    with pytest.raises(QuestionBankFormatError) as caught:
        load_question_bank(f'{{"questions":[{secret}]')

    assert str(caught.value) == "題庫 JSON 格式無效，請檢查語法。"
    assert secret not in str(caught.value)


def test_non_utf8_bytes_have_safe_error() -> None:
    with pytest.raises(
        QuestionBankFormatError,
        match="必須使用 UTF-8",
    ):
        load_question_bank(b"\xff\xfe\x00")


def test_missing_path_error_does_not_echo_local_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "private-folder" / "secret-bank.json"

    with pytest.raises(QuestionBankReadError) as caught:
        load_question_bank(missing)

    assert str(caught.value) == "找不到題庫檔案，請確認路徑。"
    assert str(missing) not in str(caught.value)


def test_string_source_is_json_content_not_an_implicit_path() -> None:
    with pytest.raises(QuestionBankFormatError, match="JSON 格式無效"):
        load_question_bank("missing-question-bank.json")


def test_rejects_unsupported_source_type() -> None:
    with pytest.raises(TypeError, match="Path、JSON bytes 或 JSON 字串"):
        load_question_bank(123)  # type: ignore[arg-type]
