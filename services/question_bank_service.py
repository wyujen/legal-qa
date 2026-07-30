"""Load and validate an externally supplied QA test question bank."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from models.qa_test_question import QaTestQuestion


DEFAULT_MAX_QUESTION_COUNT = 5_000
MAX_SOURCE_BYTES = 32 * 1024 * 1024


class QuestionBankError(ValueError):
    """Base class for safe, user-displayable question-bank errors."""


class QuestionBankReadError(QuestionBankError):
    """The requested question-bank file could not be read."""


class QuestionBankFormatError(QuestionBankError):
    """The JSON payload or one of its questions is invalid."""


def _normalized_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _validation_field(error: ValidationError) -> str:
    details = error.errors(include_url=False)
    if not details:
        return "資料"
    location = details[0].get("loc", ())
    if not location:
        return "資料"
    return ".".join(str(part) for part in location)


class QuestionBankService:
    """Parse a Path, UTF-8 JSON bytes, or an in-memory JSON string."""

    def __init__(
        self,
        *,
        max_questions: int = DEFAULT_MAX_QUESTION_COUNT,
    ) -> None:
        if (
            isinstance(max_questions, bool)
            or not isinstance(max_questions, int)
            or max_questions < 1
        ):
            raise ValueError("max_questions 必須是大於 0 的整數。")
        self.max_questions = max_questions

    def load(
        self,
        source: Path | bytes | str,
    ) -> list[QaTestQuestion]:
        raw = self._read_source(source)
        payload = self._parse_json(raw)
        records = self._question_records(payload)

        if not records:
            raise QuestionBankFormatError("題庫不可為空。")
        if len(records) > self.max_questions:
            raise QuestionBankFormatError(
                f"題庫題數不可超過 {self.max_questions} 題。"
            )

        questions: list[QaTestQuestion] = []
        seen_ids: dict[str, int] = {}
        seen_questions: dict[str, int] = {}
        for index, record in enumerate(records, start=1):
            try:
                question = QaTestQuestion.model_validate(record)
            except ValidationError as exc:
                field = _validation_field(exc)
                raise QuestionBankFormatError(
                    f"題庫第 {index} 題的「{field}」欄位格式無效。"
                ) from exc

            normalized_id = _normalized_key(question.question_id)
            previous_id_index = seen_ids.get(normalized_id)
            if previous_id_index is not None:
                raise QuestionBankFormatError(
                    "題庫 question_id 不可重複："
                    f"第 {previous_id_index} 題與第 {index} 題。"
                )
            seen_ids[normalized_id] = index

            normalized_question = _normalized_key(question.question)
            previous_question_index = seen_questions.get(
                normalized_question
            )
            if previous_question_index is not None:
                raise QuestionBankFormatError(
                    "題庫問題文字不可重複："
                    f"第 {previous_question_index} 題與第 {index} 題。"
                )
            seen_questions[normalized_question] = index
            questions.append(question)

        return questions

    @staticmethod
    def _read_source(source: Path | bytes | str) -> bytes | str:
        if isinstance(source, Path):
            try:
                if source.stat().st_size > MAX_SOURCE_BYTES:
                    raise QuestionBankFormatError("題庫 JSON 檔案過大。")
                return source.read_bytes()
            except QuestionBankError:
                raise
            except FileNotFoundError as exc:
                raise QuestionBankReadError(
                    "找不到題庫檔案，請確認路徑。"
                ) from exc
            except (OSError, PermissionError) as exc:
                raise QuestionBankReadError(
                    "無法讀取題庫檔案，請確認檔案權限。"
                ) from exc

        if isinstance(source, bytes):
            if len(source) > MAX_SOURCE_BYTES:
                raise QuestionBankFormatError("題庫 JSON 內容過大。")
            return source

        if isinstance(source, str):
            try:
                encoded_size = len(source.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise QuestionBankFormatError(
                    "題庫 JSON 字串不是有效的 Unicode 文字。"
                ) from exc
            if encoded_size > MAX_SOURCE_BYTES:
                raise QuestionBankFormatError("題庫 JSON 內容過大。")
            return source

        raise TypeError("題庫來源必須是 Path、JSON bytes 或 JSON 字串。")

    @staticmethod
    def _parse_json(raw: bytes | str) -> Any:
        if isinstance(raw, bytes):
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise QuestionBankFormatError(
                    "題庫檔案必須使用 UTF-8 編碼。"
                ) from exc
        else:
            text = raw.removeprefix("\ufeff")

        if not text.strip():
            raise QuestionBankFormatError("題庫 JSON 內容不可為空。")
        try:
            return json.loads(
                text,
                parse_constant=_reject_non_standard_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise QuestionBankFormatError(
                "題庫 JSON 格式無效，請檢查語法。"
            ) from exc

    @staticmethod
    def _question_records(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if set(payload) != {"questions"}:
                raise QuestionBankFormatError(
                    "題庫 JSON 物件必須只包含 questions 欄位。"
                )
            questions = payload["questions"]
            if not isinstance(questions, list):
                raise QuestionBankFormatError(
                    "題庫 questions 欄位必須是陣列。"
                )
            return questions
        raise QuestionBankFormatError(
            "題庫 JSON 根節點必須是陣列或包含 questions 的物件。"
        )


def load_question_bank(
    source: Path | bytes | str,
    *,
    max_questions: int = DEFAULT_MAX_QUESTION_COUNT,
) -> list[QaTestQuestion]:
    """Convenience wrapper for callers that do not need a service instance."""

    return QuestionBankService(max_questions=max_questions).load(source)


__all__ = [
    "DEFAULT_MAX_QUESTION_COUNT",
    "QuestionBankError",
    "QuestionBankFormatError",
    "QuestionBankReadError",
    "QuestionBankService",
    "load_question_bank",
]
