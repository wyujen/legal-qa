"""解析模型 JSON，並以本次檢索結果驗證與校正回答。"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from models.legal_qa_response import LEGAL_NOTICE, Citation, LegalQaResponse

DEFAULT_MAX_LIST_ITEMS = 6
_TModel = TypeVar("_TModel", bound=BaseModel)
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_DANGEROUS_BLOCK_RE = re.compile(
    r"<(?:script|style)\b[^>]*>.*?</(?:script|style)\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_PROVISION_MARKER = r"\[\s*provision\s*id\s*=\s*\d+\s*\]"
_PROVISION_MARKER_CLUSTER_RE = re.compile(
    _PROVISION_MARKER
    + rf"(?:\s*(?:、|,|，|和|及|與)\s*{_PROVISION_MARKER})*",
    flags=re.IGNORECASE,
)
_PROVISION_FIELD_RE = re.compile(
    r"""["']?\bprovision[\s_-]*id\b["']?\s*[:=]\s*["']?\d+["']?""",
    flags=re.IGNORECASE,
)


class ResponseValidationError(ValueError):
    """模型回答不是可安全顯示的 LegalQaResponse。"""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _strip_html(value: str) -> str:
    """移除標籤及 script/style 區塊，只保留可顯示純文字。"""

    # 先解碼 entity 再解析，並重複少量次數，避免
    # ``&amp;lt;script&amp;gt;`` 在清理完成後重新變成 HTML。
    text = value
    for _ in range(3):
        decoded = html.unescape(text)
        without_dangerous_blocks = _DANGEROUS_BLOCK_RE.sub("", decoded)
        parser = _TextExtractor()
        try:
            parser.feed(without_dangerous_blocks)
            parser.close()
            cleaned = "".join(parser.parts)
        except Exception:
            cleaned = re.sub(r"<[^>]*>", "", without_dangerous_blocks)
        if cleaned == text:
            text = cleaned
            break
        text = cleaned

    # 不讓殘留或過度編碼的角括號形成任何可解讀標籤。
    text = html.unescape(text)
    text = _DANGEROUS_BLOCK_RE.sub("", text)
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    # 裸角括號可能是合法的數值比較；改成全形可保留法律條件語意，
    # 同時確保輸出不可能被 HTML renderer 當成標籤。
    return text.replace("<", "＜").replace(">", "＞").strip()


def _sanitize_display_text(value: str) -> str:
    """Remove HTML and internal citation markers from user-visible text."""

    text = _strip_html(value)
    text = _PROVISION_MARKER_CLUSTER_RE.sub("", text)
    text = _PROVISION_FIELD_RE.sub("", text)
    text = re.sub(r"\s+([，。；：、,.!?])", r"\1", text)
    return " ".join(text.split()).strip(" \t\r\n,，、;；:：")


def _model_validate(model_type: type[_TModel], payload: Any) -> _TModel:
    validator = getattr(model_type, "model_validate", None)
    if callable(validator):
        return validator(payload)
    return model_type.parse_obj(payload)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper(mode="python")
    return model.dict()


def _extract_json_text(raw_response: str) -> str:
    text = raw_response.strip()
    if not text:
        raise ResponseValidationError("模型回傳空內容。")

    fenced = _CODE_FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()

    # 結構化輸出偶爾仍會附帶一句前言；只抽出最外層 JSON 物件。
    if not text.startswith("{") or not text.endswith("}"):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ResponseValidationError("模型回傳的 JSON 格式無法解析。")
        text = text[start : end + 1]
    return text


def parse_response(
    raw_response: str | bytes | Mapping[str, Any] | LegalQaResponse,
    response_model: type[_TModel] = LegalQaResponse,
) -> _TModel:
    """將模型輸出解析為指定 Pydantic model，不回傳未驗證文字。"""

    if isinstance(raw_response, response_model):
        parsed = raw_response
    else:
        payload: Any
        if isinstance(raw_response, bytes):
            try:
                raw_response = raw_response.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ResponseValidationError(
                    "模型回傳內容不是有效的 UTF-8。"
                ) from exc

        if isinstance(raw_response, str):
            try:
                payload = json.loads(_extract_json_text(raw_response))
            except json.JSONDecodeError as exc:
                raise ResponseValidationError(
                    "模型回傳的 JSON 格式無法解析。"
                ) from exc
        elif isinstance(raw_response, Mapping):
            payload = dict(raw_response)
        else:
            raise ResponseValidationError("模型回傳格式不受支援。")

        try:
            parsed = _model_validate(response_model, payload)
        except (ValidationError, TypeError, ValueError) as exc:
            raise ResponseValidationError("模型回傳內容不符合指定格式。") from exc

    # 將最基本的語意完整性納入 structured retry 邊界。
    summary = getattr(parsed, "summary", None)
    if summary is not None and not str(summary).strip():
        raise ResponseValidationError("模型回答的初步結論不得為空。")
    return parsed


def _field(item: object, name: str, default: Any = "") -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _clean_list(values: Sequence[str], maximum: int) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = _sanitize_display_text(str(value))
        if text:
            cleaned.append(text)
        if len(cleaned) >= maximum:
            break
    return cleaned


def validate_response(
    response: LegalQaResponse | Mapping[str, Any] | str | bytes,
    retrieval_results: Sequence[object],
    *,
    max_list_items: int = DEFAULT_MAX_LIST_ITEMS,
) -> LegalQaResponse:
    """驗證 citation allowlist、覆寫本地欄位並清除不可顯示內容。"""

    if max_list_items < 1:
        raise ValueError("列表項目數量上限必須大於零。")

    parsed = parse_response(response, LegalQaResponse)
    payload = _model_dump(parsed)

    summary = _sanitize_display_text(str(payload.get("summary", "")))
    if not summary:
        raise ResponseValidationError("模型回答的初步結論不得為空。")

    local_by_id: dict[int, object] = {}
    for result in retrieval_results:
        raw_id = _field(result, "provision_id", None)
        try:
            provision_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        local_by_id.setdefault(provision_id, result)

    valid_citations: list[Citation] = []
    seen_ids: set[int] = set()
    for citation in parsed.citations:
        provision_id = int(citation.provision_id)
        local = local_by_id.get(provision_id)
        if local is None or provision_id in seen_ids:
            continue
        seen_ids.add(provision_id)
        valid_citations.append(
            Citation(
                provision_id=provision_id,
                # Citation 顯示值完全以本地檢索資料為準，不採信模型文字。
                document_name=str(_field(local, "document_name", "")).strip(),
                article_no=str(_field(local, "article_no", "")).strip(),
            )
        )
        if len(valid_citations) >= max_list_items:
            break

    unsupported_claim = bool(parsed.can_answer and not valid_citations)
    if unsupported_claim:
        summary = "模型回答沒有可驗證的引用條文，因此無法提供受支持的初步結論。"
        conditions: list[str] = []
        exceptions: list[str] = []
        missing_information = [
            "請確認檢索結果是否包含足以支持結論的條文。"
        ]
    else:
        conditions = _clean_list(parsed.conditions, max_list_items)
        exceptions = _clean_list(parsed.exceptions, max_list_items)
        missing_information = _clean_list(
            parsed.missing_information,
            max_list_items,
        )

    payload.update(
        {
            "can_answer": bool(parsed.can_answer and valid_citations),
            "summary": summary,
            "conditions": conditions,
            "exceptions": exceptions,
            "missing_information": missing_information,
            "citations": valid_citations,
            # 免責文字由本地程式固定，不信任模型可能竄改的 notice。
            "notice": LEGAL_NOTICE,
        }
    )

    try:
        return _model_validate(LegalQaResponse, payload)
    except (ValidationError, TypeError, ValueError) as exc:
        raise ResponseValidationError("模型回傳內容驗證失敗。") from exc


class ResponseValidator:
    """保留列表上限設定的回答驗證器。"""

    def __init__(self, max_list_items: int = DEFAULT_MAX_LIST_ITEMS) -> None:
        if max_list_items < 1:
            raise ValueError("列表項目數量上限必須大於零。")
        self.max_list_items = max_list_items

    def parse(
        self,
        raw_response: str | bytes | Mapping[str, Any] | LegalQaResponse,
    ) -> LegalQaResponse:
        return parse_response(raw_response, LegalQaResponse)

    def validate(
        self,
        response: LegalQaResponse | Mapping[str, Any] | str | bytes,
        retrieval_results: Sequence[object],
    ) -> LegalQaResponse:
        return validate_response(
            response,
            retrieval_results,
            max_list_items=self.max_list_items,
        )
