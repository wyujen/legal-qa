from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from models.legal_qa_response import LegalQaResponse
from services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseFormatError,
    OllamaService,
)


VALID_RESPONSE = """{
  "can_answer": false,
  "summary": "參考條文不足。",
  "conditions": [],
  "exceptions": [],
  "missing_information": ["需要更多資料"],
  "citations": [],
  "notice": "僅供內部初步解析。"
}"""


class FakeClient:
    def __init__(self, responses: Sequence[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _service(client: FakeClient) -> OllamaService:
    return OllamaService(
        client=client,
        base_url="http://test.invalid",
        model="test-model",
        temperature=0.1,
        top_p=0.9,
        max_tokens=1200,
        thinking=False,
        timeout_seconds=1,
    )


def test_chat_uses_official_structured_format_and_options() -> None:
    client = FakeClient([{"message": {"content": VALID_RESPONSE}}])
    service = _service(client)

    raw = service.chat([{"role": "user", "content": "問題"}])

    assert raw == VALID_RESPONSE
    request = client.calls[0]
    assert request["model"] == "test-model"
    assert request["messages"] == [{"role": "user", "content": "問題"}]
    assert request["format"] == LegalQaResponse.model_json_schema()
    assert request["think"] is False
    assert set(request["format"]["required"]) == {
        "can_answer",
        "summary",
        "conditions",
        "exceptions",
        "missing_information",
        "citations",
        "notice",
    }
    assert request["format"]["additionalProperties"] is False
    assert (
        request["format"]["$defs"]["Citation"]["additionalProperties"]
        is False
    )
    assert request["options"] == {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 1200,
    }


def test_structured_chat_retries_invalid_json_once() -> None:
    client = FakeClient(
        [
            {"message": {"content": "不是 JSON"}},
            {"message": {"content": VALID_RESPONSE}},
        ]
    )
    service = _service(client)

    parsed = service.chat_structured(
        [{"role": "system", "content": "規則"}, {"role": "user", "content": "問題"}]
    )

    assert parsed.summary == "參考條文不足。"
    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "不是 JSON"}
    assert "不符合指定 JSON Schema" in retry_messages[-1]["content"]


def test_structured_chat_stops_after_one_retry() -> None:
    client = FakeClient(
        [
            {"message": {"content": "壞格式一"}},
            {"message": {"content": "壞格式二"}},
        ]
    )
    service = _service(client)

    with pytest.raises(OllamaResponseFormatError, match="已完成一次格式重試"):
        service.chat_structured([{"role": "user", "content": "問題"}])

    assert len(client.calls) == 2


def test_structured_chat_retries_an_empty_first_response() -> None:
    client = FakeClient(
        [
            {"message": {"content": " "}},
            {"message": {"content": VALID_RESPONSE}},
        ]
    )
    service = _service(client)

    parsed = service.chat_structured([{"role": "user", "content": "問題"}])

    assert parsed.summary == "參考條文不足。"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "user"


def test_structured_chat_retries_an_empty_summary() -> None:
    empty_summary = VALID_RESPONSE.replace(
        '"summary": "參考條文不足。"',
        '"summary": "   "',
    )
    client = FakeClient(
        [
            {"message": {"content": empty_summary}},
            {"message": {"content": VALID_RESPONSE}},
        ]
    )
    service = _service(client)

    parsed = service.chat_structured([{"role": "user", "content": "問題"}])

    assert parsed.summary == "參考條文不足。"
    assert len(client.calls) == 2


def test_connection_error_is_translated_to_traditional_chinese() -> None:
    client = FakeClient([ConnectionError("connection refused")])
    service = _service(client)

    with pytest.raises(OllamaConnectionError, match="無法連線到 Ollama"):
        service.chat([{"role": "user", "content": "問題"}])


def test_empty_response_is_rejected() -> None:
    client = FakeClient([{"message": {"content": "  "}}])
    service = _service(client)

    with pytest.raises(OllamaResponseFormatError, match="回傳空內容"):
        service.chat([{"role": "user", "content": "問題"}])
