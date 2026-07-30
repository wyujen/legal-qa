from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scripts.warmup_models import ModelWarmupError, warmup_models


class FakeWarmupClient:
    def __init__(self, chat_content: str = "OK") -> None:
        self.chat_content = chat_content
        self.embed_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []

    def embed(self, **kwargs: Any) -> dict[str, object]:
        self.embed_calls.append(kwargs)
        return {"embeddings": [[0.1, 0.2]]}

    def chat(self, **kwargs: Any) -> dict[str, object]:
        self.chat_calls.append(kwargs)
        return {"message": {"content": self.chat_content}}


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        ollama_base_url="http://test.invalid",
        ollama_chat_model="gemma4:e2b-it-qat",
        ollama_embedding_model="embeddinggemma",
        request_timeout_seconds=360,
    )


def test_warmup_loads_both_models_with_thinking_disabled() -> None:
    client = FakeWarmupClient()

    report = warmup_models(
        _settings(),
        client=client,
        keep_alive="30m",
    )

    assert report["embedding_ms"] >= 0
    assert report["chat_ms"] >= 0
    assert client.embed_calls == [
        {
            "model": "embeddinggemma",
            "input": ["法規 QA 模型暖機"],
            "keep_alive": "30m",
        }
    ]
    assert client.chat_calls[0]["model"] == "gemma4:e2b-it-qat"
    assert client.chat_calls[0]["think"] is False
    assert client.chat_calls[0]["options"] == {
        "temperature": 0,
        "num_predict": 2,
    }
    assert client.chat_calls[0]["keep_alive"] == "30m"


def test_warmup_rejects_empty_chat_response() -> None:
    with pytest.raises(ModelWarmupError, match="空內容"):
        warmup_models(
            _settings(),
            client=FakeWarmupClient(" "),
            keep_alive="30m",
        )
