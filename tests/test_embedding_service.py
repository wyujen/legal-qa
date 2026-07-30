from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from models.legal_provision import LegalProvision
from scripts.build_embeddings import build_embeddings
from services.embedding_service import (
    EmbeddingModelNotFoundError,
    EmbeddingService,
    EmbeddingTimeoutError,
    InvalidEmbeddingResponseError,
    OllamaUnavailableError,
)


def make_settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "ollama_base_url": "http://localhost:11434",
        "ollama_embedding_model": "embeddinggemma",
        "request_timeout_seconds": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def embed(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_embed_texts_uses_official_batch_shape_and_model() -> None:
    client = FakeClient({"embeddings": [[1, 2], [3, 4]]})
    service = EmbeddingService(make_settings(), client=client)

    vectors = service.embed_texts(["甲", "乙"])

    np.testing.assert_array_equal(
        vectors,
        np.array([[1, 2], [3, 4]], dtype=np.float32),
    )
    assert client.calls == [
        {"model": "embeddinggemma", "input": ["甲", "乙"]}
    ]


def test_invalid_embedding_count_is_rejected() -> None:
    service = EmbeddingService(
        make_settings(),
        client=FakeClient({"embeddings": [[1, 2]]}),
    )

    with pytest.raises(InvalidEmbeddingResponseError, match="筆數不一致"):
        service.embed_texts(["甲", "乙"])


def test_missing_model_error_includes_pull_command() -> None:
    error = RuntimeError("model not found")
    setattr(error, "status_code", 404)
    service = EmbeddingService(make_settings(), client=FakeClient(error))

    with pytest.raises(
        EmbeddingModelNotFoundError,
        match=r"ollama pull embeddinggemma",
    ):
        service.embed_text("甲")


def test_connection_error_is_actionable() -> None:
    service = EmbeddingService(
        make_settings(),
        client=FakeClient(ConnectionError("connection refused")),
    )

    with pytest.raises(OllamaUnavailableError, match="Ollama 已啟動"):
        service.embed_text("甲")


def test_timeout_error_is_distinct_from_connection_error() -> None:
    service = EmbeddingService(
        make_settings(),
        client=FakeClient(TimeoutError("timed out")),
    )

    with pytest.raises(EmbeddingTimeoutError, match="請求逾時"):
        service.embed_text("甲")


class FakeBatchEmbeddingService:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.batches.append(texts)
        return np.array(
            [[float(len(text)), float(index)] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


def test_build_embeddings_writes_matrix_and_complete_metadata(tmp_path) -> None:
    provisions = [
        LegalProvision(
            provision_id=index,
            document_name="測試法規",
            article_no=f"第{index}條",
            content=f"內容 {index}",
            sort_order=index,
        )
        for index in (1, 2, 3)
    ]
    provisions_path = tmp_path / "legal_provisions.json"
    embeddings_path = tmp_path / "legal_embeddings.npy"
    metadata_path = tmp_path / "embedding_metadata.json"
    provisions_path.write_text(
        json.dumps(
            [provision.model_dump(mode="json") for provision in provisions],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = make_settings(
        legal_provisions_path=provisions_path,
        legal_embeddings_path=embeddings_path,
        embedding_metadata_path=metadata_path,
    )
    fake = FakeBatchEmbeddingService()

    metadata = build_embeddings(
        settings=settings,
        embedding_service=fake,
        batch_size=2,
    )

    matrix = np.load(embeddings_path, allow_pickle=False)
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert matrix.shape == (3, 2)
    assert len(fake.batches) == 2
    assert saved_metadata == metadata
    assert metadata["status"] == "ready"
    assert metadata["embedding_model"] == "embeddinggemma"
    assert metadata["vector_dimension"] == 2
    assert metadata["provision_count"] == 3
    assert metadata["provision_ids"] == [1, 2, 3]
    assert metadata["created_at"]
    assert len(metadata["content_fingerprint"]) == 64
