"""Ollama embedding integration.

The service deliberately has a very small surface area.  A client can be
injected for tests; otherwise the official :mod:`ollama` Python client is
created lazily from the application settings.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Base class for actionable embedding errors."""


class OllamaUnavailableError(EmbeddingServiceError):
    """Raised when the local Ollama server cannot be reached."""


class EmbeddingTimeoutError(EmbeddingServiceError):
    """Raised when an Ollama embedding request exceeds the configured timeout."""


class EmbeddingModelNotFoundError(EmbeddingServiceError):
    """Raised when Ollama does not have the configured embedding model."""


class InvalidEmbeddingResponseError(EmbeddingServiceError):
    """Raised when Ollama returns an unusable vector response."""


class EmbeddingService:
    """Generate embeddings with Ollama's official Python client.

    Parameters
    ----------
    settings:
        Any settings object exposing ``ollama_base_url``,
        ``ollama_embedding_model`` and ``request_timeout_seconds``.
        ``embedding_model`` is also accepted for small test doubles.
        ``config.Settings`` is constructed when this argument is omitted.
    client:
        Optional object implementing ``embed(model=..., input=...)``.  This is
        primarily an injection seam for deterministic tests.
    """

    def __init__(self, settings: Any | None = None, client: Any | None = None) -> None:
        if settings is None:
            from config import Settings

            settings = Settings()

        self.settings = settings
        model = getattr(
            settings,
            "ollama_embedding_model",
            getattr(settings, "embedding_model", None),
        )
        if not model:
            raise ValueError(
                "設定缺少 ollama_embedding_model（或 embedding_model）。"
            )
        self.model = str(model)
        self.base_url = str(settings.ollama_base_url)
        self.timeout_seconds = float(settings.request_timeout_seconds)
        self._client = client

    @property
    def client(self) -> Any:
        """Return the injected client or create an official Ollama client."""

        if self._client is None:
            try:
                import ollama
            except ImportError as exc:  # pragma: no cover - environment issue
                raise EmbeddingServiceError(
                    "找不到 ollama Python 套件；請先執行 pip install -r requirements.txt。"
                ) from exc

            self._client = ollama.Client(
                host=self.base_url,
                timeout=self.timeout_seconds,
            )
        return self._client

    def embed_text(self, text: str) -> np.ndarray:
        """Embed one non-empty text and return a one-dimensional float array."""

        vectors = self.embed_texts([text])
        return vectors[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts and return a ``(count, dimension)`` array."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts 必須是字串序列，不可直接傳入單一字串。")

        inputs = list(texts)
        if not inputs:
            return np.empty((0, 0), dtype=np.float32)

        invalid_indexes = [
            index
            for index, value in enumerate(inputs)
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid_indexes:
            indexes = ", ".join(str(index) for index in invalid_indexes)
            raise ValueError(f"Embedding 文字不可為空（索引：{indexes}）。")

        try:
            response = self.client.embed(model=self.model, input=inputs)
        except EmbeddingServiceError:
            raise
        except Exception as exc:  # The client uses httpx and Ollama exceptions.
            logger.exception("Ollama embedding 呼叫失敗")
            raise self._translate_client_error(exc) from exc

        raw_vectors = self._extract_embeddings(response)
        try:
            vectors = np.asarray(raw_vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise InvalidEmbeddingResponseError(
                "Ollama 回傳的 embedding 不是規則的數值矩陣。"
            ) from exc

        if vectors.ndim == 1 and len(inputs) == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise InvalidEmbeddingResponseError(
                f"Ollama embedding 維度格式錯誤：預期 2 維矩陣，實際為 {vectors.ndim} 維。"
            )
        if vectors.shape[0] != len(inputs):
            raise InvalidEmbeddingResponseError(
                "Ollama embedding 筆數不一致："
                f"送出 {len(inputs)} 筆，收到 {vectors.shape[0]} 筆。"
            )
        if vectors.shape[1] == 0:
            raise InvalidEmbeddingResponseError("Ollama 回傳空的 embedding 向量。")
        if not np.isfinite(vectors).all():
            raise InvalidEmbeddingResponseError(
                "Ollama 回傳的 embedding 含 NaN 或 Infinity。"
            )
        return vectors

    # Friendly aliases for callers that distinguish queries and documents.
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_text(text)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_texts(texts)

    @staticmethod
    def _extract_embeddings(response: Any) -> Any:
        if isinstance(response, dict):
            embeddings = response.get("embeddings")
        else:
            embeddings = getattr(response, "embeddings", None)

        if embeddings is None:
            raise InvalidEmbeddingResponseError(
                "Ollama embedding 回應缺少 embeddings 欄位。"
            )
        return embeddings

    def _translate_client_error(self, exc: Exception) -> EmbeddingServiceError:
        message = str(exc).strip()
        lowered = message.casefold()
        status_code = getattr(exc, "status_code", None)
        exception_name = type(exc).__name__.casefold()

        if status_code == 404 or "not found" in lowered or "pull model" in lowered:
            return EmbeddingModelNotFoundError(
                f"Ollama 找不到 embedding 模型「{self.model}」；"
                f"請先執行 ollama pull {self.model}。"
            )

        if (
            isinstance(exc, TimeoutError)
            or "timeout" in lowered
            or "timed out" in lowered
            or "timeout" in exception_name
        ):
            return EmbeddingTimeoutError(
                "Ollama embedding 請求逾時，請確認服務負載、模型狀態與"
                " REQUEST_TIMEOUT_SECONDS 設定。"
            )

        connection_markers = (
            "connect",
            "connection",
            "refused",
            "unreachable",
        )
        if isinstance(exc, (ConnectionError, TimeoutError)) or any(
            marker in lowered or marker in exception_name
            for marker in connection_markers
        ):
            return OllamaUnavailableError(
                f"無法連線到 Ollama（{self.base_url}）；請確認 Ollama 已啟動。"
            )

        return EmbeddingServiceError(
            f"Ollama embedding 呼叫失敗（模型：{self.model}），"
            "請查看執行紀錄。"
        )


__all__ = [
    "EmbeddingModelNotFoundError",
    "EmbeddingService",
    "EmbeddingServiceError",
    "EmbeddingTimeoutError",
    "InvalidEmbeddingResponseError",
    "OllamaUnavailableError",
]
