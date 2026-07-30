"""Warm both Ollama models before the Streamlit service becomes ready."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import Settings  # noqa: E402


class ModelWarmupError(RuntimeError):
    """Raised when either required Ollama model cannot be warmed."""


def _chat_content(response: object) -> str:
    message: Any = None
    if isinstance(response, Mapping):
        message = response.get("message")
    else:
        message = getattr(response, "message", None)

    if isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return str(content or "").strip()


def warmup_models(
    settings: object | None = None,
    *,
    client: object | None = None,
    keep_alive: str | None = None,
) -> dict[str, int]:
    """Load the embedding and chat models and keep both ready for QA."""

    settings = settings or Settings()
    active_keep_alive = (
        keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"
    )

    if client is None:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - container dependency
            raise ModelWarmupError("找不到 Ollama Python 套件。") from exc
        client = ollama.Client(
            host=str(getattr(settings, "ollama_base_url")),
            timeout=float(getattr(settings, "request_timeout_seconds")),
        )

    embedding_started = time.perf_counter()
    try:
        client.embed(
            model=str(getattr(settings, "ollama_embedding_model")),
            input=["法規 QA 模型暖機"],
            keep_alive=active_keep_alive,
        )
    except Exception as exc:
        raise ModelWarmupError("Embedding 模型暖機失敗。") from exc
    embedding_ms = round((time.perf_counter() - embedding_started) * 1000)

    chat_started = time.perf_counter()
    try:
        response = client.chat(
            model=str(getattr(settings, "ollama_chat_model")),
            messages=[
                {
                    "role": "user",
                    "content": "這是服務暖機檢查，只回答 OK。",
                }
            ],
            think=False,
            options={"temperature": 0, "num_predict": 2},
            keep_alive=active_keep_alive,
        )
    except Exception as exc:
        raise ModelWarmupError("回答模型暖機失敗。") from exc
    if not _chat_content(response):
        raise ModelWarmupError("回答模型暖機回傳空內容。")
    chat_ms = round((time.perf_counter() - chat_started) * 1000)

    return {
        "embedding_ms": max(0, embedding_ms),
        "chat_ms": max(0, chat_ms),
    }


def main() -> int:
    try:
        report = warmup_models()
    except ModelWarmupError as exc:
        print(f"模型暖機失敗：{exc}", file=sys.stderr)
        return 1

    print(
        "模型暖機完成："
        f"Embedding {report['embedding_ms']} ms，"
        f"回答模型 {report['chat_ms']} ms。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
