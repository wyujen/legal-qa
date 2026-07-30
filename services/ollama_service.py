"""使用官方 Ollama Python client 取得結構化法規 QA 回答。"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from models.legal_qa_response import LegalQaResponse
from services.response_validator import ResponseValidationError, parse_response

logger = logging.getLogger(__name__)
_TModel = TypeVar("_TModel", bound=BaseModel)


class OllamaServiceError(RuntimeError):
    """可安全顯示給使用者的 Ollama 錯誤。"""


class OllamaConnectionError(OllamaServiceError):
    """無法連線 Ollama。"""


class OllamaModelNotFoundError(OllamaServiceError):
    """Ollama 找不到設定的模型。"""


class OllamaTimeoutError(OllamaServiceError):
    """Ollama 請求逾時。"""


class OllamaResponseFormatError(OllamaServiceError):
    """模型在一次重試後仍未回傳合法 JSON。"""


def _schema_for(model_type: type[BaseModel]) -> dict[str, Any]:
    schema_builder = getattr(model_type, "model_json_schema", None)
    if callable(schema_builder):
        return schema_builder()
    return model_type.schema()


def _load_default_settings() -> object | None:
    """延後匯入 config，讓 client 注入測試不依賴環境設定。"""

    try:
        from config import get_settings

        return get_settings()
    except (ImportError, AttributeError):
        try:
            from config import settings

            return settings
        except (ImportError, AttributeError):
            return None


def _configured(
    explicit: Any,
    settings: object | None,
    attribute: str,
    fallback: Any,
) -> Any:
    if explicit is not None:
        return explicit
    if settings is not None:
        value = getattr(settings, attribute, None)
        if value is not None:
            return value
    return fallback


def _normalise_messages(
    messages: str | Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if isinstance(messages, str):
        if not messages.strip():
            raise ValueError("提示詞不得為空。")
        return [{"role": "user", "content": messages}]

    normalised: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content.strip():
            raise ValueError("Ollama 訊息必須包含有效的 role 與 content。")
        normalised.append({"role": role, "content": content})
    if not normalised:
        raise ValueError("Ollama 訊息不得為空。")
    return normalised


def _response_content(response: object) -> str:
    message: Any = None
    if isinstance(response, Mapping):
        message = response.get("message")
    else:
        message = getattr(response, "message", None)

    content: Any = None
    if isinstance(message, Mapping):
        content = message.get("content")
    elif message is not None:
        content = getattr(message, "content", None)

    # 兼容極少數簡化測試 client 的回傳格式。
    if content is None:
        if isinstance(response, Mapping):
            content = response.get("response")
        else:
            content = getattr(response, "response", None)

    if content is None or not str(content).strip():
        raise OllamaResponseFormatError("Ollama 回傳空內容。")
    return str(content).strip()


def _friendly_ollama_error(exc: Exception, model: str) -> OllamaServiceError:
    message = str(exc).lower()
    class_name = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None)

    if status_code == 404 or (
        ("model" in message or "manifest" in message)
        and ("not found" in message or "pull" in message)
    ):
        return OllamaModelNotFoundError(
            f"找不到 Ollama 模型「{model}」，請先下載該模型。"
        )
    if (
        isinstance(exc, TimeoutError)
        or "timeout" in class_name
        or "timed out" in message
        or "逾時" in message
    ):
        return OllamaTimeoutError(
            "Ollama 請求逾時，請確認服務與模型狀態後再試一次。"
        )
    if (
        isinstance(exc, ConnectionError)
        or "connect" in class_name
        or "connection" in message
        or "refused" in message
    ):
        return OllamaConnectionError(
            "無法連線到 Ollama，請確認 Ollama 已啟動且服務網址正確。"
        )
    return OllamaServiceError("Ollama 呼叫失敗，請確認本機服務與模型狀態。")


class OllamaService:
    """封裝 Ollama chat 與最多一次 JSON 格式重試。"""

    def __init__(
        self,
        settings: object | None = None,
        *,
        client: object | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = settings or _load_default_settings()
        configured_base_url = _configured(
            base_url,
            settings,
            "ollama_base_url",
            None,
        )
        configured_model = _configured(
            model,
            settings,
            "ollama_chat_model",
            None,
        )
        if not configured_base_url or not str(configured_base_url).strip():
            raise ValueError("缺少 OLLAMA_BASE_URL 設定。")
        if not configured_model or not str(configured_model).strip():
            raise ValueError("缺少 OLLAMA_CHAT_MODEL 設定。")
        self.base_url = str(configured_base_url)
        self.model = str(configured_model)
        self.temperature = float(
            _configured(
                temperature,
                settings,
                "llm_temperature",
                0.1,
            )
        )
        self.top_p = float(
            _configured(top_p, settings, "llm_top_p", 0.9)
        )
        self.max_tokens = int(
            _configured(
                max_tokens,
                settings,
                "llm_max_tokens",
                1200,
            )
        )
        self.thinking = bool(
            _configured(
                thinking,
                settings,
                "llm_thinking",
                False,
            )
        )
        self.timeout_seconds = float(
            _configured(
                timeout_seconds,
                settings,
                "request_timeout_seconds",
                180,
            )
        )
        # 供同一次 QA 流程寫入開發用 JSONL 紀錄；每次 chat 開始時會重設。
        self.last_raw_response = ""
        self._client = client if client is not None else self._create_client()

    def _create_client(self) -> object:
        try:
            import ollama
        except ImportError as exc:
            raise OllamaServiceError(
                "尚未安裝 Ollama Python 套件，請先安裝專案依賴。"
            ) from exc

        try:
            return ollama.Client(
                host=self.base_url,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            logger.exception("建立 Ollama client 失敗")
            raise _friendly_ollama_error(exc, self.model) from exc

    def chat(
        self,
        messages: str | Sequence[Mapping[str, str]],
        *,
        response_model: type[BaseModel] = LegalQaResponse,
    ) -> str:
        """呼叫官方 chat API，並要求依 Pydantic JSON Schema 輸出。"""

        normalised_messages = _normalise_messages(messages)
        self.last_raw_response = ""
        try:
            response = self._client.chat(
                model=self.model,
                messages=normalised_messages,
                think=self.thinking,
                format=_schema_for(response_model),
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                },
            )
            content = _response_content(response)
            self.last_raw_response = content
            return content
        except OllamaServiceError:
            raise
        except Exception as exc:
            logger.exception("Ollama chat 呼叫失敗")
            raise _friendly_ollama_error(exc, self.model) from exc

    # 常見命名別名，讓 QA service 可以保持直觀。
    call = chat
    call_ollama = chat
    generate = chat

    def chat_structured(
        self,
        messages: str | Sequence[Mapping[str, str]],
        *,
        response_model: type[_TModel] = LegalQaResponse,
        max_attempts: int = 2,
    ) -> _TModel:
        """取得已通過 Pydantic 驗證的回答；JSON 錯誤時只重試一次。"""

        if max_attempts not in {1, 2}:
            raise ValueError("格式嘗試次數只能是 1 或 2。")

        working_messages = _normalise_messages(messages)
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            raw_response = ""
            try:
                raw_response = self.chat(
                    working_messages,
                    response_model=response_model,
                )
                return parse_response(raw_response, response_model)
            except (ResponseValidationError, OllamaResponseFormatError) as exc:
                last_error = exc
                logger.warning(
                    "模型第 %d 次回傳的結構化內容驗證失敗",
                    attempt + 1,
                )
                if attempt + 1 < max_attempts:
                    if raw_response.strip():
                        working_messages.append(
                            {"role": "assistant", "content": raw_response}
                        )
                    working_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "上一個回答不符合指定 JSON Schema。"
                                "請修正格式與欄位型別，只輸出完整 JSON 物件；"
                                "不要加入說明、Markdown、HTML 或思考過程。"
                            ),
                        }
                    )

        raise OllamaResponseFormatError(
            "模型回傳的 JSON 格式無法解析，已完成一次格式重試。"
        ) from last_error

    request_structured_response = chat_structured
    generate_structured = chat_structured


def call_ollama(
    prompt: str | Sequence[Mapping[str, str]],
    *,
    service: OllamaService | None = None,
    response_model: type[BaseModel] = LegalQaResponse,
) -> str:
    """便利函式：呼叫 Ollama 並回傳尚待解析的結構化文字。"""

    active_service = service or OllamaService()
    return active_service.chat(prompt, response_model=response_model)


def call_ollama_structured(
    prompt: str | Sequence[Mapping[str, str]],
    *,
    service: OllamaService | None = None,
    response_model: type[_TModel] = LegalQaResponse,
) -> _TModel:
    """便利函式：呼叫 Ollama，格式失敗時重試一次並回傳 Pydantic model。"""

    active_service = service or OllamaService()
    return active_service.chat_structured(
        prompt,
        response_model=response_model,
        max_attempts=2,
    )
