"""協調正規化、檢索、LLM、回答驗證與 QA 紀錄。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from config import Settings, get_settings
from models.legal_qa_response import (
    LEGAL_NOTICE,
    LegalQaResponse,
    LegalQaResult,
)
from models.retrieval_result import RetrievalResult
from services.embedding_service import EmbeddingServiceError
from services.log_service import LogService
from services.ollama_service import OllamaService, OllamaServiceError
from services.prompt_service import PromptService
from services.response_validator import (
    ResponseValidationError,
    ResponseValidator,
)
from services.retrieval_service import RetrievalDataError, RetrievalService
from services.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class QAService:
    """法規 QA 的單一應用服務邊界，所有外部依賴皆可注入測試替身。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        normalizer: object | None = None,
        retrieval_service: object | None = None,
        prompt_service: object | None = None,
        ollama_service: object | None = None,
        response_validator: object | None = None,
        log_service: object | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.normalizer = normalizer or TextNormalizer()
        # RetrievalService 會驗證磁碟索引，延後到 ask 才建立，讓 UI 即使
        # 尚未建立 embeddings 也能正常啟動並顯示可操作錯誤。
        self.retrieval_service = retrieval_service
        self.prompt_service = prompt_service or PromptService()
        self.ollama_service = ollama_service
        self.response_validator = response_validator or ResponseValidator(
            self.settings.max_list_items
        )
        self.log_service = log_service or LogService(self.settings.qa_logs_path)

    def ask(self, question: str) -> LegalQaResult:
        """執行一次完整 QA；失敗時回傳可安全顯示的繁中錯誤。"""

        started_at = time.perf_counter()
        query_id = str(uuid4())
        normalized_question = ""
        retrieval_results: list[RetrievalResult] = []
        raw_prompt: str | None = None
        raw_model_response = ""
        validated_response: LegalQaResponse | None = None
        error: str | None = None

        try:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("請輸入要解析的問題。")

            normalized_question = self._normalize(question)
            if not normalized_question:
                raise ValueError("正規化後的問題為空，請重新輸入。")

            retrieval_results = self._retrieve(normalized_question)
            if not retrieval_results:
                validated_response = LegalQaResponse(
                    can_answer=False,
                    summary="目前找不到足以回答此問題的相關條文。",
                    missing_information=[
                        "請換一種方式描述問題，或確認法規資料與 Embedding 已更新。"
                    ],
                    notice=LEGAL_NOTICE,
                )
            else:
                messages = self._build_messages(
                    normalized_question,
                    retrieval_results,
                )
                if self.settings.log_full_prompt:
                    raw_prompt = json.dumps(messages, ensure_ascii=False)

                parsed_response = self._chat_structured(messages)
                raw_model_response = str(
                    getattr(self.ollama_service, "last_raw_response", "") or ""
                )
                validated_response = self._validate_response(
                    parsed_response,
                    retrieval_results,
                )
        except (
            ValueError,
            RetrievalDataError,
            EmbeddingServiceError,
            OllamaServiceError,
            ResponseValidationError,
        ) as exc:
            error = str(exc)
            raw_model_response = str(
                getattr(self.ollama_service, "last_raw_response", "") or ""
            )
            logger.warning("法規 QA 無法完成：%s", exc)
        except Exception:
            error = "系統處理問題時發生錯誤，請查看執行紀錄後再試一次。"
            raw_model_response = str(
                getattr(self.ollama_service, "last_raw_response", "") or ""
            )
            logger.exception("法規 QA 發生未預期錯誤")

        duration_ms = max(0, round((time.perf_counter() - started_at) * 1000))
        result = LegalQaResult(
            query_id=query_id,
            question=question if isinstance(question, str) else str(question),
            normalized_question=normalized_question,
            response=validated_response,
            retrieval_results=retrieval_results,
            duration_ms=duration_ms,
            error=error,
        )
        self._save_log(
            result=result,
            raw_prompt=raw_prompt,
            raw_model_response=raw_model_response,
        )
        return result

    async def ask_async(self, question: str) -> LegalQaResult:
        """在非同步呼叫端將同步 Ollama/檔案工作移至執行緒。"""

        return await asyncio.to_thread(self.ask, question)

    def _normalize(self, question: str) -> str:
        normalize = getattr(self.normalizer, "normalize", None)
        if callable(normalize):
            return str(normalize(question))
        if callable(self.normalizer):
            return str(self.normalizer(question))
        raise TypeError("normalizer 必須是 callable 或提供 normalize()。")

    def _retrieve(self, normalized_question: str) -> list[RetrievalResult]:
        if self.retrieval_service is None:
            self.retrieval_service = RetrievalService(self.settings)
        retrieve = getattr(self.retrieval_service, "retrieve", None)
        if not callable(retrieve):
            retrieve = getattr(self.retrieval_service, "search", None)
        if not callable(retrieve):
            raise TypeError("retrieval_service 必須提供 retrieve() 或 search()。")
        # Prompt 與 citation allowlist 必須使用完全相同、最多 6 筆的集合。
        return list(retrieve(normalized_question))[:6]

    def _build_messages(
        self,
        question: str,
        retrieval_results: list[RetrievalResult],
    ) -> list[dict[str, str]]:
        builder = getattr(self.prompt_service, "build_messages", None)
        if not callable(builder):
            raise TypeError("prompt_service 必須提供 build_messages()。")
        return list(builder(question, retrieval_results))

    def _chat_structured(
        self,
        messages: list[dict[str, str]],
    ) -> LegalQaResponse:
        if self.ollama_service is None:
            self.ollama_service = OllamaService(self.settings)
        chat = getattr(self.ollama_service, "chat_structured", None)
        if not callable(chat):
            raise TypeError("ollama_service 必須提供 chat_structured()。")
        response = chat(
            messages,
            response_model=LegalQaResponse,
            max_attempts=2,
        )
        if isinstance(response, LegalQaResponse):
            return response
        if isinstance(response, Mapping):
            return LegalQaResponse.model_validate(response)
        raise ResponseValidationError("模型服務未回傳合法的結構化回答。")

    def _validate_response(
        self,
        response: LegalQaResponse,
        retrieval_results: list[RetrievalResult],
    ) -> LegalQaResponse:
        validate = getattr(self.response_validator, "validate", None)
        if callable(validate):
            return validate(response, retrieval_results)
        if callable(self.response_validator):
            return self.response_validator(response, retrieval_results)
        raise TypeError("response_validator 必須是 callable 或提供 validate()。")

    def _save_log(
        self,
        *,
        result: LegalQaResult,
        raw_prompt: str | None,
        raw_model_response: str,
    ) -> None:
        retrieval_scores = [
            {
                "provision_id": item.provision_id,
                "vector_score": item.vector_score,
                "keyword_score": item.keyword_score,
                "final_score": item.final_score,
            }
            for item in result.retrieval_results
        ]
        record = {
            "query_id": result.query_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question": result.question,
            "normalized_question": result.normalized_question,
            "retrieved_provision_ids": [
                item.provision_id for item in result.retrieval_results
            ],
            "retrieval_scores": retrieval_scores,
            "model_name": self.settings.ollama_chat_model,
            "embedding_model": self.settings.ollama_embedding_model,
            "raw_prompt": raw_prompt,
            "raw_model_response": raw_model_response,
            "validated_response": (
                result.response.model_dump(mode="json")
                if result.response is not None
                else {}
            ),
            "duration_ms": result.duration_ms,
            "error": result.error,
            "feedback": None,
            "feedback_reason": None,
        }
        saver = getattr(self.log_service, "save_log", None)
        if not callable(saver):
            saver = getattr(self.log_service, "save", None)
        if callable(saver):
            try:
                saver(record)
            except Exception:
                # 注入的自訂 logger 也不可破壞主流程。
                logger.exception("自訂 QA Log Service 發生錯誤")
        else:
            logger.error("log_service 未提供 save_log() 或 save()")


def create_qa_service(settings: Settings | None = None) -> QAService:
    """建立使用本地檔案與 Ollama 的預設 QA Service。"""

    return QAService(settings=settings)


async def ask(question: str) -> LegalQaResult:
    """符合非同步整合情境的 module-level 便利介面。"""

    return await create_qa_service().ask_async(question)
