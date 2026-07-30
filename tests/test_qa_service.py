from __future__ import annotations

from config import Settings
from models.legal_qa_response import Citation, LegalQaResponse
from models.retrieval_result import RetrievalResult
from services.qa_service import QAService
from services.retrieval_service import RetrievalDataError


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        provision_id=4,
        document_name="測試法規",
        article_no="第4條",
        title="補件",
        content="申請文件不完備者，應於七日內補件。",
        vector_score=0.8,
        keyword_score=1.0,
        final_score=0.87,
    )


class FakeRetrievalService:
    def __init__(self, results) -> None:
        self.results = results
        self.questions: list[str] = []

    def retrieve(self, question: str):
        self.questions.append(question)
        return self.results


class FailingRetrievalService:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.questions: list[str] = []

    def retrieve(self, question: str):
        self.questions.append(question)
        raise self.error


class FakePromptService:
    def __init__(self) -> None:
        self.calls = []

    def build_messages(self, question, provisions):
        self.calls.append((question, provisions))
        return [{"role": "user", "content": question}]


class FakeOllamaService:
    def __init__(self) -> None:
        self.calls = []
        self.last_raw_response = '{"can_answer":true}'

    def chat_structured(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return LegalQaResponse(
            can_answer=True,
            summary="初步可於七日內補件。",
            conditions=["已收到補件通知"],
            citations=[
                Citation(
                    provision_id=4,
                    document_name="模型捏造名稱",
                    article_no="第999條",
                )
            ],
            notice="僅供內部參考。",
        )


class FakeLogService:
    def __init__(self) -> None:
        self.records = []

    def save_log(self, record) -> bool:
        self.records.append(record)
        return True


def _settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, log_full_prompt=True)


def test_complete_qa_flow_uses_mocks_and_validates_citation(tmp_path) -> None:
    retrieval = FakeRetrievalService([_retrieval_result()])
    prompt = FakePromptService()
    ollama = FakeOllamaService()
    log = FakeLogService()
    service = QAService(
        _settings(tmp_path),
        retrieval_service=retrieval,
        prompt_service=prompt,
        ollama_service=ollama,
        log_service=log,
    )

    result = service.ask("  補資料期限是多久？  ")

    assert result.error is None
    assert result.normalized_question == "補件期限是多久?"
    assert result.response is not None
    assert result.response.can_answer is True
    assert result.response.citations[0].document_name == "測試法規"
    assert result.response.citations[0].article_no == "第4條"
    assert retrieval.questions == ["補件期限是多久?"]
    assert len(prompt.calls) == 1
    assert len(ollama.calls) == 1
    assert log.records[0]["retrieved_provision_ids"] == [4]
    assert log.records[0]["raw_prompt"]
    assert log.records[0]["raw_model_response"]


def test_no_relevant_provision_does_not_call_model(tmp_path) -> None:
    retrieval = FakeRetrievalService([])
    ollama = FakeOllamaService()
    log = FakeLogService()
    service = QAService(
        _settings(tmp_path),
        retrieval_service=retrieval,
        ollama_service=ollama,
        log_service=log,
    )

    result = service.ask("完全無關的問題")

    assert result.error is None
    assert result.response is not None
    assert result.response.can_answer is False
    assert ollama.calls == []
    assert log.records[0]["retrieved_provision_ids"] == []


def test_empty_question_returns_friendly_error_and_logs(tmp_path) -> None:
    log = FakeLogService()
    service = QAService(
        _settings(tmp_path),
        retrieval_service=FakeRetrievalService([]),
        log_service=log,
    )

    result = service.ask("   ")

    assert result.error == "請輸入要解析的問題。"
    assert result.response is None
    assert log.records[0]["error"] == "請輸入要解析的問題。"


def test_database_retrieval_error_is_actionable_without_calling_ollama(
    tmp_path,
) -> None:
    retrieval = FailingRetrievalService(
        RetrievalDataError(
            "無法連線到 PostgreSQL 法規資料庫，請確認資料庫服務已啟動。"
        )
    )
    ollama = FakeOllamaService()
    service = QAService(
        Settings(data_dir=tmp_path),
        retrieval_service=retrieval,
        ollama_service=ollama,
    )

    result = service.ask("這條規定是什麼？")

    assert result.response is None
    assert result.error == (
        "無法連線到 PostgreSQL 法規資料庫，請確認資料庫服務已啟動。"
    )
    assert retrieval.questions == ["這條規定是什麼?"]
    assert ollama.calls == []
