from __future__ import annotations

from models.qa_test_question import QaTestQuestion
from models.retrieval_result import RetrievalResult
from scripts.evaluate_question_bank import evaluate_questions


def _question(
    question_id: str,
    expected_provision_id: int,
) -> QaTestQuestion:
    return QaTestQuestion(
        question_id=question_id,
        question=f"{question_id} 的口語問題",
        expected_answer="有法規支持的答案",
        document_name="測試法規",
        article_no="第一條",
        expected_keywords=["答案"],
        expected_provision_ids=[expected_provision_id],
    )


def _result(provision_id: int) -> RetrievalResult:
    return RetrievalResult(
        provision_id=provision_id,
        document_name="測試法規",
        article_no="第一條",
        title="",
        content="測試內容",
        vector_score=0.8,
        keyword_score=0.5,
        final_score=0.7,
    )


class FakeRetriever:
    def __init__(self, results_by_question: dict[str, list[int]]) -> None:
        self.results_by_question = results_by_question

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        ids = self.results_by_question[question]
        return [_result(provision_id) for provision_id in ids[:top_k]]


def test_evaluate_questions_reports_recall_mrr_and_misses() -> None:
    questions = [_question("Q001", 10), _question("Q002", 20)]
    retriever = FakeRetriever(
        {
            "Q001 的口語問題": [99, 10, 30],
            "Q002 的口語問題": [40, 50, 60],
        }
    )

    report = evaluate_questions(questions, retriever, top_k=3)

    assert report.question_count == 2
    assert report.hit_count == 1
    assert report.recall == 0.5
    assert report.mean_reciprocal_rank == 0.25
    assert [miss.question_id for miss in report.misses] == ["Q002"]
    assert report.misses[0].retrieved_ids == (40, 50, 60)


def test_evaluate_questions_accepts_any_expected_provision() -> None:
    question = _question("Q001", 10).model_copy(
        update={"expected_provision_ids": [10, 11]}
    )
    retriever = FakeRetriever({"Q001 的口語問題": [11]})

    report = evaluate_questions([question], retriever, top_k=1)

    assert report.recall == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert not report.misses
