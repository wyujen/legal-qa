"""Evaluate whether question-bank items retrieve their expected provisions."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import Settings  # noqa: E402
from models.qa_test_question import QaTestQuestion  # noqa: E402
from models.retrieval_result import RetrievalResult  # noqa: E402
from services.question_bank_service import load_question_bank  # noqa: E402
from services.qa_service import QAService  # noqa: E402
from services.retrieval_service import RetrievalService  # noqa: E402


class QuestionRetriever(Protocol):
    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]: ...


@dataclass(frozen=True)
class RetrievalMiss:
    question_id: str
    question: str
    expected_ids: tuple[int, ...]
    retrieved_ids: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationReport:
    question_count: int
    hit_count: int
    reciprocal_rank_sum: float
    misses: tuple[RetrievalMiss, ...]

    @property
    def recall(self) -> float:
        if self.question_count == 0:
            return 0.0
        return self.hit_count / self.question_count

    @property
    def mean_reciprocal_rank(self) -> float:
        if self.question_count == 0:
            return 0.0
        return self.reciprocal_rank_sum / self.question_count


class _DiscardLogService:
    """Keep evaluation queries out of the user-facing QA history."""

    @staticmethod
    def save_log(record: object) -> None:
        del record


def evaluate_questions(
    questions: Sequence[QaTestQuestion],
    retriever: QuestionRetriever,
    *,
    top_k: int,
) -> EvaluationReport:
    """Measure expected-provision Recall@K and MRR without using gold in search."""

    if top_k < 1:
        raise ValueError("top_k 必須至少為 1。")

    hit_count = 0
    reciprocal_rank_sum = 0.0
    misses: list[RetrievalMiss] = []

    for question in questions:
        results = retriever.retrieve(question.question, top_k=top_k)
        retrieved_ids = tuple(result.provision_id for result in results)
        expected_ids = tuple(question.expected_provision_ids)
        expected_set = set(expected_ids)
        matching_rank = next(
            (
                rank
                for rank, provision_id in enumerate(retrieved_ids, start=1)
                if provision_id in expected_set
            ),
            None,
        )

        if matching_rank is None:
            misses.append(
                RetrievalMiss(
                    question_id=question.question_id,
                    question=question.question,
                    expected_ids=expected_ids,
                    retrieved_ids=retrieved_ids,
                )
            )
            continue

        hit_count += 1
        reciprocal_rank_sum += 1 / matching_rank

    return EvaluationReport(
        question_count=len(questions),
        hit_count=hit_count,
        reciprocal_rank_sum=reciprocal_rank_sum,
        misses=tuple(misses),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "逐題送出真正的問題文字，檢查預期條文是否出現在檢索結果中。"
        )
    )
    parser.add_argument(
        "--questions",
        type=Path,
        help="題庫 JSON；預設使用 DATA_DIR/qa_test_questions.json。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="檢索結果數；預設使用 RETRIEVAL_TOP_K。",
    )
    parser.add_argument(
        "--minimum-recall",
        type=float,
        help=(
            "可選的 Recall@K 門檻；未指定時只報告結果，不因未達滿分失敗。"
        ),
    )
    parser.add_argument(
        "--live-ids",
        help=(
            "再以真正回答模型驗證指定題號，以逗號分隔；"
            "例如 Q001,Q020,Q080。"
        ),
    )
    return parser


def _evaluate_live_answers(
    questions: Sequence[QaTestQuestion],
    settings: Settings,
    question_ids: Sequence[str],
) -> bool:
    questions_by_id = {question.question_id: question for question in questions}
    unknown_ids = [
        question_id
        for question_id in question_ids
        if question_id not in questions_by_id
    ]
    if unknown_ids:
        print(f"找不到題號：{', '.join(unknown_ids)}", file=sys.stderr)
        return False

    service = QAService(settings, log_service=_DiscardLogService())
    all_passed = True
    for question_id in question_ids:
        question = questions_by_id[question_id]
        result = service.ask(question.question)
        response = result.response
        cited_ids = (
            {
                citation.provision_id
                for citation in response.citations
            }
            if response is not None
            else set()
        )
        expected_ids = set(question.expected_provision_ids)
        smoke_passed = bool(
            result.error is None
            and response is not None
            and response.can_answer
            and cited_ids.intersection(expected_ids)
        )
        all_passed = all_passed and smoke_passed
        status = "SMOKE_PASS" if smoke_passed else "SMOKE_FAIL"
        summary = response.summary if response is not None else ""
        retrieved_ids = [
            retrieval.provision_id
            for retrieval in result.retrieval_results
        ]
        print(
            f"{status} {question_id} {result.duration_ms} ms；"
            f"expected={sorted(expected_ids)}；"
            f"retrieved={retrieved_ids}；"
            f"cited={sorted(cited_ids)}；{summary}"
        )

    return all_passed


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings()
    question_path = args.questions or settings.qa_test_questions_path
    top_k = args.top_k or settings.retrieval_top_k

    if top_k < 1:
        print("--top-k 必須至少為 1。", file=sys.stderr)
        return 2
    if (
        args.minimum_recall is not None
        and not 0 <= args.minimum_recall <= 1
    ):
        print("--minimum-recall 必須介於 0 與 1 之間。", file=sys.stderr)
        return 2

    questions = load_question_bank(question_path)
    report = evaluate_questions(
        questions,
        RetrievalService(settings),
        top_k=top_k,
    )

    print(
        f"題數 {report.question_count}；"
        f"命中 {report.hit_count}；"
        f"Recall@{top_k} {report.recall:.1%}；"
        f"MRR {report.mean_reciprocal_rank:.3f}"
    )
    for miss in report.misses:
        print(
            f"MISS {miss.question_id}：expected={list(miss.expected_ids)}；"
            f"retrieved={list(miss.retrieved_ids)}；{miss.question}"
        )

    retrieval_passed = (
        args.minimum_recall is None
        or report.recall >= args.minimum_recall
    )
    if not args.live_ids:
        return 0 if retrieval_passed else 1

    live_ids = [
        value.strip()
        for value in args.live_ids.split(",")
        if value.strip()
    ]
    if not live_ids:
        print("--live-ids 不可為空。", file=sys.stderr)
        return 2
    live_passed = _evaluate_live_answers(
        questions,
        settings,
        live_ids,
    )
    return 0 if retrieval_passed and live_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
