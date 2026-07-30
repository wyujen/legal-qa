"""Streamlit 內部法規解析助手。"""

from __future__ import annotations

import logging
import random
from typing import Any

import streamlit as st
from pydantic import ValidationError

from config import Settings, get_settings
from models.legal_qa_response import LegalQaResult
from models.qa_test_question import QaTestQuestion
from services.log_service import LogService
from services.qa_service import create_qa_service
from services.question_bank_service import QuestionBankError, load_question_bank


NEGATIVE_FEEDBACK_REASONS = (
    "找錯條文",
    "解釋錯誤",
    "回答太模糊",
    "沒有回答問題",
    "其他",
)


def _configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "question_input": "",
        "qa_result": None,
        "feedback_submitted": False,
        "show_negative_feedback": False,
        "question_bank_run_requested": False,
        "random_question_ids": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _set_test_question(question: str, run_immediately: bool = False) -> None:
    st.session_state["question_input"] = question
    st.session_state["qa_result"] = None
    st.session_state["feedback_submitted"] = False
    st.session_state["show_negative_feedback"] = False
    st.session_state["question_bank_run_requested"] = run_immediately


def _clear_state() -> None:
    st.session_state["question_input"] = ""
    st.session_state["qa_result"] = None
    st.session_state["feedback_submitted"] = False
    st.session_state["show_negative_feedback"] = False
    st.session_state["question_bank_run_requested"] = False


def _refresh_random_questions(question_ids: tuple[str, ...]) -> None:
    count = min(3, len(question_ids))
    if count == 0:
        st.session_state["random_question_ids"] = []
        return

    previous = set(st.session_state.get("random_question_ids", []))
    fresh_pool = [item for item in question_ids if item not in previous]
    pool = fresh_pool if len(fresh_pool) >= count else list(question_ids)
    st.session_state["random_question_ids"] = random.SystemRandom().sample(
        pool,
        count,
    )


def _set_random_test_question(
    question_texts: tuple[str, ...],
    run_immediately: bool,
) -> None:
    if not question_texts:
        return
    question = random.SystemRandom().choice(question_texts)
    _set_test_question(question, run_immediately)


def _render_random_questions(questions: list[QaTestQuestion]) -> None:
    st.subheader("隨機範例問題")
    if not questions:
        st.caption("題庫尚未載入，暫時無法顯示隨機題目。")
        return

    questions_by_id = {
        question.question_id: question for question in questions
    }
    expected_count = min(3, len(questions_by_id))
    random_ids = list(st.session_state.get("random_question_ids", []))
    if (
        len(random_ids) != expected_count
        or any(item not in questions_by_id for item in random_ids)
    ):
        _refresh_random_questions(tuple(questions_by_id))
        random_ids = list(st.session_state["random_question_ids"])

    st.button(
        "換一批隨機題",
        key="refresh_random_questions",
        on_click=_refresh_random_questions,
        args=(tuple(questions_by_id),),
    )
    columns = st.columns(expected_count)
    for index, (column, question_id) in enumerate(
        zip(columns, random_ids, strict=True),
        start=1,
    ):
        question = questions_by_id[question_id]
        column.button(
            f"隨機題 {index}",
            key=f"random_question_{question_id}",
            help=question.question,
            on_click=_set_test_question,
            args=(question.question, False),
            use_container_width=True,
        )
        column.caption(question.question)
    st.button(
        "隨機抽一題並解析",
        key="run_random_question",
        type="primary",
        on_click=_set_random_test_question,
        args=(tuple(item.question for item in questions), True),
        use_container_width=True,
    )


def _render_question_bank(settings: Settings) -> list[QaTestQuestion]:
    with st.expander("測試題庫：內建 100 題或上傳 JSON"):
        source = st.radio(
            "題庫來源",
            ("內建 100 題", "上傳 JSON"),
            horizontal=True,
            key="question_bank_source",
        )
        question_source: object | None = None
        if source == "內建 100 題":
            question_source = settings.qa_test_questions_path
        else:
            uploaded = st.file_uploader(
                "選擇題庫 JSON",
                type=("json",),
                key="question_bank_upload",
                max_upload_size=4,
                help=(
                    "支援 JSON 陣列，或只含 questions 陣列的 JSON 物件；"
                    "上傳後會先驗證欄位、題號及重複問題。"
                ),
            )
            if uploaded is None:
                st.caption("請上傳題庫；檔案只在目前瀏覽器工作階段使用。")
                return []
            question_source = uploaded.getvalue()

        try:
            questions = load_question_bank(
                question_source,
                max_questions=100,
            )
        except QuestionBankError as exc:
            st.error(str(exc))
            return []

        st.success(f"已載入 {len(questions)} 題。")
        questions_by_id = {
            question.question_id: question for question in questions
        }
        question_ids = list(questions_by_id)
        selection_key = "question_bank_selected_id"
        if st.session_state.get(selection_key) not in questions_by_id:
            st.session_state[selection_key] = question_ids[0]

        selected_id = st.selectbox(
            "選擇測試問題",
            question_ids,
            key=selection_key,
            format_func=lambda item: (
                f"{item}｜{questions_by_id[item].question}"
            ),
        )
        selected: QaTestQuestion = questions_by_id[selected_id]
        st.caption(
            f"{selected.document_name} {selected.article_no}　｜　"
            "預期 Provision ID："
            + "、".join(
                str(item) for item in selected.expected_provision_ids
            )
        )
        if st.checkbox(
            "顯示預期答案與關鍵字",
            key="question_bank_show_expected",
        ):
            st.text(selected.expected_answer)
            st.caption(
                "預期關鍵字："
                + "、".join(selected.expected_keywords)
            )

        fill_column, run_column = st.columns(2)
        fill_column.button(
            "帶入問題",
            key="question_bank_fill",
            on_click=_set_test_question,
            args=(selected.question, False),
            use_container_width=True,
        )
        run_column.button(
            "帶入並解析",
            key="question_bank_run",
            type="primary",
            on_click=_set_test_question,
            args=(selected.question, True),
            use_container_width=True,
        )

        if source == "內建 100 題":
            try:
                bank_bytes = settings.qa_test_questions_path.read_bytes()
            except OSError:
                st.caption("內建題庫目前無法下載，但仍可在本頁選題。")
            else:
                st.download_button(
                    "下載內建題庫 JSON",
                    data=bank_bytes,
                    file_name="qa_test_questions.json",
                    mime="application/json",
                    use_container_width=True,
                )
        return questions


def _render_items(title: str, items: list[str], empty_text: str) -> None:
    st.subheader(title)
    if items:
        for item in items:
            # 模型文字一律以純文字元件顯示，不交給 Markdown/HTML renderer。
            st.text(f"• {item}")
    else:
        st.caption(empty_text)


def _render_answer(result: LegalQaResult) -> None:
    if result.error:
        st.error(result.error)
        return
    if result.response is None:
        st.error("系統沒有產生可顯示且已驗證的回答。")
        return

    response = result.response
    st.subheader("初步結論")
    if response.can_answer:
        st.text(response.summary)
    else:
        st.warning("依目前檢索與引用驗證結果，無法形成受支持的回答。")
        st.text(response.summary)

    _render_items("適用條件", response.conditions, "參考條文未列出特定條件。")
    _render_items("可能例外", response.exceptions, "目前未辨識出可能例外。")
    _render_items(
        "尚需確認事項",
        response.missing_information,
        "目前沒有額外待確認事項。",
    )

    st.subheader("引用條文")
    # 使用本次檢索快照，避免回答完成後資料庫更新造成引用版本漂移。
    retrieved_provisions = {
        item.provision_id: item for item in result.retrieval_results
    }

    if not response.citations:
        st.caption("這次回答沒有通過驗證的引用條文。")
    for citation in response.citations:
        provision = retrieved_provisions.get(citation.provision_id)
        label = f"{citation.document_name} {citation.article_no}"
        with st.expander(label):
            if provision is None:
                st.warning("本次檢索快照中找不到這筆 Provision ID。")
            else:
                if provision.title:
                    st.caption(provision.title)
                # 完整法條只取自 PostgreSQL 檢索結果，不使用模型生成內容。
                st.write(provision.content)

    st.info(response.notice)
    st.caption(f"查詢編號：{result.query_id}　處理耗時：{result.duration_ms} ms")


def _render_retrieval_debug(result: LegalQaResult) -> None:
    with st.expander("查看檢索結果"):
        if not result.retrieval_results:
            st.caption("本次沒有通過相關度門檻的條文。")
            return
        rows = [
            {
                "Provision ID": item.provision_id,
                "法規名稱": item.document_name,
                "條號": item.article_no,
                "向量分數": round(item.vector_score, 4),
                "關鍵字分數": round(item.keyword_score, 4),
                "最終分數": round(item.final_score, 4),
            }
            for item in result.retrieval_results
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _save_feedback(
    settings: Settings,
    result: LegalQaResult,
    feedback: str,
    reason: str | None = None,
) -> bool:
    service = LogService(settings.qa_logs_path)
    return service.update_feedback(result.query_id, feedback, reason)


def _render_feedback(result: LegalQaResult, settings: Settings) -> None:
    st.subheader("這次回答有幫助嗎？")
    if st.session_state["feedback_submitted"]:
        st.success("謝謝，你的回饋已記錄。")
        return

    helpful_column, unhelpful_column = st.columns(2)
    if helpful_column.button(
        "有幫助",
        key=f"helpful_{result.query_id}",
        use_container_width=True,
    ):
        if _save_feedback(settings, result, "helpful"):
            st.session_state["feedback_submitted"] = True
            st.rerun()
        else:
            st.error("回饋暫時無法寫入，主要回答不受影響。")

    if unhelpful_column.button(
        "沒有幫助",
        key=f"unhelpful_{result.query_id}",
        use_container_width=True,
    ):
        st.session_state["show_negative_feedback"] = True

    if st.session_state["show_negative_feedback"]:
        reason = st.selectbox(
            "沒有幫助的原因（選填）",
            NEGATIVE_FEEDBACK_REASONS,
            key=f"feedback_reason_{result.query_id}",
        )
        if st.button("送出回饋", key=f"submit_feedback_{result.query_id}"):
            if _save_feedback(settings, result, "not_helpful", reason):
                st.session_state["feedback_submitted"] = True
                st.rerun()
            else:
                st.error("回饋暫時無法寫入，主要回答不受影響。")


def main() -> None:
    st.set_page_config(
        page_title="內部法規解析助手",
        page_icon="⚖️",
        layout="centered",
    )
    try:
        settings = get_settings()
    except (ValidationError, ValueError):
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).exception("載入應用程式設定失敗")
        st.error("設定檔內容無效，請檢查 `.env` 的數值與必要欄位。")
        st.stop()
    _configure_logging(settings)
    _initialize_state()

    st.title("內部法規解析助手")
    st.caption(
        "資料來源：PostgreSQL＋pgvector　｜　"
        f"回答模型：{settings.ollama_chat_model}　｜　"
        f"Embedding：{settings.ollama_embedding_model}"
    )
    st.warning("本工具僅供內部概念驗證與初步解析，不構成正式法律意見。")

    questions = _render_question_bank(settings)
    _render_random_questions(questions)

    st.text_area(
        "請輸入法規問題",
        key="question_input",
        height=130,
        placeholder="例如：申請文件不齊全時，應在幾日內補件？",
    )
    parse_column, clear_column = st.columns([3, 1])
    parse_clicked = parse_column.button(
        "開始解析",
        type="primary",
        use_container_width=True,
    )
    clear_column.button(
        "清除",
        on_click=_clear_state,
        use_container_width=True,
    )

    run_requested = bool(
        st.session_state.get("question_bank_run_requested", False)
    )
    if parse_clicked or run_requested:
        st.session_state["question_bank_run_requested"] = False
        question = str(st.session_state["question_input"])
        if not question.strip():
            st.warning("請先輸入要解析的問題。")
        else:
            with st.spinner("正在檢索法條並進行初步解析…"):
                service = create_qa_service(settings)
                st.session_state["qa_result"] = service.ask(question)
            st.session_state["feedback_submitted"] = False
            st.session_state["show_negative_feedback"] = False

    result = st.session_state.get("qa_result")
    if isinstance(result, LegalQaResult):
        st.divider()
        _render_answer(result)
        _render_retrieval_debug(result)
        _render_feedback(result, settings)


if __name__ == "__main__":
    main()
