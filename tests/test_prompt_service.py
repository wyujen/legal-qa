from models.retrieval_result import RetrievalResult
from services.prompt_service import SYSTEM_PROMPT, build_messages, build_prompt


def _result(
    provision_id: int,
    article_no: str,
    content: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        provision_id=provision_id,
        document_name="測試法規",
        article_no=article_no,
        title=f"{article_no}標題",
        content=content,
        vector_score=score,
        keyword_score=score,
        final_score=score,
    )


def test_prompt_contains_question_provision_id_and_full_content() -> None:
    result = _result(3, "第3條", "申請人應於期限內提出完整文件。", 0.9)

    prompt = build_prompt("申請期限為何？", [result])

    assert "申請期限為何？" in prompt
    assert "[ProvisionId=3]" in prompt
    assert "申請人應於期限內提出完整文件。" in prompt
    assert "測試法規 第3條" in prompt


def test_prompt_only_contains_retrieved_provisions_and_orders_by_score() -> None:
    lower = _result(1, "第1條", "低分條文", 0.1)
    higher = _result(2, "第2條", "高分條文", 0.9)

    prompt = build_prompt("問題", [lower, higher])

    assert prompt.index("[ProvisionId=2]") < prompt.index("[ProvisionId=1]")
    assert "未傳入的法條文字" not in prompt


def test_prompt_limits_references_to_six() -> None:
    results = [
        _result(index, f"第{index}條", f"條文{index}", float(index))
        for index in range(1, 8)
    ]

    prompt = build_prompt("問題", results)

    assert prompt.count("\n[ProvisionId=") == 6
    assert "[ProvisionId=1]" not in prompt


def test_long_reference_keeps_query_focused_excerpt() -> None:
    long_content = (
        ("與本題無關的沿革說明。" * 100)
        + "\n日間部四年制一至三年級每學期應修16至25學分，"
        "四年級應修9至25學分。\n"
        + ("其他不相關規定。" * 100)
    )
    result = _result(8, "第三條", long_content, 1.0)

    prompt = build_prompt(
        "日間部四年制各年級每學期應修多少學分？",
        [result],
    )

    assert "一至三年級每學期應修16至25學分" in prompt
    assert prompt.count("與本題無關的沿革說明") < 5
    assert prompt.count("其他不相關規定") < 5


def test_system_prompt_defends_rules_and_does_not_request_reasoning() -> None:
    assert "僅能根據參考條文回答" in SYSTEM_PROMPT
    assert "不得創造" in SYSTEM_PROMPT
    assert "不可信" in SYSTEM_PROMPT
    assert "忽略規則" in SYSTEM_PROMPT
    assert "can_answer 只判斷「核心問題」" in SYSTEM_PROMPT
    assert "can_answer 必須為 true" in SYSTEM_PROMPT
    assert "僅為主題相關但沒有支持回答內容" in SYSTEM_PROMPT
    assert "ProvisionId 只能填在 citations" in SYSTEM_PROMPT
    assert "不同階段" in SYSTEM_PROMPT
    assert "不得" in SYSTEM_PROMPT
    assert "自行挑選其中一種情境" in SYSTEM_PROMPT
    assert "精確數值" in SYSTEM_PROMPT
    assert "不得輸出" in SYSTEM_PROMPT
    assert "思考過程" in SYSTEM_PROMPT


def test_messages_use_system_and_user_roles_and_include_schema() -> None:
    messages = build_messages(
        "是否可以申請？",
        [_result(1, "第1條", "申請應符合規定。", 1.0)],
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert '"can_answer"' in messages[1]["content"]
    assert '"citations"' in messages[1]["content"]
    assert "只輸出 JSON 物件" in messages[1]["content"]


def test_empty_question_is_rejected() -> None:
    try:
        build_prompt("   ", [])
    except ValueError as exc:
        assert "問題不得為空" in str(exc)
    else:
        raise AssertionError("空問題應觸發 ValueError")
