import pytest

from models.legal_qa_response import LEGAL_NOTICE, LegalQaResponse
from models.retrieval_result import RetrievalResult
from services.response_validator import (
    ResponseValidationError,
    parse_response,
    validate_response,
)


def _retrieval(
    provision_id: int = 3,
    document_name: str = "本地法規",
    article_no: str = "第3條",
) -> RetrievalResult:
    return RetrievalResult(
        provision_id=provision_id,
        document_name=document_name,
        article_no=article_no,
        title="申請條件",
        content="申請人應提出文件。",
        vector_score=0.9,
        keyword_score=0.8,
        final_score=0.865,
    )


def _response(**updates: object) -> LegalQaResponse:
    payload = {
        "can_answer": True,
        "summary": "依條文可提出申請。",
        "conditions": ["應提出文件"],
        "exceptions": [],
        "missing_information": [],
        "citations": [
            {
                "provision_id": 3,
                "document_name": "模型捏造名稱",
                "article_no": "第999條",
            }
        ],
        "notice": "僅供內部初步解析。",
    }
    payload.update(updates)
    return LegalQaResponse.model_validate(payload)


def test_valid_citation_is_kept_and_local_fields_overwrite_model() -> None:
    validated = validate_response(_response(), [_retrieval()])

    assert validated.can_answer is True
    assert len(validated.citations) == 1
    assert validated.citations[0].provision_id == 3
    assert validated.citations[0].document_name == "本地法規"
    assert validated.citations[0].article_no == "第3條"


def test_invalid_provision_id_is_removed_and_answer_becomes_false() -> None:
    response = _response(
        citations=[
            {
                "provision_id": 999,
                "document_name": "不存在",
                "article_no": "第999條",
            }
        ]
    )

    validated = validate_response(response, [_retrieval()])

    assert validated.citations == []
    assert validated.can_answer is False
    assert "沒有可驗證的引用條文" in validated.summary
    assert validated.conditions == []
    assert validated.missing_information


def test_mixed_citations_keep_only_allowlisted_ids() -> None:
    response = _response(
        citations=[
            {
                "provision_id": 999,
                "document_name": "不存在",
                "article_no": "第999條",
            },
            {
                "provision_id": 3,
                "document_name": "錯誤名稱",
                "article_no": "錯誤條號",
            },
        ]
    )

    validated = validate_response(response, [_retrieval()])

    assert [item.provision_id for item in validated.citations] == [3]
    assert validated.can_answer is True


def test_empty_or_html_only_summary_is_invalid() -> None:
    with pytest.raises(ResponseValidationError, match="初步結論不得為空"):
        validate_response(_response(summary="  "), [_retrieval()])

    with pytest.raises(ResponseValidationError, match="初步結論不得為空"):
        validate_response(_response(summary="<script>bad()</script>"), [_retrieval()])


def test_html_is_removed_and_lists_are_limited() -> None:
    response = _response(
        summary="<b>初步</b><script>alert(1)</script>結論",
        conditions=[f"<i>條件{index}</i>" for index in range(8)],
        exceptions=["<img src=x onerror=bad>例外"],
        missing_information=["<strong>補充資料</strong>"],
        notice="<p>內部使用</p>",
    )

    validated = validate_response(
        response,
        [_retrieval()],
        max_list_items=3,
    )

    assert validated.summary == "初步結論"
    assert validated.conditions == ["條件0", "條件1", "條件2"]
    assert validated.exceptions == ["例外"]
    assert validated.missing_information == ["補充資料"]
    assert validated.notice == LEGAL_NOTICE
    assert all("<" not in value for value in validated.conditions)


def test_encoded_html_cannot_reappear_after_sanitizing() -> None:
    validated = validate_response(
        _response(
            summary="&lt;script&gt;alert(1)&lt;/script&gt;安全結論",
            conditions=["&amp;lt;b&amp;gt;條件&amp;lt;/b&amp;gt;"],
        ),
        [_retrieval()],
    )

    assert "<" not in validated.summary
    assert "alert(1)" not in validated.summary
    assert validated.summary == "安全結論"
    assert validated.conditions == ["條件"]


def test_numeric_comparison_symbols_keep_their_meaning() -> None:
    validated = validate_response(
        _response(
            summary="金額 < 1000 元且 > 0 元時適用。",
            conditions=["年齡 < 18 歲或 > 65 歲"],
        ),
        [_retrieval()],
    )

    assert validated.summary == "金額 ＜ 1000 元且 ＞ 0 元時適用。"
    assert validated.conditions == ["年齡 ＜ 18 歲或 ＞ 65 歲"]


def test_internal_provision_markers_are_removed_from_display_text() -> None:
    validated = validate_response(
        _response(
            summary=(
                "根據參考條文[ProvisionId=128]、[ProvisionId=129]"
                "和[ProvisionId=130]，延修生不作延修學期成績排名。"
            ),
            conditions=[
                "依[ProvisionId = 128]規定辦理",
                'provision_id":"128"',
            ],
        ),
        [_retrieval()],
    )

    assert validated.summary == (
        "根據參考條文，延修生不作延修學期成績排名。"
    )
    assert validated.conditions == ["依規定辦理"]
    assert "ProvisionId" not in validated.summary


def test_parse_response_accepts_json_code_fence_but_returns_model() -> None:
    raw = """```json
    {
      "can_answer": false,
      "summary": "資料不足。",
      "conditions": [],
      "exceptions": [],
      "missing_information": ["需要申請日期"],
      "citations": [],
      "notice": "僅供內部使用。"
    }
    ```"""

    parsed = parse_response(raw)

    assert isinstance(parsed, LegalQaResponse)
    assert parsed.can_answer is False
    assert parsed.summary == "資料不足。"


def test_malformed_json_is_never_returned_as_answer() -> None:
    with pytest.raises(ResponseValidationError, match="JSON 格式無法解析"):
        parse_response("這不是 JSON")
