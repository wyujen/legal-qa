"""文字正規化行為測試。"""

from __future__ import annotations

import pytest

from services.text_normalizer import TextNormalizer, normalize_text


def test_normalize_collapses_and_trims_whitespace() -> None:
    assert normalize_text(" \t第一段 \n  第二段　 ", synonyms={}) == "第一段 第二段"


def test_normalize_unifies_width_case_and_basic_punctuation() -> None:
    source = "ＡＢＣ１２３，測試！「ＯＫ」。"

    assert normalize_text(source, synonyms={}) == 'abc123,測試!"ok".'


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("請問要怎麼補資料", "請問要怎麼補件"),
        ("已經逾期", "已經超過期限"),
        ("老師的資格", "教師的資格"),
    ],
)
def test_normalize_replaces_default_synonyms(source: str, expected: str) -> None:
    assert normalize_text(source) == expected


def test_custom_synonyms_are_simultaneous_and_longest_term_wins() -> None:
    normalizer = TextNormalizer({"補資料": "補件", "資料": "文件", "補件": "申請"})

    # 單次取代不會把「補資料」先換成「補件」後又連鎖換成「申請」。
    assert normalizer.normalize("補資料及資料") == "補件及文件"


def test_normalize_preserves_article_numbers_and_digit_order() -> None:
    assert normalize_text("  第１２３條 及 第10條之2  ", synonyms={}) == (
        "第123條 及 第10條之2"
    )


def test_empty_text_is_supported() -> None:
    assert normalize_text(" \n\t ") == ""


def test_non_string_input_has_clear_error() -> None:
    with pytest.raises(TypeError, match="必須是字串"):
        normalize_text(123)  # type: ignore[arg-type]
