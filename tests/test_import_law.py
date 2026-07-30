"""純文字法規匯入測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.import_law import (
    LawImportError,
    LawImportWarning,
    import_law_file,
    parse_law_text,
)
from services.document_service import load_provisions


def test_parse_spaced_chinese_and_inline_arabic_articles() -> None:
    source = """第一章 總則

第 一 條
第一條第一段。
第一條第二段。

第2條 第二條同列內容。
"""

    provisions = parse_law_text(source, "測試法規")

    assert [item.provision_id for item in provisions] == [1, 2]
    assert [item.article_no for item in provisions] == ["第 一 條", "第2條"]
    assert provisions[0].chapter_name == "第一章 總則"
    assert provisions[0].content == "第一條第一段。\n第一條第二段。"
    assert provisions[1].content == "第二條同列內容。"
    assert [item.sort_order for item in provisions] == [1, 2]


def test_parse_tracks_chapter_and_section_context() -> None:
    source = """第一章總則
第一節 通則
第1條 第一章內容
第二章 附則
第 二 條 第二章內容
"""

    provisions = parse_law_text(source, "測試法規")

    assert provisions[0].chapter_name == "第一章總則"
    assert provisions[0].section_name == "第一節 通則"
    assert provisions[1].chapter_name == "第二章 附則"
    assert provisions[1].section_name == ""


def test_unparsed_preamble_is_reported_instead_of_silently_discarded() -> None:
    source = """這是無法辨識的前言
另一行前言
第1條 有效條文
"""

    with pytest.warns(LawImportWarning, match="第 1 至 2 行.*未匯入"):
        provisions = parse_law_text(source, "測試法規")

    assert len(provisions) == 1


def test_empty_article_is_warned_and_not_written() -> None:
    source = """第1條
第2條 有內容
"""

    with pytest.warns(LawImportWarning, match="第1條.*沒有條文內容"):
        provisions = parse_law_text(source, "測試法規")

    assert [item.article_no for item in provisions] == ["第2條"]


def test_duplicate_article_number_is_warned_but_both_are_retained() -> None:
    source = """第 一 條 第一份內容
第一條 第二份內容
"""

    with pytest.warns(LawImportWarning, match="條號.*重複"):
        provisions = parse_law_text(source, "測試法規")

    assert len(provisions) == 2
    assert [item.provision_id for item in provisions] == [1, 2]


def test_import_file_writes_readable_utf8_json(tmp_path: Path) -> None:
    source_path = tmp_path / "來源法規.txt"
    output_path = tmp_path / "legal_provisions.json"
    source_path.write_text("第1條 中文條文內容。", encoding="utf-8")

    imported = import_law_file(
        source_path,
        output_path,
        "中文測試法規",
        source_url="https://example.test/law",
    )
    loaded = load_provisions(output_path)
    raw_json = output_path.read_text(encoding="utf-8")

    assert len(imported) == len(loaded) == 1
    assert loaded[0].document_name == "中文測試法規"
    assert loaded[0].source_url == "https://example.test/law"
    assert "中文條文內容" in raw_json
    assert "\\u4e2d" not in raw_json
    assert json.loads(raw_json)[0]["article_no"] == "第1條"


def test_invalid_or_empty_input_has_clear_error() -> None:
    with pytest.raises(LawImportError, match="文字為空"):
        parse_law_text(" \n ", "測試法規")

    with pytest.warns(LawImportWarning, match="無法解析"):
        with pytest.raises(LawImportError, match="未找到"):
            parse_law_text("這不是可辨識的法條", "測試法規")
