"""將 UTF-8 純文字法規解析為本專案使用的 JSON 格式。"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# 直接執行 ``python scripts/import_law.py`` 時，Python 只會將 scripts/
# 放入模組搜尋路徑；在匯入專案模組前補入 repository root。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings  # noqa: E402
from models.legal_provision import LegalProvision  # noqa: E402
from services.document_service import save_provisions  # noqa: E402


LOGGER = logging.getLogger(__name__)

_NUMBER_CHARACTER = "0-9０-９〇○零一二三四五六七八九十百千萬兩"
_SPACED_NUMBER = (
    rf"[{_NUMBER_CHARACTER}]"
    rf"(?:[ \t\u3000]*[{_NUMBER_CHARACTER}])*"
)
_ARTICLE_NUMBER = (
    rf"{_SPACED_NUMBER}"
    rf"(?:(?:[ \t\u3000]*[-－–][ \t\u3000]*"
    rf"|[ \t\u3000]*之[ \t\u3000]*){_SPACED_NUMBER})*"
)

ARTICLE_PATTERN = re.compile(
    rf"^(?P<article_no>"
    rf"第[ \t\u3000]*{_ARTICLE_NUMBER}[ \t\u3000]*條"
    rf"(?:[ \t\u3000]*之[ \t\u3000]*{_SPACED_NUMBER})?"
    rf")(?:[ \t\u3000]+(?P<content>\S.*))?$"
)
CHAPTER_PATTERN = re.compile(
    rf"^(?P<header>第[ \t\u3000]*{_SPACED_NUMBER}"
    rf"[ \t\u3000]*章(?:[ \t\u3000]*\S.*)?)$"
)
SECTION_PATTERN = re.compile(
    rf"^(?P<header>第[ \t\u3000]*{_SPACED_NUMBER}"
    rf"[ \t\u3000]*節(?:[ \t\u3000]*\S.*)?)$"
)


class LawImportError(ValueError):
    """純文字法規無法讀取或未能產生有效條文。"""


class LawImportWarning(UserWarning):
    """匯入時有原始內容無法安全歸入條文。"""


@dataclass(slots=True)
class _ArticleDraft:
    article_no: str
    chapter_name: str
    section_name: str
    header_line_number: int
    content_lines: list[str] = field(default_factory=list)


def _emit_warning(message: str) -> None:
    warnings.warn(message, LawImportWarning, stacklevel=3)


def _trim_outer_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _canonical_article_no(article_no: str) -> str:
    normalized = unicodedata.normalize("NFKC", article_no)
    return re.sub(r"\s+", "", normalized)


def _unparsed_warning(unparsed_lines: list[tuple[int, str]]) -> None:
    if not unparsed_lines:
        return

    start_line = unparsed_lines[0][0]
    end_line = unparsed_lines[-1][0]
    location = (
        f"第 {start_line} 行"
        if start_line == end_line
        else f"第 {start_line} 至 {end_line} 行"
    )
    preview = " ".join(text.strip() for _, text in unparsed_lines)
    if len(preview) > 120:
        preview = f"{preview[:117]}..."
    _emit_warning(f"{location}無法解析，未匯入內容：{preview}")
    unparsed_lines.clear()


def parse_law_text(
    text: str,
    document_name: str,
    *,
    source_url: str = "",
    start_provision_id: int = 1,
) -> list[LegalProvision]:
    """依章、節及「第 X 條」標題解析純文字法規。

    條號會保留原始字寬與內部空白；法條正文也保留原始換行。無法安全
    歸入任何條文的非空內容會發出 :class:`LawImportWarning`。
    """

    if not isinstance(text, str):
        raise TypeError("法規原文必須是字串。")
    if not document_name.strip():
        raise LawImportError("法規名稱不可為空。")
    if start_provision_id < 1:
        raise LawImportError("起始 provision_id 必須大於 0。")
    if not text.strip():
        raise LawImportError("輸入法規文字為空。")

    provisions: list[LegalProvision] = []
    chapter_name = ""
    section_name = ""
    current_article: _ArticleDraft | None = None
    unparsed_lines: list[tuple[int, str]] = []
    seen_article_numbers: dict[str, int] = {}

    def finalize_article() -> None:
        nonlocal current_article
        if current_article is None:
            return

        content_lines = _trim_outer_blank_lines(current_article.content_lines)
        content = "\n".join(line.rstrip() for line in content_lines)
        if not content.strip():
            _emit_warning(
                f"第 {current_article.header_line_number} 行的"
                f"「{current_article.article_no}」沒有條文內容，未匯入。"
            )
            current_article = None
            return

        canonical_number = _canonical_article_no(current_article.article_no)
        if canonical_number in seen_article_numbers:
            first_line = seen_article_numbers[canonical_number]
            _emit_warning(
                f"第 {current_article.header_line_number} 行的條號"
                f"「{current_article.article_no}」重複（首次出現於第 "
                f"{first_line} 行）；兩筆內容均保留並使用不同 provision_id。"
            )
        else:
            seen_article_numbers[canonical_number] = (
                current_article.header_line_number
            )

        provision_id = start_provision_id + len(provisions)
        provisions.append(
            LegalProvision(
                provision_id=provision_id,
                document_name=document_name.strip(),
                chapter_name=current_article.chapter_name,
                section_name=current_article.section_name,
                article_no=current_article.article_no,
                paragraph_no=None,
                subparagraph_no=None,
                title="",
                content=content,
                search_text="",
                sort_order=len(provisions) + 1,
                source_url=source_url,
                is_active=True,
            )
        )
        current_article = None

    for line_number, original_line in enumerate(text.splitlines(), start=1):
        candidate = original_line.strip()

        chapter_match = CHAPTER_PATTERN.fullmatch(candidate)
        if chapter_match is not None:
            _unparsed_warning(unparsed_lines)
            finalize_article()
            chapter_name = chapter_match.group("header")
            section_name = ""
            continue

        section_match = SECTION_PATTERN.fullmatch(candidate)
        if section_match is not None:
            _unparsed_warning(unparsed_lines)
            finalize_article()
            section_name = section_match.group("header")
            continue

        article_match = ARTICLE_PATTERN.fullmatch(candidate)
        if article_match is not None:
            _unparsed_warning(unparsed_lines)
            finalize_article()
            current_article = _ArticleDraft(
                article_no=article_match.group("article_no").rstrip(),
                chapter_name=chapter_name,
                section_name=section_name,
                header_line_number=line_number,
            )
            inline_content = (article_match.group("content") or "").strip()
            if inline_content:
                current_article.content_lines.append(inline_content)
            continue

        if current_article is not None:
            current_article.content_lines.append(original_line.rstrip())
        elif candidate:
            unparsed_lines.append((line_number, original_line))

    _unparsed_warning(unparsed_lines)
    finalize_article()

    if not provisions:
        raise LawImportError("未找到含有效內容的「第 X 條」條文。")
    return provisions


def import_law_file(
    input_path: str | Path,
    output_path: str | Path,
    document_name: str,
    *,
    source_url: str = "",
) -> list[LegalProvision]:
    """讀取 UTF-8 純文字、解析條文並寫入 UTF-8 JSON。"""

    source_path = Path(input_path)
    try:
        text = source_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到法規原始檔：{source_path}") from exc
    except UnicodeDecodeError as exc:
        raise LawImportError(
            f"法規原始檔不是有效的 UTF-8 文字：{source_path}"
        ) from exc
    except OSError as exc:
        raise OSError(f"無法讀取法規原始檔：{source_path}") from exc

    provisions = parse_law_text(
        text,
        document_name,
        source_url=source_url,
    )
    save_provisions(output_path, provisions)
    return provisions


def build_argument_parser() -> argparse.ArgumentParser:
    """建立命令列參數解析器。"""

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="將 UTF-8 純文字法規匯入 legal_provisions.json。",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="UTF-8 純文字法規檔路徑。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.legal_provisions_path,
        help=f"輸出 JSON 路徑（預設：{settings.legal_provisions_path}）。",
    )
    parser.add_argument(
        "--document-name",
        default=settings.law_name,
        help=f"法規名稱（預設：{settings.law_name}）。",
    )
    parser.add_argument(
        "--source-url",
        default="",
        help="選填的原始法規網址。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """執行法規匯入 CLI，成功回傳 0，失敗回傳 1。"""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.captureWarnings(True)
    args = build_argument_parser().parse_args(argv)

    try:
        provisions = import_law_file(
            input_path=args.input,
            output_path=args.output,
            document_name=args.document_name,
            source_url=args.source_url,
        )
    except (FileNotFoundError, LawImportError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "匯入完成：共 %d 筆條文，輸出至 %s",
        len(provisions),
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
