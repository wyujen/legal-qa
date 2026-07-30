"""供檢索與問句處理使用的輕量文字正規化工具。"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType


DEFAULT_SYNONYMS: Mapping[str, str] = MappingProxyType(
    {
        "補資料": "補件",
        "逾期": "超過期限",
        "老師": "教師",
    }
)

# NFKC 已能處理大部分全形 ASCII 標點；下列字元不會由 NFKC 全部統一，
# 因此明確轉成便於搜尋及比對的半形形式。
_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "。": ".",
        "、": ",",
        "；": ";",
        "：": ":",
        "？": "?",
        "！": "!",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "—": "-",
        "–": "-",
    }
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_characters(text: str) -> str:
    """統一字寬、基本標點、英文大小寫及空白。"""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_PUNCTUATION_TRANSLATION)
    normalized = normalized.lower()
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def _replace_synonyms(text: str, synonyms: Mapping[str, str]) -> str:
    """以單次比對取代同義詞，避免取代結果被再次連鎖取代。"""

    normalized_synonyms: dict[str, str] = {}
    for source, target in synonyms.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError("同義詞的來源與目標都必須是字串。")

        normalized_source = _normalize_characters(source)
        if not normalized_source:
            continue
        normalized_synonyms[normalized_source] = _normalize_characters(target)

    if not normalized_synonyms:
        return text

    # 較長的詞優先，可避免「補資料」與「資料」等重疊詞造成短詞先命中。
    pattern = re.compile(
        "|".join(
            re.escape(source)
            for source in sorted(normalized_synonyms, key=len, reverse=True)
        )
    )
    return pattern.sub(lambda match: normalized_synonyms[match.group(0)], text)


def normalize_text(
    text: str,
    synonyms: Mapping[str, str] | None = None,
) -> str:
    """正規化文字，且不改寫條號中的數字或數字順序。

    ``synonyms=None`` 會使用內建的少量同義詞；傳入空字典則可關閉同義詞
    取代，適合需要忠實保留法條用語的情境。
    """

    if not isinstance(text, str):
        raise TypeError("待正規化內容必須是字串。")

    normalized = _normalize_characters(text)
    selected_synonyms = DEFAULT_SYNONYMS if synonyms is None else synonyms
    return _replace_synonyms(normalized, selected_synonyms)


# 保留簡潔的函式名稱，讓呼叫端可使用 ``normalize(question)``。
normalize = normalize_text


class TextNormalizer:
    """可注入自訂同義詞的無狀態正規化器。"""

    def __init__(self, synonyms: Mapping[str, str] | None = None) -> None:
        self._synonyms = DEFAULT_SYNONYMS if synonyms is None else dict(synonyms)

    def normalize(self, text: str) -> str:
        """回傳正規化後文字。"""

        return normalize_text(text, self._synonyms)
