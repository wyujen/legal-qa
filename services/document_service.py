"""UTF-8 JSON 法規文件的讀取、驗證與儲存。"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from models.legal_provision import LegalProvision


DEFAULT_PROVISIONS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "legal_provisions.json"
)


class DocumentFormatError(ValueError):
    """法規文件不是合法 JSON 陣列或不符合資料模型。"""


def _validate_provision(item: object, index: int) -> LegalProvision:
    if isinstance(item, LegalProvision):
        return item
    if not isinstance(item, Mapping):
        raise DocumentFormatError(f"第 {index} 筆法規資料必須是 JSON 物件。")

    try:
        return LegalProvision.model_validate(dict(item))
    except ValidationError as exc:
        raise DocumentFormatError(f"第 {index} 筆法規資料欄位格式錯誤。") from exc


def _ensure_unique_ids(provisions: Iterable[LegalProvision]) -> None:
    seen_ids: set[int] = set()
    duplicate_ids: set[int] = set()
    for provision in provisions:
        if provision.provision_id in seen_ids:
            duplicate_ids.add(provision.provision_id)
        seen_ids.add(provision.provision_id)

    if duplicate_ids:
        formatted_ids = "、".join(str(item) for item in sorted(duplicate_ids))
        raise DocumentFormatError(f"provision_id 不可重複：{formatted_ids}。")


def load_provisions(path: str | Path = DEFAULT_PROVISIONS_PATH) -> list[LegalProvision]:
    """從 UTF-8 JSON 陣列載入並驗證法規條文。"""

    document_path = Path(path)
    try:
        raw_text = document_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到法規資料檔：{document_path}") from exc
    except UnicodeDecodeError as exc:
        raise DocumentFormatError(
            f"法規資料檔不是有效的 UTF-8 文字：{document_path}"
        ) from exc
    except OSError as exc:
        raise OSError(f"無法讀取法規資料檔：{document_path}") from exc

    try:
        payload: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DocumentFormatError(
            f"法規 JSON 格式錯誤（第 {exc.lineno} 行、第 {exc.colno} 欄）："
            f"{document_path}"
        ) from exc

    if not isinstance(payload, list):
        raise DocumentFormatError("法規 JSON 最外層必須是陣列。")

    provisions = [
        _validate_provision(item, index)
        for index, item in enumerate(payload, start=1)
    ]
    _ensure_unique_ids(provisions)
    return provisions


def _serialize_provisions(
    provisions: Iterable[LegalProvision | Mapping[str, object]],
) -> list[dict[str, Any]]:
    validated = [
        _validate_provision(item, index)
        for index, item in enumerate(provisions, start=1)
    ]
    _ensure_unique_ids(validated)
    return [provision.model_dump(mode="json") for provision in validated]


def save_provisions(
    path: str | Path,
    provisions: Iterable[LegalProvision | Mapping[str, object]],
) -> None:
    """驗證後以無 BOM 的 UTF-8 JSON 原子寫入法規條文。"""

    document_path = Path(path)
    payload = _serialize_provisions(provisions)

    temporary_path: Path | None = None
    try:
        document_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=document_path.parent,
            prefix=f".{document_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                indent=2,
            )
            temporary_file.write("\n")
        temporary_path.replace(document_path)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise OSError(f"無法寫入法規資料檔：{document_path}") from exc


def load_legal_provisions(
    path: str | Path = DEFAULT_PROVISIONS_PATH,
) -> list[LegalProvision]:
    """``load_provisions`` 的語意化別名。"""

    return load_provisions(path)


def save_legal_provisions(
    provisions: Iterable[LegalProvision | Mapping[str, object]],
    path: str | Path = DEFAULT_PROVISIONS_PATH,
) -> None:
    """以「資料在前、路徑在後」的便利介面儲存條文。"""

    save_provisions(path, provisions)


class DocumentService:
    """綁定單一 JSON 路徑的法規文件服務。"""

    def __init__(self, path: str | Path = DEFAULT_PROVISIONS_PATH) -> None:
        self.path = Path(path)

    def load_provisions(self) -> list[LegalProvision]:
        """載入目前路徑中的所有法規條文。"""

        return load_provisions(self.path)

    def save_provisions(
        self,
        provisions: Iterable[LegalProvision | Mapping[str, object]],
    ) -> None:
        """將法規條文儲存至目前路徑。"""

        save_provisions(self.path, provisions)

    def get_by_id(self, provision_id: int) -> LegalProvision | None:
        """依唯一 ID 取得條文；查無資料時回傳 ``None``。"""

        return next(
            (
                provision
                for provision in self.load_provisions()
                if provision.provision_id == provision_id
            ),
            None,
        )
