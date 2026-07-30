"""以 JSONL 儲存 QA 過程，並提供不影響主流程的回饋更新。"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"無法序列化 {type(value).__name__}")


class LogService:
    """追加 QA 紀錄；任何磁碟錯誤只寫 Python log 並回傳失敗。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save_log(self, record: Mapping[str, Any]) -> bool:
        """以單行 UTF-8 JSON 追加紀錄，不將錯誤傳回 QA 主流程。"""

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(
                dict(record),
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_default,
            )
            with self.path.open("a", encoding="utf-8", newline="\n") as log_file:
                log_file.write(serialized)
                log_file.write("\n")
            return True
        except (OSError, TypeError, ValueError):
            logger.exception("寫入 QA JSONL 紀錄失敗：%s", self.path)
            return False

    # 讓依賴注入測試可以使用較短的命名。
    save = save_log

    def update_feedback(
        self,
        query_id: str,
        feedback: str,
        feedback_reason: str | None = None,
    ) -> bool:
        """以原子取代方式更新指定查詢的回饋欄位。"""

        if not query_id.strip() or not feedback.strip():
            logger.warning("忽略缺少 query_id 或 feedback 的回饋")
            return False

        try:
            if not self.path.exists():
                logger.warning("QA Log 尚不存在，無法儲存回饋：%s", query_id)
                return False

            raw_lines = self.path.read_text(encoding="utf-8").splitlines()
            records: list[dict[str, Any] | str] = []
            matched = False
            for raw_line in raw_lines:
                if not raw_line.strip():
                    continue
                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError:
                    records.append(raw_line)
                    continue

                if isinstance(item, dict) and item.get("query_id") == query_id:
                    item["feedback"] = feedback
                    item["feedback_reason"] = feedback_reason or None
                    matched = True
                records.append(item)

            if not matched:
                logger.warning("QA Log 找不到 query_id，無法儲存回饋：%s", query_id)
                return False

            temporary_path = self.path.with_name(f".{self.path.name}.tmp")
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as temporary_file:
                for item in records:
                    if isinstance(item, str):
                        temporary_file.write(item)
                    else:
                        temporary_file.write(
                            json.dumps(
                                item,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                default=_json_default,
                            )
                        )
                    temporary_file.write("\n")
            temporary_path.replace(self.path)
            return True
        except (OSError, TypeError, ValueError):
            logger.exception("更新 QA 回饋失敗：%s", query_id)
            return False
