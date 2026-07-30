from __future__ import annotations

import json

from services.log_service import LogService


def test_save_and_update_feedback(tmp_path) -> None:
    path = tmp_path / "qa_logs.jsonl"
    service = LogService(path)

    assert service.save_log(
        {
            "query_id": "query-1",
            "question": "補件期限多久？",
            "feedback": None,
            "feedback_reason": None,
        }
    )
    assert service.update_feedback("query-1", "not_helpful", "找錯條文")

    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["feedback"] == "not_helpful"
    assert record["feedback_reason"] == "找錯條文"


def test_log_write_error_does_not_raise(tmp_path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    service = LogService(directory)

    assert service.save_log({"query_id": "query-1"}) is False
