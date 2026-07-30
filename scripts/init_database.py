"""Initialize the local PostgreSQL + pgvector schema."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import Settings  # noqa: E402
from services.database_service import (  # noqa: E402
    DatabaseServiceError,
    PostgresDatabase,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        database = PostgresDatabase(Settings())
        database.initialize_schema()
        database.ping()
    except DatabaseServiceError as exc:
        print(f"資料庫初始化失敗：{exc}", file=sys.stderr)
        return 1

    print("PostgreSQL＋pgvector schema 已初始化完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
