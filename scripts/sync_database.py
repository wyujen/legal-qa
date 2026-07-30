"""Synchronize validated collector JSON into PostgreSQL + pgvector."""

from __future__ import annotations

import argparse
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
from services.database_sync_service import (  # noqa: E402
    DatabaseSyncError,
    DatabaseSyncService,
)
from services.document_service import (  # noqa: E402
    DocumentFormatError,
    load_provisions,
)
from services.embedding_service import EmbeddingServiceError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "將 legal_provisions.json 增量同步至 PostgreSQL，"
            "只為新增或內容變更的條文產生 Embedding。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="輸入 JSON；預設使用設定中的 data/legal_provisions.json。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding 批次大小（預設 32）。",
    )
    parser.add_argument(
        "--full-snapshot",
        action="store_true",
        help=(
            "確認輸入是完整現行法規快照，並停用資料庫中未出現的舊條文。"
        ),
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="略過冪等 schema 初始化。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings()
    input_path = args.input or settings.legal_provisions_path
    try:
        provisions = load_provisions(input_path)
        database = PostgresDatabase(settings)
        if not args.skip_init:
            database.initialize_schema()
        service = DatabaseSyncService(settings, database=database)
        summary = service.sync(
            provisions,
            source_path=input_path,
            batch_size=args.batch_size,
            full_snapshot=args.full_snapshot,
        )
    except (
        DatabaseServiceError,
        DatabaseSyncError,
        DocumentFormatError,
        EmbeddingServiceError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"資料庫同步失敗：{exc}", file=sys.stderr)
        return 1

    print(
        "同步完成："
        f"{summary.document_count} 部法規、"
        f"{summary.provision_count} 筆條文；"
        f"新建／更新 {summary.embedded_count} 筆向量，"
        f"沿用 {summary.reused_embedding_count} 筆，"
        f"停用 {summary.deactivated_count} 筆。"
    )
    print(f"collection run：{summary.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
