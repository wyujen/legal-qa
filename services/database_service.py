"""PostgreSQL connection and schema lifecycle helpers.

The application opens short-lived connections on demand.  Importing this
module never connects to PostgreSQL, so the Streamlit home page can still load
and show a friendly error when the local database is stopped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class DatabaseServiceError(RuntimeError):
    """Base class for user-safe PostgreSQL errors."""


class DatabaseDependencyError(DatabaseServiceError):
    """Required Python database packages are missing."""


class DatabaseConnectionError(DatabaseServiceError):
    """PostgreSQL could not be reached."""


class DatabaseNotInitializedError(DatabaseServiceError):
    """The database schema or pgvector extension is missing."""


class DatabaseQueryError(DatabaseServiceError):
    """A database operation failed without exposing its DSN."""


def _load_postgres_dependencies() -> tuple[Any, Any]:
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise DatabaseDependencyError(
            "尚未安裝 PostgreSQL Python 套件，請執行 "
            "pip install -r requirements.txt。"
        ) from exc
    return psycopg, register_vector


def _translate_database_error(exc: Exception) -> DatabaseServiceError:
    """Translate psycopg errors without leaking credentials or SQL."""

    sqlstate = str(getattr(exc, "sqlstate", "") or "")
    if sqlstate in {"42P01", "42704", "42883"}:
        return DatabaseNotInitializedError(
            "PostgreSQL 法規資料庫尚未初始化，請執行 "
            "python scripts/init_database.py。"
        )

    class_name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    connection_markers = (
        "operationalerror",
        "connection",
        "could not connect",
        "connection refused",
        "timeout",
        "server closed",
    )
    if any(
        marker in class_name or marker in message
        for marker in connection_markers
    ):
        return DatabaseConnectionError(
            "無法連線到 PostgreSQL 法規資料庫，請確認資料庫服務已啟動。"
        )
    return DatabaseQueryError(
        "PostgreSQL 法規資料庫操作失敗，請查看應用程式執行紀錄。"
    )


class PostgresDatabase:
    """Create short-lived psycopg connections and apply the local schema."""

    def __init__(
        self,
        settings: Any,
        *,
        connection_factory: Any | None = None,
    ) -> None:
        self.database_url = str(settings.database_url)
        self.connect_timeout_seconds = int(
            settings.database_connect_timeout_seconds
        )
        self.schema_path = Path(settings.database_schema_path)
        self._connection_factory = connection_factory

    def _open_connection(self, *, autocommit: bool) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory(autocommit=autocommit)

        psycopg, _ = _load_postgres_dependencies()
        try:
            return psycopg.connect(
                self.database_url,
                connect_timeout=self.connect_timeout_seconds,
                autocommit=autocommit,
            )
        except DatabaseServiceError:
            raise
        except Exception as exc:
            logger.exception("連線 PostgreSQL 失敗")
            raise _translate_database_error(exc) from exc

    @contextmanager
    def connect(
        self,
        *,
        autocommit: bool = False,
        register_vectors: bool = True,
    ) -> Iterator[Any]:
        connection: Any | None = None
        try:
            connection = self._open_connection(autocommit=autocommit)
            if register_vectors and self._connection_factory is None:
                _, register_vector = _load_postgres_dependencies()
                register_vector(connection)
            yield connection
        except DatabaseServiceError:
            raise
        except Exception as exc:
            logger.exception("PostgreSQL 操作失敗")
            raise _translate_database_error(exc) from exc
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.exception("關閉 PostgreSQL connection 失敗")

    def initialize_schema(self) -> None:
        """Create pgvector and all idempotent application tables/indexes."""

        try:
            schema_sql = self.schema_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            logger.exception("讀取資料庫 schema 失敗：%s", self.schema_path)
            raise DatabaseQueryError(
                "無法讀取 PostgreSQL schema 檔案。"
            ) from exc

        statements = [
            statement.strip()
            for statement in schema_sql.split(";")
            if statement.strip()
        ]
        try:
            with self.connect(
                autocommit=True,
                register_vectors=False,
            ) as connection:
                with connection.cursor() as cursor:
                    for statement in statements:
                        cursor.execute(statement)
        except DatabaseServiceError:
            raise
        except Exception as exc:
            logger.exception("初始化 PostgreSQL schema 失敗")
            raise _translate_database_error(exc) from exc

    def ping(self) -> None:
        with self.connect(autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()


__all__ = [
    "DatabaseConnectionError",
    "DatabaseDependencyError",
    "DatabaseNotInitializedError",
    "DatabaseQueryError",
    "DatabaseServiceError",
    "PostgresDatabase",
]
