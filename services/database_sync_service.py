"""Validated, incremental JSON-to-PostgreSQL synchronization."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from models.legal_provision import LegalProvision
from services.database_service import PostgresDatabase
from services.retrieval_service import compact_keyword_text


logger = logging.getLogger(__name__)


class DatabaseSyncError(RuntimeError):
    """The input snapshot cannot be synchronized safely."""


@dataclass(frozen=True, slots=True)
class SyncSummary:
    run_id: str
    document_count: int
    provision_count: int
    embedded_count: int
    reused_embedding_count: int
    deactivated_count: int
    source_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PreparedProvision:
    provision: LegalProvision
    content_hash: str
    embedding_text: str
    embedding_input_hash: str
    search_compact: str

    @property
    def stable_key(self) -> tuple[str, str, int | None, int | None]:
        item = self.provision
        return (
            item.document_name,
            item.article_no,
            item.paragraph_no,
            item.subparagraph_no,
        )


@dataclass(frozen=True, slots=True)
class _ExistingProvision:
    provision_id: int
    stable_key: tuple[str, str, int | None, int | None]
    embedding_input_hash: str
    has_current_embedding: bool


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(canonical)


def build_embedding_text(provision: LegalProvision) -> str:
    """Build a complete input even if collector search_text is stale.

    Collector-provided search terms are retained as optional retrieval hints,
    while the official title and content are always present in the vector input.
    """

    canonical = " ".join(
        part
        for part in (
            provision.document_name,
            provision.chapter_name,
            provision.section_name,
            provision.article_no,
            provision.title,
            provision.content,
        )
        if part
    )
    canonical = " ".join(canonical.split())
    provided = " ".join(provision.search_text.split())
    if not provided or provided == canonical:
        return canonical
    return f"{canonical} {provided}"


def _prepare_provisions(
    provisions: Sequence[LegalProvision],
) -> list[_PreparedProvision]:
    if not provisions:
        raise DatabaseSyncError("法規快照不可為空。")

    expected_orders = list(range(1, len(provisions) + 1))
    actual_orders = [item.sort_order for item in provisions]
    if actual_orders != expected_orders:
        raise DatabaseSyncError(
            "sort_order 必須依輸入順序從 1 全域連續編號。"
        )

    prepared: list[_PreparedProvision] = []
    seen_keys: dict[tuple[str, str, int | None, int | None], int] = {}
    seen_ids: set[int] = set()
    document_urls: dict[str, set[str]] = defaultdict(set)
    for item in provisions:
        if not item.is_active:
            raise DatabaseSyncError(
                "正式收集 JSON 只能包含 is_active=true 的現行條文。"
            )
        if item.provision_id in seen_ids:
            raise DatabaseSyncError(
                f"provision_id 不可重複：{item.provision_id}。"
            )
        seen_ids.add(item.provision_id)
        key = (
            item.document_name,
            item.article_no,
            item.paragraph_no,
            item.subparagraph_no,
        )
        if key in seen_keys:
            raise DatabaseSyncError(
                "法規快照含重複 stable key："
                f"{item.document_name}／{item.article_no}／"
                f"{item.paragraph_no}／{item.subparagraph_no}。"
            )
        seen_keys[key] = item.provision_id
        if item.source_url:
            document_urls[item.document_name].add(item.source_url)
        dump = item.model_dump(mode="json")
        embedding_text = build_embedding_text(item)
        prepared.append(
            _PreparedProvision(
                provision=item,
                content_hash=_canonical_hash(dump),
                embedding_text=embedding_text,
                embedding_input_hash=_sha256_text(embedding_text),
                search_compact=compact_keyword_text(embedding_text),
            )
        )

    conflicts = [
        name for name, urls in document_urls.items() if len(urls) > 1
    ]
    if conflicts:
        raise DatabaseSyncError(
            "同一法規含多個不同 source_url："
            + "、".join(sorted(conflicts))
            + "。"
        )
    return prepared


def _is_allowed_first_paragraph_migration(
    old_key: tuple[str, str, int | None, int | None],
    new_key: tuple[str, str, int | None, int | None],
) -> bool:
    """Implement the confirmed Q18 rule for a newly split first paragraph."""

    return (
        old_key[0] == new_key[0]
        and old_key[1] == new_key[1]
        and old_key[2] is None
        and new_key[2] == 1
        and old_key[3] == new_key[3]
    )


class DatabaseSyncService:
    """Synchronize a complete or incremental collector snapshot."""

    def __init__(
        self,
        settings: Any,
        *,
        database: PostgresDatabase | None = None,
        embedding_service: Any | None = None,
    ) -> None:
        self.settings = settings
        self.database = database or PostgresDatabase(settings)
        if embedding_service is None:
            from services.embedding_service import EmbeddingService

            embedding_service = EmbeddingService(settings)
        self.embedding_service = embedding_service
        self.embedding_model = str(settings.ollama_embedding_model)
        self.embedding_dimension = int(settings.embedding_dimension)

    def sync(
        self,
        provisions: Sequence[LegalProvision],
        *,
        source_path: str | Path,
        batch_size: int = 32,
        full_snapshot: bool = False,
    ) -> SyncSummary:
        if batch_size < 1:
            raise ValueError("batch_size 必須至少為 1。")

        prepared = _prepare_provisions(provisions)
        source_fingerprint = _canonical_hash(
            [item.provision.model_dump(mode="json") for item in prepared]
        )
        with self._synchronization_lock():
            existing = self._load_existing()
            self._validate_identity(existing, prepared)

            to_embed = [
                item
                for item in prepared
                if not (
                    item.provision.provision_id in existing
                    and existing[
                        item.provision.provision_id
                    ].has_current_embedding
                    and existing[
                        item.provision.provision_id
                    ].embedding_input_hash
                    == item.embedding_input_hash
                )
            ]
            vectors = self._generate_embeddings(
                to_embed,
                batch_size=batch_size,
            )
            run_uuid = uuid4()
            deactivated_count = self._write_snapshot(
                prepared,
                vectors=vectors,
                run_id=run_uuid,
                source_path=str(Path(source_path)),
                source_fingerprint=source_fingerprint,
                full_snapshot=full_snapshot,
            )
        return SyncSummary(
            run_id=str(run_uuid),
            document_count=len(
                {item.provision.document_name for item in prepared}
            ),
            provision_count=len(prepared),
            embedded_count=len(to_embed),
            reused_embedding_count=len(prepared) - len(to_embed),
            deactivated_count=deactivated_count,
            source_fingerprint=source_fingerprint,
        )

    @contextmanager
    def _synchronization_lock(self):
        """Serialize real sync commands without holding a long transaction."""

        connect = getattr(self.database, "connect", None)
        if not callable(connect):
            yield
            return

        with connect(autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_lock(hashtext(%s))",
                    ("legal_qa_database_sync",),
                )
            try:
                yield
            finally:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT pg_advisory_unlock(hashtext(%s))",
                        ("legal_qa_database_sync",),
                    )

    def _load_existing(self) -> dict[int, _ExistingProvision]:
        sql = """
            SELECT
                p.provision_id,
                d.document_name,
                p.article_no,
                p.paragraph_no,
                p.subparagraph_no,
                p.embedding_input_hash,
                (
                    e.provision_id IS NOT NULL
                    AND e.embedding_input_hash = p.embedding_input_hash
                ) AS has_current_embedding
            FROM legal_provisions AS p
            JOIN legal_documents AS d
              ON d.document_id = p.document_id
            LEFT JOIN provision_embeddings AS e
              ON e.provision_id = p.provision_id
             AND e.embedding_model = %(embedding_model)s
        """
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, {"embedding_model": self.embedding_model})
                rows = cursor.fetchall()
        return {
            int(row[0]): _ExistingProvision(
                provision_id=int(row[0]),
                stable_key=(
                    str(row[1]),
                    str(row[2]),
                    row[3],
                    row[4],
                ),
                embedding_input_hash=str(row[5]),
                has_current_embedding=bool(row[6]),
            )
            for row in rows
        }

    @staticmethod
    def _validate_identity(
        existing: Mapping[int, _ExistingProvision],
        incoming: Sequence[_PreparedProvision],
    ) -> None:
        existing_by_key = {
            value.stable_key: value.provision_id
            for value in existing.values()
        }
        for item in incoming:
            provision_id = item.provision.provision_id
            previous = existing.get(provision_id)
            if previous is not None and previous.stable_key != item.stable_key:
                if not _is_allowed_first_paragraph_migration(
                    previous.stable_key,
                    item.stable_key,
                ):
                    raise DatabaseSyncError(
                        f"provision_id {provision_id} 已對應其他 stable key，"
                        "已停止同步。"
                    )
            key_owner = existing_by_key.get(item.stable_key)
            if key_owner is not None and key_owner != provision_id:
                raise DatabaseSyncError(
                    "stable key 已由其他 provision_id 使用："
                    f"{item.stable_key}（既有 {key_owner}，"
                    f"輸入 {provision_id}）。"
                )

    def _generate_embeddings(
        self,
        items: Sequence[_PreparedProvision],
        *,
        batch_size: int,
    ) -> dict[int, np.ndarray]:
        vectors_by_id: dict[int, np.ndarray] = {}
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            texts = [item.embedding_text for item in batch]
            embed_documents = getattr(
                self.embedding_service,
                "embed_documents",
                None,
            )
            if callable(embed_documents):
                raw_vectors = embed_documents(texts)
            else:
                embed_texts = getattr(
                    self.embedding_service,
                    "embed_texts",
                    None,
                )
                if not callable(embed_texts):
                    raise TypeError(
                        "embedding_service 必須提供 embed_documents() "
                        "或 embed_texts()。"
                    )
                raw_vectors = embed_texts(texts)

            matrix = np.asarray(raw_vectors, dtype=np.float32)
            expected_shape = (len(batch), self.embedding_dimension)
            if matrix.shape != expected_shape:
                raise DatabaseSyncError(
                    "Embedding 維度或筆數不正確："
                    f"預期 {expected_shape}，實際 {matrix.shape}。"
                )
            if not np.isfinite(matrix).all():
                raise DatabaseSyncError(
                    "Embedding 含 NaN 或 Infinity，資料庫未更新。"
                )
            norms = np.linalg.norm(matrix, axis=1)
            if np.any(norms <= 0):
                raise DatabaseSyncError(
                    "Embedding 含零向量，無法建立 cosine 向量索引。"
                )
            for item, vector in zip(batch, matrix, strict=True):
                vectors_by_id[item.provision.provision_id] = vector.copy()
        return vectors_by_id

    def _write_snapshot(
        self,
        prepared: Sequence[_PreparedProvision],
        *,
        vectors: Mapping[int, np.ndarray],
        run_id: UUID,
        source_path: str,
        source_fingerprint: str,
        full_snapshot: bool,
    ) -> int:
        grouped: dict[str, list[_PreparedProvision]] = defaultdict(list)
        for item in prepared:
            grouped[item.provision.document_name].append(item)

        deactivated_count = 0
        with self.database.connect() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO collection_runs (
                            run_id,
                            source_fingerprint,
                            source_path,
                            embedding_model,
                            full_snapshot,
                            status,
                            document_count,
                            provision_count,
                            embedded_count,
                            reused_embedding_count
                        ) VALUES (
                            %(run_id)s,
                            %(source_fingerprint)s,
                            %(source_path)s,
                            %(embedding_model)s,
                            %(full_snapshot)s,
                            'running',
                            %(document_count)s,
                            %(provision_count)s,
                            %(embedded_count)s,
                            %(reused_embedding_count)s
                        )
                        """,
                        {
                            "run_id": run_id,
                            "source_fingerprint": source_fingerprint,
                            "source_path": source_path,
                            "embedding_model": self.embedding_model,
                            "full_snapshot": full_snapshot,
                            "document_count": len(grouped),
                            "provision_count": len(prepared),
                            "embedded_count": len(vectors),
                            "reused_embedding_count": (
                                len(prepared) - len(vectors)
                            ),
                        },
                    )

                    document_ids: dict[str, int] = {}
                    for document_name, items in grouped.items():
                        source_url = next(
                            (
                                item.provision.source_url
                                for item in items
                                if item.provision.source_url
                            ),
                            "",
                        )
                        document_hash = _canonical_hash(
                            [
                                item.provision.model_dump(mode="json")
                                for item in items
                            ]
                        )
                        cursor.execute(
                            """
                            INSERT INTO legal_documents (
                                document_name,
                                source_url,
                                content_hash,
                                is_active,
                                last_collection_run_id
                            ) VALUES (
                                %(document_name)s,
                                %(source_url)s,
                                %(content_hash)s,
                                true,
                                %(run_id)s
                            )
                            ON CONFLICT (document_name) DO UPDATE SET
                                source_url = CASE
                                    WHEN EXCLUDED.source_url <> ''
                                    THEN EXCLUDED.source_url
                                    ELSE legal_documents.source_url
                                END,
                                content_hash = EXCLUDED.content_hash,
                                is_active = true,
                                last_collected_at = now(),
                                last_collection_run_id = EXCLUDED.last_collection_run_id
                            RETURNING document_id
                            """,
                            {
                                "document_name": document_name,
                                "source_url": source_url,
                                "content_hash": document_hash,
                                "run_id": run_id,
                            },
                        )
                        document_ids[document_name] = int(
                            cursor.fetchone()[0]
                        )

                    for item in prepared:
                        provision = item.provision
                        cursor.execute(
                            """
                            INSERT INTO legal_provisions (
                                provision_id,
                                document_id,
                                chapter_name,
                                section_name,
                                article_no,
                                paragraph_no,
                                subparagraph_no,
                                title,
                                content,
                                search_text,
                                search_compact,
                                sort_order,
                                source_url,
                                is_active,
                                content_hash,
                                embedding_input_hash,
                                last_collection_run_id
                            ) VALUES (
                                %(provision_id)s,
                                %(document_id)s,
                                %(chapter_name)s,
                                %(section_name)s,
                                %(article_no)s,
                                %(paragraph_no)s,
                                %(subparagraph_no)s,
                                %(title)s,
                                %(content)s,
                                %(search_text)s,
                                %(search_compact)s,
                                %(sort_order)s,
                                %(source_url)s,
                                true,
                                %(content_hash)s,
                                %(embedding_input_hash)s,
                                %(run_id)s
                            )
                            ON CONFLICT (provision_id) DO UPDATE SET
                                document_id = EXCLUDED.document_id,
                                chapter_name = EXCLUDED.chapter_name,
                                section_name = EXCLUDED.section_name,
                                article_no = EXCLUDED.article_no,
                                paragraph_no = EXCLUDED.paragraph_no,
                                subparagraph_no = EXCLUDED.subparagraph_no,
                                title = EXCLUDED.title,
                                content = EXCLUDED.content,
                                search_text = EXCLUDED.search_text,
                                search_compact = EXCLUDED.search_compact,
                                sort_order = EXCLUDED.sort_order,
                                source_url = EXCLUDED.source_url,
                                is_active = true,
                                content_hash = EXCLUDED.content_hash,
                                embedding_input_hash = EXCLUDED.embedding_input_hash,
                                last_collected_at = now(),
                                last_collection_run_id = EXCLUDED.last_collection_run_id
                            """,
                            {
                                "provision_id": provision.provision_id,
                                "document_id": document_ids[
                                    provision.document_name
                                ],
                                "chapter_name": provision.chapter_name,
                                "section_name": provision.section_name,
                                "article_no": provision.article_no,
                                "paragraph_no": provision.paragraph_no,
                                "subparagraph_no": provision.subparagraph_no,
                                "title": provision.title,
                                "content": provision.content,
                                "search_text": item.embedding_text,
                                "search_compact": item.search_compact,
                                "sort_order": provision.sort_order,
                                "source_url": provision.source_url,
                                "content_hash": item.content_hash,
                                "embedding_input_hash": (
                                    item.embedding_input_hash
                                ),
                                "run_id": run_id,
                            },
                        )

                        vector = vectors.get(provision.provision_id)
                        if vector is not None:
                            cursor.execute(
                                """
                                INSERT INTO provision_embeddings (
                                    provision_id,
                                    embedding_model,
                                    embedding_dimension,
                                    embedding_input_hash,
                                    embedding
                                ) VALUES (
                                    %(provision_id)s,
                                    %(embedding_model)s,
                                    %(embedding_dimension)s,
                                    %(embedding_input_hash)s,
                                    %(embedding)s
                                )
                                ON CONFLICT (
                                    provision_id,
                                    embedding_model
                                ) DO UPDATE SET
                                    embedding_dimension =
                                        EXCLUDED.embedding_dimension,
                                    embedding_input_hash =
                                        EXCLUDED.embedding_input_hash,
                                    embedding = EXCLUDED.embedding,
                                    created_at = now()
                                """,
                                {
                                    "provision_id": provision.provision_id,
                                    "embedding_model": self.embedding_model,
                                    "embedding_dimension": (
                                        self.embedding_dimension
                                    ),
                                    "embedding_input_hash": (
                                        item.embedding_input_hash
                                    ),
                                    "embedding": vector,
                                },
                            )

                    if full_snapshot:
                        incoming_ids = [
                            item.provision.provision_id for item in prepared
                        ]
                        cursor.execute(
                            """
                            UPDATE legal_provisions
                            SET
                                is_active = false,
                                last_collected_at = now(),
                                last_collection_run_id = %(run_id)s
                            WHERE is_active = true
                              AND NOT (
                                  provision_id =
                                  ANY(%(incoming_ids)s::bigint[])
                              )
                            """,
                            {
                                "run_id": run_id,
                                "incoming_ids": incoming_ids,
                            },
                        )
                        deactivated_count = max(cursor.rowcount, 0)
                        cursor.execute(
                            """
                            UPDATE legal_documents AS d
                            SET
                                is_active = EXISTS (
                                    SELECT 1
                                    FROM legal_provisions AS p
                                    WHERE p.document_id = d.document_id
                                      AND p.is_active = true
                                ),
                                last_collected_at = now(),
                                last_collection_run_id = %(run_id)s
                            """,
                            {"run_id": run_id},
                        )

                    cursor.execute(
                        """
                        UPDATE collection_runs
                        SET
                            status = 'succeeded',
                            completed_at = now()
                        WHERE run_id = %(run_id)s
                        """,
                        {"run_id": run_id},
                    )
        return deactivated_count


__all__ = [
    "DatabaseSyncError",
    "DatabaseSyncService",
    "SyncSummary",
    "build_embedding_text",
]
