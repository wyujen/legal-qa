"""PostgreSQL + pgvector repository for legal provision retrieval."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from models.legal_provision import LegalProvision
from models.retrieval_candidate import RetrievalCandidate
from services.database_service import PostgresDatabase


logger = logging.getLogger(__name__)


class PostgresProvisionRepository:
    """Retrieve only the nearest active provisions from PostgreSQL."""

    def __init__(
        self,
        settings: Any,
        *,
        database: PostgresDatabase | None = None,
    ) -> None:
        self.settings = settings
        self.database = database or PostgresDatabase(settings)
        self.embedding_dimension = int(settings.embedding_dimension)

    def search_candidates(
        self,
        query_vector: Sequence[float] | np.ndarray,
        *,
        embedding_model: str,
        limit: int,
        compact_query: str = "",
        keyword_terms: Sequence[str] = (),
    ) -> list[RetrievalCandidate]:
        to_numpy = getattr(query_vector, "to_numpy", None)
        if callable(to_numpy):
            query_vector = to_numpy()
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.embedding_dimension:
            raise ValueError(
                "查詢 embedding 維度不一致："
                f"預期 {self.embedding_dimension}，實際 {vector.shape}。"
            )
        if not np.isfinite(vector).all():
            raise ValueError("查詢 embedding 含 NaN 或 Infinity。")
        if float(np.linalg.norm(vector)) <= 0:
            raise ValueError("查詢 embedding 是零向量，無法計算 cosine 相似度。")
        if limit < 1:
            return []

        sql = """
            WITH vector_candidates AS MATERIALIZED (
                SELECT e.provision_id
                FROM provision_embeddings AS e
                JOIN legal_provisions AS p
                  ON p.provision_id = e.provision_id
                 AND p.embedding_input_hash = e.embedding_input_hash
                JOIN legal_documents AS d
                  ON d.document_id = p.document_id
                WHERE e.embedding_model = %(embedding_model)s
                  AND p.is_active = true
                  AND d.is_active = true
                ORDER BY e.embedding <=> %(query_vector)s
                LIMIT %(limit)s
            ),
            keyword_scored AS MATERIALIZED (
                SELECT
                    p.provision_id,
                    CASE
                        WHEN %(compact_query)s <> ''
                         AND strpos(
                             p.search_compact,
                             %(compact_query)s
                         ) > 0
                            THEN 1.0
                        WHEN cardinality(
                            %(keyword_terms)s::text[]
                        ) = 0
                            THEN 0.0
                        ELSE (
                            SELECT count(*)::double precision
                                   / cardinality(
                                       %(keyword_terms)s::text[]
                                   )
                            FROM unnest(
                                %(keyword_terms)s::text[]
                            ) AS term(value)
                            WHERE strpos(
                                p.search_compact,
                                term.value
                            ) > 0
                        )
                    END AS keyword_score
                FROM legal_provisions AS p
                JOIN legal_documents AS d
                  ON d.document_id = p.document_id
                WHERE p.is_active = true
                  AND d.is_active = true
                  AND (
                      %(compact_query)s <> ''
                      OR cardinality(
                          %(keyword_terms)s::text[]
                      ) > 0
                  )
            ),
            keyword_candidates AS (
                SELECT provision_id
                FROM keyword_scored
                WHERE keyword_score > 0
                ORDER BY keyword_score DESC, provision_id
                LIMIT %(limit)s
            ),
            candidate_ids AS (
                SELECT provision_id FROM vector_candidates
                UNION
                SELECT provision_id FROM keyword_candidates
            )
            SELECT
                p.provision_id,
                d.document_name,
                p.chapter_name,
                p.section_name,
                p.article_no,
                p.paragraph_no,
                p.subparagraph_no,
                p.title,
                p.content,
                p.search_text,
                p.sort_order,
                p.source_url,
                p.is_active,
                1 - (e.embedding <=> %(query_vector)s) AS vector_score
            FROM candidate_ids AS candidates
            JOIN provision_embeddings AS e
              ON e.provision_id = candidates.provision_id
            JOIN legal_provisions AS p
              ON p.provision_id = e.provision_id
             AND p.embedding_input_hash = e.embedding_input_hash
            JOIN legal_documents AS d
              ON d.document_id = p.document_id
            WHERE e.embedding_model = %(embedding_model)s
              AND p.is_active = true
              AND d.is_active = true
            ORDER BY p.provision_id
        """
        parameters = {
            "query_vector": vector,
            "embedding_model": embedding_model,
            "limit": int(limit),
            "compact_query": str(compact_query),
            "keyword_terms": [str(item) for item in keyword_terms],
        }

        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT set_config(
                        'hnsw.ef_search',
                        %(hnsw_ef_search)s,
                        true
                    )
                    """,
                    {
                        "hnsw_ef_search": str(max(100, int(limit) * 2))
                    },
                )
                cursor.execute(
                    """
                    SELECT set_config(
                        'hnsw.iterative_scan',
                        'strict_order',
                        true
                    )
                    """
                )
                cursor.execute(sql, parameters)
                rows = cursor.fetchall()

        candidates: list[RetrievalCandidate] = []
        for row in rows:
            provision = LegalProvision(
                provision_id=int(row[0]),
                document_name=str(row[1]),
                chapter_name=str(row[2] or ""),
                section_name=str(row[3] or ""),
                article_no=str(row[4]),
                paragraph_no=row[5],
                subparagraph_no=row[6],
                title=str(row[7] or ""),
                content=str(row[8]),
                search_text=str(row[9]),
                sort_order=int(row[10]),
                source_url=str(row[11] or ""),
                is_active=bool(row[12]),
            )
            candidates.append(
                RetrievalCandidate(
                    provision=provision,
                    vector_score=float(row[13]),
                )
            )
        return candidates

    def current_counts(self) -> tuple[int, int]:
        """Return active document and provision counts for diagnostics."""

        sql = """
            SELECT
                count(DISTINCT d.document_id),
                count(p.provision_id)
            FROM legal_documents AS d
            LEFT JOIN legal_provisions AS p
              ON p.document_id = d.document_id
             AND p.is_active = true
            WHERE d.is_active = true
        """
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
        return int(row[0]), int(row[1])


__all__ = ["PostgresProvisionRepository"]
