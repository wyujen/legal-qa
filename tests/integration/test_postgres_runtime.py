from __future__ import annotations

import os

import pytest

from config import Settings
from services.database_service import PostgresDatabase
from services.postgres_repository import PostgresProvisionRepository


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.postgres


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL 未設定，略過 PostgreSQL 整合測試。",
)
def test_pgvector_schema_and_sample_retrieval_are_ready() -> None:
    settings = Settings(database_url=TEST_DATABASE_URL)
    database = PostgresDatabase(settings)
    repository = PostgresProvisionRepository(
        settings,
        database=database,
    )

    with database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector'
                """
            )
            assert cursor.fetchone()[0]
            cursor.execute(
                """
                SELECT p.provision_id, e.embedding
                FROM provision_embeddings AS e
                JOIN legal_provisions AS p
                  ON p.provision_id = e.provision_id
                WHERE e.embedding_model = %s
                  AND p.is_active = true
                  AND p.embedding_input_hash = e.embedding_input_hash
                ORDER BY p.provision_id
                LIMIT 1
                """,
                (settings.ollama_embedding_model,),
            )
            provision_id, vector = cursor.fetchone()

    candidates = repository.search_candidates(
        vector,
        embedding_model=settings.ollama_embedding_model,
        limit=6,
    )

    assert candidates
    matching = next(
        item
        for item in candidates
        if item.provision.provision_id == provision_id
    )
    assert matching.vector_score == pytest.approx(1.0, abs=1e-5)
    document_count, provision_count = repository.current_counts()
    assert document_count >= 1
    assert provision_count >= 1
