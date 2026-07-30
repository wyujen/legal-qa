from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from models.legal_provision import LegalProvision
from models.retrieval_candidate import RetrievalCandidate
from services.database_service import DatabaseConnectionError
from services.retrieval_service import (
    RetrievalDataError,
    RetrievalIndexNotBuiltError,
    RetrievalService,
    cosine_similarity,
    keyword_score,
    provisions_fingerprint,
)


class FakeEmbeddingService:
    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.calls: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return self.vector


class FakeProvisionSearchRepository:
    def __init__(
        self,
        candidates: list[RetrievalCandidate] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.candidates = list(candidates or [])
        self.error = error
        self.calls: list[dict[str, object]] = []

    def search_candidates(
        self,
        query_vector,
        *,
        embedding_model: str,
        limit: int,
        compact_query: str = "",
        keyword_terms=(),
    ) -> list[RetrievalCandidate]:
        self.calls.append(
            {
                "query_vector": np.asarray(
                    query_vector,
                    dtype=np.float32,
                ).copy(),
                "embedding_model": embedding_model,
                "limit": limit,
                "compact_query": compact_query,
                "keyword_terms": tuple(keyword_terms),
            }
        )
        if self.error is not None:
            raise self.error
        return list(self.candidates)


def make_settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "vector_weight": 0.65,
        "keyword_weight": 0.35,
        "retrieval_top_k": 6,
        "retrieval_candidate_k": 50,
        "retrieval_min_score": -1.0,
        "ollama_embedding_model": "embeddinggemma",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_provision(
    provision_id: int,
    *,
    content: str,
    title: str = "",
    active: bool = True,
) -> LegalProvision:
    return LegalProvision(
        provision_id=provision_id,
        document_name="測試法規",
        chapter_name="第一章",
        section_name="",
        article_no=f"第{provision_id}條",
        paragraph_no=None,
        subparagraph_no=None,
        title=title,
        content=content,
        search_text="",
        sort_order=provision_id,
        source_url="",
        is_active=active,
    )


def make_service(
    provisions: list[LegalProvision],
    embeddings: np.ndarray,
    query_vector: list[float],
    **setting_overrides: object,
) -> tuple[RetrievalService, FakeEmbeddingService]:
    fake = FakeEmbeddingService(query_vector)
    service = RetrievalService(
        settings=make_settings(**setting_overrides),
        embedding_service=fake,
        provisions=provisions,
        embeddings=embeddings,
    )
    return service, fake


def make_database_service(
    candidates: list[RetrievalCandidate],
    query_vector: list[float],
    *,
    repository: FakeProvisionSearchRepository | None = None,
    **setting_overrides: object,
) -> tuple[
    RetrievalService,
    FakeEmbeddingService,
    FakeProvisionSearchRepository,
]:
    fake_embedding = FakeEmbeddingService(query_vector)
    fake_repository = repository or FakeProvisionSearchRepository(candidates)
    service = RetrievalService(
        settings=make_settings(**setting_overrides),
        embedding_service=fake_embedding,
        repository=fake_repository,
    )
    return service, fake_embedding, fake_repository


def ready_metadata(
    provisions: list[LegalProvision],
    *,
    dimension: int = 2,
) -> dict[str, object]:
    return {
        "status": "ready",
        "embedding_model": "embeddinggemma",
        "vector_dimension": dimension,
        "created_at": "2026-07-28T00:00:00+00:00",
        "provision_count": len(provisions),
        "provision_ids": [item.provision_id for item in provisions],
        "content_fingerprint": provisions_fingerprint(provisions),
    }


def test_cosine_similarity_has_expected_order() -> None:
    scores = cosine_similarity(
        [1.0, 0.0],
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )

    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(2**-0.5)
    assert scores[0] == pytest.approx(0.0)


def test_keyword_score_is_query_term_coverage() -> None:
    assert keyword_score("租賃 終止", "租賃契約得依法終止") == pytest.approx(1.0)
    assert keyword_score("租賃 損害", "本條處理租賃契約") == pytest.approx(0.5)
    assert keyword_score("租賃 終止", "買賣契約") == pytest.approx(0.0)


def test_hybrid_score_uses_configurable_weights() -> None:
    provisions = [
        make_provision(1, content="完全無關的文字"),
        make_provision(2, content="租賃契約"),
    ]
    service, _ = make_service(
        provisions,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [1.0, 0.0],
        vector_weight=0.2,
        keyword_weight=0.8,
    )

    results = service.retrieve("租賃")

    assert [result.provision_id for result in results] == [2, 1]
    assert results[0].vector_score == pytest.approx(0.0)
    assert results[0].keyword_score == pytest.approx(1.0)
    assert results[0].final_score == pytest.approx(0.8)
    assert results[1].final_score == pytest.approx(0.2)


def test_retrieve_sorts_then_applies_top_k_and_threshold() -> None:
    provisions = [
        make_provision(1, content="甲"),
        make_provision(2, content="乙"),
        make_provision(3, content="丙"),
    ]
    service, _ = make_service(
        provisions,
        np.array(
            [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]],
            dtype=np.float32,
        ),
        [1.0, 0.0],
        vector_weight=1.0,
        keyword_weight=0.0,
    )

    results = service.retrieve("查詢", top_k=1, min_score=0.5)

    assert [result.provision_id for result in results] == [1]


def test_database_retrieval_passes_bounded_query_context_and_reranks() -> None:
    vector_match = make_provision(1, content="完全無關的文字")
    keyword_match = make_provision(2, content="租賃終止之要件")
    service, fake_embedding, repository = make_database_service(
        [
            RetrievalCandidate(
                provision=vector_match,
                vector_score=0.95,
            ),
            RetrievalCandidate(
                provision=keyword_match,
                vector_score=0.2,
            ),
        ],
        [1.0, 0.0],
        vector_weight=0.2,
        keyword_weight=0.8,
        retrieval_candidate_k=40,
    )

    results = service.retrieve("租賃終止")

    assert [item.provision_id for item in results] == [2, 1]
    assert results[0].keyword_score == pytest.approx(1.0)
    assert results[0].final_score == pytest.approx(0.84)
    assert fake_embedding.calls == ["租賃終止"]
    assert len(repository.calls) == 1
    call = repository.calls[0]
    np.testing.assert_array_equal(
        call["query_vector"],
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert call["embedding_model"] == "embeddinggemma"
    assert call["limit"] == 40
    assert call["compact_query"] == "租賃終止"
    assert call["keyword_terms"] == ("租賃", "終止", "賃終")


def test_database_retrieval_filters_inactive_and_non_finite_candidates() -> None:
    service, _, _ = make_database_service(
        [
            RetrievalCandidate(
                provision=make_provision(
                    1,
                    content="租賃終止",
                    active=False,
                ),
                vector_score=1.0,
            ),
            RetrievalCandidate(
                provision=make_provision(2, content="租賃終止"),
                vector_score=float("nan"),
            ),
            RetrievalCandidate(
                provision=make_provision(3, content="租賃終止"),
                vector_score=0.5,
            ),
        ],
        [1.0, 0.0],
    )

    results = service.retrieve("租賃終止")

    assert [item.provision_id for item in results] == [3]


def test_database_error_is_wrapped_without_calling_repository_twice() -> None:
    repository = FakeProvisionSearchRepository(
        error=DatabaseConnectionError(
            "無法連線到 PostgreSQL 法規資料庫，請確認資料庫服務已啟動。"
        )
    )
    service, _, repository = make_database_service(
        [],
        [1.0, 0.0],
        repository=repository,
    )

    with pytest.raises(RetrievalDataError, match="無法連線到 PostgreSQL"):
        service.retrieve("租賃")

    assert len(repository.calls) == 1


def test_empty_dataset_returns_without_embedding_call() -> None:
    service, fake = make_service(
        [],
        np.empty((0, 0), dtype=np.float32),
        [1.0, 0.0],
    )

    assert service.retrieve("租賃") == []
    assert fake.calls == []


def test_zero_vectors_score_zero_without_nan() -> None:
    assert cosine_similarity(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    ).tolist() == [0.0, 0.0]

    service, _ = make_service(
        [make_provision(1, content="甲")],
        np.array([[0.0, 0.0]], dtype=np.float32),
        [0.0, 0.0],
        vector_weight=1.0,
        keyword_weight=0.0,
    )
    result = service.retrieve("查詢")[0]
    assert result.vector_score == 0.0
    assert result.final_score == 0.0


def test_query_and_index_dimension_mismatch_is_clear() -> None:
    service, _ = make_service(
        [make_provision(1, content="甲")],
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        [1.0, 0.0],
    )

    with pytest.raises(RetrievalDataError, match="維度不一致"):
        service.retrieve("查詢")


def test_provision_and_embedding_count_mismatch_fails_at_startup() -> None:
    with pytest.raises(RetrievalDataError, match="筆數不一致"):
        make_service(
            [make_provision(1, content="甲")],
            np.empty((0, 2), dtype=np.float32),
            [1.0, 0.0],
        )


def test_metadata_model_mismatch_requires_rebuild() -> None:
    with pytest.raises(RetrievalDataError, match="模型不一致"):
        RetrievalService(
            settings=make_settings(),
            embedding_service=FakeEmbeddingService([1.0, 0.0]),
            provisions=[make_provision(1, content="甲")],
            embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            metadata={
                "status": "ready",
                "embedding_model": "other-model",
                "vector_dimension": 2,
                "provision_count": 1,
                "provision_ids": [1],
            },
        )


def test_complete_ready_metadata_with_matching_fingerprint_is_accepted() -> None:
    provisions = [make_provision(1, content="租賃契約")]
    service = RetrievalService(
        settings=make_settings(),
        embedding_service=FakeEmbeddingService([1.0, 0.0]),
        provisions=provisions,
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        metadata=ready_metadata(provisions),
    )

    assert service.retrieve("租賃")[0].provision_id == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"embedding_model": ""}, "embedding_model"),
        ({"provision_count": "1"}, "provision_count"),
        ({"vector_dimension": None}, "vector_dimension"),
        ({"created_at": None}, "created_at"),
        ({"provision_ids": None}, "provision_ids"),
    ],
)
def test_invalid_ready_metadata_values_are_rejected(
    updates: dict[str, object],
    message: str,
) -> None:
    provisions = [make_provision(1, content="甲")]
    metadata = ready_metadata(provisions)
    metadata.update(updates)

    with pytest.raises(RetrievalDataError, match=message):
        RetrievalService(
            settings=make_settings(),
            embedding_service=FakeEmbeddingService([1.0, 0.0]),
            provisions=provisions,
            embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            metadata=metadata,
        )


def test_changed_provision_content_invalidates_existing_index() -> None:
    original = [make_provision(1, content="原始內容")]
    changed = [make_provision(1, content="已更新內容")]

    with pytest.raises(RetrievalDataError, match="索引已過期"):
        RetrievalService(
            settings=make_settings(),
            embedding_service=FakeEmbeddingService([1.0, 0.0]),
            provisions=changed,
            embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            metadata=ready_metadata(original),
        )


def test_not_built_metadata_has_actionable_command(tmp_path) -> None:
    provisions_path = tmp_path / "legal_provisions.json"
    provisions_path.write_text("[]", encoding="utf-8")
    metadata_path = tmp_path / "embedding_metadata.json"
    metadata_path.write_text('{"status": "not_built"}', encoding="utf-8")
    settings = make_settings(
        legal_provisions_path=provisions_path,
        legal_embeddings_path=tmp_path / "legal_embeddings.npy",
        embedding_metadata_path=metadata_path,
    )

    with pytest.raises(
        RetrievalIndexNotBuiltError,
        match=r"python scripts/build_embeddings.py",
    ):
        RetrievalService(
            settings=settings,
            embedding_service=FakeEmbeddingService([1.0, 0.0]),
        )
