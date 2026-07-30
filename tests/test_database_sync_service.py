from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest

from models.legal_provision import LegalProvision
from services import database_sync_service as sync_module
from services.database_sync_service import (
    DatabaseSyncError,
    DatabaseSyncService,
    build_embedding_text,
)


def make_settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "ollama_embedding_model": "embeddinggemma",
        "embedding_dimension": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_provision(
    provision_id: int,
    *,
    article_no: str | None = None,
    content: str | None = None,
    sort_order: int | None = None,
    document_name: str = "測試法規",
    source_url: str = "https://example.test/law",
    paragraph_no: int | None = None,
    active: bool = True,
) -> LegalProvision:
    return LegalProvision(
        provision_id=provision_id,
        document_name=document_name,
        article_no=article_no or f"第{provision_id}條",
        paragraph_no=paragraph_no,
        content=content or f"第 {provision_id} 筆內容",
        sort_order=provision_id if sort_order is None else sort_order,
        source_url=source_url,
        is_active=active,
    )


def embedding_hash(provision: LegalProvision) -> str:
    return hashlib.sha256(
        provision.search_text.encode("utf-8")
    ).hexdigest()


def test_embedding_text_does_not_duplicate_equivalent_whitespace() -> None:
    provision = LegalProvision(
        provision_id=1,
        document_name="測試法規",
        chapter_name="第一章",
        article_no="第1條",
        title="目的",
        content="第一段。\n\n第二段。",
        search_text=(
            "測試法規   第一章\n第1條\t目的  第一段。\n第二段。"
        ),
        sort_order=1,
    )

    embedding_text = build_embedding_text(provision)

    assert embedding_text == "測試法規 第一章 第1條 目的 第一段。 第二段。"
    assert embedding_text.count("測試法規") == 1


class FakeEmbeddingService:
    def __init__(
        self,
        *,
        dimension: int = 3,
        override: np.ndarray | None = None,
    ) -> None:
        self.dimension = dimension
        self.override = override
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.batches.append(list(texts))
        if self.override is not None:
            return self.override
        return np.array(
            [
                [
                    float(len(text)),
                    float(batch_index + 1),
                    1.0,
                ][: self.dimension]
                for batch_index, text in enumerate(texts)
            ],
            dtype=np.float32,
        )


class RecordingSyncService(DatabaseSyncService):
    def __init__(
        self,
        *,
        existing: dict[int, object] | None = None,
        embedding_service: object | None = None,
        deactivated_count: int = 0,
    ) -> None:
        super().__init__(
            make_settings(),
            database=object(),
            embedding_service=embedding_service or FakeEmbeddingService(),
        )
        self.existing = dict(existing or {})
        self.deactivated_count = deactivated_count
        self.write_calls: list[dict[str, object]] = []

    def _load_existing(self):
        return dict(self.existing)

    def _write_snapshot(
        self,
        prepared,
        *,
        vectors,
        run_id,
        source_path,
        source_fingerprint,
        full_snapshot,
    ) -> int:
        self.write_calls.append(
            {
                "prepared": list(prepared),
                "vectors": dict(vectors),
                "run_id": run_id,
                "source_path": source_path,
                "source_fingerprint": source_fingerprint,
                "full_snapshot": full_snapshot,
            }
        )
        return self.deactivated_count


def existing_provision(
    provision: LegalProvision,
    *,
    embedding_input_hash: str | None = None,
    has_current_embedding: bool = True,
):
    return sync_module._ExistingProvision(
        provision_id=provision.provision_id,
        stable_key=(
            provision.document_name,
            provision.article_no,
            provision.paragraph_no,
            provision.subparagraph_no,
        ),
        embedding_input_hash=(
            embedding_input_hash or embedding_hash(provision)
        ),
        has_current_embedding=has_current_embedding,
    )


@pytest.mark.parametrize(
    ("provisions", "message"),
    [
        ([], "不可為空"),
        (
            [make_provision(1, sort_order=2)],
            "sort_order",
        ),
        (
            [make_provision(1, active=False)],
            "is_active=true",
        ),
        (
            [
                make_provision(1, article_no="第1條", sort_order=1),
                make_provision(2, article_no="第1條", sort_order=2),
            ],
            "重複 stable key",
        ),
        (
            [
                make_provision(
                    1,
                    article_no="第1條",
                    sort_order=1,
                    source_url="https://example.test/one",
                ),
                make_provision(
                    2,
                    article_no="第2條",
                    sort_order=2,
                    source_url="https://example.test/two",
                ),
            ],
            "多個不同 source_url",
        ),
    ],
)
def test_sync_rejects_unsafe_snapshot_before_embedding_or_write(
    provisions: list[LegalProvision],
    message: str,
    tmp_path: Path,
) -> None:
    embedding = FakeEmbeddingService()
    service = RecordingSyncService(embedding_service=embedding)

    with pytest.raises(DatabaseSyncError, match=message):
        service.sync(provisions, source_path=tmp_path / "input.json")

    assert embedding.batches == []
    assert service.write_calls == []


def test_sync_rejects_duplicate_incoming_provision_id(
    tmp_path: Path,
) -> None:
    embedding = FakeEmbeddingService()
    service = RecordingSyncService(embedding_service=embedding)
    provisions = [
        make_provision(1, article_no="第1條", sort_order=1),
        make_provision(1, article_no="第2條", sort_order=2),
    ]

    with pytest.raises(DatabaseSyncError, match="provision_id"):
        service.sync(provisions, source_path=tmp_path / "input.json")

    assert embedding.batches == []
    assert service.write_calls == []


def test_sync_only_embeds_new_or_stale_provisions(
    tmp_path: Path,
) -> None:
    reusable = make_provision(1, sort_order=1)
    stale = make_provision(2, sort_order=2)
    new = make_provision(3, sort_order=3)
    embedding = FakeEmbeddingService()
    service = RecordingSyncService(
        existing={
            1: existing_provision(reusable),
            2: existing_provision(
                stale,
                embedding_input_hash="0" * 64,
            ),
        },
        embedding_service=embedding,
    )

    summary = service.sync(
        [reusable, stale, new],
        source_path=tmp_path / "input.json",
        batch_size=2,
    )

    assert embedding.batches == [[stale.search_text, new.search_text]]
    assert summary.embedded_count == 2
    assert summary.reused_embedding_count == 1
    assert summary.provision_count == 3
    assert summary.document_count == 1
    assert summary.deactivated_count == 0
    assert UUID(summary.run_id)
    assert len(summary.source_fingerprint) == 64
    assert len(service.write_calls) == 1
    write = service.write_calls[0]
    assert set(write["vectors"]) == {2, 3}
    assert write["full_snapshot"] is False


def test_missing_current_embedding_is_regenerated_even_when_hash_matches(
    tmp_path: Path,
) -> None:
    provision = make_provision(1)
    embedding = FakeEmbeddingService()
    service = RecordingSyncService(
        existing={
            1: existing_provision(
                provision,
                has_current_embedding=False,
            )
        },
        embedding_service=embedding,
    )

    summary = service.sync(
        [provision],
        source_path=tmp_path / "input.json",
    )

    assert embedding.batches == [[provision.search_text]]
    assert summary.embedded_count == 1
    assert summary.reused_embedding_count == 0


def test_sync_batches_embeddings_and_propagates_full_snapshot(
    tmp_path: Path,
) -> None:
    provisions = [
        make_provision(index, sort_order=index)
        for index in range(1, 6)
    ]
    embedding = FakeEmbeddingService()
    service = RecordingSyncService(
        embedding_service=embedding,
        deactivated_count=4,
    )

    summary = service.sync(
        provisions,
        source_path=tmp_path / "input.json",
        batch_size=2,
        full_snapshot=True,
    )

    assert [len(batch) for batch in embedding.batches] == [2, 2, 1]
    assert summary.embedded_count == 5
    assert summary.deactivated_count == 4
    assert service.write_calls[0]["full_snapshot"] is True


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((1, 2), dtype=np.float32),
        np.array([[1.0, np.nan, 2.0]], dtype=np.float32),
        np.array([[1.0, np.inf, 2.0]], dtype=np.float32),
    ],
)
def test_invalid_embedding_batch_never_reaches_database_write(
    vectors: np.ndarray,
    tmp_path: Path,
) -> None:
    service = RecordingSyncService(
        embedding_service=FakeEmbeddingService(override=vectors)
    )

    with pytest.raises(DatabaseSyncError, match="Embedding"):
        service.sync(
            [make_provision(1)],
            source_path=tmp_path / "input.json",
        )

    assert service.write_calls == []


def test_identity_collision_is_rejected_before_embedding(
    tmp_path: Path,
) -> None:
    old = make_provision(1, article_no="第1條")
    incoming = make_provision(1, article_no="第99條")
    embedding = FakeEmbeddingService()
    service = RecordingSyncService(
        existing={1: existing_provision(old)},
        embedding_service=embedding,
    )

    with pytest.raises(DatabaseSyncError, match="其他 stable key"):
        service.sync(
            [incoming],
            source_path=tmp_path / "input.json",
        )

    assert embedding.batches == []
    assert service.write_calls == []


def test_first_paragraph_split_may_keep_original_provision_id(
    tmp_path: Path,
) -> None:
    old = make_provision(1, article_no="第1條")
    first_paragraph = make_provision(
        1,
        article_no="第1條",
        paragraph_no=1,
    )
    embedding = FakeEmbeddingService()
    service = RecordingSyncService(
        existing={1: existing_provision(old)},
        embedding_service=embedding,
    )

    summary = service.sync(
        [first_paragraph],
        source_path=tmp_path / "input.json",
    )

    assert summary.embedded_count == 0
    assert summary.reused_embedding_count == 1
    assert embedding.batches == []
