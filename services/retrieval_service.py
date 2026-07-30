"""In-memory hybrid retrieval using NumPy cosine and transparent keywords."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from models.legal_provision import LegalProvision
from models.retrieval_candidate import RetrievalCandidate
from models.retrieval_result import RetrievalResult
from services.database_service import DatabaseServiceError
from services.embedding_service import EmbeddingService
from services.postgres_repository import PostgresProvisionRepository

logger = logging.getLogger(__name__)


class RetrievalDataError(RuntimeError):
    """Raised when the on-disk retrieval index is missing or inconsistent."""


class RetrievalIndexNotBuiltError(RetrievalDataError):
    """Raised when ``scripts/build_embeddings.py`` has not been run."""


class ProvisionSearchRepository(Protocol):
    def search_candidates(
        self,
        query_vector: Sequence[float] | np.ndarray,
        *,
        embedding_model: str,
        limit: int,
        compact_query: str = "",
        keyword_terms: Sequence[str] = (),
    ) -> list[RetrievalCandidate]: ...


def cosine_similarity(
    query_vector: Sequence[float] | np.ndarray,
    document_vectors: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return cosine similarity against every document vector.

    Zero-length vectors have a similarity of ``0`` rather than producing NaN.
    Shape and embedding-dimension mistakes fail explicitly.
    """

    query = np.asarray(query_vector, dtype=np.float64)
    documents = np.asarray(document_vectors, dtype=np.float64)

    if query.ndim != 1:
        raise ValueError(
            f"查詢 embedding 必須是一維向量，實際 shape={query.shape}。"
        )
    if query.size == 0:
        raise ValueError("查詢 embedding 不可為空。")

    if documents.size == 0:
        if documents.ndim not in (1, 2):
            raise ValueError(
                f"法規 embeddings 必須是二維矩陣，實際 shape={documents.shape}。"
            )
        if documents.ndim == 2 and documents.shape[1] not in (0, query.size):
            raise ValueError(
                "Embedding 維度不一致："
                f"查詢為 {query.size}，法規為 {documents.shape[1]}。"
            )
        return np.empty(0, dtype=np.float64)

    if documents.ndim != 2:
        raise ValueError(
            f"法規 embeddings 必須是二維矩陣，實際 shape={documents.shape}。"
        )
    if documents.shape[1] != query.size:
        raise ValueError(
            "Embedding 維度不一致："
            f"查詢為 {query.size}，法規為 {documents.shape[1]}。"
        )
    if not np.isfinite(query).all() or not np.isfinite(documents).all():
        raise ValueError("Embedding 含 NaN 或 Infinity。")

    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)
    denominators = document_norms * query_norm
    dot_products = documents @ query

    similarities = np.zeros(documents.shape[0], dtype=np.float64)
    np.divide(
        dot_products,
        denominators,
        out=similarities,
        where=denominators > 0,
    )
    # Floating point error can otherwise yield values such as 1.0000000002.
    return np.clip(similarities, -1.0, 1.0)


_ARTICLE_PATTERN = re.compile(
    r"第\s*[0-9０-９零〇一二三四五六七八九十百千兩两之\-－]+\s*條"
    r"(?:\s*之\s*[0-9０-９零〇一二三四五六七八九十]+)?"
)
_LEXICAL_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")
_LOW_INFORMATION_TERMS = {
    "可以",
    "是否",
    "什麼",
    "何謂",
    "如何",
    "怎麼",
    "哪些",
    "哪個",
    "請問",
    "規定",
    "依據",
    "法律",
    "問題",
}


def _normalize_keyword_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def compact_keyword_text(value: str) -> str:
    """Return the same compact representation used by keyword scoring."""

    return re.sub(
        r"[^\w\u3400-\u9fff]+",
        "",
        _normalize_keyword_text(value),
    )


def extract_keyword_terms(query: str) -> set[str]:
    """Extract explainable query terms.

    Latin words/numbers remain whole.  Chinese runs are represented by
    two-character terms, which lets a natural-language question overlap legal
    prose without requiring an external tokenizer.  Common question words are
    ignored.
    """

    normalized = _normalize_keyword_text(query)
    terms: set[str] = set()

    for match in _ARTICLE_PATTERN.finditer(normalized):
        terms.add(re.sub(r"\s+", "", match.group(0)))

    for token in _LEXICAL_PATTERN.findall(normalized):
        if re.fullmatch(r"[a-z0-9]+", token):
            if token not in _LOW_INFORMATION_TERMS:
                terms.add(token)
            continue

        if len(token) == 1:
            continue
        if len(token) == 2:
            if token not in _LOW_INFORMATION_TERMS:
                terms.add(token)
            continue

        for index in range(len(token) - 1):
            bigram = token[index : index + 2]
            if bigram not in _LOW_INFORMATION_TERMS:
                terms.add(bigram)

    return terms


def keyword_score(query: str, searchable_text: str) -> float:
    """Score keyword overlap as ``matched query terms / all query terms``."""

    normalized_query = _normalize_keyword_text(query)
    normalized_text = _normalize_keyword_text(searchable_text)
    if not normalized_query or not normalized_text:
        return 0.0

    # A complete phrase match is the strongest possible lexical signal.
    compact_query = compact_keyword_text(normalized_query)
    compact_text = compact_keyword_text(normalized_text)
    if compact_query and compact_query in compact_text:
        return 1.0

    terms = extract_keyword_terms(normalized_query)
    if not terms:
        return 0.0
    matched = sum(term in compact_text for term in terms)
    return matched / len(terms)


def _value(record: Any, name: str, default: Any = "") -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def provision_search_text(provision: LegalProvision | Mapping[str, Any]) -> str:
    """Return all fields that provide useful human-visible lexical signals."""

    search_text = str(_value(provision, "search_text", "") or "")
    if search_text:
        return search_text
    return " ".join(
        str(_value(provision, field, "") or "")
        for field in (
            "document_name",
            "chapter_name",
            "section_name",
            "article_no",
            "title",
            "content",
        )
    )


def provisions_fingerprint(
    provisions: Sequence[LegalProvision | Mapping[str, Any]],
) -> str:
    """建立包含條文內容與順序的穩定 SHA-256，防止沿用舊向量。"""

    canonical_provisions = [
        (
            provision
            if isinstance(provision, LegalProvision)
            else LegalProvision.model_validate(provision)
        ).model_dump(mode="json")
        for provision in provisions
    ]
    canonical_json = json.dumps(
        canonical_provisions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class RetrievalService:
    """Rank legal provisions with configurable vector/keyword hybrid scoring."""

    def __init__(
        self,
        settings: Any | None = None,
        embedding_service: Any | None = None,
        *,
        provisions: Sequence[LegalProvision | Mapping[str, Any]] | None = None,
        embeddings: Sequence[Sequence[float]] | np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
        repository: ProvisionSearchRepository | None = None,
        vector_weight: float | None = None,
        keyword_weight: float | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        if settings is None:
            from config import Settings

            settings = Settings()

        self.settings = settings
        self.embedding_service = embedding_service or EmbeddingService(settings)
        self.vector_weight = float(
            settings.vector_weight if vector_weight is None else vector_weight
        )
        self.keyword_weight = float(
            settings.keyword_weight if keyword_weight is None else keyword_weight
        )
        self.top_k = int(settings.retrieval_top_k if top_k is None else top_k)
        self.candidate_k = int(
            getattr(settings, "retrieval_candidate_k", max(self.top_k, 50))
        )
        self.min_score = float(
            settings.retrieval_min_score if min_score is None else min_score
        )
        self._validate_configuration()

        self.repository = repository
        self._database_mode = bool(
            repository is not None
            or (
                provisions is None
                and embeddings is None
                and getattr(settings, "database_url", None)
            )
        )
        if self._database_mode:
            self.repository = repository or PostgresProvisionRepository(
                settings
            )
            self.provisions = []
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.metadata = {}
            return

        loaded_metadata = metadata
        if provisions is None:
            provisions = self._load_provisions(
                Path(settings.legal_provisions_path)
            )
        if embeddings is None:
            if loaded_metadata is None:
                loaded_metadata = self._load_metadata(
                    Path(settings.embedding_metadata_path)
                )
            self._require_built_metadata(loaded_metadata)
            embeddings = self._load_embeddings(
                Path(settings.legal_embeddings_path)
            )

        self.provisions = [
            self._coerce_provision(provision) for provision in provisions
        ]
        self.embeddings = self._coerce_embedding_matrix(embeddings)
        self.metadata = dict(loaded_metadata or {})
        self._validate_index()

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Return the highest-scoring active provisions."""

        if not isinstance(question, str):
            raise TypeError("question 必須是字串。")
        if not question.strip():
            return []

        limit = self.top_k if top_k is None else int(top_k)
        if limit <= 0:
            return []
        threshold = self.min_score if min_score is None else float(min_score)

        if not self._database_mode and not self.provisions:
            return []

        query_vector = self._embed_query(question)
        if self._database_mode:
            candidates = self._database_candidates(question, query_vector)
            return self._rank_candidates(
                question,
                candidates,
                limit=limit,
                threshold=threshold,
            )

        try:
            vector_scores = cosine_similarity(query_vector, self.embeddings)
        except ValueError as exc:
            raise RetrievalDataError(f"無法計算檢索分數：{exc}") from exc
        candidates = [
            RetrievalCandidate(
                provision=provision,
                vector_score=float(vector_scores[index]),
            )
            for index, provision in enumerate(self.provisions)
            if bool(provision.is_active)
        ]
        return self._rank_candidates(
            question,
            candidates,
            limit=limit,
            threshold=threshold,
        )

    def _database_candidates(
        self,
        question: str,
        query_vector: np.ndarray,
    ) -> list[RetrievalCandidate]:
        if self.repository is None:
            raise RetrievalDataError("PostgreSQL 檢索 repository 尚未建立。")
        try:
            return list(
                self.repository.search_candidates(
                    query_vector,
                    embedding_model=str(
                        getattr(
                            self.settings,
                            "ollama_embedding_model",
                            "",
                        )
                    ),
                    limit=self.candidate_k,
                    compact_query=compact_keyword_text(question),
                    keyword_terms=sorted(extract_keyword_terms(question)),
                )
            )
        except DatabaseServiceError as exc:
            raise RetrievalDataError(str(exc)) from exc
        except ValueError as exc:
            raise RetrievalDataError(f"無法查詢法規向量：{exc}") from exc

    def _rank_candidates(
        self,
        question: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        limit: int,
        threshold: float,
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        weight_total = self.vector_weight + self.keyword_weight
        vector_weight = self.vector_weight / weight_total
        lexical_weight = self.keyword_weight / weight_total

        for candidate in candidates:
            provision = candidate.provision
            if not bool(provision.is_active):
                continue

            lexical_score = keyword_score(
                question,
                provision_search_text(provision),
            )
            vector_score = float(candidate.vector_score)
            final_score = (
                vector_weight * vector_score + lexical_weight * lexical_score
            )
            if (
                not np.isfinite(vector_score)
                or not np.isfinite(final_score)
                or final_score < threshold
            ):
                continue

            results.append(
                RetrievalResult(
                    provision_id=provision.provision_id,
                    document_name=provision.document_name,
                    article_no=provision.article_no,
                    title=provision.title,
                    content=provision.content,
                    vector_score=vector_score,
                    keyword_score=float(lexical_score),
                    final_score=float(final_score),
                )
            )

        results.sort(
            key=lambda result: (-result.final_score, result.provision_id)
        )
        return results[:limit]

    # A short alias is useful in orchestration code.
    search = retrieve

    def calculate_keyword_score(
        self,
        question: str,
        provision_or_text: LegalProvision | Mapping[str, Any] | str,
    ) -> float:
        text = (
            provision_or_text
            if isinstance(provision_or_text, str)
            else provision_search_text(provision_or_text)
        )
        return keyword_score(question, text)

    def _embed_query(self, question: str) -> np.ndarray:
        embed_query = getattr(self.embedding_service, "embed_query", None)
        if callable(embed_query):
            vector = embed_query(question)
        else:
            embed_text = getattr(self.embedding_service, "embed_text", None)
            if not callable(embed_text):
                raise TypeError(
                    "embedding_service 必須提供 embed_query() 或 embed_text()。"
                )
            vector = embed_text(question)
        return np.asarray(vector, dtype=np.float32)

    def _validate_configuration(self) -> None:
        if self.vector_weight < 0 or self.keyword_weight < 0:
            raise ValueError("vector_weight 與 keyword_weight 不可為負數。")
        if self.vector_weight + self.keyword_weight <= 0:
            raise ValueError("vector_weight 與 keyword_weight 不可同時為 0。")
        if self.top_k < 1:
            raise ValueError("top_k 必須至少為 1。")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k 不可小於 top_k。")
        if not np.isfinite(self.min_score):
            raise ValueError("min_score 必須是有限數值。")

    @staticmethod
    def _coerce_provision(
        provision: LegalProvision | Mapping[str, Any],
    ) -> LegalProvision:
        if isinstance(provision, LegalProvision):
            return provision
        return LegalProvision.model_validate(provision)

    @staticmethod
    def _coerce_embedding_matrix(
        embeddings: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        try:
            matrix = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RetrievalDataError(
                "法規 embeddings 必須是規則的數值矩陣。"
            ) from exc

        if matrix.size == 0 and matrix.ndim == 1:
            return np.empty((0, 0), dtype=np.float32)
        if matrix.ndim != 2:
            raise RetrievalDataError(
                f"法規 embeddings 必須是二維矩陣，實際 shape={matrix.shape}。"
            )
        if not np.isfinite(matrix).all():
            raise RetrievalDataError("法規 embeddings 含 NaN 或 Infinity。")
        return matrix

    def _validate_index(self) -> None:
        if self.embeddings.shape[0] != len(self.provisions):
            raise RetrievalDataError(
                "法規與 embedding 筆數不一致："
                f"法規 {len(self.provisions)} 筆，embedding "
                f"{self.embeddings.shape[0]} 筆。請重新執行 "
                "python scripts/build_embeddings.py。"
            )

        if not self.metadata:
            return

        status = str(self.metadata.get("status", "")).casefold()
        if status in {"not_built", "missing", "pending"}:
            raise RetrievalIndexNotBuiltError(
                "Embedding 尚未建立；請執行 python scripts/build_embeddings.py。"
            )
        if status != "ready":
            raise RetrievalDataError(
                "Embedding metadata 的 status 無效；"
                "請重新執行 python scripts/build_embeddings.py。"
            )

        indexed_model = self.metadata.get("embedding_model")
        if not isinstance(indexed_model, str) or not indexed_model.strip():
            raise RetrievalDataError(
                "Embedding metadata 的 embedding_model 無效；"
                "請重新建立 embeddings。"
            )
        configured_model = str(
            getattr(
                self.settings,
                "ollama_embedding_model",
                getattr(self.settings, "embedding_model", ""),
            )
        )
        if not configured_model or configured_model != indexed_model:
            raise RetrievalDataError(
                f"Embedding 模型不一致：索引使用「{indexed_model}」，"
                f"目前設定為「{configured_model}」。請重新建立 embeddings。"
            )

        metadata_count = self.metadata.get("provision_count")
        if (
            isinstance(metadata_count, bool)
            or not isinstance(metadata_count, int)
            or metadata_count < 0
        ):
            raise RetrievalDataError(
                "Embedding metadata 的 provision_count 無效；"
                "請重新建立 embeddings。"
            )
        if metadata_count != len(self.provisions):
            raise RetrievalDataError(
                "Embedding metadata 的法規筆數與資料不一致；"
                "請重新執行 python scripts/build_embeddings.py。"
            )

        dimension = self.metadata.get("vector_dimension")
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            or (metadata_count > 0 and dimension == 0)
        ):
            raise RetrievalDataError(
                "Embedding metadata 的 vector_dimension 無效；"
                "請重新建立 embeddings。"
            )
        if dimension != self.embeddings.shape[1]:
            raise RetrievalDataError(
                "Embedding metadata 的向量維度與 .npy 檔不一致；"
                "請重新執行 python scripts/build_embeddings.py。"
            )

        metadata_ids = self.metadata.get("provision_ids")
        expected_ids = [provision.provision_id for provision in self.provisions]
        if (
            not isinstance(metadata_ids, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in metadata_ids
            )
        ):
            raise RetrievalDataError(
                "Embedding metadata 的 provision_ids 無效；"
                "請重新建立 embeddings。"
            )
        if metadata_ids != expected_ids:
            raise RetrievalDataError(
                "Embedding metadata 的 provision_ids 與法規順序不一致；"
                "請重新執行 python scripts/build_embeddings.py。"
            )

        created_at = self.metadata.get("created_at")
        if not isinstance(created_at, str) or not created_at.strip():
            raise RetrievalDataError(
                "Embedding metadata 的 created_at 無效；請重新建立 embeddings。"
            )
        try:
            parsed_created_at = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise RetrievalDataError(
                "Embedding metadata 的 created_at 不是合法 ISO-8601；"
                "請重新建立 embeddings。"
            ) from exc
        if parsed_created_at.tzinfo is None:
            raise RetrievalDataError(
                "Embedding metadata 的 created_at 必須包含時區；"
                "請重新建立 embeddings。"
            )

        indexed_fingerprint = self.metadata.get("content_fingerprint")
        if (
            not isinstance(indexed_fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", indexed_fingerprint)
        ):
            raise RetrievalDataError(
                "Embedding metadata 缺少有效的 content_fingerprint；"
                "請重新建立 embeddings。"
            )
        current_fingerprint = provisions_fingerprint(self.provisions)
        if indexed_fingerprint != current_fingerprint:
            raise RetrievalDataError(
                "法規內容已變更，現有 Embedding 索引已過期；"
                "請重新執行 python scripts/build_embeddings.py。"
            )

    @staticmethod
    def _load_provisions(path: Path) -> list[LegalProvision]:
        if not path.exists():
            raise RetrievalDataError(f"找不到法規資料：{path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.exception("讀取法規 JSON 失敗：%s", path)
            raise RetrievalDataError(
                f"無法讀取法規 JSON：{path}。請檢查檔案格式與權限。"
            ) from exc
        if not isinstance(payload, list):
            raise RetrievalDataError("法規 JSON 根節點必須是陣列。")
        try:
            return [LegalProvision.model_validate(item) for item in payload]
        except Exception as exc:
            logger.exception("驗證法規 JSON 失敗：%s", path)
            raise RetrievalDataError(
                "法規 JSON 欄位格式錯誤，請檢查執行紀錄。"
            ) from exc

    @staticmethod
    def _load_embeddings(path: Path) -> np.ndarray:
        if not path.exists():
            raise RetrievalIndexNotBuiltError(
                f"找不到 embedding 檔案：{path}。"
                "請執行 python scripts/build_embeddings.py。"
            )
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            logger.exception("讀取 embedding 檔案失敗：%s", path)
            raise RetrievalDataError(
                f"無法讀取 embedding 檔案：{path}。請重新建立 embeddings。"
            ) from exc

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise RetrievalIndexNotBuiltError(
                f"找不到 embedding metadata：{path}。"
                "請執行 python scripts/build_embeddings.py。"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.exception("讀取 embedding metadata 失敗：%s", path)
            raise RetrievalDataError(
                f"無法讀取 embedding metadata：{path}。請重新建立 embeddings。"
            ) from exc
        if not isinstance(payload, dict):
            raise RetrievalDataError("Embedding metadata 根節點必須是物件。")
        return payload

    @staticmethod
    def _require_built_metadata(metadata: Mapping[str, Any]) -> None:
        status = str(metadata.get("status", "")).casefold()
        if status in {
            "not_built",
            "missing",
            "pending",
        }:
            raise RetrievalIndexNotBuiltError(
                "Embedding 尚未建立；請執行 python scripts/build_embeddings.py。"
            )
        if not metadata:
            raise RetrievalDataError(
                "Embedding metadata 是空的；"
                "請重新執行 python scripts/build_embeddings.py。"
            )

        required_groups = {
            "embedding 模型": ("embedding_model", "model"),
            "向量維度": (
                "vector_dimension",
                "dimension",
                "embedding_dimension",
            ),
            "建立時間": ("created_at",),
            "法規筆數": ("provision_count", "count"),
            "provision_ids": ("provision_ids",),
        }
        missing = [
            label
            for label, alternatives in required_groups.items()
            if not any(key in metadata for key in alternatives)
        ]
        if missing:
            raise RetrievalDataError(
                "Embedding metadata 缺少"
                f"「{'、'.join(missing)}」；請重新執行 "
                "python scripts/build_embeddings.py。"
            )


compute_cosine_similarity = cosine_similarity
compute_keyword_score = keyword_score

__all__ = [
    "RetrievalDataError",
    "RetrievalIndexNotBuiltError",
    "RetrievalService",
    "compute_cosine_similarity",
    "compute_keyword_score",
    "compact_keyword_text",
    "cosine_similarity",
    "extract_keyword_terms",
    "keyword_score",
    "provision_search_text",
    "provisions_fingerprint",
]
