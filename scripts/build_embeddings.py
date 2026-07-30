"""Build the local NumPy embedding index.

Run from the repository root:

    python scripts/build_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ``python scripts/build_embeddings.py`` puts ``scripts`` (not the repository
# root) on sys.path.  Add the root before importing project modules.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import Settings  # noqa: E402
from services.document_service import load_provisions  # noqa: E402
from services.embedding_service import EmbeddingService  # noqa: E402
from services.retrieval_service import provisions_fingerprint  # noqa: E402


LOGGER = logging.getLogger("build_embeddings")


def _embedding_model_name(settings: Any) -> str:
    model = getattr(
        settings,
        "ollama_embedding_model",
        getattr(settings, "embedding_model", None),
    )
    if not model:
        raise ValueError(
            "設定缺少 ollama_embedding_model（或 embedding_model）。"
        )
    return str(model)


def _atomic_save_npy(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npy",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.save(temporary_file, matrix, allow_pickle=False)
        temporary_path.replace(path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        temporary_path.replace(path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def build_embeddings(
    *,
    settings: Any | None = None,
    provisions_path: str | Path | None = None,
    embeddings_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    embedding_service: Any | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Generate vectors, save ``.npy`` plus metadata, and return metadata."""

    settings = settings or Settings()
    if batch_size < 1:
        raise ValueError("batch_size 必須至少為 1。")

    source_path = Path(provisions_path or settings.legal_provisions_path)
    output_path = Path(embeddings_path or settings.legal_embeddings_path)
    output_metadata_path = Path(
        metadata_path or settings.embedding_metadata_path
    )
    provisions = load_provisions(source_path)
    service = embedding_service or EmbeddingService(settings)

    LOGGER.info(
        "準備以 %s 建立 %d 筆法規 embeddings",
        _embedding_model_name(settings),
        len(provisions),
    )

    matrices: list[np.ndarray] = []
    for start in range(0, len(provisions), batch_size):
        batch = provisions[start : start + batch_size]
        texts = [provision.search_text for provision in batch]
        embed_documents = getattr(service, "embed_documents", None)
        if callable(embed_documents):
            vectors = embed_documents(texts)
        else:
            embed_texts = getattr(service, "embed_texts", None)
            if not callable(embed_texts):
                raise TypeError(
                    "embedding_service 必須提供 embed_documents() 或 embed_texts()。"
                )
            vectors = embed_texts(texts)

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(batch):
            raise ValueError(
                "Embedding service 回傳 shape 不正確："
                f"批次預期 {len(batch)} 筆，實際為 {matrix.shape}。"
            )
        if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
            raise ValueError("Embedding service 回傳空向量、NaN 或 Infinity。")
        matrices.append(matrix)
        LOGGER.info("已完成 %d/%d 筆", min(start + len(batch), len(provisions)), len(provisions))

    if matrices:
        dimensions = {matrix.shape[1] for matrix in matrices}
        if len(dimensions) != 1:
            raise ValueError(
                f"不同批次的 embedding 維度不一致：{sorted(dimensions)}。"
            )
        embedding_matrix = np.concatenate(matrices, axis=0)
    else:
        embedding_matrix = np.empty((0, 0), dtype=np.float32)

    metadata = {
        "status": "ready",
        "embedding_model": _embedding_model_name(settings),
        "vector_dimension": int(embedding_matrix.shape[1]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provision_count": len(provisions),
        "provision_ids": [
            provision.provision_id for provision in provisions
        ],
        "content_fingerprint": provisions_fingerprint(provisions),
    }

    _atomic_save_npy(output_path, embedding_matrix)
    # Metadata is committed last so ``status=ready`` never points to a missing
    # vector file during a normal build.
    _atomic_save_json(output_metadata_path, metadata)
    LOGGER.info(
        "完成：%s（shape=%s），metadata=%s",
        output_path,
        embedding_matrix.shape,
        output_metadata_path,
    )
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="以本機 Ollama 建立法規 embedding 索引。"
    )
    parser.add_argument(
        "--provisions",
        type=Path,
        help="法規 JSON 路徑；預設使用 config.Settings。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="輸出 .npy 路徑；預設使用 config.Settings。",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="輸出 metadata JSON 路徑；預設使用 config.Settings。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="每次送往 Ollama 的文字筆數（預設 32）。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        build_embeddings(
            provisions_path=args.provisions,
            embeddings_path=args.output,
            metadata_path=args.metadata,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        LOGGER.error("建立 embeddings 失敗：%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
