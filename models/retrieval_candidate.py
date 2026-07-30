"""Internal candidate returned by the PostgreSQL vector search."""

from __future__ import annotations

from dataclasses import dataclass

from models.legal_provision import LegalProvision


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    provision: LegalProvision
    vector_score: float


__all__ = ["RetrievalCandidate"]
