"""集中管理應用程式設定與資料路徑。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_DIMENSION = 768


class Settings(BaseSettings):
    """從環境變數或專案根目錄的 .env 載入設定。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e2b-it-qat"
    ollama_embedding_model: str = "embeddinggemma"

    database_url: str = (
        "postgresql://legal_qa:legal_qa_local@127.0.0.1:5432/legal_qa"
    )
    database_connect_timeout_seconds: int = Field(default=5, ge=1, le=60)
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    retrieval_candidate_k: int = Field(default=50, ge=1, le=1000)
    retrieval_min_score: float = Field(default=0.12, ge=-1.0, le=1.0)
    vector_weight: float = Field(default=0.65, ge=0.0)
    keyword_weight: float = Field(default=0.35, ge=0.0)

    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    llm_max_tokens: int = Field(default=1200, ge=1)
    llm_thinking: bool = False
    request_timeout_seconds: float = Field(default=360.0, gt=0.0)

    law_name: str = "測試法規"
    log_full_prompt: bool = False
    log_level: str = "INFO"
    max_list_items: int = Field(default=6, ge=1, le=20)

    data_dir: Path = BASE_DIR / "data"

    @model_validator(mode="after")
    def validate_retrieval_weights(self) -> "Settings":
        """避免兩個檢索權重同時為零。"""

        if self.vector_weight + self.keyword_weight <= 0:
            raise ValueError("VECTOR_WEIGHT 與 KEYWORD_WEIGHT 不可同時為 0")
        if self.retrieval_candidate_k < self.retrieval_top_k:
            raise ValueError(
                "RETRIEVAL_CANDIDATE_K 不可小於 RETRIEVAL_TOP_K"
            )
        if not self.database_url.strip():
            raise ValueError("DATABASE_URL 不可為空")
        return self

    @property
    def legal_provisions_path(self) -> Path:
        return self.data_dir / "legal_provisions.json"

    @property
    def legal_embeddings_path(self) -> Path:
        return self.data_dir / "legal_embeddings.npy"

    @property
    def embedding_metadata_path(self) -> Path:
        return self.data_dir / "embedding_metadata.json"

    @property
    def qa_logs_path(self) -> Path:
        return self.data_dir / "qa_logs.jsonl"

    @property
    def qa_test_questions_path(self) -> Path:
        return self.data_dir / "qa_test_questions.json"

    @property
    def database_schema_path(self) -> Path:
        return BASE_DIR / "database" / "schema.sql"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """回傳程序內共用、不可變更來源的設定快取。"""

    return Settings()
