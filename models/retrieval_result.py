"""混合檢索結果模型。"""

from pydantic import BaseModel, ConfigDict, Field


class RetrievalResult(BaseModel):
    """單筆法條與向量、關鍵字及合併分數。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provision_id: int = Field(gt=0)
    document_name: str = Field(min_length=1)
    article_no: str = Field(min_length=1)
    title: str = ""
    content: str = Field(min_length=1)
    vector_score: float
    keyword_score: float
    final_score: float
