"""本地法規條文資料模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LegalProvision(BaseModel):
    """一筆可檢索的法規條文或依項切分的條文片段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provision_id: int = Field(gt=0)
    document_name: str = Field(min_length=1)
    chapter_name: str = ""
    section_name: str = ""
    article_no: str = Field(min_length=1)
    paragraph_no: int | None = Field(default=None, ge=1)
    subparagraph_no: int | None = Field(default=None, ge=1)
    title: str = ""
    content: str = Field(min_length=1)
    search_text: str = ""
    sort_order: int = Field(ge=0)
    source_url: str = ""
    is_active: bool = True

    @model_validator(mode="after")
    def populate_search_text(self) -> "LegalProvision":
        """缺少 search_text 時，以可辨識的本地欄位安全補齊。"""

        if not self.search_text:
            self.search_text = " ".join(
                part
                for part in (
                    self.document_name,
                    self.chapter_name,
                    self.section_name,
                    self.article_no,
                    self.title,
                    self.content,
                )
                if part
            )
        return self
