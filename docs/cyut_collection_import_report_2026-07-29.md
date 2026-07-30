# 朝陽法規 MVP 正式資料匯入報告

匯入日期：2026-07-29

## 匯入來源

原始交付目錄：

`C:\Users\wyujen\Documents\Codex\2026-07-29\chrome-plugin-chrome-openai-bundled-file\outputs`

| 檔案 | 大小 | SHA-256 |
| --- | ---: | --- |
| `source_law.txt` | 758,633 bytes | `4C19BCB99E7D65A87606E6F0886C5EACC1DD9D91948E01E001D74CB0083646DB` |
| `legal_provisions.json` | 2,565,447 bytes | `523BB8FE135835DD3F0DA65E49AC0F9FC367E6C6C295C39CC23CE2AFC875834A` |
| `collection_warnings.json` | 4,968 bytes | `02F1BBCAD6935E364F1DD394E59E18B1E5CBDDDD7EDA991D2BAE850B5F4692DD` |

原有範例 `source_law.txt` 與 `legal_provisions.json` 已先備份至
`data/backups/2026-07-29-pre-official-sync/`，再以交付檔替換。

## 獨立驗證

- 223 部現行法規、2,234 筆條文。
- `provision_id` 為 9 至 2242，全部唯一。
- `sort_order` 為 1 至 2234，依輸入順序連續。
- 穩定 key 無重複，全部為 `is_active=true`。
- 全部資料通過專案 `LegalProvision` schema 與同步前驗證。
- TXT 的 223 個文件區塊與 JSON 的法規名稱、來源網址一一相符。
- JSON 中每筆正文都可在對應 TXT 原文找到。
- 6 份內容頁明確標示廢止的法規及 1 份同名同文重複來源均未混入現行資料。

## 警告處理

原始 `collection_warnings.json` 保留不變，共 15 筆：

- `inactive_detail_page`：6 筆。
- `duplicate_document`：1 筆。
- `removed_old_provision`：8 筆。

8 筆移除事件全部是舊 MVP 的「測試法規」ID 1 至 8。完整快照同步只將
這些資料標記為 inactive，未永久刪除。

本次另發現警告報告未包含決策文件要求的 `severity`、`context`，且沒有
列出不同 key 但正文相同的非阻擋性事件。這不影響 TXT、JSON 正文或 QA
資料庫正確性，但下次正式化收集器時應補齊警告輸出契約。

## PostgreSQL 與向量同步

- 正式資料：223 部 active 法規、2,234 筆 active 條文。
- Embedding：2,234 筆，模型 `embeddinggemma`，維度 768。
- 舊測試條文：8 筆 inactive。
- HNSW cosine index 已由實際查詢計畫確認使用。
- 第一次同步後發現 672 筆向量輸入會因換行空白差異重複拼接；同步器已先
  正規化 canonical text，再增量重建這 672 筆向量。
- 最後一次冪等同步結果：新建／更新 0 筆、沿用 2,234 筆、停用 0 筆。

## 驗證結果

- 單元測試：87 passed，1 skipped。
- 真實 PostgreSQL 整合測試：1 passed。
- Streamlit 健康端點：HTTP 200。
- 問句「學生因重病經核准，最多可以延期註冊多久？」的正確 Unicode
  查詢中，Provision 134 與 53 分別排名第 1、2。
- 完整 QA 正確回答「至多二星期」，並引用上述兩筆正式條文。
