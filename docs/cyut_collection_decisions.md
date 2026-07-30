# 朝陽科技大學法規收集程序決策

狀態：已確認，可依本文件開始實作
日期：2026-07-29

本文件回覆《法規收集程序需求確認文件》Q1–Q27。除 Q26 改採 C 且預設關閉外，其餘均採需求文件的建議預設。

## 決策摘要

| 題號 | 決策 | 確認內容 |
| --- | --- | --- |
| Q1 | A | 收集朝陽科技大學全校法規資料庫中的全部現行法規。 |
| Q2 | A | 完全排除廢止法規，不寫入正式 TXT 或 JSON。 |
| Q3 | A | 沿革與修正紀錄不納入正文，只保留正式現行條文。 |
| Q4 | A | 第一版不收附件、修正對照表或 PDF，只收網頁顯示的現行條文。 |
| Q5 | A | 同名且正文相同者去重，只保留較正式或主要分類頁，並記錄其他來源網址供稽核。 |
| Q6 | A | 同名但正文不同時停止該次正式發布，產生高優先級錯誤，等待人工選擇。 |
| Q7 | A | 沒有「第 X 條」時，以來源中的最高層正式編號作為 `article_no`，例如「一、」。 |
| Q8 | A | 單一條文超過 2,000 個 Unicode 字元時，才依正式項次拆分。 |
| Q9 | A | 條文的自然段依原始順序視為項，填入 `paragraph_no = 1, 2, 3...`；「一、二、三」等款次不得誤判為項。 |
| Q10 | A | 款、目不拆成獨立 Chunk，保留於所屬條或項的 `content`，`subparagraph_no` 維持 `null`。 |
| Q11 | A | 現行文本中的「（刪除）」或「（停止適用）」條號仍保留。 |
| Q12 | A | 只有來源明確提供條文標題時才填入 `title`；否則使用空字串，不推測或摘要。 |
| Q13 | A | 所有法規合併為單一 `source_law.txt`，保留法規名稱、來源網址及清楚分隔線。 |
| Q14 | A | 合併連續空行，但保留章、節、條及段落邊界。 |
| Q15 | A | 穩定 key 固定為 `document_name + article_no + paragraph_no + subparagraph_no`；`null` 保持空值，不轉成 `0` 或空字串。 |
| Q16 | A | 舊 JSON 若有重複 key，停止發布並報錯，不猜測要沿用的 `provision_id`。 |
| Q17 | A | 法規改名視為新文件，產生新的 `provision_id`。 |
| Q18 | A | 條文由未拆分改為依項拆分時，第一項沿用原條文 ID，其餘項使用新 ID。 |
| Q19 | A | 官方來源已移除的舊條文不寫入新 JSON；warning 列出舊 key 與 ID。資料庫不得未經確認就永久刪除該資料。 |
| Q20 | A | 依官方索引順序及各法規內章、節、條、項順序，從 1 全域連續編排 `sort_order`。 |
| Q21 | A | key 相同即視為重複；正文相同但 key 不同只列 warning，不自動刪除。 |
| Q22 | A | warning 同時顯示於終端並寫入 `collection_warnings.json`。 |
| Q23 | A | 遇到無法解析內容時仍產生可供檢查的 staging 輸出及報告，但命令回傳非零；不得自動同步資料庫。 |
| Q24 | A | 網路錯誤或部分頁面下載失敗時，不覆寫既有 staging 正式檔，產生錯誤報告並以非零狀態結束。 |
| Q25 | A | 新增獨立 CLI `scripts/collect_cyut_laws.py`，將網站收集與既有純文字匯入器分離。 |
| Q26 | C | 由命令列選項控制資料庫同步與 Embedding；預設關閉，不在一般收集完成後自動執行。 |
| Q27 | A | staging 輸出沿用 `data/source_law.txt` 與 `data/legal_provisions.json`。 |

## 解析與識別碼補充規則

- 只複製官方現行文字，不摘要、不改寫，也不補寫來源沒有提供的標題。
- `content` 必須保留條、項內的款、目文字及原始順序。
- `search_text` 由法規名稱、章、節、條號、標題及正文組合產生，不寫入來源沒有的法律內容。
- 去重或沿用 ID 前，先完成 Unicode、換行及空白正規化；正規化不得改變實質文字。
- 新 key 使用尚未使用過的新 `provision_id`，不得回收已移除條文的舊 ID。
- Q18 的第一項沿用規則只適用於能唯一確認為同一條文的情況；無法唯一確認時視為阻擋性錯誤。

## 錯誤、阻擋性警告與非阻擋性警告

所有事件均寫入 `collection_warnings.json`，至少包含：

- `severity`：`fatal`、`blocking_warning` 或 `warning`
- `code`
- `document_name`
- `source_url`
- `message`
- 可定位來源內容的 `context`

### `fatal`

下列情況必須回傳非零狀態、不覆寫既有 staging 正式檔，也不得同步 PostgreSQL：

- 官方索引或任一預期頁面下載失敗，重試後仍無法取得完整資料。
- 同名法規正文不同且無法自動判定正式版本。
- 舊 JSON 存在重複穩定 key、重複 `provision_id` 或無法安全沿用 ID。
- 輸出不符合 schema、必要欄位缺漏、排序或 ID 完整性驗證失敗。
- 暫存檔寫入、完整驗證或原子替換失敗。

### `blocking_warning`

無法解析部分內容時，仍可產生本次候選 staging TXT、JSON 及報告供人工檢查，但：

- 命令回傳非零狀態。
- 不自動執行 `scripts/sync_database.py`。
- 人工確認或修正後才可發布至資料庫。

官方來源中原有但本次消失的舊 key 也至少列為 `blocking_warning`；在人工確認前，資料庫內既有資料不得永久刪除或停用。

### `warning`

不造成資料缺頁、識別碼歧義或 schema 錯誤的事件屬非阻擋性 warning，例如：

- 同名、同正文的多個網址已依 Q5 去重。
- 不同 key 恰有相同正文，依 Q21 保留全部資料。
- 網頁多餘空行或排版空白已正規化。

這類 warning 可回傳成功狀態並發布完整 staging 輸出；若使用者明確指定同步選項，也可同步資料庫。

## PostgreSQL + pgvector 整合

### 資料角色

- `data/source_law.txt`：人工稽核、來源快照及重新解析依據。
- `data/legal_provisions.json`：收集器與資料庫同步程序之間的 staging／交換格式。
- `data/collection_warnings.json`：本次收集的錯誤與警告報告。
- PostgreSQL + pgvector：QA 執行時的正式查詢資料層；應用程式不再全量載入 JSON 與 `.npy`。

TXT 與 JSON 仍須完整產生並驗證，但不作為大量法規下的線上檢索資料庫。

### 同步程序

新增 `scripts/sync_database.py`，職責如下：

1. 完整驗證 `legal_provisions.json`、穩定 key、ID、排序及本次警告狀態。
2. 以穩定 key 對 PostgreSQL 執行 transaction 內的增量 upsert。
3. 為每筆資料保存內容 fingerprint、Embedding 模型名稱及向量維度。
4. 只有下列資料呼叫 Embedding：
   - 新增條文；
   - `search_text` 或其他會影響向量的內容已變更；
   - 向量缺漏；
   - Embedding 模型或維度不符。
5. fingerprint、模型及維度均相同者沿用既有向量，不重新計算。
6. 條文與 `vector(768)` 必須在同一 transaction 成功寫入；任一步驟失敗即 rollback。
7. 舊 key 本次未出現時先報告，經人工確認後才標記非現行；不自動永久刪除。

建議操作方式：

```powershell
# 只收集及產生 staging 檔；預設不連資料庫、不產生 Embedding
.\.venv\Scripts\python.exe scripts\collect_cyut_laws.py

# 人工或自動化檢查通過後，明確執行增量同步
.\.venv\Scripts\python.exe scripts\sync_database.py

# 收集 CLI 的選用捷徑；僅在沒有 fatal/blocking_warning 時允許
.\.venv\Scripts\python.exe scripts\collect_cyut_laws.py --sync-database
```

`--sync-database` 預設為 `false`。若未提供此選項，收集器只能輸出 staging 檔及後續指令提示。完整重建向量應另設明確的 `scripts/sync_database.py --rebuild-all`，不得由一般收集流程隱式觸發。

### 模型與 QA 流程

模型設定維持不變：

- 回答模型：`gemma4:e2b-it-qat`
- Embedding 模型：`embeddinggemma`
- 向量維度：768
- PostgreSQL／pgvector 先取得候選條文，沿用既有排序或 rerank 邏輯，最後仍只將 Top 6 條文交給回答模型。

本次資料層改造不得改變既有提示詞、結構化回答驗證、引用驗證、免責聲明或 QA 模型行為。
