# 本機多法規 AI QA 測試系統

這是一個可直接執行的繁體中文法規 QA MVP。法規正文與 768 維
Embedding 正式存放在 PostgreSQL＋pgvector；使用者提出問題後，系統只
取出向量及中文關鍵字候選，混合排序成 Top 6，再交給本機
`gemma4:e2b-it-qat` 回答。

模型流程維持完全本機：

- 回答模型：`gemma4:e2b-it-qat`
- Embedding：`embeddinggemma`
- 法規資料庫：PostgreSQL 17＋pgvector 0.8.5
- UI：Streamlit
- 結構與引用驗證：Pydantic

`data/source_law.txt` 與 `data/legal_provisions.json` 仍是收集、交換及
人工稽核格式，但 QA 執行時不再全量載入 JSON、`.npy` 或所有法條。

本工具只供內部概念驗證與初步解析，不構成正式法律意見。

## 系統需求

- Windows 10/11
- Docker Desktop（WSL2 backend）
- NVIDIA GPU 與可供 WSL2 使用的驅動；目前預設 Compose 會保留 1 張 GPU
- 建議 32 GB RAM；資料庫容器已限制為 768 MB，但 Docker Desktop 本身
  仍會使用額外記憶體

已在本機 RTX 4060 Ti 8 GB、Docker Compose 2.35、PostgreSQL 17、
pgvector 0.8.5、Ollama 0.32.5、`gemma4:e2b-it-qat` 與
`embeddinggemma` 實際驗證。

若只使用 Compose，不必另外安裝 Python 或 Windows 版 Ollama。只有開發、
執行測試或使用原生模式時才需要 Python 3.11／3.12。

## 最快啟動：完整 Docker Compose

在 PowerShell 進入專案根目錄：

```powershell
Copy-Item .env.example .env
docker compose config
docker compose up -d --build
```

第一次啟動會依序：

1. 啟動 PostgreSQL＋pgvector。
2. 啟動支援 NVIDIA GPU 的 Ollama 0.32.5。
3. 將 `gemma4:e2b-it-qat` 與 `embeddinggemma` 下載至持久化
   `ollama_models` volume；目前合計約 4.9 GB。
4. 驗證模型 ID，避免資料庫沿用由不同 Embedding 模型產生的舊向量。
5. 暖機 Embedding 與回答模型；Gemma 4 第一次載入可能需要數分鐘。
6. 冪等建立 schema，並把完整法規快照增量同步至資料庫。
7. 前述工作成功後才啟動 Streamlit。

查看包含一次性初始化服務的狀態：

```powershell
docker compose ps --all
docker compose logs -f ollama-init warmup bootstrap app
```

`ollama-init`、`warmup` 與 `bootstrap` 成功執行後顯示 `Exited (0)`
是正常現象。
當 `app` 顯示 `healthy` 後開啟：

```text
http://127.0.0.1:8501/
```

預設只在本機映射下列連接埠：

- Streamlit：`127.0.0.1:8501`
- PostgreSQL：`127.0.0.1:5432`
- 容器版 Ollama：`127.0.0.1:11435`

容器版 Ollama 使用 11435，刻意避開 Windows 原生 Ollama 的 11434。
正式使用 Compose 版確認無誤後，建議退出 Windows 原生 Ollama，避免兩套
服務同時搶用 GPU／記憶體。

日常重新啟動只需要：

```powershell
docker compose up -d
```

停止全部服務但保留資料與模型：

```powershell
docker compose down
```

PostgreSQL 存在 `legal-qa_legal_qa_pgdata`，模型存在
`legal-qa_ollama_models`。除非確定要永久刪除法規資料庫與重新下載全部
模型，否則不要執行 `docker compose down -v`。

## 法規初始化與後續同步

同步器會：

1. 驗證 UTF-8 JSON、全域連續 `sort_order`、唯一 `provision_id` 與穩定
   key。
2. 防止同一 ID 被改配到不同條文。
3. 自行把正式正文納入 Embedding input，不盲目信任可能過期的
   `search_text`。
4. 只替新增、內容變更、缺少向量或模型不同的條文產生 Embedding。
5. 在向量全部成功後，以單一 PostgreSQL transaction 寫入正式資料。
6. 相同資料再次同步時直接沿用原有向量。

完整 Compose 啟動時，`bootstrap` 會自動執行完整快照同步。更新
`data/legal_provisions.json` 後，可手動重新執行：

```powershell
docker compose run --rm bootstrap
```

只有確認輸入代表「完整現行法規快照」時才能加上
`--full-snapshot`。此選項會將資料庫中未出現在輸入的舊條文標記為
非現行。若要測試單一或部分法規，改用 App 映像執行且不要加
`--full-snapshot`：

```powershell
docker compose run --rm bootstrap `
  python scripts/sync_database.py --input data/legal_provisions.json
```

## 原生開發模式

需要修改程式或執行測試時，可只讓 Compose 啟動資料庫，App 與 Ollama
維持 Windows 原生：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d db
ollama pull gemma4:e2b-it-qat
ollama pull embeddinggemma
.\.venv\Scripts\python.exe scripts\sync_database.py --full-snapshot
.\.venv\Scripts\python.exe -m streamlit run app.py
```

原生模式使用 `.env` 中的 `127.0.0.1:5432` 與 `localhost:11434`；
Compose 內部則會自動覆寫成 `db:5432` 與 `ollama:11434`。

## 法規收集輸出

收集程序應產生：

- `data/source_law.txt`：官方原文、法規名稱、來源網址與文件分隔。
- `data/legal_provisions.json`：符合 `LegalProvision` schema 的 staging
  資料。
- `data/collection_warnings.json`：未處理內容與警告稽核報告。

朝陽科技大學收集器 Q1–Q27 的完整確認結果見
[`docs/cyut_collection_decisions.md`](docs/cyut_collection_decisions.md)。

重要原則：

- 只收現行正文，不摘要、不改寫。
- 廢止法規、沿革、附件與修正對照表不混入正式 JSON。
- 同名同正文去重；同名不同正文停止發布並交由人工判定。
- 無「第 X 條」時以來源最高層正式編號作為 `article_no`。
- 超過 2,000 字才依正式項次拆分；款、目不拆。
- 法規改名視為新文件。
- JSON 是 staging／交換格式，PostgreSQL 才是 QA runtime 資料源。

## 啟動 QA

完整 Compose 模式：

```powershell
docker compose up -d
docker compose ps --all
```

開啟：

```text
http://127.0.0.1:8501/
```

首頁即使資料庫暫時停止也能顯示；送出問題時會得到不含 DSN、帳號或
密碼的友善錯誤。

### 100 題測試題庫

`data/qa_test_questions.json` 內建 100 題可由現行法規直接支持的測試
問題。問題採第一人稱、實際情境及日常用語，例如「我可以使用宿舍網路
自己架設伺服器或網站嗎？」，不預設使用者已違規，也不要求模型照背條文。
首頁會從目前題庫隨機顯示 3 題，並提供：

- 換一批隨機題。
- 點選隨機題後帶入問題框。
- 隨機抽一題並立即解析。
- 從 100 題清單指定題目後帶入或直接解析。
- 顯示預期答案、關鍵字及 Provision ID。
- 下載內建題庫，或上傳自訂 JSON 題庫。

上傳檔案最多 4 MB、最多 100 題。支援 JSON 陣列或
`{"questions": [...]}`，每題格式如下：

```json
{
  "question_id": "CYUT-QA-001",
  "question": "問題文字",
  "expected_answer": "預期答案",
  "document_name": "法規名稱",
  "article_no": "條號",
  "expected_keywords": ["答案關鍵字"],
  "expected_provision_ids": [123]
}
```

題庫的預期答案及關鍵字只供人工與評測比對，不會送入模型 Prompt。
網頁仍逐題執行 QA，避免一次執行 100 題占滿本機模型與記憶體。

維護題庫時，問題文字應遵守：

- 從使用者尚不知道答案的狀態提問。
- 先問能不能、怎麼辦、何時申請或需要什麼條件。
- 不在問題中預先放入處分、停權、門檻數字或正確結論。
- 只有在身分、事件或程序階段會改變問題時，才保留該情境。
- 不為提高檢索或回答分數而改回條文用語。

可另外量測正確條文是否進入 Top K；預設只報告真實結果，不要求
100% 才通過：

```powershell
python scripts/evaluate_question_bank.py --top-k 6
```

若要設定自動化門檻，可加上 `--minimum-recall 0.9`。也可用
`--live-ids Q001,Q020,Q080` 抽樣呼叫真正的回答模型；此模式的
`SMOKE_PASS` 只表示模型可回答且引用命中，內容是否正確仍應對照
預期答案人工評閱。

## 執行流程

```text
問題正規化
  → embeddinggemma 產生問題向量
  → pgvector HNSW 向量候選
  + PostgreSQL 中文 bigram／完整片語候選
  → Python 沿用既有 Hybrid Score
  → Top 6 PostgreSQL 條文快照
  → 依問題擷取長條文的相關段落，避免超過模型 context
  → 防注入 Prompt + Pydantic JSON Schema
  → Ollama gemma4:e2b-it-qat
  → 結構、citation allowlist 與內部 ID 顯示清理
  → Streamlit 顯示本次檢索快照中的正式全文
  → JSONL QA 紀錄與使用者回饋
```

引用全文直接使用該次 PostgreSQL 檢索快照，不會在回答完成後重讀
JSON，也不會因稍後的資料庫更新而顯示另一版本內容。

## PostgreSQL schema

主要資料表：

| 資料表 | 用途 |
| --- | --- |
| `collection_runs` | 每次成功同步的來源指紋與統計 |
| `legal_documents` | 多部法規的名稱、來源及有效狀態 |
| `legal_provisions` | 現行條文、穩定 ID、內容與 Embedding input hash |
| `provision_embeddings` | 各模型的 `vector(768)` 與 input hash |

目前版本保存「多部法規的最新現況」，不提供歷史版本查詢。舊條文可被
標記為非現行，但修正前的完整正文不會另建版本保存。

資料庫檢索只會使用同時符合以下條件的向量：

- 法規與條文均為現行。
- `embedding_model` 等於目前設定。
- 向量的 `embedding_input_hash` 等於條文目前內容。

因此條文更新後不會誤用舊向量。

## 設定

設定從根目錄 `.env` 載入：

| 變數 | 預設值 | 用途 |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://legal_qa:legal_qa_local@127.0.0.1:5432/legal_qa` | PostgreSQL DSN |
| `POSTGRES_PASSWORD` | `legal_qa_local` | Compose 本機資料庫密碼 |
| `DATABASE_CONNECT_TIMEOUT_SECONDS` | `5` | DB 連線逾時 |
| `EMBEDDING_DIMENSION` | `768` | 固定配合 embeddinggemma 與 schema |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服務 |
| `OLLAMA_CHAT_MODEL` | `gemma4:e2b-it-qat` | Gemma 4 回答模型 |
| `OLLAMA_EMBEDDING_MODEL` | `embeddinggemma` | 文件與問題 Embedding |
| `OLLAMA_CHAT_MODEL_ID` | `07ea59a47401` | Compose 模型 ID 安全檢查 |
| `OLLAMA_EMBEDDING_MODEL_ID` | `85462619ee72` | 防止誤用不同模型產生的既有向量 |
| `OLLAMA_IMAGE` | `ollama/ollama:0.32.5` | 固定的 Ollama 容器版本 |
| `OLLAMA_DOCKER_PORT` | `11435` | 容器版 Ollama 的本機除錯連接埠 |
| `OLLAMA_KEEP_ALIVE` | `30m` | 模型閒置後保留時間，避免重複冷啟動 |
| `OLLAMA_MAX_LOADED_MODELS` | `2` | 同時保留 Embedding 與回答模型 |
| `OLLAMA_NUM_PARALLEL` | `1` | 同時推論數，避免 MVP 併發耗盡記憶體 |
| `APP_PORT` | `8501` | Streamlit 本機連接埠 |
| `RETRIEVAL_CANDIDATE_K` | `50` | 向量及關鍵字各自候選上限 |
| `RETRIEVAL_TOP_K` | `6` | 最後交給回答模型的條文上限 |
| `RETRIEVAL_MIN_SCORE` | `0.12` | 最低混合分數 |
| `VECTOR_WEIGHT` | `0.65` | 向量分數權重 |
| `KEYWORD_WEIGHT` | `0.35` | 關鍵字分數權重 |
| `LLM_TEMPERATURE` | `0.1` | 回答溫度 |
| `LLM_TOP_P` | `0.9` | nucleus sampling |
| `LLM_MAX_TOKENS` | `1200` | Ollama `num_predict` |
| `LLM_THINKING` | `false` | 關閉 Gemma 4 額外思考輸出，穩定 JSON 與延遲 |
| `REQUEST_TIMEOUT_SECONDS` | `360` | Ollama 逾時，涵蓋 Gemma 4 首次載入 |
| `LOG_FULL_PROMPT` | `false` | 是否保存完整問題與參考條文 |
| `MAX_LIST_ITEMS` | `6` | 回答列表及引用上限 |

`EMBEDDING_DIMENSION` 在本版必須是 768。`RETRIEVAL_CANDIDATE_K` 不得
小於 `RETRIEVAL_TOP_K`，向量與關鍵字權重不可同時為 0。

目前 RTX 4060 Ti 8 GB 可同時保留 Embedding 與回答模型，因此預設
`OLLAMA_MAX_LOADED_MODELS=2`；仍限制 `OLLAMA_NUM_PARALLEL=1`，且模型
閒置 30 分鐘後會釋放。若部署機器記憶體不足，可把 loaded models 改為
`1`，代價是每題在兩個模型之間切換時會顯著增加等待時間。

正式環境必須更換範例密碼，並確保 `POSTGRES_PASSWORD` 與
原生模式 `DATABASE_URL` 內的密碼一致。Compose 內部 DSN 會直接由
`POSTGRES_PASSWORD` 組成。

## 測試

快速離線測試不需要 PostgreSQL 或 Ollama：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

PostgreSQL 整合測試必須明確指定測試資料庫：

```powershell
$env:TEST_DATABASE_URL = `
  "postgresql://legal_qa:legal_qa_local@127.0.0.1:5432/legal_qa"
.\.venv\Scripts\python.exe -m pytest tests\integration
```

測試涵蓋：

- 收集 JSON 及唯一 ID 驗證。
- 增量同步、向量沿用、批次及錯誤 rollback 前防護。
- PostgreSQL repository 候選參數與 Hybrid rerank。
- pgvector extension、實際向量資料與資料庫檢索。
- Prompt、防注入、結構化輸出、引用 allowlist、QA log 及 Streamlit。
- 100 題題庫 schema、正式 Provision ID、隨機題與網頁載入流程。

## 舊的 NPY 工具

`scripts/build_embeddings.py`、`data/legal_embeddings.npy` 與
`data/embedding_metadata.json` 只保留供舊測試與遷移比對；正式 UI 與
QA 不會讀取它們。新流程一律執行：

```powershell
docker compose run --rm bootstrap
```

系統不會在 PostgreSQL 失敗時自動 fallback 到 JSON／NPY，避免無聲使用
過期法規。

## 常見問題

### 無法連線 PostgreSQL

```powershell
docker compose up -d db
docker compose ps --all
docker compose logs db
docker compose run --rm bootstrap
```

### 資料庫有條文但查不到

重新執行增量同步。同步器會只補齊缺漏或過期的向量：

```powershell
docker compose run --rm bootstrap
```

### 無法連線 Ollama 或找不到模型

```powershell
docker compose up -d ollama
docker compose run --rm ollama-init
docker compose exec ollama ollama list
docker compose logs ollama
```

若 `ollama` 無法啟動並顯示 GPU 錯誤，先確認 Docker Desktop 使用 WSL2
backend，並在 PowerShell 驗證：

```powershell
docker run --rm --gpus all ubuntu nvidia-smi
```

若初始化回報模型 ID 不相符，不要直接略過檢查。這表示 Registry 中同名
tag 已指向不同模型；必須先決定新的 Embedding 模型名稱並重建向量，再
更新 `.env` 的預期 ID。

### 要停止服務但保留資料

```powershell
docker compose down
```

兩個 Docker named volume 仍會保留；下次執行
`docker compose up -d` 即可繼續使用。
