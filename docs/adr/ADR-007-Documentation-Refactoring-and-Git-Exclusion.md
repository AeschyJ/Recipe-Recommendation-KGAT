# ADR-007: 全專案文檔重構與 Git 巨型檔案排除規範 (Documentation Refactoring & Git Exclusion Standard)

* **狀態**: 接受 (Accepted)
* **日期**: 2026-07-29
* **決策者**: Antigravity Core Team

---

## 背景 (Context)

隨著專案進入實驗量化與論文寫作階段，現有文檔出現以下問題：
1. **文檔與代碼不一致**：部分舊文檔仍參照舊版腳本檔名與舊版硬體套件 (IPEX 依賴 vs PyTorch Native XPU)。
2. **Git 版本控制風險**：專案目錄中累積大量模型備份包 (`model_checkpoints_backup.zip`, 4.03 GB)、LaTeX 論文包 (`paper_latex.zip`, 77.1 MB)、中間編譯檔 (`.aux`, `.log`, `.fls`, `.fdb_latexmk`) 及暫存 diff 紀錄檔。若上傳至 GitHub，將因超過 100MB 限制而引發 Push 失敗並膨脹倉庫體積。
3. **維護與自動化腳本缺口**：新增的 `.bat` 批次檔、`scripts/` 工具與 NotebookLM 彙編腳本未記錄於系統主手冊。

---

## 決策 (Decisions)

1. **全面更新繁體中文文檔體系**：
   - 升級 `README.md`，提供完整專案架構 Mermaid 圖、`uv` 安裝手冊、批次檔與 CLI 指引及評估指標總覽。
   - 更新 `docs/architecture.md`, `docs/api_reference.md`, `docs/development.md`，確保類別名稱、參數規格與腳本用途與 `src/` 最新原始碼 100% 吻合。
2. **強化 `.gitignore` 防錯與規則定義**：
   - 顯式排除 `*.zip` (`model_checkpoints_backup.zip`, `paper_latex.zip`)。
   - 排除 AI 工具內部目錄 (`.agent/`, `.agents/`, `.gemini/`) 與中介目錄 (`tmp/`)。
   - 排除所有 LaTeX 編譯產物與暫存 diff 紀錄 (`*_diffs.txt`)。
3. **規範 Conventional Commits 版本同步**：
   - 使用標準 Conventional Commits 格式（例如 `docs(readme): ...`與 `chore(git): ...`）進行 GitHub 版本推動。

---

## 折衷與影響 (Trade-offs & Consequences)

* **優點**：
  * 防止 4GB 大檔誤上傳導致 GitHub Remote Block。
  * 文檔架構清晰完整，具備可維護性與直接可讀性。
  * 提供 NotebookLM 與 LLM 精簡閱讀資料集之說明。
* **缺點**：
  * 本地大檔備份（如 `.zip`）需獨立備份至外部雲端儲存空間，Git 庫僅追蹤代碼與文檔。
