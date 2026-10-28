# ADR-008: 口試簡報與學位論文學術語用、名詞一致性暨數學符號嚴謹化規範 (Oral Defense & Thesis Rigor Alignment)

## 狀態
**Status:** 已接受 (Accepted) - 2026-10

## 背景與問題 (Context)
在完成碩士學位論文口試簡報初稿與指導教授面談後，指導教授針對口試簡報主檔 (`Paper/Beamer/defense.tex`) 以及學位論文主文 (`Paper/Main/`) 提出了四項重大學術嚴謹度與表達問題：

1. **非臺灣學術語用與大陸詞彙殘留**：簡報中出現「基線」、「數據」、「網絡」、「優化/優化器」、「歸一化」等非臺灣學術慣用語，在臺大工學院碩士口試場合極易招致評審委員對文獻素養與研究嚴謹度的質疑。
2. **專有名詞與稱謂跨文檔歧異**：核心對照模型 LightGCN 在簡報各頁混用「頂尖基線」、「強基線」、「最強圖卷積基線」等多頭稱謂，缺乏論文主文「在三個選取 Baseline 中表現最佳」的受詞邊界限定，存在誇大宣稱 (Overclaiming) 與定義飄移風險；且 Slide 8 演進表中存在分類縮寫 (CF/FM/GNN/KGAT) 層級錯位與同義反覆。
3. **Baseline 縮寫首見缺乏全稱與自明性**：BPR-MF、NFM、LightGCN、CKG 等模型與指標首次出現時未提供完整英文全銜與中文審定名詞，違反簡報自明性 (Self-contained) 原則。
4. **數學公式理論漏洞與忠實度符號語意不清**：
   - BPR 正則化項 $\lambda \|\mathbf{\Theta}\|_2^2$ 缺少參數集合 $\Theta$ 明確定義，負號排版不良，且文字誤將參數集合稱為單一「參數矩陣」並誤用 2-範數（實際應為各權重矩陣之 Frobenius 範數平方和）。
   - 忠實度公式原式寫作 $p_{uv \setminus S}$ 與 $p_{uv \mid S}$，存在嚴重離散數學與機率論硬傷：下標頂點對 $(u, v)$ 無法與邊集合 $S$ 做差集運算（型別衝突），且垂直線「$\mid$」易被審查委員誤解為貝氏條件機率（偽條件機率陷阱），指導教授直指「意義不清」。

---

## 決策內容 (Decision)

### 1. 臺灣本土學術語用標準化與簡體文獻遷移痕跡清零 (Language Sanitization)
依據中華民國國家教育研究院（雙語詞彙、學術名詞暨辭書資訊網）及國科會計算機科學門規範，全篇清查替換：
- **`網絡` $\rightarrow$ `網路` (Network)**：統一附錄 F 等處之網路架構稱謂。
- **`歸一化` $\rightarrow$ `正規化` (Normalization)**：Softmax 相容性映射統稱「注意力 Softmax 正規化權重」。
- **`優化 / 優化器` $\rightarrow$ `最佳化 / 最佳化器` (Optimization / Optimizer)**：損失函數標題與 Adam 稱謂一律採用工學院審定名詞「最佳化器」。
- **`基線` $\rightarrow$ `基準模型 / 基準方法 (Baseline)`**：全面廢除生硬之「基線」二字，回歸學界標準之「基準模型」。
- **`數據` $\rightarrow$ 分流對齊**：指涉日誌與輸入集合概念時統一為「資料 (Data)」（如資料稀疏性、互動資料）；指涉量化統計量時統一為「數值 (Values)」或「實證結果 (Results)」。
- **全篇字形與動詞規範**：交互特徵 $\rightarrow$ 互動特徵、圖拓撲 $\rightarrow$ 圖拓樸（木部）、構建 $\rightarrow$ 建構、獨立運行 $\rightarrow$ 獨立執行。

### 2. Baseline 分類層級校正與 LightGCN 全域一致稱謂 (Taxonomy & Consistency)
- **消除誇大宣稱風險**：廢除「頂尖基線」，對齊論文第四章 Line 140 之嚴謹邊界，簡報全篇統一規範為：
  - RQ1 頁面：`相較於基準模型 (LightGCN) 帶來何種排序表現影響？`
  - 核心發現與結論：`相較於表現最佳之基準模型 LightGCN（在評估之 Baseline 中）`
- **修正 Slide 8 演進架構表之層級錯位**：
  - 矩陣分解更正為 `\textbf{矩陣分解 (MF)}`（原錯置為 CF）
  - 特徵互動更正為 `\textbf{因子分解機 (FM)}`（原錯置為 FM 與特徵混淆）
  - 圖卷積更正為 `\textbf{圖卷積網路 (GCN)}`（原錯置為泛指之 GNN）
  - 圖注意力修正為 `\textbf{知識圖注意力 (KGAT)}`（消除同義反覆）

### 3. Beamer 16:9 表底星號統釋模式與自明性指標解碼 (Acronym & Anti-overflow Standard)
針對 Beamer 16:9 垂直維度受限特性，避免行內嵌入過長全銜引發 Overfull \vbox：
- **表底星號統釋模式 (Bottom Notes Pattern)**：於 Slide 8 框體底端統一加註：
  $$\text{\textbf{*基準模型全稱}：\textbf{BPR-MF}: Bayesian Personalized Ranking Matrix Factorization；\textbf{NFM}: Neural Factorization Machines；\textbf{LightGCN}: Light Graph Convolutional Network}$$
- **指標首次登場全稱解碼**：於 Slide 12 加註 HR@20（命中率）、NDCG@20（正規化折損累計增益）；於附錄 A1 加註 P@K（精確度）並標明在留一法單正樣本下 $\text{P}@K = \text{HR}@K / K$ 之數學等價關係。
- **版面溢出根除調校**：配合負間距微調與長條圖高度優化 (`0.58\textheight`)，全篇 30 頁達到完全零高度溢出 (**0 Overfull \vbox**)。

### 4. BPR 正則化集合嚴密化與 Fidelity 圖擾動狀態函數標準記號 (Mathematical Rigor)
- **BPR Loss 正則化嚴格定義**：
  將負號顯式外提至求和符號外部，明確定義跨層可學習參數集合 $\Theta \triangleq \{\mathbf{E}, \{\mathbf{W}_r\}_{r \in \mathcal{R}}, \{\mathbf{W}_1^{(l)}, \mathbf{W}_2^{(l)}\}_{l=1}^L\}$，並修正為各參數矩陣之 Frobenius 範數平方和：
  $$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i^+, i^-) \in \mathcal{D}} \ln \sigma\Big(\hat{y}(u, i^+) - \hat{y}(u, i^-)\Big) + \lambda \sum_{\mathbf{W} \in \Theta} \|\mathbf{W}\|_F^2$$
- **忠實度公式（Fidelity）標準化升級**：
  徹底揚棄 $p_{uv \setminus S}$ 與 $p_{uv \mid S}$。對齊 IEEE TPAMI 2022、NeurIPS 2019 與 EMNLP 2024 等圖神經網路解釋性主流論文記號，確立改採**「圖擾動狀態預測差值函數」**記號：
  $$F^+ = \hat{p}_{uv}(\mathcal{G}) - \hat{p}_{uv}(\mathcal{G}_{\setminus S}) \quad (\text{必要性 / Prediction Drop})$$
  $$F^- = \hat{p}_{uv}(\mathcal{G}) - \hat{p}_{uv}(\mathcal{G}_S) \quad (\text{充分性 / Prediction Gap})$$
  - $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：完整協同知識圖拓樸；
  - $\mathcal{G}_{\setminus S} \triangleq (\mathcal{V}, \mathcal{E} \setminus S)$：移除解釋路徑邊後的殘缺圖；
  - $\mathcal{G}_S \triangleq (\mathcal{V}, S)$：僅保留解釋路徑邊的極度稀疏子圖；
  - $\hat{p}_{uv}(\cdot)$：模型在特定圖狀態下針對物品 $v$ 之推薦預測機率。

---

## 影響與效益 (Consequences)

### 正面影響 (Positive Impacts)
1. **學術合規性與本土化 100% 達標**：徹底肅清簡體論文直譯痕跡，符合國科會與臺大工學院口試委員的審閱標準。
2. **答辯防禦力大幅強化**：消除「誇大 LightGCN 為 SOTA」與「忠實度偽條件機率」兩大口試易受質疑點，具備完備的理論與文獻支撐說帖。
3. **跨文檔嚴密一致**：口試簡報 `defense.tex` 與學位論文 `chapter03_methodology.tex`、`chapter04_experiments.tex` 在模型分類、損失函數定義與 Fidelity 符號上達成 100% 同步。
4. **版面編譯零瑕疵**：Beamer 30 頁輸出零 Overfull \vbox，論文 83 頁 XeLaTeX 編譯全數通過。

### 延伸追蹤 (Follow-ups)
- 口試答辯時，若委員詢問 Fidelity 的數學定義，直接援引本 ADR 決策四之圖拓撲擾動狀態映射概念與 IEEE TPAMI 2022 論文標準進行說明。
