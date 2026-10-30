# 論文最後一頁末行「白框」符號之技術調查與學術意涵解析

- **調查對象**：`Paper/Main/main.pdf` 最後一頁（早期版本第 97 頁 / 最新完整目錄編譯版第 99 頁，正文編碼第 81 頁）末行右側白框圖形
- **對應原始碼**：[`Paper/Main/back/appendix04.tex`](Paper/Main/back/appendix04.tex#L99-L124)（`\begin{proof} ... \end{proof}`）
- **設定定義**：[`Paper/Main/ntusetup.tex`](Paper/Main/ntusetup.tex#L68-L75)（`amsthm`, `\newtheorem`, `\renewcommand{\proofname}{證明}`, `\qedsymbol`）

---

## 1. 調查結論概述

**核心結論**：該白框是**完結符號**！它是國際數學、資訊工程與電腦科學學術論文中標準且規範的**「證畢符號」（Q.E.D. Symbol / Halmos Symbol / 墓碑符號 Tombstone）**。

它**絕非字型缺失（Tofu / Missing Glyph）、不是編碼亂碼，也不是排版瑕疵**。

其由 LaTeX 的美國數學學會定理巨集套件 `amsthm` 在 `\begin{proof}` 環境結束（`\end{proof}`）時，自動於證明末行最右側邊界（Flush Right）繪製而成。

---

## 2. 深入技術驗證紀錄（PDF 底層向量與原始碼定位）

### 2.1 頁碼動態變化說明（97 頁 vs 99 頁）
- 在初期未完全展開目錄（TOC）或附錄 D 剛引入的第一階段編譯中，`main.pdf` 總頁數為 97 頁（正文頁碼 79）。
- 隨著目錄重新寫入 `main.aux` 與 `main.toc` 並完成二次編譯，前置目錄由 2 頁擴展至 4 頁，使得論文總頁數推進至 **99 頁**（正文頁碼 81）。
- 無論是第 97 頁或第 99 頁，該白框皆精確出現在**附錄 D 形式化數學證明的最後一行**。

### 2.2 PDF 向量結構分析（PyMuPDF 幾何提取）
透過解析 `main.pdf` 最後一頁底層 PDF Drawing 向量指令：
* **文字區塊**（Y 座標 509.8 ～ 522.3）：
  `bbox: (85.0, 509.8, 317.2, 522.3), text: '出之路徑權重在IEEE 754 浮點精度下恆等。\n'`
* **末端白框向量圖元**（Y 座標 511.3 ～ 519.4，X 座標 501.8 ～ 509.4）：
  * 左垂直線：`(501.80, 519.41) -> (501.80, 511.34)`
  * 頂水平線：`(502.00, 511.54) -> (509.17, 511.54)`
  * 底水平線：`(502.00, 519.21) -> (509.17, 519.21)`
  * 右垂直線：`(509.37, 519.41) -> (509.37, 511.34)`
  * 線寬（stroke width）：`0.3985 pt`（標準 TeX `\hrule` / `\vrule` 規線寬度）
  * 填色（fill）：`None`（空心矩形 / 白框）
  * 對齊邊界：X = 509.4（恰好緊貼版面最右側邊界 Margin 510.2 pt）

**技術事實確認**：
若為字型缺字，PDF 中會存在一筆使用特定字型指向 `.notdef` 字符的 Text Span；然而在此處，PyMuPDF 完全偵測不到字元文字，而是由 **4 條精確的向量規線組合而成的封閉幾何矩形**。此即 TeX 底層巨集 `\openbox`（定義為 `\vrule` 與 `\hrule` 包裹之方塊）的典型繪圖特徵。

### 2.3 原始碼映射分析
在 [`Paper/Main/back/appendix04.tex`](Paper/Main/back/appendix04.tex#L91-L124) 中：
```latex
91: \begin{proposition}[路徑集合與分數等價性]
92: 令 \mathcal{P}_{\text{NX}}(u, v, 3) 為全圖 \mathcal{G} 上由 NetworkX ...
...
97: \end{proposition}
98: 
99: \begin{proof}
100: 本證明採用雙向子集包含法分別證明 ...
...
123: 由於兩者演算法均讀取相同 KGAT 檢查點之靜態注意力陣列 \mathbf{A}，... 故兩者計算得出之路徑權重在 IEEE 754 浮點精度下恆等。
124: \end{proof}
```
全本論文僅有附錄 D 包含形式化數學定理證明（`proposition` 與 `proof`）。當編譯器讀取至第 124 行 `\end{proof}` 時，會自動調用 `\qed` 巨集，在段落尾端輸出 `\qedsymbol`。

---

## 3. 學術意涵與歷史背景

1. **Halmos 符號（墓碑符號 Tombstone）**：
   * 著名匈牙利裔美國數學家保羅·哈爾莫斯（Paul Halmos）於 1950 年在其專著《測度論》（*Measure Theory*）中首次提出以方塊符號標記數學證明完畢，以取代傳統繁複的拉丁文縮寫 **Q.E.D.**（*quod erat demonstrandum*，直譯為「這就是所要證明的」，中文常作「證畢」）。
   * 由於外觀形似微型墓碑，在數學與理論計算機科學界被普遍暱稱為「墓碑符號」（Tombstone）。
2. **AMS-LaTeX 與現代頂刊規範**：
   * 美國數學學會（AMS, American Mathematical Society）在制定 `amsthm` 巨集套件時，將預設證畢標記訂為空心方框 $\square$（`\openbox`），並在證明環境結尾以 `\hfill` 自動靠右對齊。
   * 包含 IEEE、ACM、SIAM、Springer、Elsevier 等頂級電腦科學期刊與國際會議論文，均廣泛採用此標準符號。

---

## 4. 為何非數學背景讀者容易感到疑惑？

在中文排版與一般軟體文字介面中，空心方框 $\square$ 經常被用作字型缺失或亂碼時的替代符號（又稱「豆腐塊」Tofu block 或 `.notdef`）。
當工學院評審或口試委員若主要專注於推薦系統工程架構，而較少審閱理論數學證明時，初次在中文段落結尾看見孤立的空心方框，容易直覺產生「此處是否漏字或編碼錯誤」的疑惑。

---

## 5. 處置建議與自訂調整方案

### 5.1 建議做法：保留（強烈推薦）
在工學院碩士論文中，附錄 D 為「CSR 雙向局部交集演算法虛擬碼與等價性理論證明」，既然前文採用了嚴謹的 `命題`（Proposition）與 `證明`（Proof）環境，結尾保留標準的 Halmos 證畢白框 $\square$ 是**非常專業且符合學術規範的正式排版**，能清楚向口試委員表達「形式化數學推導在此完整閉環」。

### 5.2 若需消除評審疑惑的備選自訂方案
若作者希望徹底杜絕口試委員產生「缺字方塊」之誤解，已在 [`Paper/Main/ntusetup.tex`](Paper/Main/ntusetup.tex#L68-L75) 預置了配置開關，只需取消註解即可立即切換：

1. **切換為實心黑框（$\blacksquare$）**：
   ```latex
   \renewcommand{\qedsymbol}{$\blacksquare$}
   ```
   *優點*：保留 Halmos 墓碑意象，但實心方塊明顯不同於空心缺字豆腐塊。
2. **切換為傳統英文字樣（Q.E.D.）**：
   ```latex
   \renewcommand{\qedsymbol}{\textbf{Q.E.D.}}
   ```
   *優點*：國際通用拉丁縮寫，明確指示證明完結。
3. **切換為中文顯式標記（（證畢））**：
   ```latex
   \renewcommand{\qedsymbol}{（證畢）}
   ```
   *優點*：最符合中文學術論文習慣，任何評審委員皆能一目了然。
4. **完全隱藏（不顯示任何符號）**：
   ```latex
   \renewcommand{\qedsymbol}{}
   ```
