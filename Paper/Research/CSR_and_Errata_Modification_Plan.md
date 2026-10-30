# 碩士論文、口試簡報與展示系統：CSR 核心貢獻整合與勘誤清零綜合實施方案

> **文件狀態**：調查與規劃專用基準文檔 (Investigation & Implementation Blueprint)  
> **關聯依據**：`Paper/Research/CSR_Contribution_Thesis_Integration.md`、`Paper/Research/Errata_Alignment_Checklist.md`、`Paper/Research/path_extraction_csr_vs_networkx.md`、`ADR-008`、`ADR-009`  
> **目標範圍**：學位論文 (`Paper/Main/`)、口試簡報 (`Paper/Beamer/`)、展示系統 (`demo/`)

---

## 執行摘要 (Executive Summary)

本研究針對 720 萬邊級之 Food.com 食譜協同知識圖 ($\mathcal{G}_{\text{CKG}}$)，全面檢驗既有論文草稿、口試投影片與展示系統之實作。調查確認：
1. **CSR 核心貢獻之學術定調**：揭露主流 Graph XAI 在大規模圖拓樸上的「玩具資料集盲區」與 NetworkX 單向 DFS 的延遲崩潰（749.95 ms，高壓力 2.20 s，建圖動態耗時 13.01 s / 1.58 GB）；以三部圖拓樸定界 $L=3$ 排他性推論單元，提出 CSR 雙向局部交集向量化演算法，達成純遍歷 **11,205.2 倍** 提速（降至 **0.067 ms**）、端到端 **205,550.2 倍** 提速、記憶體節省 **93%**，並在 18 組嚴格基準測試中達成 **100% 拓樸雙射與浮點數值等價**，且具備度數突破 22,000 之超級節點非對稱免疫屏障。
2. **勘誤全景精確定位與實證解決**：
   - **Errata 1 (Demo 使用者 ID 錯位)**：確認前端頂部顯示 Remapped ID (`17049`)，但後端匯出工具使用 Raw ID (`144146`)，已定位至 `demo/export_demo_cases.py` 與 `src/generate_explanations.py`。
   - **Errata 2 (簡報附錄 D1 權重不符)**：經查 `NotebookLM/notebooklm_explanation.md`，User #2410864 的真實模型注意力貢獻度精確為 **0.7670**，簡報文字 $\alpha = 0.767$ 與論文第 4 章完全一致，左側向量圖 `user_141540.pdf` 標註之 `0.7538` 為舊 Checkpoint 殘留。
   - **Errata 3 (附錄 B1 高頻食材數量算術落差)**：經全面比對，`appendix03.tex` 中 Base (13) 與 Top 0.5% (61) 兩者互斥且聯集為 74；簡報附錄 B1 僅印出 54 個食材，精確遺漏了 `carrots`, `parsley`, `green onions`, `garlic clove`, `onion`, `garlic cloves`, `cinnamon` 共 7 個，補回後即精確吻合 61。
   - **Errata 4 (ADR-009 數據同步)**：更新舊版 3,970 倍為最新嚴格 18 組測試之 11,205.2 倍與 205,550.2 倍。
   - **Errata 5 (Demo 忠實度公式)**：將 `P_orig - P_f+` 與 `P_orig - P_f-` 改為 $\hat{p}(\mathcal{G}) - \hat{p}(\mathcal{G} \setminus S)$ 與 $\hat{p}(\mathcal{G}) - \hat{p}(\mathcal{G}_S)$。
   - **Errata 6~9 (字形與語用語料)**：全面更正「拓撲 $\to$ 拓樸」、「正則化 $\to$ 正規化」、「噪音 $\to$ 雜訊」及簡報日期。

---

## 第一部分：學位論文 (`Paper/Main/`) 整合修改方案

### 1.1 第 1 章：緒論 (`Paper/Main/contents/chapter01_introduction.tex`)

#### 修改要點 1：語意與拓樸標準詞彙清洗
- **Line 11**：將「構建出一個龐大且高度互連的語義網路」替換為「**建構**出一個龐大且高度互連的**語意**網路」。
- **Line 22**：將「食譜知識圖具有與商品知識圖截然不同的圖拓撲特性」替換為「圖**拓樸**特性」。

#### 修改要點 2：擴充第 1.4 節研究貢獻（新增第四大貢獻）
在 Line 50 之後新增第 4 點貢獻：
```latex
  \item \textbf{大規模協同知識圖之即時可解釋路徑萃取與互動展示：}
  本研究揭露現有圖神經網路可解釋性開源實作慣性依賴單向深度優先搜尋，在真實規模協同知識圖（逾 47 萬節點、720 萬條邊）上引發嚴重之記憶體膨脹（動態建圖需 1.58~GB）與秒級延遲瓶頸（平均尋路 749.95~ms，高壓力使用者達 2.20~s）。為此，本研究基於三部圖拓樸約束提出基於壓縮稀疏列（CSR）連續記憶體之雙向局部交集尋路演算法，在數學上嚴格證明其與全局遍歷具備 100\% 拓樸雙射與數值等價性，將平均尋路延遲降至 0.067~ms（純遍歷提速逾 1.1 萬倍，含動態建圖端到端提速逾 20 萬倍），記憶體大幅節省 93\%，並成功對超級節點（度數逾 2.2 萬）形成非對稱免疫屏障，奠定大規模 XAI 即時推論與展示系統之工程基礎。
```

#### 修改要點 3：更新第 1.5 節論文架構說明
同步更新第三章與第四章簡述，標明加入 CSR 雙向路徑萃取演算法理論證明、多軸向消融實驗與即時互動展示系統。

---

### 1.2 第 2 章：文獻探討 (`Paper/Main/contents/chapter02_related_work.tex`)

#### 修改要點 1：圖可解釋性小節末段植入黃金三角文獻
在 Line 91（第 2.4 節末段）植入黃金三角串聯劇本：
```latex
\subsection*{大規模圖可解釋性之計算瓶頸與即時推論挑戰}

儘管基於注意力權重與事後子圖擾動的可解釋方法能有效評估推薦忠實度，然而誠如 \citet{yuan2022explainability} 於圖神經網路可解釋性綜述中所指出，事後解釋子圖與多跳路徑檢索在大規模圖拓樸上面臨嚴峻的指數級鄰域擴張開銷，難以直接滿足線上即時推論 (Online Serving) 之低延遲需求。既有開源實作（如 DIG、CF-GNNExplainer）普遍在小型資料集（如 Cora、MovieLens-100k）上依賴 NetworkX 等泛用套件進行離線批次評估，忽略了物件導向三層巢狀雜湊字典在百萬邊級圖譜上所引發的記憶體膨脹與快取失效問題。為突破此一工程盲區，本研究結合 \citet{pohl1971bi} 提出之雙向會合搜尋 (Bi-directional Search) 理論，將三跳路徑搜尋複雜度由單向深度優先搜尋之 $\mathcal{O}(d^3)$ 降階為中間相遇之 $\mathcal{O}(d^2)$；同時遵循 \citet{kepner2016graphblas} 於 GraphBLAS 標準所確立之稀疏線性代數圖計算典範，採用壓縮稀疏列 (Compressed Sparse Row, CSR) 連續記憶體陣列封裝全圖拓樸與注意力權重，實現兼具快取行親和性 (Cache-Line Affinity) 與 100\% 拓樸等價性之微秒級路徑萃取架構。
```

#### 修改要點 2：詞彙規範清洗
- Line 15：「進一步優化了」$\to$「進一步**最佳化**了」。
- Line 19、65：「高階拓撲」$\to$「高階**拓樸**」。
- Line 59：「構建協同知識圖」$\to$「**建構**協同知識圖」。
- Line 77、88：「圖的拓撲結構」$\to$「圖的**拓樸**結構」。

---

### 1.3 第 3 章：研究方法 (`Paper/Main/contents/chapter03_methodology.tex`)

在 Line 178（忠實度量化指標小節）後新增第 3.6 節：

```latex
\section{大規模協同知識圖之 CSR 雙向局部交集尋路架構}
\label{sec:csr_bidirectional_method}

為克服傳統圖論套件在百萬邊級協同知識圖上進行可解釋路徑萃取時面臨的記憶體膨脹與搜尋延遲瓶頸，本研究提出基於壓縮稀疏列 (CSR) 連續記憶體陣列之雙向局部交集 (Bi-directional Meet-in-the-Middle) 尋路演算法。

\subsection{結構性三部圖拓樸定界與 $L=3$ 排他性證明}
本研究所建構之協同知識圖 $\mathcal{G} = (\mathcal{V}_{\text{all}}, \mathcal{E}_{\text{all}})$ 本質為嚴格的三部圖 (Tripartite Graph)，頂點集合劃分為使用者 $\mathcal{U}$、食譜 $\mathcal{V}$ 與知識圖實體 $\mathcal{E}_{\text{kg}}$，邊集合滿足 $\mathcal{E}_{\text{all}} \subseteq (\mathcal{U} \times \mathcal{V}) \cup (\mathcal{V} \times \mathcal{E}_{\text{kg}})$，同類節點之間無直連邊。

給定查詢使用者 $u \in \mathcal{U}$ 與推薦候選食譜 $v \in \mathcal{V}$（且 $(u, v) \notin \mathcal{E}$）：
\begin{enumerate}[leftmargin=2em]
  \item \textbf{長度 $L=2$ 恆為空集：}因 $u$ 之鄰域 $\mathcal{N}(u) \subseteq \mathcal{V}$，而 $v$ 之鄰域 $\mathcal{N}(v) \subseteq (\mathcal{U} \cup \mathcal{E}_{\text{kg}})$，兩者交集恆為空：$\mathcal{N}(u) \cap \mathcal{N}(v) = \emptyset$。因此，二跳路徑在數學上恆不存在。
  \item \textbf{長度 $L=3$ 為唯一最小推論單元：}路徑形式為 $(u \to r_1 \to m \to v)$，其中 $r_1 \in \mathcal{N}(u) \subseteq \mathcal{V}$ 為歷史食譜，$m \in \mathcal{N}(r_1) \cap \mathcal{N}(v)$ 為歷史食譜與推薦食譜之共鄰中繼實體。若 $m \in \mathcal{U}$ 則對應協同過濾路徑；若 $m \in \mathcal{E}_{\text{kg}}$ 則對應知識圖語意路徑。此結構精準吻合本研究採納之 $L=3$ KGAT 模型的因果感受野 (Receptive Field)。
\end{enumerate}

\subsection{CSR 稀疏矩陣連續記憶體設計}
遵循 \citet{kepner2016graphblas} 標準，全圖拓樸與邊權重封裝於三個扁平 C 連續陣列：列偏移量 $\mathbf{indptr}$、行索引 $\mathbf{indices}$ 與注意力權重 $\mathbf{data}$。對於任意節點 $i$，其所有鄰居在記憶體中連續儲存於 $\mathbf{indices}[\mathbf{indptr}[i] : \mathbf{indptr}[i+1]]$。現代 CPU 在單次快取行載入（64 位元組）即可一口氣載入 16 個 32-bit 相鄰節點識別碼，硬體預取器能以極高效能運作，徹底杜絕 Python 巢狀字典指標跳轉引發之快取失效。

\subsection{雙向相遇搜尋演算法與閉環剪枝}
演算法如演算法~\ref{alg:csr_search} 所示。前向遍歷歷史食譜 $r_1 \in \mathcal{N}(u) \setminus \{u, v\}$；後向則將目標食譜 $v$ 的鄰居存入經剪枝之候選集合 $\mathcal{S}_v = \text{HashSet}(\mathcal{N}(v)) \setminus \{u, v\}$。中繼判定僅需對 $m \in \mathcal{N}(r_1)$ 執行 $\mathcal{O}(1)$ 集合成員查驗，並施加 $m \ne r_1 \land m \ne v$ 防自環約束。時間複雜度由單向 DFS 之 $\mathcal{O}(d_u \cdot d_r \cdot d_m)$ 降至 $\mathcal{O}(d_u \cdot d_r \cdot 1)$。

\begin{algorithm}[htbp]
\caption{CSR 雙向局部交集路徑萃取演算法}
\label{alg:csr_search}
\KwIn{查詢使用者 $u$, 目標食譜 $v$, CSR 圖結構 $(\mathbf{indptr}, \mathbf{indices})$, 注意力權重 $\mathbf{A}$, 門檻 $K$}
\KwOut{Top-$K$ 排序之解釋路徑集合 $\mathcal{P}^*$}
$\mathcal{P} \gets \emptyset$\;
$\mathcal{N}(u) \gets \mathbf{indices}[\mathbf{indptr}[u] : \mathbf{indptr}[u+1]]$\;
$\mathcal{N}(v) \gets \mathbf{indices}[\mathbf{indptr}[v] : \mathbf{indptr}[v+1]]$\;
$\mathcal{S}_v \gets \text{HashSet}(\mathcal{N}(v)) \setminus \{u, v\}$ \tcp*{後向候選剪枝集合}
\If{$v \in \mathcal{N}(u)$}{
  $w_{uv} \gets \text{GetWeight}(u, v, \mathbf{A})$\;
  \If{$w_{uv} > 10^{-6}$}{$\mathcal{P} \gets \mathcal{P} \cup \{ ([u, v], w_{uv}) \}$}\;
}
\ForEach{$r_1 \in \mathcal{N}(u) \setminus \{u, v\}$}{
  $w_{u, r_1} \gets \text{GetWeight}(u, r_1, \mathbf{A})$\;
  \If{$w_{u, r_1} \le 10^{-6}$}{\textbf{continue}\;}
  $\mathcal{N}(r_1) \gets \mathbf{indices}[\mathbf{indptr}[r_1] : \mathbf{indptr}[r_1+1]]$\;
  \ForEach{$m \in \mathcal{N}(r_1)$}{
    \If{$m \in \mathcal{S}_v \land m \ne r_1$}{
      $w_{r_1, m} \gets \text{GetWeight}(r_1, m, \mathbf{A})$\;
      $w_{m, v} \gets \text{GetWeight}(m, v, \mathbf{A})$\;
      \If{$w_{r_1, m} > 10^{-6} \land w_{m, v} > 10^{-6}$}{
        $\text{Score} \gets w_{u, r_1} \times w_{r_1, m} \times w_{m, v}$\;
        $\mathcal{P} \gets \mathcal{P} \cup \{ ([u, r_1, m, v], \text{Score}) \}$\;
      }
    }
  }
}
$\mathcal{P}^* \gets \text{SortByScoreDescending}(\mathcal{P})[1:K]$\;
\Return{$\mathcal{P}^*$}
\end{algorithm}

\subsection{拓樸雙射與數值等價性定理證明}
\textbf{定理：}令 $\mathcal{P}_{\text{NX}}(u, v, 3)$ 為 NetworkX 單向深度優先搜尋產出之邊長度 $\le 3$ 簡單路徑集合，$\mathcal{P}_{\text{CSR}}(u, v)$ 為演算法~\ref{alg:csr_search} 產出之路徑集合，則兩者嚴格雙射相等：$\mathcal{P}_{\text{CSR}} \equiv \mathcal{P}_{\text{NX}}$，且對應路徑分數在浮點精度下數值完全恆等。

\textbf{證明要點：}
(1) 任取 $p = [u, r_1, m, v] \in \mathcal{P}_{\text{CSR}}$，因 $r_1 \notin \{u, v\}$ 且 $m \in \mathcal{S}_v \implies m \notin \{u, r_1, v\}$，所有頂點互異，且邊均存在，為合法簡單路徑，必被單向 DFS 探訪，故 $\mathcal{P}_{\text{CSR}} \subseteq \mathcal{P}_{\text{NX}}$；
(2) 反之，任取 $p \in \mathcal{P}_{\text{NX}}$ 且邊數 $\le 3$，由簡單路徑頂點互異性與邊相連性，其必滿足 $r_1 \in \mathcal{N}(u) \setminus \{u, v\}$ 及 $m \in \mathcal{N}(r_1) \cap \mathcal{N}(v) \setminus \{u, r_1, v\}$，故必被雙向相遇擷取，$\mathcal{P}_{\text{NX}} \subseteq \mathcal{P}_{\text{CSR}}$；
(3) 路徑分數均為相同邊權重之連乘積，數值恆等。證畢。
```

---

### 1.4 第 4 章：實驗結果 (`Paper/Main/contents/chapter04_experiments.tex`)

在 Line 424（案例分析之後）新增第 4.5 節：

```latex
\section{大規模 CKG 即時路徑萃取與推論效能實證}
\label{sec:path_extraction_benchmark}

為驗證 §\ref{sec:csr_bidirectional_method} 所提出之 CSR 雙向局部交集尋路架構在真實工業級規模圖譜上的推論效率，本節在包含 473,577 個節點與 7,200,165 條有向邊之 Food.com 協同知識圖上進行系統性基準評測。

\subsection{圖資料結構建構開銷}
表~\ref{tab:graph_construction_benchmark} 呈現在相同環境下不同圖資料結構之建構效能。傳統 XAI 開源實作（NetworkX 動態帶權模式）在每次使用者查詢時依據注意力篩選有效邊動態建圖，耗時高達 13.01 秒且暴增 1.58 GB 記憶體；相較之下，CSR 稀疏矩陣建構僅需 190.82 ms，核心陣列記憶體僅 56.74 MB（節省達 93\%），每秒吞吐量達 5.24 次。

\begin{table}[htbp]
  \centering
  \small
  \caption{圖資料結構建構開銷對比（全圖 47.3 萬節點、720 萬有向邊）}
  \label{tab:graph_construction_benchmark}
  \begin{tabular}{l cccc}
    \toprule
    資料結構與建圖模式 & 建構耗時 & 核心陣列 RAM & 行程 RSS 增幅 & 每秒吞吐上限 \\
    \midrule
    \textbf{CSR 稀疏矩陣 (Scipy CSR)} & \textbf{190.82 ms} & \textbf{56.74 MB} & \textbf{113.98 MB} & \textbf{5.24 次/秒} \\
    NetworkX 靜態拓樸 & 7,432.2 ms & 純物件無連續陣列 & 852.33 MB & 0.13 次/秒 \\
    NetworkX 動態帶權 (傳統 XAI 模式) & 13,007.2 ms & 純物件無連續陣列 & 1,581.41 MB & 0.077 次/秒 \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{$2 \times 2$ 因子消融與搜尋延遲評估}
本研究選取 18 組具統計代表性之測試案例（涵蓋展示案例、中度互動、高度壓力、超級節點與孤立節點），評估四象限演算法效能，結果如表~\ref{tab:traversal_latency_benchmark} 所示。實測證明，CSR 雙向演算法在所有 18 組案例中與 NetworkX 單向搜尋達成 \textbf{100\% 拓樸雙射與數值等價}，平均遍歷延遲自 749.95 ms 驟降至 0.067 ms（純遍歷提速 \textbf{11,205.2 倍}）。若計入動態建圖之端到端查詢延遲，CSR 架構更達成 \textbf{205,550.2 倍} 之極限加速。

\begin{table}[htbp]
  \centering
  \footnotesize
  \caption{路徑搜尋延遲四象限消融對比（三次獨立運行平均值）}
  \label{tab:traversal_latency_benchmark}
  \begin{tabular}{l cccc cc}
    \toprule
    測試分組 & NetworkX (Uni) & NetworkX (Bi) & CSR (Uni) & CSR (Bi, 本研究) & 遍歷提速比 & 等價性 \\
    \midrule
    典型口試案例 ($n=5$) & 51.73 ms & 0.032 ms & 0.206 ms & \textbf{0.013 ms} & 3,979.2$\times$ & 100\% PASS \\
    中度互動使用者 ($n=5$) & 326.04 ms & 0.094 ms & 0.783 ms & \textbf{0.027 ms} & 12,075.6$\times$ & 100\% PASS \\
    高度壓力使用者 ($n=5$) & 1,854.92 ms & 0.704 ms & 5.577 ms & \textbf{0.134 ms} & 13,842.7$\times$ & 100\% PASS \\
    超級節點案例 ($n=2$) & 730.56 ms & 0.511 ms & 4.003 ms & \textbf{0.092 ms} & 7,940.9$\times$ & 100\% PASS \\
    非連通節點對 ($n=1$) & 874.44 ms & 0.878 ms & 5.555 ms & \textbf{0.150 ms} & 5,829.6$\times$ & 100\% PASS \\
    \midrule
    \textbf{總體平均表現} & \textbf{749.95 ms} & \textbf{0.336 ms} & \textbf{2.577 ms} & \textbf{0.067 ms} & \textbf{11,205.2$\times$} & \textbf{100\% PASS} \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{演算法複雜度與硬體快取機制解耦}
透過 $2 \times 2$ 矩陣分解加速來源：
\begin{enumerate}[leftmargin=2em]
  \item \textbf{演算法複雜度增益 ($\mathcal{O}(d^3) \to \mathcal{O}(d^2)$)：}在 NetworkX 內部改用雙向搜尋帶來 2,230.2 倍提速；在 CSR 內部帶來 38.5 倍提速。
  \item \textbf{資料結構與快取親和增益 (Python 字典 $\to$ C 陣列)：}在單向搜尋下帶來 291.0 倍提速；在雙向搜尋下帶來 5.0 倍提速。
\end{enumerate}

\subsection{非對稱超級節點屏障機制}
即便圖譜已修剪高頻食材，圖中仍存在度數高達 22,095 之超級節點 (Entity 14883)。傳統單向搜尋在探訪該實體時引發 876.26 ms 嚴重阻塞；而本研究之雙向演算法將後向集合 $\mathcal{S}_v$ 錨定於食譜端（鄰居數 $<100$，小於 2 KB，常駐 CPU L1 快取），前向對超級節點僅做 $\mathcal{O}(1)$ 成員查驗，完全旁路超級節點的兩萬個鄰域展開，搜尋延遲維持於 0.038~0.145 ms。
```

---

## 第二部分：口試簡報 (`Paper/Beamer/defense.tex`) 修正方案

### 2.1 最高優先級勘誤修正

#### 修正 1：附錄 D1 注意力權重數值統一（Errata 2）
- **位置**：`defense.tex` Line 619
- **現狀**：右側文字欄寫 `\item \textbf{注意力權重}：$\alpha = 0.767$`，左側圖片標註 `0.7538`。
- **證據**：查 `NotebookLM/notebooklm_explanation.md` Line 848，最佳模型權重精確為 `0.7670428`，論文第 4 章 Line 403 亦為 `0.767`。
- **行動**：維持文字 $\alpha = 0.767$（或 0.7670）；重繪或修訂 `Paper/Main/figures/user_141540.pdf` 將連線標籤統一為 `0.7670`。

#### 修正 2：附錄 B1 高頻食材數量算術落差補齊（Errata 3）
- **位置**：`defense.tex` Lines 575–578
- **現狀**：方框標註 61 個實體，但清點僅印出 54 個食材（缺少 7 個）。
- **證據**：清查確認遺漏之 7 個實體為：`cinnamon`, `onion`, `garlic clove`, `garlic cloves`, `carrots`, `green onions`, `parsley`。
- **行動**：將 Line 577 替換為完整 61 個食材清單（完全對齊 `appendix03.tex`）：
```latex
  \begin{examplebox}{高頻食材過濾清單 (Top 0.5\% - 61 個實體)}
    \tiny
    \texttt{carrot, red onion, onion, unsalted butter, green onion, zucchini, ground beef, ground cinnamon, cinnamon, honey, cream cheese, mayonnaise, fresh parsley, potatoes, celery, red bell pepper, tomatoes, parsley, dijon mustard, orange juice, cayenne pepper, margarine, olive oil, soy sauce, dried oregano, vegetable oil, garlic clove, garlic powder, worcestershire sauce, flour, carrots, powdered sugar, paprika, lemon juice, kosher salt, cheddar cheese, milk, garlic cloves, sour cream, chili powder, nutmeg, fresh lemon juice, fresh ground black pepper, mozzarella cheese, cornstarch, parmesan cheese, green onions, heavy cream, chicken broth, raisins, granulated sugar, brown sugar, vanilla extract, extra virgin olive oil, bacon, walnuts, pecans, salt and pepper, ground cumin, pepper, egg}
  \end{examplebox}
```

#### 修正 3：詞彙與字形清零（Errata 6, 7, 8）
- **Slide 16 (Line 370)**：將「觀察模型在殘缺**拓撲** $\mathcal{G}_{\setminus S}$ 上的預測降幅」改為「殘缺**拓樸**」。
- **附錄 A2 (Line 563)**：將「均勻聚合具隱含**正則化**」改為「隱含**正規化**」。
- **附錄 B1, B2 (Lines 569, 570, 581)**：將「噪音」全數替換為台灣學術標準「**雜訊**」（「食材雜訊過濾清單」、「基礎調味料雜訊清單」、「標籤雜訊過濾清單」）。

---

### 2.2 CSR 核心貢獻投影片植入藍圖

#### 主體頁新增：Slide 19（介於 Case C 與結論頁之間）
```latex
\begin{frame}{RQ3 延伸實踐：大規模 CKG 即時路徑萃取加速與互動展示}
  \begin{columns}[c]
    \begin{column}{0.50\textwidth}
      \begin{mybox}{ntublue}{CSR 雙向局部交集向量化演算法效益}
        \footnotesize
        \begin{itemize}
          \item \textbf{百萬邊建圖開銷}：CSR 連續陣列僅 \textbf{56.74 MB / 190 ms}（NetworkX 需 1.58 GB / 13.01 s，節省 93\%）。
          \item \textbf{路徑萃取延遲}：平均自 749.95 ms 降至 \textbf{0.067 ms}（\textbf{純遍歷提速 11,205 倍}；端到端提速 \textbf{20.5 萬倍}）。
          \item \textbf{拓樸雙射等價}：18 組嚴格基準全數 \textbf{100\% PASS}。
          \item \textbf{非對稱超節點屏障}：度數 22,095 超熱門實體延遲鎖定在 \textbf{0.145 ms}，完全旁路巨型鄰域展開。
        \end{itemize}
      \end{mybox}
    \end{column}

    \begin{column}{0.48\textwidth}
      \centering
      \includegraphics[width=\linewidth, keepaspectratio]{../Main/figures/demo_screenshot.png}
      \vspace{0.1cm}
      \small \textbf{展示系統}：連線推論 / 離線雙軌架構 (8501 埠)
    \end{column}
  \end{columns}
\end{frame}
```

#### 主體頁改版：Slide 20 論文主要貢獻總結（擴充為四大貢獻）
將原先三欄重構為 4 個精緻區塊（2x2 格點佈局），加入「**貢獻四：即時路徑萃取與展示系統**：提出基於 CSR 連續陣列之雙向會合尋路演算法，於 720 萬邊圖譜上將平均尋路延遲降至 0.067 ms（提速逾 1.1 萬倍），並建構高可靠雙軌展示系統。」

#### 附錄頁新增：附錄 G1 與 G2（接在附錄 F 之後）
- **附錄 G1**：`附錄 G1: 大規模 CKG 路徑萃取基準測試完整數據 (18 組對照)`（放入 18 組樣本之四象限消融延遲表）。
- **附錄 G2**：`附錄 G2: CSR 雙向局部交集演算法理論雙射等價證明與快取機制`（三部圖定界、$L=3$ 排他性證明與 L1 Cache 常駐圖解）。

---

## 第三部分：展示系統 (`demo/`) 修正與資料契約規範

### 3.1 最高優先級勘誤修正

#### 修正 1：前端展示系統 User ID「圖文張冠李戴」修訂（Errata 1）
- **問題根源**：`demo/export_demo_cases.py` 在封裝資料時，頂部選單使用 `uid`（如 `17049`），但路徑節點物件之 `name` 與 `path_description` 調用 `get_node_name` 產生了原始 ID 字串（如 `User 144146`）。
- **修復代碼（在 `demo/export_demo_cases.py`）**：
  在產生 `explanations` 前端資料結構時，針對 `USER` 節點強制將顯示名稱規範為一致之 Remapped ID：
  ```python
  # 修正前：name = "User 144146", path_desc = "User 144146 -> ..."
  # 修正後：
  for exp in rec.get("explanations", []):
      fixed_details = []
      for node in exp.get("path_details", []):
          node_copy = dict(node)
          if node_copy.get("type") == "USER":
              node_copy["name"] = f"User {node_copy['id_remapped']}"
          fixed_details.append(node_copy)
      fixed_desc = exp.get("path_description", "")
      # 正則或字串替換開頭的原始使用者 ID 為一致的 User {uid}
      if fixed_details and fixed_details[0]["type"] == "USER":
          first_name = fixed_details[0]["name"]
          parts = fixed_desc.split(" -> ")
          if len(parts) > 1:
              fixed_desc = f"{first_name} -> " + " -> ".join(parts[1:])
      exps.append({
          "path_structure": exp.get("path_structure", ""),
          "path_description": fixed_desc,
          "path_details": fixed_details,
          "contribution_score": exp.get("contribution_score", 0.0),
      })
  ```
- **執行重產**：執行 `python demo/export_demo_cases.py`，更新 `demo_cases.json` 與 `demo_cases_data.js`，重新產出前端截圖。

#### 修正 2：忠實度公式記號對齊（Errata 5）
- **位置**：`demo/web/index.html` Lines 144 與 149
- **修訂為**：
  - Line 144：`<div class="probe-stat-note" id="fPlusNote">p̂(&#x1D43A;) &minus; p̂(&#x1D43A; \ &#x1D446;)</div>`
  - Line 149：`<div class="probe-stat-note" id="fMinusNote">p̂(&#x1D43A;) &minus; p̂(&#x1D43A;<sub>&#x1D446;</sub>)</div>`

#### 修正 3：`ADR-009` 第 3.1 節數據同步（Errata 4）
- **位置**：`docs/adr/ADR-009-Oral-Defense-Demo-Packaging.md` Line 61
- **修訂為**：
  `尋路時間自 749.95 毫秒驟降至 0.067 毫秒（提速達 11,205.2 倍；計入動態建圖端到端提速達 205,550.2 倍），記憶體開銷縮減 93%（自 1.58 GB 降至 56.74 MB），並與 NetworkX 圖論完整搜尋達成 100% 拓樸雙射與數值等價性。`

---

### 3.2 後端推論服務 API 接口與資料契約規範 (Data Contract)

遵循 Subagent 定義規範，提供推論端點之強型別契約：

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=0, le=226569, description="內部重新映射之使用者識別碼 (0~226569)")
    depth: str = Field(default="3", regex="^(1|2|3|wo_kg|wo_attn|bprmf|lightgcn|nfm)$", description="模型深度或消融實驗類型")

class PathNodeDetail(BaseModel):
    id_remapped: int = Field(..., description="圖中內部重新映射 ID")
    id_original: str = Field(..., description="原始資料庫字串 ID")
    type: str = Field(..., regex="^(USER|RECIPE|INGREDIENT|TAG)$", description="節點類型")
    name: str = Field(..., description="展示名稱（使用者一律為 User {id_remapped}）")

class ExplanationPath(BaseModel):
    path_structure: str = Field(..., description="元路徑模式，如 USER -> RECIPE -> TAG -> RECIPE")
    path_description: str = Field(..., description="人類可讀路徑字串")
    path_details: List[PathNodeDetail] = Field(..., description="路徑節點詳細序列")
    contribution_score: float = Field(..., ge=0.0, le=1.0, description="注意力權重連乘分數")

class FidelityMetrics(BaseModel):
    f_plus: Optional[float] = Field(None, description="必要性忠實度 F+")
    f_minus: Optional[float] = Field(None, description="充分性忠實度 F-")
    prob_f_plus: Optional[float] = Field(None, description="遮蔽解釋邊後之預測機率")
    prob_f_minus: Optional[float] = Field(None, description="僅保留解釋邊後之預測機率")

class RecommendResponse(BaseModel):
    status: str = Field(default="success", description="狀態碼：success 或 error")
    user_id: int
    depth: str
    recommended_item_id: int
    recommended_item_name: str
    score: float
    prob: float
    is_test_hit: bool
    user_n_interactions: int
    item_kg_degree: int
    explanations: List[ExplanationPath]
    fidelity: FidelityMetrics
    inference_time_ms: float = Field(..., description="推論與尋路耗時（毫秒）")

class FidelityProbeRequest(BaseModel):
    user_id: int = Field(..., ge=0, le=226569)
    item_id: int = Field(..., ge=0, le=231636)
    depth: str = Field(default="3")
    mode: str = Field(..., regex="^(f_plus|f_minus)$", description="探針擾動模式")
    path_nodes: Optional[List[List[int]]] = Field(None, description="指定擾動之路徑節點清單")

class FidelityProbeResponse(BaseModel):
    status: str = "success"
    mode: str
    original_prob: float
    probed_prob: float
    fidelity_delta: float
    execution_time_ms: float
```

#### 錯誤分類碼表 (Error Taxonomy)
| 錯誤代碼 (Code) | HTTP 狀態 | 觸發原因 (Condition) | 回傳 JSON 範例 |
| :--- | :---: | :--- | :--- |
| `USER_NOT_FOUND` | 404 | 輸入之 `user_id` 超出 [0, 226569] 範圍 | `{"error": "USER_NOT_FOUND", "message": "使用者 ID 999999 超出範圍。"}` |
| `MODEL_NOT_READY` | 503 | 伺服器模型權重尚未載入完成 | `{"error": "MODEL_NOT_READY", "message": "模型 KGAT-L3 正在初始化。"}` |
| `INVALID_PROBE_MODE`| 400 | 探針模式非 `f_plus` 或 `f_minus` | `{"error": "INVALID_PROBE_MODE", "message": "未知模式。"}` |

---

## 第四部分：下游交接路線圖與驗證清單

```
[Phase 1: 檔案修正] 
  ├── Paper/Main/ (ch1, ch2, ch3, ch4) 植入 CSR 演算法、理論等價性與消融數據
  ├── Paper/Beamer/ (defense.tex) 修復 Errata 2, 3, 6, 7, 8，新增 Slide 19 與附錄 G
  ├── demo/export_demo_cases.py 修復 User ID 錯位 (Errata 1) 並重新匯出
  └── demo/web/index.html 修正公式下標 (Errata 5)

[Phase 2: 自主閉環驗證]
  ├── pwsh 執行測試：python -m pytest tests/test_demo_server.py
  ├── 論文編譯測試：於 Paper/Main/ 執行 xelatex main.tex && bibtex main
  └── 簡報編譯測試：於 Paper/Beamer/ 執行 xelatex defense.tex

[Phase 3: 最終交付物更新]
  ├── demo 重新截圖並覆蓋 Paper/Main/figures/
  └── 更新 README.md、CHANGELOG.md 與 HANDOVER.md
```
