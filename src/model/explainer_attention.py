import matplotlib.pyplot as plt
import networkx as nx
import torch


class KGATAttentionExplainer:
    def __init__(self, model):
        """
        KGAT Attention 解釋器.
        專門用於解釋 KGATAttention 模型，直接利用模型回傳的 Attention Weights。

        Args:
            model: 訓練好的 KGATAttention 模型
        """
        self.model = model
        self.model.eval()

    def explain(
        self, indices, edge_types, num_nodes, user_ids, item_ids, n_hops=2, top_k=10
    ):
        """
        解釋推薦原因。

        Args:
            indices: torch.LongTensor (2, E)
            num_nodes: int
            user_ids: int or tensor
            item_ids: int or tensor
            n_hops: int
            top_k: int

        Returns:
            explanation (dict)
        """
        # 確保輸入是 Tensor
        if isinstance(user_ids, int):
            user_ids = torch.LongTensor([user_ids])
        if isinstance(item_ids, int):
            item_ids = torch.LongTensor([item_ids])

        user_ids = user_ids.to(indices.device)
        item_ids = item_ids.to(indices.device)

        # 1. 取得 Attention Weights
        # KGATAttention.forward(..., return_attention=True) 會回傳 (scores, attentions)
        with torch.no_grad():
            scores, attentions = self.model(
                indices,
                edge_types,
                num_nodes,
                user_ids,
                item_ids,
                return_attention=True,
            )

        # attentions 是一個 list，包含每一層 GNN 的 attention weights
        # attentions[i] shape: (E, 1) or (E, heads)
        # 這裡假設單頭注意力 (E, 1)

        # 我們可以将所有層的 attention 取平均，或者只看最後一層
        # 這裡採用平均策略，更能反映整個訊息傳遞過程
        if len(attentions) > 0:
            # stack: (L, E, 1) -> mean: (E, 1) -> squeeze: (E,)
            final_att = torch.mean(torch.stack(attentions, dim=0), dim=0).squeeze()
        else:
            print("No attention weights returned.")
            return None

        # 2. 構建解釋圖 (使用 Attention 作為權重)
        u_id = user_ids.item()
        # 注意: KGATAttention 預期輸入的 item_ids 是原始 ID，但在圖中已偏移
        i_id_global = self.model.n_users + item_ids.item()

        edge_indices = indices.t().cpu().numpy()  # (E, 2)
        edge_weights = final_att.cpu().numpy()  # (E, )

        # 建立 NetworkX 圖
        G = nx.Graph()
        for (src, dst), w in zip(edge_indices, edge_weights):
            # 只加入權重非零的邊 (雖 softmax 後通常都非零，但可設閾值)
            if w > 1e-6:
                G.add_edge(src, dst, weight=float(w))

        # 3. 尋找路徑
        important_paths = []
        try:
            # KGAT 預設 2 層，所以我們找長度 <= 3 的路徑 (nodes: u -> e1 -> e2 -> i)
            paths = list(
                nx.all_simple_paths(
                    G, source=u_id, target=i_id_global, cutoff=n_hops + 1
                )
            )

            path_scores = []
            for path in paths:
                # 計算路徑分數
                # 這裡定義路徑分數為邊權重的乘積 (Joint Probability 概念)
                score = 1.0
                for k in range(len(path) - 1):
                    w = G[path[k]][path[k + 1]].get("weight", 0.0)
                    score *= w
                path_scores.append((path, score))

            # 排序並取 Top K
            path_scores.sort(key=lambda x: x[1], reverse=True)
            important_paths = path_scores[:top_k]

        except nx.NetworkXNoPath:
            print(f"No path found between User {u_id} and Item {i_id_global}")

        # 4. 建立子圖
        explanation_subgraph = nx.Graph()
        for path, score in important_paths:
            nx.add_path(explanation_subgraph, path, weight=score)

        return {
            "subgraph": explanation_subgraph,
            "top_paths": important_paths,
            "target_score": scores.item(),
        }

    def explain_fast_csr(
        self,
        indices,
        edge_types,
        num_nodes,
        user_id,
        item_id,
        csr_adj,
        top_k=3,
        n_hops=2,
    ):
        """基於壓縮稀疏矩陣 (CSR) 之局部 3-hop 閉環向量化尋路演算法.

        透過查詢 u 鄰域食譜 r1 與目標食譜 v 之共鄰中繼 m in N(r1) cap N(v)，
        精確窮舉所有 3-hop 簡單路徑，避免全局圖構建與 DFS，實現數毫秒級尋路。
        """
        import numpy as np

        if isinstance(user_id, (int, np.integer)):
            u_tensor = torch.LongTensor([int(user_id)])
            u_id = int(user_id)
        else:
            u_tensor = user_id
            u_id = int(user_id.item())

        if isinstance(item_id, (int, np.integer)):
            i_tensor = torch.LongTensor([int(item_id)])
            i_id = int(item_id)
        else:
            i_tensor = item_id
            i_id = int(item_id.item())

        i_id_global = self.model.n_users + i_id

        # 邊界保護：若節點編號超出 CSR 矩陣維度或為負值，安全回傳空子圖
        if (
            u_id < 0
            or u_id >= csr_adj.shape[0]
            or i_id_global < 0
            or i_id_global >= csr_adj.shape[0]
        ):
            return {
                "subgraph": nx.Graph(),
                "top_paths": [],
                "target_score": 0.0,
                "search_time": 0.0,
                "search_time_ms": 0.0,
            }

        u_tensor = u_tensor.to(indices.device)
        i_tensor = i_tensor.to(indices.device)

        with torch.no_grad():
            scores, attentions = self.model(
                indices,
                edge_types,
                num_nodes,
                u_tensor,
                i_tensor,
                return_attention=True,
            )

        if len(attentions) > 0:
            final_att = torch.mean(torch.stack(attentions, dim=0), dim=0).squeeze().cpu().numpy()
            del attentions
            if indices.device.type == "xpu":
                torch.xpu.empty_cache()
            elif indices.device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            return None

        indptr = csr_adj.indptr
        col_indices = csr_adj.indices
        edge_data = csr_adj.data

        def get_edge_weight(a, b):
            start_a, end_a = indptr[a], indptr[a + 1]
            cols_a = col_indices[start_a:end_a]
            pos_b = np.searchsorted(cols_a, b)
            e1 = edge_data[start_a + pos_b] - 1 if (pos_b < len(cols_a) and cols_a[pos_b] == b) else -1

            start_b, end_b = indptr[b], indptr[b + 1]
            cols_b = col_indices[start_b:end_b]
            pos_a = np.searchsorted(cols_b, a)
            e2 = edge_data[start_b + pos_a] - 1 if (pos_a < len(cols_b) and cols_b[pos_a] == a) else -1

            e = max(e1, e2)
            if e >= 0:
                w = float(final_att[e])
                return w if w > 1e-6 else 0.0
            return 0.0

        import time
        t_search_0 = time.perf_counter()

        u_nbrs = col_indices[indptr[u_id]:indptr[u_id + 1]]
        recipes_u = [r for r in u_nbrs if r != u_id and r != i_id_global]

        v_nbrs = col_indices[indptr[i_id_global]:indptr[i_id_global + 1]]
        v_cand_set = set(m for m in v_nbrs if m != u_id and m != i_id_global)

        path_scores = []
        # 若存在直連邊且 n_hops >= 1
        if i_id_global in u_nbrs:
            w0 = get_edge_weight(u_id, i_id_global)
            if w0 > 1e-6:
                path_scores.append(([u_id, i_id_global], w0))

        # 搜尋 3-hop 簡單路徑: u -> r1 -> m -> i_id_global
        if n_hops >= 2:
            for r1 in recipes_u:
                w_u_r1 = get_edge_weight(u_id, r1)
                if w_u_r1 <= 1e-6:
                    continue
                r1_nbrs = col_indices[indptr[r1]:indptr[r1 + 1]]
                for m in r1_nbrs:
                    if m in v_cand_set and m != r1:
                        w_r1_m = get_edge_weight(r1, m)
                        if w_r1_m <= 1e-6:
                            continue
                        w_m_v = get_edge_weight(m, i_id_global)
                        if w_m_v <= 1e-6:
                            continue
                        score = w_u_r1 * w_r1_m * w_m_v
                        path_scores.append(([u_id, int(r1), int(m), i_id_global], score))

        path_scores.sort(key=lambda x: x[1], reverse=True)
        important_paths = path_scores[:top_k]
        search_time = time.perf_counter() - t_search_0

        explanation_subgraph = nx.Graph()
        for path, score in important_paths:
            nx.add_path(explanation_subgraph, path, weight=score)

        return {
            "subgraph": explanation_subgraph,
            "top_paths": important_paths,
            "target_score": scores.item(),
            "search_time": search_time,
            "search_time_ms": search_time * 1000,
        }

    def visualize(self, explanation, id_maps=None):
        """
        視覺化
        """
        graph = explanation["subgraph"]
        if graph.number_of_nodes() == 0:
            print("Empty explanation graph.")
            return

        pos = nx.spring_layout(graph)
        plt.figure(figsize=(10, 8))

        # Draw Nodes
        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="lightblue")

        # Draw Labels
        labels = {node: str(node) for node in graph.nodes()}
        # TODO: Implement ID mapping to names if id_maps is provided
        nx.draw_networkx_labels(graph, pos, labels=labels)

        # Draw Edges with width proportional to attention
        weights = [graph[u][v]["weight"] for u, v in graph.edges()]

        if weights:
            max_w = max(weights)
            min_w = min(weights)
            # Normalize for visualization width (1 ~ 5)
            if max_w > min_w:
                width = [(w - min_w) / (max_w - min_w) * 4 + 1 for w in weights]
            else:
                width = [2 for _ in weights]
        else:
            width = []

        nx.draw_networkx_edges(graph, pos, width=width, edge_color="gray", alpha=0.7)

        plt.title("KGAT Attention Explanation")
        plt.axis("off")
        plt.show()
