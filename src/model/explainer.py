import types

import matplotlib.pyplot as plt
import networkx as nx
import torch


class SparseMMFunction(torch.autograd.Function):
    """
    Memory-efficient Sparse x Dense Matrix Multiplication for Gradient Calculation.
    Avoids materializing (N, N) dense gradient matrix for the sparse adjacency.
    """

    @staticmethod
    def forward(ctx, adj, features):
        # adj: Sparse Tensor
        # features: Dense Tensor
        ctx.save_for_backward(adj, features)
        return torch.sparse.mm(adj, features)

    @staticmethod
    def backward(ctx, grad_output):
        adj, features = ctx.saved_tensors
        grad_adj = None
        grad_features = None

        # 1. Gradient w.r.t features (Dense)
        # dL/dX = A^T * dL/dY
        if ctx.needs_input_grad[1]:
            grad_features = torch.sparse.mm(adj.t(), grad_output)

        # 2. Gradient w.r.t adj (Sparse)
        # We compute gradient only for the non-zero elements (indices) of adj.
        if ctx.needs_input_grad[0]:
            indices = adj._indices()
            row, col = indices

            # Efficient computation: dot product of rows
            # grad_val[k] = sum(grad_output[row[k]] * features[col[k]])
            # This avoids creating (N, N) dense matrix

            # Gather relevant rows
            grad_out_rows = grad_output[row]  # (E, out_dim)
            feat_rows = features[
                col
            ]  # (E, in_dim) aka (E, out_dim since W is applied later) wait mm is first.
            # features is (N, in_dim). grad_output is (N, in_dim) (result of sparse.mm)

            vals_grad = (grad_out_rows * feat_rows).sum(dim=1)

            # Create sparse gradient tensor
            grad_adj = torch.sparse_coo_tensor(
                indices, vals_grad, adj.shape, device=adj.device
            )

        return grad_adj, grad_features


class KGATExplainer:
    def __init__(self, model):
        """
        KGAT 解釋器 (Gradient-based Saliency).

        Args:
            model: 訓練好的 KGAT 模型 (Pure PyTorch)
        """
        self.model = model
        self.model.eval()
        self._patch_model()

    def _patch_model(self):
        """
        Monkey-patch GNN layers to use memory-efficient SparseMM.
        """
        for layer in self.model.aggregator_layers:
            layer.forward = types.MethodType(self._manual_gnn_forward, layer)

    @staticmethod
    def _manual_gnn_forward(self, adj, features):
        """
        Patched forward method for GNNLayer.
        """
        # Original: h_neigh = torch.sparse.mm(adj, features)
        h_neigh = SparseMMFunction.apply(adj, features)

        term1 = self.W1(features + h_neigh)
        term2 = self.W2(features * h_neigh)

        h_out = self.leaky_relu(term1 + term2)
        return h_out

    def explain(self, adj, user_ids, item_ids, n_hops=2, top_k=10):
        """
        解釋為何模型推薦了特定物品給特定使用者。
        計算 Adjacency Matrix 上的梯度，以此作為邊的重要性分數。

        Args:
            adj: torch.sparse_coo_tensor (N, N), 歸一化後的鄰接矩陣
            user_ids: int or tensor, 目標使用者 ID (全域 ID)
            item_ids: int or tensor, 目標物品 ID (全域 ID)
            n_hops: int, 考慮的跳數 (對應 GNN 層數)
            top_k: int, 取前 k 條最重要的路徑

        Returns:
            explanation (dict): 包含 'subgraph' (nx.DiGraph), 'important_edges' (list), 'score'
        """
        # 確保輸入是 Tensor
        # 確保輸入是 Tensor
        if not torch.is_tensor(user_ids):
            user_ids = torch.as_tensor(user_ids, dtype=torch.long).view(-1)
        if not torch.is_tensor(item_ids):
            item_ids = torch.as_tensor(item_ids, dtype=torch.long).view(-1)

        user_ids = user_ids.to(adj.device)
        item_ids = item_ids.to(adj.device)

        # 1. 準備可微分的 Adjacency Matrix Values
        # 注意: torch.sparse 不支援直接對 values 求導，我們需要重建一個新的 sparse tensor
        indices = adj._indices()
        values = adj._values().detach().clone()
        values.requires_grad_(True)

        adj_grad = torch.sparse_coo_tensor(indices, values, adj.shape).to(adj.device)

        # 2. Forward Pass
        # 將模型設為 train 模式以啟用梯度追蹤 (但不更新權重)
        self.model.eval()  # 保持 eval 以關閉 dropout
        # 雖然是 eval，但我們手動對 input (adj_grad) 求導是允許的

        scores = self.model(adj_grad, user_ids, item_ids)

        # 3. Backward Pass (計算梯度)
        # 我們只關心目標分數對 adj values 的梯度
        target_score = scores.sum()
        target_score.backward()

        # 梯度即為重要性 (Saliency)
        # 取絕對值，因為負影響也是一種影響 (或者只取正值視需求而定)
        grads = values.grad
        edge_importance = torch.abs(grads)

        # 4. 提取子圖 (Subgraph Extraction)
        # 我們只對與 User 或 Item 相關的鄰居感興趣 (k-hop)
        # 使用 networkx 來找路徑比較方便

        # 轉換為 CPU 處理圖結構
        u_id = user_ids.item()
        # Item ID 在圖中是偏移過的 (n_users + item_id)
        # model.n_users 是 KGAT 儲存的 User 數量
        i_id_global = self.model.n_users + item_ids.item()

        edge_indices = indices.t().cpu().numpy()
        edge_weights = edge_importance.detach().cpu().numpy()

        # 建立一個臨時圖包含所有邊權重
        # 為了效率，先過濾掉梯度為 0 的邊
        mask = edge_weights > 0
        active_edges = edge_indices[mask]
        active_weights = edge_weights[mask]

        # 構建 NetworkX 圖
        G = nx.Graph()  # 無向圖
        for (src, dst), w in zip(active_edges, active_weights):
            G.add_edge(src, dst, weight=w)

        # 找出從 User 到 Item 的路徑 (限制長度)
        important_paths = []
        try:
            # 尋找所有簡單路徑 (限制長度 <= n_hops + 1)
            # 因為 KGAT 是 2 層，User -> Entity -> Item 是 2 跳
            # 注意: target 必須是全域 ID
            paths = list(
                nx.all_simple_paths(
                    G, source=u_id, target=i_id_global, cutoff=n_hops + 1
                )
            )

            # 計算路徑分數 (路徑上邊權重的總和或平均)
            path_scores = []
            for path in paths:
                score = 0
                for k in range(len(path) - 1):
                    # 累加邊權重
                    score += G[path[k]][path[k + 1]]["weight"]
                path_scores.append((path, score))

            # 取前 top_k
            path_scores.sort(key=lambda x: x[1], reverse=True)
            important_paths = path_scores[:top_k]

        except nx.NetworkXNoPath:
            print(
                f"No path found between User {u_id} and Item {i_id_global} within {n_hops + 1} hops."
            )

        # 建構解釋子圖
        explanation_subgraph = nx.Graph()
        for path, score in important_paths:
            nx.add_path(explanation_subgraph, path, weight=score)

        return {
            "subgraph": explanation_subgraph,
            "top_paths": important_paths,
            "target_score": target_score.item(),
        }

    def visualize(self, explanation, id_maps=None):
        """
        視覺化解釋子圖。

        Args:
            explanation: `explain` 方法的回傳值
            id_maps: dict, 可選，包含 'user_map', 'item_map', 'ingredient_map' 等，用於顯示真實名稱
        """
        graph = explanation["subgraph"]
        if graph.number_of_nodes() == 0:
            print("Empty explanation graph.")
            return

        pos = nx.spring_layout(graph)
        plt.figure(figsize=(10, 8))

        # 繪製節點
        # 可以根據節點類型上不同顏色 (需知道 ID range)
        # 這裡簡化統一繪製
        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="lightblue")

        # 繪製標籤
        labels = {}
        for node in graph.nodes():
            labels[node] = str(node)  # 預設顯示 ID
            # 如果有 id_maps，可以嘗試反查名稱 (需實作反查邏輯)

        nx.draw_networkx_labels(graph, pos, labels=labels)

        # 繪製邊 (粗細代表權重)
        weights = [graph[u][v]["weight"] for u, v in graph.edges()]

        # 正規化權重以便繪圖
        if weights:
            max_w = max(weights)
            min_w = min(weights)
            if max_w > min_w:
                params = [(w - min_w) / (max_w - min_w) * 5 + 1 for w in weights]
            else:
                params = [2 for _ in weights]
        else:
            params = []

        nx.draw_networkx_edges(graph, pos, width=params, edge_color="gray", alpha=0.6)

        plt.title("KGAT Explanation (Gradient-based Saliency)")
        plt.axis("off")
        plt.show()
