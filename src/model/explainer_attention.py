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

    def explain(self, indices, num_nodes, user_ids, item_ids, n_hops=2, top_k=10):
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
                indices, num_nodes, user_ids, item_ids, return_attention=True
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
