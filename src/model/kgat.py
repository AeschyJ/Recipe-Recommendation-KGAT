import torch
import torch.nn as nn
import torch.nn.functional as F


class KGAT(nn.Module):
    def __init__(
        self,
        n_users,
        n_entities,
        n_relations,
        embed_dim=64,
        layers=[64, 32],
        mess_dropout=[0.1, 0.1],
        adj_adj_dropout=[0.0, 0.0],
    ):
        super(KGAT, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = embed_dim

        # Embeddings
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.entity_embed = nn.Embedding(n_entities, embed_dim)
        self.relation_embed = nn.Embedding(n_relations, embed_dim)

        # Weight matrices for transductive KGE (e.g. TransR)
        self.W_R = nn.Parameter(torch.Tensor(n_relations, embed_dim, embed_dim))

        # Graph Attention Layers
        self.aggregator_layers = nn.ModuleList()
        in_dim = embed_dim
        for out_dim in layers:
            self.aggregator_layers.append(GNNLayer(in_dim, out_dim))
            in_dim = out_dim

        self.mess_dropout = mess_dropout
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.W_R)

    def forward(self, adj, user_ids, pos_item_ids, neg_item_ids=None):
        """
        adj: torch.sparse_coo_tensor (Adjacency Matrix)
        user_ids: tensor of user indices
        pos_item_ids: tensor of positive item indices
        neg_item_ids: (Optional) tensor of negative item indices
        """
        # 1. 優化：預先提取稀疏矩陣索引，避免在每一層重複提取與檢查 layout
        if adj.layout == torch.sparse_coo:
            target, neighbor = adj.indices()[0], adj.indices()[1]
            values = adj.values()
        elif adj.layout == torch.sparse_csr:
            coo = adj.to_sparse_coo()
            target, neighbor = coo.indices()
            values = coo.values()
        else:
            raise ValueError(f"Unsupported adjacency layout: {adj.layout}")

        # Initial features: Concatenate user and entity embeddings
        all_embed = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)

        # List to store embeddings at each layer
        ego_embeddings = [all_embed]

        for i, layer in enumerate(self.aggregator_layers):
            # 傳遞預提取的矩陣參數
            all_embed = layer(all_embed, target, neighbor, values)
            all_embed = F.normalize(all_embed, p=2, dim=1)
            ego_embeddings.append(all_embed)

        # 2. 合併多層特徵
        final_embed = torch.cat(ego_embeddings, dim=1)

        # 3. 提取對應的嵌入
        u_embed = final_embed[user_ids]
        pos_i_embed = final_embed[self.n_users + pos_item_ids]

        # 計算正樣本分數
        pos_scores = torch.sum(u_embed * pos_i_embed, dim=1)

        if neg_item_ids is not None:
            neg_i_embed = final_embed[self.n_users + neg_item_ids]
            neg_scores = torch.sum(u_embed * neg_i_embed, dim=1)
            return pos_scores, neg_scores

        return pos_scores


class SparseAggregateFunction(torch.autograd.Function):
    """
    自定義聚合算子，用於節省記憶體。
    不保留訊息矩陣 (E, dim)，而是在 Backward 時重新計算。
    """

    @staticmethod
    def forward(ctx, features, target, neighbor, values, num_nodes):
        # 儲存必要的張量供 Backward 使用
        ctx.save_for_backward(features, target, neighbor, values)
        ctx.num_nodes = num_nodes

        # 計算聚集後的結果 h_neigh = sum(neighbor_features * edge_values)
        # 這裡會暫時建立大型 msg 張量，但 Function 結束後即會釋放其在 Forward Graph 的引用
        msg = features[neighbor] * values.unsqueeze(1)
        h_neigh = torch.zeros(
            num_nodes, features.shape[1], device=features.device, dtype=features.dtype
        )
        h_neigh.index_add_(0, target, msg)
        return h_neigh

    @staticmethod
    def backward(ctx, grad_h_neigh):
        features, target, neighbor, values = ctx.saved_tensors
        num_nodes = ctx.num_nodes

        # 梯度計算：
        # 1. 對 features 的梯度 = grad_h_neigh[target] * values
        # 這裡重新計算加權梯度，避免 Forward 儲存它
        weighted_grad = grad_h_neigh[target] * values.unsqueeze(1)
        grad_features = torch.zeros_like(features)
        grad_features.index_add_(0, neighbor, weighted_grad)

        # 2. 對 values 的梯度 = (grad_h_neigh[target] * features[neighbor]).sum(dim=1)
        # 用於處理權重學習 (如果有需要的話)
        grad_values = (grad_h_neigh[target] * features[neighbor]).sum(dim=1)

        return grad_features, None, None, grad_values, None


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, features, target, neighbor, values):
        """
        features: (N, in_dim)
        target: source indices for index_add
        neighbor: neighbor indices for gathering message
        values: edge values (weights)
        """
        num_nodes = features.shape[0]

        # 使用記憶體優化的聚合算子 (避免在 Forward 保存大型中間矩陣)
        h_neigh = SparseAggregateFunction.apply(
            features, target, neighbor, values, num_nodes
        )

        # Bi-Interaction Aggregation
        h_out = self.leaky_relu(
            self.W1(features + h_neigh) + self.W2(features * h_neigh)
        )
        return h_out
