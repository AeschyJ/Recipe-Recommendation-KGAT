import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayerAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayerAttention, self).__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.W_att = nn.Linear(in_dim, out_dim)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, indices, features, edge_types, num_nodes, return_attention=False):
        """
        Memory-optimized Forward pass using Gather-Scatter (Message Passing).
        Avoids OOM by not creating large N*N matrices or broadcasting dense tensors.

        Args:
            indices: (2, E) Edge indices
            features: (N, D) Node features
            edge_types: (E,) Relation type for each edge (0: Ingredient, 1: Tag, etc.)
            num_nodes: Total number of nodes
        """
        src, dst = indices[0], indices[1]
        device = features.device

        # 1. Attention Input
        h_trans = self.W_att(features)
        h_src = h_trans[src]
        h_dst = h_trans[dst]
        del h_trans  # Release memory

        # 2. Attention Coefficients
        edge_h = torch.cat([h_src, h_dst], dim=1)

        # LeakyReLU(a^T [Wh_i || Wh_j])
        # Add numerical scaling factor: 1 / sqrt(out_dim)
        scaling = 1.0 / (self.W_att.out_features**0.5)

        # 計算 Attention Score
        # shape: (E, 1)
        e_ij_raw = torch.matmul(edge_h, self.a)

        # 特徵加權：針對 Ingredient 類型的路徑給予額外 Bias
        # 假設 Relation 0 = Ingredient, Relation 1 = Tag, Relation >= 2 = Others (Interaction)
        # 我們希望 Ingredient 的初始權重較高
        # Create a bias tensor based on edge_types
        # 這裡簡單給 Ingredient (type 0) 加分，其他不加

        # 注意：edge_types 可能包含互動邊 (Users-Items)，需確認其 Relation ID
        # 在 train_att.py 中構造圖時，KG 部分是 0 和 1，User-Item 互動我們視為另一種 Relation (例如 2)
        # 這裡我們給 Relation 0 (Ingredient) 一個 bonus

        if edge_types is not None:
            # 初始化 Bias，Ingredient 給 5.0，其他 0
            # 使用 where 來向量化操作
            # 確保 edge_types 與 e_ij 在同一裝置
            is_ingredient = (edge_types == 0).float().unsqueeze(1)  # (E, 1)
            bias = is_ingredient * 5.0
            e_ij_raw = e_ij_raw + bias

        e_ij = self.leaky_relu(e_ij_raw * scaling)
        del edge_h, e_ij_raw

        # 3. Softmax & Message Passing
        # To support mixed precision (AMP) properly, we respect the incoming feature dtype
        current_dtype = features.dtype

        # Temporarily disable autocast for scatter operations if they cause issues,
        # but keep data types consistent.
        with (
            torch.autocast(device_type=device.type, enabled=False)
            if device.type != "cpu"
            else torch.no_grad()
        ):
            # 確保 e_ij 精度足夠進行 exp
            e_ij = e_ij.float()

            # 數值穩定性優化：使用 Scatter Max
            max_score = torch.full(
                (num_nodes, 1), -1e9, device=device, dtype=e_ij.dtype
            )
            max_score.scatter_reduce_(
                0, dst.unsqueeze(1), e_ij.detach(), reduce="amax", include_self=False
            )

            e_ij = e_ij - max_score[dst]
            exp_e = torch.exp(e_ij)

            denom = torch.zeros(num_nodes, 1, device=device, dtype=e_ij.dtype)
            denom.scatter_add_(0, dst.unsqueeze(1), exp_e)

            alpha = exp_e / (denom[dst] + 1e-9)
            del exp_e, denom, max_score

            # Message Passing
            # 修正：不要強制轉為 float()，確保與 layer 權重 (可能是 BFloat16) 一致
            h_msg_all = self.W1(features)
            msg = h_msg_all[src]
            del h_msg_all

            weighted_msg = msg * alpha.to(current_dtype)
            del msg

            h_neigh = torch.zeros(
                num_nodes, self.W1.out_features, device=device, dtype=current_dtype
            )
            h_neigh.index_add_(0, dst, weighted_msg)
            del weighted_msg

        # Convert back to original dtype (e.g. float16 if using AMP)
        h_neigh = h_neigh.to(current_dtype)
        alpha = alpha.to(current_dtype)

        # 4. Bi-Interaction
        # Re-compute h_self in original precision
        h_self = (
            self.W1(features) if features.shape[1] != h_neigh.shape[1] else features
        )

        # Step-by-step aggregation to save memory
        sum_h = h_self + h_neigh
        prod_h = h_self * h_neigh
        w2_prod = self.W2(prod_h)
        h_out = self.leaky_relu(sum_h + w2_prod)

        if return_attention:
            return h_out, alpha
        return h_out


class KGATAttention(nn.Module):
    def __init__(self, n_users, n_entities, n_relations, embed_dim=32, layers=[32]):
        super(KGATAttention, self).__init__()
        self.n_users = n_users
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.entity_embed = nn.Embedding(n_entities, embed_dim)
        self.relation_embed = nn.Embedding(n_relations, embed_dim)

        self.aggregator_layers = nn.ModuleList()
        in_dim = embed_dim
        for out_dim in layers:
            self.aggregator_layers.append(GNNLayerAttention(in_dim, out_dim))
            in_dim = out_dim

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

    def forward(
        self, indices, edge_types, num_nodes, user_ids, item_ids, return_attention=False
    ):
        """
        indices: (2, E) LongTensor
        edge_types: (E,) LongTensor -- added for relation-aware attention
        num_nodes: int
        """
        all_embed = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        ego_embeddings = [all_embed]
        attentions = []

        for layer in self.aggregator_layers:
            if return_attention:
                all_embed, att = layer(indices, all_embed, edge_types, num_nodes, True)
                attentions.append(att)
            else:
                all_embed = layer(indices, all_embed, edge_types, num_nodes, False)

            all_embed = F.normalize(all_embed, p=2, dim=1)
            ego_embeddings.append(all_embed)

        final_embed = torch.cat(ego_embeddings, dim=1)
        u_embed = final_embed[user_ids]
        i_embed = final_embed[self.n_users + item_ids]

        scores = torch.sum(u_embed * i_embed, dim=1)

        if return_attention:
            return scores, attentions
        return scores
