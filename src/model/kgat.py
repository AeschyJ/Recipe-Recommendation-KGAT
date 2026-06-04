import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayerAttention(nn.Module):
    def __init__(self, in_dim, out_dim, force_fp32_agg=True):
        super(GNNLayerAttention, self).__init__()
        self.force_fp32_agg = force_fp32_agg
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.W_att = nn.Linear(in_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, indices, features, e_r, num_nodes, return_attention=False):
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

        # 2. Relation-Aware Attention Coefficients
        # 原論文公式: pi(h, r, t) = (W_r e_t)^T tanh(W_r e_h + e_r)
        # 為解決記憶體與 XPU 負擔，我們使用共享 W_att：(W_att e_t)^T tanh(W_att e_h + e_r)

        # 若發生維度不匹配 (如多層遞減)，我們對 e_r 進行截斷 (通常 L=64=embed_dim)
        if e_r.shape[1] > h_src.shape[1]:
            e_r_aligned = e_r[:, : h_src.shape[1]]
        elif e_r.shape[1] < h_src.shape[1]:
            # Padding
            pad = torch.zeros(
                e_r.shape[0],
                h_src.shape[1] - e_r.shape[1],
                device=device,
                dtype=e_r.dtype,
            )
            e_r_aligned = torch.cat([e_r, pad], dim=1)
        else:
            e_r_aligned = e_r

        # e_ij_raw = sum_d( h_dst * tanh(h_src + e_r) )
        e_ij_raw = torch.sum(
            h_dst * torch.tanh(h_src + e_r_aligned), dim=-1, keepdim=True
        )
        scaling = 1.0 / (self.W_att.out_features**0.5)

        e_ij = self.leaky_relu(e_ij_raw * scaling)
        del e_ij_raw

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
            # 這裡修正為直接聚合「原始特徵 (Raw features)」
            msg = features[src]
            
            if self.force_fp32_agg:
                # ★ XPU bfloat16 index_add_ 效能 Bug Workaround (適用於淺層模型)
                # 將聚合部分強制以 float32 進行
                weighted_msg = msg.float() * alpha.float()
                h_neigh = torch.zeros(
                    num_nodes, features.shape[1], device=device, dtype=torch.float32
                )
                h_neigh.index_add_(0, dst, weighted_msg)
                h_neigh = h_neigh.to(current_dtype)
            else:
                # 為了避免 OOM，深層模型 (L>1) 退回使用 bf16 原生 index_add_
                weighted_msg = msg * alpha.to(current_dtype)
                h_neigh = torch.zeros(
                    num_nodes, features.shape[1], device=device, dtype=current_dtype
                )
                h_neigh.index_add_(0, dst, weighted_msg)
                
            del msg
            del weighted_msg

        # 4. Bi-Interaction
        # 與 train_bi_interaction.py 的公式完美對齊:
        # h_out = LeakyReLU( W1(u + N_u) + W2(u * N_u) )
        sum_h = features + h_neigh
        prod_h = features * h_neigh

        h_out = self.leaky_relu(self.W1(sum_h) + self.W2(prod_h))

        if return_attention:
            return h_out, alpha
        return h_out


class KGATAttention(nn.Module):
    def __init__(
        self,
        n_users,
        n_entities,
        n_relations,
        embed_dim=32,
        layers=[32],
        mess_dropout=[0.1],
    ):
        super(KGATAttention, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = embed_dim
        self.mess_dropout = mess_dropout

        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.entity_embed = nn.Embedding(n_entities, embed_dim)
        self.relation_embed = nn.Embedding(n_relations, embed_dim)

        self.aggregator_layers = nn.ModuleList()
        in_dim = embed_dim
        # 當模型深度 > 1 時，關閉強制 float32 聚合以避免 VRAM OOM
        force_fp32_agg = len(layers) <= 3
        
        for out_dim in layers:
            self.aggregator_layers.append(GNNLayerAttention(in_dim, out_dim, force_fp32_agg=force_fp32_agg))
            in_dim = out_dim

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

    def forward(
        self,
        indices,
        edge_types,
        num_nodes,
        user_ids,
        pos_item_ids,
        neg_item_ids=None,
        return_attention=False,
    ):
        """
        indices: (2, E) LongTensor
        edge_types: (E,) LongTensor
        num_nodes: int
        """
        all_embed = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        ego_embeddings = [all_embed]
        attentions = []

        # 取得 Relation Embedded (Shape: [E, embed_dim])
        e_r = self.relation_embed(edge_types)

        for i, layer in enumerate(self.aggregator_layers):
            if return_attention:
                all_embed, att = layer(indices, all_embed, e_r, num_nodes, True)
                attentions.append(att)
            else:
                # Activation Checkpointing:
                # 為了避免多層 GNN (如 L=3) 中 E x D 的龐大中間張量撐爆 VRAM (OOM)，
                # 訓練階段使用 checkpoint 重算中間梯度，用運算時間換取大幅記憶體節省。
                if self.training and all_embed.requires_grad:
                    from torch.utils.checkpoint import checkpoint

                    all_embed = checkpoint(
                        layer,
                        indices,
                        all_embed,
                        e_r,
                        num_nodes,
                        False,
                        use_reentrant=False,
                    )
                else:
                    all_embed = layer(indices, all_embed, e_r, num_nodes, False)

            all_embed = F.normalize(all_embed, p=2, dim=1)

            # 加入 Message Dropout 機制 (與無注意力版維持公平基準)
            if (
                self.training
                and len(self.mess_dropout) > i
                and self.mess_dropout[i] > 0
            ):
                all_embed = F.dropout(
                    all_embed, p=self.mess_dropout[i], training=self.training
                )

            ego_embeddings.append(all_embed)

        final_embed = torch.cat(ego_embeddings, dim=1)
        u_embed = final_embed[user_ids]
        pos_i_embed = final_embed[self.n_users + pos_item_ids]

        pos_scores = torch.sum(u_embed * pos_i_embed, dim=1)

        if neg_item_ids is not None:
            neg_i_embed = final_embed[self.n_users + neg_item_ids]
            neg_scores = torch.sum(u_embed * neg_i_embed, dim=1)
            if return_attention:
                return pos_scores, neg_scores, attentions
            return pos_scores, neg_scores

        if return_attention:
            return pos_scores, attentions
        return pos_scores

    def get_final_embeddings(self, indices, edge_types, num_nodes):
        """
        一次性前向傳播提取全圖特徵，專供 Evaluate 階段加速使用 (不需要每 Batch 都重算全圖 GNN)
        """
        all_embed = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        ego_embeddings = [all_embed]
        e_r = self.relation_embed(edge_types)

        for layer in self.aggregator_layers:
            # Inference 不需要 Dropout 和 Attention Weight 導出
            all_embed = layer(indices, all_embed, e_r, num_nodes, False)
            all_embed = F.normalize(all_embed, p=2, dim=1)
            ego_embeddings.append(all_embed)

        return torch.cat(ego_embeddings, dim=1)


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, force_fp32_agg=True):
        super(GNNLayer, self).__init__()
        self.force_fp32_agg = force_fp32_agg
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, features, target, neighbor, values):
        """
        features: (N, in_dim)
        target: source indices for index_add_
        neighbor: neighbor indices for gathering message
        values: edge values (weights)
        """
        num_nodes = features.shape[0]
        current_dtype = features.dtype
        device = features.device

        # ★ XPU bfloat16 index_add_ 效能 Bug Workaround
        # Intel XPU 的 index_add_ 在 bfloat16 下有嚴重的效能退化 (140x slower)，
        # 原因是高度數節點 (max_degree=22096) 造成原子寫入衝突在 bf16 下
        # 無法有效合併。解法與 KGATAttention 一致：在聚合區段強制使用 float32。
        # 但針對深層模型(L>1)，因會觸發OOM，需退回 bf16 原生 index_add_
        
        if self.force_fp32_agg:
            with torch.autocast(device_type=device.type, enabled=False):
                msg = features.float()[neighbor] * values.float().unsqueeze(1)

                h_neigh = torch.zeros(
                    num_nodes, features.shape[1], device=device, dtype=torch.float32
                )
                h_neigh.index_add_(0, target, msg)
                del msg

            h_neigh = h_neigh.to(current_dtype)
        else:
            msg = features[neighbor] * values.to(current_dtype).unsqueeze(1)
            h_neigh = torch.zeros(
                num_nodes, features.shape[1], device=device, dtype=current_dtype
            )
            h_neigh.index_add_(0, target, msg)
            del msg

        # Bi-Interaction Aggregation
        h_out = self.leaky_relu(
            self.W1(features + h_neigh) + self.W2(features * h_neigh)
        )
        return h_out


class KGAT_BiInteraction(nn.Module):
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
        super(KGAT_BiInteraction, self).__init__()
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
        # 當模型深度 > 1 時，關閉強制 float32 聚合以避免 VRAM OOM
        force_fp32_agg = len(layers) <= 1
        
        for out_dim in layers:
            self.aggregator_layers.append(GNNLayer(in_dim, out_dim, force_fp32_agg=force_fp32_agg))
            in_dim = out_dim

        self.mess_dropout = mess_dropout
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.W_R)

    def forward(self, target, neighbor, values, num_nodes, user_ids, pos_item_ids, neg_item_ids=None):
        """
        target: (E,) dense LongTensor — 稀疏鄰接矩陣的 source indices
        neighbor: (E,) dense LongTensor — 稀疏鄰接矩陣的 destination indices
        values: (E,) dense Tensor — 正規化後的邊權重
        num_nodes: int — 總節點數
        user_ids: tensor of user indices
        pos_item_ids: tensor of positive item indices
        neg_item_ids: (Optional) tensor of negative item indices
        """

        # Initial features: Concatenate user and entity embeddings
        all_embed = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)

        # List to store embeddings at each layer
        ego_embeddings = [all_embed]

        for i, layer in enumerate(self.aggregator_layers):
            all_embed = layer(all_embed, target, neighbor, values)
            all_embed = F.normalize(all_embed, p=2, dim=1)

            # 實作原論文的 Message Dropout (避免 overfitting)
            if (
                self.training
                and len(self.mess_dropout) > i
                and self.mess_dropout[i] > 0
            ):
                all_embed = F.dropout(
                    all_embed, p=self.mess_dropout[i], training=self.training
                )

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

    def get_final_embeddings(self, adj):
        """
        一次性前向傳播提取全圖特徵，專供 Evaluate 階段加速使用
        """
        if adj.layout == torch.sparse_coo:
            target, neighbor = adj.indices()[0], adj.indices()[1]
            values = adj.values()
        elif adj.layout == torch.sparse_csr:
            coo = adj.to_sparse_coo()
            target, neighbor = coo.indices()
            values = coo.values()
        else:
            raise ValueError(f"Unsupported adjacency layout: {adj.layout}")

        all_embed = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        ego_embeddings = [all_embed]

        for layer in self.aggregator_layers:
            all_embed = layer(all_embed, target, neighbor, values)
            all_embed = F.normalize(all_embed, p=2, dim=1)
            ego_embeddings.append(all_embed)

        return torch.cat(ego_embeddings, dim=1)
