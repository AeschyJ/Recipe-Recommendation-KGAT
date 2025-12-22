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

    def forward(self, indices, features, num_nodes, return_attention=False):
        """
        Memory-optimized Forward pass using Gather-Scatter (Message Passing).
        Avoids OOM by not creating large N*N matrices or broadcasting dense tensors.
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
        e_ij = self.leaky_relu(torch.matmul(edge_h, self.a))
        del edge_h  # Release memory

        # 3. Softmax & Message Passing
        # Use simple softmax for numerical stability test (or reimplement scatter softmax if needed)
        # Note: In Colab we used a simplified softmax or disabled autodetect.
        # Here we follow the stable logic strictly.

        # To support mixed precision (AMP) properly without error, force float32 for delicate ops
        current_dtype = features.dtype

        # Temporarily disable autocast for sparse/scatter operations to avoid type mismatch
        # or NotImplemetedError for Half types in some ops
        with (
            torch.autocast(device_type=device.type, enabled=False)
            if device.type != "cpu"
            else torch.no_grad()
        ):
            # Force float32 for stability
            e_ij = e_ij.float()
            e_ij = e_ij - e_ij.max()
            exp_e = torch.exp(e_ij)

            denom = torch.zeros(num_nodes, 1, device=device)
            denom.scatter_add_(0, dst.unsqueeze(1), exp_e)

            alpha = exp_e / (denom[dst] + 1e-9)
            del exp_e, denom

            # Message Passing
            # Ensure W1 input is float32 if we are in this block
            h_msg_all = self.W1(features.float())
            msg = h_msg_all[src]
            del h_msg_all

            weighted_msg = msg * alpha
            del msg

            h_neigh = torch.zeros(num_nodes, self.W1.out_features, device=device)
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

    def forward(self, indices, num_nodes, user_ids, item_ids, return_attention=False):
        """
        indices: (2, E) LongTensor
        num_nodes: int
        """
        all_embed = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        ego_embeddings = [all_embed]
        attentions = []

        for layer in self.aggregator_layers:
            if return_attention:
                all_embed, att = layer(indices, all_embed, num_nodes, True)
                attentions.append(att)
            else:
                all_embed = layer(indices, all_embed, num_nodes, False)

            all_embed = F.normalize(all_embed, p=2, dim=1)
            ego_embeddings.append(all_embed)

        final_embed = torch.cat(ego_embeddings, dim=1)
        u_embed = final_embed[user_ids]
        i_embed = final_embed[self.n_users + item_ids]

        scores = torch.sum(u_embed * i_embed, dim=1)

        if return_attention:
            return scores, attentions
        return scores
