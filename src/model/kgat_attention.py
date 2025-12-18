import torch
import torch.nn as nn
import torch.nn.functional as F


class KGATAttention(nn.Module):
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
        super(KGATAttention, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = embed_dim

        # Embeddings
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.entity_embed = nn.Embedding(n_entities, embed_dim)
        self.relation_embed = nn.Embedding(n_relations, embed_dim)

        # Graph Attention Layers
        self.aggregator_layers = nn.ModuleList()
        in_dim = embed_dim
        for out_dim in layers:
            self.aggregator_layers.append(GNNLayerAttention(in_dim, out_dim))
            in_dim = out_dim

        self.mess_dropout = mess_dropout
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

    def forward(self, adj, user_ids, item_ids, return_attention=False):
        """
        adj: torch.sparse_coo_tensor.
        """
        # Initial features
        all_embed = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        ego_embeddings = [all_embed]

        indices = adj._indices()
        num_nodes = adj.shape[0]

        attentions = []

        for i, layer in enumerate(self.aggregator_layers):
            if return_attention:
                all_embed, att = layer(
                    indices, all_embed, num_nodes, return_attention=True
                )
                attentions.append(att)
            else:
                all_embed = layer(indices, all_embed, num_nodes)

            all_embed = F.normalize(all_embed, p=2, dim=1)
            ego_embeddings.append(all_embed)

        final_embed = torch.cat(ego_embeddings, dim=1)

        u_embed = final_embed[user_ids]
        i_embed = final_embed[self.n_users + item_ids]

        # Inner Product Score
        scores = torch.sum(u_embed * i_embed, dim=1)

        if return_attention:
            return scores, attentions
        return scores


class GNNLayerAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayerAttention, self).__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)

        # Attention parameters (GAT style)
        self.W_att = nn.Linear(in_dim, out_dim)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data)

        self.leaky_relu = nn.LeakyReLU()

    def _edge_softmax(self, logits, indices, num_nodes):
        src, dst = indices[0], indices[1]
        exp_logits = torch.exp(logits)
        denom_sum = torch.zeros(num_nodes, 1, device=logits.device)
        denom_sum.scatter_add_(0, dst.unsqueeze(1), exp_logits)
        edge_denom = denom_sum[dst]
        attentions = exp_logits / (edge_denom + 1e-9)
        return attentions

    def forward(self, indices, features, num_nodes, return_attention=False):
        # 1. Attention Score Computation
        h_trans = self.W_att(features)

        src, dst = indices[0], indices[1]
        src_h = h_trans[src]
        dst_h = h_trans[dst]

        edge_h_cat = torch.cat([src_h, dst_h], dim=1)
        logits = self.leaky_relu(torch.matmul(edge_h_cat, self.a))

        alpha = self._edge_softmax(logits, indices, num_nodes)

        # 2. Aggregation
        adj_att = torch.sparse_coo_tensor(
            indices, alpha.squeeze(), (num_nodes, num_nodes)
        )

        h_neigh_msg = self.W1(features)
        h_neigh = torch.sparse.mm(adj_att, h_neigh_msg)

        # 3. Bi-Interaction
        h_self = (
            self.W1(features) if features.shape[1] != h_neigh.shape[1] else features
        )
        term1 = h_self + h_neigh

        h_prod = h_self * h_neigh
        term2 = self.W2(h_prod)

        h_out = self.leaky_relu(term1 + term2)

        if return_attention:
            return h_out, alpha
        else:
            return h_out
