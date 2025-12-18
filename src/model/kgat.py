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

    def forward(self, adj, user_ids, item_ids):
        """
        adj: torch.sparse_coo_tensor (Adjacency Matrix)
        user_ids: tensor of user indices
        item_ids: tensor of item indices
        """
        # 1. Update Entity Embeddings via Knowledge Graph Aggregation

        # Initial features: Concatenate user and entity embeddings
        all_embed = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)

        # List to store embeddings at each layer
        ego_embeddings = [all_embed]

        for i, layer in enumerate(self.aggregator_layers):
            all_embed = layer(adj, all_embed)
            # Normalize to prevent exploding embeddings
            all_embed = F.normalize(all_embed, p=2, dim=1)
            ego_embeddings.append(all_embed)

        # Concatenate embeddings from all layers
        final_embed = torch.cat(ego_embeddings, dim=1)

        # Retrieve user and item embeddings
        u_embed = final_embed[user_ids]
        # Note: Item IDs in the graph are offset by n_users
        i_embed = final_embed[self.n_users + item_ids]

        # Score via Inner Product
        scores = torch.sum(u_embed * i_embed, dim=1)
        return scores


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, adj, features):
        """
        adj: Sparse Tensor (N, N)
        features: (N, in_dim)
        """
        # Message passing: A * H
        # torch.sparse.mm is optimized for Sparse x Dense multiplication
        h_neigh = torch.sparse.mm(adj, features)

        # Bi-Interaction Aggregation
        # f_a = W1 * (h + h_neigh)
        # f_b = W2 * (h * h_neigh)

        term1 = self.W1(features + h_neigh)
        term2 = self.W2(features * h_neigh)

        h_out = self.leaky_relu(term1 + term2)
        return h_out
